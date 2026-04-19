from __future__ import annotations

import json
import logging

from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from server.config import Settings
from server.models import (
    AckResponse,
    CreateGloveMappingRequest,
    DeviceEventFrame,
    DeviceInfoRequest,
    GloveMapping,
    GloveMappingListResponse,
    machine_control_frame_adapter,
    status_frame_adapter,
)
from server.state import (
    NetworkState,
    StateConflictError,
    StateError,
    StateNotFoundError,
    StateValidationError,
)

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    app = FastAPI(
        title="Smart Glove Network API",
        version="0.1.0",
        summary="HTTP and WebSocket endpoints matching docs/network-openapi.yaml",
    )
    app.state.settings = settings
    app.state.network_state = NetworkState()

    if settings.allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.allowed_origins),
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["*"],
        )

    @app.exception_handler(RequestValidationError)
    async def handle_request_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"ok": False, "error": "invalid_request", "details": exc.errors()},
        )

    @app.get("/health", response_model=AckResponse, tags=["Health"])
    async def health() -> AckResponse:
        return AckResponse()

    @app.get("/ready", response_model=AckResponse, tags=["Health"])
    async def ready() -> AckResponse:
        return AckResponse()

    @app.post("/v1/device-info", response_model=AckResponse, tags=["Device Info"])
    async def publish_device_info(
        payload: DeviceInfoRequest, request: Request
    ) -> AckResponse:
        state: NetworkState = request.app.state.network_state
        state.register_device(payload)
        logger.info(
            "registered device_info device_id=%s kind=%s events=%s",
            payload.device_id,
            payload.kind,
            payload.events,
        )
        return AckResponse()

    @app.get(
        "/v1/gloves/{glove_device_id}/mappings",
        response_model=GloveMappingListResponse,
        tags=["Glove Mappings"],
    )
    async def list_glove_mappings(
        glove_device_id: str, request: Request
    ) -> GloveMappingListResponse:
        state: NetworkState = request.app.state.network_state
        try:
            return GloveMappingListResponse(items=state.list_mappings(glove_device_id))
        except StateNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except StateValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post(
        "/v1/gloves/{glove_device_id}/mappings",
        response_model=GloveMapping,
        status_code=status.HTTP_201_CREATED,
        tags=["Glove Mappings"],
    )
    async def create_glove_mapping(
        glove_device_id: str,
        payload: CreateGloveMappingRequest,
        request: Request,
    ) -> GloveMapping:
        state: NetworkState = request.app.state.network_state
        try:
            mapping = state.create_mapping(glove_device_id, payload)
        except StateNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except StateConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except StateValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        logger.info(
            "registered mapping glove_device_id=%s source_event=%s target_machine_id=%s target_event=%s",
            glove_device_id,
            payload.source_event,
            payload.target_machine_id,
            payload.target_event,
        )
        return mapping

    @app.delete(
        "/v1/gloves/{glove_device_id}/mappings/{mapping_id}",
        status_code=status.HTTP_204_NO_CONTENT,
        tags=["Glove Mappings"],
    )
    async def delete_glove_mapping(
        glove_device_id: str, mapping_id: str, request: Request
    ) -> Response:
        state: NetworkState = request.app.state.network_state
        try:
            state.delete_mapping(glove_device_id, mapping_id)
        except StateNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except StateValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        logger.info(
            "deleted mapping glove_device_id=%s mapping_id=%s",
            glove_device_id,
            mapping_id,
        )
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.websocket("/v1/ws")
    async def open_status_websocket(websocket: WebSocket) -> None:
        state: NetworkState = websocket.app.state.network_state
        current_device_id: str | None = None
        await websocket.accept()

        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    return

                text_payload = message.get("text")
                if text_payload is None:
                    await websocket.close(
                        code=status.WS_1003_UNSUPPORTED_DATA,
                        reason="text frames required",
                    )
                    return

                try:
                    raw_frame = json.loads(text_payload)
                except json.JSONDecodeError:
                    await websocket.close(
                        code=status.WS_1003_UNSUPPORTED_DATA,
                        reason="invalid json",
                    )
                    return

                try:
                    frame = status_frame_adapter.validate_python(raw_frame)
                except ValidationError:
                    await websocket.close(
                        code=status.WS_1008_POLICY_VIOLATION,
                        reason="invalid frame",
                    )
                    return

                current_device_id = frame.device_id
                if state.record_frame(frame):
                    logger.info("device %s is online", frame.device_id)
                logger.info(
                    "received frame kind=%s device_id=%s",
                    frame.kind,
                    frame.device_id,
                )
                if isinstance(frame, DeviceEventFrame):
                    for machine_socket, trigger in state.build_machine_triggers(frame):
                        try:
                            await machine_socket.send_json(trigger.model_dump(mode="json"))
                            logger.info(
                                "triggered machine device_id=%s event=%s source_device_id=%s source_event=%s",
                                trigger.device_id,
                                trigger.event,
                                trigger.source_device_id,
                                trigger.source_event,
                            )
                        except RuntimeError:
                            state.unregister_machine_socket(trigger.device_id, machine_socket)
        except WebSocketDisconnect:
            return
        finally:
            if current_device_id is not None and state.mark_disconnected(
                current_device_id
            ):
                logger.info("device %s is offline", current_device_id)

    @app.websocket("/v1/machine-ws")
    async def open_machine_control_websocket(websocket: WebSocket) -> None:
        state: NetworkState = websocket.app.state.network_state
        current_device_id: str | None = None
        await websocket.accept()

        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    return

                text_payload = message.get("text")
                if text_payload is None:
                    await websocket.close(
                        code=status.WS_1003_UNSUPPORTED_DATA,
                        reason="text frames required",
                    )
                    return

                try:
                    raw_frame = json.loads(text_payload)
                except json.JSONDecodeError:
                    await websocket.close(
                        code=status.WS_1003_UNSUPPORTED_DATA,
                        reason="invalid json",
                    )
                    return

                try:
                    frame = machine_control_frame_adapter.validate_python(raw_frame)
                except ValidationError:
                    await websocket.close(
                        code=status.WS_1008_POLICY_VIOLATION,
                        reason="invalid frame",
                    )
                    return

                current_device_id = frame.device_id
                try:
                    state.register_machine_socket(frame.device_id, websocket)
                except StateError:
                    await websocket.close(
                        code=status.WS_1008_POLICY_VIOLATION,
                        reason="machine device must be registered first",
                    )
                    return

                if state.record_frame(frame):
                    logger.info("device %s is online", frame.device_id)
                logger.info(
                    "received machine frame kind=%s device_id=%s",
                    frame.kind,
                    frame.device_id,
                )
        except WebSocketDisconnect:
            return
        finally:
            if current_device_id is not None:
                state.unregister_machine_socket(current_device_id, websocket)
            if current_device_id is not None and state.mark_disconnected(
                current_device_id
            ):
                logger.info("device %s is offline", current_device_id)

    return app


app = create_app()
