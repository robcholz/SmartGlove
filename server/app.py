from __future__ import annotations

import json
import logging

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from server.config import Settings
from server.models import AckResponse, DeviceInfoRequest, status_frame_adapter
from server.state import NetworkState

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
            allow_methods=["GET", "POST"],
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
            "registered device_info device_id=%s events=%s",
            payload.device_id,
            payload.events,
        )
        return AckResponse()

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
                state.record_frame(frame)
                logger.info(
                    "received frame kind=%s device_id=%s",
                    frame.kind,
                    frame.device_id,
                )
        except WebSocketDisconnect:
            return
        finally:
            if current_device_id is not None:
                state.mark_disconnected(current_device_id)

    return app


app = create_app()
