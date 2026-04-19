from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from fastapi import WebSocket
from server.models import (
    CreateGloveMappingRequest,
    DeviceEventFrame,
    DeviceInfoRequest,
    DeviceOnlineFrame,
    GloveMapping,
    MachineOnlineFrame,
    MachineResultFrame,
    MachineTriggerFrame,
    SensorBatchFrame,
)

StoredFrame = (
    DeviceOnlineFrame
    | DeviceEventFrame
    | SensorBatchFrame
    | MachineOnlineFrame
    | MachineResultFrame
    | MachineTriggerFrame
)


class StateError(Exception):
    pass


class StateNotFoundError(StateError):
    pass


class StateConflictError(StateError):
    pass


class StateValidationError(StateError):
    pass


@dataclass
class DeviceRecord:
    kind: str | None = None
    events: tuple[str, ...] = ()
    connected: bool = False
    last_seen_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True)
class MappingRecord:
    id: str
    source_device_id: str
    source_event: str
    target_machine_id: str
    target_event: str

    def to_model(self) -> GloveMapping:
        return GloveMapping(
            id=self.id,
            source_device_id=self.source_device_id,
            source_event=self.source_event,
            target_machine_id=self.target_machine_id,
            target_event=self.target_event,
        )


class NetworkState:
    def __init__(self, max_frames: int = 1024) -> None:
        self._devices: dict[str, DeviceRecord] = {}
        self._recent_frames: deque[dict[str, Any]] = deque(maxlen=max_frames)
        self._mappings: dict[str, MappingRecord] = {}
        self._machine_sockets: dict[str, WebSocket] = {}

    def register_device(self, payload: DeviceInfoRequest) -> None:
        existing = self._devices.get(payload.device_id)
        self._devices[payload.device_id] = DeviceRecord(
            kind=payload.kind,
            events=tuple(payload.events),
            connected=existing.connected if existing is not None else False,
        )

    def create_mapping(
        self, glove_device_id: str, payload: CreateGloveMappingRequest
    ) -> GloveMapping:
        glove_record = self._require_device_kind(glove_device_id, "glove")
        machine_record = self._require_device_kind(payload.target_machine_id, "machine")
        if payload.source_event not in glove_record.events:
            raise StateValidationError(
                f"glove {glove_device_id} did not declare source event {payload.source_event}"
            )
        if payload.target_event not in machine_record.events:
            raise StateValidationError(
                f"machine {payload.target_machine_id} did not declare target event {payload.target_event}"
            )

        for existing in self._mappings.values():
            if (
                existing.source_device_id == glove_device_id
                and existing.source_event == payload.source_event
                and existing.target_machine_id == payload.target_machine_id
                and existing.target_event == payload.target_event
            ):
                raise StateConflictError("mapping already exists")

        mapping = MappingRecord(
            id=f"map_{uuid4().hex[:12]}",
            source_device_id=glove_device_id,
            source_event=payload.source_event,
            target_machine_id=payload.target_machine_id,
            target_event=payload.target_event,
        )
        self._mappings[mapping.id] = mapping
        return mapping.to_model()

    def list_mappings(self, glove_device_id: str) -> list[GloveMapping]:
        self._require_device_kind(glove_device_id, "glove")
        return [
            mapping.to_model()
            for mapping in self._mappings.values()
            if mapping.source_device_id == glove_device_id
        ]

    def delete_mapping(self, glove_device_id: str, mapping_id: str) -> None:
        self._require_device_kind(glove_device_id, "glove")
        mapping = self._mappings.get(mapping_id)
        if mapping is None or mapping.source_device_id != glove_device_id:
            raise StateNotFoundError(f"mapping {mapping_id} not found")
        del self._mappings[mapping_id]

    def record_frame(self, frame: StoredFrame) -> bool:
        record = self._devices.setdefault(frame.device_id, DeviceRecord())
        record.last_seen_at = datetime.now(UTC)
        became_connected = False
        if isinstance(frame, (DeviceOnlineFrame, MachineOnlineFrame)):
            became_connected = not record.connected
            record.connected = True

        self._recent_frames.append(frame.model_dump(mode="json"))
        return became_connected

    def mark_disconnected(self, device_id: str) -> bool:
        record = self._devices.get(device_id)
        if record is not None:
            was_connected = record.connected
            record.connected = False
            record.last_seen_at = datetime.now(UTC)
            return was_connected
        return False

    def register_machine_socket(self, device_id: str, websocket: WebSocket) -> None:
        self._require_device_kind(device_id, "machine")
        self._machine_sockets[device_id] = websocket

    def unregister_machine_socket(self, device_id: str, websocket: WebSocket) -> None:
        current = self._machine_sockets.get(device_id)
        if current is websocket:
            del self._machine_sockets[device_id]

    def build_machine_triggers(
        self, frame: DeviceEventFrame
    ) -> list[tuple[WebSocket, MachineTriggerFrame]]:
        triggers: list[tuple[WebSocket, MachineTriggerFrame]] = []
        for mapping in self._mappings.values():
            if (
                mapping.source_device_id != frame.device_id
                or mapping.source_event != frame.event
            ):
                continue

            websocket = self._machine_sockets.get(mapping.target_machine_id)
            if websocket is None:
                continue

            trigger = MachineTriggerFrame(
                kind="machine.trigger",
                request_id=f"req_{uuid4().hex[:12]}",
                device_id=mapping.target_machine_id,
                event=mapping.target_event,
                source_device_id=frame.device_id,
                source_event=frame.event,
                payload=frame.payload,
            )
            self._recent_frames.append(trigger.model_dump(mode="json"))
            triggers.append((websocket, trigger))

        return triggers

    def snapshot(self) -> dict[str, Any]:
        return {
            "devices": {
                device_id: {
                    "kind": record.kind,
                    "events": list(record.events),
                    "connected": record.connected,
                    "last_seen_at": record.last_seen_at.isoformat(),
                }
                for device_id, record in self._devices.items()
            },
            "mappings": [
                mapping.to_model().model_dump(mode="json")
                for mapping in self._mappings.values()
            ],
            "recent_frames": list(self._recent_frames),
        }

    def _require_device_kind(self, device_id: str, kind: str) -> DeviceRecord:
        record = self._devices.get(device_id)
        if record is None:
            raise StateNotFoundError(f"device {device_id} not found")
        if record.kind != kind:
            raise StateValidationError(
                f"device {device_id} must have kind {kind}, got {record.kind}"
            )
        return record
