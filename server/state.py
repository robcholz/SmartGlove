from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from server.models import (
    DeviceEventFrame,
    DeviceInfoRequest,
    DeviceOnlineFrame,
    SensorBatchFrame,
)

StoredFrame = DeviceOnlineFrame | DeviceEventFrame | SensorBatchFrame


@dataclass
class DeviceRecord:
    kind: str | None = None
    events: tuple[str, ...] = ()
    connected: bool = False
    last_seen_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class NetworkState:
    def __init__(self, max_frames: int = 1024) -> None:
        self._devices: dict[str, DeviceRecord] = {}
        self._recent_frames: deque[dict[str, Any]] = deque(maxlen=max_frames)

    def register_device(self, payload: DeviceInfoRequest) -> None:
        existing = self._devices.get(payload.device_id)
        self._devices[payload.device_id] = DeviceRecord(
            kind=payload.kind,
            events=tuple(payload.events),
            connected=existing.connected if existing is not None else False,
        )

    def record_frame(self, frame: StoredFrame) -> bool:
        record = self._devices.setdefault(frame.device_id, DeviceRecord())
        record.last_seen_at = datetime.now(UTC)
        became_connected = False
        if isinstance(frame, DeviceOnlineFrame):
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
            "recent_frames": list(self._recent_frames),
        }
