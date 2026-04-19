from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, TypeAdapter
from typing_extensions import Annotated

DeviceId = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{12}$")]
Axis3 = tuple[float, float, float]
JsonValue = dict[str, Any] | list[Any] | str | float | int | bool | None


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AckResponse(BaseModel):
    ok: bool = True


class DeviceInfoRequest(StrictModel):
    device_id: DeviceId
    events: list[str] = Field(default_factory=list)


class FlexReadings(StrictModel):
    thumb: float
    index: float
    middle: float
    ring: float
    pinky: float


class BufferedSensorSample(StrictModel):
    imu_acc: Axis3
    vel: Axis3
    flex: FlexReadings


class DeviceOnlineFrame(StrictModel):
    kind: Literal["device.online"]
    device_id: DeviceId


class DeviceEventFrame(StrictModel):
    kind: Literal["device.event"]
    device_id: DeviceId
    event: str
    payload: JsonValue


class SensorBatchFrame(StrictModel):
    kind: Literal["sensor.batch"]
    device_id: DeviceId
    sample_rate_hz: int = Field(ge=1)
    start_tick_ms: int = Field(ge=0)
    samples: list[BufferedSensorSample]


StatusFrame = Annotated[
    DeviceOnlineFrame | DeviceEventFrame | SensorBatchFrame,
    Field(discriminator="kind"),
]

status_frame_adapter = TypeAdapter(StatusFrame)
