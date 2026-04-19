from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, TypeAdapter
from typing_extensions import Annotated

DeviceId = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{12}$")]
DeviceKind = Literal["glove", "machine"]
Axis3 = tuple[float, float, float]
JsonValue = dict[str, Any] | list[Any] | str | float | int | bool | None


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AckResponse(BaseModel):
    ok: bool = True


class DeviceInfoRequest(StrictModel):
    device_id: DeviceId
    kind: DeviceKind
    events: list[str] = Field(default_factory=list)


class CreateGloveMappingRequest(StrictModel):
    source_event: str
    target_machine_id: DeviceId
    target_event: str


class GloveMapping(StrictModel):
    id: str
    source_device_id: DeviceId
    source_event: str
    target_machine_id: DeviceId
    target_event: str


class GloveMappingListResponse(StrictModel):
    items: list[GloveMapping]


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


class MachineOnlineFrame(StrictModel):
    kind: Literal["machine.online"]
    device_id: DeviceId


class MachineResultFrame(StrictModel):
    kind: Literal["machine.result"]
    device_id: DeviceId
    request_id: str
    status: Literal["ok", "error"]
    payload: JsonValue


class MachineTriggerFrame(StrictModel):
    kind: Literal["machine.trigger"]
    request_id: str
    device_id: DeviceId
    event: str
    source_device_id: DeviceId
    source_event: str
    payload: JsonValue


StatusFrame = Annotated[
    DeviceOnlineFrame | DeviceEventFrame | SensorBatchFrame,
    Field(discriminator="kind"),
]

MachineControlFrame = Annotated[
    MachineOnlineFrame | MachineResultFrame,
    Field(discriminator="kind"),
]

status_frame_adapter = TypeAdapter(StatusFrame)
machine_control_frame_adapter = TypeAdapter(MachineControlFrame)
