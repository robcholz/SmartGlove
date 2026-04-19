from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

from tests.network_contract import validate_device_info_request, validate_ws_frame
from tests.network_server import (
    CapturedWsFrame,
    MockNetworkServer,
    resolve_bind_host_ip,
)
from tests.provision_e2e import (
    ProcessCapture,
    build_firmware_env,
    cargo_profile_flag,
    resolve_credentials,
    run_command,
)
from tests.provision_test import run_test

CONFIG_PATH = Path("tests/network_runtime_config.env")
EXPECTED_EVENT_NAME = "network.ready"
EXPECTED_EVENT_PAYLOAD = {"phase": "provisioned"}
EXPECTED_EVENT_PAYLOAD_JSON = json.dumps(EXPECTED_EVENT_PAYLOAD, separators=(",", ":"))
FIRMWARE_START_ATTEMPTS = 3


@dataclass(frozen=True)
class Scenario:
    sample_rate_hz: int
    batch_samples: int
    batch_count: int
    flush_interval_ms: int

    @property
    def name(self) -> str:
        return (
            f"rate={self.sample_rate_hz}Hz batch={self.batch_samples} "
            f"count={self.batch_count} flush={self.flush_interval_ms}ms"
        )


@dataclass(frozen=True)
class ScenarioResult:
    scenario: Scenario
    device_id: str
    ws_frames: int
    ws_bytes: int
    total_samples: int
    first_batch_tick_ms: int
    last_batch_tick_ms: int
    device_span_ms: float
    host_span_ms: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build, flash, and run incremental SmartGlove network end-to-end tests."
    )
    parser.add_argument("--ssid")
    parser.add_argument("--password")
    parser.add_argument("--env-file", type=Path, default=Path(".env.local"))
    parser.add_argument("--device-name", default="SmartGlove Provision")
    parser.add_argument("--scan-timeout", type=float, default=40.0)
    parser.add_argument("--status-timeout", type=float, default=30.0)
    parser.add_argument("--firmware-timeout", type=float, default=45.0)
    parser.add_argument("--firmware-log", type=Path)
    parser.add_argument("--allow-name-fallback", action="store_true")
    parser.add_argument("--serial-port")
    parser.add_argument("--serial-baudrate", type=int, default=115200)
    parser.add_argument("--serial-log", type=Path)
    parser.add_argument("--expect-serial", action="append", default=[])
    parser.add_argument("--build-mode", choices=["release", "debug"], default="release")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--host-ip")
    parser.add_argument(
        "--sample-rates",
        default="100,200,400,800",
        help="Comma-separated sample rates in Hz to test incrementally.",
    )
    parser.add_argument(
        "--batch-sizes",
        default="10",
        help="Comma-separated batch sizes to test for each sample rate.",
    )
    parser.add_argument(
        "--batch-count",
        type=int,
        default=10,
        help="Number of sensor batches to send per scenario.",
    )
    parser.add_argument(
        "--flush-interval-ms",
        type=int,
        default=100,
        help="Reporter flush interval to embed in the test firmware config.",
    )
    return parser


def parse_csv_ints(raw: str, field_name: str) -> list[int]:
    values = [chunk.strip() for chunk in raw.split(",") if chunk.strip()]
    if not values:
        raise SystemExit(f"{field_name} must contain at least one integer")

    parsed: list[int] = []
    for value in values:
        try:
            parsed.append(int(value))
        except ValueError as err:
            raise SystemExit(
                f"{field_name} contains invalid integer {value!r}"
            ) from err

    if any(value <= 0 for value in parsed):
        raise SystemExit(f"{field_name} values must be greater than zero")

    return parsed


def build_scenarios(args: argparse.Namespace) -> list[Scenario]:
    sample_rates = parse_csv_ints(args.sample_rates, "sample-rates")
    batch_sizes = parse_csv_ints(args.batch_sizes, "batch-sizes")
    if args.batch_count <= 0:
        raise SystemExit("batch-count must be greater than zero")
    if args.flush_interval_ms <= 0:
        raise SystemExit("flush-interval-ms must be greater than zero")

    return [
        Scenario(
            sample_rate_hz=sample_rate_hz,
            batch_samples=batch_samples,
            batch_count=args.batch_count,
            flush_interval_ms=args.flush_interval_ms,
        )
        for sample_rate_hz in sample_rates
        for batch_samples in batch_sizes
    ]


def write_runtime_config(host_ip: str, port: int, scenario: Scenario) -> str:
    previous = CONFIG_PATH.read_text(encoding="utf-8")
    CONFIG_PATH.write_text(
        "\n".join(
            [
                f"DEVICE_INFO_URL=http://{host_ip}:{port}/v1/device-info",
                f"STATUS_WS_URL=ws://{host_ip}:{port}/v1/ws",
                f"EVENT_NAME={EXPECTED_EVENT_NAME}",
                f"EVENT_PAYLOAD_JSON={EXPECTED_EVENT_PAYLOAD_JSON}",
                f"SAMPLE_RATE_HZ={scenario.sample_rate_hz}",
                f"BATCH_SAMPLES={scenario.batch_samples}",
                f"BATCH_COUNT={scenario.batch_count}",
                f"FLUSH_INTERVAL_MS={scenario.flush_interval_ms}",
                "POST_RUN_DELAY_MS=250",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return previous


def restore_runtime_config(previous: str) -> None:
    CONFIG_PATH.write_text(previous, encoding="utf-8")


def assert_device_info(payload: dict) -> str:
    validate_device_info_request(payload)
    if payload["kind"] != "glove":
        raise AssertionError(payload)
    if payload["events"] != [EXPECTED_EVENT_NAME]:
        raise AssertionError(payload)
    return payload["device_id"]


def assert_ws_frames(
    frames: list[CapturedWsFrame],
    expected_device_id: str,
    scenario: Scenario,
) -> ScenarioResult:
    expected_frame_count = scenario.batch_count + 2
    if len(frames) != expected_frame_count:
        raise AssertionError(
            f"expected {expected_frame_count} websocket frames, got {len(frames)}"
        )

    payloads = [frame.payload for frame in frames]
    for payload in payloads:
        validate_ws_frame(payload)

    online, event, *batches = payloads

    if online != {"kind": "device.online", "device_id": expected_device_id}:
        raise AssertionError(online)

    if event != {
        "kind": "device.event",
        "device_id": expected_device_id,
        "event": EXPECTED_EVENT_NAME,
        "payload": EXPECTED_EVENT_PAYLOAD,
    }:
        raise AssertionError(event)

    if len(batches) != scenario.batch_count:
        raise AssertionError(
            f"expected {scenario.batch_count} sensor batches, got {len(batches)}"
        )

    total_samples = 0
    first_batch_frame = frames[2]
    last_batch_frame = frames[-1]
    previous_tick_ms = -1

    for index, batch in enumerate(batches):
        if batch["kind"] != "sensor.batch":
            raise AssertionError(batch)
        if batch["device_id"] != expected_device_id:
            raise AssertionError(batch)
        if batch["sample_rate_hz"] != scenario.sample_rate_hz:
            raise AssertionError(batch)
        if len(batch["samples"]) != scenario.batch_samples:
            raise AssertionError(batch)
        if batch["start_tick_ms"] < previous_tick_ms:
            raise AssertionError(batch)
        previous_tick_ms = batch["start_tick_ms"]
        total_samples += len(batch["samples"])

        first_sample = batch["samples"][0]
        if set(first_sample.keys()) != {"imu_acc", "vel", "flex"}:
            raise AssertionError(first_sample)
        if len(first_sample["imu_acc"]) != 3 or len(first_sample["vel"]) != 3:
            raise AssertionError(first_sample)
        if set(first_sample["flex"]) != {"thumb", "index", "middle", "ring", "pinky"}:
            raise AssertionError(first_sample)

        if index > 0 and batch["start_tick_ms"] == batches[index - 1]["start_tick_ms"]:
            raise AssertionError("sensor batch start_tick_ms did not advance")

    first_tick = first_batch_frame.payload["start_tick_ms"]
    last_tick = last_batch_frame.payload["start_tick_ms"]
    device_span_ms = (
        last_tick
        + ((scenario.batch_samples - 1) * 1000.0 / scenario.sample_rate_hz)
        - first_tick
    )
    host_span_ms = max(
        0.0,
        (last_batch_frame.received_at - first_batch_frame.received_at) * 1000.0,
    )

    return ScenarioResult(
        scenario=scenario,
        device_id=expected_device_id,
        ws_frames=len(frames),
        ws_bytes=sum(frame.size_bytes for frame in frames),
        total_samples=total_samples,
        first_batch_tick_ms=int(first_tick),
        last_batch_tick_ms=int(last_tick),
        device_span_ms=device_span_ms,
        host_span_ms=host_span_ms,
    )


def print_benchmark(result: ScenarioResult) -> None:
    host_samples_per_second = (
        result.total_samples / (result.host_span_ms / 1000.0)
        if result.host_span_ms > 0
        else 0.0
    )
    device_samples_per_second = (
        result.total_samples / (result.device_span_ms / 1000.0)
        if result.device_span_ms > 0
        else 0.0
    )
    print(
        "benchmark "
        f"{result.scenario.name} "
        f"frames={result.ws_frames} "
        f"bytes={result.ws_bytes} "
        f"samples={result.total_samples} "
        f"device_span_ms={result.device_span_ms:.1f} "
        f"host_span_ms={result.host_span_ms:.1f} "
        f"device_samples_per_sec={device_samples_per_second:.1f} "
        f"host_samples_per_sec={host_samples_per_second:.1f}",
        flush=True,
    )


def run_one_scenario(
    args: argparse.Namespace,
    scenario: Scenario,
    host_ip: str,
    firmware_env: dict[str, str],
) -> ScenarioResult:
    server = MockNetworkServer()
    server.start()
    assert server.actual_port is not None

    previous_config = write_runtime_config(host_ip, server.actual_port, scenario)
    try:
        if not args.skip_build:
            run_command(
                [
                    "cargo",
                    "build",
                    *cargo_profile_flag(args.build_mode),
                    "--bin",
                    "provision_network",
                ],
                env=firmware_env,
            )

        last_startup_error: Exception | None = None
        for attempt in range(1, FIRMWARE_START_ATTEMPTS + 1):
            firmware = ProcessCapture(
                [
                    "cargo",
                    "run",
                    *cargo_profile_flag(args.build_mode),
                    "--bin",
                    "provision_network",
                ],
                log_path=args.firmware_log,
                env=firmware_env,
            )
            firmware.start()

            try:
                try:
                    firmware.wait_for(
                        "PROVISION_NETWORK_READY", timeout=args.firmware_timeout
                    )
                    firmware.wait_for(
                        "PROVISION_STATUS Broadcasting", timeout=args.firmware_timeout
                    )
                except (TimeoutError, RuntimeError) as err:
                    last_startup_error = err
                    if attempt == FIRMWARE_START_ATTEMPTS:
                        raise
                    print(
                        f"retrying firmware startup after {type(err).__name__}: {err} "
                        f"({attempt}/{FIRMWARE_START_ATTEMPTS})",
                        flush=True,
                    )
                    continue

                result = asyncio.run(run_test(args))
                if result != 0:
                    raise SystemExit(result)

                firmware.wait_for(
                    "NETWORK_DEVICE_INFO_SENT", timeout=args.firmware_timeout
                )
                firmware.wait_for("NETWORK_WS_CONNECTED", timeout=args.firmware_timeout)
                firmware.wait_for("NETWORK_ONLINE_SENT", timeout=args.firmware_timeout)
                firmware.wait_for("NETWORK_EVENT_SENT", timeout=args.firmware_timeout)
                firmware.wait_for("NETWORK_STREAM_DONE", timeout=args.firmware_timeout)

                device_info = server.wait_for_device_info(timeout=args.firmware_timeout)
                expected_device_id = assert_device_info(device_info)
                frames = server.wait_for_ws_frames(
                    scenario.batch_count + 2, timeout=args.firmware_timeout
                )
                return assert_ws_frames(frames, expected_device_id, scenario)
            finally:
                firmware.stop()

        assert last_startup_error is not None
        raise last_startup_error
    finally:
        restore_runtime_config(previous_config)
        server.stop()


def main() -> int:
    args = build_parser().parse_args()
    args.ssid, args.password = resolve_credentials(args)
    args.allow_name_fallback = True

    scenarios = build_scenarios(args)
    if args.skip_build and len(scenarios) > 1:
        raise SystemExit("--skip-build can only be used with a single scenario")

    host_ip = args.host_ip or resolve_bind_host_ip()
    firmware_env = build_firmware_env(args.serial_port)

    print(f"contract: docs/network-openapi.yaml", flush=True)
    print(f"scenario_count={len(scenarios)}", flush=True)

    completed: list[ScenarioResult] = []
    for index, scenario in enumerate(scenarios, start=1):
        print(f"scenario {index}/{len(scenarios)} {scenario.name}", flush=True)
        result = run_one_scenario(args, scenario, host_ip, firmware_env)
        print_benchmark(result)
        completed.append(result)

    if completed:
        peak = max(completed, key=lambda result: result.scenario.sample_rate_hz)
        print(
            "summary "
            f"max_verified_rate_hz={peak.scenario.sample_rate_hz} "
            f"scenarios_passed={len(completed)}",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
