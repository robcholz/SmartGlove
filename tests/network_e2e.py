from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from tests.network_server import MockNetworkServer, resolve_bind_host_ip
from tests.provision_e2e import (
    ProcessCapture,
    cargo_profile_flag,
    resolve_credentials,
    run_command,
)
from tests.provision_test import run_test

CONFIG_PATH = Path("tests/network_runtime_config.env")
EXPECTED_EVENT_NAME = "network.ready"
EXPECTED_EVENT_PAYLOAD = '{"phase":"provisioned"}'


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build, flash, and run the SmartGlove network end-to-end test."
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
    return parser


def write_runtime_config(host_ip: str, port: int) -> str:
    previous = CONFIG_PATH.read_text(encoding="utf-8")
    CONFIG_PATH.write_text(
        "\n".join(
            [
                f"DEVICE_INFO_URL=http://{host_ip}:{port}/v1/device-info",
                f"STATUS_WS_URL=ws://{host_ip}:{port}/v1/ws",
                f"EVENT_NAME={EXPECTED_EVENT_NAME}",
                f"EVENT_PAYLOAD_JSON={EXPECTED_EVENT_PAYLOAD}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return previous


def restore_runtime_config(previous: str) -> None:
    CONFIG_PATH.write_text(previous, encoding="utf-8")


def build_firmware_env() -> dict[str, str]:
    env = os.environ.copy()
    env["CARGO_TERM_COLOR"] = "always"
    return env


def assert_device_info(payload: dict) -> str:
    assert isinstance(payload.get("device_id"), str), payload
    assert len(payload["device_id"]) == 12, payload
    assert payload["events"] == [EXPECTED_EVENT_NAME], payload
    return payload["device_id"]


def assert_ws_messages(messages: list[dict], expected_device_id: str) -> None:
    assert len(messages) >= 3, messages

    online, event, batch = messages[:3]

    assert online == {
        "kind": "device.online",
        "device_id": expected_device_id,
    }, online

    assert event == {
        "kind": "device.event",
        "device_id": expected_device_id,
        "event": EXPECTED_EVENT_NAME,
        "payload": {"phase": "provisioned"},
    }, event

    assert batch["kind"] == "sensor.batch", batch
    assert batch["device_id"] == expected_device_id, batch
    assert batch["sample_rate_hz"] == 100, batch
    assert isinstance(batch["start_tick_ms"], int) and batch["start_tick_ms"] >= 0, batch
    assert len(batch["samples"]) == 10, batch

    first_sample = batch["samples"][0]
    assert set(first_sample.keys()) == {"imu_acc", "vel", "flex"}, first_sample
    assert len(first_sample["imu_acc"]) == 3, first_sample
    assert len(first_sample["vel"]) == 3, first_sample
    assert set(first_sample["flex"].keys()) == {
        "thumb",
        "index",
        "middle",
        "ring",
        "pinky",
    }, first_sample


def main() -> int:
    args = build_parser().parse_args()
    args.ssid, args.password = resolve_credentials(args)
    args.allow_name_fallback = True

    host_ip = args.host_ip or resolve_bind_host_ip()
    server = MockNetworkServer()
    server.start()
    assert server.actual_port is not None

    previous_config = write_runtime_config(host_ip, server.actual_port)
    firmware_env = build_firmware_env()

    try:
        if not args.skip_build:
            run_command(
                ["cargo", "build", *cargo_profile_flag(args.build_mode), "--bin", "provision_network"],
                env=firmware_env,
            )

        firmware = ProcessCapture(
            ["cargo", "run", *cargo_profile_flag(args.build_mode), "--bin", "provision_network"],
            log_path=args.firmware_log,
            env=firmware_env,
        )
        firmware.start()

        try:
            firmware.wait_for("PROVISION_NETWORK_READY", timeout=args.firmware_timeout)
            firmware.wait_for("PROVISION_STATUS Broadcasting", timeout=args.firmware_timeout)

            result = asyncio.run(run_test(args))

            firmware.wait_for("NETWORK_DEVICE_INFO_SENT", timeout=args.firmware_timeout)
            firmware.wait_for("NETWORK_WS_CONNECTED", timeout=args.firmware_timeout)
            firmware.wait_for("NETWORK_ONLINE_SENT", timeout=args.firmware_timeout)
            firmware.wait_for("NETWORK_EVENT_SENT", timeout=args.firmware_timeout)
            firmware.wait_for("NETWORK_BATCH_SENT", timeout=args.firmware_timeout)

            device_info = server.wait_for_device_info(timeout=args.firmware_timeout)
            expected_device_id = assert_device_info(device_info)

            messages = server.wait_for_ws_messages(3, timeout=args.firmware_timeout)
            assert_ws_messages(messages, expected_device_id)

            return result
        finally:
            firmware.stop()
    finally:
        restore_runtime_config(previous_config)
        server.stop()


if __name__ == "__main__":
    raise SystemExit(main())
