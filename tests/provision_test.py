from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from bleak import BleakClient, BleakScanner

from tests.serial_capture import SerialCapture

PROVISION_SERVICE_UUID = "000012ff-0000-1000-8000-00805f9b34fb"
PROVISION_CREDENTIALS_UUID = "0000ff01-0000-1000-8000-00805f9b34fb"
PROVISION_STATUS_UUID = "0000ff02-0000-1000-8000-00805f9b34fb"
PROVISION_MANUFACTURER_ID = 0xFFFF
PROVISION_MAGIC = b"SG"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the SmartGlove BLE provisioning test.")
    parser.add_argument("--device-name", default="SmartGlove Provision")
    parser.add_argument("--ssid", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--scan-timeout", type=float, default=15.0)
    parser.add_argument("--status-timeout", type=float, default=50.0)
    parser.add_argument(
        "--allow-name-fallback",
        action="store_true",
        help="Allow matching by device name when the manufacturer marker is unavailable.",
    )
    parser.add_argument("--serial-port")
    parser.add_argument("--serial-baudrate", type=int, default=115200)
    parser.add_argument("--serial-log", type=Path)
    parser.add_argument(
        "--expect-serial",
        action="append",
        default=[],
        help="Serial substrings to wait for after the BLE workflow completes.",
    )
    return parser


async def run_test(args: argparse.Namespace) -> int:
    serial_capture: SerialCapture | None = None
    if args.serial_port:
        serial_capture = SerialCapture(
            args.serial_port,
            baudrate=args.serial_baudrate,
            log_path=args.serial_log,
        )
        serial_capture.start()

    try:
        if serial_capture is not None:
            try:
                serial_capture.wait_for("PROVISION_READY", timeout=10.0)
            except TimeoutError:
                print("warning: PROVISION_READY was not seen on serial before BLE scan", flush=True)

        device = await BleakScanner.find_device_by_filter(
            lambda _, advertisement: advertisement_matches(args, advertisement),
            timeout=args.scan_timeout,
        )
        if device is None:
            raise RuntimeError(f"unable to find BLE device {args.device_name!r}")

        print(f"found device: {device.address} ({device.name})", flush=True)

        status_queue: asyncio.Queue[str] = asyncio.Queue()

        def handle_status(_: str, data: bytearray) -> None:
            status = bytes(data).decode("utf-8", errors="replace")
            print(f"[ble] status={status}", flush=True)
            status_queue.put_nowait(status)

        async with BleakClient(device) as client:
            await client.start_notify(PROVISION_STATUS_UUID, handle_status)
            payload = f"{args.ssid}\n{args.password}".encode("utf-8")
            await client.write_gatt_char(PROVISION_CREDENTIALS_UUID, payload, response=True)

            while True:
                status = await asyncio.wait_for(status_queue.get(), timeout=args.status_timeout)
                if status == "connected":
                    print("provisioning succeeded", flush=True)
                    break
                if status.startswith("connection_failed:"):
                    raise RuntimeError(status)

        if serial_capture is not None:
            for needle in args.expect_serial:
                serial_capture.wait_for(needle, timeout=10.0)

        return 0
    finally:
        if serial_capture is not None:
            serial_capture.stop()


def main() -> int:
    args = build_parser().parse_args()
    return asyncio.run(run_test(args))


def advertisement_matches(args: argparse.Namespace, advertisement) -> bool:
    manufacturer_data = advertisement.manufacturer_data or {}
    marker = manufacturer_data.get(PROVISION_MANUFACTURER_ID)
    if marker == PROVISION_MAGIC:
        return True

    if getattr(args, "allow_name_fallback", False):
        return (
            advertisement.local_name == args.device_name
            or PROVISION_SERVICE_UUID.lower()
            in {service.lower() for service in (advertisement.service_uuids or [])}
        )

    return False


if __name__ == "__main__":
    raise SystemExit(main())
