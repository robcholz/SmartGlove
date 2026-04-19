from __future__ import annotations

import argparse
import asyncio
import os
import errno
import queue
import signal
import subprocess
import threading
import time
from pathlib import Path

from tests.provision_test import run_test


class ProcessCapture:
    def __init__(
        self,
        command: list[str],
        log_path: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self.command = command
        self.log_path = log_path
        self.env = env
        self.process: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lines: queue.Queue[str] = queue.Queue()
        self._master_fd: int | None = None

    def start(self) -> None:
        print(f"+ {' '.join(self.command)}", flush=True)
        if os.name == "nt":
            self.process = subprocess.Popen(
                self.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=self.env,
            )
        else:
            master_fd, slave_fd = os.openpty()
            self._master_fd = master_fd
            try:
                self.process = subprocess.Popen(
                    self.command,
                    stdin=slave_fd,
                    stdout=slave_fd,
                    stderr=slave_fd,
                    text=False,
                    close_fds=True,
                    env=self.env,
                )
            finally:
                os.close(slave_fd)
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self.process is None:
            return

        if self.process.poll() is None:
            try:
                if os.name == "nt":
                    self.process.terminate()
                else:
                    self.process.send_signal(signal.SIGINT)
            except ProcessLookupError:
                pass

            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)

        if self._thread is not None:
            self._thread.join(timeout=1.0)

        if self._master_fd is not None:
            os.close(self._master_fd)
            self._master_fd = None

    def wait_for(self, needle: str, timeout: float) -> str:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                line = self._lines.get(timeout=min(0.2, remaining))
            except queue.Empty:
                if self.process is not None and self.process.poll() is not None:
                    raise RuntimeError(
                        f"process exited before producing line containing {needle!r}"
                    ) from None
                continue
            if needle in line:
                return line
        raise TimeoutError(f"timed out waiting for process line containing: {needle}")

    def _read_loop(self) -> None:
        assert self.process is not None

        log_file = self.log_path.open("a", encoding="utf-8") if self.log_path else None
        try:
            if self._master_fd is not None:
                self._read_pty(log_file)
            else:
                assert self.process.stdout is not None
                for raw_line in self.process.stdout:
                    if self._stop_event.is_set():
                        break
                    self._handle_line(raw_line.rstrip("\n"), log_file)
        finally:
            if log_file is not None:
                log_file.close()

    def _read_pty(self, log_file) -> None:
        assert self._master_fd is not None
        buffer = ""

        while not self._stop_event.is_set():
            try:
                chunk = os.read(self._master_fd, 4096)
            except OSError as exc:
                if exc.errno == errno.EIO:
                    break
                raise

            if not chunk:
                break

            buffer += chunk.decode("utf-8", errors="replace")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                self._handle_line(line.rstrip("\r"), log_file)

        if buffer:
            self._handle_line(buffer.rstrip("\r"), log_file)

    def _handle_line(self, line: str, log_file) -> None:
        print(f"[firmware] {line}", flush=True)
        if log_file is not None:
            log_file.write(line + "\n")
            log_file.flush()
        self._lines.put(line)


def load_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("\"'")
    return values


def resolve_credentials(args: argparse.Namespace) -> tuple[str, str]:
    env_values = load_env_file(args.env_file)
    ssid = args.ssid or env_values.get("WIFI_SSID")
    password = args.password or env_values.get("WIFI_PASSWORD")

    if not ssid or password is None:
        raise SystemExit(
            "missing Wi-Fi credentials: pass --ssid/--password or define WIFI_SSID and "
            "WIFI_PASSWORD in .env.local"
        )

    return ssid, password


def resolve_espflash_port(preferred_port: str | None = None) -> str:
    if preferred_port:
        return preferred_port

    env_port = os.environ.get("ESPFLASH_PORT")
    if env_port:
        return env_port

    completed = subprocess.run(
        ["espflash", "list-ports"],
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        raise SystemExit(
            "failed to resolve ESPFLASH_PORT: `espflash list-ports` did not succeed"
        )

    raw_ports = [
        line.split()[0]
        for line in completed.stdout.splitlines()
        if line.startswith("/dev/")
    ]
    preferred_ports = [port for port in raw_ports if port.startswith("/dev/cu.")]
    candidates = sorted(dict.fromkeys(preferred_ports or raw_ports))
    if not candidates:
        raise SystemExit("failed to resolve ESPFLASH_PORT: no ESP serial ports found")
    if len(candidates) != 1:
        raise SystemExit(
            "failed to resolve ESPFLASH_PORT automatically: found "
            + ", ".join(candidates)
            + "; pass --serial-port or set ESPFLASH_PORT"
        )

    return candidates[0]


def build_firmware_env(serial_port: str | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["CARGO_TERM_COLOR"] = "always"
    env["ESPFLASH_PORT"] = resolve_espflash_port(serial_port)
    print(f"using ESPFLASH_PORT={env['ESPFLASH_PORT']}", flush=True)
    return env


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build, flash, and run the SmartGlove BLE provisioning test end to end."
    )
    parser.add_argument("--ssid")
    parser.add_argument("--password")
    parser.add_argument("--env-file", type=Path, default=Path(".env.local"))
    parser.add_argument("--device-name", default="SmartGlove Provision")
    parser.add_argument("--scan-timeout", type=float, default=15.0)
    parser.add_argument("--status-timeout", type=float, default=30.0)
    parser.add_argument("--firmware-timeout", type=float, default=30.0)
    parser.add_argument("--firmware-log", type=Path)
    parser.add_argument("--serial-port")
    parser.add_argument("--serial-baudrate", type=int, default=115200)
    parser.add_argument("--serial-log", type=Path)
    parser.add_argument("--expect-serial", action="append", default=[])
    parser.add_argument(
        "--expect-firmware",
        action="append",
        default=["PROVISION_DONE success"],
        help="Firmware log substrings to verify after BLE provisioning succeeds.",
    )
    parser.add_argument("--build-mode", choices=["release", "debug"], default="release")
    parser.add_argument("--skip-build", action="store_true")
    return parser


def cargo_profile_flag(build_mode: str) -> list[str]:
    return ["--release"] if build_mode == "release" else []


def run_command(command: list[str], env: dict[str, str] | None = None) -> None:
    print(f"+ {' '.join(command)}", flush=True)
    completed = subprocess.run(command, check=False, env=env)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> int:
    args = build_parser().parse_args()
    args.ssid, args.password = resolve_credentials(args)
    firmware_env = build_firmware_env(args.serial_port)

    if not args.skip_build:
        run_command(
            [
                "cargo",
                "build",
                *cargo_profile_flag(args.build_mode),
                "--bin",
                "provision",
            ],
            env=firmware_env,
        )

    firmware = ProcessCapture(
        ["cargo", "run", *cargo_profile_flag(args.build_mode), "--bin", "provision"],
        log_path=args.firmware_log,
        env=firmware_env,
    )
    firmware.start()

    try:
        firmware.wait_for("PROVISION_READY", timeout=args.firmware_timeout)
        firmware.wait_for(
            "PROVISION_STATUS Broadcasting", timeout=args.firmware_timeout
        )

        result = asyncio.run(run_test(args))

        for needle in args.expect_firmware:
            firmware.wait_for(needle, timeout=args.firmware_timeout)

        return result
    finally:
        firmware.stop()


if __name__ == "__main__":
    raise SystemExit(main())
