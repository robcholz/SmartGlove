from __future__ import annotations

import argparse
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import serial


@dataclass
class SerialCapture:
    port: str
    baudrate: int = 115200
    log_path: Path | None = None
    echo: bool = True
    _serial: serial.Serial | None = field(init=False, default=None)
    _thread: threading.Thread | None = field(init=False, default=None)
    _stop_event: threading.Event = field(init=False, default_factory=threading.Event)
    _lines: queue.Queue[str] = field(init=False, default_factory=queue.Queue)

    def start(self) -> None:
        self._serial = serial.Serial(self.port, self.baudrate, timeout=0.2)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._serial is not None:
            self._serial.close()
            self._serial = None

    def wait_for(self, needle: str, timeout: float) -> str:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                line = self._lines.get(timeout=min(0.2, remaining))
            except queue.Empty:
                continue
            if needle in line:
                return line
        raise TimeoutError(f"timed out waiting for serial line containing: {needle}")

    def _read_loop(self) -> None:
        assert self._serial is not None

        log_file = self.log_path.open("a", encoding="utf-8") if self.log_path else None
        try:
            while not self._stop_event.is_set():
                raw = self._serial.readline()
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="replace").rstrip()
                if self.echo:
                    print(f"[serial] {line}", flush=True)
                if log_file is not None:
                    log_file.write(line + "\n")
                    log_file.flush()
                self._lines.put(line)
        finally:
            if log_file is not None:
                log_file.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture ESP serial output.")
    parser.add_argument("--port", required=True, help="Serial port, for example /dev/tty.usbmodem1101")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--log-path", type=Path)
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds to capture. 0 means until Ctrl+C.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    capture = SerialCapture(args.port, baudrate=args.baudrate, log_path=args.log_path)
    capture.start()
    try:
        if args.duration > 0:
            time.sleep(args.duration)
        else:
            while True:
                time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        capture.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
