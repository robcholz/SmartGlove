from __future__ import annotations

import argparse
import csv
import os
import re
import select
import subprocess
import sys
import termios
import threading
import time
import tty
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import serial
from serial.tools import list_ports


AXIS_RE = re.compile(
    r"imu (?P<kind>acc|vec): x=(?P<x>-?\d+(?:\.\d+)?), y=(?P<y>-?\d+(?:\.\d+)?), z=(?P<z>-?\d+(?:\.\d+)?)"
)
FINGER_RE = re.compile(
    r"(?P<finger>thumb|index|middle|ring|pinky) flex sensor: (?P<value>-?\d+)"
)
SAMPLE_RE = re.compile(
    r"sample acc=(?P<acc_x>-?\d+(?:\.\d+)?),(?P<acc_y>-?\d+(?:\.\d+)?),(?P<acc_z>-?\d+(?:\.\d+)?) "
    r"vec=(?P<vec_x>-?\d+(?:\.\d+)?),(?P<vec_y>-?\d+(?:\.\d+)?),(?P<vec_z>-?\d+(?:\.\d+)?) "
    r"flex=(?P<thumb>-?\d+),(?P<index>-?\d+),(?P<middle>-?\d+),(?P<ring>-?\d+),(?P<pinky>-?\d+)"
)
SESSION_FILE_RE = re.compile(r"^(?P<timestamp>\d+)_session(?P<index>\d+)\.csv$")
USB_PORT_RE = re.compile(r"(usb|serial|uart|slab|wch|acm)", re.IGNORECASE)
CSV_FIELDS = [
    "acc_x",
    "acc_y",
    "acc_z",
    "vec_x",
    "vec_y",
    "vec_z",
    "thumb",
    "index",
    "middle",
    "ring",
    "pinky",
    "label",
]
SAMPLE_FIELDS = [field for field in CSV_FIELDS if field != "label"]


@dataclass
class SampleBuffer:
    values: dict[str, str] = field(default_factory=dict)

    def reset(self) -> None:
        self.values.clear()

    def start_sample(self, x: str, y: str, z: str) -> None:
        self.values = {
            "acc_x": x,
            "acc_y": y,
            "acc_z": z,
        }

    def has_active_sample(self) -> bool:
        return bool(self.values)

    def is_complete(self) -> bool:
        return all(field in self.values for field in SAMPLE_FIELDS)


@dataclass
class SessionHandle:
    session_index: int
    output_path: Path
    csv_file: object
    writer: csv.DictWriter
    label: str
    samples_written: int = 0


@dataclass
class OutputState:
    recent_lines: deque[str] = field(default_factory=lambda: deque(maxlen=80))
    lock: threading.Lock = field(default_factory=threading.Lock)

    def push(self, line: str) -> None:
        with self.lock:
            self.recent_lines.append(line)

    def tail(self) -> list[str]:
        with self.lock:
            return list(self.recent_lines)


class SessionRecorder:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: SessionHandle | None = None

    def start_session(self, output_path: Path, session_index: int, label: str) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = output_path.open("w", newline="", encoding="utf-8")
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        csv_file.flush()

        with self._lock:
            self._active = SessionHandle(
                session_index=session_index,
                output_path=output_path,
                csv_file=csv_file,
                writer=writer,
                label=label,
            )

    def stop_session(self) -> tuple[Path, int] | None:
        with self._lock:
            active = self._active
            self._active = None

        if active is None:
            return None

        active.csv_file.close()
        return active.output_path, active.samples_written

    def emit_sample(self, sample: SampleBuffer) -> bool:
        with self._lock:
            active = self._active
            if active is None:
                sample.reset()
                return False

            row = dict(sample.values)
            row["label"] = active.label

            active.writer.writerow(row)
            active.csv_file.flush()
            active.samples_written += 1

        sample.reset()
        return True

    def is_recording(self) -> bool:
        with self._lock:
            return self._active is not None


class RawKeyboard:
    def __enter__(self) -> "RawKeyboard":
        self._fd = sys.stdin.fileno()
        self._old_settings = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)

    def read_key(self, timeout: float = 0.1) -> str | None:
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        if not ready:
            return None
        return sys.stdin.read(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect labeled IMU + flex sensor sessions from the glove."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("dataset"),
        help="Root directory for labeled session CSVs. Defaults to ./dataset.",
    )
    parser.add_argument(
        "--bin",
        default="driver",
        help="Cargo bin to run for sensor streaming. Defaults to driver.",
    )
    parser.add_argument(
        "--port",
        help="Serial port for espflash. Defaults to ESPFLASH_PORT or an interactive selection.",
    )
    parser.add_argument(
        "--verbose-stream",
        action="store_true",
        help="Print raw serial sensor lines while capturing.",
    )
    return parser


def update_sample(sample: SampleBuffer, line: str) -> bool:
    sample_match = SAMPLE_RE.search(line)
    if sample_match:
        sample.values = {field: sample_match.group(field) for field in SAMPLE_FIELDS}
        return sample.is_complete()

    axis_match = AXIS_RE.search(line)
    if axis_match:
        kind = axis_match.group("kind")
        x = axis_match.group("x")
        y = axis_match.group("y")
        z = axis_match.group("z")

        if kind == "acc":
            sample.start_sample(x, y, z)
        elif sample.has_active_sample():
            sample.values["vec_x"] = x
            sample.values["vec_y"] = y
            sample.values["vec_z"] = z
        return sample.is_complete()

    finger_match = FINGER_RE.search(line)
    if finger_match and sample.has_active_sample():
        sample.values[finger_match.group("finger")] = finger_match.group("value")
        return sample.is_complete()

    return False


def load_existing_labels(dataset_dir: Path) -> list[str]:
    if not dataset_dir.exists():
        return []
    return sorted(path.name for path in dataset_dir.iterdir() if path.is_dir())


def prompt_label(dataset_dir: Path) -> str:
    existing_labels = load_existing_labels(dataset_dir)
    print("Select a label to record.", flush=True)
    if existing_labels:
        print("Existing labels:", flush=True)
        for index, label in enumerate(existing_labels, start=1):
            print(f"  {index}. {label}", flush=True)
        print("  n. new label", flush=True)

    while True:
        if existing_labels:
            raw = input("label choice or new label name: ").strip()
            if not raw:
                continue
            if raw.lower() == "n":
                label = input("new label name: ").strip()
            elif raw.isdigit() and 1 <= int(raw) <= len(existing_labels):
                label = existing_labels[int(raw) - 1]
            else:
                label = raw
        else:
            label = input("label name: ").strip()

        if not label:
            print("label cannot be empty", flush=True)
            continue
        if any(part in {"", ".", ".."} for part in label.split("/")) or "/" in label:
            print("label cannot contain '/' or parent-path segments", flush=True)
            continue
        return label


def prompt_session_count() -> int:
    while True:
        raw = input("number of sessions to record: ").strip()
        try:
            count = int(raw)
        except ValueError:
            print("please enter a whole number", flush=True)
            continue

        if count <= 0:
            print("session count must be at least 1", flush=True)
            continue
        return count


def next_session_index(label_dir: Path) -> int:
    max_index = 0
    if label_dir.exists():
        for path in label_dir.glob("*_session*.csv"):
            match = SESSION_FILE_RE.match(path.name)
            if match:
                max_index = max(max_index, int(match.group("index")))
    return max_index + 1


def build_output_path(label_dir: Path, session_index: int) -> Path:
    timestamp = int(time.time())
    return label_dir / f"{timestamp}_session{session_index}.csv"


def list_serial_ports() -> list[str]:
    ports = sorted(port.device for port in list_ports.comports())
    preferred = [port for port in ports if USB_PORT_RE.search(port)]
    others = [port for port in ports if port not in preferred]
    return preferred + others


def prompt_serial_port(cli_port: str | None) -> str:
    env_port = os.environ.get("ESPFLASH_PORT")
    if cli_port:
        print(f"using serial port from --port: {cli_port}", flush=True)
        return cli_port
    if env_port:
        print(f"using serial port from ESPFLASH_PORT: {env_port}", flush=True)
        return env_port

    ports = list_serial_ports()
    if not ports:
        raise RuntimeError("no serial ports found; connect the glove and try again")
    print(f"auto-selected serial port: {ports[0]}", flush=True)
    return ports[0]


def wait_for_space(
    keyboard: RawKeyboard,
    serial_failed: threading.Event,
    prompt: str,
) -> None:
    if prompt:
        print(prompt, flush=True)
    while True:
        if serial_failed.is_set():
            raise RuntimeError("serial reader stopped while waiting for input")

        key = keyboard.read_key(timeout=0.1)
        if key is None:
            continue
        if key == " ":
            return
        if key in {"\x03", "\x04"}:
            raise KeyboardInterrupt


def stream_sensor_output(
    serial_port: serial.Serial,
    recorder: SessionRecorder,
    serial_ready: threading.Event,
    serial_failed: threading.Event,
    output_state: OutputState,
    verbose_stream: bool,
    stop_event: threading.Event,
) -> None:
    sample = SampleBuffer()
    try:
        while not stop_event.is_set():
            try:
                raw_line = serial_port.readline()
            except serial.SerialException as exc:
                if stop_event.is_set():
                    break
                output_state.push(f"serial read error: {exc}")
                serial_failed.set()
                break
            if not raw_line:
                continue

            line = raw_line.decode("utf-8", errors="replace").rstrip()
            output_state.push(line)
            is_sensor_line = bool(
                SAMPLE_RE.search(line) or AXIS_RE.search(line) or FINGER_RE.search(line)
            )
            if verbose_stream or not is_sensor_line or not serial_ready.is_set():
                print(line, flush=True)
            if is_sensor_line:
                serial_ready.set()
            if update_sample(sample, line):
                recorder.emit_sample(sample)
    finally:
        if not stop_event.is_set():
            serial_failed.set()


def wait_for_serial_data(
    serial_ready: threading.Event,
    serial_failed: threading.Event,
    output_state: OutputState,
) -> None:
    print("waiting for live sensor data from the serial monitor...", flush=True)
    while True:
        if serial_ready.wait(timeout=0.1):
            print("sensor stream detected; ready to record", flush=True)
            return
        if serial_failed.is_set():
            tail = output_state.tail()
            if tail:
                print("recent subprocess output:", flush=True)
                for line in tail[-20:]:
                    print(line, flush=True)
            raise RuntimeError("serial reader stopped before any sensor data was seen")


def firmware_image_path(bin_name: str) -> Path:
    return Path("target") / "xtensa-esp32s3-espidf" / "debug" / bin_name


def run_command(command: list[str], env: dict[str, str]) -> None:
    print(f"+ {' '.join(command)}", flush=True)
    subprocess.run(command, check=True, env=env)


def open_serial_with_retry(
    port: str, baudrate: int = 115200, timeout: float = 0.1
) -> serial.Serial:
    deadline = time.time() + 10.0
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            return serial.Serial(port=port, baudrate=baudrate, timeout=timeout)
        except serial.SerialException as exc:
            last_error = exc
            time.sleep(0.25)
    raise RuntimeError(f"failed to open serial port {port}: {last_error}")


def main() -> int:
    args = build_parser().parse_args()
    dataset_dir = args.dataset_dir
    port = prompt_serial_port(args.port)
    print(f"espflash port: {port}", flush=True)

    env = os.environ.copy()
    env["ESPFLASH_PORT"] = port
    image_path = firmware_image_path(args.bin)
    run_command(["cargo", "build", "--bin", args.bin], env)
    run_command(
        [
            "espflash",
            "flash",
            "--port",
            port,
            "--non-interactive",
            str(image_path),
        ],
        env,
    )

    serial_port = open_serial_with_retry(port)
    recorder = SessionRecorder()
    serial_ready = threading.Event()
    serial_failed = threading.Event()
    stop_event = threading.Event()
    output_state = OutputState()
    reader = threading.Thread(
        target=stream_sensor_output,
        args=(
            serial_port,
            recorder,
            serial_ready,
            serial_failed,
            output_state,
            args.verbose_stream,
            stop_event,
        ),
        daemon=True,
    )
    reader.start()

    completed_sessions = 0
    session_count = 0

    try:
        wait_for_serial_data(serial_ready, serial_failed, output_state)
        label = prompt_label(dataset_dir)
        session_count = prompt_session_count()
        label_dir = dataset_dir / label
        first_session_index = next_session_index(label_dir)
        print(f"dataset label directory: {label_dir}", flush=True)
        with RawKeyboard() as keyboard:
            for offset in range(session_count):
                session_index = first_session_index + offset
                output_path = build_output_path(label_dir, session_index)

                wait_for_space(
                    keyboard,
                    serial_failed,
                    (
                        f"Session {offset + 1}/{session_count} "
                        f"(session index {session_index}). Press space to start recording."
                    ),
                )

                recorder.start_session(output_path, session_index, label)
                print(
                    f"recording session {session_index} -> {output_path}. Press space to stop.",
                    flush=True,
                )

                wait_for_space(keyboard, serial_failed, "")
                result = recorder.stop_session()
                if result is None:
                    raise RuntimeError("recording session was not active")

                saved_path, samples_written = result
                completed_sessions += 1
                print(
                    f"saved session {session_index}: {samples_written} samples -> {saved_path}",
                    flush=True,
                )
    except KeyboardInterrupt:
        print("\nstopping capture...", flush=True)
    finally:
        result = recorder.stop_session()
        if result is not None:
            saved_path, samples_written = result
            print(
                f"saved partial session: {samples_written} samples -> {saved_path}",
                flush=True,
            )

        stop_event.set()
        reader.join(timeout=2)
        try:
            serial_port.close()
        except Exception:
            pass

    print(f"completed sessions: {completed_sessions}/{session_count}", flush=True)
    if (
        serial_failed.is_set()
        and not stop_event.is_set()
        and completed_sessions < session_count
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
