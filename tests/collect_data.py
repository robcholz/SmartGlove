from __future__ import annotations

import argparse
import csv
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


AXIS_RE = re.compile(
    r"imu (?P<kind>acc|vec): x=(?P<x>-?\d+(?:\.\d+)?), y=(?P<y>-?\d+(?:\.\d+)?), z=(?P<z>-?\d+(?:\.\d+)?)"
)
FINGER_RE = re.compile(
    r"(?P<finger>thumb|index|middle|ring|pinky) flex sensor: (?P<value>-?\d+)"
)
CSV_FIELDS = [
    "captured_at_local",
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
]


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
        required = CSV_FIELDS[1:]
        return all(field in self.values for field in required)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run driver_probe and save parsed IMU + flex sensor samples to CSV."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory for the captured CSV. Defaults to the current directory.",
    )
    return parser


def build_output_path(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    return output_dir / f"{timestamp}.csv"


def emit_sample(writer: csv.DictWriter, csv_file, sample: SampleBuffer) -> None:
    row = {
        "captured_at_local": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        **sample.values,
    }
    writer.writerow(row)
    csv_file.flush()
    sample.reset()


def update_sample(sample: SampleBuffer, line: str) -> bool:
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


def main() -> int:
    args = build_parser().parse_args()
    output_path = build_output_path(args.output_dir)
    sample = SampleBuffer()
    samples_written = 0
    interrupted = False

    command = ["cargo", "run", "--bin", "driver_probe"]
    print(f"+ {' '.join(command)}", flush=True)
    print(f"writing samples to {output_path}", flush=True)

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert process.stdout is not None

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()

        try:
            for raw_line in process.stdout:
                line = raw_line.rstrip()
                print(line, flush=True)
                if update_sample(sample, line):
                    emit_sample(writer, csv_file, sample)
                    samples_written += 1
        except KeyboardInterrupt:
            interrupted = True
            print("\nstopping capture...", flush=True)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()

    print(f"saved {samples_written} samples to {output_path}", flush=True)
    if interrupted:
        return 0
    return process.returncode or 0


if __name__ == "__main__":
    raise SystemExit(main())
