from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


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
NUMERIC_FIELDS = [field for field in CSV_FIELDS if field != "label"]
ACC_FIELDS = ["acc_x", "acc_y", "acc_z"]
VEC_FIELDS = ["vec_x", "vec_y", "vec_z"]
FLEX_FIELDS = ["thumb", "index", "middle", "ring", "pinky"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize one SmartGlove capture CSV."
    )
    parser.add_argument("csv_path", type=Path, help="Path to the captured CSV file.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional image output path. If set, save the figure there.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the interactive plot window.",
    )
    return parser


def load_csv(csv_path: Path) -> tuple[dict[str, list[float]], str | None]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != CSV_FIELDS:
            raise ValueError(
                f"unexpected CSV columns: {reader.fieldnames}; expected {CSV_FIELDS}"
            )

        columns = {field: [] for field in NUMERIC_FIELDS}
        labels: set[str] = set()
        for row in reader:
            for field in NUMERIC_FIELDS:
                columns[field].append(float(row[field]))
            labels.add(row["label"])

    label = next(iter(labels)) if len(labels) == 1 else None
    return columns, label


def plot_group(
    axis: plt.Axes,
    x_values: list[int],
    columns: dict[str, list[float]],
    fields: list[str],
    title: str,
) -> None:
    for field in fields:
        axis.plot(x_values, columns[field], label=field, linewidth=1.2)
    axis.set_title(title)
    axis.set_xlabel("sample")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="upper right")


def main() -> int:
    args = build_parser().parse_args()
    csv_path = args.csv_path.expanduser().resolve()
    columns, label = load_csv(csv_path)
    if not columns[NUMERIC_FIELDS[0]]:
        print(f"no samples found in {csv_path}", flush=True)
        return 0
    sample_count = len(columns[NUMERIC_FIELDS[0]])
    x_values = list(range(sample_count))

    figure, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    title = f"SmartGlove Capture: {csv_path.name} ({sample_count} samples)"
    if label:
        title += f" label={label}"
    figure.suptitle(title)

    plot_group(axes[0], x_values, columns, ACC_FIELDS, "Accelerometer")
    plot_group(axes[1], x_values, columns, VEC_FIELDS, "Gyroscope / Vec")
    plot_group(axes[2], x_values, columns, FLEX_FIELDS, "Flex Sensors")
    axes[2].set_xlabel("sample index")

    figure.tight_layout()

    if args.output is not None:
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=160, bbox_inches="tight")
        print(f"saved figure to {output_path}", flush=True)

    if not args.no_show:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
