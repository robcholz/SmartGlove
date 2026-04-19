from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.gesture_model import (
    DATASET_DIR,
    DEFAULT_SEED,
    DEFAULT_SEQUENCE_LENGTH,
    FEATURE_COLUMNS,
    PROCESSED_DATA_DIR,
    ProcessedDataMetadata,
    SessionRecord,
    discover_dataset_records,
    load_session_array,
    print_dataset_summary,
    processed_feature_path,
    processed_label_path,
    processed_length_path,
    processed_metadata_path,
    resample_sequence,
    save_array,
    stratified_split_records,
    write_processed_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resample raw gesture sessions into a fixed-length processed dataset."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def build_split_arrays(
    split_records: list[tuple[Path, str]],
    labels: list[str],
    split_name: str,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[SessionRecord]]:
    label_to_index = {label: index for index, label in enumerate(labels)}
    features: list[np.ndarray] = []
    label_indices: list[int] = []
    original_lengths: list[int] = []
    manifest_records: list[SessionRecord] = []

    for csv_path, label in split_records:
        session_values = load_session_array(csv_path)
        resampled = resample_sequence(session_values, sequence_length)
        label_index = label_to_index[label]

        features.append(resampled)
        label_indices.append(label_index)
        original_lengths.append(len(session_values))
        manifest_records.append(
            SessionRecord(
                path=str(csv_path),
                label=label,
                label_index=label_index,
                split=split_name,
                original_length=len(session_values),
                resampled_length=sequence_length,
            )
        )

    return (
        np.stack(features).astype(np.float32),
        np.asarray(label_indices, dtype=np.int64),
        np.asarray(original_lengths, dtype=np.int64),
        manifest_records,
    )


def main() -> None:
    args = parse_args()
    raw_records = discover_dataset_records(args.dataset_dir)
    labels = sorted({label for _, label in raw_records})
    splits = stratified_split_records(
        raw_records,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    manifest_records: list[SessionRecord] = []
    split_counts: dict[str, int] = {}
    for split_name, split_records in splits.items():
        feature_array, label_array, length_array, split_manifest = build_split_arrays(
            split_records=split_records,
            labels=labels,
            split_name=split_name,
            sequence_length=args.sequence_length,
        )
        save_array(processed_feature_path(args.output_dir, split_name), feature_array)
        save_array(processed_label_path(args.output_dir, split_name), label_array)
        save_array(processed_length_path(args.output_dir, split_name), length_array)
        manifest_records.extend(split_manifest)
        split_counts[split_name] = int(len(split_records))

        print(
            f"{split_name}_split sessions={len(split_records)} "
            f"features_shape={feature_array.shape}"
        )

    original_lengths = np.asarray([record.original_length for record in manifest_records], dtype=np.int64)
    label_counts = Counter(record.label for record in manifest_records)
    metadata = ProcessedDataMetadata(
        dataset_dir=str(args.dataset_dir.resolve()),
        processed_dir=str(args.output_dir.resolve()),
        sequence_length=args.sequence_length,
        feature_columns=FEATURE_COLUMNS,
        labels=labels,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        split_counts=split_counts,
        label_counts=dict(sorted(label_counts.items())),
        original_length_stats={
            "min": float(original_lengths.min()),
            "max": float(original_lengths.max()),
            "mean": float(original_lengths.mean()),
            "median": float(np.median(original_lengths)),
        },
        records=manifest_records,
    )
    write_processed_metadata(metadata, processed_metadata_path(args.output_dir))
    print_dataset_summary(metadata)


if __name__ == "__main__":
    main()
