#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from sklearn.metrics import accuracy_score

from training.postprocess import (
    GestureConvNet,
    TRAINING_DIR,
    build_label_encoder_from_metadata,
    build_scaler_from_metadata,
    build_windows,
    export_onnx,
    load_dataframe,
    load_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SmartGlove inference in Python against the dataset"
    )
    parser.add_argument("--weights", type=Path, default=TRAINING_DIR / "glove_model.pt")
    parser.add_argument("--onnx", type=Path, default=TRAINING_DIR / "glove_model.onnx")
    parser.add_argument(
        "--metadata", type=Path, default=TRAINING_DIR / "glove_model_metadata.json"
    )
    parser.add_argument("--split", choices=["train", "val", "all"], default="val")
    parser.add_argument(
        "--limit", type=int, default=5, help="How many sample predictions to print"
    )
    return parser.parse_args()


def select_split_rows(df, split: str, metadata):
    if split == "all":
        return df.copy()
    subjects = metadata.train_subjects if split == "train" else metadata.val_subjects
    return df[df[metadata.subject_column].astype(str).isin(subjects)].copy()


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def main() -> None:
    args = parse_args()
    metadata = load_metadata(args.metadata)
    label_encoder = build_label_encoder_from_metadata(metadata)
    scaler = build_scaler_from_metadata(metadata)

    df, _ = load_dataframe([Path(path) for path in metadata.csv_files])
    split_df = select_split_rows(df, args.split, metadata)
    windows, labels = build_windows(
        split_df,
        label_encoder=label_encoder,
        scaler=scaler,
        majority_threshold=metadata.majority_threshold,
    )

    model = GestureConvNet(
        input_channels=len(metadata.sensor_columns),
        num_classes=len(metadata.label_classes),
    )
    state_dict = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        torch_logits = model(torch.from_numpy(windows).float()).cpu().numpy()

    if not args.onnx.exists():
        export_onnx(model, args.onnx)

    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    onnx_logits = np.stack(
        [
            session.run(
                None, {input_name: windows[index : index + 1].astype(np.float32)}
            )[0][0]
            for index in range(len(windows))
        ],
        axis=0,
    )

    torch_pred = torch_logits.argmax(axis=1)
    onnx_pred = onnx_logits.argmax(axis=1)
    truth = labels.astype(np.int64)

    print(f"labels={metadata.label_classes}")
    print(f"split={args.split} windows={len(windows)}")
    print(f"torch_accuracy={accuracy_score(truth, torch_pred):.4f}")
    print(f"onnx_accuracy={accuracy_score(truth, onnx_pred):.4f}")
    print(f"torch_vs_onnx_agreement={(torch_pred == onnx_pred).mean():.4f}")
    print(f"onnx_max_abs_error={np.max(np.abs(torch_logits - onnx_logits)):.8f}")

    sample_count = min(args.limit, len(windows))
    if sample_count == 0:
        return

    probs = softmax(onnx_logits[:sample_count])
    for index in range(sample_count):
        predicted_index = int(onnx_pred[index])
        actual_index = int(truth[index])
        print(
            f"sample={index} actual={metadata.label_classes[actual_index]} "
            f"predicted={metadata.label_classes[predicted_index]} "
            f"confidence={probs[index, predicted_index]:.4f}"
        )


if __name__ == "__main__":
    main()
