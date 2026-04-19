from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.gesture_model import (
    GestureMLP,
    PROCESSED_DATA_DIR,
    TRAINING_DIR,
    build_scaler_from_metadata,
    confusion_matrix_payload,
    evaluate_predictions,
    load_model_metadata,
    load_processed_split,
    make_artifact_paths,
    per_class_accuracy,
    softmax,
    transform_features,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the trained gesture classifier on the held-out test split."
    )
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=TRAINING_DIR)
    parser.add_argument("--repeat", type=int, default=50)
    return parser.parse_args()


def benchmark_torch(model: GestureMLP, features: np.ndarray, repeat: int) -> dict[str, float]:
    sample = torch.from_numpy(features[:1]).float()
    for _ in range(5):
        with torch.no_grad():
            model(sample)

    timings_ms: list[float] = []
    for _ in range(repeat):
        start = time.perf_counter()
        with torch.no_grad():
            model(sample)
        timings_ms.append((time.perf_counter() - start) * 1000.0)
    return {
        "mean_ms": float(np.mean(timings_ms)),
        "median_ms": float(np.median(timings_ms)),
        "p95_ms": float(np.percentile(timings_ms, 95)),
    }


def benchmark_onnx(session: ort.InferenceSession, input_name: str, features: np.ndarray, repeat: int) -> dict[str, float]:
    sample = features[:1].astype(np.float32)
    for _ in range(5):
        session.run(None, {input_name: sample})

    timings_ms: list[float] = []
    for _ in range(repeat):
        start = time.perf_counter()
        session.run(None, {input_name: sample})
        timings_ms.append((time.perf_counter() - start) * 1000.0)
    return {
        "mean_ms": float(np.mean(timings_ms)),
        "median_ms": float(np.median(timings_ms)),
        "p95_ms": float(np.percentile(timings_ms, 95)),
    }


def predict_onnx_logits(session: ort.InferenceSession, input_name: str, features: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            session.run(None, {input_name: features[index : index + 1].astype(np.float32)})[0][0]
            for index in range(len(features))
        ],
        axis=0,
    )


def main() -> None:
    args = parse_args()
    artifacts = make_artifact_paths(args.artifacts_dir)
    metadata = load_model_metadata(artifacts.metadata_json)
    scaler = build_scaler_from_metadata(metadata)

    test_features, test_labels = load_processed_split(args.processed_dir, "test")
    test_features = transform_features(test_features, scaler)

    model = GestureMLP(
        sequence_length=metadata.sequence_length,
        feature_count=len(metadata.feature_columns),
        num_classes=len(metadata.labels),
        hidden_dims=metadata.hidden_dims,
    )
    state_dict = torch.load(artifacts.weights, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        torch_logits = model(torch.from_numpy(test_features).float()).cpu().numpy()

    session = ort.InferenceSession(str(artifacts.onnx), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    onnx_logits = predict_onnx_logits(session, input_name, test_features)

    torch_predictions = torch_logits.argmax(axis=1)
    onnx_predictions = onnx_logits.argmax(axis=1)
    agreement = float(np.mean(torch_predictions == onnx_predictions))

    torch_metrics = evaluate_predictions(test_labels, torch_predictions)
    onnx_metrics = evaluate_predictions(test_labels, onnx_predictions)
    torch_latency = benchmark_torch(model, test_features, args.repeat)
    onnx_latency = benchmark_onnx(session, input_name, test_features, args.repeat)

    sample_probs = softmax(onnx_logits[:5])
    samples = []
    for index in range(min(5, len(test_features))):
        predicted = int(onnx_predictions[index])
        actual = int(test_labels[index])
        samples.append(
            {
                "index": index,
                "actual": metadata.labels[actual],
                "predicted": metadata.labels[predicted],
                "confidence": float(sample_probs[index, predicted]),
            }
        )

    payload = {
        "torch_metrics": torch_metrics,
        "onnx_metrics": onnx_metrics,
        "torch_latency_ms": torch_latency,
        "onnx_latency_ms": onnx_latency,
        "torch_onnx_agreement": agreement,
        "onnx_max_abs_error": float(np.max(np.abs(torch_logits - onnx_logits))),
        "per_class_accuracy": per_class_accuracy(test_labels, onnx_predictions, metadata.labels),
        "confusion_matrix": confusion_matrix_payload(test_labels, onnx_predictions, metadata.labels),
        "sample_predictions": samples,
        "artifacts": {
            "weights_bytes": artifacts.weights.stat().st_size if artifacts.weights.exists() else 0,
            "onnx_bytes": artifacts.onnx.stat().st_size if artifacts.onnx.exists() else 0,
            "espdl_bytes": artifacts.espdl.stat().st_size if artifacts.espdl.exists() else 0,
        },
    }
    write_json(artifacts.benchmark_json, payload)

    print(f"torch_accuracy={torch_metrics['accuracy']:.4f}")
    print(f"onnx_accuracy={onnx_metrics['accuracy']:.4f}")
    print(f"onnx_macro_f1={onnx_metrics['macro_f1']:.4f}")
    print(f"torch_onnx_agreement={agreement:.4f}")
    print(f"torch_latency_ms={torch_latency}")
    print(f"onnx_latency_ms={onnx_latency}")
    print(f"benchmark_json={artifacts.benchmark_json}")


if __name__ == "__main__":
    main()
