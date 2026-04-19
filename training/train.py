from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from esp_ppq.api import espdl_quantize_onnx
from torch.utils.data import DataLoader, TensorDataset

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.gesture_model import (
    COMPONENT_MODEL_PATH,
    DEFAULT_SEED,
    GestureMLP,
    ModelMetadata,
    PROCESSED_DATA_DIR,
    RUST_METADATA_PATH,
    TRAINING_DIR,
    build_scaler_from_metadata,
    evaluate_predictions,
    export_onnx,
    fit_feature_scaler,
    load_model_metadata,
    load_processed_metadata,
    load_processed_split,
    load_quantized_examples,
    make_artifact_paths,
    per_class_accuracy,
    set_random_seed,
    transform_features,
    validate_onnx_export,
    write_model_metadata,
    write_rust_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a gesture classifier and export ONNX + ESP-DL artifacts."
    )
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=TRAINING_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[128, 64])
    parser.add_argument("--target", default="esp32s3", choices=["esp32s3", "esp32p4", "c"])
    parser.add_argument("--bits", type=int, default=8, choices=[8, 16])
    parser.add_argument("--calibration-steps", type=int, default=32)
    parser.add_argument("--skip-quantization", action="store_true")
    parser.add_argument("--skip-copy-component-model", action="store_true")
    return parser.parse_args()


def make_loader(features: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(features), torch.from_numpy(labels))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def compute_class_weights(labels: np.ndarray, class_count: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=class_count).astype(np.float32)
    counts[counts == 0.0] = 1.0
    weights = counts.sum() / (class_count * counts)
    return torch.from_numpy(weights)


def run_epoch(
    model: GestureMLP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for batch_features, batch_labels in loader:
        optimizer.zero_grad()
        logits = model(batch_features)
        loss = criterion(logits, batch_labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * batch_features.size(0)
        total_samples += batch_features.size(0)
    return total_loss / max(1, total_samples)


def predict_logits(model: GestureMLP, features: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(features).float()).cpu().numpy()
    return logits


def quantize_model(
    onnx_path: Path,
    espdl_path: Path,
    calibration_features: np.ndarray,
    calibration_steps: int,
    target: str,
    bits: int,
) -> None:
    calibration_dataset = TensorDataset(torch.from_numpy(calibration_features))
    calibration_loader = DataLoader(
        calibration_dataset,
        batch_size=min(16, len(calibration_dataset)),
        shuffle=False,
        collate_fn=lambda batch: torch.stack([item[0] for item in batch], dim=0).cpu(),
    )
    sample_shape = calibration_features.shape[1:]
    espdl_quantize_onnx(
        onnx_import_file=str(onnx_path),
        espdl_export_file=str(espdl_path),
        calib_dataloader=calibration_loader,
        calib_steps=min(calibration_steps, len(calibration_loader)),
        input_shape=[1, sample_shape[0], sample_shape[1]],
        inputs=None,
        target=target,
        num_of_bits=bits,
        collate_fn=None,
        dispatching_override=None,
        device="cpu",
        error_report=True,
        skip_export=False,
        export_test_values=True,
        verbose=1,
    )


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)
    torch.set_num_threads(max(1, min(8, (torch.get_num_threads() or 1))))

    processed_metadata = load_processed_metadata(args.processed_dir)
    train_features, train_labels = load_processed_split(args.processed_dir, "train")
    val_features, val_labels = load_processed_split(args.processed_dir, "val")
    test_features, test_labels = load_processed_split(args.processed_dir, "test")

    scaler = fit_feature_scaler(train_features)
    train_features = transform_features(train_features, scaler)
    val_features = transform_features(val_features, scaler)
    test_features = transform_features(test_features, scaler)

    class_weights = compute_class_weights(train_labels, len(processed_metadata.labels))
    train_loader = make_loader(train_features, train_labels, args.batch_size, shuffle=True)

    model = GestureMLP(
        sequence_length=processed_metadata.sequence_length,
        feature_count=len(processed_metadata.feature_columns),
        num_classes=len(processed_metadata.labels),
        hidden_dims=args.hidden_dims,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    best_state: dict[str, torch.Tensor] | None = None
    best_val_macro_f1 = -1.0
    best_val_accuracy = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, criterion)
        train_logits = predict_logits(model, train_features)
        val_logits = predict_logits(model, val_features)
        train_predictions = train_logits.argmax(axis=1)
        val_predictions = val_logits.argmax(axis=1)
        train_metrics = evaluate_predictions(train_labels, train_predictions)
        val_metrics = evaluate_predictions(val_labels, val_predictions)

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_val_accuracy = val_metrics["accuracy"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        print(
            f"epoch={epoch:03d} loss={train_loss:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

    if best_state is None:
        raise RuntimeError("Training finished without a best checkpoint.")

    model.load_state_dict(best_state)
    artifacts = make_artifact_paths(args.artifacts_dir)
    torch.save(model.state_dict(), artifacts.weights)
    export_onnx(model, train_features.shape[1:], artifacts.onnx)
    onnx_max_abs_error = validate_onnx_export(model, artifacts.onnx, val_features)

    train_predictions = predict_logits(model, train_features).argmax(axis=1)
    val_predictions = predict_logits(model, val_features).argmax(axis=1)
    test_predictions = predict_logits(model, test_features).argmax(axis=1)
    train_metrics = evaluate_predictions(train_labels, train_predictions)
    val_metrics = evaluate_predictions(val_labels, val_predictions)
    test_metrics = evaluate_predictions(test_labels, test_predictions)

    metadata = ModelMetadata(
        processed_dir=str(args.processed_dir.resolve()),
        sequence_length=processed_metadata.sequence_length,
        feature_columns=processed_metadata.feature_columns,
        labels=processed_metadata.labels,
        scaler_mean=scaler.mean_.astype(float).tolist(),
        scaler_scale=scaler.scale_.astype(float).tolist(),
        train_size=int(len(train_features)),
        val_size=int(len(val_features)),
        test_size=int(len(test_features)),
        seed=args.seed,
        hidden_dims=[int(value) for value in args.hidden_dims],
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        train_accuracy=train_metrics["accuracy"],
        val_accuracy=val_metrics["accuracy"],
        val_macro_f1=val_metrics["macro_f1"],
        test_accuracy=test_metrics["accuracy"],
        test_macro_f1=test_metrics["macro_f1"],
        onnx_max_abs_error=onnx_max_abs_error,
    )
    write_model_metadata(metadata, artifacts.metadata_json)
    write_rust_metadata(metadata, RUST_METADATA_PATH)

    print(f"train_accuracy={train_metrics['accuracy']:.4f}")
    print(f"val_accuracy={val_metrics['accuracy']:.4f}")
    print(f"test_accuracy={test_metrics['accuracy']:.4f}")
    print(f"test_macro_f1={test_metrics['macro_f1']:.4f}")
    print(f"test_per_class_accuracy={per_class_accuracy(test_labels, test_predictions, metadata.labels)}")
    print(f"weights={artifacts.weights}")
    print(f"onnx={artifacts.onnx}")
    print(f"metadata={artifacts.metadata_json}")

    if args.skip_quantization:
        return

    quantize_model(
        onnx_path=artifacts.onnx,
        espdl_path=artifacts.espdl,
        calibration_features=train_features,
        calibration_steps=args.calibration_steps,
        target=args.target,
        bits=args.bits,
    )

    quantized_examples = load_quantized_examples(artifacts.espdl_info)
    metadata.espdl_input_shape = quantized_examples.input_tensor.shape
    metadata.espdl_input_exponent = quantized_examples.input_tensor.exponent
    metadata.espdl_output_shape = quantized_examples.output_tensor.shape
    metadata.espdl_output_exponent = quantized_examples.output_tensor.exponent
    write_model_metadata(metadata, artifacts.metadata_json)
    write_rust_metadata(metadata, RUST_METADATA_PATH, quantized_examples=quantized_examples)

    if not args.skip_copy_component_model:
        COMPONENT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(artifacts.espdl, COMPONENT_MODEL_PATH)
        print(f"component_model={COMPONENT_MODEL_PATH}")

    print(f"espdl={artifacts.espdl}")
    print(f"espdl_info={artifacts.espdl_info}")
    print(f"espdl_quant={artifacts.espdl_quant}")


if __name__ == "__main__":
    main()
