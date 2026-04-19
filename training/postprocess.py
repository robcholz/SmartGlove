#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import json
import math
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import torch
from esp_ppq.api import espdl_quantize_onnx
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import kagglehub


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = Path(__file__).resolve().parent
COMPONENT_MODEL_PATH = (
    REPO_ROOT / "src" / "experiment" / "components" / "espdl_experiment" / "model.espdl"
)
RUST_METADATA_PATH = REPO_ROOT / "src" / "inference" / "generated.rs"

DEFAULT_DATASET = "krishnagarigipati/final-datasets"
WINDOW_SIZE = 154
WINDOW_STEP = 77
SENSOR_COLS = [
    "flex_1",
    "flex_2",
    "flex_3",
    "flex_4",
    "flex_5",
    "ACCx",
    "ACCy",
    "ACCz",
]
LABEL_COL = "label"
SUBJECT_COL = "subject"


@dataclass
class ArtifactPaths:
    weights: Path
    onnx: Path
    espdl: Path
    espdl_info: Path
    espdl_quant: Path
    metadata_json: Path


@dataclass
class TrainingMetadata:
    dataset_ref: str
    dataset_path: str
    csv_files: list[str]
    sensor_columns: list[str]
    label_column: str
    subject_column: str
    label_classes: list[str]
    scaler_mean: list[float]
    scaler_scale: list[float]
    train_subjects: list[str]
    val_subjects: list[str]
    input_shape: list[int]
    window_size: int
    window_step: int
    majority_threshold: float
    train_windows: int
    val_windows: int
    val_accuracy: float
    onnx_max_abs_error: float
    espdl_input_shape: list[int] | None = None
    espdl_input_exponent: int | None = None
    espdl_output_shape: list[int] | None = None
    espdl_output_exponent: int | None = None


@dataclass
class QuantizedTensorExample:
    name: str
    shape: list[int]
    exponent: int
    values: list[int]


@dataclass
class QuantizedModelExamples:
    input_tensor: QuantizedTensorExample
    output_tensor: QuantizedTensorExample


class GestureConvNet(nn.Module):
    def __init__(self, input_channels: int, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 24, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(24, 48, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train, export, and quantize the SmartGlove model"
    )
    parser.add_argument(
        "--dataset", default=DEFAULT_DATASET, help="Kaggle dataset reference"
    )
    parser.add_argument(
        "--csv", type=Path, help="Use a local CSV instead of downloading from Kaggle"
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--majority-threshold", type=float, default=0.8)
    parser.add_argument("--calibration-steps", type=int, default=32)
    parser.add_argument(
        "--target", default="esp32s3", choices=["c", "esp32s3", "esp32p4"]
    )
    parser.add_argument("--bits", type=int, default=8, choices=[8, 16])
    parser.add_argument("--artifacts-dir", type=Path, default=TRAINING_DIR)
    parser.add_argument("--weights-name", default="glove_model.pt")
    parser.add_argument("--onnx-name", default="glove_model.onnx")
    parser.add_argument("--espdl-name", default="glove_model.espdl")
    parser.add_argument(
        "--skip-quantization",
        action="store_true",
        help="Only train and export ONNX, skip ESP-DL quantization",
    )
    parser.add_argument(
        "--copy-component-model",
        action="store_true",
        help="Copy the exported .espdl file into the ESP-IDF component embed path",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def canonicalize_column(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "flex1": "flex_1",
        "flex_1": "flex_1",
        "flex2": "flex_2",
        "flex_2": "flex_2",
        "flex3": "flex_3",
        "flex_3": "flex_3",
        "flex4": "flex_4",
        "flex_4": "flex_4",
        "flex5": "flex_5",
        "flex_5": "flex_5",
        "accx": "ACCx",
        "accy": "ACCy",
        "accz": "ACCz",
        "label": "label",
        "subject": "subject",
    }
    renamed = {}
    for col in df.columns:
        key = canonicalize_column(col)
        if key in aliases:
            renamed[col] = aliases[key]
    return df.rename(columns=renamed)


def find_candidate_csvs(dataset_root: Path) -> list[Path]:
    candidates: list[Path] = []
    for csv_path in sorted(dataset_root.rglob("*.csv")):
        try:
            sample = rename_columns(pd.read_csv(csv_path, nrows=4))
        except Exception:
            continue

        required = set(SENSOR_COLS + [LABEL_COL])
        if required.issubset(sample.columns):
            candidates.append(csv_path)
    if not candidates:
        raise FileNotFoundError(
            f"No CSV files with the expected SmartGlove schema were found under {dataset_root}"
        )
    return candidates


def download_dataset(dataset_ref: str) -> Path:
    dataset_path = Path(kagglehub.dataset_download(dataset_ref))
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"KaggleHub reported {dataset_path}, but it does not exist"
        )
    return dataset_path


def load_dataframe(csv_paths: Iterable[Path]) -> tuple[pd.DataFrame, list[Path]]:
    frames: list[pd.DataFrame] = []
    accepted_paths: list[Path] = []

    for csv_path in csv_paths:
        df = rename_columns(pd.read_csv(csv_path))
        required = set(SENSOR_COLS + [LABEL_COL])
        if not required.issubset(df.columns):
            continue
        if SUBJECT_COL not in df.columns:
            df[SUBJECT_COL] = csv_path.stem
        df = df.dropna(subset=SENSOR_COLS + [LABEL_COL, SUBJECT_COL]).copy()
        df[SUBJECT_COL] = df[SUBJECT_COL].astype(str)
        frames.append(df)
        accepted_paths.append(csv_path)

    if not frames:
        raise ValueError(
            "Found CSV files, but none contained the required SmartGlove columns"
        )

    combined = pd.concat(frames, ignore_index=True)
    return combined, accepted_paths


def split_subjects(
    subjects: np.ndarray,
    validation_fraction: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    unique_subjects = np.array(sorted({str(subject) for subject in subjects}))
    if unique_subjects.size < 2:
        return unique_subjects.tolist(), []

    val_size = max(1, int(math.ceil(unique_subjects.size * validation_fraction)))
    val_size = min(val_size, unique_subjects.size - 1)
    train_subjects, val_subjects = train_test_split(
        unique_subjects,
        test_size=val_size,
        random_state=seed,
        shuffle=True,
    )
    return sorted(train_subjects.tolist()), sorted(val_subjects.tolist())


def build_windows(
    df: pd.DataFrame,
    label_encoder: LabelEncoder,
    scaler: StandardScaler,
    majority_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    num_classes = len(label_encoder.classes_)
    windows: list[np.ndarray] = []
    labels: list[int] = []

    for _, group in df.groupby(SUBJECT_COL, sort=False):
        sensors = scaler.transform(group[SENSOR_COLS].to_numpy()).astype(np.float32)
        encoded_labels = label_encoder.transform(group[LABEL_COL].astype(str))

        if len(group) < WINDOW_SIZE:
            continue

        for start in range(0, len(group) - WINDOW_SIZE + 1, WINDOW_STEP):
            end = start + WINDOW_SIZE
            label_slice = encoded_labels[start:end]
            counts = np.bincount(label_slice, minlength=num_classes)
            target = int(np.argmax(counts))
            majority_ratio = float(counts[target]) / float(WINDOW_SIZE)
            if majority_ratio < majority_threshold:
                continue

            window = sensors[start:end].T
            windows.append(window)
            labels.append(target)

    if not windows:
        raise ValueError(
            "No training windows were generated. Check the dataset size and majority threshold."
        )

    return np.stack(windows).astype(np.float32), np.asarray(labels, dtype=np.int64)


def make_loaders(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val)),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, val_loader


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    learning_rate: float,
) -> tuple[nn.Module, float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    best_state = None
    best_val_accuracy = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * batch_x.size(0)
            train_count += batch_x.size(0)

        val_accuracy = evaluate_model(model, val_loader)
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        avg_train_loss = train_loss / max(1, train_count)
        print(
            f"epoch={epoch:02d} train_loss={avg_train_loss:.4f} "
            f"val_accuracy={val_accuracy:.4f}"
        )

    if best_state is None:
        raise RuntimeError("Training finished without producing a model checkpoint")

    model.load_state_dict(best_state)
    model.eval()
    return model, best_val_accuracy


def evaluate_model(model: nn.Module, loader: DataLoader) -> float:
    preds: list[int] = []
    targets: list[int] = []
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in loader:
            logits = model(batch_x)
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
            targets.extend(batch_y.cpu().numpy().tolist())
    if not targets:
        return 0.0
    return float(accuracy_score(targets, preds))


def export_onnx(model: nn.Module, onnx_path: Path) -> None:
    dummy_input = torch.zeros(1, len(SENSOR_COLS), WINDOW_SIZE, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=13,
        input_names=["sensor_window"],
        output_names=["logits"],
        dynamic_axes=None,
        do_constant_folding=True,
        dynamo=False,
    )


def validate_onnx(model: nn.Module, onnx_path: Path, sample: np.ndarray) -> float:
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    sample_tensor = torch.from_numpy(sample[:1]).float()
    with torch.no_grad():
        torch_output = model(sample_tensor).cpu().numpy()
    ort_output = session.run(None, {input_name: sample_tensor.numpy()})[0]

    max_abs_error = float(np.max(np.abs(torch_output - ort_output)))
    print(f"onnx_max_abs_error={max_abs_error:.8f}")
    return max_abs_error


def collate_calibration_batch(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    inputs = torch.stack([item[0] for item in batch], dim=0)
    return inputs.cpu()


def quantize_model(
    onnx_path: Path,
    espdl_path: Path,
    calibration_windows: np.ndarray,
    calibration_steps: int,
    target: str,
    num_bits: int,
) -> None:
    calibration_dataset = TensorDataset(
        torch.from_numpy(calibration_windows),
        torch.zeros(len(calibration_windows), dtype=torch.int64),
    )
    calibration_loader = DataLoader(
        calibration_dataset,
        batch_size=min(16, len(calibration_dataset)),
        shuffle=False,
        collate_fn=collate_calibration_batch,
    )

    espdl_quantize_onnx(
        onnx_import_file=str(onnx_path),
        espdl_export_file=str(espdl_path),
        calib_dataloader=calibration_loader,
        calib_steps=min(calibration_steps, len(calibration_loader)),
        input_shape=[1, len(SENSOR_COLS), WINDOW_SIZE],
        inputs=None,
        target=target,
        num_of_bits=num_bits,
        collate_fn=None,
        dispatching_override=None,
        device="cpu",
        error_report=True,
        skip_export=False,
        export_test_values=True,
        verbose=1,
    )


def ensure_output_paths(
    artifacts_dir: Path, weights_name: str, onnx_name: str, espdl_name: str
) -> ArtifactPaths:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    espdl_base = artifacts_dir / espdl_name
    return ArtifactPaths(
        weights=artifacts_dir / weights_name,
        onnx=artifacts_dir / onnx_name,
        espdl=espdl_base,
        espdl_info=espdl_base.with_suffix(".info"),
        espdl_quant=espdl_base.with_suffix(".json"),
        metadata_json=artifacts_dir / "glove_model_metadata.json",
    )


def write_metadata(metadata: TrainingMetadata, output_path: Path) -> None:
    output_path.write_text(
        json.dumps(asdict(metadata), indent=2) + "\n", encoding="utf-8"
    )


def load_metadata(metadata_path: Path | None = None) -> TrainingMetadata:
    metadata_path = metadata_path or (TRAINING_DIR / "glove_model_metadata.json")
    return TrainingMetadata(**json.loads(metadata_path.read_text(encoding="utf-8")))


def build_label_encoder_from_metadata(metadata: TrainingMetadata) -> LabelEncoder:
    encoder = LabelEncoder()
    encoder.classes_ = np.asarray(metadata.label_classes, dtype=object)
    return encoder


def build_scaler_from_metadata(metadata: TrainingMetadata) -> StandardScaler:
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(metadata.scaler_mean, dtype=np.float64)
    scaler.scale_ = np.asarray(metadata.scaler_scale, dtype=np.float64)
    scaler.var_ = scaler.scale_**2
    scaler.n_features_in_ = len(metadata.sensor_columns)
    return scaler


def write_rust_metadata(metadata: TrainingMetadata, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = ", ".join(json.dumps(label) for label in metadata.label_classes)
    means = ", ".join(f"{value:.8f}" for value in metadata.scaler_mean)
    scales = ", ".join(f"{value:.8f}" for value in metadata.scaler_scale)
    espdl_input_shape = (
        metadata.espdl_input_shape
        if metadata.espdl_input_shape is not None
        else metadata.input_shape
    )
    espdl_output_shape = (
        metadata.espdl_output_shape if metadata.espdl_output_shape is not None else []
    )
    output = f"""// Generated by training/postprocess.py. Do not edit manually.
pub const MODEL_WINDOW_SIZE: usize = {metadata.window_size};
pub const MODEL_WINDOW_STEP: usize = {metadata.window_step};
pub const MODEL_FEATURE_COUNT: usize = {len(metadata.sensor_columns)};
pub const MODEL_OUTPUT_COUNT: usize = {len(metadata.label_classes)};
pub const ESPDL_INPUT_RANK: usize = {len(espdl_input_shape)};
pub const ESPDL_INPUT_SHAPE: [usize; {len(espdl_input_shape)}] = [{", ".join(str(value) for value in espdl_input_shape)}];
pub const ESPDL_INPUT_EXPONENT: i32 = {metadata.espdl_input_exponent if metadata.espdl_input_exponent is not None else 0};
pub const ESPDL_OUTPUT_RANK: usize = {len(espdl_output_shape)};
pub const ESPDL_OUTPUT_SHAPE: [usize; {len(espdl_output_shape)}] = [{", ".join(str(value) for value in espdl_output_shape)}];
pub const ESPDL_OUTPUT_EXPONENT: i32 = {metadata.espdl_output_exponent if metadata.espdl_output_exponent is not None else 0};
pub const MODEL_LABELS: [&str; {len(metadata.label_classes)}] = [{labels}];
pub const FEATURE_MEANS: [f32; {len(metadata.scaler_mean)}] = [{means}];
pub const FEATURE_SCALES: [f32; {len(metadata.scaler_scale)}] = [{scales}];
"""
    output_path.write_text(output, encoding="utf-8")


def append_rust_quantized_examples(
    output_path: Path, examples: QuantizedModelExamples
) -> None:
    input_values = ", ".join(str(value) for value in examples.input_tensor.values)
    output_values = ", ".join(str(value) for value in examples.output_tensor.values)
    addition = f"""
pub const ESPDL_TEST_INPUT: [i8; {len(examples.input_tensor.values)}] = [{input_values}];
pub const ESPDL_TEST_OUTPUT: [i8; {len(examples.output_tensor.values)}] = [{output_values}];
"""
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(addition)


def parse_quantized_tensor_example(
    info_text: str, section_header: str
) -> QuantizedTensorExample:
    pattern = re.compile(
        rf"{section_header}:\s*%(?P<name>[\w/\.]+), shape: \[(?P<shape>[^\]]+)\], "
        rf"exponents: \[(?P<exponent>-?\d+)\],\s*value: array\(\[(?P<values>.*?)\],\s*dtype=(?P<dtype>\w+)\)",
        re.S,
    )
    match = pattern.search(info_text)
    if match is None:
        raise ValueError(
            f"Could not parse {section_header} from {TRAINING_DIR / 'glove_model.info'}"
        )

    shape = [
        int(part.strip()) for part in match.group("shape").split(",") if part.strip()
    ]
    logical_size = math.prod(shape)
    values = np.fromstring(
        match.group("values").replace("\n", " "), sep=",", dtype=np.int64
    )
    values = values[:logical_size].astype(np.int64).tolist()
    return QuantizedTensorExample(
        name=match.group("name"),
        shape=shape,
        exponent=int(match.group("exponent")),
        values=values,
    )


def load_quantized_examples(info_path: Path) -> QuantizedModelExamples:
    info_text = info_path.read_text(encoding="utf-8")
    input_tensor = parse_quantized_tensor_example(info_text, "test inputs value")
    output_tensor = parse_quantized_tensor_example(info_text, "test outputs value")
    return QuantizedModelExamples(
        input_tensor=input_tensor, output_tensor=output_tensor
    )


def prepare_dataset(args: argparse.Namespace) -> tuple[pd.DataFrame, list[Path], str]:
    if args.csv is not None:
        dataset_path = args.csv.resolve()
        df, csv_paths = load_dataframe([dataset_path])
        return df, csv_paths, str(dataset_path.parent)

    dataset_root = download_dataset(args.dataset)
    csv_paths = find_candidate_csvs(dataset_root)
    df, accepted_paths = load_dataframe(csv_paths)
    return df, accepted_paths, str(dataset_root)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    torch.set_num_threads(max(1, min(8, (torch.get_num_threads() or 1))))

    artifact_paths = ensure_output_paths(
        artifacts_dir=args.artifacts_dir.resolve(),
        weights_name=args.weights_name,
        onnx_name=args.onnx_name,
        espdl_name=args.espdl_name,
    )

    df, csv_paths, dataset_path = prepare_dataset(args)
    print(
        f"loaded_rows={len(df)} csv_files={len(csv_paths)} dataset_path={dataset_path}"
    )

    train_subjects, val_subjects = split_subjects(
        df[SUBJECT_COL].to_numpy(),
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )

    if val_subjects:
        train_df = df[df[SUBJECT_COL].isin(train_subjects)].copy()
        val_df = df[df[SUBJECT_COL].isin(val_subjects)].copy()
    else:
        train_df, val_df = train_test_split(
            df,
            test_size=args.validation_fraction,
            random_state=args.seed,
            shuffle=True,
        )
        train_subjects = sorted(train_df[SUBJECT_COL].astype(str).unique().tolist())
        val_subjects = sorted(val_df[SUBJECT_COL].astype(str).unique().tolist())

    label_encoder = LabelEncoder()
    label_encoder.fit(df[LABEL_COL].astype(str))

    scaler = StandardScaler()
    scaler.fit(train_df[SENSOR_COLS])

    x_train, y_train = build_windows(
        train_df, label_encoder, scaler, args.majority_threshold
    )
    x_val, y_val = build_windows(val_df, label_encoder, scaler, args.majority_threshold)
    print(
        f"train_windows={len(x_train)} val_windows={len(x_val)} "
        f"classes={len(label_encoder.classes_)}"
    )

    train_loader, val_loader = make_loaders(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        batch_size=args.batch_size,
    )

    model = GestureConvNet(
        input_channels=len(SENSOR_COLS), num_classes=len(label_encoder.classes_)
    )
    model, val_accuracy = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )

    torch.save(model.state_dict(), artifact_paths.weights)
    export_onnx(model, artifact_paths.onnx)
    onnx_max_abs_error = validate_onnx(model, artifact_paths.onnx, x_val)

    metadata = TrainingMetadata(
        dataset_ref=args.dataset,
        dataset_path=dataset_path,
        csv_files=[str(path) for path in csv_paths],
        sensor_columns=SENSOR_COLS,
        label_column=LABEL_COL,
        subject_column=SUBJECT_COL,
        label_classes=label_encoder.classes_.tolist(),
        scaler_mean=scaler.mean_.astype(float).tolist(),
        scaler_scale=scaler.scale_.astype(float).tolist(),
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        input_shape=[1, len(SENSOR_COLS), WINDOW_SIZE],
        window_size=WINDOW_SIZE,
        window_step=WINDOW_STEP,
        majority_threshold=args.majority_threshold,
        train_windows=int(len(x_train)),
        val_windows=int(len(x_val)),
        val_accuracy=float(val_accuracy),
        onnx_max_abs_error=onnx_max_abs_error,
    )
    write_metadata(metadata, artifact_paths.metadata_json)
    write_rust_metadata(metadata, RUST_METADATA_PATH)

    print(f"weights={artifact_paths.weights}")
    print(f"onnx={artifact_paths.onnx}")
    print(f"metadata={artifact_paths.metadata_json}")
    print(f"rust_metadata={RUST_METADATA_PATH}")

    if args.skip_quantization:
        return

    quantize_model(
        onnx_path=artifact_paths.onnx,
        espdl_path=artifact_paths.espdl,
        calibration_windows=x_train,
        calibration_steps=args.calibration_steps,
        target=args.target,
        num_bits=args.bits,
    )
    print(f"espdl={artifact_paths.espdl}")
    print(f"espdl_info={artifact_paths.espdl_info}")
    print(f"espdl_quant={artifact_paths.espdl_quant}")

    quantized_examples = load_quantized_examples(artifact_paths.espdl_info)
    metadata.espdl_input_shape = quantized_examples.input_tensor.shape
    metadata.espdl_input_exponent = quantized_examples.input_tensor.exponent
    metadata.espdl_output_shape = quantized_examples.output_tensor.shape
    metadata.espdl_output_exponent = quantized_examples.output_tensor.exponent
    write_metadata(metadata, artifact_paths.metadata_json)
    write_rust_metadata(metadata, RUST_METADATA_PATH)
    append_rust_quantized_examples(RUST_METADATA_PATH, quantized_examples)

    if args.copy_component_model:
        COMPONENT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(artifact_paths.espdl, COMPONENT_MODEL_PATH)
        print(f"component_model={COMPONENT_MODEL_PATH}")


if __name__ == "__main__":
    main()
