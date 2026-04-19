from __future__ import annotations

import json
import math
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = REPO_ROOT / "dataset"
PROCESSED_DATA_DIR = REPO_ROOT / "processed_data"
TRAINING_DIR = Path(__file__).resolve().parent
COMPONENT_MODEL_PATH = (
    REPO_ROOT / "src" / "inference" / "components" / "espdl_experiment" / "model.espdl"
)
RUST_METADATA_PATH = REPO_ROOT / "src" / "inference" / "generated.rs"

FEATURE_COLUMNS = [
    "thumb",
    "index",
    "middle",
    "ring",
    "pinky",
    "acc_x",
    "acc_y",
    "acc_z",
]
IMU_FEATURE_INDICES = [
    FEATURE_COLUMNS.index(name) for name in ["acc_x", "acc_y", "acc_z"]
]
LABEL_COLUMN = "label"
DEFAULT_SEQUENCE_LENGTH = 230
DEFAULT_SEED = 7

SESSION_FILE_RE = re.compile(r"^(?P<timestamp>\d+)_session(?P<index>\d+)\.csv$")


@dataclass
class SessionRecord:
    path: str
    label: str
    label_index: int
    split: str
    original_length: int
    resampled_length: int


@dataclass
class ProcessedDataMetadata:
    dataset_dir: str
    processed_dir: str
    sequence_length: int
    feature_columns: list[str]
    labels: list[str]
    train_ratio: float
    val_ratio: float
    test_ratio: float
    seed: int
    split_counts: dict[str, int]
    label_counts: dict[str, int]
    original_length_stats: dict[str, float]
    records: list[SessionRecord]


@dataclass
class ArtifactPaths:
    weights: Path
    onnx: Path
    espdl: Path
    espdl_info: Path
    espdl_quant: Path
    metadata_json: Path
    benchmark_json: Path


@dataclass
class ModelMetadata:
    processed_dir: str
    sequence_length: int
    feature_columns: list[str]
    model_input_dim: int
    labels: list[str]
    scaler_mean: list[float]
    scaler_scale: list[float]
    train_size: int
    val_size: int
    test_size: int
    seed: int
    hidden_dims: list[int]
    epochs: int
    batch_size: int
    learning_rate: float
    train_accuracy: float
    val_accuracy: float
    val_macro_f1: float
    test_accuracy: float
    test_macro_f1: float
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
    dtype: str
    values: list[float]


@dataclass
class QuantizedModelExamples:
    input_tensor: QuantizedTensorExample
    output_tensor: QuantizedTensorExample


class GestureMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: Sequence[int] = (128, 64),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Flatten()]
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def discover_dataset_records(dataset_dir: Path) -> list[tuple[Path, str]]:
    records: list[tuple[Path, str]] = []
    for label_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
        for csv_path in sorted(label_dir.glob("*.csv")):
            records.append((csv_path, label_dir.name))
    if not records:
        raise FileNotFoundError(f"No CSV files found under {dataset_dir}")
    return records


def load_session_array(csv_path: Path) -> np.ndarray:
    frame = pd.read_csv(csv_path)
    required = set(FEATURE_COLUMNS + [LABEL_COLUMN])
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")
    values = frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=True)
    if len(values) == 0:
        raise ValueError(f"{csv_path} contains no samples")
    return values


def resample_sequence(values: np.ndarray, target_length: int) -> np.ndarray:
    source_length, feature_count = values.shape
    if source_length == target_length:
        return values.astype(np.float32, copy=True)
    if source_length == 1:
        return np.repeat(values, target_length, axis=0).astype(np.float32, copy=False)

    source_positions = np.linspace(0.0, 1.0, num=source_length, dtype=np.float32)
    target_positions = np.linspace(0.0, 1.0, num=target_length, dtype=np.float32)
    resampled = np.empty((target_length, feature_count), dtype=np.float32)
    for feature_index in range(feature_count):
        resampled[:, feature_index] = np.interp(
            target_positions,
            source_positions,
            values[:, feature_index].astype(np.float32, copy=False),
        )
    return resampled


def stratified_split_records(
    records: list[tuple[Path, str]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[tuple[Path, str]]]:
    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, rel_tol=1e-6):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    if not records:
        raise ValueError("records must not be empty")

    paths = [path for path, _ in records]
    labels = [label for _, label in records]

    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        paths,
        labels,
        test_size=test_ratio,
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )

    val_fraction_within_train_val = val_ratio / (train_ratio + val_ratio)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths,
        train_val_labels,
        test_size=val_fraction_within_train_val,
        random_state=seed,
        shuffle=True,
        stratify=train_val_labels,
    )

    return {
        "train": list(zip(train_paths, train_labels, strict=True)),
        "val": list(zip(val_paths, val_labels, strict=True)),
        "test": list(zip(test_paths, test_labels, strict=True)),
    }


def save_array(path: Path, array: np.ndarray) -> None:
    ensure_dir(path.parent)
    np.save(path, array)


def load_array(path: Path, dtype: np.dtype | None = None) -> np.ndarray:
    array = np.load(path)
    if dtype is not None:
        return array.astype(dtype, copy=False)
    return array


def processed_feature_path(processed_dir: Path, split: str) -> Path:
    return processed_dir / f"{split}_features.npy"


def processed_label_path(processed_dir: Path, split: str) -> Path:
    return processed_dir / f"{split}_labels.npy"


def processed_length_path(processed_dir: Path, split: str) -> Path:
    return processed_dir / f"{split}_original_lengths.npy"


def processed_metadata_path(processed_dir: Path) -> Path:
    return processed_dir / "metadata.json"


def model_metadata_path(artifacts_dir: Path) -> Path:
    return artifacts_dir / "glove_model_metadata.json"


def benchmark_output_path(artifacts_dir: Path) -> Path:
    return artifacts_dir / "benchmark.json"


def write_json(path: Path, payload: object) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def load_processed_metadata(processed_dir: Path) -> ProcessedDataMetadata:
    return ProcessedDataMetadata(**read_json(processed_metadata_path(processed_dir)))


def load_model_metadata(metadata_path: Path) -> ModelMetadata:
    return ModelMetadata(**read_json(metadata_path))


def load_processed_split(
    processed_dir: Path, split: str
) -> tuple[np.ndarray, np.ndarray]:
    features = load_array(
        processed_feature_path(processed_dir, split), dtype=np.float32
    )
    labels = load_array(processed_label_path(processed_dir, split), dtype=np.int64)
    return features, labels


def extract_sequence_features(features: np.ndarray) -> np.ndarray:
    processed = features.astype(np.float32, copy=False)
    imu_values = processed[:, :, IMU_FEATURE_INDICES]
    imu_centered = imu_values - imu_values.mean(axis=1, keepdims=True)
    diffs = np.diff(processed, axis=1, prepend=processed[:, :1, :])
    imu_diffs = np.diff(imu_values, axis=1, prepend=imu_values[:, :1, :])

    blocks = [
        processed.mean(axis=1),
        processed.std(axis=1),
        processed.min(axis=1),
        processed.max(axis=1),
        processed.max(axis=1) - processed.min(axis=1),
        processed[:, 0, :],
        processed[:, -1, :],
        processed[:, -1, :] - processed[:, 0, :],
        np.sqrt((diffs**2).mean(axis=1)),
        np.sqrt((imu_centered**2).mean(axis=1)),
        np.sqrt((imu_diffs**2).mean(axis=1)),
    ]
    return np.concatenate(blocks, axis=1).astype(np.float32, copy=False)


def fit_feature_scaler(features: np.ndarray) -> StandardScaler:
    features = extract_sequence_features(features)
    scaler = StandardScaler()
    scaler.fit(features)
    return scaler


def transform_features(features: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    features = extract_sequence_features(features)
    transformed = scaler.transform(features)
    return transformed.astype(np.float32, copy=False)


def make_artifact_paths(artifacts_dir: Path) -> ArtifactPaths:
    ensure_dir(artifacts_dir)
    espdl_base = artifacts_dir / "glove_model.espdl"
    return ArtifactPaths(
        weights=artifacts_dir / "glove_model.pt",
        onnx=artifacts_dir / "glove_model.onnx",
        espdl=espdl_base,
        espdl_info=espdl_base.with_suffix(".info"),
        espdl_quant=espdl_base.with_suffix(".json"),
        metadata_json=model_metadata_path(artifacts_dir),
        benchmark_json=benchmark_output_path(artifacts_dir),
    )


def export_onnx(model: nn.Module, input_dim: int, output_path: Path) -> None:
    dummy = torch.zeros(1, input_dim, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        output_path,
        export_params=True,
        opset_version=13,
        input_names=["sensor_window"],
        output_names=["logits"],
        dynamic_axes=None,
        do_constant_folding=True,
        dynamo=False,
    )


def validate_onnx_export(
    model: nn.Module,
    onnx_path: Path,
    sample_batch: np.ndarray,
) -> float:
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    sample_tensor = torch.from_numpy(sample_batch[:1]).float()
    with torch.no_grad():
        torch_output = model(sample_tensor).cpu().numpy()
    onnx_output = session.run(None, {input_name: sample_tensor.numpy()})[0]
    return float(np.max(np.abs(torch_output - onnx_output)))


def write_model_metadata(metadata: ModelMetadata, output_path: Path) -> None:
    write_json(output_path, asdict(metadata))


def write_processed_metadata(
    metadata: ProcessedDataMetadata, output_path: Path
) -> None:
    write_json(output_path, asdict(metadata))


def build_scaler_from_metadata(metadata: ModelMetadata) -> StandardScaler:
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(metadata.scaler_mean, dtype=np.float64)
    scaler.scale_ = np.asarray(metadata.scaler_scale, dtype=np.float64)
    scaler.var_ = scaler.scale_**2
    scaler.n_features_in_ = len(metadata.scaler_mean)
    return scaler


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def evaluate_predictions(
    labels: np.ndarray, predictions: np.ndarray
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_precision": float(
            precision_score(labels, predictions, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(labels, predictions, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
    }


def per_class_accuracy(
    labels: np.ndarray, predictions: np.ndarray, class_names: Sequence[str]
) -> dict[str, float]:
    results: dict[str, float] = {}
    for index, class_name in enumerate(class_names):
        mask = labels == index
        if not np.any(mask):
            results[class_name] = 0.0
            continue
        results[class_name] = float(np.mean(predictions[mask] == labels[mask]))
    return results


def confusion_matrix_payload(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: Sequence[str],
) -> dict[str, object]:
    matrix = confusion_matrix(labels, predictions, labels=np.arange(len(class_names)))
    return {
        "labels": list(class_names),
        "matrix": matrix.astype(int).tolist(),
    }


def write_rust_metadata(
    metadata: ModelMetadata,
    output_path: Path,
    quantized_examples: QuantizedModelExamples | None = None,
) -> None:
    def rust_f32_literal(value: float) -> str:
        literal = np.format_float_positional(np.float32(value), unique=True, trim="k")
        if "." not in literal and "e" not in literal and "E" not in literal:
            literal += ".0"
        return literal

    labels = ", ".join(json.dumps(label) for label in metadata.labels)
    means = ", ".join(rust_f32_literal(value) for value in metadata.scaler_mean)
    scales = ", ".join(rust_f32_literal(value) for value in metadata.scaler_scale)
    espdl_input_shape = metadata.espdl_input_shape or [1, metadata.model_input_dim]
    espdl_output_shape = metadata.espdl_output_shape or [1, len(metadata.labels)]
    output = f"""// Generated by train.py. Do not edit manually.
pub const MODEL_WINDOW_SIZE: usize = {metadata.sequence_length};
pub const MODEL_WINDOW_STEP: usize = 1;
pub const RAW_FEATURE_COUNT: usize = {len(metadata.feature_columns)};
pub const MODEL_INPUT_FEATURE_COUNT: usize = {metadata.model_input_dim};
pub const MODEL_OUTPUT_COUNT: usize = {len(metadata.labels)};
pub const ESPDL_INPUT_RANK: usize = {len(espdl_input_shape)};
pub const ESPDL_INPUT_SHAPE: [usize; {len(espdl_input_shape)}] = [{", ".join(str(value) for value in espdl_input_shape)}];
pub const ESPDL_INPUT_EXPONENT: i32 = {metadata.espdl_input_exponent if metadata.espdl_input_exponent is not None else 0};
pub const ESPDL_OUTPUT_RANK: usize = {len(espdl_output_shape)};
pub const ESPDL_OUTPUT_SHAPE: [usize; {len(espdl_output_shape)}] = [{", ".join(str(value) for value in espdl_output_shape)}];
pub const ESPDL_OUTPUT_EXPONENT: i32 = {metadata.espdl_output_exponent if metadata.espdl_output_exponent is not None else 0};
pub const MODEL_LABELS: [&str; {len(metadata.labels)}] = [{labels}];
pub const MODEL_FEATURE_MEANS: [f32; {len(metadata.scaler_mean)}] = [{means}];
pub const MODEL_FEATURE_SCALES: [f32; {len(metadata.scaler_scale)}] = [{scales}];
"""
    if quantized_examples is not None:
        if quantized_examples.input_tensor.dtype.startswith("float"):
            input_rust_type = "f32"
            input_values = ", ".join(
                rust_f32_literal(float(value))
                for value in quantized_examples.input_tensor.values
            )
        else:
            input_rust_type = "i8"
            input_values = ", ".join(
                str(int(value)) for value in quantized_examples.input_tensor.values
            )
        output_values = ", ".join(
            str(int(value)) for value in quantized_examples.output_tensor.values
        )
        output += f"""

pub const ESPDL_TEST_INPUT: [{input_rust_type}; {len(quantized_examples.input_tensor.values)}] = [{input_values}];
pub const ESPDL_TEST_OUTPUT: [i8; {len(quantized_examples.output_tensor.values)}] = [{output_values}];
"""
    else:
        output += """

pub const ESPDL_TEST_INPUT: [f32; 0] = [];
pub const ESPDL_TEST_OUTPUT: [i8; 0] = [];
"""
    ensure_dir(output_path.parent)
    output_path.write_text(output, encoding="utf-8")


def parse_quantized_tensor_example(
    info_text: str, section_header: str, info_path: Path
) -> QuantizedTensorExample:
    pattern = re.compile(
        rf"{section_header}:\s*%(?P<name>[\w/\.]+), shape: \[(?P<shape>[^\]]+)\], "
        rf"exponents: \[(?P<exponent>-?\d+)\],\s*value: array\(\[(?P<values>.*?)\],\s*dtype=(?P<dtype>\w+)\)",
        re.S,
    )
    match = pattern.search(info_text)
    if match is None:
        raise ValueError(f"Could not parse {section_header} from {info_path}")

    shape = [
        int(part.strip()) for part in match.group("shape").split(",") if part.strip()
    ]
    logical_size = math.prod(shape)
    dtype = match.group("dtype")
    value_dtype = np.float64 if dtype.startswith("float") else np.int64
    values = np.fromstring(
        match.group("values").replace("\n", " "), sep=",", dtype=value_dtype
    )
    sliced = values[:logical_size]
    if dtype.startswith("float"):
        parsed_values = sliced.astype(np.float64).tolist()
    else:
        parsed_values = sliced.astype(np.int64).tolist()
    return QuantizedTensorExample(
        name=match.group("name"),
        shape=shape,
        exponent=int(match.group("exponent")),
        values=parsed_values,
        dtype=dtype,
    )


def load_quantized_examples(info_path: Path) -> QuantizedModelExamples:
    info_text = info_path.read_text(encoding="utf-8")
    return QuantizedModelExamples(
        input_tensor=parse_quantized_tensor_example(
            info_text, "test inputs value", info_path
        ),
        output_tensor=parse_quantized_tensor_example(
            info_text, "test outputs value", info_path
        ),
    )


def print_dataset_summary(metadata: ProcessedDataMetadata) -> None:
    print(
        "dataset_summary "
        f"sessions={sum(metadata.split_counts.values())} "
        f"sequence_length={metadata.sequence_length} "
        f"splits={metadata.split_counts}"
    )
