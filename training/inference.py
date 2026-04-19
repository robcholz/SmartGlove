from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import onnxruntime as ort
import serial
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.gesture_model import (
    GestureMLP,
    TRAINING_DIR,
    build_scaler_from_metadata,
    load_model_metadata,
    softmax,
)
from tests.collect_data import (
    SampleBuffer,
    firmware_image_path,
    open_serial_with_retry,
    prompt_serial_port,
    run_command,
    update_sample,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run live gesture inference on this computer using glove serial data."
    )
    parser.add_argument("--metadata", type=Path, default=TRAINING_DIR / "glove_model_metadata.json")
    parser.add_argument("--weights", type=Path, default=TRAINING_DIR / "glove_model.pt")
    parser.add_argument("--onnx", type=Path, default=TRAINING_DIR / "glove_model.onnx")
    parser.add_argument("--model-type", choices=["onnx", "torch"], default="onnx")
    parser.add_argument("--bin", default="driver")
    parser.add_argument("--port")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-flash", action="store_true")
    parser.add_argument("--predict-every", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--verbose-stream", action="store_true")
    parser.add_argument("--max-seconds", type=float)
    return parser.parse_args()


def build_frame(sample: SampleBuffer) -> np.ndarray:
    return np.asarray(
        [
            float(sample.values["thumb"]),
            float(sample.values["index"]),
            float(sample.values["middle"]),
            float(sample.values["ring"]),
            float(sample.values["pinky"]),
            float(sample.values["acc_x"]),
            float(sample.values["acc_y"]),
            float(sample.values["acc_z"]),
            float(sample.values["vec_x"]),
            float(sample.values["vec_y"]),
            float(sample.values["vec_z"]),
        ],
        dtype=np.float32,
    )


def format_topk(probabilities: np.ndarray, labels: list[str], top_k: int) -> str:
    top_indices = np.argsort(probabilities)[::-1][:top_k]
    return ", ".join(f"{labels[index]}:{probabilities[index]:.3f}" for index in top_indices)


def main() -> None:
    args = parse_args()
    metadata = load_model_metadata(args.metadata)
    scaler = build_scaler_from_metadata(metadata)
    port = prompt_serial_port(args.port)
    env = dict(os.environ)
    env["ESPFLASH_PORT"] = port

    if not args.skip_build:
        run_command(["cargo", "build", "--bin", args.bin], env)
    if not args.skip_flash:
        run_command(
            [
                "espflash",
                "flash",
                "--port",
                port,
                "--non-interactive",
                str(firmware_image_path(args.bin)),
            ],
            env,
        )

    if args.model_type == "onnx":
        session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        model = None
    else:
        session = None
        input_name = None
        model = GestureMLP(
            sequence_length=metadata.sequence_length,
            feature_count=len(metadata.feature_columns),
            num_classes=len(metadata.labels),
            hidden_dims=metadata.hidden_dims,
        )
        state_dict = torch.load(args.weights, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()

    serial_port = open_serial_with_retry(port)
    sample = SampleBuffer()
    window: deque[np.ndarray] = deque(maxlen=metadata.sequence_length)
    sample_counter = 0
    start_time = time.monotonic()

    print(
        f"listening_on={port} sequence_length={metadata.sequence_length} "
        f"model_type={args.model_type}"
    )
    try:
        while True:
            if args.max_seconds is not None and (time.monotonic() - start_time) >= args.max_seconds:
                break

            raw_line = serial_port.readline()
            if not raw_line:
                continue
            line = raw_line.decode("utf-8", errors="replace").rstrip()
            is_complete = update_sample(sample, line)
            if args.verbose_stream and line:
                print(line)
            if not is_complete:
                continue

            frame = build_frame(sample)
            sample.reset()
            window.append(frame)
            sample_counter += 1

            if len(window) < metadata.sequence_length:
                continue
            if sample_counter % args.predict_every != 0:
                continue

            window_array = np.stack(window, axis=0).astype(np.float32)
            normalized = scaler.transform(window_array).astype(np.float32)
            batch = normalized[None, :, :]

            if session is not None and input_name is not None:
                logits = session.run(None, {input_name: batch})[0][0]
            else:
                with torch.no_grad():
                    logits = model(torch.from_numpy(batch).float()).cpu().numpy()[0]

            probabilities = softmax(logits[None, :])[0]
            predicted_index = int(np.argmax(probabilities))
            confidence = float(probabilities[predicted_index])
            if confidence < args.min_confidence:
                continue

            print(
                f"sample={sample_counter} "
                f"predicted={metadata.labels[predicted_index]} "
                f"confidence={confidence:.3f} "
                f"topk=[{format_topk(probabilities, metadata.labels, args.top_k)}]"
            )
    except KeyboardInterrupt:
        pass
    finally:
        try:
            serial_port.close()
        except serial.SerialException:
            pass


if __name__ == "__main__":
    main()
