#!/usr/bin/env python3
"""Run TensorRT inference for Day 5 ONNX-exported models on Jetson Orin Nano."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

try:
    import tensorrt as trt
    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda
except Exception as exc:  # pragma: no cover - Jetson-only dependency path
    raise RuntimeError("This script must run on Jetson with TensorRT and PyCUDA installed.") from exc


PACKAGE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE_DIR = PACKAGE_DIR / "bundle"
DEFAULT_ENGINE_DIR = PACKAGE_DIR / "engines"

MODEL_FILES = {
    "fashion": {
        "name": "fashion_mnist_cnn",
        "shape": (1, 28, 28, 1),
    },
    "catsdogs": {
        "name": "cats_dogs_transfer",
        "shape": (1, 96, 96, 3),
    },
}


def load_metadata(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_engine(path: Path):
    logger = trt.Logger(trt.Logger.WARNING)
    with path.open("rb") as handle, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(handle.read())
    if engine is None:
        raise RuntimeError(f"Unable to deserialize TensorRT engine: {path}")
    return engine


def _tensor_names(engine):
    if hasattr(engine, "num_io_tensors"):
        names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        input_names = [name for name in names if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT]
        output_names = [name for name in names if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT]
        return input_names, output_names
    input_names = [engine.get_binding_name(i) for i in range(engine.num_bindings) if engine.binding_is_input(i)]
    output_names = [engine.get_binding_name(i) for i in range(engine.num_bindings) if not engine.binding_is_input(i)]
    return input_names, output_names


def _tensor_shape(engine, context, name: str):
    if hasattr(engine, "get_tensor_shape"):
        shape = tuple(context.get_tensor_shape(name))
        if any(dim < 0 for dim in shape):
            shape = tuple(engine.get_tensor_shape(name))
        return shape
    return tuple(engine.get_binding_shape(engine.get_binding_index(name)))


def _tensor_dtype(engine, name: str):
    if hasattr(engine, "get_tensor_dtype"):
        return trt.nptype(engine.get_tensor_dtype(name))
    return trt.nptype(engine.get_binding_dtype(engine.get_binding_index(name)))


def execute_engine(engine, input_array: np.ndarray, warmup: int) -> tuple[np.ndarray, float]:
    context = engine.create_execution_context()
    input_names, output_names = _tensor_names(engine)
    input_name = input_names[0]
    output_name = output_names[0]

    if hasattr(context, "set_input_shape"):
        context.set_input_shape(input_name, tuple(input_array.shape))
    elif -1 in tuple(engine.get_binding_shape(engine.get_binding_index(input_name))):
        context.set_binding_shape(engine.get_binding_index(input_name), tuple(input_array.shape))

    output_shape = tuple(int(dim) for dim in _tensor_shape(engine, context, output_name))
    output_dtype = _tensor_dtype(engine, output_name)
    input_array = np.ascontiguousarray(input_array.astype(np.float32))
    output_array = np.empty(output_shape, dtype=output_dtype)

    d_input = cuda.mem_alloc(input_array.nbytes)
    d_output = cuda.mem_alloc(output_array.nbytes)
    stream = cuda.Stream()

    if hasattr(context, "set_tensor_address"):
        context.set_tensor_address(input_name, int(d_input))
        context.set_tensor_address(output_name, int(d_output))
        execute = lambda: context.execute_async_v3(stream_handle=stream.handle)
    else:
        bindings = [0] * engine.num_bindings
        bindings[engine.get_binding_index(input_name)] = int(d_input)
        bindings[engine.get_binding_index(output_name)] = int(d_output)
        execute = lambda: context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

    cuda.memcpy_htod_async(d_input, input_array, stream)
    for _ in range(warmup):
        execute()
    stream.synchronize()

    start = time.perf_counter()
    execute()
    cuda.memcpy_dtoh_async(output_array, d_output, stream)
    stream.synchronize()
    latency_ms = (time.perf_counter() - start) * 1000.0
    return output_array, latency_ms


def preprocess_image(image_path: Path, metadata: dict[str, object]) -> np.ndarray:
    shape = metadata.get("input_shape", [1, 224, 224, 3])
    height = int(shape[1])
    width = int(shape[2])
    channels = int(shape[3])

    if channels == 1:
        image = Image.open(image_path).convert("L").resize((width, height))
        array = np.asarray(image, dtype=np.float32)[..., np.newaxis] / 255.0
    else:
        image = Image.open(image_path).convert("RGB").resize((width, height))
        array = np.asarray(image, dtype=np.float32) / 255.0
    return array[np.newaxis, ...].astype(np.float32)


def load_input(args: argparse.Namespace, metadata: dict[str, object], model_key: str) -> np.ndarray:
    if args.npy is not None:
        return np.load(args.npy).astype(np.float32)
    if args.image is not None:
        return preprocess_image(args.image, metadata)

    sample_path = metadata.get("sample_path")
    if sample_path:
        candidate = Path(str(sample_path))
        if not candidate.exists():
            candidate = args.bundle_dir / candidate.name
        if candidate.exists():
            return np.load(candidate).astype(np.float32)

    fallback = args.bundle_dir / ("fashion_mnist_sample.npy" if model_key == "fashion" else "cats_dogs_sample.npy")
    if fallback.exists():
        return np.load(fallback).astype(np.float32)
    raise FileNotFoundError("No input provided and no bundled sample .npy was found.")


def summarize(output: np.ndarray, metadata: dict[str, object]) -> dict[str, object]:
    values = np.asarray(output, dtype=np.float32).reshape(-1)
    class_names = list(metadata.get("class_names", []))
    threshold = metadata.get("prediction_threshold")
    activation = metadata.get("output_activation", "")
    task = metadata.get("task", "")

    if values.size == 1 or task == "binary_classification" or activation == "sigmoid":
        score = float(values[0])
        predicted_index = int(score >= float(threshold or 0.5))
        confidence = score if predicted_index == 1 else 1.0 - score
    else:
        predicted_index = int(np.argmax(values))
        confidence = float(values[predicted_index])

    predicted_class = class_names[predicted_index] if predicted_index < len(class_names) else str(predicted_index)
    return {
        "predicted_index": predicted_index,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "raw_output": values.tolist(),
        "output_shape": list(output.shape),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_FILES), required=True)
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp16")
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE_DIR)
    parser.add_argument("--engine-dir", type=Path, default=DEFAULT_ENGINE_DIR)
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--npy", type=Path)
    parser.add_argument("--warmup", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = MODEL_FILES[args.model]
    model_name = spec["name"]
    metadata_path = args.bundle_dir / f"{model_name}.json"
    engine_path = args.engine or (args.engine_dir / f"{model_name}_{args.precision}.engine")

    metadata = load_metadata(metadata_path)
    input_array = load_input(args, metadata, args.model)
    engine = load_engine(engine_path)
    output, latency_ms = execute_engine(engine, input_array, args.warmup)
    result = summarize(output, metadata)
    result.update(
        {
            "latency_ms": latency_ms,
            "engine_path": str(engine_path),
            "input_shape": list(input_array.shape),
            "model": args.model,
        }
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
