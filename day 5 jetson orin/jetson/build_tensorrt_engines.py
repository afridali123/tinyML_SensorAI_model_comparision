#!/usr/bin/env python3
"""Build TensorRT engines from the Day 5 ONNX bundle on Jetson Orin Nano."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


PACKAGE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE_DIR = PACKAGE_DIR / "bundle"
DEFAULT_ENGINE_DIR = PACKAGE_DIR / "engines"

MODEL_SPECS = {
    "fashion": {
        "name": "fashion_mnist_cnn",
        "shape": (1, 28, 28, 1),
    },
    "catsdogs": {
        "name": "cats_dogs_transfer",
        "shape": (1, 96, 96, 3),
    },
}


def configure_workspace(config: Any, trt: Any, workspace_mb: int) -> None:
    workspace_bytes = workspace_mb * 1024 * 1024
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    else:
        config.max_workspace_size = workspace_bytes


def build_engine(
    trt: Any,
    onnx_path: Path,
    engine_path: Path,
    precision: str,
    workspace_mb: int,
    expected_shape: tuple[int, ...],
) -> None:
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    if not onnx_path.is_file():
        raise FileNotFoundError(f"Missing ONNX file: {onnx_path}")

    if not parser.parse(onnx_path.read_bytes()):
        errors = "\n".join(str(parser.get_error(index)) for index in range(parser.num_errors))
        raise RuntimeError(f"Unable to parse {onnx_path}:\n{errors}")

    config = builder.create_builder_config()
    configure_workspace(config, trt, workspace_mb)

    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            raise RuntimeError("This TensorRT platform does not report fast FP16 support.")
        config.set_flag(trt.BuilderFlag.FP16)

    input_tensor = network.get_input(0)
    parsed_shape = tuple(int(dimension) for dimension in input_tensor.shape)
    if any(dimension < 0 for dimension in parsed_shape):
        profile = builder.create_optimization_profile()
        profile.set_shape(input_tensor.name, expected_shape, expected_shape, expected_shape)
        config.add_optimization_profile(profile)
    elif parsed_shape != expected_shape:
        raise RuntimeError(
            f"Unexpected input shape for {onnx_path.name}: {parsed_shape}; expected {expected_shape}."
        )

    print(f"Building {engine_path} from {onnx_path}")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError(f"TensorRT failed to build an engine from {onnx_path}")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized_engine))
    print(f"Wrote {engine_path} ({engine_path.stat().st_size} bytes)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE_DIR)
    parser.add_argument("--engine-dir", type=Path, default=DEFAULT_ENGINE_DIR)
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp16")
    parser.add_argument("--workspace", type=int, default=1024, help="TensorRT workspace size in MiB")
    parser.add_argument(
        "--model",
        choices=["all", *MODEL_SPECS],
        default="all",
        help="Build both models or one selected model",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        import tensorrt as trt
    except ImportError as exc:
        raise RuntimeError("TensorRT Python bindings are required; run this script on the Jetson.") from exc

    selected_models = MODEL_SPECS if args.model == "all" else {args.model: MODEL_SPECS[args.model]}
    for spec in selected_models.values():
        model_name = str(spec["name"])
        build_engine(
            trt=trt,
            onnx_path=args.bundle_dir / f"{model_name}.onnx",
            engine_path=args.engine_dir / f"{model_name}_{args.precision}.engine",
            precision=args.precision,
            workspace_mb=args.workspace,
            expected_shape=spec["shape"],
        )

    print(f"TensorRT engines written to {args.engine_dir}")


if __name__ == "__main__":
    main()