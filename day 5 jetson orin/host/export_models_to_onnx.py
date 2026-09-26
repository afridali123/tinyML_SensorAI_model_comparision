"""Export Notebook 09/10 Keras models to ONNX and prepare a Jetson Orin bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = ROOT / "day 5 jetson orin"
BUNDLE_DIR = PACKAGE_ROOT / "bundle"
KERAS_DIR = ROOT / "models" / "keras"
NOTEBOOK10_OUTPUT_DIR = ROOT / "notebook10_outputs" / "models"

FASHION_MODEL_PATH = KERAS_DIR / "notebook09_fashion_mnist_cnn.keras"
CATS_DOGS_MODEL_CANDIDATES = [
    KERAS_DIR / "notebook10_cats_dogs_transfer.keras",
    NOTEBOOK10_OUTPUT_DIR / "notebook10_cats_dogs_transfer.keras",
]

FASHION_CLASSES = [
    "T-shirt",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]
CATS_DOGS_CLASSES = ["cats", "dogs"]


def _lazy_import_ml() -> tuple[Any, Any, Any, Any]:
    import onnx
    import onnxruntime as ort
    import tensorflow as tf
    import tf2onnx

    return onnx, ort, tf, tf2onnx


def _find_first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _build_fashion_model(tf: Any) -> Any:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(16, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def ensure_fashion_model(tf: Any, train_missing: bool) -> tuple[Path, Path]:
    (_, _), (x_test_raw, _) = tf.keras.datasets.fashion_mnist.load_data()
    sample = (x_test_raw[:1].astype("float32") / 255.0)[..., np.newaxis]
    sample_path = BUNDLE_DIR / "fashion_mnist_sample.npy"
    np.save(sample_path, sample.astype("float32"))

    if FASHION_MODEL_PATH.exists():
        return FASHION_MODEL_PATH, sample_path

    if not train_missing:
        raise FileNotFoundError(
            f"Missing {FASHION_MODEL_PATH}. Run Notebook 09 with a save step, or rerun this script with --train-missing-fashion."
        )

    (x_train_raw, y_train), _ = tf.keras.datasets.fashion_mnist.load_data()
    x_train = (x_train_raw[:6000].astype("float32") / 255.0)[..., np.newaxis]
    y_train = y_train[:6000]
    model = _build_fashion_model(tf)
    model.fit(
        x_train[:5000],
        y_train[:5000],
        validation_data=(x_train[5000:], y_train[5000:]),
        epochs=2,
        batch_size=64,
        verbose=1,
    )
    FASHION_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(FASHION_MODEL_PATH)
    return FASHION_MODEL_PATH, sample_path


def ensure_cats_dogs_model() -> tuple[Path, Path | None]:
    model_path = _find_first_existing(CATS_DOGS_MODEL_CANDIDATES)
    if model_path is None:
        raise FileNotFoundError(
            "Missing Notebook 10 model. Run notebooks/solutions/10_keras_cats_dogs_transfer_learning.ipynb first."
        )

    sample_candidates = [
        ROOT / "data" / "processed" / "notebook10_cats_dogs_sample.npy",
        ROOT / "notebook10_data" / "processed" / "notebook10_cats_dogs_sample.npy",
    ]
    sample_path = _find_first_existing(sample_candidates)
    return model_path, sample_path


def export_onnx_model(
    tf2onnx: Any,
    onnx: Any,
    ort: Any,
    tf: Any,
    model_path: Path,
    output_path: Path,
    metadata_path: Path,
    input_shape: tuple[int, ...],
    class_names: list[str],
    sample_path: Path | None,
    task: str,
    prediction_threshold: float | None,
    output_activation: str,
) -> dict[str, object]:
    model = tf.keras.models.load_model(model_path)
    signature = [tf.TensorSpec(shape=input_shape, dtype=tf.float32, name="input")]
    tf2onnx.convert.from_keras(model, input_signature=signature, opset=13, output_path=str(output_path))

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    comparison: dict[str, object] = {}
    if sample_path is not None and sample_path.exists():
        sample = np.load(sample_path).astype("float32")
        keras_output = model.predict(sample, verbose=0)
        session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
        onnx_output = session.run(None, {session.get_inputs()[0].name: sample})[0]
        diff = np.abs(keras_output - onnx_output)
        comparison = {
            "max_abs_error": float(diff.max()),
            "mean_abs_error": float(diff.mean()),
            "class_match": bool(np.array_equal(keras_output.argmax(axis=-1), onnx_output.argmax(axis=-1)))
            if keras_output.shape[-1] > 1
            else bool((keras_output.reshape(-1) >= 0.5)[0] == (onnx_output.reshape(-1) >= 0.5)[0]),
        }

    metadata = {
        "model_name": output_path.stem,
        "source_model": str(model_path),
        "task": task,
        "input_name": "input",
        "output_name": "output",
        "input_shape": list(input_shape),
        "input_dtype": "float32",
        "layout": "NHWC",
        "normalization": {"scale": 1.0 / 255.0, "offset": 0.0, "note": "Image inputs should be converted to float32 0..1 before inference."},
        "class_names": class_names,
        "onnx_opset": 13,
        "prediction_threshold": prediction_threshold,
        "output_activation": output_activation,
        "sample_path": str(sample_path) if sample_path is not None else None,
        "onnx_parity": comparison,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-missing-fashion", action="store_true")
    args = parser.parse_args()

    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    onnx, ort, tf, tf2onnx = _lazy_import_ml()

    fashion_model_path, fashion_sample_path = ensure_fashion_model(tf, args.train_missing_fashion)
    cats_dogs_model_path, cats_dogs_sample_path = ensure_cats_dogs_model()

    outputs = []
    outputs.append(
        export_onnx_model(
            tf2onnx,
            onnx,
            ort,
            tf,
            model_path=fashion_model_path,
            output_path=BUNDLE_DIR / "fashion_mnist_cnn.onnx",
            metadata_path=BUNDLE_DIR / "fashion_mnist_cnn.json",
            input_shape=(1, 28, 28, 1),
            class_names=FASHION_CLASSES,
            sample_path=fashion_sample_path,
            task="multiclass_classification",
            prediction_threshold=None,
            output_activation="softmax",
        )
    )
    outputs.append(
        export_onnx_model(
            tf2onnx,
            onnx,
            ort,
            tf,
            model_path=cats_dogs_model_path,
            output_path=BUNDLE_DIR / "cats_dogs_transfer.onnx",
            metadata_path=BUNDLE_DIR / "cats_dogs_transfer.json",
            input_shape=(1, 96, 96, 3),
            class_names=CATS_DOGS_CLASSES,
            sample_path=cats_dogs_sample_path,
            task="binary_classification",
            prediction_threshold=0.5,
            output_activation="sigmoid",
        )
    )

    if cats_dogs_sample_path is not None and cats_dogs_sample_path.exists():
        shutil.copy2(cats_dogs_sample_path, BUNDLE_DIR / "cats_dogs_sample.npy")

    summary_path = BUNDLE_DIR / "export_summary.json"
    summary_path.write_text(json.dumps({"models": outputs}, indent=2), encoding="utf-8")
    print(json.dumps({"bundle_dir": str(BUNDLE_DIR), "summary": str(summary_path), "models": outputs}, indent=2))


if __name__ == "__main__":
    main()
