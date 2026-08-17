"""Classical ML and 1D CNN training helpers for Day 5."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

from .benchmarking import benchmark_callable
from .config import ROOT


CLASSICAL_MODELS_DIR = ROOT / "models" / "classical"
KERAS_MODELS_DIR = ROOT / "models" / "keras"
TFLITE_MODELS_DIR = ROOT / "models" / "tflite"
REPORT_METRICS_DIR = ROOT / "reports" / "day5" / "metrics"


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Compute required classification metrics."""

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, digits=4, zero_division=0),
    }


def _latency_per_sample(model: Any, x_eval: np.ndarray, runs: int = 150) -> dict[str, float]:
    rng = np.random.default_rng(123)
    idx_pool = np.arange(len(x_eval))

    def _predict_once() -> Any:
        idx = int(rng.choice(idx_pool))
        return model.predict(x_eval[idx : idx + 1])

    stats = benchmark_callable(_predict_once, runs=runs, warmup=min(20, max(5, runs // 5)))
    return stats


def train_classical_models(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
) -> dict[str, dict[str, Any]]:
    """Train logistic regression and random forest baselines."""

    CLASSICAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_METRICS_DIR.mkdir(parents=True, exist_ok=True)

    specs: dict[str, Pipeline] = {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        solver="lbfgs",
                        multi_class="multinomial",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("model", RandomForestClassifier(n_estimators=180, random_state=42, n_jobs=-1)),
            ]
        ),
    }

    results: dict[str, dict[str, Any]] = {}
    for model_name, pipeline in specs.items():
        train_start = time.perf_counter()
        pipeline.fit(train_features, train_labels)
        train_time_ms = (time.perf_counter() - train_start) * 1000.0

        preds = pipeline.predict(test_features)
        metrics = compute_metrics(test_labels, preds)
        latency = _latency_per_sample(pipeline, test_features)

        model_path = CLASSICAL_MODELS_DIR / f"sensor_{model_name}.joblib"
        joblib.dump(pipeline, model_path)
        size_bytes = model_path.stat().st_size

        metrics.update(
            {
                "train_time_ms": float(train_time_ms),
                "latency_ms": latency,
                "model_path": str(model_path),
                "model_size_bytes": int(size_bytes),
            }
        )

        metrics_path = REPORT_METRICS_DIR / f"{model_name}_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        results[model_name] = metrics

    return results


def build_cnn_model(window_size: int, num_classes: int = 4) -> tf.keras.Model:
    """Build a compact edge-oriented 1D CNN."""

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(window_size, 1)),
            tf.keras.layers.Conv1D(16, 5, activation="relu", padding="same"),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Conv1D(32, 5, activation="relu", padding="same"),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def _prepare_cnn_inputs(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    train_mean = float(np.mean(x_train))
    train_std = float(np.std(x_train) + 1e-8)

    def _norm(arr: np.ndarray) -> np.ndarray:
        return ((arr - train_mean) / train_std)[..., np.newaxis].astype(np.float32)

    return _norm(x_train), _norm(x_val), _norm(x_test), {"mean": train_mean, "std": train_std}


def _keras_latency_stats(model: tf.keras.Model, x_eval: np.ndarray, runs: int = 150) -> dict[str, float]:
    rng = np.random.default_rng(1234)
    idx_pool = np.arange(len(x_eval))

    def _infer_once() -> np.ndarray:
        idx = int(rng.choice(idx_pool))
        return model.predict(x_eval[idx : idx + 1], verbose=0)

    return benchmark_callable(_infer_once, runs=runs, warmup=min(20, max(5, runs // 5)))


def train_cnn_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 15,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Train, evaluate, and save the 1D CNN model."""

    KERAS_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_METRICS_DIR.mkdir(parents=True, exist_ok=True)

    x_train_n, x_val_n, x_test_n, norm_stats = _prepare_cnn_inputs(x_train, x_val, x_test)

    model = build_cnn_model(window_size=x_train.shape[1], num_classes=4)
    checkpoint_path = KERAS_MODELS_DIR / "sensor_1d_cnn.keras"

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ]

    history = model.fit(
        x_train_n,
        y_train,
        validation_data=(x_val_n, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=callbacks,
    )

    best_model = tf.keras.models.load_model(checkpoint_path)
    test_probs = best_model.predict(x_test_n, verbose=0)
    test_pred = np.argmax(test_probs, axis=1)
    metrics = compute_metrics(y_test, test_pred)

    latency = _keras_latency_stats(best_model, x_test_n)
    size_bytes = checkpoint_path.stat().st_size

    results = {
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
        "metrics": metrics,
        "latency_ms": latency,
        "normalization": norm_stats,
        "model_path": str(checkpoint_path),
        "model_size_bytes": int(size_bytes),
        "parameter_count": int(best_model.count_params()),
    }

    metrics_path = REPORT_METRICS_DIR / "cnn_metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    return {
        **results,
        "model": best_model,
        "x_test_normalized": x_test_n,
        "test_predictions": test_pred,
    }


def convert_to_tflite(
    keras_model: tf.keras.Model,
    representative_data: np.ndarray,
) -> dict[str, str]:
    """Convert Keras CNN to TFLite FP32 and attempt INT8 conversion."""

    TFLITE_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    fp32_path = TFLITE_MODELS_DIR / "sensor_cnn_fp32.tflite"
    int8_path = TFLITE_MODELS_DIR / "sensor_cnn_int8.tflite"

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    fp32_model = converter.convert()
    fp32_path.write_bytes(fp32_model)

    output = {
        "fp32_path": str(fp32_path),
        "fp32_size_bytes": str(fp32_path.stat().st_size),
    }

    try:
        converter_int8 = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]

        def representative_dataset() -> Any:
            for sample in representative_data[:256]:
                yield [sample[np.newaxis, ...].astype(np.float32)]

        converter_int8.representative_dataset = representative_dataset
        converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter_int8.inference_input_type = tf.int8
        converter_int8.inference_output_type = tf.int8
        int8_model = converter_int8.convert()
        int8_path.write_bytes(int8_model)
        output["int8_path"] = str(int8_path)
        output["int8_size_bytes"] = str(int8_path.stat().st_size)
    except Exception as exc:  # pragma: no cover - environment-dependent path
        output["int8_error"] = str(exc)

    return output
