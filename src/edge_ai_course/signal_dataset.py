"""Dataset assembly utilities with leakage-safe splitting for Day 5."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT
from .signal_features import extract_features
from .signal_generator import CLASS_ID_TO_NAME, SignalConfig, generate_source_signals


SENSOR_DATA_ROOT = ROOT / "data" / "sensor"
SENSOR_RAW_DIR = SENSOR_DATA_ROOT / "raw"
SENSOR_PROCESSED_DIR = SENSOR_DATA_ROOT / "processed"
SENSOR_METADATA_DIR = SENSOR_DATA_ROOT / "metadata"


def _stratified_source_split(
    source_records: list[dict[str, object]],
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, list[dict[str, object]]]:
    train_records: list[dict[str, object]] = []
    val_records: list[dict[str, object]] = []
    test_records: list[dict[str, object]] = []

    rng = np.random.default_rng(seed)
    for class_id in sorted(CLASS_ID_TO_NAME):
        class_subset = [r for r in source_records if int(r["class_id"]) == class_id]
        indices = np.arange(len(class_subset))
        rng.shuffle(indices)

        n_total = len(class_subset)
        n_test = max(1, int(round(n_total * test_fraction)))
        n_val = max(1, int(round(n_total * val_fraction)))
        n_train = max(1, n_total - n_val - n_test)

        # Guard against rounding causing overflow.
        if n_train + n_val + n_test > n_total:
            n_train = n_total - n_val - n_test
        if n_train <= 0:
            n_train = 1
            if n_val > 1:
                n_val -= 1
            elif n_test > 1:
                n_test -= 1

        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val : n_train + n_val + n_test]

        train_records.extend(class_subset[i] for i in train_idx)
        val_records.extend(class_subset[i] for i in val_idx)
        test_records.extend(class_subset[i] for i in test_idx)

    return {"train": train_records, "validation": val_records, "test": test_records}


def create_windows(signal: np.ndarray, window_size: int, overlap_fraction: float) -> np.ndarray:
    """Create overlapping windows from one source signal."""

    if not (0.0 <= overlap_fraction < 1.0):
        raise ValueError("overlap_fraction must be in [0, 1)")
    if window_size <= 0:
        raise ValueError("window_size must be > 0")
    if signal.size < window_size:
        raise ValueError("signal is shorter than window_size")

    step_size = max(1, int(round(window_size * (1.0 - overlap_fraction))))
    windows = []
    for start in range(0, signal.size - window_size + 1, step_size):
        windows.append(signal[start : start + window_size])
    return np.asarray(windows, dtype=np.float32)


def _windows_to_feature_rows(
    split_name: str,
    split_records: list[dict[str, object]],
    window_size: int,
    overlap_fraction: float,
    sampling_rate: int,
) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray, list[str]]:
    rows: list[dict[str, object]] = []
    x_windows: list[np.ndarray] = []
    y_labels: list[int] = []
    source_ids: list[str] = []

    for record in split_records:
        source_id = str(record["source_signal_id"])
        class_id = int(record["class_id"])
        class_name = str(record["class_name"])
        windows = create_windows(
            signal=np.asarray(record["signal"], dtype=np.float32),
            window_size=window_size,
            overlap_fraction=overlap_fraction,
        )
        for local_id, window in enumerate(windows):
            feats = extract_features(window, sampling_rate=sampling_rate)
            row = {
                "split": split_name,
                "source_signal_id": source_id,
                "window_id": f"{source_id}_w{local_id:03d}",
                "class_id": class_id,
                "class_name": class_name,
                **feats,
            }
            rows.append(row)
            x_windows.append(window)
            y_labels.append(class_id)
            source_ids.append(source_id)

    return rows, np.asarray(x_windows, dtype=np.float32), np.asarray(y_labels, dtype=np.int32), source_ids


def build_sensor_dataset(
    config: SignalConfig | None = None,
    mode: str = "normal",
    window_size: int = 256,
    overlap_fraction: float = 0.5,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> dict[str, object]:
    """Generate source signals and produce leakage-safe window datasets."""

    if abs((train_fraction + val_fraction + test_fraction) - 1.0) > 1e-6:
        raise ValueError("train/val/test fractions must sum to 1.0")

    cfg = config or SignalConfig()
    source_examples = generate_source_signals(config=cfg, mode=mode)
    source_records = [
        {
            "source_signal_id": ex.source_signal_id,
            "class_id": ex.class_id,
            "class_name": ex.class_name,
            "sample_rate": ex.sample_rate,
            "signal": ex.signal,
        }
        for ex in source_examples
    ]

    split = _stratified_source_split(
        source_records=source_records,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=cfg.random_seed,
    )

    rows_train, x_train, y_train, source_train = _windows_to_feature_rows(
        split_name="train",
        split_records=split["train"],
        window_size=window_size,
        overlap_fraction=overlap_fraction,
        sampling_rate=cfg.sampling_rate,
    )
    rows_val, x_val, y_val, source_val = _windows_to_feature_rows(
        split_name="validation",
        split_records=split["validation"],
        window_size=window_size,
        overlap_fraction=overlap_fraction,
        sampling_rate=cfg.sampling_rate,
    )
    rows_test, x_test, y_test, source_test = _windows_to_feature_rows(
        split_name="test",
        split_records=split["test"],
        window_size=window_size,
        overlap_fraction=overlap_fraction,
        sampling_rate=cfg.sampling_rate,
    )

    feature_table = pd.DataFrame(rows_train + rows_val + rows_test)

    return {
        "config": cfg,
        "mode": mode,
        "source_records": source_records,
        "feature_table": feature_table,
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_test,
        "y_test": y_test,
        "source_ids_train": source_train,
        "source_ids_val": source_val,
        "source_ids_test": source_test,
        "window_size": window_size,
        "overlap_fraction": overlap_fraction,
    }


def save_dataset_artifacts(dataset: dict[str, object]) -> dict[str, Path]:
    """Save raw arrays, feature table, and metadata for Day 5."""

    SENSOR_RAW_DIR.mkdir(parents=True, exist_ok=True)
    SENSOR_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SENSOR_METADATA_DIR.mkdir(parents=True, exist_ok=True)

    feature_path = SENSOR_PROCESSED_DIR / "sensor_features.csv"
    dataset["feature_table"].to_csv(feature_path, index=False)

    arrays_path = SENSOR_PROCESSED_DIR / "sensor_windows.npz"
    np.savez_compressed(
        arrays_path,
        x_train=dataset["x_train"],
        y_train=dataset["y_train"],
        x_val=dataset["x_val"],
        y_val=dataset["y_val"],
        x_test=dataset["x_test"],
        y_test=dataset["y_test"],
    )

    source_path = SENSOR_RAW_DIR / "sensor_source_signals.npz"
    source_ids = np.array([r["source_signal_id"] for r in dataset["source_records"]], dtype=object)
    class_ids = np.array([r["class_id"] for r in dataset["source_records"]], dtype=np.int32)
    signals = np.stack([np.asarray(r["signal"], dtype=np.float32) for r in dataset["source_records"]])
    np.savez_compressed(source_path, source_ids=source_ids, class_ids=class_ids, signals=signals)

    metadata = {
        "mode": dataset["mode"],
        "signal_config": asdict(dataset["config"]),
        "window_size": dataset["window_size"],
        "overlap_fraction": dataset["overlap_fraction"],
        "split_window_counts": {
            "train": int(len(dataset["x_train"])),
            "validation": int(len(dataset["x_val"])),
            "test": int(len(dataset["x_test"])),
        },
        "source_counts": {
            "train": int(len(set(dataset["source_ids_train"]))),
            "validation": int(len(set(dataset["source_ids_val"]))),
            "test": int(len(set(dataset["source_ids_test"]))),
        },
    }
    metadata_path = SENSOR_METADATA_DIR / "dataset_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "feature_csv": feature_path,
        "window_npz": arrays_path,
        "source_npz": source_path,
        "metadata_json": metadata_path,
    }


def assert_no_source_leakage(dataset: dict[str, object]) -> None:
    """Assert that source IDs do not overlap across splits."""

    train_ids = set(dataset["source_ids_train"])
    val_ids = set(dataset["source_ids_val"])
    test_ids = set(dataset["source_ids_test"])

    if train_ids & val_ids:
        raise AssertionError("Train/validation source leakage detected")
    if train_ids & test_ids:
        raise AssertionError("Train/test source leakage detected")
    if val_ids & test_ids:
        raise AssertionError("Validation/test source leakage detected")
