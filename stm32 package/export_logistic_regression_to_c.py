"""Export the Day 5 Logistic Regression model to C header arrays."""

from __future__ import annotations

from pathlib import Path
import json

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "classical" / "sensor_logistic_regression.joblib"
FEATURES_CSV = PROJECT_ROOT / "data" / "sensor" / "processed" / "sensor_features.csv"
OUTPUT_DIR = PROJECT_ROOT / "stm32 package" / "generated"
OUTPUT_HEADER = OUTPUT_DIR / "sensor_model_data.h"
OUTPUT_MANIFEST = OUTPUT_DIR / "sensor_model_manifest.json"

EXCLUDED_COLUMNS = {"split", "source_signal_id", "window_id", "class_id", "class_name"}
CLASS_NAMES = ["normal", "imbalance", "impulse_fault", "high_frequency_disturbance"]


def _c_float(value: float) -> str:
    return f"{float(value):.9g}f"


def _format_vector(values) -> str:
    return ", ".join(_c_float(value) for value in values)


def _format_matrix(rows) -> str:
    return ",\n".join(f"    {{ {_format_vector(row)} }}" for row in rows)


def _format_string_array(values) -> str:
    return ",\n".join(f'    "{value}"' for value in values)


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing model: {MODEL_PATH}\nRun notebooks/solutions/13_classical_ml_sensor_classification.ipynb first."
        )
    if not FEATURES_CSV.exists():
        raise FileNotFoundError(
            f"Missing feature table: {FEATURES_CSV}\nRun Notebook 12 or Notebook 13 first."
        )

    pipeline = joblib.load(MODEL_PATH)
    scaler = pipeline.named_steps["scaler"]
    model = pipeline.named_steps["model"]

    features = pd.read_csv(FEATURES_CSV, nrows=1)
    feature_columns = [column for column in features.columns if column not in EXCLUDED_COLUMNS]

    if len(feature_columns) != len(scaler.mean_):
        raise ValueError(
            f"Feature count mismatch: CSV has {len(feature_columns)}, model expects {len(scaler.mean_)}."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    header = f"""#ifndef SENSOR_MODEL_DATA_H
#define SENSOR_MODEL_DATA_H

#define SENSOR_NUM_FEATURES {len(feature_columns)}
#define SENSOR_NUM_CLASSES {len(CLASS_NAMES)}
#define SENSOR_WINDOW_SIZE 256

static const char *const SENSOR_FEATURE_NAMES[SENSOR_NUM_FEATURES] = {{
{_format_string_array(feature_columns)}
}};

static const char *const SENSOR_CLASS_NAMES[SENSOR_NUM_CLASSES] = {{
{_format_string_array(CLASS_NAMES)}
}};

static const float SENSOR_SCALER_MEAN[SENSOR_NUM_FEATURES] = {{
    {_format_vector(scaler.mean_)}
}};

static const float SENSOR_SCALER_SCALE[SENSOR_NUM_FEATURES] = {{
    {_format_vector(scaler.scale_)}
}};

static const float SENSOR_MODEL_COEFFICIENTS[SENSOR_NUM_CLASSES][SENSOR_NUM_FEATURES] = {{
{_format_matrix(model.coef_)}
}};

static const float SENSOR_MODEL_INTERCEPTS[SENSOR_NUM_CLASSES] = {{
    {_format_vector(model.intercept_)}
}};

#endif
"""
    OUTPUT_HEADER.write_text(header, encoding="utf-8")

    manifest = {
        "model_path": str(MODEL_PATH),
        "features_csv": str(FEATURES_CSV),
        "output_header": str(OUTPUT_HEADER),
        "feature_columns": feature_columns,
        "class_names": CLASS_NAMES,
    }
    OUTPUT_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Exported STM32 model data: {OUTPUT_HEADER}")
    print(f"Feature count: {len(feature_columns)}")
    print(f"Class count: {len(CLASS_NAMES)}")


if __name__ == "__main__":
    main()
