"""Reporting helpers for Day 5 model comparison."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ROOT


REPORTS_DIR = ROOT / "reports" / "day5"
METRICS_DIR = REPORTS_DIR / "metrics"


def model_size_to_kb(size_bytes: int) -> float:
    return float(size_bytes) / 1024.0


def estimate_classical_pipeline_memory(feature_count: int, model_size_bytes: int) -> dict[str, float]:
    """Educational host-side RAM estimate for feature-based pipeline."""

    input_window_bytes = 256 * 4
    feature_vector_bytes = feature_count * 8
    fft_buffer_bytes = 256 * 16
    return {
        "input_window_bytes": float(input_window_bytes),
        "feature_vector_bytes": float(feature_vector_bytes),
        "fft_buffer_bytes": float(fft_buffer_bytes),
        "model_bytes": float(model_size_bytes),
        "estimated_total_bytes": float(input_window_bytes + feature_vector_bytes + fft_buffer_bytes + model_size_bytes),
    }


def estimate_cnn_pipeline_memory(model_size_bytes: int, parameter_count: int) -> dict[str, float]:
    """Educational host-side RAM estimate for CNN pipeline."""

    input_tensor_bytes = 256 * 4
    activation_bytes = 32 * 128 * 4
    temp_buffer_bytes = 32 * 128 * 4
    parameter_bytes = parameter_count * 4
    return {
        "input_tensor_bytes": float(input_tensor_bytes),
        "activations_bytes": float(activation_bytes),
        "temp_buffers_bytes": float(temp_buffer_bytes),
        "weights_bytes": float(parameter_bytes),
        "model_file_bytes": float(model_size_bytes),
        "estimated_total_bytes": float(
            input_tensor_bytes + activation_bytes + temp_buffer_bytes + parameter_bytes
        ),
    }


def build_comparison_table(
    classical_results: dict[str, dict[str, Any]],
    cnn_results: dict[str, Any],
) -> pd.DataFrame:
    """Create model comparison table used in notebook/report sections."""

    rows = []
    for model_name, result in classical_results.items():
        rows.append(
            {
                "model": model_name,
                "accuracy": result["accuracy"],
                "f1_macro": result["f1_macro"],
                "model_size_kb": model_size_to_kb(result["model_size_bytes"]),
                "latency_mean_ms": result["latency_ms"]["mean_ms"],
                "latency_p95_ms": result["latency_ms"]["p95_ms"],
            }
        )

    rows.append(
        {
            "model": "cnn_1d",
            "accuracy": cnn_results["metrics"]["accuracy"],
            "f1_macro": cnn_results["metrics"]["f1_macro"],
            "model_size_kb": model_size_to_kb(cnn_results["model_size_bytes"]),
            "latency_mean_ms": cnn_results["latency_ms"]["mean_ms"],
            "latency_p95_ms": cnn_results["latency_ms"]["p95_ms"],
        }
    )

    return pd.DataFrame(rows)


def _table_to_markdown(frame: pd.DataFrame) -> str:
    """Render a markdown table without requiring optional tabulate dependency."""

    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in frame.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def write_day5_report(
    dataset_summary: dict[str, Any],
    classical_results: dict[str, dict[str, Any]],
    cnn_results: dict[str, Any],
    comparison_table: pd.DataFrame,
    output_path: Path | None = None,
) -> Path:
    """Write the required day5_model_comparison.md report."""

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    output = output_path or REPORTS_DIR / "day5_model_comparison.md"

    best_classical_name = max(classical_results, key=lambda key: classical_results[key]["f1_macro"])
    best_classical = classical_results[best_classical_name]

    recommendation = ""
    if cnn_results["metrics"]["f1_macro"] > best_classical["f1_macro"] + 0.02:
        recommendation = (
            "The 1D CNN gives stronger predictive quality on this synthetic benchmark. "
            "For ultra-low-power targets, deploy with TFLite quantization and validate RAM/energy on-device."
        )
    else:
        recommendation = (
            "The best classical model is competitive and may be simpler for tiny devices. "
            "However, the feature extraction cost (FFT + statistics) must be included in the deployment budget."
        )

    md = [
        "# Day 5 Sensor AI Model Comparison",
        "",
        "## Dataset",
        f"- Mode: {dataset_summary['mode']}",
        f"- Sampling rate: {dataset_summary['sampling_rate_hz']} Hz",
        f"- Signal duration: {dataset_summary['duration_seconds']} s",
        f"- Window size: {dataset_summary['window_size']} samples",
        f"- Overlap: {dataset_summary['overlap_fraction']}",
        "",
        "## Feature-Based Model",
        f"- Best classical model: **{best_classical_name}**",
        f"- Accuracy: {best_classical['accuracy']:.4f}",
        f"- F1 macro: {best_classical['f1_macro']:.4f}",
        "",
        "## 1D CNN",
        f"- Accuracy: {cnn_results['metrics']['accuracy']:.4f}",
        f"- F1 macro: {cnn_results['metrics']['f1_macro']:.4f}",
        f"- Parameters: {cnn_results['parameter_count']}",
        "",
        "## Accuracy Comparison",
        _table_to_markdown(comparison_table),
        "",
        "## Latency Comparison",
        "Host-side latency reflects educational comparison only. Embedded target latency must be measured on-device.",
        "",
        "## Model Size Comparison",
        "Model file sizes are measured from serialized artifacts.",
        "",
        "## TinyML Considerations",
        "- Compare full pipeline cost: acquisition + preprocessing + model inference.",
        "- Classical models may have tiny classifiers but can require expensive feature extraction.",
        "- CNNs can simplify the signal path and are quantization-friendly but may increase memory needs.",
        "",
        "## Limitations",
        "- Synthetic dataset; domain transfer to real machinery needs validation.",
        "- Host latency and RAM estimates are not MCU power measurements.",
        "",
        "## Recommended Deployment Strategy",
        recommendation,
    ]

    output.write_text("\n".join(md), encoding="utf-8")

    # Also keep a machine-readable summary.
    summary_json = {
        "best_classical": best_classical_name,
        "best_classical_f1": best_classical["f1_macro"],
        "cnn_f1": cnn_results["metrics"]["f1_macro"],
        "recommendation": recommendation,
    }
    (METRICS_DIR / "day5_summary.json").write_text(json.dumps(summary_json, indent=2), encoding="utf-8")

    return output
