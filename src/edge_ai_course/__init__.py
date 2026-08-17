"""Standalone Day 5 Sensor AI utilities."""

from .model_analysis import build_comparison_table, write_day5_report
from .signal_dataset import assert_no_source_leakage, build_sensor_dataset, save_dataset_artifacts
from .signal_features import extract_features
from .signal_generator import CLASS_ID_TO_NAME, SignalConfig, generate_source_signals
from .signal_models import convert_to_tflite, train_classical_models, train_cnn_model

__all__ = [
    "CLASS_ID_TO_NAME",
    "SignalConfig",
    "assert_no_source_leakage",
    "build_comparison_table",
    "build_sensor_dataset",
    "convert_to_tflite",
    "extract_features",
    "generate_source_signals",
    "save_dataset_artifacts",
    "train_classical_models",
    "train_cnn_model",
    "write_day5_report",
]
