"""Configuration and project paths for the standalone Day 5 Sensor package."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
SENSOR_DATA_DIR = DATA_DIR / "sensor"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
NOTEBOOKS_DIR = ROOT / "notebooks"

DEFAULT_RANDOM_SEED = 42
DEFAULT_SAMPLE_RATE = 1000
DEFAULT_DURATION_SECONDS = 4.0
DEFAULT_WINDOW_SIZE = 256
DEFAULT_OVERLAP = 0.5
