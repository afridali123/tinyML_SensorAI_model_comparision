# Day 5 Sensor AI - Solutions Notebook Guide

This folder contains complete instructor solution notebooks for Day 5.

## Notebook Sequence
1. `11_1d_signal_generation.ipynb`
2. `12_1d_feature_engineering.ipynb`
3. `13_classical_ml_sensor_classification.ipynb`
4. `14_1d_cnn_sensor_classification.ipynb`
5. `15_day5_sensor_ai_assignment.ipynb`

Run in order from a clean kernel.

## What Each Notebook Delivers

### 11 - 1D Signal Generation
- Generates four synthetic machine-condition classes.
- Plots waveform and FFT for each class.
- Includes noise-level sensitivity visualization.
- Saves plots under `reports/day5/plots/`.

### 12 - Feature Engineering
- Builds leakage-safe split at source-signal level.
- Applies windowing (`256` samples, `50%` overlap by default).
- Extracts required time and frequency features.
- Saves `data/sensor/processed/sensor_features.csv` and `sensor_windows.npz`.
- Produces feature-distribution and correlation plots.

### 13 - Classical ML Classification
- Trains Logistic Regression and Random Forest baselines.
- Reports accuracy, balanced accuracy, precision, recall, F1, confusion matrices.
- Shows feature-importance analysis.
- Saves models in `models/classical/` and metrics in `reports/day5/metrics/`.

### 14 - 1D CNN Classification
- Trains compact 1D CNN with early stopping and checkpointing.
- Plots train/validation learning curves.
- Evaluates with full classification metrics and confusion matrix.
- Converts to TFLite FP32 and attempts INT8 conversion.
- Saves artifacts in `models/keras/` and `models/tflite/`.

### 15 - End-to-End Assignment Solution
- Executes full comparison pipeline.
- Builds final comparison table (accuracy, F1, size, latency).
- Generates comparison plots and final report:
  - `reports/day5/day5_model_comparison.md`
- Includes TinyML-oriented memory and deployment reasoning.

## Expected Outputs After Running All Solution Notebooks
- Dataset artifacts under `data/sensor/`
- Trained models under `models/classical`, `models/keras`, `models/tflite`
- Plots under `reports/day5/plots`
- Metrics under `reports/day5/metrics`
- Final report at `reports/day5/day5_model_comparison.md`

## Quick Execution Command (already validated)
Use environment Python from repository `.venv`, then execute notebooks 11 to 15 in order with `nbclient`.
