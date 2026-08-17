# Day 5 Sensor AI (Standalone)

This folder contains a fully isolated Day 5 implementation for 1D Sensor AI and TinyML-oriented model comparison.

## Structure
- `notebooks/student/` and `notebooks/solutions/`: Day 5 notebooks 11-15
- `src/edge_ai_course/`: standalone signal generation, feature extraction, dataset, models, and reporting modules
- `data/sensor/`: generated source and processed datasets
- `models/`: trained classical, Keras, and TFLite artifacts
- `reports/day5/`: plots, metrics, and final markdown report
- `tests/`: unit tests for the Day 5 modules

## Recommended workflow
1. Run solution notebooks 11-15 from a clean kernel.
2. Run tests under `tests/`.
3. Review `reports/day5/day5_model_comparison.md`.
