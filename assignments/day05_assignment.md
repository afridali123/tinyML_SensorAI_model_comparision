# Day 05 Assignment - Sensor AI Model Comparison

## Objective
Build an end-to-end 1D sensor classification workflow and compare a feature-based classical model with a raw-signal 1D CNN.

## Required Tasks
1. Generate all four classes of synthetic source signals.
2. Add configurable noise and fault severity.
3. Plot waveform and FFT per class.
4. Create sliding windows with overlap.
5. Prevent source-level train/test leakage.
6. Extract time-domain and frequency-domain features.
7. Train and evaluate at least one classical classifier.
8. Train and evaluate a 1D CNN baseline.
9. Create confusion matrices and classification reports.
10. Benchmark latency and measure model sizes.
11. Compare models with accuracy/F1/size/latency plots.
12. Provide a TinyML deployment recommendation.

## Deliverables
- Dataset generation code
- Signal and FFT plots
- Feature table (`data/sensor/processed/sensor_features.csv`)
- Classical model results and saved model(s)
- 1D CNN results and saved model
- Latency/model-size comparison
- Final TinyML recommendation

## Environment Metadata
Include in your notebook output:
- Python version
- TensorFlow version
- scikit-learn version
- Random seed
- Dataset mode and split settings

## Final Research Question
Which model is more suitable for ultra-low-power deployment and why?
Use evidence from:
- accuracy and F1
- model size
- end-to-end pipeline complexity
- latency
- quantization potential
- implementation risk
