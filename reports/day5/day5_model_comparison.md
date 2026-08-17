# Day 5 Sensor AI Model Comparison

## Dataset
- Mode: quick
- Sampling rate: 1000 Hz
- Signal duration: 4.0 s
- Window size: 256 samples
- Overlap: 0.5

## Feature-Based Model
- Best classical model: **logistic_regression**
- Accuracy: 0.9177
- F1 macro: 0.9165

## 1D CNN
- Accuracy: 0.8833
- F1 macro: 0.8817
- Parameters: 3876

## Accuracy Comparison
| model | accuracy | f1_macro | model_size_kb | latency_mean_ms | latency_p95_ms |
| --- | --- | --- | --- | --- | --- |
| logistic_regression | 0.9177 | 0.9165 | 2.4434 | 0.0924 | 0.0986 |
| random_forest | 0.8708 | 0.8696 | 4899.9365 | 19.7209 | 21.3257 |
| cnn_1d | 0.8833 | 0.8817 | 82.5762 | 44.1697 | 47.8378 |

## Latency Comparison
Host-side latency reflects educational comparison only. Embedded target latency must be measured on-device.

## Model Size Comparison
Model file sizes are measured from serialized artifacts.

## TinyML Considerations
- Compare full pipeline cost: acquisition + preprocessing + model inference.
- Classical models may have tiny classifiers but can require expensive feature extraction.
- CNNs can simplify the signal path and are quantization-friendly but may increase memory needs.

## Limitations
- Synthetic dataset; domain transfer to real machinery needs validation.
- Host latency and RAM estimates are not MCU power measurements.

## Recommended Deployment Strategy
The best classical model is competitive and may be simpler for tiny devices. However, the feature extraction cost (FFT + statistics) must be included in the deployment budget.