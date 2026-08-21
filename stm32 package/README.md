# STM32 Package for Day 5 Sensor Classical ML

This package exports the Day 5 classical Logistic Regression sensor model from Notebook 13 into plain C/C++ code for STM32 inference.

The target workflow is:

1. Train the classical model in `notebooks/solutions/13_classical_ml_sensor_classification.ipynb`.
2. Export the trained scikit-learn pipeline into C arrays.
3. Copy the generated C files into an STM32CubeIDE project.
4. Extract features from a 256-sample sensor window on STM32.
5. Run model inference and get one of the four class IDs.

## Why Logistic Regression?

Notebook 13 trains two classical models:

- Logistic Regression
- Random Forest

For STM32, Logistic Regression is the better first deployment target because it is small, deterministic, and easy to express as C arrays:

```text
raw sensor window -> feature extraction -> standardization -> linear logits -> softmax -> predicted class
```

Random Forest can also be exported, but it creates much larger C code because every decision tree must be represented.

## Folder Contents

```text
stm32 package/
  README.md
  export_logistic_regression_to_c.py
  include/
    sensor_model.h
  src/
    sensor_model.c
    main_example.c
  generated/
    sensor_model_data.h   # created by the exporter
```

## Step 1: Train Notebook 13

Run Notebook 13 first:

```text
notebooks/solutions/13_classical_ml_sensor_classification.ipynb
```

This creates:

```text
models/classical/sensor_logistic_regression.joblib
data/sensor/processed/sensor_features.csv
```

The `.joblib` file stores the trained scaler plus Logistic Regression model. The CSV file provides the exact feature order used during training.

## Step 2: Export Model Data to C

From the `day 5 sensor` folder, run:

```powershell
python "stm32 package/export_logistic_regression_to_c.py"
```

The script writes:

```text
stm32 package/generated/sensor_model_data.h
```

That generated header contains:

- feature names
- class names
- scaler means
- scaler scales
- Logistic Regression weights
- Logistic Regression biases

## Step 3: Add Files to STM32CubeIDE

Copy these files into your STM32 project:

```text
Core/Inc/sensor_model.h
Core/Inc/sensor_model_data.h
Core/Src/sensor_model.c
```

You can also copy `src/main_example.c` as a reference, but do not replace your STM32Cube-generated `main.c` directly unless you know what you are changing.

## Step 4: Use the Model in STM32 C Code

Basic usage:

```c
#include "sensor_model.h"

float window[SENSOR_WINDOW_SIZE];
float features[SENSOR_NUM_FEATURES];
float probabilities[SENSOR_NUM_CLASSES];

// Fill window[] with 256 sensor samples first.
sensor_extract_features(window, SENSOR_WINDOW_SIZE, 1000.0f, features);
int predicted_class = sensor_model_predict(features, probabilities);
```

Class mapping:

```text
0 = normal
1 = imbalance
2 = impulse_fault
3 = high_frequency_disturbance
```

Use `sensor_model_class_name(predicted_class)` to get the class name string.

## Step 5: Connect Real Sensor Data

For real hardware, replace the example window with samples from your sensor path, such as:

- ADC DMA buffer
- I2C accelerometer samples
- SPI accelerometer samples
- UART streamed sensor data

The model expects a fixed 256-sample window and a sampling rate close to the training rate of 1000 Hz.

## Important Student Notes

The STM32 inference code must use the same preprocessing used during Python training:

1. Same window size: `256` samples.
2. Same sampling rate assumption: `1000 Hz`.
3. Same feature order as `sensor_features.csv`.
4. Same scaler mean and scale from the trained Python pipeline.
5. Same class ID order from the notebook.

If any of these change, retrain and rerun the exporter.

## Production Note

The included feature extractor uses a simple C implementation and a direct DFT for portability. It is easy to read for students, but for production STM32 firmware you should replace the DFT section with CMSIS-DSP FFT functions for better speed.
