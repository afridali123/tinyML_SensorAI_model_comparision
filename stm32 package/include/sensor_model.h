#ifndef SENSOR_MODEL_H
#define SENSOR_MODEL_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

#include "sensor_model_data.h"

int sensor_extract_features(
    const float *window,
    size_t window_length,
    float sampling_rate_hz,
    float features[SENSOR_NUM_FEATURES]
);

int sensor_model_predict(
    const float features[SENSOR_NUM_FEATURES],
    float probabilities[SENSOR_NUM_CLASSES]
);

const char *sensor_model_class_name(int class_id);

#ifdef __cplusplus
}
#endif

#endif
