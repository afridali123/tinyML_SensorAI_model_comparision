#include "sensor_model.h"

#include <stdio.h>

int main(void) {
    float window[SENSOR_WINDOW_SIZE] = {0};
    float features[SENSOR_NUM_FEATURES];
    float probabilities[SENSOR_NUM_CLASSES];

    for (int i = 0; i < SENSOR_WINDOW_SIZE; ++i) {
        window[i] = 0.1f;
    }

    if (sensor_extract_features(window, SENSOR_WINDOW_SIZE, 1000.0f, features) != 0) {
        printf("Feature extraction failed\n");
        return 1;
    }

    int predicted_class = sensor_model_predict(features, probabilities);
    printf("Predicted class: %d (%s)\n", predicted_class, sensor_model_class_name(predicted_class));

    for (int i = 0; i < SENSOR_NUM_CLASSES; ++i) {
        printf("  %s: %.3f\n", sensor_model_class_name(i), probabilities[i]);
    }

    return 0;
}
