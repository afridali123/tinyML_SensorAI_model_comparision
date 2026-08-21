#include "sensor_model.h"

#include <float.h>
#include <math.h>
#include <stddef.h>

#define SENSOR_PI 3.14159265358979323846f
#define SENSOR_EPSILON 1.0e-12f

static void sort_copy(float *values, size_t length) {
    for (size_t i = 1; i < length; ++i) {
        float key = values[i];
        size_t j = i;
        while (j > 0 && values[j - 1] > key) {
            values[j] = values[j - 1];
            --j;
        }
        values[j] = key;
    }
}

static float safe_scale(float value) {
    return fabsf(value) < SENSOR_EPSILON ? 1.0f : value;
}

int sensor_extract_features(
    const float *window,
    size_t window_length,
    float sampling_rate_hz,
    float features[SENSOR_NUM_FEATURES]
) {
    if (window == NULL || features == NULL || window_length != SENSOR_WINDOW_SIZE || sampling_rate_hz <= 0.0f) {
        return -1;
    }

    float sum = 0.0f;
    float abs_sum = 0.0f;
    float square_sum = 0.0f;
    float maximum = -FLT_MAX;
    float minimum = FLT_MAX;
    float max_abs = 0.0f;
    float sorted[SENSOR_WINDOW_SIZE];

    for (size_t i = 0; i < window_length; ++i) {
        float value = window[i];
        float abs_value = fabsf(value);
        sum += value;
        abs_sum += abs_value;
        square_sum += value * value;
        maximum = value > maximum ? value : maximum;
        minimum = value < minimum ? value : minimum;
        max_abs = abs_value > max_abs ? abs_value : max_abs;
        sorted[i] = value;
    }

    float length_f = (float)window_length;
    float mean = sum / length_f;
    float variance = 0.0f;
    float skew_sum = 0.0f;
    float kurtosis_sum = 0.0f;

    for (size_t i = 0; i < window_length; ++i) {
        float centered = window[i] - mean;
        variance += centered * centered;
    }
    variance /= length_f;

    float std = sqrtf(variance) + SENSOR_EPSILON;
    for (size_t i = 0; i < window_length; ++i) {
        float z = (window[i] - mean) / std;
        float z2 = z * z;
        skew_sum += z2 * z;
        kurtosis_sum += z2 * z2;
    }

    sort_copy(sorted, window_length);
    float median = 0.5f * (sorted[(window_length / 2) - 1] + sorted[window_length / 2]);
    float rms = sqrtf(square_sum / length_f);

    float dominant_frequency = 0.0f;
    float dominant_magnitude = 0.0f;
    float spectral_energy = 0.0f;
    float spectral_weighted_sum = 0.0f;
    float spectral_magnitude_sum = 0.0f;
    float band_low = 0.0f;
    float band_mid = 0.0f;
    float band_high = 0.0f;
    float power_bins[(SENSOR_WINDOW_SIZE / 2) + 1];
    float total_power = 0.0f;

    for (size_t k = 0; k <= window_length / 2; ++k) {
        float real_part = 0.0f;
        float imag_part = 0.0f;
        for (size_t n = 0; n < window_length; ++n) {
            float angle = -2.0f * SENSOR_PI * (float)k * (float)n / length_f;
            real_part += window[n] * cosf(angle);
            imag_part += window[n] * sinf(angle);
        }

        float magnitude = sqrtf((real_part * real_part) + (imag_part * imag_part));
        float frequency = ((float)k * sampling_rate_hz) / length_f;
        float power = magnitude * magnitude;

        power_bins[k] = power;
        total_power += power;
        spectral_energy += power;
        spectral_weighted_sum += frequency * magnitude;
        spectral_magnitude_sum += magnitude;

        if (magnitude > dominant_magnitude) {
            dominant_magnitude = magnitude;
            dominant_frequency = frequency;
        }
        if (frequency >= 0.0f && frequency < 60.0f) {
            band_low += power;
        } else if (frequency >= 60.0f && frequency < 180.0f) {
            band_mid += power;
        } else if (frequency >= 180.0f && frequency < 500.0f) {
            band_high += power;
        }
    }

    float spectral_entropy = 0.0f;
    if (total_power > 0.0f) {
        float normalizer = log2f((float)((SENSOR_WINDOW_SIZE / 2) + 1));
        for (size_t k = 0; k <= window_length / 2; ++k) {
            float probability = power_bins[k] / total_power;
            probability = probability < SENSOR_EPSILON ? SENSOR_EPSILON : probability;
            spectral_entropy -= probability * log2f(probability);
        }
        spectral_entropy /= normalizer;
    }

    features[0] = mean;
    features[1] = std;
    features[2] = variance;
    features[3] = rms;
    features[4] = maximum;
    features[5] = minimum;
    features[6] = maximum - minimum;
    features[7] = median;
    features[8] = abs_sum / length_f;
    features[9] = skew_sum / length_f;
    features[10] = kurtosis_sum / length_f;
    features[11] = max_abs / (rms + SENSOR_EPSILON);
    features[12] = square_sum;
    features[13] = dominant_frequency;
    features[14] = dominant_magnitude;
    features[15] = spectral_magnitude_sum > 0.0f ? spectral_weighted_sum / spectral_magnitude_sum : 0.0f;
    features[16] = spectral_energy;
    features[17] = spectral_entropy;
    features[18] = band_low;
    features[19] = band_mid;
    features[20] = band_high;

    return 0;
}

int sensor_model_predict(
    const float features[SENSOR_NUM_FEATURES],
    float probabilities[SENSOR_NUM_CLASSES]
) {
    if (features == NULL || probabilities == NULL) {
        return -1;
    }

    float logits[SENSOR_NUM_CLASSES];
    float max_logit = -FLT_MAX;
    for (size_t class_index = 0; class_index < SENSOR_NUM_CLASSES; ++class_index) {
        float logit = SENSOR_MODEL_INTERCEPTS[class_index];
        for (size_t feature_index = 0; feature_index < SENSOR_NUM_FEATURES; ++feature_index) {
            float standardized = (features[feature_index] - SENSOR_SCALER_MEAN[feature_index]) /
                safe_scale(SENSOR_SCALER_SCALE[feature_index]);
            logit += SENSOR_MODEL_COEFFICIENTS[class_index][feature_index] * standardized;
        }
        logits[class_index] = logit;
        max_logit = logit > max_logit ? logit : max_logit;
    }

    float probability_sum = 0.0f;
    int best_class = 0;
    float best_probability = -1.0f;
    for (size_t class_index = 0; class_index < SENSOR_NUM_CLASSES; ++class_index) {
        probabilities[class_index] = expf(logits[class_index] - max_logit);
        probability_sum += probabilities[class_index];
    }
    probability_sum = probability_sum <= 0.0f ? 1.0f : probability_sum;

    for (size_t class_index = 0; class_index < SENSOR_NUM_CLASSES; ++class_index) {
        probabilities[class_index] /= probability_sum;
        if (probabilities[class_index] > best_probability) {
            best_probability = probabilities[class_index];
            best_class = (int)class_index;
        }
    }

    return best_class;
}

const char *sensor_model_class_name(int class_id) {
    if (class_id < 0 || class_id >= SENSOR_NUM_CLASSES) {
        return "unknown";
    }
    return SENSOR_CLASS_NAMES[class_id];
}
