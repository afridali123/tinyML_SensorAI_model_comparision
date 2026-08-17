import numpy as np

from edge_ai_course.signal_features import extract_features


def test_feature_keys_and_finite_values() -> None:
    x = np.sin(np.linspace(0.0, 8.0 * np.pi, 256)).astype(np.float32)
    features = extract_features(x, sampling_rate=1000)
    expected = {
        "mean",
        "std",
        "variance",
        "rms",
        "peak_to_peak",
        "skewness",
        "kurtosis",
        "dominant_frequency",
        "spectral_entropy",
    }
    assert expected.issubset(features)
    assert all(np.isfinite(v) for v in features.values())
    assert features["rms"] >= 0.0
    assert features["peak_to_peak"] >= 0.0
