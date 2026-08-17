from edge_ai_course.signal_dataset import assert_no_source_leakage, build_sensor_dataset
from edge_ai_course.signal_generator import SignalConfig


def test_window_shape_and_leakage() -> None:
    dataset = build_sensor_dataset(config=SignalConfig(random_seed=42), mode="quick", window_size=256, overlap_fraction=0.5)
    assert dataset["x_train"].ndim == 2
    assert dataset["x_train"].shape[1] == 256
    assert_no_source_leakage(dataset)


def test_feature_table_has_expected_columns() -> None:
    dataset = build_sensor_dataset(config=SignalConfig(random_seed=42), mode="quick", window_size=256, overlap_fraction=0.5)
    frame = dataset["feature_table"]
    for column in ["source_signal_id", "window_id", "class_id", "class_name", "rms", "dominant_frequency"]:
        assert column in frame.columns
