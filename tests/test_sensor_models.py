import numpy as np

from edge_ai_course.signal_dataset import build_sensor_dataset
from edge_ai_course.signal_generator import SignalConfig
from edge_ai_course.signal_models import build_cnn_model, train_classical_models


def test_classical_models_train_and_predict() -> None:
    dataset = build_sensor_dataset(config=SignalConfig(random_seed=42), mode="quick", window_size=256, overlap_fraction=0.5)
    frame = dataset["feature_table"]
    columns = [c for c in frame.columns if c not in {"split", "source_signal_id", "window_id", "class_id", "class_name"}]
    train_df = frame[frame["split"] == "train"]
    test_df = frame[frame["split"] == "test"]

    results = train_classical_models(
        train_df[columns].to_numpy(dtype=np.float32),
        train_df["class_id"].to_numpy(dtype=np.int32),
        test_df[columns].to_numpy(dtype=np.float32),
        test_df["class_id"].to_numpy(dtype=np.int32),
    )
    assert "logistic_regression" in results
    assert "random_forest" in results


def test_cnn_output_shape() -> None:
    model = build_cnn_model(window_size=256, num_classes=4)
    out = model(np.zeros((2, 256, 1), dtype=np.float32), training=False)
    assert tuple(out.shape) == (2, 4)
