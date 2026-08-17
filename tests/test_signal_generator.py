import numpy as np

from edge_ai_course.signal_generator import CLASS_ID_TO_NAME, SignalConfig, generate_source_signals, time_axis


def test_time_axis_length() -> None:
    cfg = SignalConfig(sampling_rate=1000, duration_seconds=4.0)
    t = time_axis(cfg)
    assert len(t) == 4000


def test_generation_reproducible() -> None:
    cfg = SignalConfig(random_seed=42)
    a = generate_source_signals(cfg, mode="quick", samples_per_class=2)
    b = generate_source_signals(cfg, mode="quick", samples_per_class=2)
    assert np.allclose(a[0].signal, b[0].signal)


def test_all_classes_generated() -> None:
    cfg = SignalConfig(random_seed=1)
    examples = generate_source_signals(cfg, mode="quick", samples_per_class=1)
    class_ids = {ex.class_id for ex in examples}
    assert class_ids == set(CLASS_ID_TO_NAME)
