"""Synthetic 1D sensor signal generator for Day 5 labs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


MODE_SAMPLES_PER_CLASS = {
    "quick": 50,
    "normal": 250,
    "extended": 1000,
}

CLASS_ID_TO_NAME = {
    0: "normal_operation",
    1: "imbalance",
    2: "impulse_fault",
    3: "high_frequency_disturbance",
}


@dataclass(frozen=True)
class SignalConfig:
    """Configuration for synthetic signal generation."""

    sampling_rate: int = 1000
    duration_seconds: float = 4.0
    base_frequency_hz: float = 30.0
    amplitude: float = 1.0
    noise_level: float = 0.05
    random_seed: int = 42


@dataclass(frozen=True)
class SignalExample:
    """One generated signal with metadata."""

    source_signal_id: str
    class_id: int
    class_name: str
    sample_rate: int
    signal: np.ndarray


def time_axis(config: SignalConfig) -> np.ndarray:
    """Build a uniformly sampled time axis."""

    sample_count = int(config.sampling_rate * config.duration_seconds)
    return np.arange(sample_count, dtype=np.float32) / float(config.sampling_rate)


def _base_wave(t: np.ndarray, frequency_hz: float, amplitude: float, phase: float) -> np.ndarray:
    return amplitude * np.sin(2.0 * np.pi * frequency_hz * t + phase)


def _ringing_pulse(length: int, center: int, width: int, amplitude: float, damping: float) -> np.ndarray:
    pulse = np.zeros(length, dtype=np.float32)
    start = max(0, center - width)
    end = min(length, center + width)
    idx = np.arange(start, end)
    local_t = idx - center
    envelope = np.exp(-damping * np.abs(local_t))
    carrier = np.sin(2.0 * np.pi * 0.25 * local_t)
    pulse[start:end] = amplitude * envelope * carrier
    pulse[center:center + 1] += amplitude
    return pulse


def generate_class_signal(
    class_id: int,
    config: SignalConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate one synthetic sample for a specific machine condition class."""

    if class_id not in CLASS_ID_TO_NAME:
        raise ValueError(f"Unsupported class_id: {class_id}")

    t = time_axis(config)
    amp = config.amplitude * float(rng.uniform(0.85, 1.15))
    base_f = config.base_frequency_hz * float(rng.uniform(0.92, 1.08))
    phase = float(rng.uniform(0.0, 2.0 * np.pi))

    normal = (
        _base_wave(t, base_f, amp, phase)
        + 0.10 * _base_wave(t, 2.0 * base_f, amp, phase / 2.0)
        + rng.normal(0.0, config.noise_level, size=t.shape)
    )

    if class_id == 0:
        signal = normal
    elif class_id == 1:
        modulation = 1.0 + 0.12 * np.sin(2.0 * np.pi * 0.5 * t + phase)
        imbalance = (
            modulation * _base_wave(t, base_f, 1.35 * amp, phase)
            + 0.35 * _base_wave(t, 2.0 * base_f, amp, phase)
            + 0.20 * _base_wave(t, 3.0 * base_f, amp, phase)
        )
        signal = imbalance + rng.normal(0.0, config.noise_level * 1.2, size=t.shape)
    elif class_id == 2:
        signal = normal.copy()
        impulse_count = int(rng.integers(8, 14))
        pulse_positions = np.linspace(200, len(t) - 200, impulse_count).astype(int)
        jitter = rng.integers(-30, 31, size=impulse_count)
        severity = float(rng.uniform(0.8, 1.6))
        for center in np.clip(pulse_positions + jitter, 50, len(t) - 51):
            signal += _ringing_pulse(
                length=len(signal),
                center=int(center),
                width=20,
                amplitude=severity,
                damping=0.22,
            )
    else:
        hf = float(rng.uniform(180.0, 300.0))
        hf_amp = float(rng.uniform(0.35, 0.70))
        signal = normal + hf_amp * _base_wave(t, hf, amp, phase / 3.0)
        signal += rng.normal(0.0, config.noise_level * 1.3, size=t.shape)

    return signal.astype(np.float32)


def generate_source_signals(
    config: SignalConfig,
    mode: str = "normal",
    samples_per_class: int | None = None,
) -> list[SignalExample]:
    """Generate source signals for all four classes."""

    if samples_per_class is None:
        if mode not in MODE_SAMPLES_PER_CLASS:
            raise ValueError(f"Unsupported mode: {mode}")
        samples_per_class = MODE_SAMPLES_PER_CLASS[mode]

    rng = np.random.default_rng(config.random_seed)
    examples: list[SignalExample] = []

    for class_id, class_name in CLASS_ID_TO_NAME.items():
        for index in range(samples_per_class):
            source_signal_id = f"c{class_id:01d}_s{index:04d}"
            signal = generate_class_signal(class_id=class_id, config=config, rng=rng)
            examples.append(
                SignalExample(
                    source_signal_id=source_signal_id,
                    class_id=class_id,
                    class_name=class_name,
                    sample_rate=config.sampling_rate,
                    signal=signal,
                )
            )
    return examples
