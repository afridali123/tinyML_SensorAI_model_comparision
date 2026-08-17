"""Feature extraction for 1D sensor windows."""

from __future__ import annotations

import math

import numpy as np


DEFAULT_FREQUENCY_BANDS_HZ = {
    "low": (0.0, 60.0),
    "mid": (60.0, 180.0),
    "high": (180.0, 500.0),
}


def fft_components(signal: np.ndarray, sampling_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute positive frequency bins and magnitudes."""

    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / float(sampling_rate))
    mags = np.abs(spectrum)
    return freqs.astype(np.float32), mags.astype(np.float32)


def _safe_std(signal: np.ndarray) -> float:
    return float(np.std(signal)) + 1e-12


def _spectral_entropy(magnitudes: np.ndarray) -> float:
    power = np.square(magnitudes.astype(np.float64))
    total = float(np.sum(power))
    if total <= 0.0:
        return 0.0
    prob = power / total
    prob = np.clip(prob, 1e-12, 1.0)
    entropy = -float(np.sum(prob * np.log2(prob)))
    return entropy / math.log2(len(prob)) if len(prob) > 1 else 0.0


def _band_energy(freqs: np.ndarray, mags: np.ndarray, low_hz: float, high_hz: float) -> float:
    mask = (freqs >= low_hz) & (freqs < high_hz)
    if not np.any(mask):
        return 0.0
    return float(np.sum(np.square(mags[mask])))


def extract_features(
    signal: np.ndarray,
    sampling_rate: int,
    frequency_bands_hz: dict[str, tuple[float, float]] | None = None,
) -> dict[str, float]:
    """Extract required time and frequency domain features from one window."""

    x = signal.astype(np.float64)
    std = _safe_std(x)
    mean = float(np.mean(x))
    variance = float(np.var(x))
    rms = float(np.sqrt(np.mean(np.square(x))))
    x_max = float(np.max(x))
    x_min = float(np.min(x))
    p2p = float(x_max - x_min)
    median = float(np.median(x))
    abs_mean = float(np.mean(np.abs(x)))
    centered = x - mean
    skewness = float(np.mean(np.power(centered / std, 3)))
    kurtosis = float(np.mean(np.power(centered / std, 4)))
    crest_factor = float(np.max(np.abs(x)) / (rms + 1e-12))
    energy = float(np.sum(np.square(x)))

    freqs, mags = fft_components(signal=x.astype(np.float32), sampling_rate=sampling_rate)
    dominant_idx = int(np.argmax(mags))
    dominant_frequency = float(freqs[dominant_idx])
    dominant_magnitude = float(mags[dominant_idx])
    spectral_energy = float(np.sum(np.square(mags)))

    if spectral_energy <= 0.0:
        spectral_centroid = 0.0
    else:
        spectral_centroid = float(np.sum(freqs * mags) / (np.sum(mags) + 1e-12))

    spectral_entropy = _spectral_entropy(mags)
    bands = frequency_bands_hz or DEFAULT_FREQUENCY_BANDS_HZ
    low_band_energy = _band_energy(freqs, mags, *bands["low"])
    mid_band_energy = _band_energy(freqs, mags, *bands["mid"])
    high_band_energy = _band_energy(freqs, mags, *bands["high"])

    return {
        "mean": mean,
        "std": std,
        "variance": variance,
        "rms": rms,
        "maximum": x_max,
        "minimum": x_min,
        "peak_to_peak": p2p,
        "median": median,
        "absolute_mean": abs_mean,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "crest_factor": crest_factor,
        "signal_energy": energy,
        "dominant_frequency": dominant_frequency,
        "dominant_magnitude": dominant_magnitude,
        "spectral_centroid": spectral_centroid,
        "spectral_energy": spectral_energy,
        "spectral_entropy": spectral_entropy,
        "low_band_energy": low_band_energy,
        "mid_band_energy": mid_band_energy,
        "high_band_energy": high_band_energy,
    }
