"""Visualization helpers for Day 5 notebooks."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .signal_features import fft_components


PLOTS_DIR = Path("reports/day5/plots")


def ensure_plots_dir() -> Path:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    return PLOTS_DIR


def plot_signal_and_fft(
    signal: np.ndarray,
    sampling_rate: int,
    title_prefix: str,
    zoom_seconds: float = 0.25,
    save_path: Path | None = None,
) -> None:
    """Plot full waveform, zoomed waveform, and FFT magnitude."""

    t = np.arange(signal.size, dtype=np.float32) / float(sampling_rate)
    freq, mag = fft_components(signal, sampling_rate)
    zoom_samples = min(signal.size, int(zoom_seconds * sampling_rate))

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    axes[0].plot(t, signal)
    axes[0].set_title(f"{title_prefix} - Full Signal")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Amplitude")

    axes[1].plot(t[:zoom_samples], signal[:zoom_samples])
    axes[1].set_title(f"{title_prefix} - Zoom ({zoom_seconds:.2f}s)")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Amplitude")

    axes[2].plot(freq, mag)
    axes[2].set_title(f"{title_prefix} - Frequency Spectrum")
    axes[2].set_xlabel("Frequency [Hz]")
    axes[2].set_ylabel("Magnitude")
    axes[2].set_xlim(0, sampling_rate / 2)

    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=140)
    plt.show()


def plot_feature_distribution(
    frame: pd.DataFrame,
    feature_name: str,
    class_column: str = "class_name",
    save_path: Path | None = None,
) -> None:
    """Plot a feature grouped by class using boxplots."""

    fig, ax = plt.subplots(figsize=(7, 4))
    frame.boxplot(column=feature_name, by=class_column, ax=ax)
    ax.set_title(f"{feature_name} by class")
    ax.set_xlabel("Class")
    ax.set_ylabel(feature_name)
    fig.suptitle("")
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=140)
    plt.show()


def plot_confusion_matrix(conf_matrix: np.ndarray, class_names: list[str], title: str) -> None:
    """Render confusion matrix heatmap."""

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(conf_matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(class_names)), class_names, rotation=35, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)

    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(j, i, str(int(conf_matrix[i, j])), ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()


def plot_comparison_bars(summary_table: pd.DataFrame, save_dir: Path | None = None) -> None:
    """Plot separate comparison charts for accuracy, F1, size, and latency."""

    targets = [
        ("accuracy", "Accuracy", "Accuracy"),
        ("f1_macro", "F1 Macro", "F1 score"),
        ("model_size_kb", "Model Size (KB)", "KB"),
        ("latency_mean_ms", "Mean Latency", "ms / inference"),
    ]

    for column, title, ylabel in targets:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.bar(summary_table["model"], summary_table[column], color=["#1f77b4", "#2ca02c", "#ff7f0e"])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Model")
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()

        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_dir / f"comparison_{column}.png", dpi=140)

        plt.show()
