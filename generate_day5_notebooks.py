from __future__ import annotations

from pathlib import Path
import nbformat as nbf

BASE = Path(__file__).resolve().parent
NB_SOL = BASE / "notebooks" / "solutions"
NB_STU = BASE / "notebooks" / "student"

NB_SOL.mkdir(parents=True, exist_ok=True)
NB_STU.mkdir(parents=True, exist_ok=True)


def md_cell(text: str):
    return nbf.v4.new_markdown_cell(text, metadata={"language": "markdown"})


def code_cell(text: str):
    return nbf.v4.new_code_cell(text, metadata={"language": "python"})


common_imports = """
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path.cwd().resolve()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from edge_ai_course.signal_generator import SignalConfig, CLASS_ID_TO_NAME, generate_source_signals
from edge_ai_course.signal_dataset import build_sensor_dataset, save_dataset_artifacts, assert_no_source_leakage
from edge_ai_course.signal_features import extract_features, fft_components
from edge_ai_course.signal_models import train_classical_models, train_cnn_model, convert_to_tflite
from edge_ai_course.signal_visualization import plot_signal_and_fft, plot_feature_distribution, plot_confusion_matrix, plot_comparison_bars
from edge_ai_course.model_analysis import (
    build_comparison_table,
    write_day5_report,
    estimate_classical_pipeline_memory,
    estimate_cnn_pipeline_memory,
)
""".strip()


nb11_sol = nbf.v4.new_notebook()
nb11_sol.cells = [
    md_cell(
        """# 11 - 1D Signal Generation (Solution)\n\nThis notebook generates synthetic sensor signals for four machine conditions and compares waveform and FFT views.\n\nMATLAB to Python note:\n- MATLAB: `fft(x)`\n- Python: `np.fft.rfft(x)`"""
    ),
    code_cell(common_imports),
    code_cell(
        """
config = SignalConfig(sampling_rate=1000, duration_seconds=4.0, random_seed=42)
examples = generate_source_signals(config=config, mode="quick")
print(f"Generated {len(examples)} source signals")
class_counts = pd.Series([e.class_name for e in examples]).value_counts().sort_index()
class_counts
""".strip()
    ),
    code_cell(
        """
plot_dir = Path("reports/day5/plots")
plot_dir.mkdir(parents=True, exist_ok=True)

# One representative signal per class with full/zoom/FFT views.
for class_id, class_name in CLASS_ID_TO_NAME.items():
    signal = next(e.signal for e in examples if e.class_id == class_id)
    plot_signal_and_fft(
        signal=signal,
        sampling_rate=config.sampling_rate,
        title_prefix=class_name,
        save_path=plot_dir / f"11_signal_fft_{class_name}.png",
    )
""".strip()
    ),
    code_cell(
        """
# Configurable experiment: noise and fault severity proxy via sampling different seeds.
noise_levels = [0.02, 0.06, 0.12]
fig, axes = plt.subplots(len(noise_levels), 1, figsize=(10, 6), sharex=True)
for row, noise in enumerate(noise_levels):
    cfg = SignalConfig(sampling_rate=1000, duration_seconds=4.0, noise_level=noise, random_seed=123 + row)
    sig = generate_source_signals(cfg, mode="quick", samples_per_class=1)[0].signal
    t = np.arange(sig.size) / cfg.sampling_rate
    axes[row].plot(t[:500], sig[:500])
    axes[row].set_title(f"Noise level = {noise}")
    axes[row].set_ylabel("Amplitude")
axes[-1].set_xlabel("Time [s]")
plt.tight_layout()
plt.show()
""".strip()
    ),
]

nb12_sol = nbf.v4.new_notebook()
nb12_sol.cells = [
    md_cell(
        """# 12 - Windowing and Feature Engineering (Solution)\n\nThis notebook creates leakage-safe train/validation/test splits at the source-signal level before windowing."""
    ),
    code_cell(common_imports),
    code_cell(
        """
dataset = build_sensor_dataset(
    config=SignalConfig(sampling_rate=1000, duration_seconds=4.0, random_seed=42),
    mode="quick",
    window_size=256,
    overlap_fraction=0.5,
)
assert_no_source_leakage(dataset)
paths = save_dataset_artifacts(dataset)
print(paths)

features = dataset["feature_table"]
print("Feature table shape:", features.shape)
print(features[["split", "class_name"]].value_counts().head())
features.head()
""".strip()
    ),
    code_cell(
        """
# Window duration explanation
sampling_rate = dataset["config"].sampling_rate
window_size = dataset["window_size"]
window_duration = window_size / sampling_rate
print(f"window_duration = {window_size} / {sampling_rate} = {window_duration:.3f} seconds")

# Visualize overlapping windows from one source signal.
one_signal = dataset["x_train"][0]
second_window = dataset["x_train"][1]
plt.figure(figsize=(10, 3))
plt.plot(one_signal, label="window 1")
plt.plot(second_window, label="window 2", alpha=0.7)
plt.title("Overlapping windows (example)")
plt.xlabel("Sample index")
plt.ylabel("Amplitude")
plt.legend()
plt.tight_layout()
plt.show()
""".strip()
    ),
    code_cell(
        """
plot_dir = Path("reports/day5/plots")
plot_dir.mkdir(parents=True, exist_ok=True)

for feature in ["rms", "kurtosis", "dominant_frequency", "low_band_energy", "mid_band_energy", "high_band_energy"]:
    plot_feature_distribution(features, feature, save_path=plot_dir / f"12_{feature}_by_class.png")

corr = features.select_dtypes(include=["number"]).corr()
plt.figure(figsize=(8, 6))
plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Feature Correlation Heatmap")
plt.colorbar()
plt.tight_layout()
plt.savefig(plot_dir / "12_feature_correlation.png", dpi=140)
plt.show()
""".strip()
    ),
]

nb13_sol = nbf.v4.new_notebook()
nb13_sol.cells = [
    md_cell("# 13 - Classical ML Sensor Classification (Solution)"),
    code_cell(common_imports),
    code_cell(
        """
# Load or regenerate dataset artifacts.
features_csv = Path("data/sensor/processed/sensor_features.csv")
if not features_csv.exists():
    data_bundle = build_sensor_dataset(config=SignalConfig(), mode="quick", window_size=256, overlap_fraction=0.5)
    save_dataset_artifacts(data_bundle)

features = pd.read_csv(features_csv)
feature_columns = [c for c in features.columns if c not in {"split", "source_signal_id", "window_id", "class_id", "class_name"}]

train_df = features[features["split"] == "train"]
test_df = features[features["split"] == "test"]

x_train = train_df[feature_columns].to_numpy(dtype=np.float32)
y_train = train_df["class_id"].to_numpy(dtype=np.int32)
x_test = test_df[feature_columns].to_numpy(dtype=np.float32)
y_test = test_df["class_id"].to_numpy(dtype=np.int32)

classical_results = train_classical_models(x_train, y_train, x_test, y_test)
classical_results
""".strip()
    ),
    code_cell(
        """
# Confusion matrices and model interpretability snippets.
class_names = [CLASS_ID_TO_NAME[i] for i in sorted(CLASS_ID_TO_NAME)]
for model_name, result in classical_results.items():
    print(f"\\n=== {model_name} ===")
    print(result["classification_report"])
    plot_confusion_matrix(np.array(result["confusion_matrix"]), class_names, f"{model_name} confusion matrix")

# Random Forest importance from serialized pipeline.
import joblib
rf = joblib.load(Path(classical_results["random_forest"]["model_path"]))
rf_model = rf.named_steps["model"]
importance = pd.Series(rf_model.feature_importances_, index=feature_columns).sort_values(ascending=False).head(12)
plt.figure(figsize=(8, 4))
importance.plot(kind="bar")
plt.title("Random Forest Top Feature Importances")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()
""".strip()
    ),
]

nb14_sol = nbf.v4.new_notebook()
nb14_sol.cells = [
    md_cell("# 14 - 1D CNN Sensor Classification (Solution)"),
    code_cell(common_imports),
    code_cell(
        """
bundle = build_sensor_dataset(config=SignalConfig(), mode="quick", window_size=256, overlap_fraction=0.5)
assert_no_source_leakage(bundle)
save_dataset_artifacts(bundle)

cnn_results = train_cnn_model(
    x_train=bundle["x_train"],
    y_train=bundle["y_train"],
    x_val=bundle["x_val"],
    y_val=bundle["y_val"],
    x_test=bundle["x_test"],
    y_test=bundle["y_test"],
    epochs=12,
    batch_size=64,
)
print(cnn_results["metrics"])
""".strip()
    ),
    code_cell(
        """
# Training curves
hist = cnn_results["history"]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
axes[0].plot(hist["loss"], label="train")
axes[0].plot(hist["val_loss"], label="val")
axes[0].set_title("Loss")
axes[0].set_xlabel("Epoch")
axes[0].legend()
axes[1].plot(hist["accuracy"], label="train")
axes[1].plot(hist["val_accuracy"], label="val")
axes[1].set_title("Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].legend()
plt.tight_layout()
plt.show()

plot_confusion_matrix(
    np.array(cnn_results["metrics"]["confusion_matrix"]),
    [CLASS_ID_TO_NAME[i] for i in sorted(CLASS_ID_TO_NAME)],
    "1D CNN confusion matrix",
)
""".strip()
    ),
    code_cell(
        """
# Optional TFLite conversion
x_train_norm = ((bundle["x_train"] - cnn_results["normalization"]["mean"]) / cnn_results["normalization"]["std"])[..., np.newaxis]
tflite_info = convert_to_tflite(cnn_results["model"], representative_data=x_train_norm)
tflite_info
""".strip()
    ),
]

nb15_sol = nbf.v4.new_notebook()
nb15_sol.cells = [
    md_cell("# 15 - Day 5 Assignment (Solution)"),
    code_cell(common_imports),
    code_cell(
        """
# End-to-end pipeline execution for final recommendation.
bundle = build_sensor_dataset(config=SignalConfig(), mode="quick", window_size=256, overlap_fraction=0.5)
assert_no_source_leakage(bundle)
save_dataset_artifacts(bundle)

features = bundle["feature_table"]
feature_columns = [c for c in features.columns if c not in {"split", "source_signal_id", "window_id", "class_id", "class_name"}]
train_df = features[features["split"] == "train"]
test_df = features[features["split"] == "test"]

classical_results = train_classical_models(
    train_df[feature_columns].to_numpy(dtype=np.float32),
    train_df["class_id"].to_numpy(dtype=np.int32),
    test_df[feature_columns].to_numpy(dtype=np.float32),
    test_df["class_id"].to_numpy(dtype=np.int32),
)

cnn_results = train_cnn_model(
    x_train=bundle["x_train"],
    y_train=bundle["y_train"],
    x_val=bundle["x_val"],
    y_val=bundle["y_val"],
    x_test=bundle["x_test"],
    y_test=bundle["y_test"],
    epochs=10,
    batch_size=64,
)

comparison = build_comparison_table(classical_results, cnn_results)
comparison
""".strip()
    ),
    code_cell(
        """
plot_dir = Path("reports/day5/plots")
plot_dir.mkdir(parents=True, exist_ok=True)
plot_comparison_bars(comparison, save_dir=plot_dir)

dataset_summary = {
    "mode": "quick",
    "sampling_rate_hz": bundle["config"].sampling_rate,
    "duration_seconds": bundle["config"].duration_seconds,
    "window_size": bundle["window_size"],
    "overlap_fraction": bundle["overlap_fraction"],
}
report_path = write_day5_report(dataset_summary, classical_results, cnn_results, comparison)
report_path
""".strip()
    ),
    code_cell(
        """
# TinyML reasoning support: rough RAM estimates and recommendation evidence.
best_classical_name = max(classical_results, key=lambda k: classical_results[k]["f1_macro"])
best_classical = classical_results[best_classical_name]

classical_ram = estimate_classical_pipeline_memory(
    feature_count=len(feature_columns),
    model_size_bytes=best_classical["model_size_bytes"],
)
cnn_ram = estimate_cnn_pipeline_memory(
    model_size_bytes=cnn_results["model_size_bytes"],
    parameter_count=cnn_results["parameter_count"],
)

print("Best classical model:", best_classical_name)
print("Classical RAM estimate:", classical_ram)
print("CNN RAM estimate:", cnn_ram)
print("Power note: actual energy per inference must be measured on device with voltage/current logging.")
""".strip()
    ),
]


def student_stub(title: str, objectives: str, todo_cells: list[str]):
    nb = nbf.v4.new_notebook()
    cells = [
        md_cell(f"# {title}\\n\\n{objectives}"),
        code_cell(common_imports),
    ]
    for todo in todo_cells:
        cells.append(code_cell(todo))
    nb.cells = cells
    return nb


nb11_stu = student_stub(
    "11 - 1D Signal Generation (Student)",
    "Generate and visualize all four classes. Include waveform and FFT views.",
    [
        "# TODO: Create SignalConfig and generate source signals in quick mode.\n# TODO: Print class distribution and verify expected counts.",
        "# TODO: Plot full waveform and first 0.25s zoom for one sample per class.",
        "# TODO: Compute and plot FFT for each class. Label axes clearly.",
    ],
)

nb12_stu = student_stub(
    "12 - 1D Feature Engineering (Student)",
    "Create leakage-safe windows and extract required features.",
    [
        "# TODO: Build dataset with window_size=256 and overlap=0.5.\n# TODO: Assert no source leakage.",
        "# TODO: Save dataset artifacts including sensor_features.csv and sensor_windows.npz.",
        "# TODO: Visualize RMS, kurtosis, dominant frequency, and spectral band energies by class.",
    ],
)

nb13_stu = student_stub(
    "13 - Classical ML Sensor Classification (Student)",
    "Train and evaluate Logistic Regression and Random Forest pipelines.",
    [
        "# TODO: Load feature table and split into train/test by the split column.",
        "# TODO: Train classical models and print metrics (accuracy, precision, recall, F1, confusion matrix).",
        "# TODO: Save model files and report model size + latency.",
    ],
)

nb14_stu = student_stub(
    "14 - 1D CNN Sensor Classification (Student)",
    "Train and evaluate a compact 1D CNN on raw windows.",
    [
        "# TODO: Build leakage-safe dataset and train 1D CNN with EarlyStopping and ModelCheckpoint.",
        "# TODO: Plot training/validation loss and accuracy curves.",
        "# TODO: Evaluate on test split and compare against classical baseline.",
        "# TODO: Attempt TFLite FP32 and INT8 conversion using train-only representative data.",
    ],
)

nb15_stu = student_stub(
    "15 - Day 5 Sensor AI Assignment (Student)",
    "Complete the full comparison and answer the TinyML deployment recommendation question.",
    [
        "# TODO: End-to-end pipeline execution from data generation to final comparison table.",
        "# TODO: Include environment details (Python, TensorFlow, scikit-learn, random seed, config).",
        "# TODO: Generate final recommendation based on accuracy, model size, latency, and pipeline complexity.",
    ],
)


def save_nb(nb, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, path)


save_nb(nb11_sol, NB_SOL / "11_1d_signal_generation.ipynb")
save_nb(nb12_sol, NB_SOL / "12_1d_feature_engineering.ipynb")
save_nb(nb13_sol, NB_SOL / "13_classical_ml_sensor_classification.ipynb")
save_nb(nb14_sol, NB_SOL / "14_1d_cnn_sensor_classification.ipynb")
save_nb(nb15_sol, NB_SOL / "15_day5_sensor_ai_assignment.ipynb")

save_nb(nb11_stu, NB_STU / "11_1d_signal_generation.ipynb")
save_nb(nb12_stu, NB_STU / "12_1d_feature_engineering.ipynb")
save_nb(nb13_stu, NB_STU / "13_classical_ml_sensor_classification.ipynb")
save_nb(nb14_stu, NB_STU / "14_1d_cnn_sensor_classification.ipynb")
save_nb(nb15_stu, NB_STU / "15_day5_sensor_ai_assignment.ipynb")

print("Generated Day 5 notebooks in standalone folder")
