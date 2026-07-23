"""Train the reference-notebook GRU on raw, filtered, and normalized windows.

The existing preprocessing window functions are reused for every variant.
Each five-second window is converted into a labeled feature row before it is
passed to the model.  Labels follow the reference notebook: baseline/amusement
are non-stress (0), and stress is 1.

Run from the project root:
    .venv\\Scripts\\python.exe Stress-Detection-System\\models\\train_window_feature_models.py

For a quick verification run (three subjects):
    .venv\\Scripts\\python.exe Stress-Detection-System\\models\\train_window_feature_models.py --subjects S2 S3 S4
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import signal, stats
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, GRU
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "preprocessing"))
sys.path.insert(0, str(PROJECT_ROOT / "feature_engineering"))
from windowing_filtered import create_windows_filtered
from windowing_normalised import create_windows_normalized
from feature_extraction import statistical_features

WESAD_ROOT = PROJECT_ROOT / "Data" / "Raw" / "WESAD"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "06_feature_model"
BINARY_LABELS = {1: 0, 2: 1, 3: 0}
VALID_LABELS = set(BINARY_LABELS)
TARGET_FS = 4
WINDOW_SEC = 5
OVERLAP = 0.5
RANDOM_SEED = 42


def resample_to_target(values: np.ndarray, target_length: int) -> np.ndarray:
    """Match the reference notebook's common 4 Hz alignment step."""
    values = np.asarray(values, dtype=float)
    return signal.resample(values, target_length, axis=0)


def _sos_filter(values: np.ndarray, fs: float, cutoff, kind: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    nyq = fs / 2
    if kind == "band":
        sos = signal.butter(4, [cutoff[0] / nyq, cutoff[1] / nyq], btype="band", output="sos")
    else:
        sos = signal.butter(4, cutoff / nyq, btype="low", output="sos")
    return signal.sosfiltfilt(sos, values, axis=0)


def filter_wrist_signals(wrist: dict) -> dict:
    """Use the same filter intent as preprocessing/01_filtering.ipynb."""
    return {
        "ACC": _sos_filter(wrist["ACC"], 32, 2.0, "low"),
        "BVP": _sos_filter(wrist["BVP"], 64, (0.7, 5.0), "band"),
        "EDA": _sos_filter(wrist["EDA"], 4, 1.0, "low"),
        "TEMP": _sos_filter(wrist["TEMP"], 4, 0.5, "low"),
    }


def align_signals(wrist: dict, labels: np.ndarray) -> tuple[dict, np.ndarray]:
    target_length = int(len(labels) / 700 * TARGET_FS)
    source = {key: wrist[key] for key in ("ACC", "BVP", "EDA", "TEMP")}
    aligned = {key: resample_to_target(value, target_length) for key, value in source.items()}
    aligned_labels = np.rint(resample_to_target(labels, target_length)).astype(int)
    return aligned, aligned_labels


def normalize_per_subject(signals: dict) -> dict:
    """The normalized variant is filtered first, then z-scored per subject."""
    result = {}
    for name, values in signals.items():
        values = np.asarray(values, dtype=float)
        result[name] = StandardScaler().fit_transform(values.reshape(len(values), -1)).reshape(values.shape)
    return result


def windows_to_features(window_result: dict, subject: str) -> pd.DataFrame:
    """Create one complete, labeled feature row for each existing 5-second window."""
    windows = window_result["windows"]
    labels = np.asarray(window_result["labels"])
    rows = []
    channel_map = {"ACC": ("x", "y", "z"), "BVP": ("value",), "EDA": ("value",), "TEMP": ("value",)}
    for index, label in enumerate(labels):
        if int(label) not in VALID_LABELS:
            continue
        row = {"subject": subject, "window_index": index, "label_original": int(label), "stress_label": BINARY_LABELS[int(label)]}
        for signal_name, channel_names in channel_map.items():
            data = np.asarray(windows[signal_name][index])
            if data.ndim == 1:
                data = data[:, None]
            for channel, channel_name in enumerate(channel_names):
                features = statistical_features(data[:, channel])
                row.update({f"{signal_name}_{channel_name}_{feature}": value for feature, value in features.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def save_windows_before_feature_extraction(variant: str, subject: str, window_result: dict) -> None:
    """Persist the exact labeled inputs that feature extraction receives."""
    output_dir = OUTPUT_ROOT / variant / "windows_before_feature_extraction"
    output_dir.mkdir(parents=True, exist_ok=True)
    windows = window_result["windows"]
    np.savez_compressed(
        output_dir / f"{subject}_windows.npz",
        labels=np.asarray(window_result["labels"]),
        **{name: np.asarray(values) for name, values in windows.items()},
    )
    labels = pd.DataFrame({
        "subject": subject,
        "window_index": np.arange(len(window_result["labels"])),
        "label_original": window_result["labels"],
    })
    labels["stress_label"] = labels["label_original"].map(BINARY_LABELS)
    labels.to_csv(output_dir / f"{subject}_window_labels.csv", index=False)
    manifest = pd.DataFrame([
        {"subject": subject, "signal": name, "window_shape": " x ".join(map(str, np.asarray(values).shape)),
         "window_count": len(values), "samples_per_window": np.asarray(values).shape[1]}
        for name, values in windows.items()
    ])
    manifest.to_csv(output_dir / f"{subject}_window_manifest.csv", index=False)

    # A visual record of the first five windows, before any feature is made.
    fig, axes = plt.subplots(len(windows), 1, figsize=(11, 2.7 * len(windows)), squeeze=False)
    for ax, (name, values) in zip(axes.ravel(), windows.items()):
        values = np.asarray(values)
        for index in range(min(5, len(values))):
            sample = values[index]
            if sample.ndim == 2:
                sample = sample[:, 0]
            ax.plot(sample, alpha=0.75, linewidth=1, label=f"window {index}")
        ax.set(title=f"{subject} {variant}: {name} windows before feature extraction", xlabel="Sample inside 5-second window")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", ncol=5, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / f"{subject}_window_preview_before_feature_extraction.png", dpi=170)
    plt.close(fig)


def build_features(subjects: list[str]) -> dict[str, pd.DataFrame]:
    frames = {"raw": [], "filtered": [], "filtered_normalized": []}
    fs_dict = {"ACC": TARGET_FS, "BVP": TARGET_FS, "EDA": TARGET_FS, "TEMP": TARGET_FS}
    for subject in subjects:
        source = WESAD_ROOT / subject / f"{subject}.pkl"
        if not source.exists():
            print(f"[skip] missing {source}")
            continue
        print(f"Loading and using preprocessing windowing for {subject}...")
        with source.open("rb") as handle:
            record = pickle.load(handle, encoding="latin1")
        raw_aligned, labels = align_signals(record["signal"]["wrist"], record["label"])
        filtered_aligned, _ = align_signals(filter_wrist_signals(record["signal"]["wrist"]), record["label"])
        normalized_aligned = normalize_per_subject(filtered_aligned)
        variants = {
            "raw": create_windows_filtered(raw_aligned, fs_dict, labels, TARGET_FS, WINDOW_SEC, OVERLAP),
            "filtered": create_windows_filtered(filtered_aligned, fs_dict, labels, TARGET_FS, WINDOW_SEC, OVERLAP),
            "filtered_normalized": create_windows_normalized(normalized_aligned, fs_dict, labels, TARGET_FS, WINDOW_SEC, OVERLAP),
        }
        for name, window_result in variants.items():
            save_windows_before_feature_extraction(name, subject, window_result)
            frames[name].append(windows_to_features(window_result, subject))
        del record
    return {name: pd.concat(parts, ignore_index=True) for name, parts in frames.items() if parts}


def plot_feature_overview(features: pd.DataFrame, output_dir: Path, variant: str) -> None:
    numeric = features.drop(columns=["subject", "window_index", "label_original", "stress_label"])
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    sns.countplot(data=features, x="stress_label", ax=axes[0], hue="stress_label", legend=False, palette="Set2")
    axes[0].set_xticks([0, 1], ["Non-stress", "Stress"])
    axes[0].set_title(f"{variant}: feature rows by class")
    summary = numeric.groupby(features["stress_label"]).mean().T.iloc[:30]
    sns.heatmap(summary, cmap="vlag", center=0, ax=axes[1], cbar_kws={"label": "Feature mean"})
    axes[1].set_title(f"{variant}: first 30 feature means by class")
    axes[1].set_xlabel("Stress label")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_overview_before_model.png", dpi=180)
    plt.close(fig)


def train_variant(variant: str, features: pd.DataFrame) -> dict:
    output_dir = OUTPUT_ROOT / variant
    output_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_dir / "features_before_model.csv", index=False)
    plot_feature_overview(features, output_dir, variant)

    columns = [c for c in features.columns if c not in {"subject", "window_index", "label_original", "stress_label"}]
    X = features[columns].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    y = features["stress_label"].to_numpy(dtype=np.int32)
    subjects = features["subject"].to_numpy()
    unique_subjects = np.array(sorted(np.unique(subjects)))
    rng = np.random.default_rng(RANDOM_SEED)
    test_subjects = rng.choice(unique_subjects, size=max(1, len(unique_subjects) // 3), replace=False)
    train_mask = ~np.isin(subjects, test_subjects)
    if len(np.unique(y[train_mask])) < 2 or len(np.unique(y[~train_mask])) < 2:
        raise ValueError("Both classes must occur in both subject-wise splits; supply more subjects.")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_mask])
    X_test = scaler.transform(X[~train_mask])
    np.savez_compressed(
        output_dir / "model_ready_scaled_features_before_model.npz",
        X_train=X_train, X_test=X_test, y_train=y[train_mask], y_test=y[~train_mask],
        feature_names=np.asarray(columns), train_subjects=subjects[train_mask], test_subjects=subjects[~train_mask],
    )
    pd.DataFrame({"feature_name": columns}).to_csv(output_dir / "feature_columns_before_model.csv", index=False)
    # The reference notebook uses GRU. Here each extracted feature is one GRU step.
    X_train = X_train[..., None]
    X_test = X_test[..., None]
    y_train, y_test = y[train_mask], y[~train_mask]
    weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weight = dict(zip(np.unique(y_train), weights))

    tf.keras.utils.set_random_seed(RANDOM_SEED)
    model = Sequential([
        GRU(64, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.3), GRU(32), Dropout(0.3), Dense(16, activation="relu"),
        Dropout(0.2), Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=32,
                        class_weight=class_weight, callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)], verbose=2)
    history_df = pd.DataFrame(history.history)
    history_df.index += 1
    history_df.to_csv(output_dir / "training_history.csv", index_label="epoch")
    model.save(output_dir / "stress_gru.keras")
    joblib.dump(scaler, output_dir / "feature_scaler.joblib")

    probs = model.predict(X_test, verbose=0).ravel()
    pred = (probs >= 0.5).astype(int)
    # Keep the full feature rows alongside the prediction: this is the
    # auditable "after model" counterpart of features_before_model.csv.
    after = features.loc[~train_mask].copy()
    after["predicted_stress"] = pred
    after["stress_probability"] = probs
    after.to_csv(output_dir / "features_after_model.csv", index=False)
    report = classification_report(y_test, pred, target_names=["non_stress", "stress"], output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(output_dir / "classification_report.csv")
    auc = roc_auc_score(y_test, probs)
    pd.DataFrame([{"variant": variant, "test_subjects": ",".join(test_subjects), "test_accuracy": float(np.mean(pred == y_test)), "test_auc": auc,
                   "n_features": len(columns), "n_train": len(y_train), "n_test": len(y_test)}]).to_csv(output_dir / "metrics.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    axes[0].plot(history_df["accuracy"], label="train"); axes[0].plot(history_df["val_accuracy"], label="validation")
    axes[0].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy"); axes[0].legend(); axes[0].grid(alpha=.3)
    axes[1].plot(history_df["loss"], label="train"); axes[1].plot(history_df["val_loss"], label="validation")
    axes[1].set(title="Loss", xlabel="Epoch", ylabel="Binary cross-entropy"); axes[1].legend(); axes[1].grid(alpha=.3)
    fpr, tpr, _ = roc_curve(y_test, probs)
    axes[2].plot(fpr, tpr, label=f"AUC = {auc:.3f}"); axes[2].plot([0,1],[0,1], "k--")
    axes[2].set(title="ROC curve", xlabel="False-positive rate", ylabel="True-positive rate"); axes[2].legend(); axes[2].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(output_dir / "training_and_roc.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt="d", cmap="Blues", ax=ax, xticklabels=["Non-stress", "Stress"], yticklabels=["Non-stress", "Stress"])
    ax.set(xlabel="Predicted", ylabel="True", title=f"{variant}: confusion matrix")
    fig.tight_layout(); fig.savefig(output_dir / "confusion_matrix_after_model.png", dpi=180); plt.close(fig)
    return {"variant": variant, "accuracy": float(np.mean(pred == y_test)), "auc": auc, "test_subjects": ",".join(test_subjects)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", default=sorted(p.name for p in WESAD_ROOT.glob("S*") if (p / f"{p.name}.pkl").exists()))
    args = parser.parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    feature_sets = build_features(args.subjects)
    results = [train_variant(name, table) for name, table in feature_sets.items()]
    pd.DataFrame(results).to_csv(OUTPUT_ROOT / "model_comparison.csv", index=False)
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
