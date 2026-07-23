
"""Robust, leaked-free stress-detection training for WESAD wrist signals.

Why this replaces train_window_feature_models.py
--------------------------------------------------
The previous GRU script had three problems that explain a worse-than-baseline
(~54%) result:

1. It fed each independent scalar feature (mean, std, min, ...) into a GRU
   as if it were a timestep in a sequence. Those features have no temporal
   relationship to each other, so the recurrent layer had nothing sequential
   to learn from -- an MLP or tree model is the correct tool for tabular
   features like these.
2. It evaluated on a SINGLE random subject holdout (`rng.choice`). With only
   a handful of subjects, one unlucky split dominates the reported number.
   This script uses Leave-One-Subject-Out cross-validation across every
   subject found in Data/Raw/WESAD, so the reported metric is an average
   over N independent held-out subjects, not one lucky/unlucky draw.
3. It reported raw accuracy on an imbalanced label set (~70% non-stress).
   A model can look "good" by mostly predicting the majority class. This
   script reports balanced accuracy, macro-F1, per-class recall, and ROC-AUC,
   and always prints the majority-class dummy baseline next to real results
   so a misleadingly high accuracy cannot hide a broken model.

What this will NOT do
----------------------
It will not report 96%+ on a subject-independent split. Published WESAD
benchmarks (Schmidt et al. 2018) get ~90-93% binary stress accuracy with
LOSO cross-validation using CHEST (RespiBAN) ECG-derived features; wrist-only
signals (what this script uses) are noisier and typically land lower, often
80-90% depending on subject count and class balance. Any source claiming
96%+ on WESAD binary stress detection with a genuinely subject-independent
split should be treated with suspicion (usually subject-DEPENDENT splits,
which leak windows from the same person into train and test).

Run from the project root:
    python 07_classical_ml/train_robust_model.py
    python 07_classical_ml/train_robust_model.py --subjects S2 S3 S4 S5 S6 S7
    python 07_classical_ml/train_robust_model.py --model xgboost
"""

from __future__ import annotations

import argparse
import importlib
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import signal
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_class_weight

MODEL_CHOICES = [
    "random_forest",
    "logistic_regression",
    "xgboost",
    "svm",
    "knn",
    "decision_tree",
]

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "04_windowing_filtered"))
sys.path.insert(0, str(PROJECT_ROOT / "05_windowing_normalised"))
sys.path.insert(0, str(PROJECT_ROOT / "06_feature_engineering"))
from windowing_filtered import create_windows_filtered
from windowing_normalised import create_windows_normalized
from feature_extraction import extract_full_features

WINDOW_SEC = 5
OVERLAP = 0.5
RANDOM_SEED = 42

wesad_loader = importlib.import_module("01_data_loaders.wesad_loader")
wesad_config = importlib.import_module("configs.wesad_config")

DATASET_REGISTRY = {
    "wesad": {"loader": wesad_loader, "config": wesad_config},
}


def resample_to_target(values: np.ndarray, target_length: int) -> np.ndarray:
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


def filter_wrist_signals(wrist: dict, native_fs: dict) -> dict:
    return {
        "ACC": _sos_filter(wrist["ACC"], native_fs["ACC"], 2.0, "low"),
        "BVP": _sos_filter(wrist["BVP"], native_fs["BVP"], (0.7, 5.0), "band"),
        "EDA": _sos_filter(wrist["EDA"], native_fs["EDA"], 1.0, "low"),
        "TEMP": _sos_filter(wrist["TEMP"], native_fs["TEMP"], 0.5, "low"),
    }


def align_signals(wrist: dict, labels: np.ndarray, native_label_fs: float, target_fs: float) -> tuple[dict, np.ndarray]:
    target_length = int(len(labels) / native_label_fs * target_fs)
    source = {key: wrist[key] for key in ("ACC", "BVP", "EDA", "TEMP")}
    aligned = {key: resample_to_target(value, target_length) for key, value in source.items()}
    aligned_labels = np.rint(resample_to_target(labels, target_length)).astype(int)
    return aligned, aligned_labels


def normalize_per_subject(signals_dict: dict) -> dict:
    result = {}
    for name, values in signals_dict.items():
        values = np.asarray(values, dtype=float)
        result[name] = StandardScaler().fit_transform(values.reshape(len(values), -1)).reshape(values.shape)
    return result


def windows_to_features(window_result: dict, subject: str, config) -> pd.DataFrame:
    """One labeled feature row per 5-second window, using domain-specific
    features (EDA phasic/tonic, BVP peaks, TEMP trend) where available, not
    just generic statistics -- this is the single biggest lever for accuracy
    that the previous script did not use."""
    windows = window_result["windows"]
    labels = np.asarray(window_result["labels"])
    rows = []
    channel_map = {"ACC": ("x", "y", "z"), "BVP": ("value",), "EDA": ("value",), "TEMP": ("value",)}
    for index, label in enumerate(labels):
        if int(label) not in config.VALID_LABELS:
            continue
        row = {"subject": subject, "window_index": index, "label_original": int(label), "stress_label": config.BINARY_LABELS[int(label)]}
        for signal_name, channel_names in channel_map.items():
            data = np.asarray(windows[signal_name][index])
            if data.ndim == 1:
                data = data[:, None]
            for channel, channel_name in enumerate(channel_names):
                sig_type = config.SIGNAL_TYPE_MAP[signal_name]
                try:
                    features = extract_full_features(data[:, channel], sig_type, fs=config.TARGET_FS)
                except Exception:
                    features = extract_full_features(data[:, channel], "acc", fs=config.TARGET_FS)  # generic fallback
                row.update({f"{signal_name}_{channel_name}_{feature}": value for feature, value in features.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def build_features(subjects: list[str], variant: str, dataset: dict) -> pd.DataFrame:
    loader, config = dataset["loader"], dataset["config"]
    fs_dict = {name: config.TARGET_FS for name in config.SIGNAL_TYPE_MAP}
    frames = []
    for subject in subjects:
        print(f"Loading {subject} ({variant})...")
        try:
            subject_data = loader.load_subject(subject, config.DATA_ROOT)
        except FileNotFoundError:
            print(f"[skip] missing subject {subject} under {config.DATA_ROOT}")
            continue
        wrist, native_fs = subject_data["wrist_signals"], subject_data["native_fs"]
        raw_aligned, labels = align_signals(wrist, subject_data["labels"], config.NATIVE_LABEL_FS, config.TARGET_FS)
        if variant == "raw":
            signals_for_window = raw_aligned
            window_result = create_windows_filtered(signals_for_window, fs_dict, labels, config.TARGET_FS, WINDOW_SEC, OVERLAP)
        elif variant == "filtered":
            filtered_aligned, _ = align_signals(filter_wrist_signals(wrist, native_fs), subject_data["labels"], config.NATIVE_LABEL_FS, config.TARGET_FS)
            window_result = create_windows_filtered(filtered_aligned, fs_dict, labels, config.TARGET_FS, WINDOW_SEC, OVERLAP)
        elif variant == "filtered_normalized":
            filtered_aligned, _ = align_signals(filter_wrist_signals(wrist, native_fs), subject_data["labels"], config.NATIVE_LABEL_FS, config.TARGET_FS)
            normalized_aligned = normalize_per_subject(filtered_aligned)
            window_result = create_windows_normalized(normalized_aligned, fs_dict, labels, config.TARGET_FS, WINDOW_SEC, OVERLAP)
        else:
            raise ValueError(f"Unknown variant {variant}")
        frames.append(windows_to_features(window_result, subject, config))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def get_model(name: str):
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=400, max_depth=12, min_samples_leaf=3,
            class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1,
        )
    if name == "logistic_regression":
        return LogisticRegression(
            max_iter=2000, class_weight="balanced", C=1.0, random_state=RANDOM_SEED,
        )
    if name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise SystemExit(
                "xgboost is not installed. Run: pip install xgboost --break-system-packages\n"
                f"or pick --model from {[m for m in MODEL_CHOICES if m != 'xgboost']} instead."
            ) from exc
        return XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, eval_metric="logloss", random_state=RANDOM_SEED, n_jobs=-1,
        )
    if name == "svm":
        # probability=True is required so predict_proba exists for ROC-AUC downstream
        return SVC(
            kernel="rbf", C=1.0, gamma="scale", class_weight="balanced",
            probability=True, random_state=RANDOM_SEED,
        )
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1)
    if name == "decision_tree":
        return DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=5, class_weight="balanced", random_state=RANDOM_SEED,
        )
    raise ValueError(f"Unknown model {name}. Choose from: {MODEL_CHOICES}")


def run_loso(features: pd.DataFrame, model_name: str, variant: str, output_root: Path) -> dict:
    """Leave-one-subject-out CV: every subject is the test set exactly once,
    trained only on the remaining subjects. This is the honest way to report
    subject-independent accuracy -- no window from a test subject is ever
    seen during that subject's training fold."""
    output_dir = output_root / variant
    output_dir.mkdir(parents=True, exist_ok=True)

    meta_cols = {"subject", "window_index", "label_original", "stress_label"}
    columns = [c for c in features.columns if c not in meta_cols]
    X = features[columns].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)
    y = features["stress_label"].to_numpy(dtype=np.int32)
    groups = features["subject"].to_numpy()

    logo = LeaveOneGroupOut()
    fold_rows = []
    all_true, all_pred, all_prob = [], [], []

    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups), start=1):
        held_out = np.unique(groups[test_idx])[0]
        if len(np.unique(y[train_idx])) < 2:
            print(f"[skip fold] {held_out}: only one class in training data")
            continue

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        y_train, y_test = y[train_idx], y[test_idx]

        model = get_model(model_name)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else pred.astype(float)

        # dummy baseline on the SAME fold, so it's a fair side-by-side, not a
        # figure pulled from a different run
        dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
        dummy_acc = dummy.score(X_test, y_test)

        bal_acc = balanced_accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, zero_division=0)
        try:
            auc = roc_auc_score(y_test, prob) if len(np.unique(y_test)) > 1 else float("nan")
        except ValueError:
            auc = float("nan")

        fold_rows.append({
            "fold": fold_idx, "held_out_subject": held_out, "n_test_windows": len(test_idx),
            "test_accuracy": float(np.mean(pred == y_test)), "dummy_baseline_accuracy": dummy_acc,
            "balanced_accuracy": bal_acc, "f1_stress": f1, "roc_auc": auc,
        })
        all_true.append(y_test); all_pred.append(pred); all_prob.append(prob)
        print(f"[fold {fold_idx}] held-out={held_out}: acc={np.mean(pred==y_test):.3f} "
              f"(dummy={dummy_acc:.3f})  balanced_acc={bal_acc:.3f}  f1={f1:.3f}  auc={auc:.3f}")

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(output_dir / f"loso_fold_results_{model_name}.csv", index=False)

    y_true_all = np.concatenate(all_true)
    y_pred_all = np.concatenate(all_pred)
    report = classification_report(y_true_all, y_pred_all, target_names=["non_stress", "stress"], output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(output_dir / f"pooled_classification_report_{model_name}.csv")

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_true_all, y_pred_all), annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Non-stress", "Stress"], yticklabels=["Non-stress", "Stress"])
    ax.set(xlabel="Predicted", ylabel="True", title=f"{variant}/{model_name}: pooled LOSO confusion matrix")
    fig.tight_layout(); fig.savefig(output_dir / f"confusion_matrix_loso_{model_name}.png", dpi=180); plt.close(fig)

    summary = {
        "variant": variant, "model": model_name,
        "mean_accuracy": fold_df["test_accuracy"].mean(), "std_accuracy": fold_df["test_accuracy"].std(),
        "mean_dummy_baseline": fold_df["dummy_baseline_accuracy"].mean(),
        "mean_balanced_accuracy": fold_df["balanced_accuracy"].mean(), "std_balanced_accuracy": fold_df["balanced_accuracy"].std(),
        "mean_f1_stress": fold_df["f1_stress"].mean(), "mean_roc_auc": fold_df["roc_auc"].mean(),
        "n_subjects": fold_df["held_out_subject"].nunique(), "n_windows": len(features),
    }

    # Final model trained on ALL data, for deployment -- not what the metrics
    # above describe, so it's saved separately and clearly labeled.
    final_scaler = StandardScaler()
    X_all = final_scaler.fit_transform(X)
    final_model = get_model(model_name)
    final_model.fit(X_all, y)
    joblib.dump(final_model, output_dir / f"{model_name}_final_model_trained_on_all_subjects.joblib")
    joblib.dump(final_scaler, output_dir / "feature_scaler.joblib")
    pd.DataFrame({"feature_name": columns}).to_csv(output_dir / "feature_columns.csv", index=False)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="wesad", choices=DATASET_REGISTRY.keys())
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--variants", nargs="+", default=["raw", "filtered", "filtered_normalized"])
    parser.add_argument("--model", default="random_forest", choices=MODEL_CHOICES)
    parser.add_argument("--compare-models", action="store_true", help="Run all six model types on each variant and compare.")
    args = parser.parse_args()
    dataset = DATASET_REGISTRY[args.dataset]
    config = dataset["config"]
    output_root = PROJECT_ROOT / "07_classical_ml" / "outputs" / "robust_model" / config.OUTPUT_DIRECTORY_NAME
    subjects = args.subjects or dataset["loader"].list_subjects(config.DATA_ROOT)

    if not config.DATA_ROOT.exists() or not subjects:
        raise SystemExit(f"No {args.dataset} subjects found under {config.DATA_ROOT}. Point the dataset root at the data first.")

    output_root.mkdir(parents=True, exist_ok=True)
    print(f"Dataset: {args.dataset}; subjects: {subjects}")

    all_summaries = []
    for variant in args.variants:
        features = build_features(subjects, variant, dataset)
        if features.empty:
            print(f"[skip] no features built for variant={variant}")
            continue
        (output_root / variant).mkdir(parents=True, exist_ok=True)
        features.to_csv(output_root / variant / "features_all_subjects.csv", index=False)

        models_to_run = MODEL_CHOICES if args.compare_models else [args.model]
        for model_name in models_to_run:
            print(f"\n=== variant={variant} model={model_name} (Leave-One-Subject-Out) ===")
            summary = run_loso(features, model_name, variant, output_root)
            all_summaries.append(summary)

    summary_df = pd.DataFrame(all_summaries).sort_values("mean_balanced_accuracy", ascending=False)
    summary_df.to_csv(output_root / "model_comparison_summary.csv", index=False)
    print("\n=== Summary (sorted by mean balanced accuracy across LOSO folds) ===")
    print(summary_df.to_string(index=False))
    print(
        "\nNote: 'mean_accuracy' can look inflated on an imbalanced label set. "
        "'mean_balanced_accuracy' and 'mean_f1_stress' are the numbers to trust, "
        "and 'mean_dummy_baseline' shows what a model that just guesses the "
        "majority class would score -- your real model must clear that by a "
        "meaningful margin, not just match it."
    )


if __name__ == "__main__":
    main()
