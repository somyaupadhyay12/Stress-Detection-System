"""Create ML-ready WESAD feature tables and train grouped classical baselines.

This is the pre-IoT training path for the wrist signals that are closest to
the planned hardware: PPG/BVP, EDA/GSR, temperature and accelerometer data.
It deliberately evaluates unseen people, rather than randomly split windows.

Run from the Stress-Detection-System folder:
    python models/train_classical_models.py

The script creates one feature table per preprocessing variant (raw,
filtered, filtered_normalized) and evaluates Dummy, Logistic Regression,
Random Forest and SVM models using GroupKFold by participant.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "preprocessing"))
sys.path.insert(0, str(PROJECT_ROOT / "feature_engineering"))
from windowing_filtered import create_windows_filtered
from windowing_normalised import create_windows_normalized

WESAD_ROOT = PROJECT_ROOT / "Data" / "Raw" / "WESAD"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "07_classical_ml"
TARGET_FS = 4
WINDOW_SECONDS = 5
OVERLAP = 0.5
RANDOM_SEED = 42
LABEL_MAP = {1: 0, 2: 1, 3: 0}  # baseline/amusement => non-stress; stress => stress
METADATA_COLUMNS = ["subject_id", "window_index", "original_label", "stress_label"]


def resample(values: np.ndarray, length: int) -> np.ndarray:
    return signal.resample(np.asarray(values, dtype=float), length, axis=0)


def low_or_band_filter(values: np.ndarray, fs: float, cutoff: float | tuple[float, float], kind: str) -> np.ndarray:
    nyquist = fs / 2
    if kind == "band":
        sos = signal.butter(4, [cutoff[0] / nyquist, cutoff[1] / nyquist], btype="band", output="sos")
    else:
        sos = signal.butter(4, cutoff / nyquist, btype="low", output="sos")
    return signal.sosfiltfilt(sos, np.asarray(values, dtype=float), axis=0)


def align_wrist_signals(wrist: dict, labels: np.ndarray) -> tuple[dict, np.ndarray]:
    target_length = int(len(labels) / 700 * TARGET_FS)
    aligned = {
        name: resample(wrist[name], target_length)
        for name in ("ACC", "BVP", "EDA", "TEMP")
    }
    return aligned, np.rint(resample(labels, target_length)).astype(int)


def filtered_wrist_signals(wrist: dict) -> dict:
    return {
        "ACC": low_or_band_filter(wrist["ACC"], 32, 2.0, "low"),
        "BVP": low_or_band_filter(wrist["BVP"], 64, (0.7, 5.0), "band"),
        "EDA": low_or_band_filter(wrist["EDA"], 4, 1.0, "low"),
        "TEMP": low_or_band_filter(wrist["TEMP"], 4, 0.5, "low"),
    }


def zscore_per_subject(signals: dict) -> dict:
    result = {}
    for name, values in signals.items():
        values = np.asarray(values, dtype=float)
        flat = values.reshape(len(values), -1)
        scale = flat.std(axis=0)
        scale[scale == 0] = 1.0
        result[name] = ((flat - flat.mean(axis=0)) / scale).reshape(values.shape)
    return result


def fast_statistical_features(values: np.ndarray) -> dict[str, float]:
    """The project's generic feature set, implemented for batch-scale training.

    It matches the established feature names while avoiding thousands of
    individual SciPy ``skew``/``kurtosis`` calls, which made a full 15-subject
    run impractically slow.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return {key: np.nan for key in (
            "mean", "median", "std", "variance", "min", "max", "range", "iqr", "skewness", "kurtosis",
            "rms", "energy", "entropy", "mav", "auc", "slope", "zero_crossing_rate", "cv",
        )}
    mean = float(values.mean())
    std = float(values.std())
    variance = float(values.var())
    minimum, maximum = float(values.min()), float(values.max())
    value_range = maximum - minimum
    q25, q75 = np.percentile(values, [25, 75])
    centered = values - mean
    if std <= np.finfo(float).eps * max(1.0, abs(mean), float(np.abs(values).max())):
        skewness = kurtosis = 0.0
    else:
        standardized = centered / std
        skewness = float(np.mean(standardized ** 3))
        kurtosis = float(np.mean(standardized ** 4) - 3.0)
    if value_range <= np.finfo(float).eps * max(1.0, abs(mean), float(np.abs(values).max())):
        entropy = 0.0
    else:
        try:
            counts, _ = np.histogram(values, bins=10)
            probabilities = counts[counts > 0] / len(values)
            entropy = float(-(probabilities * np.log2(probabilities)).sum())
        except ValueError:  # Subnormal-value ranges can underflow histogram bin edges.
            entropy = 0.0
    slope = float(np.polyfit(np.arange(len(values)), values, 1)[0])
    return {
        "mean": mean, "median": float(np.median(values)), "std": std, "variance": variance,
        "min": minimum, "max": maximum, "range": value_range, "iqr": float(q75 - q25),
        "skewness": skewness, "kurtosis": kurtosis, "rms": float(np.sqrt(np.mean(values ** 2))),
        "energy": float(np.sum(values ** 2)), "entropy": entropy, "mav": float(np.mean(np.abs(values))),
        "auc": float(np.trapezoid(values)), "slope": slope,
        "zero_crossing_rate": float(np.count_nonzero(np.diff(np.signbit(values))) / (len(values) - 1)),
        "cv": float(std / abs(mean)) if abs(mean) > np.finfo(float).eps else 0.0,
    }


def vectorized_feature_frame(windows: np.ndarray, prefix: str) -> pd.DataFrame:
    """Compute the established 18 generic features for every window at once."""
    values = np.asarray(windows, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    mean, median = values.mean(axis=1), np.median(values, axis=1)
    std, variance = values.std(axis=1), values.var(axis=1)
    minimum, maximum = values.min(axis=1), values.max(axis=1)
    value_range = maximum - minimum
    q25, q75 = np.percentile(values, [25, 75], axis=1)
    centered = values - mean[:, None]
    safe_std = np.where(std > np.finfo(float).eps, std, 1.0)
    standardized = centered / safe_std[:, None]
    skewness = np.where(std > np.finfo(float).eps, np.mean(standardized ** 3, axis=1), 0.0)
    kurtosis = np.where(std > np.finfo(float).eps, np.mean(standardized ** 4, axis=1) - 3.0, 0.0)
    # Ten-bin Shannon entropy per window; only this small operation remains row-wise.
    entropy_values = []
    for row, rng in zip(values, value_range):
        if rng <= np.finfo(float).eps:
            entropy_values.append(0.0)
            continue
        try:
            counts, _ = np.histogram(row, bins=10)
            probabilities = counts[counts > 0] / len(row)
            entropy_values.append(float(-(probabilities * np.log2(probabilities)).sum()))
        except ValueError:
            entropy_values.append(0.0)
    entropy = np.asarray(entropy_values)
    sample_index = np.arange(values.shape[1])
    centered_index = sample_index - sample_index.mean()
    slope = centered.dot(centered_index) / np.sum(centered_index ** 2)
    data = {
        "mean": mean, "median": median, "std": std, "variance": variance, "min": minimum, "max": maximum,
        "range": value_range, "iqr": q75 - q25, "skewness": skewness, "kurtosis": kurtosis,
        "rms": np.sqrt(np.mean(values ** 2, axis=1)), "energy": np.sum(values ** 2, axis=1), "entropy": entropy,
        "mav": np.mean(np.abs(values), axis=1), "auc": np.trapezoid(values, axis=1), "slope": slope,
        "zero_crossing_rate": np.count_nonzero(np.diff(np.signbit(values), axis=1), axis=1) / (values.shape[1] - 1),
        "cv": np.divide(std, np.abs(mean), out=np.zeros_like(std), where=np.abs(mean) > np.finfo(float).eps),
    }
    return pd.DataFrame({f"{prefix}_{name}": column for name, column in data.items()})


def windows_to_feature_rows(window_result: dict, subject_id: str) -> pd.DataFrame:
    """Turn each labelled five-second window into a single ML feature row."""
    labels = np.asarray(window_result["labels"], dtype=int)
    valid = np.isin(labels, list(LABEL_MAP))
    result = pd.DataFrame({
        "subject_id": subject_id,
        "window_index": np.flatnonzero(valid),
        "original_label": labels[valid],
        "stress_label": [LABEL_MAP[label] for label in labels[valid]],
    })
    channel_names = {"ACC": ("x", "y", "z"), "BVP": ("value",), "EDA": ("value",), "TEMP": ("value",)}
    for signal_name, names in channel_names.items():
        values = np.asarray(window_result["windows"][signal_name])[valid]
        if values.ndim == 2:
            values = values[:, :, None]
        for channel, channel_name in enumerate(names):
            result = pd.concat([result, vectorized_feature_frame(values[:, :, channel], f"{signal_name}_{channel_name}")], axis=1)
    return result


def create_feature_tables(subjects: list[str], variants_requested: list[str]) -> dict[str, pd.DataFrame]:
    tables = {name: [] for name in variants_requested}
    sampling_rates = {"ACC": TARGET_FS, "BVP": TARGET_FS, "EDA": TARGET_FS, "TEMP": TARGET_FS}
    for subject_id in subjects:
        source = WESAD_ROOT / subject_id / f"{subject_id}.pkl"
        if not source.exists():
            print(f"[skip] {source} is missing")
            continue
        print(f"Preparing windows and features for {subject_id}...")
        with source.open("rb") as file:
            record = pickle.load(file, encoding="latin1")
        raw, labels = align_wrist_signals(record["signal"]["wrist"], record["label"])
        variants = {}
        if "raw" in tables:
            variants["raw"] = raw
        if "filtered" in tables or "filtered_normalized" in tables:
            filtered, _ = align_wrist_signals(filtered_wrist_signals(record["signal"]["wrist"]), record["label"])
            if "filtered" in tables:
                variants["filtered"] = filtered
            if "filtered_normalized" in tables:
                variants["filtered_normalized"] = zscore_per_subject(filtered)
        for name, values in variants.items():
            window_function = create_windows_normalized if name == "filtered_normalized" else create_windows_filtered
            windows = window_function(values, sampling_rates, labels, TARGET_FS, WINDOW_SECONDS, OVERLAP)
            tables[name].append(windows_to_feature_rows(windows, subject_id))
    return {name: pd.concat(parts, ignore_index=True) for name, parts in tables.items() if parts}


def feature_columns(table: pd.DataFrame) -> list[str]:
    return [column for column in table.columns if column not in METADATA_COLUMNS]


def model_definitions() -> dict[str, Pipeline]:
    # Scaling belongs in each pipeline: it is fit only on the training fold.
    return {
        "dummy": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", DummyClassifier(strategy="prior"))]),
        "logistic_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_SEED)),
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=RANDOM_SEED)),
        ]),
        "svm_rbf": Pipeline([
            ("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()),
            ("model", SVC(C=1.0, kernel="rbf", class_weight="balanced", probability=True, random_state=RANDOM_SEED)),
        ]),
    }


def evaluate_variant(name: str, table: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    output_dir = OUTPUT_ROOT / name
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "ml_ready_feature_table.csv", index=False)
    columns = feature_columns(table)
    X = table[columns].replace([np.inf, -np.inf], np.nan)
    y = table["stress_label"].astype(int)
    groups = table["subject_id"].astype(str)
    all_subjects = sorted(groups.unique())
    if len(all_subjects) < 6:
        raise ValueError("At least six subjects are required for a meaningful grouped evaluation.")

    held_out = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_SEED)
    train_index, test_index = next(held_out.split(X, y, groups))
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    groups_train = groups.iloc[train_index]
    cv = GroupKFold(n_splits=min(5, len(groups_train.unique())))
    scoring = {"f1": "f1", "precision": "precision", "recall": "recall", "accuracy": "accuracy"}
    rows, fitted = [], {}

    for model_name, pipeline in model_definitions().items():
        print(f"  Evaluating {name} / {model_name}...")
        scores = cross_validate(pipeline, X_train, y_train, groups=groups_train, cv=cv, scoring=scoring, n_jobs=1, return_train_score=False)
        pipeline.fit(X_train, y_train)
        prediction = pipeline.predict(X_test)
        rows.append({
            "variant": name, "model": model_name,
            "cv_f1_mean": scores["test_f1"].mean(), "cv_f1_std": scores["test_f1"].std(),
            "cv_precision_mean": scores["test_precision"].mean(), "cv_recall_mean": scores["test_recall"].mean(),
            "test_f1": f1_score(y_test, prediction, zero_division=0),
            "test_precision": precision_score(y_test, prediction, zero_division=0),
            "test_recall": recall_score(y_test, prediction, zero_division=0),
            "test_accuracy": (prediction == y_test).mean(),
            "n_features": len(columns), "train_subjects": ",".join(sorted(groups_train.unique())),
            "test_subjects": ",".join(sorted(groups.iloc[test_index].unique())),
        })
        fitted[model_name] = (pipeline, prediction)

    results = pd.DataFrame(rows).sort_values(["cv_f1_mean", "test_f1"], ascending=False)
    results.to_csv(output_dir / "model_comparison.csv", index=False)
    best_name = results.iloc[0]["model"]
    best_model, prediction = fitted[best_name]
    joblib.dump(best_model, output_dir / "best_stress_model.joblib")
    pd.DataFrame({"feature_name": columns}).to_csv(output_dir / "feature_order.csv", index=False)
    pd.DataFrame(classification_report(y_test, prediction, target_names=["non_stress", "stress"], output_dict=True, zero_division=0)).T.to_csv(output_dir / "best_model_classification_report.csv")
    pd.DataFrame(confusion_matrix(y_test, prediction), index=["actual_non_stress", "actual_stress"], columns=["predicted_non_stress", "predicted_stress"]).to_csv(output_dir / "best_model_confusion_matrix.csv")
    predictions = table.iloc[test_index][METADATA_COLUMNS].copy()
    predictions["predicted_stress"] = prediction
    predictions["correct"] = prediction == y_test.to_numpy()
    if hasattr(best_model, "predict_proba"):
        predictions["stress_probability"] = best_model.predict_proba(X_test)[:, 1]
    predictions.to_csv(output_dir / "held_out_predictions.csv", index=False)
    metadata = {
        "chosen_model": best_name, "feature_count": len(columns), "window_seconds": WINDOW_SECONDS,
        "overlap": OVERLAP, "target_sampling_rate_hz": TARGET_FS, "label_map": LABEL_MAP,
        "evaluation": "GroupShuffleSplit held-out subjects plus GroupKFold training cross-validation",
        "warning": "This research model is not a medical diagnostic device.",
    }
    (output_dir / "deployment_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return results, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", default=sorted(path.name for path in WESAD_ROOT.glob("S*") if (path / f"{path.name}.pkl").exists()))
    parser.add_argument("--variants", nargs="+", default=["raw", "filtered", "filtered_normalized"])
    args = parser.parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    tables = create_feature_tables(args.subjects, args.variants)
    missing = set(args.variants) - set(tables)
    if missing:
        raise ValueError(f"Unknown or unavailable variants: {sorted(missing)}")
    comparisons = [evaluate_variant(name, tables[name])[0] for name in args.variants]
    pd.concat(comparisons, ignore_index=True).sort_values(["cv_f1_mean", "test_f1"], ascending=False).to_csv(OUTPUT_ROOT / "all_model_comparisons.csv", index=False)
    print("Finished. See outputs/07_classical_ml/all_model_comparisons.csv")


if __name__ == "__main__":
    main()
