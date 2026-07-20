"""Run the reproducible raw/filtered/normalized WESAD signal experiments.

This fills the gap between the existing preprocessing utilities and the
research comparison requested for the project.  It deliberately reuses the
project's windowing functions and generic ``statistical_features`` extractor.
Every experiment has the same subject-wise split, labels, window length,
overlap and classifier; only the input signal selection or preprocessing
variant changes.

Run from the repository root::

    .venv\\Scripts\\python.exe 07_classical_ml\\run_signalwise_experiments.py

Use ``--subjects S2 S3 S4`` for a short smoke run.  The default uses every
available WESAD subject and writes results to ``07_classical_ml/outputs``.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "04_windowing_filtered"))
sys.path.insert(0, str(PROJECT_ROOT / "05_windowing_normalised"))
sys.path.insert(0, str(PROJECT_ROOT / "06_feature_engineering"))
from windowing_filtered import create_windows_filtered
from windowing_normalised import create_windows_normalized
from train_classical_models import vectorized_feature_frame

WESAD_ROOT = PROJECT_ROOT / "Data" / "Raw" / "WESAD"
OUTPUT_ROOT = PROJECT_ROOT / "07_classical_ml" / "outputs" / "signalwise"
TARGET_FS = 4
WINDOW_SECONDS = 5
OVERLAP = 0.5
RANDOM_SEED = 42
LABEL_MAP = {1: 0, 2: 1, 3: 0}  # baseline/amusement -> non-stress; stress -> stress
METADATA_COLUMNS = ["subject_id", "window_index", "original_label", "stress_label"]

# Key: stable experiment name, source device/key, source rate, filter design.
# IBI is represented by an interpolated beat-to-beat interval time series so
# that it can share the project's fixed-length windowing implementation.
SIGNALS = {
    "E4_ACC": ("wrist", "ACC", 32.0, ("low", 2.0)),
    "E4_BVP": ("wrist", "BVP", 64.0, ("band", (0.7, 5.0))),
    "E4_EDA": ("wrist", "EDA", 4.0, ("low", 1.0)),
    "E4_HR": ("e4_csv", "HR", 1.0, None),
    "E4_IBI": ("e4_csv", "IBI", None, None),
    "E4_TEMP": ("wrist", "TEMP", 4.0, ("low", 0.5)),
    "RB_ACC": ("chest", "ACC", 700.0, ("low", 20.0)),
    "RB_ECG": ("chest", "ECG", 700.0, ("band", (0.5, 40.0))),
    "RB_EDA": ("chest", "EDA", 700.0, ("low", 5.0)),
    "RB_EMG": ("chest", "EMG", 700.0, ("band", (20.0, 250.0))),
    "RB_RESP": ("chest", "Resp", 700.0, ("band", (0.05, 2.0))),
    "RB_TEMP": ("chest", "Temp", 700.0, ("low", 0.5)),
}


def resample(values: np.ndarray, length: int) -> np.ndarray:
    """Resample samples along their time axis, retaining any channels."""
    return signal.resample(np.asarray(values, dtype=float), length, axis=0)


def align_source(values: np.ndarray, source_fs: float, target_length: int) -> np.ndarray:
    """Efficient anti-aliased alignment for WESAD's integer sample rates."""
    values = np.asarray(values, dtype=float)
    if source_fs in (4.0, 32.0, 64.0, 700.0):
        down = int(source_fs / TARGET_FS)
        aligned = signal.resample_poly(values, 1, down, axis=0) if down > 1 else values
        if len(aligned) == target_length:
            return aligned
    return resample(values, target_length)


def low_or_band_filter(values: np.ndarray, fs: float, design: tuple[str, object] | None) -> np.ndarray:
    """Zero-phase Butterworth filter; gracefully leave too-short signals raw."""
    if design is None:
        return np.asarray(values, dtype=float)
    kind, cutoff = design
    nyquist = fs / 2
    normalized = np.asarray(cutoff, dtype=float) / nyquist
    if np.any(normalized <= 0) or np.any(normalized >= 1):
        raise ValueError(f"Invalid cutoff {cutoff} Hz for {fs} Hz signal")
    sos = signal.butter(4, normalized, btype="band" if kind == "band" else "low", output="sos")
    values = np.asarray(values, dtype=float)
    try:
        return signal.sosfiltfilt(sos, values, axis=0)
    except ValueError:
        return values


def load_e4_csv(subject_id: str, name: str, target_length: int) -> np.ndarray:
    """Load E4 HR or convert timestamped IBI beats to a common time series."""
    folder = WESAD_ROOT / subject_id / f"{subject_id}_E4_Data"
    if name == "HR":
        rows = pd.read_csv(folder / "HR.csv", header=None).iloc[2:, 0].astype(float).to_numpy()
        return resample(rows, target_length)
    ibi = pd.read_csv(folder / "IBI.csv", header=None, names=["time", "ibi"])
    ibi["time"] = pd.to_numeric(ibi["time"], errors="coerce")
    ibi["ibi"] = pd.to_numeric(ibi["ibi"], errors="coerce")
    ibi = ibi.dropna()
    if len(ibi) < 2:
        return np.full(target_length, np.nan)
    timeline = np.linspace(float(ibi.time.iloc[0]), float(ibi.time.iloc[-1]), target_length)
    return np.interp(timeline, ibi.time.to_numpy(float), ibi.ibi.to_numpy(float))


def load_subject_signals(subject_id: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Load and align every requested source to the common 4 Hz time base."""
    path = WESAD_ROOT / subject_id / f"{subject_id}.pkl"
    with path.open("rb") as handle:
        record = pickle.load(handle, encoding="latin1")
    labels_original = np.asarray(record["label"])
    target_length = int(len(labels_original) / 700 * TARGET_FS)
    labels = np.rint(resample(labels_original, target_length)).astype(int)
    data: dict[str, np.ndarray] = {}
    for output_name, (device, key, _fs, _filter) in SIGNALS.items():
        if device == "e4_csv":
            values = load_e4_csv(subject_id, key, target_length)
        else:
            values = align_source(record["signal"][device][key], float(_fs), target_length)
        data[output_name] = np.asarray(values, dtype=float)
    return data, labels


def preprocess_variant(values: np.ndarray, signal_name: str, variant: str) -> np.ndarray:
    """Apply exactly the requested preprocessing variant, per subject."""
    _device, _key, fs, design = SIGNALS[signal_name]
    if variant == "raw":
        return values
    # Filtering occurs before common-rate alignment in the source pipeline.
    # Here the values are already 4 Hz; design filters remain valid only up to
    # 2 Hz, so source-rate filtering is performed by ``build_feature_tables``.
    if variant == "filtered":
        return values
    flat = values.reshape(len(values), -1)
    scale = flat.std(axis=0)
    scale[scale == 0] = 1.0
    return ((flat - flat.mean(axis=0)) / scale).reshape(values.shape)


def source_filtered_aligned(record: dict, subject_id: str, signal_name: str, target_length: int) -> np.ndarray:
    """Align with anti-aliasing, then apply a stable 4-Hz comparison filter."""
    device, key, fs, design = SIGNALS[signal_name]
    if device == "e4_csv":
        # HR and IBI have no defensible native-rate filter configured.
        return load_e4_csv(subject_id, key, target_length)
    values = align_source(record["signal"][device][key], float(fs), target_length)
    if design is None:
        return values
    kind, cutoff = design
    if kind == "band":
        low, high = cutoff
        design = (kind, (low, min(high, TARGET_FS / 2 - 0.05)))
        if low >= design[1][1]:
            # EMG's native 20--250 Hz band is not retained at this common
            # rate; remove the DC component instead of using an invalid band.
            return values - values.mean(axis=0)
    else:
        design = (kind, min(cutoff, TARGET_FS / 2 - 0.05))
    return low_or_band_filter(values, TARGET_FS, design)


def generic_feature_rows(windows: np.ndarray, prefix: str) -> pd.DataFrame:
    """Use the established vectorized generic extractor for every channel."""
    windows = np.asarray(windows, dtype=float)
    if windows.ndim == 2:
        windows = windows[:, :, None]
    parts = []
    channel_names = ["value"] if windows.shape[2] == 1 else ["x", "y", "z"]
    for channel, channel_name in enumerate(channel_names):
        parts.append(vectorized_feature_frame(windows[:, :, channel], f"{prefix}_{channel_name}"))
    return pd.concat(parts, axis=1)


def window_features(values: np.ndarray, labels: np.ndarray, signal_name: str, subject_id: str, variant: str) -> pd.DataFrame:
    windower = create_windows_normalized if variant == "filtered_normalized" else create_windows_filtered
    result = windower({signal_name: values}, {signal_name: TARGET_FS}, labels, TARGET_FS, WINDOW_SECONDS, OVERLAP)
    valid = np.isin(result["labels"], list(LABEL_MAP))
    feature_frame = generic_feature_rows(result["windows"][signal_name][valid], signal_name).reset_index(drop=True)
    metadata = pd.DataFrame({
        "subject_id": subject_id,
        "window_index": np.flatnonzero(valid),
        "original_label": result["labels"][valid],
    })
    metadata["stress_label"] = metadata["original_label"].map(LABEL_MAP).astype(int)
    return pd.concat([metadata, feature_frame], axis=1)


def build_feature_tables(subjects: list[str], variants: list[str]) -> tuple[dict[tuple[str, str], pd.DataFrame], pd.DataFrame]:
    frames: dict[tuple[str, str], list[pd.DataFrame]] = {(signal_name, variant): [] for signal_name in SIGNALS for variant in variants}
    stage_rows = []
    for subject_id in subjects:
        source = WESAD_ROOT / subject_id / f"{subject_id}.pkl"
        if not source.exists():
            continue
        print(f"Loading {subject_id}...", flush=True)
        started = time.perf_counter()
        with source.open("rb") as handle:
            record = pickle.load(handle, encoding="latin1")
        label_native = np.asarray(record["label"])
        target_length = int(len(label_native) / 700 * TARGET_FS)
        labels = np.rint(resample(label_native, target_length)).astype(int)
        for signal_name, (device, key, _fs, _design) in SIGNALS.items():
            raw = load_e4_csv(subject_id, key, target_length) if device == "e4_csv" else align_source(record["signal"][device][key], float(_fs), target_length)
            filtered = source_filtered_aligned(record, subject_id, signal_name, target_length)
            for variant in variants:
                values = raw if variant == "raw" else filtered
                values = preprocess_variant(values, signal_name, variant)
                table = window_features(values, labels, signal_name, subject_id, variant)
                frames[(signal_name, variant)].append(table)
                stage_rows.append({"stage": "windowing_and_features", "subject_id": subject_id, "signal": signal_name,
                                   "variant": variant, "input_samples": int(len(values)), "input_shape": "x".join(map(str, values.shape)),
                                   "windows": len(table), "features": len(table.columns) - len(METADATA_COLUMNS),
                                   "seconds": time.perf_counter() - started})
        del record
        print(f"Prepared all signals for {subject_id}")
    tables = {key: pd.concat(parts, ignore_index=True) for key, parts in frames.items() if parts}
    return tables, pd.DataFrame(stage_rows)


def make_model() -> Pipeline:
    # One fixed model/pipeline for every table.  Scaling is fit only on train.
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_SEED)),
    ])


def split_subjects(table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    groups = table.subject_id.astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_SEED)
    train, test = next(splitter.split(table, table.stress_label, groups))
    return train, test


def evaluate_table(experiment: str, variant: str, table: pd.DataFrame, output_dir: Path) -> dict:
    feature_names = [name for name in table.columns if name not in METADATA_COLUMNS]
    train_index, test_index = split_subjects(table)
    X = table[feature_names].replace([np.inf, -np.inf], np.nan)
    y = table.stress_label.astype(int)
    model = make_model()
    began = time.perf_counter(); model.fit(X.iloc[train_index], y.iloc[train_index]); train_seconds = time.perf_counter() - began
    began = time.perf_counter(); predicted = model.predict(X.iloc[test_index]); probability = model.predict_proba(X.iloc[test_index])[:, 1]; inference_seconds = time.perf_counter() - began
    y_test = y.iloc[test_index]
    target = output_dir / experiment / variant
    target.mkdir(parents=True, exist_ok=True)
    table.to_csv(target / "features.csv", index=False)
    predictions = table.iloc[test_index][METADATA_COLUMNS].copy()
    predictions["predicted_stress"] = predicted; predictions["stress_probability"] = probability
    predictions.to_csv(target / "held_out_predictions.csv", index=False)
    joblib.dump(model, target / "model.joblib")
    return {
        "experiment": experiment, "variant": variant, "model": "logistic_regression", "n_samples": len(table),
        "n_features": len(feature_names), "train_subjects": ",".join(sorted(table.iloc[train_index].subject_id.unique())),
        "test_subjects": ",".join(sorted(table.iloc[test_index].subject_id.unique())), "accuracy": accuracy_score(y_test, predicted),
        "precision": precision_score(y_test, predicted, zero_division=0), "recall": recall_score(y_test, predicted, zero_division=0),
        "f1": f1_score(y_test, predicted, zero_division=0), "roc_auc": roc_auc_score(y_test, probability),
        "training_seconds": train_seconds, "inference_seconds": inference_seconds, "inference_ms_per_window": 1000 * inference_seconds / len(test_index),
    }


def main() -> None:
    global SIGNALS
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", default=sorted(p.name for p in WESAD_ROOT.glob("S*") if (p / f"{p.name}.pkl").exists()))
    parser.add_argument("--variants", nargs="+", default=["raw", "filtered", "filtered_normalized"])
    parser.add_argument("--signals", nargs="+", default=list(SIGNALS), help="Signal IDs to include; defaults to every signal.")
    args = parser.parse_args()
    invalid = set(args.variants) - {"raw", "filtered", "filtered_normalized"}
    if invalid: raise ValueError(f"Unknown variants: {sorted(invalid)}")
    unknown_signals = set(args.signals) - set(SIGNALS)
    if unknown_signals: raise ValueError(f"Unknown signals: {sorted(unknown_signals)}")
    SIGNALS = {name: SIGNALS[name] for name in args.signals}
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    tables, stages = build_feature_tables(args.subjects, args.variants)
    stages.to_csv(OUTPUT_ROOT / "stage_record.csv", index=False)
    rows = []
    # Individual signal experiments.
    for signal_name in SIGNALS:
        for variant in args.variants:
            rows.append(evaluate_table(signal_name, variant, tables[(signal_name, variant)], OUTPUT_ROOT))
            print(f"Evaluated {signal_name} / {variant}")
    # All-signal tables are joined by subject/window/label; signals have a
    # common target rate and window schedule, so this is a true fusion table.
    for variant in args.variants:
        combined = None
        for signal_name in SIGNALS:
            table = tables[(signal_name, variant)]
            combined = table if combined is None else combined.merge(table, on=METADATA_COLUMNS, validate="one_to_one")
        rows.append(evaluate_table("ALL_SIGNALS", variant, combined, OUTPUT_ROOT))
    results = pd.DataFrame(rows).sort_values(["experiment", "variant"])
    results.to_csv(OUTPUT_ROOT / "signalwise_model_comparison.csv", index=False)
    pivot = results.pivot(index="experiment", columns="variant", values=["accuracy", "precision", "recall", "f1", "roc_auc", "training_seconds", "inference_seconds"])
    pivot.to_csv(OUTPUT_ROOT / "signalwise_comparison_table.csv")
    protocol = {"subjects": args.subjects, "variants": args.variants, "target_fs_hz": TARGET_FS, "window_seconds": WINDOW_SECONDS,
                "overlap": OVERLAP, "label_map": LABEL_MAP, "split": "GroupShuffleSplit(test_size=0.20, random_state=42)",
                "model": "median imputation -> StandardScaler -> balanced LogisticRegression(max_iter=3000, random_state=42)",
                "elapsed_seconds": time.perf_counter() - started}
    (OUTPUT_ROOT / "experiment_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(f"Completed {len(rows)} experiments. Results: {OUTPUT_ROOT / 'signalwise_model_comparison.csv'}")


if __name__ == "__main__":
    main()
