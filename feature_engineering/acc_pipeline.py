"""Complete missing ACC-only artifacts without changing other signal paths.

Creates persistent E4 and RespiBAN ACC arrays, labelled windows, axis plus
vector-magnitude features, scaled classical matrices, and DL-ready sequences.
It reuses the repository's normalizer, windowers, and generic feature code.
Default subjects match the existing completed signal-wise experiment (S2--S4).
"""
from __future__ import annotations

import argparse, pickle, sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "02_normalization"), str(ROOT / "04_windowing_filtered"), str(ROOT / "05_windowing_normalised"), str(ROOT / "06_feature_engineering")]
from normalize_signals import z_score_normalize
from windowing_filtered import create_windows_filtered
from windowing_normalised import create_windows_normalized
from feature_extraction import statistical_features

WESAD = ROOT / "Data" / "Raw" / "WESAD"
PROCESSED = ROOT / "Data" / "Processed" / "ACC"
OUTPUT = ROOT / "06_feature_engineering" / "outputs" / "acc"
PLOTS = ROOT / "plots"
TARGET_FS, WINDOW_SECONDS, OVERLAP = 4, 5, 0.5
LABEL_MAP = {1: 0, 2: 1, 3: 0}
DEVICES = {"E4_ACC": ("wrist", "ACC", 32, 2.0), "RB_ACC": ("chest", "ACC", 700, 20.0)}
META = ["subject_id", "window_index", "original_label", "stress_label"]

def resample_to(values: np.ndarray, length: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return signal.resample(values, length, axis=0)

def lowpass(values: np.ndarray, fs: float, cutoff: float) -> np.ndarray:
    sos = signal.butter(4, cutoff / (fs / 2), btype="low", output="sos")
    return signal.sosfiltfilt(sos, np.asarray(values, dtype=float), axis=0)

def normalize_axes(values: np.ndarray) -> np.ndarray:
    """Use the established z-score helper independently for x/y/z axes."""
    return np.column_stack([z_score_normalize(values[:, axis]) for axis in range(values.shape[1])])

def valid_windows(values: np.ndarray, labels: np.ndarray, variant: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    builder = create_windows_normalized if variant == "filtered_normalized" else create_windows_filtered
    result = builder({"ACC": values}, {"ACC": TARGET_FS}, labels, TARGET_FS, WINDOW_SECONDS, OVERLAP)
    keep = np.isin(result["labels"], list(LABEL_MAP))
    return result["windows"]["ACC"][keep], result["labels"][keep], np.flatnonzero(keep)

def acc_feature_frame(windows: np.ndarray, labels: np.ndarray, indices: np.ndarray, subject: str, device: str) -> pd.DataFrame:
    magnitude = np.linalg.norm(windows, axis=2)
    values = {"x": windows[:, :, 0], "y": windows[:, :, 1], "z": windows[:, :, 2], "magnitude": magnitude}
    frame = pd.DataFrame({"subject_id": subject, "window_index": indices, "original_label": labels})
    frame["stress_label"] = frame.original_label.map(LABEL_MAP).astype(int)
    for axis, rows in values.items():
        extracted = pd.DataFrame([statistical_features(row) for row in rows])
        frame = pd.concat([frame, extracted.add_prefix(f"{device}_{axis}_")], axis=1)
    return frame

def save_plot(device: str, raw: np.ndarray, filtered: np.ndarray, normalized: np.ndarray, subject: str) -> None:
    folder = PLOTS / ("filtered" if device == "E4_ACC" else "filtered") / device
    folder.mkdir(parents=True, exist_ok=True)
    n = min(len(raw), TARGET_FS * 60)
    t = np.arange(n) / TARGET_FS
    for axis, label in enumerate(("X", "Y", "Z")):
        fig, ax = plt.subplots(figsize=(11, 3.5))
        ax.plot(t, raw[:n, axis], label="Raw", alpha=.75)
        ax.plot(t, filtered[:n, axis], label="Filtered", alpha=.8)
        ax.plot(t, normalized[:n, axis], label="Filtered + normalized", alpha=.8)
        ax.set(title=f"{device} {label}: raw, filtered, and normalized ({subject})", xlabel="Time (s)", ylabel="Acceleration")
        ax.grid(alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig(folder / f"{device}_{label.lower()}_raw_filtered_normalized.png", dpi=320); plt.close(fig)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", default=["S2", "S3", "S4"])
    args = parser.parse_args()
    all_features: dict[tuple[str, str], list[pd.DataFrame]] = {(d, v): [] for d in DEVICES for v in ("raw", "filtered", "filtered_normalized")}
    dl_windows: dict[tuple[str, str], list[np.ndarray]] = {(d, v): [] for d in DEVICES for v in ("raw", "filtered", "filtered_normalized")}
    dl_labels: dict[tuple[str, str], list[np.ndarray]] = {(d, v): [] for d in DEVICES for v in ("raw", "filtered", "filtered_normalized")}
    statistics = []
    for subject in args.subjects:
        path = WESAD / subject / f"{subject}.pkl"
        if not path.exists(): print(f"[skip] {path}"); continue
        print(f"Loading ACC for {subject}...", flush=True)
        with path.open("rb") as handle: record = pickle.load(handle, encoding="latin1")
        length = int(len(record["label"]) / 700 * TARGET_FS)
        labels = np.rint(resample_to(record["label"], length)).astype(int)
        for device, (source, key, source_fs, cutoff) in DEVICES.items():
            raw = resample_to(record["signal"][source][key], length)
            filtered = resample_to(lowpass(record["signal"][source][key], source_fs, cutoff), length)
            normalized = normalize_axes(filtered)
            variants = {"raw": raw, "filtered": filtered, "filtered_normalized": normalized}
            for variant, values in variants.items():
                destination = PROCESSED / variant / device; destination.mkdir(parents=True, exist_ok=True)
                windows, window_labels, window_indices = valid_windows(values, labels, variant)
                np.savez_compressed(destination / f"{subject}_{device}.npz", values=values, labels=labels, windows=windows, window_labels=window_labels, window_indices=window_indices, sampling_rate_hz=TARGET_FS, window_seconds=WINDOW_SECONDS, overlap=OVERLAP)
                manifest = pd.DataFrame([{"subject_id":subject,"device":device,"variant":variant,"aligned_shape":"x".join(map(str, values.shape)),"window_shape":"x".join(map(str, windows.shape)),"window_count":len(windows),"sampling_rate_hz":TARGET_FS,"window_seconds":WINDOW_SECONDS,"overlap":OVERLAP}])
                manifest.to_csv(destination / f"{subject}_{device}_manifest.csv", index=False)
                features = acc_feature_frame(windows, window_labels, window_indices, subject, device)
                all_features[(device, variant)].append(features)
                magnitude = np.linalg.norm(windows, axis=2, keepdims=True)
                dl_windows[(device, variant)].append(np.concatenate([windows, magnitude], axis=2))
                dl_labels[(device, variant)].append(features.stress_label.to_numpy())
                statistics.append({"subject_id":subject,"device":device,"variant":variant,"samples":len(values),"windows":len(windows),"raw_std":float(raw.std()),"variant_std":float(values.std()),"noise_reduction_pct":float(100*(1-filtered.std()/raw.std())) if raw.std() else np.nan,"range":float(np.ptp(values))})
            if subject == args.subjects[0]: save_plot(device, raw, filtered, normalized, subject)
        del record
    OUTPUT.mkdir(parents=True, exist_ok=True); PROCESSED.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(statistics).to_csv(PROCESSED / "acc_filtering_statistics.csv", index=False)
    for (device, variant), frames in all_features.items():
        if not frames: continue
        folder = OUTPUT / variant / device; folder.mkdir(parents=True, exist_ok=True)
        features = pd.concat(frames, ignore_index=True); features.to_csv(folder / "features.csv", index=False)
        columns = [c for c in features.columns if c not in META]
        features[columns].describe().T.to_csv(folder / "feature_summary.csv")
        imputed = SimpleImputer(strategy="median").fit_transform(features[columns].replace([np.inf,-np.inf],np.nan))
        scaler = StandardScaler().fit(imputed); scaled = scaler.transform(imputed)
        np.savez_compressed(folder / "model_ready_scaled_features.npz", X=scaled, y=features.stress_label.to_numpy(), subjects=features.subject_id.to_numpy(), feature_names=np.asarray(columns))
        joblib.dump(scaler, folder / "feature_scaler.joblib")
        np.savez_compressed(folder / "dl_ready_windows.npz", X=np.concatenate(dl_windows[(device,variant)]), y=np.concatenate(dl_labels[(device,variant)]), channels=np.asarray(["x","y","z","magnitude"]), sampling_rate_hz=TARGET_FS)
        fig, ax = plt.subplots(figsize=(9,4)); counts=features.stress_label.value_counts().sort_index(); ax.bar(["Non-stress","Stress"],[counts.get(0,0),counts.get(1,0)]); ax.set(title=f"{device} {variant}: ACC feature rows by class",ylabel="Windows");ax.grid(axis="y",alpha=.3);fig.tight_layout();fig.savefig(folder / "feature_overview.png",dpi=320);plt.close(fig)
    print(f"ACC artifacts written to {PROCESSED} and {OUTPUT}")

if __name__ == "__main__": main()
