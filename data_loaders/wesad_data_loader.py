"""Centralized data loading for WESAD E4 (Empatica) and RespiBAN signals.

Keeping all file I/O and parsing here means notebooks only need to call
these functions and never touch raw file formats directly.
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Adjust this if your notebooks live somewhere other than one level below repo root
WESAD_ROOT = Path(__file__).resolve().parent / "Data" / "raw" / "WESAD"

# RespiBAN channel layout (WESAD raw .txt export): fixed column order
RESPIBAN_CHANNELS = {
    "ecg": 2, "eda": 3, "emg": 4, "temp": 5,
    "acc_x": 6, "acc_y": 7, "acc_z": 8, "resp": 9,
}


# ---------------------------------------------------------------------------
# Subject discovery
# ---------------------------------------------------------------------------
_SUBJECT_PATTERN = re.compile(r"^S\d+$")


def list_subjects(data_root: str | Path | None = None) -> list[str]:
    """Return sorted subject IDs (e.g. ['S2', 'S3', 'S4', ...]) found under data_root.

    A folder counts as a valid subject only if it matches 'S<number>' AND
    contains that subject's synchronized pickle (S<N>/S<N>.pkl), so partially
    downloaded or unrelated folders are skipped.
    """
    root = Path(data_root) if data_root is not None else WESAD_ROOT
    if not root.exists():
        return []

    subjects = []
    for entry in root.iterdir():
        if not entry.is_dir() or not _SUBJECT_PATTERN.match(entry.name):
            continue
        if (entry / f"{entry.name}.pkl").exists():
            subjects.append(entry.name)

    return sorted(subjects, key=lambda s: int(s[1:]))


# ---------------------------------------------------------------------------
# Full-subject loading (used by model_common.build_features)
# ---------------------------------------------------------------------------
# Native Empatica E4 wrist sampling rates (fixed by the device, not configurable)
E4_NATIVE_FS = {"ACC": 32.0, "BVP": 64.0, "EDA": 4.0, "TEMP": 4.0}


def load_subject(subject: str, data_root: str | Path | None = None) -> dict:
    """Load one subject's synchronized wrist signals + labels for model_common.build_features.

    Sourced from the WESAD pickle (already time-synchronized across
    devices), rather than the raw per-subject E4 CSVs, so labels and
    wrist signals are guaranteed to be aligned.

    Returns
    -------
    dict with:
      "wrist_signals": {"ACC": (n,3) array, "BVP": (n,) array, "EDA": (n,) array, "TEMP": (n,) array}
                        -- all at native E4 sampling rate, unfiltered, unresampled.
      "native_fs":      {"ACC": 32.0, "BVP": 64.0, "EDA": 4.0, "TEMP": 4.0}
      "labels":         raw label array at RespiBAN's native rate (700 Hz)

    Raises
    ------
    FileNotFoundError if the subject's pickle doesn't exist under data_root.
    """
    root = Path(data_root) if data_root is not None else WESAD_ROOT
    path = root / subject / f"{subject}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"No WESAD pickle found for subject {subject} at {path}")

    with path.open("rb") as handle:
        record = pickle.load(handle, encoding="latin1")

    wrist = record["signal"]["wrist"]
    wrist_signals = {
        "ACC": np.asarray(wrist["ACC"], dtype=float),
        "BVP": np.asarray(wrist["BVP"], dtype=float).reshape(-1),
        "EDA": np.asarray(wrist["EDA"], dtype=float).reshape(-1),
        "TEMP": np.asarray(wrist["TEMP"], dtype=float).reshape(-1),
    }

    return {
        "wrist_signals": wrist_signals,
        "native_fs": dict(E4_NATIVE_FS),
        "labels": np.asarray(record["label"]),
    }


# ---------------------------------------------------------------------------
# Empatica E4
# ---------------------------------------------------------------------------
def _e4_path(subject: str, filename: str) -> Path:
    return WESAD_ROOT / subject / f"{subject}_E4_Data" / filename


def load_e4_signal(subject: str, name: str) -> tuple[np.ndarray, float]:
    """Load a single-column E4 export (BVP, EDA, HR, TEMP).

    E4 CSVs store the start timestamp in row 1 and sampling rate in row 2;
    both are skipped here. Returns (values, sampling_rate_hz).
    """
    path = _e4_path(subject, f"{name}.csv")
    data = pd.read_csv(path, header=None, names=[name])
    fs = float(data[name].iloc[1])
    values = data[name].iloc[2:].astype(float).values
    return values, fs


def load_e4_acc(subject: str) -> tuple[np.ndarray, float]:
    """Load 3-axis E4 wrist ACC. Returns (values[n,3], sampling_rate_hz)."""
    path = _e4_path(subject, "ACC.csv")
    data = pd.read_csv(path, header=None)
    fs = float(data.iloc[1, 0])
    values = data.iloc[2:].astype(float).values
    return values, fs


def load_e4_ibi(subject: str) -> np.ndarray:
    """Load E4 IBI (inter-beat interval), cleaned of physiologically invalid values."""
    path = _e4_path(subject, "IBI.csv")
    df = pd.read_csv(path, sep="\t", header=None, comment="#")
    df = df[0].str.split(",", expand=True)
    df.columns = ["timestamp", "ibi"]
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["ibi"] = pd.to_numeric(df["ibi"], errors="coerce")
    df = df.dropna()
    df = df[(df["ibi"] >= 0.3) & (df["ibi"] <= 2.0)]
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df["ibi"].values


def load_e4_acc_from_pickle(subject: str) -> np.ndarray:
    """Alternative E4 ACC source: pulled from the WESAD synchronized pickle
    (record['signal']['wrist']['ACC']) rather than the raw per-subject CSV.
    Use this if S<N>_E4_Data/ACC.csv is not available for a subject.
    """
    record = load_wesad_pickle(subject)
    return record["signal"]["wrist"]["ACC"]


# ---------------------------------------------------------------------------
# RespiBAN (chest)
# ---------------------------------------------------------------------------
def load_respiban_raw(subject: str) -> pd.DataFrame:
    path = WESAD_ROOT / subject / f"{subject}_respiban.txt"
    return pd.read_csv(path, sep="\t", comment="#", header=None, engine="python")


def load_respiban_signal(subject: str, name: str) -> np.ndarray:
    """name in {'ecg','eda','emg','temp','resp','acc'}. 'acc' returns (n,3)."""
    data = load_respiban_raw(subject)
    if name == "acc":
        cols = [RESPIBAN_CHANNELS["acc_x"], RESPIBAN_CHANNELS["acc_y"], RESPIBAN_CHANNELS["acc_z"]]
        return data.iloc[:, cols].astype(float).values
    return data.iloc[:, RESPIBAN_CHANNELS[name]].astype(float).values


# ---------------------------------------------------------------------------
# Raw WESAD synchronized pickle (used for e.g. label alignment / ACC fallback)
# ---------------------------------------------------------------------------
def load_wesad_pickle(subject: str) -> dict:
    path = WESAD_ROOT / subject / f"{subject}.pkl"
    with path.open("rb") as handle:
        return pickle.load(handle, encoding="latin1")



# """Centralized data loading for WESAD E4 (Empatica) and RespiBAN signals.

# Keeping all file I/O and parsing here means notebooks only need to call
# these functions and never touch raw file formats directly.
# """
# from __future__ import annotations

# import pickle
# from pathlib import Path

# import numpy as np
# import pandas as pd

# # Adjust this if your notebooks live somewhere other than one level below repo root
# WESAD_ROOT = Path(__file__).resolve().parent / "Data" / "raw" / "WESAD"

# # RespiBAN channel layout (WESAD raw .txt export): fixed column order
# RESPIBAN_CHANNELS = {
#     "ecg": 2, "eda": 3, "emg": 4, "temp": 5,
#     "acc_x": 6, "acc_y": 7, "acc_z": 8, "resp": 9,
# }


# # ---------------------------------------------------------------------------
# # Empatica E4
# # ---------------------------------------------------------------------------
# def _e4_path(subject: str, filename: str) -> Path:
#     return WESAD_ROOT / subject / f"{subject}_E4_Data" / filename


# def load_e4_signal(subject: str, name: str) -> tuple[np.ndarray, float]:
#     """Load a single-column E4 export (BVP, EDA, HR, TEMP).

#     E4 CSVs store the start timestamp in row 1 and sampling rate in row 2;
#     both are skipped here. Returns (values, sampling_rate_hz).
#     """
#     path = _e4_path(subject, f"{name}.csv")
#     data = pd.read_csv(path, header=None, names=[name])
#     fs = float(data[name].iloc[1])
#     values = data[name].iloc[2:].astype(float).values
#     return values, fs


# def load_e4_acc(subject: str) -> tuple[np.ndarray, float]:
#     """Load 3-axis E4 wrist ACC. Returns (values[n,3], sampling_rate_hz)."""
#     path = _e4_path(subject, "ACC.csv")
#     data = pd.read_csv(path, header=None)
#     fs = float(data.iloc[1, 0])
#     values = data.iloc[2:].astype(float).values
#     return values, fs


# def load_e4_ibi(subject: str) -> np.ndarray:
#     """Load E4 IBI (inter-beat interval), cleaned of physiologically invalid values."""
#     path = _e4_path(subject, "IBI.csv")
#     df = pd.read_csv(path, sep="\t", header=None, comment="#")
#     df = df[0].str.split(",", expand=True)
#     df.columns = ["timestamp", "ibi"]
#     df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
#     df["ibi"] = pd.to_numeric(df["ibi"], errors="coerce")
#     df = df.dropna()
#     df = df[(df["ibi"] >= 0.3) & (df["ibi"] <= 2.0)]
#     df = df.sort_values("timestamp").reset_index(drop=True)
#     return df["ibi"].values


# def load_e4_acc_from_pickle(subject: str) -> np.ndarray:
#     """Alternative E4 ACC source: pulled from the WESAD synchronized pickle
#     (record['signal']['wrist']['ACC']) rather than the raw per-subject CSV.
#     Use this if S<N>_E4_Data/ACC.csv is not available for a subject.
#     """
#     record = load_wesad_pickle(subject)
#     return record["signal"]["wrist"]["ACC"]


# # ---------------------------------------------------------------------------
# # RespiBAN (chest)
# # ---------------------------------------------------------------------------
# def load_respiban_raw(subject: str) -> pd.DataFrame:
#     path = WESAD_ROOT / subject / f"{subject}_respiban.txt"
#     return pd.read_csv(path, sep="\t", comment="#", header=None, engine="python")


# def load_respiban_signal(subject: str, name: str) -> np.ndarray:
#     """name in {'ecg','eda','emg','temp','resp','acc'}. 'acc' returns (n,3)."""
#     data = load_respiban_raw(subject)
#     if name == "acc":
#         cols = [RESPIBAN_CHANNELS["acc_x"], RESPIBAN_CHANNELS["acc_y"], RESPIBAN_CHANNELS["acc_z"]]
#         return data.iloc[:, cols].astype(float).values
#     return data.iloc[:, RESPIBAN_CHANNELS[name]].astype(float).values


# # ---------------------------------------------------------------------------
# # Raw WESAD synchronized pickle (used for e.g. label alignment / ACC fallback)
# # ---------------------------------------------------------------------------
# def load_wesad_pickle(subject: str) -> dict:
#     path = WESAD_ROOT / subject / f"{subject}.pkl"
#     with path.open("rb") as handle:
#         return pickle.load(handle, encoding="latin1")