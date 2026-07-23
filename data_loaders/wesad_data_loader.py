"""Centralized data loading for WESAD E4 (Empatica) and RespiBAN signals.

Keeping all file I/O and parsing here means notebooks only need to call
these functions and never touch raw file formats directly.
"""
from __future__ import annotations

import pickle
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