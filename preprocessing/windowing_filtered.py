"""
windowing_filtered.py
---------------------
Creates 5-second overlapping windows from FILTERED Empatica (wrist) and
RespiBAN (chest) signals.

WHY A SEPARATE .py FILE?
A plain .py module can be imported from ANY notebook with:
    from windowing_filtered import create_windows_filtered
This is more robust than IPython magics (%store) because it does not
depend on both notebooks sharing the same kernel/profile, and it works
even if you restart a kernel.

TYPICAL SAMPLING RATES (edit if yours differ):
    Empatica E4 : ACC=32Hz, BVP=64Hz, EDA=4Hz, HR=1Hz, TEMP=4Hz
    RespiBAN    : ECG=EDA=EMG=RESP=TEMP=ACC=700Hz  (WESAD-style chest device)

HOW TO GET YOUR FILTERED DATA INTO THIS NOTEBOOK
--------------------------------------------------
Option A (recommended, robust) -- save/load with pickle:
    # --- in your FILTERING notebook, at the very end ---
    import pickle
    with open('filtered_data.pkl', 'wb') as f:
        pickle.dump(filtered_data, f)

    # --- in THIS (windowing) notebook ---
    import pickle
    with open('filtered_data.pkl', 'rb') as f:
        filtered_data = pickle.load(f)

Option B (quick, same-kernel-profile only) -- IPython %store magic:
    # --- in your FILTERING notebook, at the very end ---
    %store filtered_data

    # --- in THIS (windowing) notebook ---
    %store -r filtered_data

Either way, `filtered_data` should look like this (edit to match your
actual structure -- this is just the expected shape the functions below
assume):

    filtered_data = {
        'empatica': {
            'ACC': np.array([...]),   # or (N,3) if triaxial
            'BVP': np.array([...]),
            'EDA': np.array([...]),
            'HR' : np.array([...]),
            'TEMP': np.array([...]),
        },
        'respiban': {
            'ECG': np.array([...]),
            'EDA': np.array([...]),
            'EMG': np.array([...]),
            'RESP': np.array([...]),
            'TEMP': np.array([...]),
            'ACC': np.array([...]),
        },
        'labels': np.array([...]),   # optional, per-sample condition/stress label
    }
"""

import numpy as np
import pandas as pd

# ---- Edit these to match your actual devices' sampling rates ----
DEFAULT_FS = {
    'empatica': {'ACC': 32, 'BVP': 64, 'EDA': 4, 'HR': 1, 'TEMP': 4},
    'respiban': {'ECG': 700, 'EDA': 700, 'EMG': 700, 'RESP': 700, 'TEMP': 700, 'ACC': 700},
}


def _window_signal(signal, fs, window_sec=5, overlap=0.5):
    """Slide a window of `window_sec` seconds over a 1D (or (N, ch)) signal."""
    signal = np.asarray(signal)
    win_len = int(round(window_sec * fs))
    step = int(round(win_len * (1 - overlap)))
    if step <= 0:
        raise ValueError("overlap must be less than 1 (e.g. 0.5 for 50% overlap)")

    n = signal.shape[0]
    windows, start_idx = [], []
    start = 0
    while start + win_len <= n:
        windows.append(signal[start:start + win_len])
        start_idx.append(start)
        start += step

    return np.array(windows), np.array(start_idx)


def _window_labels(labels, fs, window_sec=5, overlap=0.5):
    """Assign each window the majority (mode) label found within it."""
    labels = np.asarray(labels)
    win_len = int(round(window_sec * fs))
    step = int(round(win_len * (1 - overlap)))

    n = len(labels)
    out = []
    start = 0
    while start + win_len <= n:
        w = labels[start:start + win_len]
        vals, counts = np.unique(w, return_counts=True)
        out.append(vals[np.argmax(counts)])
        start += step
    return np.array(out)


def create_windows_filtered(signals, fs_dict=None, labels=None, label_fs=None,
                             window_sec=5, overlap=0.5):
    """
    Build 5-second overlapping windows for a dict of filtered signals.

    Parameters
    ----------
    signals : dict
        {signal_name: 1D (or (N, ch)) array-like of FILTERED values}
        e.g. {'BVP': ..., 'EDA': ..., 'ECG': ...}
    fs_dict : dict, optional
        {signal_name: sampling_rate_hz}. Defaults to DEFAULT_FS values
        matched by name where possible; you should normally pass this
        explicitly so there's no ambiguity.
    labels : 1D array-like, optional
        Per-sample ground-truth labels (e.g. baseline/stress/amusement).
    label_fs : float, optional
        Sampling rate of `labels` (required if `labels` is given).
    window_sec : float
        Window length in seconds (default 5).
    overlap : float
        Fractional overlap between consecutive windows, 0-1 (default 0.5 = 50%).

    Returns
    -------
    dict with:
        'windows'          : {signal_name: np.array shape (n_windows, win_len[, ch])}
        'start_indices'    : {signal_name: np.array of window start sample indices}
        'labels'           : np.array of per-window majority labels (if labels given)
        'n_windows_per_signal' : dict, only present if signals produced differing
                                 window counts (expected, since fs differs per signal)
    """
    if fs_dict is None:
        fs_dict = {}

    result = {'windows': {}, 'start_indices': {}}
    n_windows_per_signal = {}

    for name, sig in signals.items():
        if name not in fs_dict:
            raise KeyError(
                f"No sampling rate given for signal '{name}'. "
                f"Pass fs_dict={{'{name}': <hz>, ...}}."
            )
        fs = fs_dict[name]
        w, idx = _window_signal(sig, fs, window_sec, overlap)
        result['windows'][name] = w
        result['start_indices'][name] = idx
        n_windows_per_signal[name] = len(w)

    if labels is not None:
        if label_fs is None:
            raise ValueError("label_fs must be provided when labels are given")
        result['labels'] = _window_labels(labels, label_fs, window_sec, overlap)

    if len(set(n_windows_per_signal.values())) > 1:
        # Expected: different signals have different fs -> different #windows
        result['n_windows_per_signal'] = n_windows_per_signal

    return result


def windows_to_dataframe(window_dict, signal_name):
    """
    Convenience: turn the windows for ONE signal into a tidy DataFrame,
    one row per window, one column per sample-within-window.
    """
    arr = window_dict['windows'][signal_name]
    df = pd.DataFrame(arr)
    df.columns = [f"{signal_name}_t{i}" for i in range(arr.shape[1])]
    return df