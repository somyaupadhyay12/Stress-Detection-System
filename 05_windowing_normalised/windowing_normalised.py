"""
windowing_normalized.py
-----------------------
Creates 5-second overlapping windows from NORMALIZED (e.g. z-scored or
min-max scaled) Empatica (wrist) and RespiBAN (chest) signals.

Same windowing logic as windowing_filtered.py, kept as a SEPARATE module
(as requested) so your normalized-data pipeline stays independent from
your filtered-data pipeline -- e.g. you may normalize per-subject, per-
session, or with a scaler fit only on training data, and want to window
that result separately.

HOW TO GET YOUR NORMALIZED DATA INTO THIS NOTEBOOK
----------------------------------------------------
Option A (recommended, robust) -- save/load with pickle:
    # --- in your NORMALIZATION notebook, at the very end ---
    import pickle
    with open('normalized_data.pkl', 'wb') as f:
        pickle.dump(normalized_data, f)

    # --- in THIS (windowing) notebook ---
    import pickle
    with open('normalized_data.pkl', 'rb') as f:
        normalized_data = pickle.load(f)

Option B (quick, same-kernel-profile only) -- IPython %store magic:
    # --- in your NORMALIZATION notebook, at the very end ---
    %store normalized_data

    # --- in THIS (windowing) notebook ---
    %store -r normalized_data

Expected shape of `normalized_data` (edit to match your real structure):

    normalized_data = {
        'empatica': {
            'ACC': np.array([...]),
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
        'labels': np.array([...]),   # optional
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


def create_windows_normalized(signals, fs_dict=None, labels=None, label_fs=None,
                               window_sec=5, overlap=0.5):
    """
    Build 5-second overlapping windows for a dict of normalized signals.

    Parameters
    ----------
    signals : dict
        {signal_name: 1D (or (N, ch)) array-like of NORMALIZED values}
    fs_dict : dict, optional
        {signal_name: sampling_rate_hz}. Pass this explicitly to avoid ambiguity.
    labels : 1D array-like, optional
        Per-sample ground-truth labels.
    label_fs : float, optional
        Sampling rate of `labels` (required if `labels` is given).
    window_sec : float
        Window length in seconds (default 5).
    overlap : float
        Fractional overlap between consecutive windows, 0-1 (default 0.5 = 50%).

    Returns
    -------
    dict with:
        'windows'              : {signal_name: np.array shape (n_windows, win_len[, ch])}
        'start_indices'        : {signal_name: np.array of window start sample indices}
        'labels'               : np.array of per-window majority labels (if labels given)
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


def _window_preview_series(windows, max_windows=5):
    """Return flattened preview data from the first few windows."""
    arr = np.asarray(windows, dtype=object if isinstance(windows, list) else None)
    if len(arr) == 0:
        return []
    previews = []
    for window in list(arr[:max_windows]):
        w = np.asarray(window)
        if w.ndim == 1:
            previews.append(w)
        elif w.ndim == 2:
            previews.append(w[:, 0])
        else:
            previews.append(w.reshape(w.shape[0], -1)[:, 0])
    return previews


def plot_all_windowed_signals(window_dict, output_dir=None, prefix="windows", max_windows=5, show=False):
    """
    Plot every signal in a window result and optionally save each plot as PNG.

    Parameters
    ----------
    window_dict : dict
        Result returned by create_windows_filtered/create_windows_normalized.
    output_dir : str or Path, optional
        Folder where PNG plots are saved. If None, plots are only shown/created.
    prefix : str
        Filename prefix for saved PNGs.
    max_windows : int
        Number of windows to overlay per signal.
    show : bool
        Show plots interactively. Keep False when running from run_all.py.
    """
    import matplotlib.pyplot as plt
    from pathlib import Path

    output_path = Path(output_dir) if output_dir else None
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)

    saved = []
    for signal_name, windows in window_dict.get('windows', {}).items():
        previews = _window_preview_series(windows, max_windows=max_windows)
        if not previews:
            continue

        fig, ax = plt.subplots(figsize=(10, 4))
        for idx, series in enumerate(previews, start=1):
            ax.plot(series, linewidth=1, alpha=0.75, label=f"window {idx}")
        ax.set_title(f"{signal_name} window preview")
        ax.set_xlabel("Sample inside window")
        ax.set_ylabel(signal_name)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()

        if output_path:
            filename = f"{prefix}_{signal_name}.png"
            file_path = output_path / filename
            fig.savefig(file_path, dpi=150, bbox_inches="tight")
            saved.append(str(file_path))

        if show:
            plt.show()
        else:
            plt.close(fig)

    return saved

