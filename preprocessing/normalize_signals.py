"""
normalize_signals.py
---------------------
Reusable normalization functions for FILTERED Empatica and RespiBAN signals.

Two standard methods are provided:
    - z_score_normalize : (x - mean) / std        -> mean 0, std 1
    - minmax_normalize   : (x - min) / (max - min) -> scaled to [0, 1]

z-score is the more common choice in WESAD-style stress-detection
pipelines, but both are here so you can pick whichever your project uses.

Import this in your NORMALIZING notebook:
    from normalize_signals import z_score_normalize, minmax_normalize, normalize_all
"""

import numpy as np


def z_score_normalize(signal):
    """(x - mean) / std. Returns array of same shape."""
    signal = np.asarray(signal, dtype=float)
    mean = np.nanmean(signal)
    std = np.nanstd(signal)
    if std == 0:
        raise ValueError("Standard deviation is 0 -- signal is constant, cannot z-score normalize.")
    return (signal - mean) / std


def minmax_normalize(signal):
    """(x - min) / (max - min). Returns array scaled to [0, 1]."""
    signal = np.asarray(signal, dtype=float)
    mn, mx = np.nanmin(signal), np.nanmax(signal)
    if mx == mn:
        raise ValueError("Max equals min -- signal is constant, cannot min-max normalize.")
    return (signal - mn) / (mx - mn)


def normalize_all(signals_dict, method='zscore'):
    """
    Normalize every signal in a dict at once.

    Parameters
    ----------
    signals_dict : dict {signal_name: 1D array}
    method : 'zscore' or 'minmax'

    Returns
    -------
    dict {signal_name: normalized 1D array}
    """
    if method == 'zscore':
        func = z_score_normalize
    elif method == 'minmax':
        func = minmax_normalize
    else:
        raise ValueError("method must be 'zscore' or 'minmax'")

    normalized = {}
    for name, sig in signals_dict.items():
        try:
            normalized[name] = func(sig)
        except ValueError as e:
            print(f"Skipping '{name}': {e}")
    return normalized