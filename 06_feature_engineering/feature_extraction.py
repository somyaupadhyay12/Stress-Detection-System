"""
feature_extraction.py
----------------------
Full feature extraction for 5-second windowed Empatica and RespiBAN
signals (works on raw, filtered, OR normalized windows -- same functions,
just point them at whichever windows_*.csv you want).

Requires: neurokit2, numpy, pandas, scipy
    pip install neurokit2 --break-system-packages   (if not already installed)

WHAT THIS COVERS
    1. GENERIC statistical + signal features, computed for EVERY signal
       (EDA, BVP, HR, IBI, TEMP, ACC, ECG, EMG, RESP, ...):
         mean, median, std, variance, min, max, range, IQR, skewness,
         kurtosis, RMS, signal energy, signal entropy, MAV, signal area
         (AUC), signal slope, zero-crossing rate, coefficient of variation
    2. EDA-SPECIFIC tonic/phasic features (SCL, SCR peak count/amplitude/
       frequency, rise time, recovery time, peak width, area under SCR peaks)
    3. HR-SPECIFIC features (mean/std/min/max/range of an HR signal)
    4. IBI/HRV-SPECIFIC features (mean RR, SDNN, RMSSD, SDSD, NN50, pNN50,
       CV of RR, median RR)
    5. BVP-SPECIFIC features (peak count, peak height stats, peak interval,
       peak frequency, signal energy, RMS)
    6. TEMPERATURE-SPECIFIC features (mean/std/min/max/range/slope/variance)

Every specific extractor below returns ONLY its extra features. Use
`extract_full_features()` to get the generic stats AND the signal-specific
features combined into one dict per window -- that's what you want for
most use cases.

HOW WINDOWED CSVs ARE STRUCTURED
    Each windows_<SIGNAL>.csv (from windowing_filtered.py / windowing_normalized.py)
    has one ROW per 5-second window, and one COLUMN per sample within that
    window (column names like SIGNAL_t0, SIGNAL_t1, ...). This module reads
    each row as a 1D array and extracts one feature-dict per window.

TYPICAL USAGE (in your feature-extraction notebook)
    import pandas as pd
    from feature_extraction import process_windowed_csv

    ecg_features = process_windowed_csv(
        'preprocessed_data/windows_filtered/windows_ECG_RB.csv',
        signal_type='ecg', fs=700
    )
    ecg_features.to_csv('preprocessed_data/features/features_ECG_RB.csv', index=False)
"""

import numpy as np
import pandas as pd
from scipy import stats

try:
    import neurokit2 as nk
    _NK_AVAILABLE = True
except ImportError:
    _NK_AVAILABLE = False


# ---------------------------------------------------------------------
# 1. GENERIC statistical + signal features -- computed for ANY signal
#    (items 1-18 of the feature spec). Use this alongside the
#    signal-specific extractors below.
# ---------------------------------------------------------------------

def statistical_features(signal):
    """
    Generic statistical + signal features that apply to any 1D window,
    regardless of signal type (EDA, BVP, HR, IBI, TEMP, ACC, ...).

    Returns a dict with: mean, median, std, variance, min, max, range,
    iqr, skewness, kurtosis, rms, energy, entropy, mav, auc, slope,
    zero_crossing_rate, cv
    """
    signal = np.asarray(signal, dtype=float)
    # Resampling and filtering can leave a rare non-finite value.  Excluding
    # all non-finite values (rather than only NaNs) keeps feature generation
    # stable for every raw/filtered/normalized window.
    signal = signal[np.isfinite(signal)]

    if len(signal) < 2:
        keys = ['mean', 'median', 'std', 'variance', 'min', 'max', 'range',
                'iqr', 'skewness', 'kurtosis', 'rms', 'energy', 'entropy',
                'mav', 'auc', 'slope', 'zero_crossing_rate', 'cv']
        return {k: np.nan for k in keys}

    mean = np.mean(signal)
    median = np.median(signal)
    std = np.std(signal)
    variance = np.var(signal)
    minimum = np.min(signal)
    maximum = np.max(signal)
    rng = maximum - minimum
    q75, q25 = np.percentile(signal, [75, 25])
    iqr = q75 - q25
    # ``np.ptp == 0`` misses near-constant, subnormal floating-point windows
    # produced by resampling.  Treat values below machine precision as flat.
    amplitude = max(1.0, float(np.max(np.abs(signal))))
    is_constant = np.ptp(signal) <= np.finfo(float).eps * amplitude
    skewness = 0.0 if is_constant else float(stats.skew(signal))
    kurtosis = 0.0 if is_constant else float(stats.kurtosis(signal))
    rms = np.sqrt(np.mean(signal ** 2))
    energy = np.sum(signal ** 2)

    # Shannon entropy computed from a 10-bin histogram of the window's values.
    if is_constant:
        entropy = 0.0
    else:
        try:
            # Counts avoid the numerical instability of density=True for
            # very narrow windows.  Convert to probabilities explicitly.
            hist, _ = np.histogram(signal, bins=10, density=False)
            probabilities = hist[hist > 0] / hist.sum()
            entropy = float(-np.sum(probabilities * np.log2(probabilities)))
        except ValueError:
            # Degenerate floating-point range: it carries no useful entropy.
            entropy = 0.0

    mav = np.mean(np.abs(signal))
    _trapz = getattr(np, 'trapezoid', None) or np.trapz  # numpy >=2.0 renamed trapz -> trapezoid
    auc = _trapz(signal)
    slope = float(np.polyfit(np.arange(len(signal)), signal, 1)[0])
    zero_crossing_rate = float(np.sum(np.diff(np.sign(signal)) != 0) / len(signal))
    cv = float(std / mean) if mean != 0 else np.nan

    return {
        'mean': float(mean), 'median': float(median), 'std': float(std),
        'variance': float(variance), 'min': float(minimum), 'max': float(maximum),
        'range': float(rng), 'iqr': float(iqr), 'skewness': skewness,
        'kurtosis': kurtosis, 'rms': float(rms), 'energy': float(energy),
        'entropy': entropy, 'mav': float(mav), 'auc': float(auc),
        'slope': slope, 'zero_crossing_rate': zero_crossing_rate, 'cv': cv,
    }


# ---------------------------------------------------------------------
# Per-signal-type feature extractors. Each takes a 1D array (one window)
# and returns a dict of {feature_name: value}.
# ---------------------------------------------------------------------

def extract_hr_features(signal):
    """
    HR-specific features (items 29-33): mean, std, min, max, range of
    an HR signal (e.g. Empatica's per-second HR window).
    """
    signal = np.asarray(signal, dtype=float)
    signal = signal[~np.isnan(signal)]
    if len(signal) == 0:
        return {'mean_hr': np.nan, 'std_hr': np.nan, 'min_hr': np.nan,
                'max_hr': np.nan, 'range_hr': np.nan}
    return {
        'mean_hr': float(np.mean(signal)),
        'std_hr': float(np.std(signal)),
        'min_hr': float(np.min(signal)),
        'max_hr': float(np.max(signal)),
        'range_hr': float(np.max(signal) - np.min(signal)),
    }


def extract_ecg_features(signal, fs):
    """HRV time-domain features + heart rate from an ECG window, via neurokit2."""
    if not _NK_AVAILABLE:
        raise ImportError("neurokit2 is required for ECG features. Run: pip install neurokit2 --break-system-packages")
    signal = np.asarray(signal, dtype=float)
    try:
        signals, info = nk.ecg_process(signal, sampling_rate=fs)
        hrv = nk.hrv_time(info, sampling_rate=fs, show=False)
        return {
            'mean_hr': float(signals['ECG_Rate'].mean()),
            'sdnn': float(hrv['HRV_SDNN'].iloc[0]) if 'HRV_SDNN' in hrv else np.nan,
            'rmssd': float(hrv['HRV_RMSSD'].iloc[0]) if 'HRV_RMSSD' in hrv else np.nan,
            'pnn50': float(hrv['HRV_pNN50'].iloc[0]) if 'HRV_pNN50' in hrv else np.nan,
            'n_rpeaks': len(info['ECG_R_Peaks']),
        }
    except Exception:
        # window too short / no clean R-peaks found -- return NaNs rather than crash the whole batch
        return {'mean_hr': np.nan, 'sdnn': np.nan, 'rmssd': np.nan, 'pnn50': np.nan, 'n_rpeaks': 0}


def extract_ibi_features(ibi_values):
    """
    IBI/HRV-specific time-domain features (items 35-42), computed directly
    from a window's IBI/RR values (in seconds). Use this for the Empatica
    IBI signal, which is already beat-to-beat intervals rather than a raw
    ECG waveform.

    Returns: mean_rr, median_rr, sdnn, rmssd, sdsd, nn50, pnn50, cv_rr, n_beats
    """
    ibi_values = np.asarray(ibi_values, dtype=float)
    ibi_values = ibi_values[~np.isnan(ibi_values)]
    keys = ['mean_rr', 'median_rr', 'sdnn', 'rmssd', 'sdsd', 'nn50', 'pnn50', 'cv_rr']
    if len(ibi_values) < 2:
        out = {k: np.nan for k in keys}
        out['n_beats'] = len(ibi_values)
        return out

    diffs_ms = np.diff(ibi_values) * 1000  # successive differences, in ms
    rmssd = np.sqrt(np.mean(diffs_ms ** 2))
    sdsd = np.std(diffs_ms)
    nn50 = int(np.sum(np.abs(diffs_ms) > 50))
    pnn50 = float(nn50 / len(diffs_ms) * 100)
    mean_rr = np.mean(ibi_values)
    sdnn = np.std(ibi_values) * 1000  # ms
    cv_rr = float(np.std(ibi_values) / mean_rr) if mean_rr != 0 else np.nan

    return {
        'mean_rr': float(mean_rr),
        'median_rr': float(np.median(ibi_values)),
        'sdnn': float(sdnn),
        'rmssd': float(rmssd),
        'sdsd': float(sdsd),
        'nn50': nn50,
        'pnn50': pnn50,
        'cv_rr': cv_rr,
        'n_beats': len(ibi_values),
    }


def extract_eda_features(signal, fs):
    """
    EDA-specific tonic + phasic features (items 19-28), via neurokit2.

    Tonic:  scl_mean, scl_std
    Phasic: scr_count, scr_mean_amplitude, scr_max_amplitude,
            scr_peak_frequency (peaks per second), scr_mean_rise_time,
            scr_mean_recovery_time, scr_mean_peak_width, scr_area_under_peaks
    """
    if not _NK_AVAILABLE:
        raise ImportError("neurokit2 is required for EDA features. Run: pip install neurokit2 --break-system-packages")
    signal = np.asarray(signal, dtype=float)
    window_sec = len(signal) / fs

    empty_result = {
        'scl_mean': np.nan, 'scl_std': np.nan, 'scr_count': 0,
        'scr_mean_amplitude': np.nan, 'scr_max_amplitude': np.nan,
        'scr_peak_frequency': 0.0, 'scr_mean_rise_time': np.nan,
        'scr_mean_recovery_time': np.nan, 'scr_mean_peak_width': np.nan,
        'scr_area_under_peaks': np.nan,
    }

    try:
        signals, info = nk.eda_process(signal, sampling_rate=fs)
        scr_peaks = np.asarray(info['SCR_Peaks'])
        scr_onsets = np.asarray(info['SCR_Onsets'])
        scr_recovery = np.asarray(info['SCR_Recovery'])
        amplitudes = np.asarray(info['SCR_Amplitude'])
        rise_times = np.asarray(info['SCR_RiseTime'])
        recovery_times = np.asarray(info['SCR_RecoveryTime'])

        n_peaks = len(scr_peaks)
        if n_peaks == 0:
            result = dict(empty_result)
            result['scl_mean'] = float(signals['EDA_Tonic'].mean())
            result['scl_std'] = float(signals['EDA_Tonic'].std())
            return result

        # peak width: onset-to-recovery duration, in seconds (NaN-safe if recovery not detected)
        valid_recovery = ~np.isnan(scr_recovery)
        if valid_recovery.any():
            widths = (scr_recovery[valid_recovery] - scr_onsets[valid_recovery]) / fs
            mean_width = float(np.mean(widths))
        else:
            mean_width = np.nan

        # area under each SCR peak: trapezoidal integration of the phasic
        # component between onset and recovery (falls back to onset-to-peak
        # if no recovery point was detected for that peak)
        phasic = signals['EDA_Phasic'].values
        _trapz = getattr(np, 'trapezoid', None) or np.trapz
        areas = []
        for i in range(n_peaks):
            start = int(scr_onsets[i])
            end = int(scr_recovery[i]) if valid_recovery[i] else int(scr_peaks[i])
            if end > start:
                areas.append(_trapz(phasic[start:end + 1]))
        area_under_peaks = float(np.mean(areas)) if len(areas) > 0 else np.nan

        return {
            'scl_mean': float(signals['EDA_Tonic'].mean()),
            'scl_std': float(signals['EDA_Tonic'].std()),
            'scr_count': n_peaks,
            'scr_mean_amplitude': float(np.nanmean(amplitudes)),
            'scr_max_amplitude': float(np.nanmax(amplitudes)),
            'scr_peak_frequency': float(n_peaks / window_sec),
            'scr_mean_rise_time': float(np.nanmean(rise_times)),
            'scr_mean_recovery_time': float(np.nanmean(recovery_times)),
            'scr_mean_peak_width': mean_width,
            'scr_area_under_peaks': area_under_peaks,
        }
    except Exception:
        return empty_result


def extract_bvp_features(signal, fs):
    """
    BVP-specific features (items 43-50), via neurokit2's PPG peak detection.

    Returns: peak_count, mean_peak_height, std_peak_height, max_peak_height,
    mean_peak_interval, peak_frequency, signal_energy, rms
    """
    if not _NK_AVAILABLE:
        raise ImportError("neurokit2 is required for BVP features. Run: pip install neurokit2 --break-system-packages")
    signal = np.asarray(signal, dtype=float)
    window_sec = len(signal) / fs

    energy = float(np.sum(signal ** 2))
    rms = float(np.sqrt(np.mean(signal ** 2)))

    try:
        signals, info = nk.ppg_process(signal, sampling_rate=fs)
        peaks = np.asarray(info['PPG_Peaks'])
        n_peaks = len(peaks)

        if n_peaks == 0:
            return {'peak_count': 0, 'mean_peak_height': np.nan, 'std_peak_height': np.nan,
                    'max_peak_height': np.nan, 'mean_peak_interval': np.nan,
                    'peak_frequency': 0.0, 'signal_energy': energy, 'rms': rms}

        peak_heights = signals['PPG_Clean'].values[peaks]
        mean_peak_interval = float(np.mean(np.diff(peaks)) / fs) if n_peaks > 1 else np.nan

        return {
            'peak_count': n_peaks,
            'mean_peak_height': float(np.mean(peak_heights)),
            'std_peak_height': float(np.std(peak_heights)),
            'max_peak_height': float(np.max(peak_heights)),
            'mean_peak_interval': mean_peak_interval,
            'peak_frequency': float(n_peaks / window_sec),
            'signal_energy': energy,
            'rms': rms,
        }
    except Exception:
        return {'peak_count': 0, 'mean_peak_height': np.nan, 'std_peak_height': np.nan,
                'max_peak_height': np.nan, 'mean_peak_interval': np.nan,
                'peak_frequency': np.nan, 'signal_energy': energy, 'rms': rms}


def extract_emg_features(signal):
    """Mean absolute value, RMS, variance, zero-crossing rate for an EMG window."""
    signal = np.asarray(signal, dtype=float)
    mav = np.mean(np.abs(signal))
    rms = np.sqrt(np.mean(signal ** 2))
    var = np.var(signal)
    zero_crossings = np.sum(np.diff(np.sign(signal)) != 0)
    return {
        'mav': float(mav),
        'rms': float(rms),
        'variance': float(var),
        'zero_crossing_rate': float(zero_crossings / len(signal)),
    }


def extract_resp_features(signal, fs):
    """Breathing rate + amplitude variability from a respiration window, via neurokit2."""
    if not _NK_AVAILABLE:
        raise ImportError("neurokit2 is required for RESP features. Run: pip install neurokit2 --break-system-packages")
    signal = np.asarray(signal, dtype=float)
    try:
        signals, info = nk.rsp_process(signal, sampling_rate=fs)
        return {
            'breathing_rate': float(signals['RSP_Rate'].mean()),
            'amplitude_mean': float(signals['RSP_Amplitude'].mean()),
            'amplitude_std': float(signals['RSP_Amplitude'].std()),
        }
    except Exception:
        return {'breathing_rate': np.nan, 'amplitude_mean': np.nan, 'amplitude_std': np.nan}


def extract_temp_features(signal):
    """
    Temperature-specific features (items 51-57): mean, std, min, max,
    range, slope (trend), variance.
    """
    signal = np.asarray(signal, dtype=float)
    signal = signal[~np.isnan(signal)]
    if len(signal) == 0:
        return {'mean_temp': np.nan, 'std_temp': np.nan, 'min_temp': np.nan,
                'max_temp': np.nan, 'range_temp': np.nan, 'slope_temp': np.nan,
                'variance_temp': np.nan}
    x = np.arange(len(signal))
    slope = float(np.polyfit(x, signal, 1)[0]) if len(signal) > 1 else 0.0
    return {
        'mean_temp': float(np.mean(signal)),
        'std_temp': float(np.std(signal)),
        'min_temp': float(np.min(signal)),
        'max_temp': float(np.max(signal)),
        'range_temp': float(np.max(signal) - np.min(signal)),
        'slope_temp': slope,
        'variance_temp': float(np.var(signal)),
    }


# ---------------------------------------------------------------------
# Combined extractor: generic statistical features (1-18) PLUS the
# signal-specific features (EDA/HR/IBI/BVP/TEMP), or generic-only for
# signal types with no dedicated extractor (e.g. ACC, EMG, RESP).
# ---------------------------------------------------------------------

_SPECIFIC_EXTRACTOR_MAP = {
    'ecg': lambda row, fs: extract_ecg_features(row, fs),
    'eda': lambda row, fs: extract_eda_features(row, fs),
    'hr': lambda row, fs: extract_hr_features(row),
    'bvp': lambda row, fs: extract_bvp_features(row, fs),
    'emg': lambda row, fs: extract_emg_features(row),
    'resp': lambda row, fs: extract_resp_features(row, fs),
    'temp': lambda row, fs: extract_temp_features(row),
    'acc': None,   # generic statistical features only
}


def extract_full_features(signal, signal_type, fs=None):
    """
    Combine the GENERIC statistical/signal features (items 1-18, computed
    for every signal type) with the SIGNAL-SPECIFIC features for the given
    signal_type (EDA, HR, BVP, TEMP get extra domain features; ACC/EMG/RESP
    fall back to generic-only or their own specific set where defined).

    Parameters
    ----------
    signal : 1D array-like
        One window of samples.
    signal_type : str
        One of: 'ecg', 'eda', 'hr', 'bvp', 'emg', 'resp', 'temp', 'acc'
    fs : float, optional
        Sampling rate in Hz. Required for 'ecg', 'eda', 'bvp', 'resp'.

    Returns
    -------
    dict combining generic + signal-specific features for this window.
    """
    signal_type = signal_type.lower()
    if signal_type not in _SPECIFIC_EXTRACTOR_MAP:
        raise ValueError(f"signal_type must be one of {list(_SPECIFIC_EXTRACTOR_MAP.keys())}")

    generic = statistical_features(signal)

    extractor = _SPECIFIC_EXTRACTOR_MAP[signal_type]
    if extractor is None:
        return generic

    if signal_type in ('ecg', 'eda', 'bvp', 'resp'):
        if fs is None:
            raise ValueError(f"fs is required for signal_type='{signal_type}'")
        specific = extractor(signal, fs)
    else:
        specific = extractor(signal, None)

    return {**generic, **specific}


# ---------------------------------------------------------------------
# Driver: apply extract_full_features to every row of a windows_*.csv file
# ---------------------------------------------------------------------

def process_windowed_csv(csv_path, signal_type, fs=None, generic_only=False):
    """
    Read a windows_<SIGNAL>.csv (one row per window) and extract features
    for every window.

    Parameters
    ----------
    csv_path : str
        Path to the windows_<SIGNAL>.csv file.
    signal_type : str
        One of: 'ecg', 'eda', 'hr', 'bvp', 'emg', 'resp', 'temp', 'acc'
        (use extract_ibi_features / process_ibi_windows directly for IBI,
        since IBI windows are variable-length and stored differently)
    fs : float, optional
        Sampling rate in Hz. Required for 'ecg', 'eda', 'bvp', 'resp'.
    generic_only : bool, default False
        If True, skip signal-specific features and return only the
        generic statistical/signal features (items 1-18).

    Returns
    -------
    pandas.DataFrame, one row per window, one column per extracted feature.
    """
    df = pd.read_csv(csv_path)

    rows = []
    for _, row in df.iterrows():
        window = row.values
        if generic_only:
            feats = statistical_features(window)
        else:
            feats = extract_full_features(window, signal_type, fs)
        rows.append(feats)

    return pd.DataFrame(rows)


def process_ibi_windows(ibi_windows, include_generic=True):
    """
    IBI windows are variable-length (different beat counts per window),
    so they can't be loaded from a fixed-column CSV like the others.
    Pass in the `ibi_windows` list you built during the windowing step
    (a list of 1D arrays, one per window).

    Parameters
    ----------
    ibi_windows : list of 1D array-like
    include_generic : bool, default True
        If True, also compute the generic statistical features (1-18) on
        each window's raw IBI values, alongside the HRV-specific features.

    Returns
    -------
    pandas.DataFrame, one row per window, one column per extracted feature.
    """
    rows = []
    for w in ibi_windows:
        feats = extract_ibi_features(w)
        if include_generic:
            feats = {**statistical_features(w), **feats}
        rows.append(feats)
    return pd.DataFrame(rows)
