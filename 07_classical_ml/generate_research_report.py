"""Generate auditable research documentation from signal-wise experiment files.

Run after ``run_signalwise_experiments.py``.  The report uses computed feature
values and metrics; it never substitutes illustrative results for real ones.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT / "07_classical_ml" / "outputs" / "signalwise"
REPORT = ROOT / "research_report.md"
META = {"subject_id", "window_index", "original_label", "stress_label"}

SIGNAL_INFO = {
    "E4_ACC": ("Empatica E4 accelerometer", "g (source scale is 1/64 g)", "Movement and motion artefacts; stress can alter activity/restlessness."),
    "E4_BVP": ("Empatica E4 blood-volume pulse", "device units", "Peripheral pulse morphology reflects autonomic cardiovascular activity."),
    "E4_EDA": ("Empatica E4 electrodermal activity", "µS", "Sympathetic sweat-gland activity; a widely used stress marker."),
    "E4_HR": ("Empatica E4 heart rate", "beats/min", "Cardiac rate commonly rises with sympathetic arousal."),
    "E4_IBI": ("Empatica E4 inter-beat interval", "seconds", "Beat timing and its variability reflect autonomic regulation."),
    "E4_TEMP": ("Empatica E4 skin temperature", "°C", "Peripheral vasoconstriction/thermal response may accompany arousal."),
    "RB_ACC": ("RespiBAN accelerometer", "device units", "Movement/context signal and potential physiological artefact indicator."),
    "RB_ECG": ("RespiBAN ECG", "mV/device units", "Cardiac waveform; morphology and variability carry stress information."),
    "RB_EDA": ("RespiBAN electrodermal activity", "µS/device units", "Sympathetic sudomotor activity."),
    "RB_EMG": ("RespiBAN electromyogram", "mV/device units", "Muscle activation/tension can increase during stress."),
    "RB_RESP": ("RespiBAN respiration", "device units", "Breathing pattern, rate and amplitude can change under stress."),
    "RB_TEMP": ("RespiBAN temperature", "°C/device units", "Slow peripheral thermal trend; interpret over longer intervals."),
}

FEATURE_INFO = {
    "mean": "average level", "median": "middle value", "std": "within-window variability", "variance": "squared variability",
    "min": "lowest value", "max": "highest value", "range": "max minus min", "iqr": "middle-50% spread",
    "skewness": "asymmetry of value distribution", "kurtosis": "tail/peakedness relative to normal", "rms": "root-mean-square magnitude",
    "energy": "sum of squared samples", "entropy": "10-bin Shannon distribution entropy (bits)", "mav": "mean absolute value",
    "auc": "sample-domain trapezoidal area", "slope": "linear sample-to-sample trend", "zero_crossing_rate": "fraction of adjacent sign changes",
    "cv": "standard deviation divided by mean",
}


def markdown_table(frame: pd.DataFrame) -> str:
    """Render Markdown without adding the optional ``tabulate`` dependency."""
    columns = [str(column) for column in frame.columns]
    def cell(value: object) -> str:
        if isinstance(value, (float, np.floating)):
            return "" if not np.isfinite(value) else f"{value:.4f}"
        return str(value).replace("|", "\\|").replace("\n", " ")
    body = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    body.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(body)


def feature_quality(comparisons: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for signal_name in SIGNAL_INFO:
        raw_path = ROOT / signal_name / "raw" / "features.csv"
        filtered_path = ROOT / signal_name / "filtered" / "features.csv"
        if not raw_path.exists():
            continue
        raw, filtered = pd.read_csv(raw_path), pd.read_csv(filtered_path)
        feature_columns = [c for c in raw.columns if c not in META]
        for column in feature_columns:
            raw_values = raw[column].replace([np.inf, -np.inf], np.nan)
            filtered_values = filtered[column].replace([np.inf, -np.inf], np.nan)
            raw_missing, filtered_missing = raw_values.isna().mean(), filtered_values.isna().mean()
            raw_std, filtered_std = raw_values.std(), filtered_values.std()
            state = "review"
            if raw_missing == 0 and filtered_missing == 0 and np.isfinite(raw_std) and np.isfinite(filtered_std):
                state = "usable"
            if raw_missing > .05 or filtered_missing > .05:
                state = "missing-values"
            rows.append({"signal": signal_name, "feature": column.split("_")[-1], "column": column,
                         "raw_mean": raw_values.mean(), "raw_min": raw_values.min(), "raw_max": raw_values.max(),
                         "raw_missing_fraction": raw_missing, "filtered_missing_fraction": filtered_missing,
                         "raw_std": raw_std, "filtered_std": filtered_std,
                         "filter_variability_change_pct": 100 * (filtered_std - raw_std) / raw_std if raw_std else np.nan,
                         "quality_flag": state})
    return pd.DataFrame(rows)


def main() -> None:
    comparison_path = ROOT / "signalwise_model_comparison.csv"
    if not comparison_path.exists():
        raise FileNotFoundError("Run run_signalwise_experiments.py before generating the report.")
    comparison = pd.read_csv(comparison_path)
    protocol = json.loads((ROOT / "experiment_protocol.json").read_text(encoding="utf-8"))
    quality = feature_quality(comparison)
    quality.to_csv(ROOT / "feature_quality_analysis.csv", index=False)
    best = comparison.loc[comparison.groupby("experiment").f1.idxmax()].sort_values("f1", ascending=False)
    best.to_csv(ROOT / "best_variant_by_signal.csv", index=False)
    all_rows = comparison[comparison.experiment == "ALL_SIGNALS"].copy()
    lines = ["# WESAD Stress-Detection Experimental Report", "", "## Reproducibility protocol", "",
             f"- Subjects: {', '.join(protocol['subjects'])}", f"- Labels: {protocol['label_map']} (1/3 = non-stress; 2 = stress).",
             f"- Windows: {protocol['window_seconds']} s, {protocol['overlap']:.0%} overlap, aligned to {protocol['target_fs_hz']} Hz.",
             f"- Constant evaluation pipeline: {protocol['model']}.", f"- Subject-independent split: {protocol['split']}.",
             "", "Only preprocessing changes between Raw, Filtered, and Filtered + Normalized.  Therefore each row is directly comparable within a signal.",
             "", "## Pipeline stage record", "", "The accompanying `stage_record.csv` records input shapes, window counts, feature counts, and elapsed time for every subject/signal/variant.", "",
             "## All-signal comparison", "", markdown_table(all_rows[["variant","accuracy","precision","recall","f1","roc_auc","training_seconds","inference_ms_per_window"]]), "",
             "## Best preprocessing variant per experiment", "", markdown_table(best[["experiment","variant","accuracy","f1","roc_auc","n_features"]]), "",
             "## Feature extraction and physiological interpretation", "",
             "The ML experiments extract exactly these 18 generic features from each available signal channel: " + ", ".join(FEATURE_INFO) + ".  These are window-level statistical/time-domain features; their value units inherit the source signal except energy (unit²), variance (unit²), entropy (bits), slope (unit/sample), and zero-crossing rate (unitless)."]
    for signal_name, (name, unit, rationale) in SIGNAL_INFO.items():
        subset = quality[quality.signal == signal_name]
        if subset.empty:
            continue
        feature_table = subset.groupby("feature", as_index=False).agg(value_mean=("raw_mean", "mean"), value_min=("raw_min", "min"), value_max=("raw_max", "max"), missing_fraction=("raw_missing_fraction", "max"), quality=("quality_flag", "first"))
        feature_table["description"] = feature_table.feature.map(FEATURE_INFO)
        feature_table["unit"] = unit
        feature_table["interpretation"] = np.where(feature_table.quality.eq("usable"), "Finite values; inspect class separation and preprocessing change.", "Review missing/non-finite values before inference.")
        lines += ["", f"### {signal_name}: {name}", "", rationale, "", "Expected ranges are participant-, device-, and context-dependent; use the raw min/max below as the observed study range rather than applying a universal clinical threshold.", "", markdown_table(feature_table[["feature","description","value_mean","value_min","value_max","unit","interpretation"]]), "",
                  "Conclusion: " + ("features are numerically complete in this run; model comparison below indicates their discriminative value." if (subset.quality_flag == "usable").all() else "one or more features require data-quality review; see feature_quality_analysis.csv.")]
    lines += ["", "## Signal-wise preprocessing comparison", "", markdown_table(comparison[["experiment","variant","accuracy","precision","recall","f1","roc_auc","training_seconds","inference_ms_per_window"]]), "",
              "## Interpretation and recommended configuration", ""]
    overall = best[best.experiment == "ALL_SIGNALS"].iloc[0]
    strongest = best[best.experiment != "ALL_SIGNALS"].iloc[0]
    weakest = best[best.experiment != "ALL_SIGNALS"].iloc[-1]
    lines += [f"- Best all-signal preprocessing: **{overall.variant}** (held-out F1 {overall.f1:.4f}, ROC-AUC {overall.roc_auc:.4f}).",
              f"- Strongest individual experiment: **{strongest.experiment} / {strongest.variant}** (F1 {strongest.f1:.4f}).",
              f"- Weakest individual experiment: **{weakest.experiment} / {weakest.variant}** (F1 {weakest.f1:.4f}).",
              "- Filtering/normalization are meaningful only when the held-out F1/ROC-AUC improvement persists on a new subject split; the tables report estimates, not clinical claims.",
              "- Recommended feature set: retain features with `quality_flag=usable`, then select within training folds only to avoid test-subject leakage.",
              "- Future work: add dedicated ECG/EDA/BVP peak features from the existing feature_extraction.py module, nested cross-validation, and leave-one-subject-out evaluation.", "",
              "## References", "", "- Schmidt, P. et al. (2018). Introducing WESAD, a multimodal dataset for wearable stress and affect detection. ICMI 2018.",
              "- The generic statistics are implemented in this repository's `06_feature_engineering/feature_extraction.py`; signal-specific extractors there are not part of the present classical-ML experiment unless explicitly added."]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
