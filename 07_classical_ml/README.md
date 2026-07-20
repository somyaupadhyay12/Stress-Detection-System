# Reproducible signal-wise experiments

Run the complete comparison from `Stress-Detection-System`:

```powershell
.\.venv\Scripts\python.exe 07_classical_ml\run_signalwise_experiments.py
.\.venv\Scripts\python.exe 07_classical_ml\generate_research_report.py
```

The first command evaluates Raw, Filtered, and Filtered + Normalized inputs for each of these signals and for `ALL_SIGNALS`: `E4_ACC`, `E4_BVP`, `E4_EDA`, `E4_HR`, `E4_IBI`, `E4_TEMP`, `RB_ACC`, `RB_ECG`, `RB_EDA`, `RB_EMG`, `RB_RESP`, `RB_TEMP`.

For a bounded smoke run, choose a small participant set and one signal:

```powershell
.\.venv\Scripts\python.exe 07_classical_ml\run_signalwise_experiments.py --subjects S2 S3 S4 --signals E4_EDA
```

All outputs are under `07_classical_ml/outputs/signalwise/`:

- `stage_record.csv` — input shape, window count, feature count, and elapsed time per stage.
- `signalwise_model_comparison.csv` and `signalwise_comparison_table.csv` — Accuracy, Precision, Recall, F1, ROC-AUC, training time, and inference time.
- `<signal>/<variant>/features.csv`, held-out predictions, and fitted model — traceable per-experiment artifacts.
- `feature_quality_analysis.csv`, `best_variant_by_signal.csv`, and `research_report.md` — feature-value, quality, interpretation, and conclusion documentation.

The experiment uses the same five-second, 50%-overlap windows; label map; GroupShuffleSplit subject split; median imputation; standardization; and balanced logistic-regression model for every comparison. Only preprocessing variant and signal selection change.
