# Classical ML stress detection

`train_robust_model.py` is the current subject-independent WESAD wrist-signal training workflow. It runs Leave-One-Subject-Out evaluation for the `raw`, `filtered`, and `filtered_normalized` preprocessing variants and supports Random Forest, Logistic Regression, XGBoost, SVM, KNN, and Decision Tree models. `2.py` and `2plot.py` remain as compatibility launchers.

Run training from the project root:

```powershell
.\.venv\Scripts\python.exe 07_classical_ml\train_robust_model.py --dataset wesad --compare-models
.\.venv\Scripts\python.exe 07_classical_ml\plot_analysis.py --dataset wesad
```

The dataset namespace is selected with `--dataset`. WESAD outputs are stored in `07_classical_ml/outputs/robust_model/WESAD/`; the plotting script auto-discovers the available model CSV files in each variant folder.

Dataset responsibilities are separated from the generic preprocessing, windowing, feature extraction, modeling, and plotting workflow:

- `01_data_loaders/` contains the common loader contract and WESAD pickle loader.
- `configs/` contains WESAD labels, sampling rates, signal types, and dataset root configuration.
- `train_robust_model.py` owns generic alignment, filtering, normalization, feature construction, and LOSO training.
- `plot_analysis.py` creates the per-subject, distribution, dummy-baseline, class-report, feature, and cross-model/variant charts.

To add a second dataset later, add (a) a loader in `01_data_loaders/`, (b) a config module in `configs/`, and (c) one entry in `DATASET_REGISTRY` in `train_robust_model.py`.

`train_classical_models.py` remains in place because `run_signalwise_experiments.py` and `run_model_comparison.py` import its shared helpers. The old GRU script is retained under `_archive/`.
