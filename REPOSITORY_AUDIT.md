# Repository Audit

Audit date: 20 July 2026. This document is the required baseline for later work. It records the repository state as inspected; it does not claim that a saved figure or model represents a completed end-to-end run for all 15 subjects.

## Architecture and execution flow

```text
Data/Raw/WESAD (15 subjects: S2--S17, excluding S12)
  -> 01_filtering notebook (exploratory filtering figures)
  -> 02_normalization notebook / normalize_signals.py
  -> 03_raw_windowing and 04/05 filtered/normalized windowing
  -> 06_feature_engineering feature functions and saved variant artifacts
  -> 07_classical_ml (classical grouped models and signal-wise comparisons)
  -> optional GRU path (train_window_feature_models.py / saved raw GRU)
  -> plots/ and report/ (visualization and LaTeX reporting)
```

The current script-first reproducible path is `07_classical_ml/run_signalwise_experiments.py`, followed by `07_classical_ml/generate_research_report.py`. It reconstructs inputs directly from `Data/Raw/WESAD`, rather than consuming a persistent `Data/Processed` dataset. The notebook runner (`run_all.py`) is a separate, older path and is not currently connected to the ML scripts.

## Folder hierarchy and purpose

| Folder/file | Current purpose and contents |
|---|---|
| `Data/Raw/WESAD/` | Full WESAD inputs for 15 subjects. Every inspected subject has `<subject>.pkl`, an E4 CSV folder, and a RespiBAN text file. |
| `Data/Processed/` | Intended processed-data location; currently empty. |
| `01_filtering/` | Exploratory filtering notebook plus extensive raw-versus-filtered E4/RespiBAN PNG plots. No reusable filtered dataset is saved. |
| `02_normalization/` | Normalization notebook and reusable `normalize_signals.py`. No normalized dataset artifact is saved. |
| `03_raw_windowing/` | Raw-windowing notebook, helper notebooks, `raw_window.py`, and raw/window-preview figures. |
| `04_windowing_filtered/` | Filtered-windowing notebook and reusable `windowing_filtered.py`. |
| `05_windowing_normalised/` | Normalized-windowing notebook and reusable `windowing_normalised.py`. |
| `06_feature_engineering/` | Generic/domain feature functions, feature notebook, and raw/filtered/filtered-normalized artifacts for S2--S4. |
| `07_classical_ml/` | Classical grouped ML, signal-wise experiment runner, optional GRU training path, result/report generators, and saved models. |
| `models/` | Legacy copy of a model notebook and outputs; duplicates `07_classical_ml/model.ipynb` in purpose and contains stale paths. |
| `notebooks/` | Day-by-day exploratory notebooks/scripts; not part of the reproducible pipeline. |
| `archive/` | Historical E4/RespiBAN notebooks; not used by the current pipeline. |
| `plots/` | PNG visualization collection, including raw, filtering, features, ML, and confusion-matrix sections. |
| `report/` | Standalone `main.tex`, selected PNG figures, and build notes. |
| `run_all.py`, `run_all.bat` | Legacy notebook orchestrator. |
| `requirements-ml.txt` | Declared ML dependencies; the local virtual environment lacks some declared plotting/report extras. |

## Notebooks

| Notebook | Purpose | Status |
|---|---|---|
| `01_filtering/01_filtering.ipynb` | E4 and RespiBAN filter exploration/plots. | Executes in saved output; no persistent filtered arrays. |
| `02_normalization/02_normalization.ipynb` | Signal normalization examples. | Executes in saved output; no persistent normalized arrays. |
| `03_raw_windowing/03_raw_windowing.ipynb` | Raw signal/window visualization. | Saved source has a JSON decode error; output copy has no saved error. |
| `03_raw_windowing/common_filter/eda*.ipynb` | EDA-specific filtering/window experiments. | Exploratory; not used by the script pipeline. |
| `04_windowing_filtered/04_windowing_filtered.ipynb` | Filtered window examples. | Saved output has no error. |
| `05_windowing_normalised/05_windowing_normalised.ipynb` | Normalized window examples. | Saved output has no error. |
| `06_feature_engineering/feature_extraction.ipynb` | Feature-extraction demonstration. | Saved output has no error. |
| `06_feature_engineering/outputs/raw/plot_feature_before_vs_after.ipynb` | Feature plotting attempt. | Broken relative path. |
| `07_classical_ml/model.ipynb`, `models/model.ipynb` | Legacy model notebook copies. | Broken relative raw-data path and failed concatenation. |
| `reference.ipynb` | Reference deep-learning/model workflow. | Saved output has no error, but is not the reproducible command-line path. |
| `notebooks/day*.ipynb` | Exploratory learning/prototyping. | Some stale relative paths and unpickling/name errors; exclude from production execution. |

## Python files

| File | Purpose | Status |
|---|---|---|
| `run_all.py` | Executes numbered preprocessing notebooks and copies figures. | Uses obsolete lowercase data-path descriptions and does not run feature/ML/report stages. |
| `02_normalization/normalize_signals.py` | Z-score/min-max normalization helpers. | Reusable, but does not persist data. |
| `03_raw_windowing/common_filter/raw_window.py` | Raw sensor CSV/text readers, window helpers, preview plots. | Exploratory/helper module; paths in main block are stale. |
| `04_windowing_filtered/windowing_filtered.py` | Generic fixed-length filtered-window builder. | Reusable and used by ML code. |
| `05_windowing_normalised/windowing_normalised.py` | Generic normalized-window builder. | Reusable and used by ML code. |
| `06_feature_engineering/feature_extraction.py` | Generic statistics plus EDA/ECG/BVP/HR/IBI/EMG/RESP/TEMP extractors. | Reusable; optional advanced features require NeuroKit2 and are not in the completed classical comparison. |
| `07_classical_ml/train_classical_models.py` | Wrist-only grouped classical baselines (Dummy, LR, RF, SVM). | Imports repaired; saved output currently covers filtered wrist features only. |
| `07_classical_ml/run_signalwise_experiments.py` | Raw/filtered/normalized per-signal and all-signal logistic-regression experiments. | Completed output exists for S2, S3, S4; needs all-subject rerun for final claims. |
| `07_classical_ml/generate_research_report.py` | Builds feature-quality CSVs and Markdown conclusions from signalwise results. | Completed for the S2--S4 run. |
| `07_classical_ml/train_window_feature_models.py` | GRU variant training path. | Exists, but results are complete only for raw and do not include every signal/variant. |
| `07_classical_ml/create_feature_model_report.py` | Auditable figures/tables for GRU feature-model run. | Depends on missing/incomplete variant outputs. |
| `08_visualization_reporting/generate_visualizations.py` | Generates reporting plots and a LaTeX project from raw data/results. | Design is useful, but first raw-record loading is slow and it requires plotting dependencies absent from `.venv`. |
| `notebooks/day*.py` | Exploratory scripts. | Not reproducible production entry points; some stale paths. |

## Existing outputs, models, reports, and visualizations

- Filtering: many E4 and RespiBAN raw/filter comparison PNGs; no saved numerical filter-efficiency/PSD/FFT table.
- Windowing: raw/filtered/normalized previews; window NPZ/manifests/labels for S2--S4 in feature-engineering outputs.
- Features: raw, filtered, and filtered-normalized feature tables, scaled matrices, and previews for S2--S4. Only raw contains the saved GRU, training history, ROC/training image, metrics, and post-model table.
- Classical ML: filtered wrist model comparison, held-out predictions, classifier report/confusion matrix, and fitted model.
- Signal-wise ML: 39 completed logistic-regression experiments (12 signals plus all-signal fusion x 3 variants) for S2--S4, including features, predictions, models, metrics, protocol, quality table, and Markdown report.
- Deep learning: a raw GRU model and its artifacts exist; filtered and filtered-normalized DL artifacts are absent.
- Visualizations: `plots/` currently has 113 PNGs, mainly copied/reorganized existing figures plus generated raw and comparison plots.
- Reports: `07_classical_ml/outputs/signalwise/research_report.md` and `report/main.tex` exist. The LaTeX source has selected figures but is not verified as compiled because no TeX executable was found.

## Signal coverage matrix

| Stage | E4 ACC/BVP/EDA/TEMP | E4 HR/IBI | RespiBAN ACC/ECG/EDA/EMG/RESP/TEMP |
|---|---|---|---|
| Raw source | Complete | Complete (CSV) | Complete |
| Exploratory filtering/plots | Substantial | Substantial | Substantial except ACC has limited dedicated filtering evidence |
| Persistent filtered/normalized arrays | Missing | Missing | Missing |
| Window artifacts | Partial, S2--S4 | Limited/plot-level | Partial, S2--S4 |
| Generic feature tables | Present through signalwise outputs | Present through signalwise outputs | Present through signalwise outputs |
| Advanced domain features | Partial/module-only | IBI module-only | ECG/EMG/RESP module-only |
| Classical ML | Complete S2--S4 signalwise | Complete S2--S4 signalwise | Complete S2--S4 signalwise |
| Deep learning | Raw combined wrist path only | Missing signal-wise | Missing |
| Final all-subject result | Missing | Missing | Missing |

## Missing or incomplete work

1. **Persistent preprocessing outputs:** filtered and normalized signal arrays are not saved to `Data/Processed`, so notebooks and scripts duplicate reconstruction logic.
2. **All-subject evaluation:** signal-wise results cover three subjects only; the full 15-subject WESAD dataset has not produced final metrics.
3. **Deep learning:** no comparable Raw/Filtered/Filtered+Normalized DL experiments for every signal or combined signals; no CNN/LSTM/MLP comparison.
4. **ACC completion:** ACC is in signal-wise classical ML, but lacks a verified deep-learning comparison and complete dedicated filtering documentation.
5. **Feature parity:** advanced EDA/ECG/BVP/IBI/RESP features exist in code but are not consistently used in ML outputs; acceleration remains generic-statistics-only by design.
6. **ML breadth:** current complete signal-wise run uses only logistic regression. The suggested SVM/RF/DT/XGBoost/KNN/NB comparison is incomplete.
7. **Evaluation visualizations:** signalwise ROC/PR curves, learning curves, per-model classification reports, and model-importance plots are incomplete/not systematically generated.
8. **Filtering validation:** PSD, FFT, frequency-response, noise/spike, SNR, and quantitative efficiency tables are missing.
9. **Normalization validation:** comprehensive before/after statistics/distributions are missing.
10. **Documentation:** root beginner README and technical architecture guide are missing; existing README files are narrow output notes.
11. **LaTeX verification:** `report/main.tex` has not been compiled and visualized; its tables are not yet generated from CSV due missing Jinja2 in `.venv`.
12. **Legacy cleanup:** multiple notebooks retain saved errors; they need path correction or explicit archival status, but should not be silently deleted.

## Duplicate, unused, and broken components

- `models/model.ipynb` and `07_classical_ml/model.ipynb` are functionally duplicate and both have stale paths.
- `archive/`, `notebooks/day*.ipynb`, and `common_filter` notebooks are exploratory/legacy and duplicate parts of the numbered pipeline.
- Empty `preprocessing/` and `feature_engineering/` directories are historical import targets; numbered module folders are the real implementation locations.
- `run_all.py` reports old paths and stops before feature engineering, ML, DL, visual reporting, and report generation.
- Saved notebook execution errors are listed above. They should be fixed only after the script path becomes the canonical pipeline.

## Priority plan

1. Establish one canonical script workflow and persist raw/filtered/normalized subject artifacts with manifests in `Data/Processed`.
2. Run and validate all 15 subjects with subject-independent splits; regenerate all signal-wise classical tables/plots/reports.
3. Complete the DL experiment matrix (at minimum GRU baseline for all three variants and all-signal fusion; add signal-wise DL only if computationally feasible).
4. Add quantitative preprocessing validation (PSD/FFT/SNR/noise/drift and normalization summaries), including ACC.
5. Integrate the existing domain-specific extractors into a controlled feature-set experiment without leakage.
6. Expand model comparison only after the canonical feature tables are frozen.
7. Repair or label legacy notebooks as archival; update `run_all.py` or replace its documentation with the canonical script order.
8. Write root README and technical documentation; compile and visually verify the LaTeX report after the final metrics/plots exist.

## Audit conclusion

The project is not starting from zero: raw data, exploratory preprocessing, reusable window/feature utilities, three-subject classical signal-wise results, a raw GRU artifact, and extensive figures already exist. It is not yet publication-ready because its authoritative all-subject persisted pipeline, complete DL matrix, quantitative preprocessing verification, and compiled final documentation remain unfinished. Future work must extend the numbered-stage modules and script-first flow rather than replace them.
