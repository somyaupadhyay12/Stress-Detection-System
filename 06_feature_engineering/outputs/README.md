# Feature Extraction and ML Output Layout

Each data variant has its own folder: `raw`, `filtered`, and
`filtered_normalized`.

- `windows_before_feature_extraction/`
  - `<subject>_windows.npz`: exact raw window arrays passed into feature extraction
  - `<subject>_window_labels.csv`: original and binary window labels
  - `<subject>_window_manifest.csv`: signal names and array shapes
  - `<subject>_window_preview_before_feature_extraction.png`: visual record of the window inputs
- `features_before_model.csv`: complete feature-extraction output before ML
- `feature_overview_before_model.png`: class distribution and feature summary
- `model_ready_scaled_features_before_model.npz`: train/test feature matrices after scaling, labels, subjects, and feature names
- `features_after_model.csv`: complete held-out feature rows with prediction and stress probability
- `training_history.csv`, `training_and_roc.png`, `confusion_matrix_after_model.png`,
  `classification_report.csv`, and `metrics.csv`: model results

`model_comparison.csv` at this folder's root compares all three variants.
