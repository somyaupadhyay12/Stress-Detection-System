# ACC Pipeline Audit

Scope: Empatica E4 ACC and RespiBAN ACC only. Inspection completed before ACC implementation.

| Stage | E4 ACC | RespiBAN ACC | Gap |
|---|---|---|---|
| Raw source | `ACC.csv` and PKL wrist signal exist for all 15 subjects. | PKL chest `ACC` exists for all 15 subjects. | None. |
| Filtering | Existing signal-wise runner applies a 2 Hz low-pass design; raw-window and filtering figures exist. | Existing runner applies a 20 Hz low-pass design before common-rate alignment; raw plots exist. | Filtered arrays, per-ACC statistics, and noise-reduction tables are not persisted. |
| Normalization | Existing per-subject z-score logic is used in the signal-wise runner. | Same. | Normalized arrays/statistics are not persisted. |
| Windowing | Existing generic windowing handles 3-axis ACC. | Same. | No ACC-specific labelled-window NPZ/manifest outputs. |
| Features | Axis-wise generic features appear in `E4_ACC` signal-wise CSVs. | Axis-wise generic features appear in `RB_ACC` CSVs. | No acceleration-magnitude feature set, feature summary, or ACC-specific feature overview. |
| Scaling/model ready | Scaler is embedded in each saved classical pipeline. | Same. | No explicit ACC-only scaled matrices for classical/DL consumers. |
| Classical ML | Raw/filtered/normalized models and predictions exist for S2--S4. | Same. | Full 15-subject run remains outside this ACC-only task. |
| Deep-learning ready inputs | E4 ACC is indirectly part of the existing combined wrist GRU path. | Not available. | No ACC-only DL-ready arrays for either device. |

Implementation must reuse `normalize_signals.py`, `windowing_filtered.py`, `windowing_normalised.py`, and `feature_extraction.py`; it must not alter other signal outputs.
