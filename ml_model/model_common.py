 

"""Shared training/evaluation logic for the per-model WESAD stress scripts.

Each concrete model file (train_logistic_regression.py, train_svm.py,
train_random_forest.py, train_knn.py, train_xgboost.py,
train_decision_tree.py) only defines a `model_factory()` and calls
`run_model(name, model_factory)` from here. Everything else -- loading
subjects, building windows/features, validating, testing, and saving
results -- lives in this module so it is identical (and only maintained in
one place) across all six models.

Every model/variant combination goes through the same 3-step process,
across four preprocessing variants: "raw" (unfiltered signal, statistical
features), "filtered" (band/low-pass filtered, statistical features),
"filtered_normalized" (filtered + per-subject z-scored, statistical
features), and "raw_windows" (unfiltered signal, but the raw window
samples themselves are flattened directly into the feature vector instead
of extracting statistical features -- lets the models learn from waveform
shape directly).

1. TRAIN/TEST SPLIT -- subjects are split with a grouped shuffle split into
   a training pool (~80% of subjects) and a locked-away TEST set (~20% of
   subjects). The test set is never touched until step 3.

2. VALIDATE -- run on the training pool ONLY, using two independent
   validation algorithms, so tuning/model-selection decisions never see
   the test set:
     a. GroupKFold CV (grouped by subject, 5 folds by default) -- fast,
        gives a variance estimate across folds.
     b. Leave-One-Subject-Out (LOSO) -- every training-pool subject held
        out exactly once, trained on the rest of the training pool. This
        is the more rigorous subject-independent estimate (see
        train_robust_model.py's docstring for why this matters more than
        a single random split).

3. TEST -- fit one model on the *entire* training pool and evaluate it
   exactly once on the locked-away test subjects. This is the single
   number that reflects genuinely unseen data, uncontaminated by any
   validation-time decisions. A separate deployment model trained on ALL
   subjects (train + test) is then saved for actual downstream use.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib

# from data_loaders import wesad_data_loader
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import signal
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, LeaveOneGroupOut, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "windowing_filtered"))
sys.path.insert(0, str(PROJECT_ROOT / "windowing_normalised"))
sys.path.insert(0, str(PROJECT_ROOT / "feature_engineering"))
from windowing_filtered import create_windows_filtered
from windowing_normalised import create_windows_normalized
import feature_extraction

extract_full_features = feature_extraction.extract_full_features


WINDOW_SEC = 5
OVERLAP = 0.5
RANDOM_SEED = 42
N_CV_FOLDS = 5

from data_loaders import wesad_data_loader
wesad_loader = importlib.import_module("data_loaders.wesad_data_loader")
 
wesad_config = importlib.import_module("configs.wesad_config")

DATASET_REGISTRY = {
    "wesad": {"loader": wesad_data_loader, "config": wesad_config},
}


# --------------------------------------------------------------------------
# Signal processing / feature building (identical to train_robust_model.py)
# --------------------------------------------------------------------------

def resample_to_target(values: np.ndarray, target_length: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return signal.resample(values, target_length, axis=0)


def _sos_filter(values: np.ndarray, fs: float, cutoff, kind: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    nyq = fs / 2
    if kind == "band":
        sos = signal.butter(4, [cutoff[0] / nyq, cutoff[1] / nyq], btype="band", output="sos")
    else:
        sos = signal.butter(4, cutoff / nyq, btype="low", output="sos")
    return signal.sosfiltfilt(sos, values, axis=0)


def filter_wrist_signals(wrist: dict, native_fs: dict) -> dict:
    return {
        "ACC": _sos_filter(wrist["ACC"], native_fs["ACC"], 2.0, "low"),
        "BVP": _sos_filter(wrist["BVP"], native_fs["BVP"], (0.7, 5.0), "band"),
        "EDA": _sos_filter(wrist["EDA"], native_fs["EDA"], 1.0, "low"),
        "TEMP": _sos_filter(wrist["TEMP"], native_fs["TEMP"], 0.5, "low"),
    }


def align_signals(wrist: dict, labels: np.ndarray, native_label_fs: float, target_fs: float) -> tuple[dict, np.ndarray]:
    target_length = int(len(labels) / native_label_fs * target_fs)
    source = {key: wrist[key] for key in ("ACC", "BVP", "EDA", "TEMP")}
    aligned = {key: resample_to_target(value, target_length) for key, value in source.items()}
    aligned_labels = np.rint(resample_to_target(labels, target_length)).astype(int)
    return aligned, aligned_labels


def normalize_per_subject(signals_dict: dict) -> dict:
    result = {}
    for name, values in signals_dict.items():
        values = np.asarray(values, dtype=float)
        result[name] = StandardScaler().fit_transform(values.reshape(len(values), -1)).reshape(values.shape)
    return result


def windows_to_features(window_result: dict, subject: str, config) -> pd.DataFrame:
    """One labeled feature row per 5-second window, using domain-specific
    features (EDA phasic/tonic, BVP peaks, TEMP trend) where available."""
    windows = window_result["windows"]
    labels = np.asarray(window_result["labels"])
    rows = []
    channel_map = {"ACC": ("x", "y", "z"), "BVP": ("value",), "EDA": ("value",), "TEMP": ("value",)}
    for index, label in enumerate(labels):
        if int(label) not in config.VALID_LABELS:
            continue
        row = {"subject": subject, "window_index": index, "label_original": int(label), "stress_label": config.BINARY_LABELS[int(label)]}
        for signal_name, channel_names in channel_map.items():
            data = np.asarray(windows[signal_name][index])
            if data.ndim == 1:
                data = data[:, None]
            for channel, channel_name in enumerate(channel_names):
                sig_type = config.SIGNAL_TYPE_MAP[signal_name]
                try:
                    features = extract_full_features(data[:, channel], sig_type, fs=config.TARGET_FS)
                except Exception:
                    features = extract_full_features(data[:, channel], "acc", fs=config.TARGET_FS)  # generic fallback
                row.update({f"{signal_name}_{channel_name}_{feature}": value for feature, value in features.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def _fixed_length(values: np.ndarray, target_len: int) -> np.ndarray:
    """Pad (edge-value) or truncate a 1-D array to an exact length, so every
    window contributes the same number of flattened columns even if a
    window at a signal boundary comes out a sample or two short/long."""
    values = np.asarray(values, dtype=float)
    if len(values) == target_len:
        return values
    if len(values) > target_len:
        return values[:target_len]
    return np.pad(values, (0, target_len - len(values)), mode="edge")


def windows_to_raw_flattened_features(window_result: dict, subject: str, config) -> pd.DataFrame:
    """One labeled feature row per window, using the RAW sample values
    themselves (flattened) as the feature vector -- no statistical or
    domain feature extraction at all. This is the 'raw_windows' variant:
    it lets the models learn directly from the waveform shape instead of
    hand-engineered summaries, at the cost of many more, less interpretable
    input columns."""
    windows = window_result["windows"]
    labels = np.asarray(window_result["labels"])
    target_len = int(round(WINDOW_SEC * config.TARGET_FS))
    channel_map = {"ACC": ("x", "y", "z"), "BVP": ("value",), "EDA": ("value",), "TEMP": ("value",)}
    rows = []
    for index, label in enumerate(labels):
        if int(label) not in config.VALID_LABELS:
            continue
        row = {"subject": subject, "window_index": index, "label_original": int(label), "stress_label": config.BINARY_LABELS[int(label)]}
        for signal_name, channel_names in channel_map.items():
            data = np.asarray(windows[signal_name][index])
            if data.ndim == 1:
                data = data[:, None]
            for channel, channel_name in enumerate(channel_names):
                values = _fixed_length(data[:, channel], target_len)
                row.update({f"{signal_name}_{channel_name}_t{t:03d}": v for t, v in enumerate(values)})
        rows.append(row)
    return pd.DataFrame(rows)


def build_features(subjects: list[str], variant: str, dataset: dict) -> pd.DataFrame:
    loader, config = dataset["loader"], dataset["config"]
    fs_dict = {name: config.TARGET_FS for name in config.SIGNAL_TYPE_MAP}
    frames = []
    for subject in subjects:
        print(f"Loading {subject} ({variant})...")
        try:
            subject_data = loader.load_subject(subject, config.DATA_ROOT)
        except FileNotFoundError:
            print(f"[skip] missing subject {subject} under {config.DATA_ROOT}")
            continue
        wrist, native_fs = subject_data["wrist_signals"], subject_data["native_fs"]
        raw_aligned, labels = align_signals(wrist, subject_data["labels"], config.NATIVE_LABEL_FS, config.TARGET_FS)
        if variant == "raw":
            window_result = create_windows_filtered(raw_aligned, fs_dict, labels, config.TARGET_FS, WINDOW_SEC, OVERLAP)
            frames.append(windows_to_features(window_result, subject, config))
        elif variant == "raw_windows":
            # Same unfiltered, aligned signal and windowing as "raw" -- the
            # only difference is that raw samples are flattened directly
            # into the feature vector instead of running feature extraction.
            window_result = create_windows_filtered(raw_aligned, fs_dict, labels, config.TARGET_FS, WINDOW_SEC, OVERLAP)
            frames.append(windows_to_raw_flattened_features(window_result, subject, config))
        elif variant == "filtered":
            filtered_aligned, _ = align_signals(filter_wrist_signals(wrist, native_fs), subject_data["labels"], config.NATIVE_LABEL_FS, config.TARGET_FS)
            window_result = create_windows_filtered(filtered_aligned, fs_dict, labels, config.TARGET_FS, WINDOW_SEC, OVERLAP)
            frames.append(windows_to_features(window_result, subject, config))
        elif variant == "filtered_normalized":
            filtered_aligned, _ = align_signals(filter_wrist_signals(wrist, native_fs), subject_data["labels"], config.NATIVE_LABEL_FS, config.TARGET_FS)
            normalized_aligned = normalize_per_subject(filtered_aligned)
            window_result = create_windows_normalized(normalized_aligned, fs_dict, labels, config.TARGET_FS, WINDOW_SEC, OVERLAP)
            frames.append(windows_to_features(window_result, subject, config))
        else:
            raise ValueError(f"Unknown variant {variant}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _feature_matrix(features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    meta_cols = {"subject", "window_index", "label_original", "stress_label"}
    columns = [c for c in features.columns if c not in meta_cols]
    X = features[columns].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float64)
    y = features["stress_label"].to_numpy(dtype=np.int32)
    groups = features["subject"].to_numpy()
    return X, y, groups, columns


def _confusion_plot(y_true, y_pred, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Non-stress", "Stress"], yticklabels=["Non-stress", "Stress"])
    ax.set(xlabel="Predicted", ylabel="True", title=title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# --------------------------------------------------------------------------
# Validation scheme 1: GroupKFold cross-validation
# --------------------------------------------------------------------------

def run_group_kfold_cv(features: pd.DataFrame, model_name: str, model_factory, variant: str, output_dir: Path) -> dict:
    """Grouped K-fold CV (by subject), scaled+imputed inside a Pipeline so
    nothing from a held-out subject leaks into fitting the scaler/imputer."""
    X, y, groups, columns = _feature_matrix(features)
    n_groups = len(np.unique(groups))
    n_splits = min(N_CV_FOLDS, n_groups)
    if n_splits < 2:
        print(f"[skip cv] {variant}/{model_name}: need at least 2 subjects, have {n_groups}")
        return {}

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model_factory()),
    ])
    cv = GroupKFold(n_splits=n_splits)
    scoring = {"f1": "f1", "balanced_accuracy": "balanced_accuracy", "precision": "precision", "recall": "recall"}
    scores = cross_validate(pipeline, X, y, groups=groups, cv=cv, scoring=scoring, n_jobs=1, return_train_score=False)

    # Pooled out-of-fold predictions, purely for a confusion matrix / report.
    y_pred = cross_val_predict(pipeline, X, y, groups=groups, cv=cv, n_jobs=1)
    report = classification_report(y, y_pred, target_names=["non_stress", "stress"], output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(output_dir / f"cv_classification_report_{model_name}.csv")
    _confusion_plot(y, y_pred, f"{variant}/{model_name}: pooled GroupKFold confusion matrix",
                     output_dir / f"cv_confusion_matrix_{model_name}.png")

    fold_df = pd.DataFrame({
        "fold": range(1, n_splits + 1),
        "f1": scores["test_f1"], "balanced_accuracy": scores["test_balanced_accuracy"],
        "precision": scores["test_precision"], "recall": scores["test_recall"],
    })
    fold_df.to_csv(output_dir / f"cv_fold_results_{model_name}.csv", index=False)

    summary = {
        "variant": variant, "model": model_name, "n_cv_folds": n_splits,
        "cv_f1_mean": scores["test_f1"].mean(), "cv_f1_std": scores["test_f1"].std(),
        "cv_balanced_accuracy_mean": scores["test_balanced_accuracy"].mean(),
        "cv_balanced_accuracy_std": scores["test_balanced_accuracy"].std(),
        "cv_precision_mean": scores["test_precision"].mean(), "cv_recall_mean": scores["test_recall"].mean(),
    }
    print(f"[cv] {variant}/{model_name}: f1={summary['cv_f1_mean']:.3f}±{summary['cv_f1_std']:.3f}  "
          f"balanced_acc={summary['cv_balanced_accuracy_mean']:.3f}±{summary['cv_balanced_accuracy_std']:.3f}")
    return summary


# --------------------------------------------------------------------------
# Validation algorithm 2: Leave-One-Subject-Out (run on the TRAIN POOL only)
# --------------------------------------------------------------------------

def run_loso_validation(features: pd.DataFrame, model_name: str, model_factory, variant: str, output_dir: Path) -> dict:
    """Within the training pool, every subject is held out exactly once,
    trained only on the remaining training-pool subjects. This never sees
    the locked-away test set -- it is purely a validation signal used to
    sanity-check the model before the single final test in step 3."""
    X, y, groups, columns = _feature_matrix(features)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    logo = LeaveOneGroupOut()
    fold_rows = []
    all_true, all_pred, all_prob = [], [], []

    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups), start=1):
        held_out = np.unique(groups[test_idx])[0]
        if len(np.unique(y[train_idx])) < 2:
            print(f"[skip fold] {held_out}: only one class in training data")
            continue

        scaler = StandardScaler()
        imputer = SimpleImputer(strategy="median")
        X_train = scaler.fit_transform(imputer.fit_transform(X[train_idx]))
        X_test = scaler.transform(imputer.transform(X[test_idx]))
        y_train, y_test = y[train_idx], y[test_idx]

        model = model_factory()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else pred.astype(float)

        # Dummy baseline on the SAME fold, so it's a fair side-by-side.
        dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
        dummy_acc = dummy.score(X_test, y_test)

        bal_acc = balanced_accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, zero_division=0)
        try:
            auc = roc_auc_score(y_test, prob) if len(np.unique(y_test)) > 1 else float("nan")
        except ValueError:
            auc = float("nan")

        fold_rows.append({
            "fold": fold_idx, "held_out_subject": held_out, "n_test_windows": len(test_idx),
            "test_accuracy": float(np.mean(pred == y_test)), "dummy_baseline_accuracy": dummy_acc,
            "balanced_accuracy": bal_acc, "f1_stress": f1, "roc_auc": auc,
        })
        all_true.append(y_test); all_pred.append(pred); all_prob.append(prob)
        print(f"[loso fold {fold_idx}] held-out={held_out}: acc={np.mean(pred==y_test):.3f} "
              f"(dummy={dummy_acc:.3f})  balanced_acc={bal_acc:.3f}  f1={f1:.3f}  auc={auc:.3f}")

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(output_dir / f"loso_validation_fold_results_{model_name}.csv", index=False)

    y_true_all = np.concatenate(all_true)
    y_pred_all = np.concatenate(all_pred)
    report = classification_report(y_true_all, y_pred_all, target_names=["non_stress", "stress"], output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(output_dir / f"loso_validation_classification_report_{model_name}.csv")
    _confusion_plot(y_true_all, y_pred_all, f"{variant}/{model_name}: pooled LOSO validation confusion matrix (train pool only)",
                     output_dir / f"loso_validation_confusion_matrix_{model_name}.png")

    summary = {
        "variant": variant, "model": model_name,
        "loso_mean_accuracy": fold_df["test_accuracy"].mean(), "loso_std_accuracy": fold_df["test_accuracy"].std(),
        "loso_mean_dummy_baseline": fold_df["dummy_baseline_accuracy"].mean(),
        "loso_mean_balanced_accuracy": fold_df["balanced_accuracy"].mean(),
        "loso_std_balanced_accuracy": fold_df["balanced_accuracy"].std(),
        "loso_mean_f1": fold_df["f1_stress"].mean(), "loso_mean_roc_auc": fold_df["roc_auc"].mean(),
        "n_train_pool_subjects": fold_df["held_out_subject"].nunique(),
    }
    print(f"[loso validation] {variant}/{model_name}: balanced_acc={summary['loso_mean_balanced_accuracy']:.3f}  "
          f"f1={summary['loso_mean_f1']:.3f}  (dummy={summary['loso_mean_dummy_baseline']:.3f})")
    return summary


# --------------------------------------------------------------------------
# Step 3: final TEST on the locked-away held-out subjects
# --------------------------------------------------------------------------

def run_final_test(train_features: pd.DataFrame, test_features: pd.DataFrame, all_features: pd.DataFrame,
                    model_name: str, model_factory, variant: str, output_dir: Path) -> dict:
    """Fit one model on the full training pool and evaluate it exactly once
    on the locked-away test subjects -- the only metrics in this pipeline
    that reflect truly unseen data untouched by validation/tuning.

    Afterwards, a separate deployment model is trained on ALL subjects
    (train + test) and saved -- for actual downstream use, not for
    reporting performance."""
    X_train, y_train, groups_train, columns = _feature_matrix(train_features)
    X_test, y_test, groups_test, _ = _feature_matrix(test_features)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train_t = scaler.fit_transform(imputer.fit_transform(X_train))
    X_test_t = scaler.transform(imputer.transform(X_test))

    model = model_factory()
    model.fit(X_train_t, y_train)
    pred = model.predict(X_test_t)
    prob = model.predict_proba(X_test_t)[:, 1] if hasattr(model, "predict_proba") else pred.astype(float)

    dummy = DummyClassifier(strategy="most_frequent").fit(X_train_t, y_train)
    dummy_acc = dummy.score(X_test_t, y_test)

    bal_acc = balanced_accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, zero_division=0)
    precision = precision_score(y_test, pred, zero_division=0)
    recall = recall_score(y_test, pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test, prob) if len(np.unique(y_test)) > 1 else float("nan")
    except ValueError:
        auc = float("nan")

    report = classification_report(y_test, pred, target_names=["non_stress", "stress"], output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(output_dir / f"test_classification_report_{model_name}.csv")
    _confusion_plot(y_test, pred, f"{variant}/{model_name}: held-out TEST confusion matrix",
                     output_dir / f"test_confusion_matrix_{model_name}.png")

    predictions = test_features[["subject", "window_index", "label_original", "stress_label"]].copy()
    predictions["predicted_stress"] = pred
    predictions["correct"] = pred == y_test
    if hasattr(model, "predict_proba"):
        predictions["stress_probability"] = prob
    predictions.to_csv(output_dir / f"test_held_out_predictions_{model_name}.csv", index=False)

    summary = {
        "test_accuracy": float(np.mean(pred == y_test)), "test_balanced_accuracy": bal_acc,
        "test_f1": f1, "test_precision": precision, "test_recall": recall, "test_roc_auc": auc,
        "test_dummy_baseline_accuracy": dummy_acc,
        "test_subjects": ",".join(sorted(set(groups_test))),
        "train_pool_subjects": ",".join(sorted(set(groups_train))),
    }
    print(f"[test] {variant}/{model_name}: balanced_acc={bal_acc:.3f}  f1={f1:.3f}  "
          f"auc={auc:.3f}  (dummy={dummy_acc:.3f})  [held-out subjects: {summary['test_subjects']}]")

    # Separate deployment model trained on ALL subjects (train + test),
    # saved for downstream use -- NOT what the metrics above describe.
    X_all, y_all, _, _ = _feature_matrix(all_features)
    final_imputer = SimpleImputer(strategy="median")
    final_scaler = StandardScaler()
    X_all_t = final_scaler.fit_transform(final_imputer.fit_transform(X_all))
    final_model = model_factory()
    final_model.fit(X_all_t, y_all)
    joblib.dump(final_model, output_dir / f"{model_name}_deployment_model_trained_on_all_subjects.joblib")
    joblib.dump(final_scaler, output_dir / "feature_scaler.joblib")
    joblib.dump(final_imputer, output_dir / "feature_imputer.joblib")
    pd.DataFrame({"feature_name": columns}).to_csv(output_dir / "feature_columns.csv", index=False)

    return summary


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def run_model(model_name: str, model_factory) -> None:
    """Entry point called by each thin per-model script. Runs the 3-step
    process (split -> validate -> test) for every requested variant.

    model_factory: zero-arg callable returning a *fresh, unfitted*
    estimator instance (called once per CV/LOSO/test fit, so it must not
    share state across calls).
    """
    parser = argparse.ArgumentParser(description=f"Train/validate/test {model_name} with GroupKFold + LOSO validation and a locked held-out test set, on WESAD.")
    parser.add_argument("--dataset", default="wesad", choices=DATASET_REGISTRY.keys())
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--variants", nargs="+", default=["raw", "filtered", "filtered_normalized", "raw_windows"])
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of subjects locked away as the final test set.")
    args = parser.parse_args()

    dataset = DATASET_REGISTRY[args.dataset]
    config = dataset["config"]
    output_root = PROJECT_ROOT / "07_classical_ml" / "outputs" / model_name / config.OUTPUT_DIRECTORY_NAME
    subjects = args.subjects or dataset["loader"].list_subjects(config.DATA_ROOT)

    if not config.DATA_ROOT.exists() or not subjects:
        raise SystemExit(f"No {args.dataset} subjects found under {config.DATA_ROOT}. Point the dataset root at the data first.")

    output_root.mkdir(parents=True, exist_ok=True)
    print(f"Model: {model_name}\nDataset: {args.dataset}; subjects: {subjects}")

    all_summaries = []
    for variant in args.variants:
        features = build_features(subjects, variant, dataset)
        if features.empty:
            print(f"[skip] no features built for variant={variant}")
            continue

        output_dir = output_root / variant
        output_dir.mkdir(parents=True, exist_ok=True)
        features.to_csv(output_dir / "features_all_subjects.csv", index=False)

        X, y, groups, _ = _feature_matrix(features)
        n_subjects = len(np.unique(groups))
        if n_subjects < 6:
            raise ValueError(f"variant={variant}: at least six subjects are required for a meaningful train/validate/test split (have {n_subjects}).")

        # -- Step 1: TRAIN/TEST SPLIT --------------------------------------
        splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=RANDOM_SEED)
        train_idx, test_idx = next(splitter.split(X, y, groups))
        train_features = features.iloc[train_idx].reset_index(drop=True)
        test_features = features.iloc[test_idx].reset_index(drop=True)
        print(f"\n=== variant={variant} model={model_name}: STEP 1/3 -- train/test split "
              f"({len(set(groups[train_idx]))} train subjects, {len(set(groups[test_idx]))} test subjects, locked) ===")

        # -- Step 2: VALIDATE (train pool only) -----------------------------
        print(f"\n=== variant={variant} model={model_name}: STEP 2/3a -- GroupKFold validation (train pool) ===")
        cv_summary = run_group_kfold_cv(train_features, model_name, model_factory, variant, output_dir)

        print(f"\n=== variant={variant} model={model_name}: STEP 2/3b -- LOSO validation (train pool) ===")
        loso_summary = run_loso_validation(train_features, model_name, model_factory, variant, output_dir)

        # -- Step 3: TEST (locked-away subjects, evaluated once) -----------
        print(f"\n=== variant={variant} model={model_name}: STEP 3/3 -- final train + held-out test ===")
        test_summary = run_final_test(train_features, test_features, features, model_name, model_factory, variant, output_dir)

        all_summaries.append({**cv_summary, **loso_summary, **test_summary})

    if not all_summaries:
        raise SystemExit("No variant produced results -- nothing to summarize.")

    summary_df = pd.DataFrame(all_summaries).sort_values("test_balanced_accuracy", ascending=False)
    summary_df.to_csv(output_root / f"{model_name}_summary.csv", index=False)
    print(f"\n=== {model_name}: summary across variants (validation + test) ===")
    print(summary_df.to_string(index=False))
    print(
        "\nNote: 'test_*' columns are the single held-out-subject numbers from step 3 -- the "
        "ones to trust, since those subjects were never seen during validation. 'cv_*' and "
        "'loso_mean_*' are validation-only signals computed on the training pool, useful for "
        "sanity-checking and comparing variants/models but not a substitute for the test metrics. "
        "'*_dummy_baseline*' shows what always guessing the majority class scores -- the real "
        "model must clear that by a meaningful margin."
    )
