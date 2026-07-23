"""Train + evaluate XGBoost on WESAD wrist signals.

Runs the 3-step train/validate/test process for every requested
preprocessing variant: locked test-set split, GroupKFold + LOSO
validation on the training pool, then a single final held-out test.
See model_common.py for the shared pipeline.

Requires xgboost: pip install xgboost --break-system-packages

Run from the project root:
    python 07_classical_ml/train_xgboost.py
    python 07_classical_ml/train_xgboost.py --subjects S2 S3 S4 S5 S6 S7
    python 07_classical_ml/train_xgboost.py --variants filtered_normalized
"""

from model_common import RANDOM_SEED, run_model

MODEL_NAME = "xgboost"

try:
    from xgboost import XGBClassifier
except ImportError as exc:
    raise SystemExit(
        "xgboost is not installed. Run: pip install xgboost --break-system-packages"
    ) from exc


def model_factory() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, eval_metric="logloss", random_state=RANDOM_SEED, n_jobs=-1,
    )


if __name__ == "__main__":
    run_model(MODEL_NAME, model_factory)
