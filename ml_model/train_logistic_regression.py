"""Train + evaluate Logistic Regression on WESAD wrist signals.

Runs the 3-step train/validate/test process for every requested
preprocessing variant: locked test-set split, GroupKFold + LOSO
validation on the training pool, then a single final held-out test.
See model_common.py for the shared pipeline.

Run from the project root:
    python 07_classical_ml/train_logistic_regression.py
    python 07_classical_ml/train_logistic_regression.py --subjects S2 S3 S4 S5 S6 S7
    python 07_classical_ml/train_logistic_regression.py --variants filtered_normalized
"""

from sklearn.linear_model import LogisticRegression

from model_common import RANDOM_SEED, run_model

MODEL_NAME = "logistic_regression"


def model_factory() -> LogisticRegression:
    return LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0, random_state=RANDOM_SEED)


if __name__ == "__main__":
    run_model(MODEL_NAME, model_factory)
