"""Train + evaluate a Random Forest on WESAD wrist signals.

Runs the 3-step train/validate/test process for every requested
preprocessing variant: locked test-set split, GroupKFold + LOSO
validation on the training pool, then a single final held-out test.
See model_common.py for the shared pipeline.

Run from the project root:
    python 07_classical_ml/train_random_forest.py
    python 07_classical_ml/train_random_forest.py --subjects S2 S3 S4 S5 S6 S7
    python 07_classical_ml/train_random_forest.py --variants filtered_normalized
"""

from sklearn.ensemble import RandomForestClassifier

from model_common import RANDOM_SEED, run_model

MODEL_NAME = "random_forest"


def model_factory() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=400, max_depth=12, min_samples_leaf=3,
        class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1,
    )


if __name__ == "__main__":
    run_model(MODEL_NAME, model_factory)
