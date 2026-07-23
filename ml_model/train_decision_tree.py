"""Train + evaluate a Decision Tree on WESAD wrist signals.

Runs the 3-step train/validate/test process for every requested
preprocessing variant: locked test-set split, GroupKFold + LOSO
validation on the training pool, then a single final held-out test.
See model_common.py for the shared pipeline.

Run from the project root:
    python 07_classical_ml/train_decision_tree.py
    python 07_classical_ml/train_decision_tree.py --subjects S2 S3 S4 S5 S6 S7
    python 07_classical_ml/train_decision_tree.py --variants filtered_normalized
"""

from sklearn.tree import DecisionTreeClassifier

from model_common import RANDOM_SEED, run_model

MODEL_NAME = "decision_tree"


def model_factory() -> DecisionTreeClassifier:
    return DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, class_weight="balanced", random_state=RANDOM_SEED)


if __name__ == "__main__":
    run_model(MODEL_NAME, model_factory)
