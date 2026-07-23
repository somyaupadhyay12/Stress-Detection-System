"""Train + evaluate an RBF-kernel SVM on WESAD wrist signals.

Runs the 3-step train/validate/test process for every requested
preprocessing variant: locked test-set split, GroupKFold + LOSO
validation on the training pool, then a single final held-out test.
See model_common.py for the shared pipeline.

Run from the project root:
    python 07_classical_ml/train_svm.py
    python 07_classical_ml/train_svm.py --subjects S2 S3 S4 S5 S6 S7
    python 07_classical_ml/train_svm.py --variants filtered_normalized
"""

from sklearn.svm import SVC

from model_common import RANDOM_SEED, run_model

MODEL_NAME = "svm"


def model_factory() -> SVC:
    # probability=True is required so predict_proba exists for ROC-AUC downstream.
    return SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", probability=True, random_state=RANDOM_SEED)


if __name__ == "__main__":
    run_model(MODEL_NAME, model_factory)
