"""Train + evaluate a K-Nearest-Neighbors classifier on WESAD wrist signals.

Runs the 3-step train/validate/test process for every requested
preprocessing variant: locked test-set split, GroupKFold + LOSO
validation on the training pool, then a single final held-out test.
See model_common.py for the shared pipeline.

Run from the project root:
    python 07_classical_ml/train_knn.py
    python 07_classical_ml/train_knn.py --subjects S2 S3 S4 S5 S6 S7
    python 07_classical_ml/train_knn.py --variants filtered_normalized
"""

from sklearn.neighbors import KNeighborsClassifier

from model_common import run_model

MODEL_NAME = "knn"


def model_factory() -> KNeighborsClassifier:
    return KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1)


if __name__ == "__main__":
    run_model(MODEL_NAME, model_factory)
