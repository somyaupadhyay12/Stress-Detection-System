"""WESAD pickle-file loader."""

import pickle
from pathlib import Path

import numpy as np


WESAD_WRIST_NATIVE_FS = {"ACC": 32, "BVP": 64, "EDA": 4, "TEMP": 4}


def load_subject(subject_id: str, root: Path) -> dict:
    """Load one WESAD subject in the common loader contract."""
    source = root / subject_id / f"{subject_id}.pkl"
    with source.open("rb") as handle:
        record = pickle.load(handle, encoding="latin1")
    wrist = record["signal"]["wrist"]
    return {
        "wrist_signals": {name: np.asarray(wrist[name]) for name in WESAD_WRIST_NATIVE_FS},
        "labels": np.asarray(record["label"]),
        "native_fs": WESAD_WRIST_NATIVE_FS.copy(),
    }


def list_subjects(root: Path) -> list[str]:
    """Discover WESAD subject folders that contain their expected pickle."""
    return sorted(path.name for path in root.glob("S*") if (path / f"{path.name}.pkl").exists())
