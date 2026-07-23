"""Contract implemented by dataset-specific loaders."""

from pathlib import Path


def load_subject(subject_id: str, root: Path) -> dict:
    """Return wrist signals, labels, and native sampling rates for one subject.

    The returned mapping has the shape ``{"wrist_signals": {...}, "labels":
    np.ndarray, "native_fs": {...}}``.
    """
    raise NotImplementedError


def list_subjects(root: Path) -> list[str]:
    """Return the available subject identifiers under ``root``."""
    raise NotImplementedError
