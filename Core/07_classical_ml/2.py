"""Compatibility launcher for :mod:`train_robust_model`.

Use ``train_robust_model.py`` for new commands; this file preserves the
previous entry point without duplicating the dataset-agnostic training code.
"""

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(Path(__file__).with_name("train_robust_model.py"), run_name="__main__")
