"""Compatibility launcher for :mod:`plot_analysis`."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(Path(__file__).with_name("plot_analysis.py"), run_name="__main__")
