"""Constants for the WESAD wrist-signal dataset."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "Data" / "Raw" / "WESAD"
# Backward-compatible alias for scripts that previously used this name.
WESAD_ROOT = DATA_ROOT
OUTPUT_DIRECTORY_NAME = "WESAD"

BINARY_LABELS = {1: 0, 2: 1, 3: 0}
VALID_LABELS = set(BINARY_LABELS)
SIGNAL_TYPE_MAP = {"ACC": "acc", "BVP": "bvp", "EDA": "eda", "TEMP": "temp"}
TARGET_FS = 4
NATIVE_LABEL_FS = 700
