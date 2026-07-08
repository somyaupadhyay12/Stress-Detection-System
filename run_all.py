"""
Run the Stress Detection preprocessing pipeline with one command.

Usage:
    python run_all.py

All run outputs are collected in the project-level outputs/ folder. Each
notebook gets its own subfolder containing an executed notebook copy and a log.
Generated data still goes to Data/preprocessed_data/ because that is the
project data folder used by the notebooks.
"""

from __future__ import annotations

import base64
import contextlib
import os
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
JUPYTER_RUNTIME_DIR = OUTPUTS_DIR / "jupyter_runtime"
JUPYTER_RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(JUPYTER_RUNTIME_DIR))
os.environ.setdefault("JUPYTER_ALLOW_INSECURE_WRITES", "1")

import nbformat
from nbclient import NotebookClient


PREPROCESSING_DIR = PROJECT_ROOT / "preprocessing"

PIPELINE_NOTEBOOKS = [
    PREPROCESSING_DIR / "01_filtering.ipynb",
    PREPROCESSING_DIR / "02_normalization.ipynb",
    PREPROCESSING_DIR / "03_raw_windowing.ipynb",
    PREPROCESSING_DIR / "04_windowing_filtered.ipynb",
    PREPROCESSING_DIR / "05_windowing_normalised.ipynb",
]


def safe_folder_name(path: Path) -> str:
    return path.stem.replace(" ", "_")


def write_log(log_path: Path, message: str) -> None:
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(message + "\n")


def slugify(value: str) -> str:
    value = value.lower().replace("?", "_").replace("-", "_")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "plot"


def infer_plot_names(source: str) -> list[str]:
    """Infer readable plot filenames from plt.title/set_title calls."""
    names: list[str] = []
    title_patterns = [
        r'plt\.title\(\s*[rRuUfFbB]*[\'"]([^\'"]+)',
        r'\.set_title\(\s*[rRuUfFbB]*[\'"]([^\'"]+)',
    ]
    for pattern in title_patterns:
        names.extend(re.findall(pattern, source))

    header = ""
    for line in source.splitlines():
        stripped = line.strip(" #")
        if stripped and any(word in stripped.lower() for word in ["empatica", "respiban", "respi", "e4"]):
            header = stripped
            break

    prefix = ""
    lower_header = header.lower()
    if "respiban" in lower_header or "respi" in lower_header:
        prefix = "respiban"
    elif "empatica" in lower_header or "e4" in lower_header:
        prefix = "empatica"

    readable = []
    for name in names:
        slug = slugify(name)
        if prefix and not slug.startswith(prefix):
            slug = f"{prefix}_{slug}"
        readable.append(slug)
    return readable


def save_embedded_png_outputs(notebook, output_dir: Path) -> int:
    """Save image/png outputs from executed notebook cells into figures/."""
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    used_names: set[str] = set()
    for cell_index, cell in enumerate(notebook.cells, start=1):
        if cell.get("cell_type") != "code":
            continue
        inferred_names = infer_plot_names(cell.get("source", ""))
        image_number = 0
        for output_index, output in enumerate(cell.get("outputs", []), start=1):
            data = output.get("data", {}) if isinstance(output, dict) else {}
            png_data = data.get("image/png")
            if not png_data:
                continue
            if isinstance(png_data, list):
                png_data = "".join(png_data)
            count += 1
            image_number += 1
            if image_number <= len(inferred_names):
                base_name = inferred_names[image_number - 1]
            else:
                base_name = f"notebook_cell_{cell_index:03d}_output_{output_index:02d}"
            final_name = base_name
            duplicate_index = 2
            while final_name in used_names:
                final_name = f"{base_name}_{duplicate_index}"
                duplicate_index += 1
            used_names.add(final_name)
            png_path = figures_dir / f"{final_name}.png"
            png_path.write_bytes(base64.b64decode(png_data))
    return count


def run_notebook(notebook_path: Path) -> Path:
    relative_name = notebook_path.relative_to(PROJECT_ROOT)
    output_dir = OUTPUTS_DIR / safe_folder_name(notebook_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "run.log"
    executed_path = output_dir / notebook_path.name

    log_path.write_text(
        f"Started: {datetime.now().isoformat(timespec='seconds')}\n"
        f"Notebook: {relative_name}\n"
        f"Working folder: {notebook_path.parent}\n\n",
        encoding="utf-8",
    )

    print(f"\nRunning {relative_name}")
    print(f"Output folder: {output_dir.relative_to(PROJECT_ROOT)}")

    if not notebook_path.exists():
        raise FileNotFoundError(f"Missing notebook: {notebook_path}")

    with notebook_path.open("r", encoding="utf-8-sig") as f:
        notebook = nbformat.read(f, as_version=4)

    client = NotebookClient(
        notebook,
        timeout=None,
        kernel_name="python3",
        resources={"metadata": {"path": str(notebook_path.parent)}},
    )

    try:
        with log_path.open("a", encoding="utf-8") as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                client.execute()
    except Exception:
        write_log(log_path, "\nFAILED")
        write_log(log_path, traceback.format_exc())
        with executed_path.open("w", encoding="utf-8") as f:
            nbformat.write(notebook, f)
        raise

    with executed_path.open("w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    png_count = save_embedded_png_outputs(notebook, output_dir)
    write_log(log_path, f"\nCompleted: {datetime.now().isoformat(timespec='seconds')}")
    write_log(log_path, f"Saved embedded PNG outputs: {png_count}")
    print(f"Saved executed notebook: {executed_path.relative_to(PROJECT_ROOT)}")
    print(f"Saved embedded PNG outputs: {png_count}")
    print(f"Saved log: {log_path.relative_to(PROJECT_ROOT)}")
    return output_dir


def write_manifest(completed_outputs: list[tuple[Path, Path]]) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUTS_DIR / "MANIFEST.txt"

    lines = [
        "Stress Detection Pipeline Outputs",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Data folders:",
        f"- Raw data: {PROJECT_ROOT / 'Data' / 'raw'}",
        f"- Preprocessed data: {PROJECT_ROOT / 'Data' / 'preprocessed_data'}",
        "",
        "Code file outputs:",
    ]

    for notebook_path, output_dir in completed_outputs:
        lines.append(f"- {notebook_path.relative_to(PROJECT_ROOT)} -> {output_dir.relative_to(PROJECT_ROOT)}")

    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nOutput manifest: {manifest_path.relative_to(PROJECT_ROOT)}")


def main() -> int:
    os.chdir(PROJECT_ROOT)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Stress Detection pipeline runner")
    print(f"Project: {PROJECT_ROOT}")
    print(f"Raw data: {PROJECT_ROOT / 'Data' / 'raw'}")
    print(f"Preprocessed data: {PROJECT_ROOT / 'Data' / 'preprocessed_data'}")
    print(f"All run outputs: {OUTPUTS_DIR}")

    completed_outputs: list[tuple[Path, Path]] = []

    try:
        for notebook_path in PIPELINE_NOTEBOOKS:
            output_dir = run_notebook(notebook_path)
            completed_outputs.append((notebook_path, output_dir))
    except Exception as exc:
        write_manifest(completed_outputs)
        print(f"\nPipeline stopped: {exc}", file=sys.stderr)
        return 1

    write_manifest(completed_outputs)
    print("\nPipeline completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



