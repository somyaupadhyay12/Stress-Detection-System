"""Aggregate every model's results and produce comparison plots + tables.

Run this AFTER training all six models (train_logistic_regression.py,
train_svm.py, train_random_forest.py, train_knn.py, train_xgboost.py,
train_decision_tree.py) -- it reads each model's `<model>_summary.csv`
(written by model_common.run_model) and builds:

1. Model comparison  -- which of the 6 models performs best (on the
   locked TEST set), per metric.
2. Variant comparison -- which preprocessing variant (raw / filtered /
   filtered_normalized / raw_windows) works best, per model and overall.
3. Validation-vs-test consistency -- whether GroupKFold CV / LOSO
   validation scores (computed on the training pool) over- or
   under-estimate the real held-out TEST performance, per model.
4. Margin-over-dummy-baseline -- how much every model beats a
   majority-class classifier on the TEST set.
5. A single ranked leaderboard (CSV) of every model x variant
   combination, sorted by test balanced accuracy.

Run from the project root:
    python 07_classical_ml/compare_all_models.py
    python 07_classical_ml/compare_all_models.py --models random_forest svm xgboost
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from model_common import DATASET_REGISTRY, PROJECT_ROOT

sns.set_theme(style="whitegrid")

ALL_MODELS = ["logistic_regression", "svm", "random_forest", "knn", "xgboost", "decision_tree"]
VARIANT_ORDER = ["raw", "filtered", "filtered_normalized", "raw_windows"]

TEST_METRICS = {
    "test_balanced_accuracy": "Test balanced accuracy",
    "test_f1": "Test F1 (stress)",
    "test_precision": "Test precision",
    "test_recall": "Test recall",
    "test_roc_auc": "Test ROC-AUC",
    "test_accuracy": "Test accuracy (raw)",
}


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_all_summaries(models: list[str], dataset_key: str) -> pd.DataFrame:
    config = DATASET_REGISTRY[dataset_key]["config"]
    frames = []
    for model_name in models:
        path = PROJECT_ROOT / "07_classical_ml" / "outputs" / model_name / config.OUTPUT_DIRECTORY_NAME / f"{model_name}_summary.csv"
        if not path.exists():
            print(f"[skip] {model_name}: no summary found at {path} -- run train_{model_name}.py first.")
            continue
        df = pd.read_csv(path)
        df["model"] = model_name
        frames.append(df)
    if not frames:
        raise SystemExit("No model summaries found. Train at least one model before comparing.")
    combined = pd.concat(frames, ignore_index=True)
    combined["variant"] = pd.Categorical(combined["variant"], categories=VARIANT_ORDER, ordered=True)
    return combined.sort_values(["model", "variant"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------

def plot_metric_by_model_and_variant(df: pd.DataFrame, metric: str, label: str, out_dir: Path) -> None:
    """Grouped bar chart: one group per model, one bar per variant. This is
    the main 'model comparison' AND 'variant comparison' plot at once."""
    pivot = df.pivot_table(index="model", columns="variant", values=metric, observed=True)
    pivot = pivot.reindex(columns=[v for v in VARIANT_ORDER if v in pivot.columns])
    pivot = pivot.loc[pivot.max(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="bar", ax=ax, width=0.75)
    ax.set_ylabel(label)
    ax.set_xlabel("Model")
    ax.set_title(f"{label}: model vs. preprocessing variant (TEST set)")
    ax.legend(title="Variant")
    ax.set_ylim(0, 1.0)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / f"compare_{metric}_by_model_variant.png", dpi=180)
    plt.close(fig)


def plot_heatmap(df: pd.DataFrame, metric: str, label: str, out_dir: Path) -> None:
    pivot = df.pivot_table(index="model", columns="variant", values=metric, observed=True)
    pivot = pivot.reindex(columns=[v for v in VARIANT_ORDER if v in pivot.columns])
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", vmin=0, vmax=1, ax=ax, cbar_kws={"label": label})
    ax.set_title(f"{label}: heatmap (TEST set)")
    ax.set_xlabel("Variant")
    ax.set_ylabel("Model")
    fig.tight_layout()
    fig.savefig(out_dir / f"compare_{metric}_heatmap.png", dpi=180)
    plt.close(fig)


def plot_validation_vs_test(df: pd.DataFrame, out_dir: Path) -> None:
    """For balanced accuracy: CV validation vs LOSO validation vs final TEST,
    averaged across variants per model. Shows whether validation scores
    were optimistic (common) or pessimistic relative to the real test."""
    grouped = df.groupby("model", observed=True)[
        ["cv_balanced_accuracy_mean", "loso_mean_balanced_accuracy", "test_balanced_accuracy"]
    ].mean()
    grouped = grouped.rename(columns={
        "cv_balanced_accuracy_mean": "GroupKFold CV (validation)",
        "loso_mean_balanced_accuracy": "LOSO (validation)",
        "test_balanced_accuracy": "Held-out TEST",
    })
    grouped = grouped.loc[grouped["Held-out TEST"].sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(9, 5))
    grouped.plot(kind="bar", ax=ax, width=0.75)
    ax.set_ylabel("Balanced accuracy")
    ax.set_xlabel("Model")
    ax.set_title("Validation vs. TEST balanced accuracy (averaged across variants)")
    ax.set_ylim(0, 1.0)
    ax.legend(title=None)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "compare_validation_vs_test.png", dpi=180)
    plt.close(fig)


def plot_dummy_margin(df: pd.DataFrame, out_dir: Path) -> None:
    """How much each model/variant beats a majority-class dummy classifier
    on the TEST set. Bars near zero mean the model isn't really learning
    anything beyond the class imbalance."""
    plot_df = df.copy()
    plot_df["margin_over_dummy"] = plot_df["test_balanced_accuracy"] - 0.5  # balanced accuracy of "always majority class" is always 0.5
    pivot = plot_df.pivot_table(index="model", columns="variant", values="margin_over_dummy", observed=True)
    pivot = pivot.reindex(columns=[v for v in VARIANT_ORDER if v in pivot.columns])
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="bar", ax=ax, width=0.75)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Test balanced accuracy − 0.5\n(margin over chance)")
    ax.set_xlabel("Model")
    ax.set_title("Margin over chance-level (0.5 balanced accuracy) on TEST set")
    ax.legend(title="Variant")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "compare_margin_over_dummy.png", dpi=180)
    plt.close(fig)


def plot_all_metrics_best_variant(df: pd.DataFrame, out_dir: Path) -> None:
    """For each model's best variant (by test balanced accuracy), show all
    test metrics side by side -- a single 'report card' plot."""
    best_rows = df.loc[df.groupby("model", observed=True)["test_balanced_accuracy"].idxmax()]
    metrics = ["test_balanced_accuracy", "test_f1", "test_precision", "test_recall", "test_roc_auc"]
    plot_df = best_rows.set_index("model")[metrics]
    plot_df = plot_df.loc[plot_df["test_balanced_accuracy"].sort_values(ascending=False).index]
    plot_df.columns = ["Balanced acc.", "F1", "Precision", "Recall", "ROC-AUC"]

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_df.plot(kind="bar", ax=ax, width=0.8)
    ax.set_ylabel("Score")
    ax.set_xlabel("Model (best variant)")
    ax.set_title("All TEST metrics, best variant per model")
    ax.set_ylim(0, 1.0)
    ax.legend(title=None, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.32))
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "compare_all_metrics_best_variant.png", dpi=180)
    plt.close(fig)

    best_rows[["model", "variant"] + metrics].to_csv(out_dir / "best_variant_per_model.csv", index=False)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Compare results across all trained models and preprocessing variants.")
    parser.add_argument("--dataset", default="wesad", choices=DATASET_REGISTRY.keys())
    parser.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    args = parser.parse_args()

    df = load_all_summaries(args.models, args.dataset)

    config = DATASET_REGISTRY[args.dataset]["config"]
    out_dir = PROJECT_ROOT / "07_classical_ml" / "outputs" / "comparison" / config.OUTPUT_DIRECTORY_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_dir / "all_results_combined.csv", index=False)

    for metric, label in TEST_METRICS.items():
        if metric not in df.columns:
            continue
        plot_metric_by_model_and_variant(df, metric, label, out_dir)
        plot_heatmap(df, metric, label, out_dir)

    plot_validation_vs_test(df, out_dir)
    plot_dummy_margin(df, out_dir)
    plot_all_metrics_best_variant(df, out_dir)

    leaderboard = df.sort_values("test_balanced_accuracy", ascending=False)[
        ["model", "variant", "test_balanced_accuracy", "test_f1", "test_precision", "test_recall",
         "test_roc_auc", "test_dummy_baseline_accuracy", "loso_mean_balanced_accuracy", "cv_balanced_accuracy_mean"]
    ]
    leaderboard.to_csv(out_dir / "leaderboard.csv", index=False)

    print(f"\nSaved comparison outputs to: {out_dir}")
    print("\n=== Leaderboard (sorted by TEST balanced accuracy) ===")
    print(leaderboard.to_string(index=False))
    best = leaderboard.iloc[0]
    print(
        f"\nBest overall: {best['model']} / {best['variant']} "
        f"-- test balanced accuracy {best['test_balanced_accuracy']:.3f}, "
        f"test F1 {best['test_f1']:.3f} "
        f"(dummy baseline {best['test_dummy_baseline_accuracy']:.3f})."
    )


if __name__ == "__main__":
    main()
