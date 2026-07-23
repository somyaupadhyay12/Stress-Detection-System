"""
Plotting suite for outputs produced by train_robust_model.py (WESAD LOSO training).

Run this AFTER train_robust_model.py has produced files under:
    07_classical_ml/outputs/robust_model/

It reads, per variant, per model (filenames are namespaced by model_name):
    <variant>/loso_fold_results_<model>.csv            (per-subject fold metrics)
    <variant>/pooled_classification_report_<model>.csv (precision/recall/f1 per class)
    <variant>/features_all_subjects.csv                 (raw feature table, shared across models)
    model_comparison_summary.csv                         (one row per variant x model)

And produces (saved as PNGs under outputs/robust_model/plots/):
    1. per_subject_metrics_<variant>_<model>.png   - grouped bar: acc/bal_acc/f1/auc per held-out subject, with dummy baseline
    2. metric_distributions_<variant>_<model>.png  - boxplots of each metric across folds
    3. model_vs_dummy_<variant>_<model>.png        - real model vs dummy baseline, per subject
    4. variant_model_comparison.png                - grouped bars comparing all variant x model combos
    5. variant_model_heatmap.png                   - heatmap of balanced accuracy across variant x model
    6. class_report_<variant>_<model>.png          - precision/recall/f1 per class, pooled
    7. feature_distributions_<variant>.png         - top discriminative features split by class (violin)
    8. feature_correlation_<variant>.png           - correlation heatmap of top features
    9. accuracy_vs_baseline_gap.png                - how much each config beats its dummy baseline

Per-fold detail plots (1, 2, 3, 6) are generated automatically for every model that has
output files present in a variant folder -- no need to pass --model, it discovers them.

Usage:
    python plot_analysis.py --root 07_classical_ml/outputs/robust_model \
        --variants raw filtered filtered_normalized
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

sns.set_theme(style="whitegrid", font_scale=1.0)


def _savefig(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")


# ---------------------------------------------------------------------------
# 1 & 2 & 3. Per-fold (per-subject) metric plots for a single variant/model
# ---------------------------------------------------------------------------

def plot_per_subject_metrics(fold_csv: Path, out_dir: Path, variant: str, model: str):
    df = pd.read_csv(fold_csv).sort_values("held_out_subject")

    # --- (1) grouped bar of all metrics per subject ---
    metrics = ["test_accuracy", "balanced_accuracy", "f1_stress", "roc_auc"]
    x = np.arange(len(df))
    width = 0.2
    fig, ax = plt.subplots(figsize=(max(8, len(df) * 0.9), 5))
    for i, m in enumerate(metrics):
        ax.bar(x + i * width, df[m], width, label=m.replace("_", " "))
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(df["held_out_subject"], rotation=0)
    ax.axhline(df["dummy_baseline_accuracy"].mean(), color="black", linestyle="--", linewidth=1,
               label="mean dummy baseline")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Held-out subject")
    ax.set_ylabel("Score")
    ax.set_title(f"{variant}/{model}: per-subject LOSO metrics")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=5)
    _savefig(fig, out_dir / f"per_subject_metrics_{variant}_{model}.png")

    # --- (2) boxplots of metric distributions across folds ---
    fig, ax = plt.subplots(figsize=(6, 5))
    melted = df[metrics].melt(var_name="metric", value_name="score")
    sns.boxplot(data=melted, x="metric", y="score", ax=ax)
    sns.stripplot(data=melted, x="metric", y="score", ax=ax, color="black", alpha=0.5, size=4)
    ax.set_xticks(range(len(metrics))); ax.set_xticklabels([m.replace("_", " ") for m in metrics], rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{variant}/{model}: metric spread across {len(df)} LOSO folds")
    _savefig(fig, out_dir / f"metric_distributions_{variant}_{model}.png")

    # --- (3) model vs dummy baseline, per subject ---
    fig, ax = plt.subplots(figsize=(max(8, len(df) * 0.9), 5))
    ax.bar(x - width / 2, df["test_accuracy"], width, label="model accuracy", color="#2E86AB")
    ax.bar(x + width / 2, df["dummy_baseline_accuracy"], width, label="dummy baseline", color="#C0C0C0")
    ax.set_xticks(x)
    ax.set_xticklabels(df["held_out_subject"])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Held-out subject")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{variant}/{model}: model vs. majority-class dummy baseline")
    ax.legend()
    _savefig(fig, out_dir / f"model_vs_dummy_{variant}_{model}.png")


# ---------------------------------------------------------------------------
# 4 & 5. Comparison across all variant x model combinations
# ---------------------------------------------------------------------------

def plot_variant_model_comparison(summary_csv: Path, out_dir: Path):
    df = pd.read_csv(summary_csv)
    df["config"] = df["variant"] + " / " + df["model"]
    df = df.sort_values("mean_balanced_accuracy", ascending=False)

    # --- (4) grouped bar comparison ---
    metrics = ["mean_accuracy", "mean_balanced_accuracy", "mean_f1_stress", "mean_roc_auc"]
    x = np.arange(len(df))
    width = 0.2
    fig, ax = plt.subplots(figsize=(max(9, len(df) * 1.1), 5.5))
    for i, m in enumerate(metrics):
        ax.bar(x + i * width, df[m], width, label=m.replace("mean_", "").replace("_", " "))
    ax.errorbar(x + 1 * width, df["mean_balanced_accuracy"], yerr=df["std_balanced_accuracy"],
                fmt="none", ecolor="black", capsize=3)
    ax.plot(x, df["mean_dummy_baseline"], "k--", marker="o", label="dummy baseline")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(df["config"], rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model / preprocessing-variant comparison (LOSO)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=5)
    _savefig(fig, out_dir / "variant_model_comparison.png")

    # --- (5) heatmap: variant vs model, balanced accuracy ---
    pivot = df.pivot(index="variant", columns="model", values="mean_balanced_accuracy")
    fig, ax = plt.subplots(figsize=(1.5 + 1.5 * pivot.shape[1], 1.5 + pivot.shape[0]))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", ax=ax, vmin=0.5, vmax=1.0,
                cbar_kws={"label": "mean balanced accuracy"})
    ax.set_title("Balanced accuracy: variant x model")
    _savefig(fig, out_dir / "variant_model_heatmap.png")

    # --- (9) gap over baseline ---
    df["gap_over_baseline"] = df["mean_balanced_accuracy"] - df["mean_dummy_baseline"]
    fig, ax = plt.subplots(figsize=(max(8, len(df) * 1.0), 5))
    colors = ["#2E86AB" if g > 0 else "#C0392B" for g in df["gap_over_baseline"]]
    ax.bar(df["config"], df["gap_over_baseline"], color=colors)
    ax.axhline(0, color="black", linewidth=1)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_ylabel("Balanced accuracy − dummy baseline")
    ax.set_title("How much each config beats guessing the majority class")
    _savefig(fig, out_dir / "accuracy_vs_baseline_gap.png")


# ---------------------------------------------------------------------------
# 6. Pooled classification report (precision/recall/f1 per class)
# ---------------------------------------------------------------------------

def plot_classification_report(report_csv: Path, out_dir: Path, variant: str, model: str):
    df = pd.read_csv(report_csv, index_col=0)
    classes = [c for c in df.index if c in ("non_stress", "stress")]
    metrics = ["precision", "recall", "f1-score"]
    sub = df.loc[classes, metrics]

    x = np.arange(len(classes))
    width = 0.25
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, m in enumerate(metrics):
        ax.bar(x + i * width, sub[m], width, label=m)
    ax.set_xticks(x + width)
    ax.set_xticklabels(classes)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{variant}/{model}: pooled precision / recall / F1 per class")
    ax.legend()
    _savefig(fig, out_dir / f"class_report_{variant}_{model}.png")


# ---------------------------------------------------------------------------
# 7 & 8. Feature-level exploration (distributions + correlation)
# ---------------------------------------------------------------------------

def plot_feature_analysis(features_csv: Path, out_dir: Path, variant: str, top_n: int = 12):
    df = pd.read_csv(features_csv)
    meta_cols = {"subject", "window_index", "label_original", "stress_label"}
    feature_cols = [c for c in df.columns if c not in meta_cols]

    # rank features by absolute mean difference between classes (simple, fast proxy
    # for discriminative power -- no model needed)
    g0 = df[df["stress_label"] == 0][feature_cols].mean()
    g1 = df[df["stress_label"] == 1][feature_cols].mean()
    spread = df[feature_cols].std().replace(0, np.nan)
    effect_size = ((g1 - g0).abs() / spread).sort_values(ascending=False)
    top_features = effect_size.head(top_n).index.tolist()

    # --- (7) violin plots of top features split by class ---
    melted = df[top_features + ["stress_label"]].melt(
        id_vars="stress_label", var_name="feature", value_name="value"
    )
    melted["class"] = melted["stress_label"].map({0: "non-stress", 1: "stress"})
    n_cols = 4
    n_rows = int(np.ceil(len(top_features) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.2 * n_rows))
    axes = np.atleast_1d(axes).flatten()
    for i, feat in enumerate(top_features):
        sub = melted[melted["feature"] == feat]
        sns.violinplot(data=sub, x="class", y="value", ax=axes[i], hue="class", legend=False)
        axes[i].set_title(feat, fontsize=9)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")
    for j in range(len(top_features), len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"{variant}: top {top_n} most class-separating features", y=1.02)
    _savefig(fig, out_dir / f"feature_distributions_{variant}.png")

    # --- (8) correlation heatmap of the same top features ---
    fig, ax = plt.subplots(figsize=(1 + 0.55 * top_n, 1 + 0.5 * top_n))
    corr = df[top_features].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax,
                annot_kws={"size": 7}, cbar_kws={"label": "Pearson r"})
    ax.set_title(f"{variant}: correlation among top {top_n} features")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    _savefig(fig, out_dir / f"feature_correlation_{variant}.png")

    # bonus: class balance per subject (helps interpret accuracy numbers)
    balance = df.groupby(["subject", "stress_label"]).size().unstack(fill_value=0)
    balance.columns = ["non_stress_windows", "stress_windows"]
    fig, ax = plt.subplots(figsize=(max(8, len(balance) * 0.8), 5))
    balance.plot(kind="bar", stacked=True, ax=ax, color=["#2E86AB", "#C0392B"])
    ax.set_ylabel("Window count")
    ax.set_title(f"{variant}: class balance per subject")
    ax.legend(title="")
    _savefig(fig, out_dir / f"class_balance_per_subject_{variant}.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def discover_models(variant_dir: Path) -> list[str]:
    """Find every model_name that has a loso_fold_results_<model>.csv in this
    variant folder, so we don't need the caller to list models manually."""
    models = []
    for path in sorted(variant_dir.glob("loso_fold_results_*.csv")):
        model_name = path.stem.replace("loso_fold_results_", "")
        models.append(model_name)
    return models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="wesad", help="Dataset namespace used by train_robust_model.py.")
    parser.add_argument("--root", default=None,
                        help="Optional output-root override; defaults to the selected dataset namespace.")
    parser.add_argument("--variants", nargs="+", default=["raw", "filtered", "filtered_normalized"])
    args = parser.parse_args()

    root = Path(args.root) if args.root else (
        Path("07_classical_ml") / "outputs" / "robust_model" / args.dataset.upper()
    )
    out_dir = root / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    for variant in args.variants:
        variant_dir = root / variant
        if not variant_dir.exists():
            print(f"[skip] {variant_dir} not found")
            continue

        models = discover_models(variant_dir)
        if not models:
            print(f"[skip] no loso_fold_results_*.csv found under {variant_dir}")
        for model_name in models:
            fold_csv = variant_dir / f"loso_fold_results_{model_name}.csv"
            report_csv = variant_dir / f"pooled_classification_report_{model_name}.csv"

            plot_per_subject_metrics(fold_csv, out_dir, variant, model_name)

            if report_csv.exists():
                plot_classification_report(report_csv, out_dir, variant, model_name)
            else:
                print(f"[skip] {report_csv} not found")

        features_csv = variant_dir / "features_all_subjects.csv"
        if features_csv.exists():
            plot_feature_analysis(features_csv, out_dir, variant)
        else:
            print(f"[skip] {features_csv} not found")

    summary_csv = root / "model_comparison_summary.csv"
    if summary_csv.exists():
        plot_variant_model_comparison(summary_csv, out_dir)
    else:
        print(f"[skip] {summary_csv} not found -- run 1.py with --compare-models or multiple variants first")

    print(f"\nAll plots written to: {out_dir}")


if __name__ == "__main__":
    main()
