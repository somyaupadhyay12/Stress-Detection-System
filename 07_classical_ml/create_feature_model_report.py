"""Create consistently labelled tables and figures for the 06_feature_model run.

Run this after train_window_feature_models.py.  It does not retrain models;
it turns their saved before/after tables into an auditable report.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "06_feature_model"
VARIANTS = {
    "raw": "Raw data through model",
    "filtered": "Filtered data through model",
    "filtered_normalized": "Filtered + normalized data through model",
}
META_COLUMNS = {"subject", "window_index", "label_original", "stress_label", "predicted_stress", "stress_probability"}


def save_variant_report(folder: Path, label: str) -> dict | None:
    before_path = folder / "features_before_model.csv"
    after_path = folder / "features_after_model.csv"
    metrics_path = folder / "metrics.csv"
    if not before_path.exists() or not after_path.exists() or not metrics_path.exists():
        return None
    before = pd.read_csv(before_path)
    after = pd.read_csv(after_path)
    feature_columns = [column for column in before.columns if column not in META_COLUMNS]
    feature_summary = before[feature_columns].describe().T[["count", "mean", "std", "min", "max"]]
    feature_summary.index.name = "feature"
    feature_summary.to_csv(folder / "feature_summary_before_model.csv")
    class_table = (before.groupby("stress_label").size().rename("window_count").reindex([0, 1], fill_value=0).rename_axis("stress_label").reset_index())
    class_table["class_name"] = class_table["stress_label"].map({0: "Non-stress", 1: "Stress"})
    class_table.to_csv(folder / "class_distribution_before_model.csv", index=False)

    prediction_table = (after.groupby(["stress_label", "predicted_stress"]).agg(
        windows=("stress_probability", "size"), mean_probability=("stress_probability", "mean"))
        .reset_index())
    prediction_table["true_class"] = prediction_table["stress_label"].map({0: "Non-stress", 1: "Stress"})
    prediction_table["predicted_class"] = prediction_table["predicted_stress"].map({0: "Non-stress", 1: "Stress"})
    prediction_table.to_csv(folder / "prediction_summary_after_model.csv", index=False)

    matrix = pd.crosstab(after["stress_label"], after["predicted_stress"]).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    matrix.index = ["True: Non-stress", "True: Stress"]
    matrix.columns = ["Predicted: Non-stress", "Predicted: Stress"]
    matrix.to_csv(folder / "confusion_matrix_after_model_table.csv")
    fig, ax = plt.subplots(figsize=(6.3, 5.2))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, linewidths=.6, linecolor="white", ax=ax)
    ax.set(title=f"Confusion Matrix: {label}", xlabel="Model prediction", ylabel="Ground truth")
    fig.tight_layout(); fig.savefig(folder / "confusion_matrix_after_model.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot_data = after.assign(true_class=after["stress_label"].map({0: "Non-stress", 1: "Stress"}))
    sns.boxplot(data=plot_data, x="true_class", y="stress_probability", hue="true_class", legend=False, palette="Set2", ax=ax)
    ax.axhline(.5, color="black", ls="--", lw=1, label="Decision threshold (0.50)")
    ax.set(title=f"Predicted stress probability: {label}", xlabel="True class", ylabel="Stress probability")
    ax.legend(loc="best"); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(folder / "prediction_probability_after_model.png", dpi=200); plt.close(fig)

    metrics = pd.read_csv(metrics_path).iloc[0].to_dict()
    metrics.update({"variant_folder": folder.name, "display_name": label, "feature_rows_before_model": len(before),
                    "held_out_rows_after_model": len(after), "feature_count": len(feature_columns)})
    return metrics


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    metrics = []
    index_rows = []
    matrices = []
    for folder_name, label in VARIANTS.items():
        folder = OUTPUT_ROOT / folder_name
        result = save_variant_report(folder, label)
        index_rows.append({"folder": folder_name, "model_label": label, "status": "complete" if result else "not yet complete"})
        if result:
            metrics.append(result)
            after = pd.read_csv(folder / "features_after_model.csv")
            matrices.append((label, pd.crosstab(after["stress_label"], after["predicted_stress"]).reindex(index=[0, 1], columns=[0, 1], fill_value=0)))
    pd.DataFrame(index_rows).to_csv(OUTPUT_ROOT / "output_record_index.csv", index=False)
    if not metrics:
        print("No completed variant outputs found yet.")
        return
    comparison = pd.DataFrame(metrics)
    comparison.to_csv(OUTPUT_ROOT / "model_comparison.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.barplot(data=comparison, x="display_name", y="test_accuracy", hue="display_name", legend=False, palette="Blues_d", ax=axes[0])
    sns.barplot(data=comparison, x="display_name", y="test_auc", hue="display_name", legend=False, palette="Greens_d", ax=axes[1])
    for ax, metric, title in zip(axes, ["test_accuracy", "test_auc"], ["Held-out Accuracy", "Held-out ROC-AUC"]):
        ax.set(title=title, xlabel="Input data passed through GRU", ylabel=metric.replace("test_", "").upper(), ylim=(0, 1))
        ax.tick_params(axis="x", rotation=15); ax.grid(axis="y", alpha=.25)
        for bar in ax.patches:
            ax.annotate(f"{bar.get_height():.3f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()), ha="center", va="bottom", fontsize=9)
    fig.suptitle("Stress Detection Model Comparison", fontweight="bold")
    fig.tight_layout(); fig.savefig(OUTPUT_ROOT / "raw_filtered_normalized_model_comparison.png", dpi=220); plt.close(fig)

    fig, axes = plt.subplots(1, len(matrices), figsize=(5.4 * len(matrices), 4.7), squeeze=False)
    for ax, (label, matrix) in zip(axes.ravel(), matrices):
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax, xticklabels=["Non-stress", "Stress"], yticklabels=["Non-stress", "Stress"])
        ax.set(title=label, xlabel="Predicted", ylabel="True")
    fig.suptitle("Confusion Matrices: Raw vs Filtered vs Filtered + Normalized", fontweight="bold")
    fig.tight_layout(); fig.savefig(OUTPUT_ROOT / "all_confusion_matrices_comparison.png", dpi=220); plt.close(fig)
    print(comparison[["display_name", "test_accuracy", "test_auc", "feature_rows_before_model", "held_out_rows_after_model"]].to_string(index=False))


if __name__ == "__main__":
    main()
