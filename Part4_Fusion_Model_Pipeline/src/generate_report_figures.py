
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

CLASS_ORDER = ["Low", "Moderate", "High", "Very High"]


def save_figure(fig, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def row_normalized_confusion(frame, split_name, palette, output_dir):
    matrix = confusion_matrix(
        frame["true_class"], frame["predicted_class"], labels=CLASS_ORDER
    )
    normalized = matrix / matrix.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(9.2, 7.6))
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.04)
    colorbar.set_label("Share of actual class", fontsize=11)

    ax.set_xticks(range(4), CLASS_ORDER, rotation=24, ha="right")
    ax.set_yticks(range(4), CLASS_ORDER)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title(
        f"{split_name} confusion matrix — color normalized within each actual class",
        pad=14,
        fontsize=15,
    )
    for row in range(4):
        for column in range(4):
            percentage = normalized[row, column]
            ax.text(
                column,
                row,
                f"{matrix[row, column]}\n({percentage:.0%})",
                ha="center",
                va="center",
                color="white" if percentage >= 0.55 else "black",
                fontsize=12,
                fontweight="bold" if row == column else "normal",
            )
    for index, label in enumerate(CLASS_ORDER):
        ax.get_yticklabels()[index].set_color(palette[label])
        ax.get_yticklabels()[index].set_fontweight("bold")
    fig.tight_layout()
    save_figure(fig, output_dir, f"{split_name.lower()}_confusion_matrix_row_normalized")


def count_confusion(frame, split_name, output_dir):
    matrix = confusion_matrix(
        frame["true_class"], frame["predicted_class"], labels=CLASS_ORDER
    )
    fig, ax = plt.subplots(figsize=(8.7, 7.1))
    image = ax.imshow(matrix, cmap="Purples", aspect="auto")
    fig.colorbar(image, ax=ax, fraction=0.045, pad=0.04, label="Posts")
    ax.set_xticks(range(4), CLASS_ORDER, rotation=24, ha="right")
    ax.set_yticks(range(4), CLASS_ORDER)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title(f"{split_name} confusion matrix — raw counts", pad=14)
    threshold = matrix.max() * 0.55
    for row in range(4):
        for column in range(4):
            ax.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] >= threshold else "black",
                fontsize=12,
            )
    fig.tight_layout()
    save_figure(fig, output_dir, f"{split_name.lower()}_confusion_matrix_counts")


def density_plot(frame, thresholds, split_name, palette, output_dir):
    fig, ax = plt.subplots(figsize=(12.5, 7.2))
    x_grid = np.linspace(1, 10, 500)
    for label in CLASS_ORDER:
        values = frame.loc[
            frame["true_class"].eq(label), "fusion_prediction"
        ].to_numpy(float)
        if len(np.unique(values)) > 1:
            density = gaussian_kde(values)(x_grid)
            ax.fill_between(
                x_grid, density, alpha=0.28, color=palette[label],
                label=f"{label} (n={len(values)})"
            )
            ax.plot(x_grid, density, color=palette[label], linewidth=2.2)
    threshold_colors = ["#D62728", "#FF8C00", "#6A0DAD"]
    threshold_names = ["Low / Moderate", "Moderate / High", "High / Very High"]
    for value, color, name in zip(thresholds, threshold_colors, threshold_names):
        ax.axvline(
            value, color=color, linestyle="--", linewidth=2.1,
            label=f"{name}: {value:.2f}"
        )
    ax.set_xlim(1, 10)
    ax.set_xlabel("Predicted continuous stress score")
    ax.set_ylabel("Density")
    ax.set_title(f"Distribution of predicted scores by true class — {split_name}", pad=12)
    ax.grid(alpha=0.22)
    ax.legend(ncol=2, frameon=True)
    fig.tight_layout()
    save_figure(fig, output_dir, f"{split_name.lower()}_score_distribution_with_thresholds")


def mae_before_after(config, output_dir):
    data = pd.DataFrame(config["active_learning_mae"])
    model_order = ["Tabular", "Transformer", "Fusion"]
    before_name = "Before active learning"
    after_name = "After active learning"
    pivot = data.pivot(index="model", columns="stage", values="mae").loc[model_order]

    x = np.arange(len(model_order))
    width = 0.34
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    before = ax.bar(x - width / 2, pivot[before_name], width, label=before_name, color="#A0A0A0")
    after = ax.bar(
        x + width / 2,
        pivot[after_name],
        width,
        label="After active learning (model colors)",
        color=[config["model_palette"][model] for model in model_order],
    )
    ax.set_xticks(x, model_order)
    ax.set_ylabel("Mean absolute error (lower is better)")
    ax.set_ylim(0, data["mae"].max() * 1.22)
    ax.set_title("Model MAE before and after active-learning expansion")
    ax.grid(axis="y", alpha=0.22)
    ax.legend()
    ax.bar_label(before, fmt="%.3f", padding=3)
    ax.bar_label(after, fmt="%.3f", padding=3)
    fig.text(
        0.5, 0.01, config["comparability_warning"],
        ha="center", va="bottom", fontsize=8.5, wrap=True
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save_figure(fig, output_dir, "mae_before_after_active_learning_grouped")
    data.to_csv(
        output_dir / "mae_before_after_active_learning_values.csv",
        index=False, encoding="utf-8-sig"
    )

    change = (pivot[after_name] / pivot[before_name] - 1.0) * 100.0
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    bars = ax.bar(
        model_order, change,
        color=[config["model_palette"][model] for model in model_order]
    )
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("MAE change (%)")
    ax.set_title("Project-stage MAE change after active-learning expansion")
    ax.grid(axis="y", alpha=0.22)
    for bar, value in zip(bars, change):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + (1 if value >= 0 else -1),
            f"{value:+.1f}%",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontweight="bold",
        )
    fig.text(
        0.5, 0.01,
        "Negative change means lower MAE. " + config["comparability_warning"],
        ha="center", va="bottom", fontsize=8.5, wrap=True
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save_figure(fig, output_dir, "active_learning_mae_percent_change")


def current_locked_mae_comparison(root, output_dir, model_palette):
    manifest = pd.read_csv(root / "data/modeling_manifest_v2.csv", dtype={"unique_post_id": "string"})
    truth = manifest.loc[
        manifest["model_role"].eq("test"), ["unique_post_id", "final_stress"]
    ]
    m1 = pd.read_csv(root / "data/member1_test_predictions_for_fusion.csv", dtype={"unique_post_id": "string"})
    m2 = pd.read_csv(root / "data/member2_test_predictions_fold_ensemble.csv", dtype={"unique_post_id": "string"})
    fusion = pd.read_csv(root / "outputs/test_predictions.csv", dtype={"unique_post_id": "string"})
    rows = []
    for model, frame, column in [
        ("Tabular", m1, "prediction"),
        ("Transformer", m2, "prediction"),
        ("Fusion", fusion, "fusion_prediction"),
    ]:
        merged = truth.merge(frame[["unique_post_id", column]], on="unique_post_id", validate="one_to_one")
        mae = np.mean(np.abs(merged["final_stress"].to_numpy(float) - merged[column].to_numpy(float)))
        rows.append({"model": model, "mae": mae})
    table = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    bars = ax.bar(
        table["model"], table["mae"],
        color=[model_palette[model] for model in table["model"]]
    )
    ax.set_ylabel("Mean absolute error")
    ax.set_title("Current locked-test MAE — directly comparable models")
    ax.set_ylim(0, table["mae"].max() * 1.22)
    ax.grid(axis="y", alpha=0.22)
    ax.bar_label(bars, fmt="%.3f", padding=3)
    fig.tight_layout()
    save_figure(fig, output_dir, "current_locked_test_mae_comparison")
    table.to_csv(output_dir / "current_locked_test_mae_values.csv", index=False, encoding="utf-8-sig")


def scatter_by_class(frame, thresholds, split_name, palette, output_dir):
    fig, ax = plt.subplots(figsize=(8.3, 7.4))
    for label in CLASS_ORDER:
        subset = frame.loc[frame["true_class"].eq(label)]
        ax.scatter(
            subset["true_stress"], subset["fusion_prediction"],
            color=palette[label], alpha=0.68, s=42,
            edgecolor="white", linewidth=0.45,
            label=f"{label} (n={len(subset)})"
        )
    ax.plot([1, 10], [1, 10], color="#333333", linestyle="--", linewidth=1.8, label="Ideal y = x")
    for boundary in [3, 5, 7]:
        ax.axvline(boundary, color="#999999", linestyle=":", linewidth=1)
    for threshold in thresholds:
        ax.axhline(threshold, color="#777777", linestyle=":", linewidth=1)
    ax.set_xlim(1, 10)
    ax.set_ylim(1, 10)
    ax.set_xlabel("True stress")
    ax.set_ylabel("Fusion prediction")
    ax.set_title(f"{split_name}: predicted versus true stress, colored by true class")
    ax.grid(alpha=0.20)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    save_figure(fig, output_dir, f"{split_name.lower()}_true_vs_predicted_by_class")


def recall_comparison(validation, test, output_dir, palette):
    val_recall = pd.Series(
        {
            label: (
                validation.loc[validation["true_class"].eq(label), "predicted_class"]
                .eq(label)
                .mean()
            )
            for label in CLASS_ORDER
        }
    )
    test_recall = pd.Series(
        {
            label: (
                test.loc[test["true_class"].eq(label), "predicted_class"]
                .eq(label)
                .mean()
            )
            for label in CLASS_ORDER
        }
    )
    requirements = np.array([0.75, 0.50, 0.50, 0.75])
    x = np.arange(4)
    width = 0.32
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    val_bars = ax.bar(x - width / 2, val_recall, width, label="Validation", color="#6BAED6")
    test_bars = ax.bar(x + width / 2, test_recall, width, label="Locked test", color="#9E9AC8")
    ax.plot(x, requirements, color="#D62728", marker="o", linestyle="--", linewidth=1.8, label="Required floor")
    ax.set_xticks(x, CLASS_ORDER)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Recall")
    ax.set_title("Per-class recall: validation versus locked test")
    ax.grid(axis="y", alpha=0.22)
    ax.legend()
    ax.bar_label(val_bars, labels=[f"{value:.0%}" for value in val_recall], padding=3)
    ax.bar_label(test_bars, labels=[f"{value:.0%}" for value in test_recall], padding=3)
    for index, label in enumerate(CLASS_ORDER):
        ax.get_xticklabels()[index].set_color(palette[label])
        ax.get_xticklabels()[index].set_fontweight("bold")
    fig.tight_layout()
    save_figure(fig, output_dir, "recall_validation_vs_test_with_requirements")


def class_metric_chart(frame, split_name, palette, output_dir):
    precision, recall, f1, support = precision_recall_fscore_support(
        frame["true_class"], frame["predicted_class"], labels=CLASS_ORDER, zero_division=0
    )
    x = np.arange(4)
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 6.5))
    p_bars = ax.bar(x - width, precision, width, label="Precision", color="#4C78A8")
    r_bars = ax.bar(x, recall, width, label="Recall", color="#F58518")
    f_bars = ax.bar(x + width, f1, width, label="F1", color="#54A24B")
    ax.set_xticks(x, [f"{label}\n(n={count})" for label, count in zip(CLASS_ORDER, support)])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"{split_name} precision, recall and F1 by class")
    ax.grid(axis="y", alpha=0.22)
    ax.legend()
    for bars, values in [(p_bars, precision), (r_bars, recall), (f_bars, f1)]:
        ax.bar_label(bars, labels=[f"{value:.0%}" for value in values], padding=3, fontsize=8)
    for index, label in enumerate(CLASS_ORDER):
        ax.get_xticklabels()[index].set_color(palette[label])
        ax.get_xticklabels()[index].set_fontweight("bold")
    fig.tight_layout()
    save_figure(fig, output_dir, f"{split_name.lower()}_precision_recall_f1")


def very_high_tradeoff(validation, thresholds, output_dir):
    t1, t2, selected_t3 = thresholds
    true_class = validation["true_class"].to_numpy(object)
    score = validation["fusion_prediction"].to_numpy(float)
    t3_grid = np.linspace(max(t2 + 0.05, 5.0), min(8.2, score.max() + 0.2), 180)
    rows = []
    for t3 in t3_grid:
        predicted = np.asarray(CLASS_ORDER, dtype=object)[
            np.digitize(score, [t1, t2, t3], right=False)
        ]
        precision, recall, _, _ = precision_recall_fscore_support(
            true_class, predicted, labels=CLASS_ORDER, zero_division=0
        )
        rows.append({
            "t3": t3,
            "very_high_precision": precision[3],
            "very_high_recall": recall[3],
            "high_recall": recall[2],
        })
    trade = pd.DataFrame(rows)
    trade.to_csv(output_dir / "very_high_threshold_tradeoff_values.csv", index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.plot(trade["t3"], trade["very_high_precision"], linewidth=2.2, label="Very-High precision")
    ax.plot(trade["t3"], trade["very_high_recall"], linewidth=2.2, label="Very-High recall")
    ax.plot(trade["t3"], trade["high_recall"], linewidth=2.0, label="High recall")
    ax.axhline(0.75, color="#D62728", linestyle="--", linewidth=1.5, label="Very-High recall floor")
    ax.axhline(0.50, color="#888888", linestyle="--", linewidth=1.2, label="High recall floor")
    ax.axvline(selected_t3, color="#6A0DAD", linestyle=":", linewidth=2.2, label=f"Selected t3 = {selected_t3:.2f}")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("High / Very-High prediction threshold")
    ax.set_ylabel("Metric")
    ax.set_title("Safety trade-off when moving the High / Very-High threshold")
    ax.grid(alpha=0.22)
    ax.legend(ncol=2)
    fig.tight_layout()
    save_figure(fig, output_dir, "very_high_threshold_tradeoff")


def importance_plots(root, output_dir):
    importance = pd.read_csv(root / "outputs/global_feature_importance.csv").head(20)
    colors = [
        "#54A24B" if f == "member2_prediction"
        else "#4C78A8" if f == "member1_prediction"
        else "#B279A2" if f.startswith("base_")
        else "#9D9D9D"
        for f in importance["feature"]
    ]
    fig, ax = plt.subplots(figsize=(10.2, 7.4))
    ax.barh(importance["feature"][::-1], importance["importance"][::-1], color=colors[::-1])
    ax.set_xlabel("CatBoost feature importance")
    ax.set_title("Top 20 global fusion features")
    ax.grid(axis="x", alpha=0.20)
    fig.tight_layout()
    save_figure(fig, output_dir, "feature_importance_top20_report")

    shap = pd.read_csv(root / "outputs/validation_shap_summary.csv").head(20)
    colors = [
        "#54A24B" if f == "member2_prediction"
        else "#4C78A8" if f == "member1_prediction"
        else "#B279A2" if f.startswith("base_")
        else "#9D9D9D"
        for f in shap["feature"]
    ]
    fig, ax = plt.subplots(figsize=(10.2, 7.4))
    ax.barh(shap["feature"][::-1], shap["mean_absolute_shap"][::-1], color=colors[::-1])
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_title("Top 20 validation SHAP contributions")
    ax.grid(axis="x", alpha=0.20)
    fig.tight_layout()
    save_figure(fig, output_dir, "shap_importance_top20_report")


def error_by_class(test, palette, output_dir):
    values = [
        np.abs(
            test.loc[test["true_class"].eq(label), "fusion_prediction"].to_numpy(float)
            - test.loc[test["true_class"].eq(label), "true_stress"].to_numpy(float)
        )
        for label in CLASS_ORDER
    ]
    fig, ax = plt.subplots(figsize=(9.5, 6.3))
    box = ax.boxplot(values, tick_labels=CLASS_ORDER, patch_artist=True, showfliers=True)
    for patch, label in zip(box["boxes"], CLASS_ORDER):
        patch.set_facecolor(palette[label])
        patch.set_alpha(0.65)
    ax.set_ylabel("Absolute error")
    ax.set_title("Locked-test absolute error distribution by true class")
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    save_figure(fig, output_dir, "test_absolute_error_by_true_class")

    rows = []
    for label in CLASS_ORDER:
        subset = test.loc[test["true_class"].eq(label)]
        under = (subset["fusion_prediction"] < subset["true_stress"]).mean()
        over = (subset["fusion_prediction"] > subset["true_stress"]).mean()
        rows.append({"class": label, "Under-predicted": under, "Over-predicted": over})
    direction = pd.DataFrame(rows).set_index("class").loc[CLASS_ORDER]
    fig, ax = plt.subplots(figsize=(9.8, 6.0))
    bottom = np.zeros(4)
    for column, color in [("Under-predicted", "#D62728"), ("Over-predicted", "#4C78A8")]:
        ax.bar(CLASS_ORDER, direction[column], bottom=bottom, label=column, color=color)
        bottom += direction[column].to_numpy()
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of class")
    ax.set_title("Locked-test error direction by true class")
    ax.legend()
    ax.grid(axis="y", alpha=0.20)
    for index, label in enumerate(CLASS_ORDER):
        ax.get_xticklabels()[index].set_color(palette[label])
        ax.get_xticklabels()[index].set_fontweight("bold")
    fig.tight_layout()
    save_figure(fig, output_dir, "test_error_direction_by_true_class")


def candidate_tradeoff(root, output_dir):
    candidates = pd.read_csv(root / "outputs/candidate_leaderboard.csv")
    feasible = candidates.loc[candidates["threshold_status"].eq("feasible")].copy()
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    for profile, marker, color in [("scalar", "o", "#4C78A8"), ("hybrid", "s", "#B279A2")]:
        subset = feasible.loc[feasible["feature_profile"].eq(profile)]
        ax.scatter(
            subset["validation_macro_f1"],
            subset["validation_very_high_precision"],
            s=58, marker=marker, color=color, alpha=0.75, label=profile.title()
        )
    selected = feasible.loc[feasible["selected"]]
    ax.scatter(
        selected["validation_macro_f1"],
        selected["validation_very_high_precision"],
        s=190, marker="*", color="#D62728", label="Selected", zorder=5
    )
    for _, row in selected.iterrows():
        ax.annotate(
            row["candidate"],
            (row["validation_macro_f1"], row["validation_very_high_precision"]),
            xytext=(8, 8), textcoords="offset points", fontsize=9
        )
    ax.set_xlabel("Validation macro F1")
    ax.set_ylabel("Validation Very-High precision")
    ax.set_title("Fusion candidate trade-off after recall constraints")
    ax.grid(alpha=0.22)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, output_dir, "fusion_candidate_tradeoff")


def system_diagram(output_dir):
    fig, ax = plt.subplots(figsize=(15, 8.5))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 9)
    ax.axis("off")

    def box(x, y, w, h, text, face, edge="#333333", fontsize=10.5):
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.03,rounding_size=0.12",
            linewidth=1.5, edgecolor=edge, facecolor=face
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, wrap=True)

    def arrow(x1, y1, x2, y2, color="#444444"):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>",
            mutation_scale=15, linewidth=1.8, color=color
        ))

    box(0.4, 3.6, 2.0, 1.5, "Enriched forum post\ntext + metadata + signal counts", "#F3F3F3")
    box(3.0, 5.6, 2.7, 1.5, "Member 1\nTabular CatBoost\n5-fold ensemble", "#DCEAF7", "#4C78A8")
    box(3.0, 2.0, 2.7, 1.5, "Member 2\nParsBERT regression\n5-fold ensemble", "#DFF0DA", "#54A24B")
    box(6.5, 3.6, 2.7, 1.5, "Fusion feature matrix\nbase predictions + tabular features\n+ disagreement features", "#EEE2F0", "#B279A2")
    box(9.8, 3.6, 2.2, 1.5, "Hybrid CatBoost\ncontinuous stress score", "#F3E1ED", "#B44C6A")
    box(12.7, 5.6, 1.9, 1.5, "Calibrated\nthresholds\n4 clinical bins", "#FBE7C6", "#F58518")
    box(12.7, 2.0, 1.9, 1.5, "Temporal state\nEWMA + trend +\nrepeated-risk rules", "#E6F2F2", "#2A9D8F")
    box(9.7, 0.15, 2.6, 1.25, "Active-learning queue\nboundary proximity +\nbase-model disagreement", "#FFF3BF", "#E3BA22")
    box(5.8, 0.15, 2.6, 1.25, "Human dual annotation\nnew confirmation labels", "#F7E1D7", "#C96B3B")

    arrow(2.4, 4.55, 3.0, 6.35)
    arrow(2.4, 4.15, 3.0, 2.75)
    arrow(5.7, 6.35, 6.5, 4.65)
    arrow(5.7, 2.75, 6.5, 4.05)
    arrow(9.2, 4.35, 9.8, 4.35)
    arrow(12.0, 4.65, 12.7, 6.1)
    arrow(12.0, 4.05, 12.7, 2.75)
    arrow(13.65, 2.0, 11.7, 1.4)
    arrow(9.7, 0.78, 8.4, 0.78)
    arrow(5.8, 0.78, 4.4, 2.0, "#C96B3B")

    ax.text(7.5, 8.35, "End-to-end fusion, monitoring and active-learning architecture",
            ha="center", va="center", fontsize=18, fontweight="bold")
    ax.text(7.5, 7.85, "Primary fusion uses leakage-safe OOF/fold-ensemble base predictions",
            ha="center", va="center", fontsize=11, color="#555555")
    fig.tight_layout()
    save_figure(fig, output_dir, "system_pipeline_diagram")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else root / "outputs/report_figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads((root / "config/report_figure_sources.json").read_text(encoding="utf-8"))
    palette = config["class_palette"]
    bundle = json.loads((root / "models/fusion_bundle.json").read_text(encoding="utf-8"))
    thresholds = bundle["thresholds"]

    validation = pd.read_csv(root / "outputs/validation_predictions.csv")
    test = pd.read_csv(root / "outputs/test_predictions.csv")

    row_normalized_confusion(validation, "Validation", palette, output_dir)
    row_normalized_confusion(test, "Test", palette, output_dir)
    count_confusion(validation, "Validation", output_dir)
    count_confusion(test, "Test", output_dir)

    density_plot(validation, thresholds, "Validation", palette, output_dir)
    density_plot(test, thresholds, "Test", palette, output_dir)

    mae_before_after(config, output_dir)
    current_locked_mae_comparison(root, output_dir, config["model_palette"])

    scatter_by_class(validation, thresholds, "Validation", palette, output_dir)
    scatter_by_class(test, thresholds, "Test", palette, output_dir)

    recall_comparison(validation, test, output_dir, palette)
    class_metric_chart(validation, "Validation", palette, output_dir)
    class_metric_chart(test, "Test", palette, output_dir)

    very_high_tradeoff(validation, thresholds, output_dir)
    importance_plots(root, output_dir)
    error_by_class(test, palette, output_dir)
    candidate_tradeoff(root, output_dir)
    system_diagram(output_dir)

    manifest = []
    for path in sorted(output_dir.iterdir()):
        if path.is_file():
            manifest.append({
                "file": path.name,
                "kind": path.suffix.lower().lstrip("."),
                "bytes": path.stat().st_size,
            })
    pd.DataFrame(manifest).to_csv(
        output_dir / "figure_manifest.csv", index=False, encoding="utf-8-sig"
    )
    print(json.dumps({"status": "complete", "files": len(manifest)}, indent=2))


if __name__ == "__main__":
    main()
