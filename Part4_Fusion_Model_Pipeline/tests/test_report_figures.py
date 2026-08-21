
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "outputs" / "report_figures"


def test_required_report_figures_exist():
    required = [
        "test_confusion_matrix_row_normalized.png",
        "test_score_distribution_with_thresholds.png",
        "mae_before_after_active_learning_grouped.png",
        "test_true_vs_predicted_by_class.png",
        "recall_validation_vs_test_with_requirements.png",
        "test_precision_recall_f1.png",
        "very_high_threshold_tradeoff.png",
        "feature_importance_top20_report.png",
        "shap_importance_top20_report.png",
        "active_learning_mae_percent_change.png",
        "system_pipeline_diagram.png",
    ]
    missing = [name for name in required if not (FIGURES / name).exists()]
    assert not missing, f"Missing report figures: {missing}"
    assert all((FIGURES / name).stat().st_size > 10_000 for name in required)
