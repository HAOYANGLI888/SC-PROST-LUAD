from training.nested_cv import run_nested_cv
from training.oof_prediction import generate_oof_analysis


def test_stage2c_toy_nested_cv_and_oof_pipeline(tmp_path):
    result = run_nested_cv(tmp_path, seeds=(42,), small_test=True)
    assert result["status"] == "passed"
    assert result["maximum_fit_test_overlap_count"] == 0
    oof = generate_oof_analysis(tmp_path, small_test=True)
    assert oof["status"] == "passed"
    required = [
        "data/processed/stage2c_oof_risk_scores.csv",
        "outputs/tables/stage2c_nested_cv_performance.csv",
        "outputs/tables/stage2c_feature_space_comparison.csv",
        "outputs/tables/stage2c_oof_multivariable_cox.csv",
        "outputs/tables/stage2c_oof_km_logrank.csv",
        "outputs/tables/stage2c_overfitting_diagnostics.csv",
        "outputs/figures/stage2c_nested_cv_cindex_boxplot.png",
        "outputs/figures/stage2c_oof_km_best_model.png",
        "outputs/reports/stage2c_rnaseq_robustness_report.md",
        "outputs/audit/stage2c/audit_report.md",
        "audit_report.md",
    ]
    for relative_path in required:
        assert (tmp_path / relative_path).is_file()

