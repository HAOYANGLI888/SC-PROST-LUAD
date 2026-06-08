from data.survival_dataset import prepare_stage2_dataset
from evaluation.stage2_evaluation import evaluate_stage2
from models.stage2_training import train_seed


def test_stage2_small_test_end_to_end(tmp_path):
    prepare_stage2_dataset(tmp_path, small_test=True)
    train_seed(tmp_path, seed=42, small_test=True)
    result = evaluate_stage2(tmp_path, small_test=True)
    assert result["status"] == "passed"
    required = [
        "data/processed/stage2_rnaseq_survival_dataset.csv",
        "data/metadata/stage2_feature_list.csv",
        "outputs/tables/stage2_model_performance.csv",
        "outputs/tables/stage2_model_performance_by_seed.csv",
        "outputs/tables/stage2_multivariable_cox.csv",
        "outputs/figures/stage2_km_best_model.png",
        "outputs/figures/stage2_time_dependent_auc_best_model.png",
        "outputs/figures/stage2_calibration_best_model.png",
        "outputs/reports/stage2_rnaseq_survival_report.md",
        "outputs/audit/stage2/audit_report.md",
    ]
    for relative_path in required:
        assert (tmp_path / relative_path).is_file()
