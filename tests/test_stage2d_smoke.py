from validation.external_validation import (
    download_geo_metadata,
    prepare_geo_cohort_small_test,
    run_external_validation,
    summarize_external_validation,
)


def test_stage2d_isolated_small_test_pipeline(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "base.yaml").write_text("project: {name: toy}\n", encoding="utf-8")
    assert download_geo_metadata(tmp_path, small_test=True)["status"] == "passed"
    assert prepare_geo_cohort_small_test(tmp_path, "GSE31210")["status"] == "passed"
    assert run_external_validation(tmp_path, small_test=True)["status"] == "passed"
    summary = summarize_external_validation(tmp_path, small_test=True)
    assert summary["dataset_mode"] == "toy_small_test"
    assert (tmp_path / "outputs" / "audit" / "stage2d_small_test" / "audit_report.md").is_file()
    assert (tmp_path / "outputs" / "reports" / "stage2d_small_test_report.md").is_file()

