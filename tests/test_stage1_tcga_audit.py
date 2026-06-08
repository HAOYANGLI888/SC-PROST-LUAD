import csv
from pathlib import Path

from data.tcga_audit import (
    AuditPaths,
    extract_patient_id,
    run_audit,
)


def test_extract_patient_id_from_sample_barcode():
    assert extract_patient_id("TCGA-AB-1234-01A") == "TCGA-AB-1234"
    assert extract_patient_id("not-a-tcga-id") is None


def test_stage1_dry_run_is_offline_and_writes_plan(tmp_path):
    result = run_audit(tmp_path, dry_run=True)
    assert result["status"] == "passed"
    assert result["network_access"] is False
    assert AuditPaths.from_root(tmp_path).dry_run_report.is_file()


def test_stage1_small_test_writes_required_outputs(tmp_path):
    result = run_audit(tmp_path, small_test=True)
    paths = AuditPaths.from_root(tmp_path)
    assert result["status"] == "passed"
    assert result["summary"]["clinical_rnaseq"] == 3
    assert result["summary"]["clinical_rnaseq_wsi"] == 2
    assert result["summary"]["clinical_rnaseq_mutation_methylation"] == 3
    assert result["summary"]["complete_multimodal"] == 2
    for output in (
        paths.patient_matrix,
        paths.survival_summary,
        paths.modality_summary,
        paths.report,
        paths.audit_mirror,
        paths.small_test_log,
    ):
        assert output.is_file()
    with paths.patient_matrix.open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
