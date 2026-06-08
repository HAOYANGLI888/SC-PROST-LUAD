from pathlib import Path

from scprost_luad.stages.stage0 import run_stage0


def test_stage0_smoke_runs_and_writes_audit():
    root = Path(__file__).resolve().parents[1]
    result = run_stage0(root)
    assert result["status"] == "passed"
    assert (root / "outputs" / "audit" / "stage0" / "audit_report.md").is_file()
    assert (root / "data" / "metadata" / "stage0_directory_manifest.csv").is_file()
