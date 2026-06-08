from pathlib import Path

from reporting.stage8 import (
    build_figure_plan,
    build_table_plan,
    load_stage8_evidence,
    scan_forbidden_overclaims,
)


ROOT = Path(__file__).resolve().parents[1]


def test_stage8_evidence_registry_uses_latest_results():
    evidence = load_stage8_evidence(ROOT)
    assert evidence.tcga_patients == 503
    assert evidence.external_total == 1193
    assert evidence.hpa_supported_count == 37
    assert evidence.cptac_samples == 110
    assert evidence.cptac_deaths == 26
    assert evidence.cptac_candidates_matched == 31
    assert set(evidence.cptac_supported_genes) == {"LDHA", "MKI67", "CDK1"}
    assert evidence.wsi_clinical_cindex > evidence.wsi_pathology_cindex


def test_stage8_plans_cover_main_and_supplementary_outputs():
    figures = build_figure_plan(ROOT)
    tables = build_table_plan(ROOT)
    assert set(figures["placement"]) == {"main", "supplementary"}
    assert figures["figure_id"].eq("Figure 5").any()
    assert tables["table_id"].eq("Table 4").any()
    assert tables["table_id"].eq("Table S7").any()
    assert tables["all_source_tables_exist"].all()


def test_stage8_manuscript_drafts_avoid_forbidden_overclaims():
    paths = [
        ROOT / "outputs" / "manuscript" / "SC_PROST_LUAD_manuscript_skeleton.md",
        ROOT / "outputs" / "reports" / "stage8_results_narrative_draft.md",
        ROOT / "outputs" / "reports" / "stage8_discussion_points.md",
        ROOT / "outputs" / "reports" / "stage8_limitations.md",
    ]
    assert all(path.exists() for path in paths)
    assert scan_forbidden_overclaims(paths) == []
