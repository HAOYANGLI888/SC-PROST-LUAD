import subprocess
import sys
from pathlib import Path


def test_stage0_script_help():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/run_stage0_smoke_test.py", "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--root" in result.stdout


def test_stage1_script_help():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/stage1_audit_tcga_luad.py", "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--dry-run" in result.stdout
    assert "--small-test" in result.stdout


def test_stage2_script_help():
    root = Path(__file__).resolve().parents[1]
    for script in (
        "scripts/stage2_prepare_rnaseq_survival.py",
        "scripts/train_stage2_baselines.py",
        "scripts/evaluate_stage2_models.py",
    ):
        result = subprocess.run(
            [sys.executable, script, "--help"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert "--small-test" in result.stdout


def test_stage2b_script_help():
    root = Path(__file__).resolve().parents[1]
    scripts = (
        "scripts/stage2_build_gdc_rnaseq_manifest.py",
        "scripts/stage2_download_gdc_rnaseq.py",
        "scripts/stage2_build_rnaseq_tpm_matrix.py",
        "scripts/stage2_import_xena_rnaseq_matrix.py",
    )
    for script in scripts:
        result = subprocess.run(
            [sys.executable, script, "--help"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0


def test_stage2c_script_help():
    root = Path(__file__).resolve().parents[1]
    scripts = (
        "scripts/stage2c_compare_feature_spaces.py",
        "scripts/stage2c_nested_cv_rnaseq.py",
        "scripts/stage2c_generate_oof_risk_scores.py",
        "scripts/stage2c_prepare_geo_validation.py",
    )
    for script in scripts:
        result = subprocess.run(
            [sys.executable, script, "--help"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert "--small-test" in result.stdout


def test_stage4_script_help():
    root = Path(__file__).resolve().parents[1]
    scripts = (
        "scripts/stage4_build_single_cell_signatures.py",
        "scripts/stage4_score_cell_states_tcga.py",
        "scripts/stage4_score_cell_states_geo.py",
    )
    for script in scripts:
        result = subprocess.run(
            [sys.executable, script, "--help"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert "--config" in result.stdout
        assert "--small-test" in result.stdout


def test_stage4b_script_help():
    root = Path(__file__).resolve().parents[1]
    scripts = (
        "scripts/stage4b_scrna_download_or_import.py",
        "scripts/stage4b_scrna_preprocess_qc.py",
        "scripts/stage4b_scrna_cell_annotation.py",
        "scripts/stage4b_scrna_score_risk_programs.py",
        "scripts/stage4b_scrna_cellular_context_validation.py",
        "scripts/stage4b_scrna_generate_figures.py",
        "scripts/stage4b_scrna_final_report.py",
    )
    for script in scripts:
        result = subprocess.run(
            [sys.executable, script, "--help"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert "--config" in result.stdout
        assert "--small-test" in result.stdout


def test_stage5_script_help():
    root = Path(__file__).resolve().parents[1]
    scripts = (
        "scripts/stage5_select_candidate_genes.py",
        "scripts/stage5_hpa_validation.py",
        "scripts/stage5_cptac_validation.py",
        "scripts/stage5_integrate_protein_evidence.py",
    )
    for script in scripts:
        result = subprocess.run(
            [sys.executable, script, "--help"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert "--config" in result.stdout
        assert "--small-test" in result.stdout


def test_stage5b_script_help():
    root = Path(__file__).resolve().parents[1]
    scripts = (
        "scripts/stage5b_cptac_download_or_import.py",
        "scripts/stage5b_cptac_preprocess.py",
        "scripts/stage5b_cptac_candidate_validation.py",
        "scripts/stage5b_cptac_survival_analysis.py",
        "scripts/stage5b_integrate_cptac_evidence.py",
    )
    for script in scripts:
        result = subprocess.run(
            [sys.executable, script, "--help"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert "--config" in result.stdout
        assert "--small-test" in result.stdout


def test_stage6a_script_help():
    root = Path(__file__).resolve().parents[1]
    scripts = (
        "scripts/stage6a_build_wsi_manifest.py",
        "scripts/stage6a_download_wsi_smallset.py",
        "scripts/stage6a_extract_wsi_patches.py",
        "scripts/stage6a_extract_patch_features.py",
        "scripts/train_stage6a_pathology_mil.py",
        "scripts/evaluate_stage6a_pathology_mil.py",
        "scripts/stage6a_check_wsi_environment.py",
        "scripts/stage6a_resume_wsi_smallset.py",
        "scripts/stage6a_run_real_smallset_pipeline.py",
    )
    for script in scripts:
        result = subprocess.run(
            [sys.executable, script, "--help"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert "--config" in result.stdout
        if script not in {
            "scripts/stage6a_check_wsi_environment.py",
            "scripts/stage6a_resume_wsi_smallset.py",
            "scripts/stage6a_run_real_smallset_pipeline.py",
        }:
            assert "--small-test" in result.stdout


def test_stage8_script_help():
    root = Path(__file__).resolve().parents[1]
    scripts = (
        "scripts/stage8_generate_manuscript_skeleton.py",
        "scripts/stage8_collect_figures_tables.py",
        "scripts/stage8_generate_results_narrative.py",
        "scripts/stage8_generate_journal_targeting_report.py",
        "scripts/stage8_final_evidence_audit.py",
    )
    for script in scripts:
        result = subprocess.run(
            [sys.executable, script, "--help"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert "--config" in result.stdout


def test_stage9_script_help():
    root = Path(__file__).resolve().parents[1]
    scripts = (
        "scripts/stage9_generate_jtm_manuscript.py",
        "scripts/stage9_generate_jtm_title_page.py",
        "scripts/stage9_generate_jtm_cover_letter.py",
        "scripts/stage9_generate_jtm_declarations.py",
        "scripts/stage9_jtm_compliance_audit.py",
        "scripts/stage9_jtm_claim_language_audit.py",
    )
    for script in scripts:
        result = subprocess.run(
            [sys.executable, script, "--help"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert "--config" in result.stdout
