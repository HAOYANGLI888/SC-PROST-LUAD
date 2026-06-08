import pandas as pd

from validation.cptac_data_import import (
    build_cptac_inventory,
    classify_cptac_file,
    create_small_test_raw_files,
    stage5b_paths,
)
from validation.cptac_luad_preprocess import preprocess_cptac_luad
from validation.cptac_protein_analysis import (
    integrate_stage5b_evidence,
    run_candidate_protein_validation,
)
from validation.cptac_survival_analysis import run_cptac_survival_analysis


def _write_stage5_inputs(root):
    table_dir = root / "outputs" / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    candidates = pd.DataFrame(
        {
            "gene_symbol": ["MKI67", "CA9", "VIM", "COL1A1", "MS4A1", "JCHAIN"],
            "ensembl_id": [""] * 6,
            "mechanism_layer": [
                "proliferation",
                "hypoxia",
                "emt_like_malignant_program",
                "caf_matrix",
                "dendritic_b_plasma_context",
                "dendritic_b_plasma_context",
            ],
            "stage4_signature": [
                "proliferative_tumor_cells",
                "hypoxia_tumor_cells",
                "emt_like_tumor_cells",
                "caf",
                "dendritic_b_plasma_cells",
                "dendritic_b_plasma_cells",
            ],
            "expected_direction": [
                "higher_in_high_risk",
                "higher_in_high_risk",
                "higher_in_high_risk",
                "higher_in_high_risk",
                "higher_in_low_risk",
                "higher_in_low_risk",
            ],
        }
    )
    candidates.to_csv(table_dir / "stage5_candidate_genes.csv", index=False)
    hpa = candidates[["gene_symbol", "mechanism_layer"]].copy()
    hpa["supported_by_HPA"] = True
    hpa["hpa_support_status"] = "qualitative_link_available"
    hpa["evidence_level"] = "moderate"
    hpa.to_csv(table_dir / "stage5_integrated_protein_evidence.csv", index=False)


def test_stage5b_file_classification_and_empty_inventory(tmp_path):
    paths = stage5b_paths(tmp_path)
    assert classify_cptac_file("CPTAC_LUAD_protein_abundance.tsv") == "protein_abundance_matrix"
    assert classify_cptac_file("clinical_survival.csv") == "clinical_metadata"
    assert classify_cptac_file("sample_annotation.csv") == "sample_annotation"
    inventory = build_cptac_inventory(paths)
    assert inventory.loc[0, "role"] == "missing_all_required_files"
    assert not bool(inventory.loc[0, "required_roles_present"])


def test_stage5b_small_test_pipeline_is_isolated_and_integrates(tmp_path):
    _write_stage5_inputs(tmp_path)
    paths = stage5b_paths(tmp_path, small_test=True)
    create_small_test_raw_files(paths)
    inventory = build_cptac_inventory(paths)
    assert "protein_abundance_matrix" in set(inventory["role"])

    preprocess = preprocess_cptac_luad(paths, inventory, small_test=True)
    assert preprocess["status"] == "processed"
    assert preprocess["candidate_available"] >= 4
    clinical = pd.read_csv(paths.processed_dir / "cptac_luad_clinical_processed.csv")
    assert set(clinical["os_event"].dropna().unique()).issubset({0.0, 1.0})
    assert (clinical["os_event"] == 0).any()

    validation = run_candidate_protein_validation(paths, small_test=True)
    assert validation["candidate_available"] >= 4
    survival = run_cptac_survival_analysis(paths, small_test=True)
    assert survival["protein_rows"] == 6
    integrated = integrate_stage5b_evidence(paths, small_test=True)
    assert integrated["genes"] == 6
    assert (paths.tables_dir / "stage5b_integrated_hpa_cptac_evidence.csv").is_file()
