from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from data.scrna_annotation import _source_harmonized_labels
from data.scrna_raw_import import (
    assemble_sparse_chunks_to_h5ad,
    build_data_inventory,
    scrna_paths,
)
from features.scrna_signature_scores import KEY_GENES, RISK_PROGRAMS
from validation.scrna_cellular_context import (
    build_cellular_context_summary,
    build_mechanism_support_matrix,
)
from validation.scrna_celltype_summary import rank_program_contexts
from validation.scrna_lowmem_reader import stratified_downsample_indices


ROOT = Path(__file__).resolve().parents[1]


def test_stage4b_programs_are_fixed_and_survival_free():
    assert RISK_PROGRAMS["hypoxia"] == [
        "LDHA",
        "CA9",
        "SLC2A1",
        "VEGFA",
        "BNIP3",
        "EGLN3",
    ]
    assert KEY_GENES == ["LDHA", "MKI67", "CDK1"]
    all_names = " ".join(RISK_PROGRAMS).lower()
    assert "survival" not in all_names
    assert "os_" not in all_names


def test_stage4b_source_annotation_is_preserved_conservatively():
    obs = pd.DataFrame(
        {
            "Cell_type.refined": [
                "Epithelial cells",
                "Fibroblasts",
                "T/NK cells",
                "B lymphocytes",
            ],
            "Cell_subtype": [
                "Malignant cells",
                "Myofibroblasts",
                "Cytotoxic CD8+ T",
                "Plasma cells",
            ],
        }
    )
    labels = _source_harmonized_labels(obs)
    assert labels.tolist() == [
        "Malignant epithelial",
        "Fibroblasts/CAF",
        "CD8 T cells",
        "Plasma cells",
    ]


def test_stage4b_small_test_outputs_are_isolated_and_readable():
    paths = scrna_paths(ROOT, small_test=True)
    assert "stage4b_small_test" in str(paths.imported_h5ad)
    assert paths.imported_h5ad.exists()
    inventory = build_data_inventory(paths.raw_dir)
    assert isinstance(inventory, pd.DataFrame)
    assert (
        ROOT
        / "outputs"
        / "reports"
        / "stage4b_small_test"
        / "stage4b_scrna_final_report.md"
    ).exists()


def test_stage4b_context_support_uses_cell_type_localization():
    frame = pd.DataFrame(
        {
            "analysis_cell_type": [
                "Malignant epithelial",
                "Fibroblasts/CAF",
                "Dendritic cells",
                "B cells",
                "Plasma cells",
            ],
            "hypoxia_score_mean": [2.0, 0.1, -0.3, -0.2, -0.1],
            "proliferation_score_mean": [2.2, -0.2, -0.4, -0.4, -0.3],
            "emt_like_score_mean": [1.2, 1.4, -0.3, -0.2, -0.2],
            "caf_matrix_score_mean": [0.1, 2.4, -0.5, -0.4, -0.4],
            "dendritic_b_plasma_context_score_mean": [-0.5, -0.4, 1.0, 1.1, 1.2],
        }
    )
    gene = pd.DataFrame(
        {
            "analysis_cell_type": ["Malignant epithelial"] * 3,
            "gene_symbol": ["LDHA", "MKI67", "CDK1"],
            "average_log_expression": [2.0, 1.8, 1.7],
            "detection_fraction": [0.8, 0.5, 0.5],
        }
    )
    support = build_mechanism_support_matrix(frame, gene)
    summary = build_cellular_context_summary(frame, gene, support)
    assert support["expected_context_in_top3"].all()
    assert "survival" not in " ".join(summary["question"]).lower()


def test_stage4b_chunked_h5ad_assembly_preserves_counts_and_metadata(tmp_path):
    work = tmp_path / "work"
    chunks = work / "chunks"
    chunks.mkdir(parents=True)
    features = pd.Series(["G1", "G2", "G3"])
    barcodes = pd.Series(["C1_LUNG_T06", "C2_LUNG_T06", "C3_LUNG_T06"])
    features.to_csv(tmp_path / "features.tsv", sep="\t", index=False, header=False)
    barcodes.to_csv(tmp_path / "barcodes.tsv", sep="\t", index=False, header=False)
    pd.DataFrame(
        {
            "Index": barcodes,
            "Barcode": ["C1", "C2", "C3"],
            "Sample": ["LUNG_T06"] * 3,
            "Sample_Origin": ["tLung"] * 3,
            "Cell_type": ["Epithelial cells"] * 3,
            "Cell_type.refined": ["Epithelial cells"] * 3,
            "Cell_subtype": ["Malignant cells"] * 3,
        }
    ).to_csv(tmp_path / "annotation.tsv", sep="\t", index=False)

    arrays = [
        (1, 1, 2, [1, 2, 3], [0, 2, 1], [0, 2, 3]),
        (2, 3, 3, [4], [2], [0, 1]),
    ]
    for chunk_id, start, end, values, indices, indptr in arrays:
        prefix = chunks / f"chunk_{chunk_id:05d}"
        np.asarray(values, dtype="<i4").tofile(prefix.with_suffix(".data.bin"))
        np.asarray(indices, dtype="<i4").tofile(prefix.with_suffix(".indices.bin"))
        np.asarray(indptr, dtype="<i4").tofile(prefix.with_suffix(".indptr.bin"))
        pd.DataFrame(
            [
                {
                    "chunk_id": chunk_id,
                    "start_cell_1based": start,
                    "end_cell_1based": end,
                    "n_cells": end - start + 1,
                    "n_genes": 3,
                    "nnz": len(values),
                    "min_value": min(values),
                    "max_value": max(values),
                    "nan_count": 0,
                    "inf_count": 0,
                    "negative_count": 0,
                }
            ]
        ).to_csv(prefix.with_suffix(".done.tsv"), sep="\t", index=False)

    destination = tmp_path / "assembled.h5ad"
    assemble_sparse_chunks_to_h5ad(
        work,
        tmp_path / "annotation.tsv",
        tmp_path / "features.tsv",
        tmp_path / "barcodes.tsv",
        destination,
    )
    data = ad.read_h5ad(destination)
    assert data.shape == (3, 3)
    assert data.X.toarray().tolist() == [[1, 0, 2], [0, 3, 0], [0, 0, 4]]
    assert data.obs["sample_id"].tolist() == ["LUNG_T06"] * 3
    assert data.obs["patient_id"].tolist() == ["P06"] * 3
    assert data.obs["Cell_subtype"].tolist() == ["Malignant cells"] * 3


def test_stage4b_lowmem_downsample_is_reproducible_and_stratified():
    obs = pd.DataFrame(
        {
            "analysis_cell_type": ["A"] * 50 + ["B"] * 30 + ["C"] * 20,
            "sample_id": ["S1"] * 25
            + ["S2"] * 25
            + ["S1"] * 15
            + ["S2"] * 15
            + ["S1"] * 10
            + ["S2"] * 10,
        }
    )
    first = stratified_downsample_indices(obs, 40, seed=42)
    second = stratified_downsample_indices(obs, 40, seed=42)
    assert first.tolist() == second.tolist()
    assert len(first) == 40
    sampled = obs.iloc[first]
    assert set(sampled["analysis_cell_type"]) == {"A", "B", "C"}
    assert set(sampled["sample_id"]) == {"S1", "S2"}


def test_stage4b_lowmem_context_distinguishes_stromal_emt():
    rows = []
    rankings = {
        "hypoxia": ["Malignant epithelial", "Epithelial", "Fibroblasts/CAF"],
        "proliferation": ["T cells", "Malignant epithelial", "Dendritic cells"],
        "emt_like": ["Fibroblasts/CAF", "Macrophages/monocytes", "Endothelial cells"],
        "caf_matrix": ["Fibroblasts/CAF", "Epithelial", "Malignant epithelial"],
        "dendritic_b_plasma_context": ["B cells", "Dendritic cells", "Plasma cells"],
    }
    for program, cell_types in rankings.items():
        for score, cell_type in zip((3.0, 2.0, 1.0), cell_types):
            rows.append(
                {
                    "program": program,
                    "analysis_cell_type": cell_type,
                    "mean_log1p_cpm_score": score,
                }
            )
    support = rank_program_contexts(pd.DataFrame(rows)).set_index("mechanism")
    assert support.loc["hypoxia", "support_status"] == "supported_primary_context"
    assert support.loc["proliferation", "support_status"] == "partially_supported"
    assert (
        support.loc["emt_like", "support_status"]
        == "supported_stromal_context_only"
    )
