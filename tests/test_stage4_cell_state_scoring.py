import numpy as np
import pandas as pd

from data.scrna_signature import curated_luad_cell_state_signatures
from features.signature_scoring import score_signatures


def test_curated_luad_signatures_include_required_states():
    signatures = curated_luad_cell_state_signatures()
    names = {signature.signature_name for signature in signatures}
    assert "emt_like_tumor_cells" in names
    assert "m2_like_macrophages" in names
    assert "cytotoxic_cd8_t_cells" in names
    assert len(signatures) >= 16
    assert all(signature.genes for signature in signatures)


def test_signature_scoring_methods_are_reproducible():
    signatures = curated_luad_cell_state_signatures()[:3]
    genes = sorted({gene for signature in signatures for gene in signature.genes})
    expression = pd.DataFrame({"sample_id": [f"S{i}" for i in range(8)]})
    for index, gene in enumerate(genes):
        expression[gene] = np.arange(8, dtype=float) + index

    mean_result = score_signatures(
        expression,
        signatures,
        sample_col="sample_id",
        method="mean_zscore",
        dataset_label="toy",
    )
    rank_result = score_signatures(
        expression,
        signatures,
        sample_col="sample_id",
        method="rank",
        dataset_label="toy",
    )

    assert set(mean_result.scores.columns) == {"sample_id", *[s.signature_name for s in signatures]}
    assert set(rank_result.scores.columns) == set(mean_result.scores.columns)
    assert mean_result.missingness["n_missing_genes"].max() == 0
    assert mean_result.scores.drop(columns="sample_id").notna().all().all()
