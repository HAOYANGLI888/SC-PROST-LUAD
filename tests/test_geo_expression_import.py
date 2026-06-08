import numpy as np
import pandas as pd

from data.geo_expression_import import (
    collapse_probes_to_genes,
    prepare_geo_expression,
    read_geo_expression,
    read_probe_annotation,
)
from data.geo_survival_preprocess import load_geo_os


def test_geo_probe_collapse_and_independent_zscore(tmp_path):
    expression = tmp_path / "expression_matrix.tsv"
    annotation = tmp_path / "probe_annotation.tsv"
    pd.DataFrame(
        {
            "probe_id": ["p1", "p2", "p3"],
            "GSM1": [1.0, 3.0, 9.0],
            "GSM2": [2.0, 4.0, 6.0],
            "GSM3": [3.0, 5.0, 3.0],
        }
    ).to_csv(expression, sep="\t", index=False)
    pd.DataFrame({"probe_id": ["p1", "p2", "p3"], "gene_symbol": ["tp53", "TP53", "EGFR"]}).to_csv(
        annotation, sep="\t", index=False
    )
    collapsed = collapse_probes_to_genes(read_geo_expression(expression), read_probe_annotation(annotation))
    assert set(collapsed.columns) == {"sample_id", "TP53", "EGFR"}
    prepared = prepare_geo_expression(expression, annotation)
    assert np.allclose(prepared[["TP53", "EGFR"]].mean(axis=0), 0.0)


def test_geo_survival_requires_usable_os(tmp_path):
    survival = tmp_path / "clinical_survival.csv"
    pd.DataFrame(
        {"sample_id": ["GSM1", "GSM2"], "OS_time": [12, 24], "OS_status": ["dead", "alive"]}
    ).to_csv(survival, index=False)
    result = load_geo_os(survival, time_unit="months")
    assert result["OS_status"].tolist() == [1, 0]
    assert result["OS_time"].iloc[0] > 300

