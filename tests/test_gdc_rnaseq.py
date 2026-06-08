from pathlib import Path

import pandas as pd

from data.gdc_rnaseq import GDCRNASeqPaths, build_tpm_matrix
from data.rnaseq_preprocess import load_rnaseq_matrix


def _write_counts(path: Path, tspan6: float, dpm1: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# gene-model: GENCODE v36\n"
        "gene_id\tgene_name\tgene_type\tunstranded\tstranded_first\tstranded_second\ttpm_unstranded\tfpkm_unstranded\tfpkm_uq_unstranded\n"
        f"ENSG00000000003.15\tTSPAN6\tprotein_coding\t1\t1\t0\t{tspan6}\t0\t0\n"
        f"ENSG00000000419.13\tDPM1\tprotein_coding\t1\t1\t0\t{dpm1}\t0\t0\n"
        "ENSG00000223972.5\tDDX11L1\ttranscribed_unprocessed_pseudogene\t1\t1\t0\t99\t0\t0\n",
        encoding="utf-8",
    )


def test_build_tpm_matrix_resolves_duplicate_patient_deterministically(tmp_path):
    paths = GDCRNASeqPaths.from_root(tmp_path)
    paths.ensure_dirs()
    rows = [
        {
            "file_id": "older",
            "file_name": "older.tsv",
            "patient_id": "TCGA-AA-0001",
            "sample_id": "TCGA-AA-0001-01A",
            "sample_type": "Primary Tumor",
            "updated_datetime": "2024-01-01T00:00:00Z",
        },
        {
            "file_id": "newer",
            "file_name": "newer.tsv",
            "patient_id": "TCGA-AA-0001",
            "sample_id": "TCGA-AA-0001-01B",
            "sample_type": "Primary Tumor",
            "updated_datetime": "2024-02-01T00:00:00Z",
        },
        {
            "file_id": "single",
            "file_name": "single.tsv",
            "patient_id": "TCGA-BB-0002",
            "sample_id": "TCGA-BB-0002-01A",
            "sample_type": "Primary Tumor",
            "updated_datetime": "2024-01-01T00:00:00Z",
        },
    ]
    pd.DataFrame(rows).to_csv(paths.file_patient_map, index=False)
    _write_counts(paths.download_dir / "older" / "older.tsv", 1.0, 2.0)
    _write_counts(paths.download_dir / "newer" / "newer.tsv", 7.0, 8.0)
    _write_counts(paths.download_dir / "single" / "single.tsv", 3.0, 4.0)
    result = build_tpm_matrix(tmp_path)
    assert result["matrix_patient_count"] == 2
    assert result["matrix_gene_count"] == 2
    assert result["duplicate_patient_count"] == 1
    matrix = pd.read_csv(paths.matrix).set_index("patient_id")
    assert matrix.loc["TCGA-AA-0001", "ENSG00000000003"] == 7.0


def test_load_rnaseq_matrix_accepts_existing_patient_id_column(tmp_path):
    path = tmp_path / "matrix.csv"
    pd.DataFrame(
        {
            "patient_id": ["TCGA-AA-0001", "TCGA-BB-0002"],
            "ENSG1": [0.0, 3.0],
        }
    ).to_csv(path, index=False)
    parsed = load_rnaseq_matrix(path)
    assert parsed["patient_id"].tolist() == ["TCGA-AA-0001", "TCGA-BB-0002"]
    assert parsed.loc[1, "ENSG1"] == 2.0
