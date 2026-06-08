import numpy as np
import pandas as pd

from data.rnaseq_preprocess import RNATrainPreprocessor, load_rnaseq_matrix


def test_load_rnaseq_matrix_normalizes_patient_barcode_and_log_transforms(tmp_path):
    path = tmp_path / "rna.csv"
    pd.DataFrame(
        {
            "sample_id": ["TCGA-AB-1234-01A", "TCGA-CD-5678-01A"],
            "GENE1": [3.0, 7.0],
            "GENE2": [0.0, 1.0],
        }
    ).to_csv(path, index=False)
    frame = load_rnaseq_matrix(path)
    assert frame["patient_id"].tolist() == ["TCGA-AB-1234", "TCGA-CD-5678"]
    assert np.isclose(frame.loc[0, "GENE1"], 2.0)


def test_rna_preprocessor_uses_training_statistics_only():
    train = pd.DataFrame({"G1": [1.0, 2.0, 3.0], "G2": [0.0, 0.0, 0.0]})
    test = pd.DataFrame({"G1": [100.0], "G2": [100.0]})
    processor = RNATrainPreprocessor(top_variable_genes=2).fit(train, patient_ids=["A", "B", "C"])
    transformed = processor.transform(test)
    assert processor.fit_patient_ids_ == ("A", "B", "C")
    assert processor.selected_genes_ == ["G1"]
    assert transformed.loc[0, "G1"] > 10
