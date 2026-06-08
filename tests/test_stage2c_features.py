import numpy as np
import pandas as pd

from features.rnaseq_feature_spaces import RNAFeatureSpace


def _fixture():
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(rng.normal(size=(24, 40)), columns=[f"GENE_{i:03d}" for i in range(40)])
    durations = np.arange(1, 25, dtype=float) * 30
    events = np.asarray([0, 1] * 12)
    ids = [f"P{i:03d}" for i in range(24)]
    return frame, durations, events, ids


def test_hvg_feature_space_fits_training_ids_only(tmp_path):
    frame, durations, events, ids = _fixture()
    transformer = RNAFeatureSpace(
        "raw_high_variance_genes_top500",
        root=tmp_path,
        small_test=True,
    ).fit(frame.iloc[:18], durations[:18], events[:18], ids[:18])
    transformed = transformer.transform(frame.iloc[18:])
    assert transformed.shape == (6, 40)
    assert set(transformer.fit_patient_ids_) == set(ids[:18])
    assert set(transformer.fit_patient_ids_).isdisjoint(ids[18:])


def test_pca_and_supervised_feature_spaces_transform_holdout(tmp_path):
    frame, durations, events, ids = _fixture()
    for name in ("PCA_25", "univariate_cox_selected_genes_inside_inner_cv", "ElasticNet_selected_genes"):
        transformer = RNAFeatureSpace(name, root=tmp_path, small_test=True).fit(
            frame.iloc[:18], durations[:18], events[:18], ids[:18]
        )
        transformed = transformer.transform(frame.iloc[18:])
        assert transformed.shape[0] == 6
        assert transformed.shape[1] > 0
        assert np.isfinite(transformed.to_numpy()).all()

