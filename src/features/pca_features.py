"""Train-fold-only PCA feature representation for RNA-seq."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from data.rnaseq_preprocess import RNATrainPreprocessor, RNASeqPreprocessError


@dataclass
class PCAFeatureTransformer:
    """Fit RNA filtering, scaling, and PCA on training patients only."""

    n_components: int = 25
    prefilter_top_variable_genes: int = 1000
    seed: int = 42

    def fit(
        self,
        frame: pd.DataFrame,
        patient_ids: Iterable[str] | None = None,
    ) -> "PCAFeatureTransformer":
        self.preprocessor_ = RNATrainPreprocessor(
            top_variable_genes=self.prefilter_top_variable_genes,
        ).fit(frame, patient_ids=patient_ids)
        scaled = self.preprocessor_.transform(frame).to_numpy()
        component_count = min(self.n_components, scaled.shape[0] - 1, scaled.shape[1])
        if component_count < 1:
            raise RNASeqPreprocessError("PCA feature space requires at least two training rows.")
        self.pca_ = PCA(n_components=component_count, random_state=self.seed)
        self.pca_.fit(scaled)
        self.feature_names_ = [f"PC{index:03d}" for index in range(1, component_count + 1)]
        self.fit_patient_ids_ = self.preprocessor_.fit_patient_ids_
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "pca_"):
            raise RNASeqPreprocessError("PCA transformer must be fit first.")
        scaled = self.preprocessor_.transform(frame).to_numpy()
        return pd.DataFrame(self.pca_.transform(scaled), index=frame.index, columns=self.feature_names_)

