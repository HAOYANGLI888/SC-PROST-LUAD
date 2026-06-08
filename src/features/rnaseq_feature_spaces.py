"""Registry and fold-local transformers for Stage 2C RNA feature spaces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from data.rnaseq_preprocess import RNATrainPreprocessor
from features.pathway_scores import PathwayScoreTransformer
from features.pca_features import PCAFeatureTransformer
from features.supervised_gene_filter import ElasticNetGeneSelector, UnivariateCoxGeneFilter


CORE_FEATURE_SPACES = (
    "clinical_only",
    "raw_high_variance_genes_top500",
    "raw_high_variance_genes_top1000",
    "raw_high_variance_genes_top3000",
    "PCA_25",
    "PCA_50",
    "PCA_100",
    "ElasticNet_selected_genes",
    "univariate_cox_selected_genes_inside_inner_cv",
)
OPTIONAL_FEATURE_SPACES = ("pathway_Hallmark_scores", "pathway_Reactome_scores")


@dataclass(frozen=True)
class FeatureSpaceResources:
    """Optional local resources for pathway feature spaces."""

    annotation_path: Path
    hallmark_gmt: Path
    reactome_gmt: Path

    @classmethod
    def from_root(cls, root: str | Path) -> "FeatureSpaceResources":
        project_root = Path(root).resolve()
        return cls(
            annotation_path=project_root / "data" / "metadata" / "stage2c_tcga_gene_annotation.csv",
            hallmark_gmt=project_root / "data" / "raw" / "gene_sets" / "h.all.v2025.1.Hs.symbols.gmt",
            reactome_gmt=project_root / "data" / "raw" / "gene_sets" / "c2.cp.reactome.v2025.1.Hs.symbols.gmt",
        )


def feature_space_inventory(root: str | Path = ".") -> pd.DataFrame:
    """List required and optional feature spaces with local readiness."""

    resources = FeatureSpaceResources.from_root(root)
    rows = [
        {"feature_space": name, "available": True, "reason": "implemented; no optional file required"}
        for name in CORE_FEATURE_SPACES
    ]
    rows.extend(
        [
            {
                "feature_space": "pathway_Hallmark_scores",
                "available": resources.hallmark_gmt.exists() and resources.annotation_path.exists(),
                "reason": "ready" if resources.hallmark_gmt.exists() and resources.annotation_path.exists()
                else f"optional GMT or annotation missing: {resources.hallmark_gmt.as_posix()}",
            },
            {
                "feature_space": "pathway_Reactome_scores",
                "available": resources.reactome_gmt.exists() and resources.annotation_path.exists(),
                "reason": "ready" if resources.reactome_gmt.exists() and resources.annotation_path.exists()
                else f"optional GMT or annotation missing: {resources.reactome_gmt.as_posix()}",
            },
        ]
    )
    return pd.DataFrame(rows)


class RNAFeatureSpace:
    """Fit and transform one RNA representation inside a training fold."""

    def __init__(self, name: str, *, root: str | Path = ".", seed: int = 42, small_test: bool = False) -> None:
        self.name = name
        self.root = Path(root).resolve()
        self.seed = seed
        self.small_test = small_test

    def fit(
        self,
        frame: pd.DataFrame,
        durations: np.ndarray,
        events: np.ndarray,
        patient_ids: Iterable[str],
    ) -> "RNAFeatureSpace":
        if self.name == "clinical_only":
            self.transformer_ = None
            self.fit_patient_ids_ = tuple(patient_ids)
            return self
        max_features = {
            "raw_high_variance_genes_top500": 60 if self.small_test else 500,
            "raw_high_variance_genes_top1000": 70 if self.small_test else 1000,
            "raw_high_variance_genes_top3000": 80 if self.small_test else 3000,
        }
        resources = FeatureSpaceResources.from_root(self.root)
        if self.name in max_features:
            transformer = RNATrainPreprocessor(top_variable_genes=max_features[self.name])
        elif self.name.startswith("PCA_"):
            transformer = PCAFeatureTransformer(
                n_components=min(int(self.name.split("_")[1]), 8 if self.small_test else 100),
                prefilter_top_variable_genes=70 if self.small_test else 1000,
                seed=self.seed,
            )
        elif self.name == "ElasticNet_selected_genes":
            transformer = ElasticNetGeneSelector(
                max_features=12 if self.small_test else 100,
                prefilter_top_variable_genes=60 if self.small_test else 500,
                epochs=8 if self.small_test else 45,
                seed=self.seed,
            )
        elif self.name == "univariate_cox_selected_genes_inside_inner_cv":
            transformer = UnivariateCoxGeneFilter(
                max_features=12 if self.small_test else 100,
                prefilter_top_variable_genes=70 if self.small_test else 1000,
            )
        elif self.name == "pathway_Hallmark_scores":
            transformer = PathwayScoreTransformer(resources.hallmark_gmt, resources.annotation_path)
        elif self.name == "pathway_Reactome_scores":
            transformer = PathwayScoreTransformer(resources.reactome_gmt, resources.annotation_path)
        else:
            raise ValueError(f"Unknown RNA feature space: {self.name}")
        if isinstance(transformer, (UnivariateCoxGeneFilter, ElasticNetGeneSelector)):
            transformer.fit(frame, durations, events, patient_ids=patient_ids)
        else:
            transformer.fit(frame, patient_ids=patient_ids)
        self.transformer_ = transformer
        self.fit_patient_ids_ = tuple(transformer.fit_patient_ids_)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "transformer_"):
            raise RuntimeError("RNA feature space must be fit before transform.")
        if self.transformer_ is None:
            return pd.DataFrame(index=frame.index)
        return self.transformer_.transform(frame)

    @property
    def feature_count(self) -> int:
        if self.transformer_ is None:
            return 0
        if hasattr(self.transformer_, "selected_genes_"):
            return len(self.transformer_.selected_genes_)
        if hasattr(self.transformer_, "feature_names_"):
            return len(self.transformer_.feature_names_)
        if hasattr(self.transformer_, "pathways_"):
            return len(self.transformer_.pathways_)
        raise RuntimeError(f"Could not resolve feature count for {self.name}.")

