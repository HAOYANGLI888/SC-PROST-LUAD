"""GMT parsing and train-fold-only pathway score calculation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from data.rnaseq_preprocess import RNASeqPreprocessError


def normalize_gene_symbol(value: object) -> str:
    return str(value or "").strip().upper()


def load_gmt(path: str | Path) -> dict[str, tuple[str, ...]]:
    """Read an MSigDB-style GMT file."""

    gmt_path = Path(path)
    if not gmt_path.exists():
        raise FileNotFoundError(f"Gene-set GMT file not found: {gmt_path}")
    gene_sets: dict[str, tuple[str, ...]] = {}
    for raw_line in gmt_path.read_text(encoding="utf-8").splitlines():
        fields = raw_line.rstrip("\n").split("\t")
        if len(fields) < 3:
            continue
        genes = tuple(symbol for symbol in (normalize_gene_symbol(item) for item in fields[2:]) if symbol)
        if genes:
            gene_sets[fields[0]] = genes
    if not gene_sets:
        raise RNASeqPreprocessError(f"GMT file contains no usable gene sets: {gmt_path}")
    return gene_sets


def load_gene_annotation(path: str | Path) -> dict[str, str]:
    """Load Ensembl-to-symbol annotation exported from GDC STAR-counts."""

    annotation_path = Path(path)
    if not annotation_path.exists():
        raise FileNotFoundError(f"TCGA gene annotation not found: {annotation_path}")
    frame = pd.read_csv(annotation_path)
    required = {"gene_id", "gene_symbol"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RNASeqPreprocessError(f"Gene annotation is missing columns: {missing}")
    return {
        str(row.gene_id): normalize_gene_symbol(row.gene_symbol)
        for row in frame.itertuples()
        if normalize_gene_symbol(row.gene_symbol)
    }


@dataclass
class PathwayScoreTransformer:
    """Calculate per-sample mean z-expression pathway scores."""

    gmt_path: str | Path
    annotation_path: str | Path
    min_genes: int = 5

    def fit(
        self,
        frame: pd.DataFrame,
        patient_ids: Iterable[str] | None = None,
    ) -> "PathwayScoreTransformer":
        annotation = load_gene_annotation(self.annotation_path)
        gene_sets = load_gmt(self.gmt_path)
        symbol_to_column: dict[str, str] = {}
        for column in frame.columns:
            symbol = annotation.get(str(column), normalize_gene_symbol(column))
            if symbol and symbol not in symbol_to_column:
                symbol_to_column[symbol] = str(column)
        pathways = {
            name: [symbol_to_column[symbol] for symbol in genes if symbol in symbol_to_column]
            for name, genes in gene_sets.items()
        }
        self.pathways_ = {name: columns for name, columns in pathways.items() if len(columns) >= self.min_genes}
        if not self.pathways_:
            raise RNASeqPreprocessError("No pathway retained enough genes after TCGA annotation matching.")
        numeric = frame.apply(pd.to_numeric, errors="coerce")
        self.gene_medians_ = numeric.median(axis=0).fillna(0.0)
        imputed = numeric.fillna(self.gene_medians_)
        self.gene_means_ = imputed.mean(axis=0)
        self.gene_scales_ = imputed.std(axis=0, ddof=0).replace(0.0, 1.0)
        scores = self._score(frame)
        self.pathway_means_ = scores.mean(axis=0)
        self.pathway_scales_ = scores.std(axis=0, ddof=0).replace(0.0, 1.0)
        self.fit_patient_ids_ = tuple(frame.index.astype(str).tolist() if patient_ids is None else patient_ids)
        return self

    def _score(self, frame: pd.DataFrame) -> pd.DataFrame:
        numeric = frame.apply(pd.to_numeric, errors="coerce")
        scaled = (numeric.fillna(self.gene_medians_) - self.gene_means_) / self.gene_scales_
        return pd.DataFrame(
            {name: scaled[columns].mean(axis=1) for name, columns in self.pathways_.items()},
            index=frame.index,
        )

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "pathways_"):
            raise RNASeqPreprocessError("Pathway score transformer must be fit first.")
        missing = sorted(set(self.gene_medians_.index) - set(frame.columns))
        if missing:
            raise RNASeqPreprocessError(f"Pathway transform is missing {len(missing)} TCGA genes.")
        scores = self._score(frame)
        return ((scores - self.pathway_means_) / self.pathway_scales_).astype("float32")

