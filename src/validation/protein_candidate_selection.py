"""Stage 5 protein/IHC candidate gene selection."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from features.cell_state_scores import load_tcga_stage2d_context
from features.signature_scoring import load_tcga_symbol_expression


class CandidateSelectionError(RuntimeError):
    """Raised when Stage 5 candidate selection cannot proceed."""


@dataclass(frozen=True)
class MechanismGeneSet:
    """A mechanism-layer gene pool for orthogonal protein validation."""

    mechanism_layer: str
    stage4_signature: str
    expected_direction: str
    genes: tuple[str, ...]


DEFAULT_MECHANISM_GENE_SETS: tuple[MechanismGeneSet, ...] = (
    MechanismGeneSet(
        "proliferation",
        "proliferative_tumor_cells",
        "higher_in_high_risk",
        ("MKI67", "TOP2A", "PCNA", "MCM2", "MCM5", "CCNB1", "CDK1"),
    ),
    MechanismGeneSet(
        "hypoxia",
        "hypoxia_tumor_cells",
        "higher_in_high_risk",
        ("CA9", "VEGFA", "LDHA", "SLC2A1", "EGLN3", "BNIP3"),
    ),
    MechanismGeneSet(
        "emt_like_malignant_program",
        "emt_like_tumor_cells",
        "higher_in_high_risk",
        ("VIM", "FN1", "SNAI1", "SNAI2", "ZEB1", "ZEB2", "CDH2", "ITGA5"),
    ),
    MechanismGeneSet(
        "caf_matrix",
        "caf",
        "higher_in_high_risk",
        ("ACTA2", "COL1A1", "COL1A2", "COL3A1", "FAP", "PDGFRB", "TAGLN"),
    ),
    MechanismGeneSet(
        "dendritic_b_plasma_context",
        "dendritic_b_plasma_cells",
        "higher_in_low_risk",
        ("CD74", "HLA-DRA", "HLA-DRB1", "MS4A1", "CD79A", "CD79B", "MZB1", "JCHAIN", "XBP1"),
    ),
)


def _project_paths(root: str | Path) -> dict[str, Path]:
    project_root = Path(root).resolve()
    return {
        "root": project_root,
        "tables": project_root / "outputs" / "tables",
        "metadata": project_root / "data" / "metadata",
        "stage4_signatures": project_root / "outputs" / "tables" / "stage4_cell_state_signature_definitions.csv",
        "stage4_correlation": project_root / "outputs" / "tables" / "stage4_tcga_cell_state_risk_correlation.csv",
        "stage4_consistency": project_root / "outputs" / "tables" / "stage4_cell_state_external_consistency.csv",
        "tcga_annotation": project_root / "data" / "metadata" / "stage2c_tcga_gene_annotation.csv",
        "frozen_pickle": project_root / "outputs" / "checkpoints" / "stage2c_tcga_fixed_rna_validation_model.pkl",
    }


def _load_stage4_tables(paths: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing = [
        str(path)
        for path in (paths["stage4_signatures"], paths["stage4_correlation"], paths["stage4_consistency"])
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Stage 5 candidate selection requires completed Stage 4 outputs: "
            + ", ".join(missing)
        )
    return (
        pd.read_csv(paths["stage4_signatures"]),
        pd.read_csv(paths["stage4_correlation"]),
        pd.read_csv(paths["stage4_consistency"]),
    )


def _gene_annotation(paths: dict[str, Path]) -> pd.DataFrame:
    if not paths["tcga_annotation"].exists():
        raise FileNotFoundError(f"TCGA gene annotation is missing: {paths['tcga_annotation']}")
    annotation = pd.read_csv(paths["tcga_annotation"], dtype=str)
    required = {"gene_id", "gene_symbol"}
    missing = sorted(required - set(annotation.columns))
    if missing:
        raise CandidateSelectionError(f"TCGA annotation is missing columns: {missing}")
    annotation["gene_symbol"] = annotation["gene_symbol"].astype(str).str.upper()
    return annotation.drop_duplicates("gene_symbol")


def _stage2d_required_symbols(paths: dict[str, Path], annotation: pd.DataFrame) -> set[str]:
    if not paths["frozen_pickle"].exists():
        return set()
    try:
        with paths["frozen_pickle"].open("rb") as handle:
            artifact = pickle.load(handle)
        genes = set(str(gene).split(".")[0] for gene in artifact.get("required_gene_ids", []))
    except Exception:
        return set()
    symbol_by_gene = {
        str(row.gene_id).split(".")[0]: str(row.gene_symbol).upper()
        for row in annotation.itertuples(index=False)
    }
    return {symbol_by_gene[gene] for gene in genes if gene in symbol_by_gene}


def _tcga_gene_risk_correlations(
    root: str | Path,
    candidate_symbols: Iterable[str],
    *,
    small_test: bool = False,
) -> pd.DataFrame:
    expression, _summary = load_tcga_symbol_expression(root, small_test=False)
    context = load_tcga_stage2d_context(root, small_test=False)
    symbols = sorted({str(symbol).upper() for symbol in candidate_symbols})
    available = [symbol for symbol in symbols if symbol in expression.columns]
    merged = context[["patient_id", "risk_score"]].merge(
        expression[["patient_id", *available]], on="patient_id", how="inner"
    )
    if small_test:
        merged = merged.head(80)
    rows = []
    for symbol in symbols:
        row = {"gene_symbol": symbol, "tcga_mrna_available": symbol in available}
        if symbol not in available or merged[symbol].nunique(dropna=True) < 2:
            row.update({"tcga_mrna_risk_spearman": np.nan, "tcga_mrna_risk_p_value": np.nan})
        else:
            valid = merged[["risk_score", symbol]].dropna()
            rho, p_value = spearmanr(valid["risk_score"], valid[symbol])
            row.update(
                {
                    "tcga_mrna_risk_spearman": float(rho),
                    "tcga_mrna_risk_p_value": float(p_value),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _mechanism_support_tables(
    stage4_correlation: pd.DataFrame,
    stage4_consistency: pd.DataFrame,
) -> pd.DataFrame:
    correlation = stage4_correlation.rename(columns={"spearman_rho": "stage4_signature_risk_rho"})
    keep = ["signature_name", "stage4_signature_risk_rho", "p_value", "q_value_bh"]
    support = stage4_consistency.merge(correlation[keep], on="signature_name", how="left")
    support["matches_predefined_risk_direction"] = (
        support["matches_predefined_risk_direction"].astype(str).str.lower() == "true"
    )
    return support


def select_candidate_genes(
    root: str | Path = ".",
    *,
    small_test: bool = False,
    genes_per_layer: int = 10,
) -> pd.DataFrame:
    """Select Stage 5 candidate genes without using protein results."""

    paths = _project_paths(root)
    _signatures, stage4_correlation, stage4_consistency = _load_stage4_tables(paths)
    annotation = _gene_annotation(paths)
    stage2d_symbols = _stage2d_required_symbols(paths, annotation)
    support = _mechanism_support_tables(stage4_correlation, stage4_consistency)
    annotation_by_symbol = annotation.set_index("gene_symbol")
    pools = DEFAULT_MECHANISM_GENE_SETS[:2] if small_test else DEFAULT_MECHANISM_GENE_SETS
    all_symbols = [gene for pool in pools for gene in pool.genes]
    gene_corr = _tcga_gene_risk_correlations(paths["root"], all_symbols, small_test=small_test)
    gene_corr = gene_corr.set_index("gene_symbol")

    rows: list[dict[str, object]] = []
    for pool in pools:
        if pool.stage4_signature == "dendritic_b_plasma_cells":
            signature_rows = support.loc[
                support["signature_name"].isin(["dendritic_cells", "b_cells", "plasma_cells"])
            ]
            stage4_rho = float(signature_rows["stage4_signature_risk_rho"].mean())
            consistency = "composite_low_risk_immune_context"
            match = bool(signature_rows["matches_predefined_risk_direction"].all())
        else:
            signature_rows = support.loc[support["signature_name"] == pool.stage4_signature]
            stage4_rho = float(signature_rows["stage4_signature_risk_rho"].iloc[0]) if not signature_rows.empty else np.nan
            consistency = (
                str(signature_rows["external_direction_consistency"].iloc[0])
                if not signature_rows.empty
                else "unknown"
            )
            match = (
                bool(signature_rows["matches_predefined_risk_direction"].iloc[0])
                if not signature_rows.empty
                else False
            )
        selected = list(pool.genes)[: max(5, min(genes_per_layer, len(pool.genes)))]
        for index, symbol in enumerate(selected, start=1):
            symbol = symbol.upper()
            ensembl = (
                str(annotation_by_symbol.loc[symbol, "gene_id"]).split(".")[0]
                if symbol in annotation_by_symbol.index
                else ""
            )
            corr_row = gene_corr.loc[symbol] if symbol in gene_corr.index else pd.Series(dtype=object)
            rho = corr_row.get("tcga_mrna_risk_spearman", np.nan)
            expected_gene_direction_supported = (
                (pool.expected_direction == "higher_in_high_risk" and pd.notna(rho) and float(rho) > 0)
                or (pool.expected_direction == "higher_in_low_risk" and pd.notna(rho) and float(rho) < 0)
            )
            rows.append(
                {
                    "gene_symbol": symbol,
                    "ensembl_id": ensembl,
                    "mechanism_layer": pool.mechanism_layer,
                    "stage4_signature": pool.stage4_signature,
                    "expected_direction": pool.expected_direction,
                    "stage4_signature_risk_rho": stage4_rho,
                    "stage4_external_consistency": consistency,
                    "stage4_matches_predefined_direction": match,
                    "tcga_mrna_available": bool(corr_row.get("tcga_mrna_available", False)),
                    "tcga_mrna_risk_spearman": rho,
                    "tcga_mrna_risk_p_value": corr_row.get("tcga_mrna_risk_p_value", np.nan),
                    "tcga_mrna_direction_matches_layer": bool(expected_gene_direction_supported),
                    "in_stage2d_fixed_input_genes": symbol in stage2d_symbols,
                    "hpa_status_before_query": "pending_stage5_hpa_validation",
                    "cptac_status_before_query": "pending_stage5_cptac_validation",
                    "priority_rank_within_layer": index,
                    "selection_rationale": (
                        "Pre-specified Stage 5 marker from a Stage 4 mechanism layer; "
                        "not selected using protein outcomes."
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    frame["candidate_priority_score"] = (
        frame["stage4_matches_predefined_direction"].astype(int) * 3
        + frame["tcga_mrna_direction_matches_layer"].astype(int) * 2
        + frame["in_stage2d_fixed_input_genes"].astype(int)
    )
    frame = frame.sort_values(
        ["mechanism_layer", "candidate_priority_score", "priority_rank_within_layer"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    return frame


def write_candidate_outputs(
    root: str | Path = ".",
    *,
    small_test: bool = False,
    genes_per_layer: int = 10,
) -> Path:
    """Run candidate selection and write the configured Stage 5 table."""

    paths = _project_paths(root)
    output = (
        paths["root"] / "outputs" / "stage5_small_test" / "tables" / "candidate_genes.csv"
        if small_test
        else paths["tables"] / "stage5_candidate_genes.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    candidates = select_candidate_genes(paths["root"], small_test=small_test, genes_per_layer=genes_per_layer)
    candidates.to_csv(output, index=False)
    return output

