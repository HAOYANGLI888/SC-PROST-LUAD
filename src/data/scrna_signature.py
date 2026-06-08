"""Single-cell and curated LUAD cell-state signature utilities.

Stage 4 uses these signatures for biological interpretation only. Survival
outcomes are never used to derive signatures in this module.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


class SCRNASignatureError(RuntimeError):
    """Raised when a single-cell signature source cannot be parsed safely."""


@dataclass(frozen=True)
class CellStateSignature:
    """A named marker-gene set used for bulk expression scoring."""

    signature_name: str
    cell_state: str
    category: str
    expected_risk_direction: str
    source: str
    description: str
    genes: tuple[str, ...]


def normalize_gene_symbol(value: object) -> str:
    """Return a conservative uppercase gene symbol."""

    text = str(value or "").strip().strip('"').upper()
    if not text or text in {"NAN", "NA", "N/A", "---"}:
        return ""
    for separator in ("///", " // ", "|", ";", ","):
        if separator in text:
            text = text.split(separator, maxsplit=1)[0].strip()
    return text


def _genes(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        symbol = normalize_gene_symbol(value)
        if symbol and symbol not in seen:
            seen.add(symbol)
            result.append(symbol)
    return tuple(result)


def curated_luad_cell_state_signatures() -> list[CellStateSignature]:
    """Return pre-specified LUAD-relevant cell-state signatures."""

    return [
        CellStateSignature(
            "malignant_epithelial_cells",
            "malignant epithelial cells",
            "tumor",
            "context_dependent",
            "curated_luad_marker_set_v1",
            "Epithelial and LUAD malignant-cell-enriched markers.",
            _genes(["EPCAM", "KRT8", "KRT18", "KRT19", "MSLN", "TACSTD2", "ALDH1A1", "CLDN4"]),
        ),
        CellStateSignature(
            "emt_like_tumor_cells",
            "EMT-like tumor cells",
            "tumor_state",
            "higher_risk",
            "curated_luad_marker_set_v1",
            "Mesenchymal transition and invasive tumor-state markers.",
            _genes(["VIM", "ZEB1", "ZEB2", "SNAI1", "SNAI2", "TWIST1", "FN1", "ITGA5", "CDH2", "SERPINE1"]),
        ),
        CellStateSignature(
            "proliferative_tumor_cells",
            "proliferative tumor cells",
            "tumor_state",
            "higher_risk",
            "curated_luad_marker_set_v1",
            "Cell-cycle and proliferation-associated tumor-state markers.",
            _genes(["MKI67", "TOP2A", "UBE2C", "CENPF", "CCNB1", "CCNB2", "AURKA", "BIRC5", "PCNA", "MCM2"]),
        ),
        CellStateSignature(
            "hypoxia_tumor_cells",
            "hypoxia tumor cells",
            "tumor_state",
            "higher_risk",
            "curated_luad_marker_set_v1",
            "Hypoxia and glycolysis-associated tumor-state markers.",
            _genes(["CA9", "VEGFA", "SLC2A1", "LDHA", "PGK1", "ENO1", "PDK1", "BNIP3", "ALDOA", "NDRG1"]),
        ),
        CellStateSignature(
            "exhausted_cd8_t_cells",
            "exhausted CD8 T cells",
            "immune",
            "higher_risk",
            "curated_luad_marker_set_v1",
            "T-cell exhaustion and dysfunctional CD8 T-cell markers.",
            _genes(["PDCD1", "LAG3", "HAVCR2", "TIGIT", "CTLA4", "TOX", "CXCL13", "ENTPD1"]),
        ),
        CellStateSignature(
            "cytotoxic_cd8_t_cells",
            "cytotoxic CD8 T cells",
            "immune",
            "lower_risk",
            "curated_luad_marker_set_v1",
            "Cytotoxic CD8 T-cell effector markers.",
            _genes(["CD8A", "CD8B", "GZMB", "GZMA", "PRF1", "NKG7", "IFNG", "GNLY", "CCL5"]),
        ),
        CellStateSignature(
            "treg_cells",
            "Treg cells",
            "immune",
            "higher_risk",
            "curated_luad_marker_set_v1",
            "Regulatory T-cell markers.",
            _genes(["FOXP3", "IL2RA", "CTLA4", "IKZF2", "TNFRSF18", "TIGIT", "CCR8", "IL10"]),
        ),
        CellStateSignature(
            "m2_like_macrophages",
            "M2-like macrophages",
            "myeloid",
            "higher_risk",
            "curated_luad_marker_set_v1",
            "Immunosuppressive macrophage-state markers.",
            _genes(["CD163", "MRC1", "MSR1", "IL10", "CCL18", "APOE", "MARCO", "VSIG4", "FOLR2"]),
        ),
        CellStateSignature(
            "m1_like_macrophages",
            "M1-like macrophages",
            "myeloid",
            "lower_risk",
            "curated_luad_marker_set_v1",
            "Inflammatory macrophage-state markers.",
            _genes(["IL1B", "TNF", "CXCL9", "CXCL10", "CXCL11", "NOS2", "CD80", "CD86", "STAT1"]),
        ),
        CellStateSignature(
            "dendritic_cells",
            "dendritic cells",
            "myeloid",
            "lower_risk",
            "curated_luad_marker_set_v1",
            "Conventional and mature dendritic-cell markers.",
            _genes(["CLEC9A", "XCR1", "BATF3", "LAMP3", "CCR7", "CD1C", "FCER1A", "ITGAX", "HLA-DRA"]),
        ),
        CellStateSignature(
            "caf",
            "CAF",
            "stroma",
            "higher_risk",
            "curated_luad_marker_set_v1",
            "Cancer-associated fibroblast and matrix-remodeling markers.",
            _genes(["COL1A1", "COL1A2", "COL3A1", "ACTA2", "FAP", "PDGFRA", "PDGFRB", "TAGLN", "DCN", "LUM"]),
        ),
        CellStateSignature(
            "endothelial_cells",
            "endothelial cells",
            "stroma",
            "context_dependent",
            "curated_luad_marker_set_v1",
            "Vascular endothelial-cell markers.",
            _genes(["PECAM1", "VWF", "KDR", "ENG", "EMCN", "ESAM", "CDH5", "CLDN5", "RAMP2"]),
        ),
        CellStateSignature(
            "b_cells",
            "B cells",
            "immune",
            "lower_risk",
            "curated_luad_marker_set_v1",
            "B-cell lineage markers.",
            _genes(["MS4A1", "CD79A", "CD79B", "BANK1", "CD19", "CD22", "HLA-DRA"]),
        ),
        CellStateSignature(
            "plasma_cells",
            "plasma cells",
            "immune",
            "lower_risk",
            "curated_luad_marker_set_v1",
            "Plasma-cell and immunoglobulin-production markers.",
            _genes(["MZB1", "XBP1", "JCHAIN", "IGKC", "IGHG1", "SDC1", "PRDM1"]),
        ),
        CellStateSignature(
            "nk_cells",
            "NK cells",
            "immune",
            "lower_risk",
            "curated_luad_marker_set_v1",
            "Natural killer cell cytotoxicity markers.",
            _genes(["NCR1", "KLRD1", "NKG7", "GNLY", "PRF1", "GZMB", "FCGR3A", "KLRF1"]),
        ),
        CellStateSignature(
            "mast_cells",
            "mast cells",
            "immune",
            "context_dependent",
            "curated_luad_marker_set_v1",
            "Mast-cell markers.",
            _genes(["TPSAB1", "TPSB2", "CPA3", "KIT", "MS4A2", "HDC", "CMA1"]),
        ),
    ]


def signatures_to_frame(signatures: Iterable[CellStateSignature]) -> pd.DataFrame:
    """Convert signatures into a flat table for audit and reuse."""

    rows = []
    for signature in signatures:
        row = asdict(signature)
        row["genes"] = ";".join(signature.genes)
        row["n_genes"] = len(signature.genes)
        rows.append(row)
    columns = [
        "signature_name",
        "cell_state",
        "category",
        "expected_risk_direction",
        "source",
        "description",
        "n_genes",
        "genes",
    ]
    return pd.DataFrame(rows, columns=columns)


def frame_to_signatures(frame: pd.DataFrame) -> list[CellStateSignature]:
    """Load signatures from a definition table."""

    required = {
        "signature_name",
        "cell_state",
        "category",
        "expected_risk_direction",
        "source",
        "description",
        "genes",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SCRNASignatureError(f"Signature table is missing columns: {missing}")
    signatures = []
    for row in frame.itertuples(index=False):
        genes = _genes(str(getattr(row, "genes")).split(";"))
        if not genes:
            continue
        signatures.append(
            CellStateSignature(
                signature_name=str(getattr(row, "signature_name")),
                cell_state=str(getattr(row, "cell_state")),
                category=str(getattr(row, "category")),
                expected_risk_direction=str(getattr(row, "expected_risk_direction")),
                source=str(getattr(row, "source")),
                description=str(getattr(row, "description")),
                genes=genes,
            )
        )
    if not signatures:
        raise SCRNASignatureError("No usable signatures were found in the definition table.")
    return signatures


def save_signature_artifacts(
    signatures: Iterable[CellStateSignature],
    table_path: str | Path,
    json_path: str | Path | None = None,
) -> pd.DataFrame:
    """Write signature definitions as CSV and optional JSON."""

    frame = signatures_to_frame(signatures)
    table = Path(table_path)
    table.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(table, index=False)
    if json_path is not None:
        path = Path(json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(frame.to_dict(orient="records"), indent=2), encoding="utf-8")
    return frame


def load_signature_artifacts(path: str | Path | None = None) -> list[CellStateSignature]:
    """Load saved signatures, falling back to the pre-specified curated set."""

    if path is None:
        return curated_luad_cell_state_signatures()
    table = Path(path)
    if not table.exists():
        return curated_luad_cell_state_signatures()
    return frame_to_signatures(pd.read_csv(table))


def signatures_from_marker_table(
    marker_path: str | Path,
    *,
    cell_type_col: str = "cell_type",
    gene_col: str = "gene_symbol",
    score_col: str | None = None,
    top_n: int = 50,
) -> list[CellStateSignature]:
    """Build signatures from a local CSV/TSV marker table."""

    path = Path(marker_path)
    if not path.exists():
        raise FileNotFoundError(f"Single-cell marker table not found: {path}")
    separator = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    frame = pd.read_csv(path, sep=separator)
    required = {cell_type_col, gene_col}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SCRNASignatureError(f"Marker table is missing columns: {missing}")
    frame = frame.copy()
    frame[gene_col] = frame[gene_col].map(normalize_gene_symbol)
    frame = frame.loc[frame[gene_col] != ""]
    if score_col and score_col in frame.columns:
        frame["_marker_rank"] = pd.to_numeric(frame[score_col], errors="coerce")
        frame = frame.sort_values([cell_type_col, "_marker_rank", gene_col], ascending=[True, False, True])
    else:
        frame = frame.sort_values([cell_type_col, gene_col])
    signatures = []
    for cell_type, group in frame.groupby(cell_type_col, sort=True):
        genes = _genes(group[gene_col].head(top_n).tolist())
        if not genes:
            continue
        name = str(cell_type).strip().lower().replace(" ", "_").replace("/", "_")
        signatures.append(
            CellStateSignature(
                signature_name=name,
                cell_state=str(cell_type),
                category="local_single_cell_marker",
                expected_risk_direction="unknown",
                source=f"local_marker_table:{path.name}",
                description="Local single-cell marker table; not derived from survival outcomes.",
                genes=genes,
            )
        )
    if not signatures:
        raise SCRNASignatureError(f"No usable markers were found in {path}")
    return signatures


def signatures_from_h5ad(
    h5ad_path: str | Path,
    *,
    cell_type_col: str = "cell_type",
    top_n: int = 50,
) -> list[CellStateSignature]:
    """Build simple cell-type signatures from a local AnnData h5ad file."""

    path = Path(h5ad_path)
    if not path.exists():
        raise FileNotFoundError(f"Single-cell h5ad not found: {path}")
    try:
        import anndata as ad  # type: ignore
    except ImportError as exc:
        raise SCRNASignatureError(
            "Reading h5ad requires anndata. Install it or provide a marker CSV."
        ) from exc
    adata = ad.read_h5ad(path)
    if cell_type_col not in adata.obs:
        raise SCRNASignatureError(
            f"h5ad obs is missing cell type column {cell_type_col!r}."
        )
    genes = [normalize_gene_symbol(gene) for gene in adata.var_names]
    valid_gene_mask = np.asarray([gene != "" for gene in genes])
    if not valid_gene_mask.any():
        raise SCRNASignatureError("h5ad var_names contain no usable gene symbols.")
    matrix = adata.X
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    matrix = np.asarray(matrix[:, valid_gene_mask], dtype=float)
    valid_genes = np.asarray(genes, dtype=object)[valid_gene_mask]
    labels = adata.obs[cell_type_col].astype(str).to_numpy()
    signatures = []
    for label in sorted(set(labels)):
        in_group = labels == label
        if int(in_group.sum()) < 3:
            continue
        group_mean = matrix[in_group].mean(axis=0)
        rest_mean = matrix[~in_group].mean(axis=0) if int((~in_group).sum()) else np.zeros_like(group_mean)
        score = group_mean - rest_mean
        order = np.argsort(-score)
        selected = _genes(valid_genes[order[:top_n]].tolist())
        if not selected:
            continue
        name = label.strip().lower().replace(" ", "_").replace("/", "_")
        signatures.append(
            CellStateSignature(
                signature_name=name,
                cell_state=label,
                category="local_single_cell_h5ad",
                expected_risk_direction="unknown",
                source=f"local_h5ad:{path.name}",
                description="Local h5ad marker set from mean expression contrast; not derived from survival outcomes.",
                genes=selected,
            )
        )
    if not signatures:
        raise SCRNASignatureError(f"No usable cell-type signatures were derived from {path}")
    return signatures


def build_signatures_from_source(
    source_path: str | Path | None,
    *,
    cell_type_col: str = "cell_type",
    gene_col: str = "gene_symbol",
    score_col: str | None = None,
    top_n: int = 50,
) -> tuple[list[CellStateSignature], str]:
    """Build signatures from a local source or return curated signatures."""

    if source_path is None:
        return curated_luad_cell_state_signatures(), "curated"
    path = Path(source_path)
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".h5ad"):
        return signatures_from_h5ad(path, cell_type_col=cell_type_col, top_n=top_n), "local_h5ad"
    if suffixes.endswith(".rds") or suffixes.endswith(".rda"):
        raise SCRNASignatureError(
            "RDS input is not read directly by this Python workflow. Export marker genes "
            "to CSV/TSV with cell_type and gene_symbol columns, or convert the object to h5ad."
        )
    return (
        signatures_from_marker_table(
            path,
            cell_type_col=cell_type_col,
            gene_col=gene_col,
            score_col=score_col,
            top_n=top_n,
        ),
        "local_marker_table",
    )

