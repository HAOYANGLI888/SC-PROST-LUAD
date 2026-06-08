"""GEO expression import helpers for future frozen-model validation."""

from __future__ import annotations

import csv
import gzip
import io
import re
from pathlib import Path
from typing import TextIO

import numpy as np
import pandas as pd


class GEOExpressionImportError(RuntimeError):
    """Raised when a GEO expression file cannot be imported safely."""


def normalize_gene_symbol(value: object) -> str:
    """Normalize external gene symbols without silently inventing mappings."""

    text = str(value or "").strip().strip('"').upper()
    return text.split("///")[0].strip()


def _open_text(path: Path) -> TextIO:
    return gzip.open(path, "rt", encoding="utf-8", errors="replace") if path.suffix.lower() == ".gz" else path.open("r", encoding="utf-8", errors="replace")


def _read_series_matrix(path: Path) -> pd.DataFrame:
    with _open_text(path) as handle:
        lines = handle.readlines()
    begin = next((index for index, line in enumerate(lines) if line.startswith("!series_matrix_table_begin")), None)
    end = next((index for index, line in enumerate(lines) if line.startswith("!series_matrix_table_end")), None)
    if begin is None or end is None or end <= begin + 1:
        raise GEOExpressionImportError(f"GEO series matrix table markers were not found: {path}")
    return pd.read_csv(io.StringIO("".join(lines[begin + 1 : end])), sep="\t")


def _parse_geo_values(line: str) -> list[str]:
    return next(csv.reader([line], delimiter="\t", quotechar='"'))[1:]


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().casefold()).strip("_")


def read_geo_series_metadata(path: str | Path) -> tuple[pd.DataFrame, dict[str, object]]:
    """Parse sample characteristics embedded in an official GEO series matrix."""

    matrix_path = Path(path)
    if not matrix_path.exists():
        raise FileNotFoundError(f"GEO series matrix not found: {matrix_path}")
    sample_ids: list[str] | None = None
    platform_ids: list[str] = []
    fields: dict[str, list[str]] = {}
    with _open_text(matrix_path) as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n\r")
            if line.startswith("!series_matrix_table_begin"):
                break
            if line.startswith("!Sample_geo_accession"):
                sample_ids = _parse_geo_values(line)
            elif line.startswith("!Sample_platform_id"):
                platform_ids = _parse_geo_values(line)
            elif line.startswith("!Sample_characteristics_ch1"):
                values = _parse_geo_values(line)
                for index, value in enumerate(values):
                    if not value.strip():
                        continue
                    key, separator, parsed_value = value.partition(":")
                    base_key = _slug(key) if separator else "characteristic"
                    parsed_value = parsed_value.strip() if separator else value.strip()
                    resolved_key = base_key
                    suffix = 1
                    while (
                        resolved_key in fields
                        and fields[resolved_key][index] != ""
                    ):
                        suffix += 1
                        resolved_key = f"{base_key}_{suffix}"
                    fields.setdefault(resolved_key, [""] * len(values))[index] = parsed_value
    if not sample_ids:
        raise GEOExpressionImportError(f"Missing !Sample_geo_accession in {matrix_path}")
    for key, values in fields.items():
        if len(values) != len(sample_ids):
            raise GEOExpressionImportError(
                f"Metadata field {key!r} has {len(values)} values for "
                f"{len(sample_ids)} samples in {matrix_path}"
            )
    return pd.DataFrame({"sample_id": sample_ids, **fields}), {
        "sample_count": len(sample_ids),
        "platform_ids": sorted(set(platform_ids)),
        "metadata_fields": sorted(fields),
    }


def read_geo_expression(path: str | Path) -> pd.DataFrame:
    """Read a GEO series matrix or user-supplied probe-by-sample table."""

    expression_path = Path(path)
    if not expression_path.exists():
        raise FileNotFoundError(f"GEO expression file not found: {expression_path}")
    if expression_path.stat().st_size == 0:
        raise GEOExpressionImportError(f"GEO expression file is empty: {expression_path}")
    if "series_matrix" in expression_path.name.lower():
        frame = _read_series_matrix(expression_path)
    else:
        separator = "\t" if expression_path.suffix.lower() in {".tsv", ".txt", ".gz"} else ","
        frame = pd.read_csv(expression_path, sep=separator)
    if frame.empty or frame.shape[1] < 2:
        raise GEOExpressionImportError("GEO expression table must contain probe IDs and sample columns.")
    first = frame.columns[0]
    frame = frame.rename(columns={first: "probe_id"})
    frame["probe_id"] = frame["probe_id"].astype(str).str.strip().str.strip('"')
    for column in frame.columns[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def read_probe_annotation(path: str | Path) -> pd.DataFrame:
    """Read a probe-to-gene-symbol map with common column aliases."""

    annotation_path = Path(path)
    if not annotation_path.exists():
        raise FileNotFoundError(f"GEO probe annotation file not found: {annotation_path}")
    separator = "\t" if annotation_path.suffix.lower() in {".tsv", ".txt", ".gz"} else ","
    frame = pd.read_csv(annotation_path, sep=separator, dtype=str)
    aliases = {
        "probe_id": ("probe_id", "ID", "ID_REF", "Probe Set ID", "probe"),
        "gene_symbol": ("gene_symbol", "Gene symbol", "Gene Symbol", "GeneSymbol", "GENE_SYMBOL", "Symbol", "symbol"),
    }
    resolved = {}
    for field, options in aliases.items():
        resolved[field] = next((column for column in options if column in frame.columns), None)
        if resolved[field] is None:
            raise GEOExpressionImportError(
                f"Probe annotation is missing {field}. Observed columns: {frame.columns.tolist()}"
            )
    table = frame[[resolved["probe_id"], resolved["gene_symbol"]]].rename(
        columns={resolved["probe_id"]: "probe_id", resolved["gene_symbol"]: "gene_symbol"}
    )
    table["probe_id"] = table["probe_id"].astype(str).str.strip().str.strip('"')
    table["gene_symbol"] = table["gene_symbol"].map(normalize_gene_symbol)
    table = table.loc[table["gene_symbol"] != ""].drop_duplicates()
    if table.empty:
        raise GEOExpressionImportError("Probe annotation contains no usable gene symbols.")
    return table


def collapse_probes_to_genes(
    expression: pd.DataFrame,
    annotation: pd.DataFrame,
    *,
    strategy: str = "mean",
) -> pd.DataFrame:
    """Collapse multiple probes per symbol and return samples x genes."""

    if strategy not in {"mean", "median", "max_variance"}:
        raise ValueError("Probe-collapse strategy must be 'mean', 'median', or 'max_variance'.")
    merged = annotation.merge(expression, on="probe_id", how="inner", validate="many_to_many")
    if merged.empty:
        raise GEOExpressionImportError("No expression probes overlap the supplied annotation.")
    sample_columns = [column for column in expression.columns if column != "probe_id"]
    if strategy in {"mean", "median"}:
        grouped = merged.groupby("gene_symbol", as_index=True)[sample_columns]
        collapsed = grouped.mean() if strategy == "mean" else grouped.median()
    else:
        variances = merged[sample_columns].var(axis=1, skipna=True)
        selected = merged.assign(_variance=variances).sort_values(
            ["gene_symbol", "_variance", "probe_id"],
            ascending=[True, False, True],
        ).drop_duplicates("gene_symbol")
        collapsed = selected.set_index("gene_symbol")[sample_columns]
    result = collapsed.T
    result.index.name = "sample_id"
    result = result.reset_index()
    return result


def zscore_external_dataset(expression: pd.DataFrame) -> pd.DataFrame:
    """Z-score each gene within one external cohort only."""

    if "sample_id" not in expression:
        raise GEOExpressionImportError("External expression table must contain sample_id.")
    genes = [column for column in expression.columns if column != "sample_id"]
    numeric = expression[genes].apply(pd.to_numeric, errors="coerce")
    medians = numeric.median(axis=0).fillna(0.0)
    numeric = numeric.fillna(medians)
    scales = numeric.std(axis=0, ddof=0).replace(0.0, 1.0)
    zscore = (numeric - numeric.mean(axis=0)) / scales
    return pd.concat([expression[["sample_id"]].reset_index(drop=True), zscore.reset_index(drop=True)], axis=1)


def prepare_geo_expression(
    expression_path: str | Path,
    annotation_path: str | Path,
    *,
    collapse_strategy: str = "mean",
) -> pd.DataFrame:
    """Import, collapse, and independently standardize one GEO cohort."""

    expression = read_geo_expression(expression_path)
    annotation = read_probe_annotation(annotation_path)
    collapsed = collapse_probes_to_genes(expression, annotation, strategy=collapse_strategy)
    return zscore_external_dataset(collapsed)


def export_tcga_gene_annotation_from_star_counts(root: str | Path = ".") -> Path:
    """Export Ensembl-to-symbol annotation from one downloaded GDC STAR-counts file."""

    project_root = Path(root).resolve()
    output = project_root / "data" / "metadata" / "stage2c_tcga_gene_annotation.csv"
    files = sorted((project_root / "data" / "raw" / "tcga_luad" / "rnaseq" / "gdc_star_counts").rglob("*.tsv"))
    if not files:
        raise FileNotFoundError("No downloaded GDC STAR-counts TSV file is available for TCGA annotation export.")
    frame = pd.read_csv(files[0], sep="\t", comment="#", dtype=str)
    required = {"gene_id", "gene_name", "gene_type"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise GEOExpressionImportError(f"GDC STAR-counts annotation columns changed or are missing: {missing}")
    annotation = frame[["gene_id", "gene_name", "gene_type"]].copy()
    annotation["gene_id"] = annotation["gene_id"].str.replace(r"\.\d+$", "", regex=True)
    annotation = annotation.loc[annotation["gene_id"].str.match(r"^ENSG", na=False)]
    annotation = annotation.rename(columns={"gene_name": "gene_symbol"})
    annotation["gene_symbol"] = annotation["gene_symbol"].map(normalize_gene_symbol)
    annotation = annotation.drop_duplicates("gene_id").sort_values("gene_id")
    output.parent.mkdir(parents=True, exist_ok=True)
    annotation.to_csv(output, index=False)
    return output


def missing_fixed_model_gene_fraction(
    expression: pd.DataFrame,
    annotation_path: str | Path,
    required_ensembl_genes: list[str],
) -> float:
    """Calculate the fraction of frozen TCGA Ensembl genes absent from GEO symbols."""

    annotation = pd.read_csv(annotation_path)
    symbol_by_gene = {
        str(row.gene_id): normalize_gene_symbol(row.gene_symbol)
        for row in annotation.itertuples()
    }
    available_symbols = {normalize_gene_symbol(column) for column in expression.columns if column != "sample_id"}
    required_symbols = {symbol_by_gene.get(gene, "") for gene in required_ensembl_genes}
    required_symbols.discard("")
    if not required_symbols:
        return 1.0
    return 1.0 - len(required_symbols & available_symbols) / len(required_symbols)
