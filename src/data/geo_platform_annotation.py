"""Parse GEO platform annotations into probe-to-gene mappings."""

from __future__ import annotations

import gzip
import io
from pathlib import Path

import pandas as pd


PROBE_ALIASES = ("ID", "ID_REF", "Probe Set ID", "probe_id")
SYMBOL_ALIASES = (
    "Gene symbol", "Gene Symbol", "GeneSymbol", "GENE_SYMBOL", "gene_symbol",
)


def _open_text(path: Path):
    if str(path).lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("rt", encoding="utf-8", errors="replace")


def _first_symbol(value: object) -> str | None:
    if pd.isna(value):
        return None
    symbol = str(value).strip()
    if not symbol or symbol in {"---", "NA", "nan"}:
        return None
    for separator in (" /// ", " // ", "|", ";", ","):
        if separator in symbol:
            symbol = symbol.split(separator, maxsplit=1)[0].strip()
    return symbol.upper() or None


def _find_column(columns: list[str], aliases: tuple[str, ...], label: str) -> str:
    lookup = {column.casefold(): column for column in columns}
    for alias in aliases:
        if alias.casefold() in lookup:
            return lookup[alias.casefold()]
    raise ValueError(f"Could not find {label} column. Available columns: {columns}")


def parse_geo_platform_annotation(path: Path) -> pd.DataFrame:
    """Return a normalized ``probe_id, gene_symbol`` mapping."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"GEO platform annotation not found: {path}")
    lines: list[str] = []
    inside = False
    with _open_text(path) as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n\r")
            if line.startswith("!platform_table_begin"):
                inside = True
                continue
            if line.startswith("!platform_table_end"):
                break
            if inside:
                lines.append(line)
    if not lines:
        raise ValueError(f"No platform table found in {path}")
    frame = pd.read_csv(io.StringIO("\n".join(lines)), sep="\t", dtype=str)
    probe = _find_column(list(frame.columns), PROBE_ALIASES, "probe ID")
    symbol = _find_column(list(frame.columns), SYMBOL_ALIASES, "gene symbol")
    result = frame[[probe, symbol]].rename(
        columns={probe: "probe_id", symbol: "gene_symbol"}
    )
    result["probe_id"] = result["probe_id"].astype(str).str.strip()
    result["gene_symbol"] = result["gene_symbol"].map(_first_symbol)
    result = result.dropna().drop_duplicates("probe_id", keep="first")
    if result.empty:
        raise ValueError(f"Platform mapping is empty after normalization: {path}")
    return result.reset_index(drop=True)

