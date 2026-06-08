# Stage 5 CPTAC/PDC Manual Download Guide

This guide documents how to add CPTAC/PDC protein abundance data for Stage 5
orthogonal protein validation.

## Scope

- CPTAC/PDC is optional orthogonal proteomic support.
- Do not retrain the Stage 2D risk model with protein data.
- Do not use protein data to revise Stage 4 signatures.
- Do not claim CPTAC validation unless a compatible protein abundance matrix is
  available and analyzed.

## Suggested Sources

- CPTAC LUAD protein abundance tables from CPTAC/PDC.
- Matched clinical or survival metadata only if publicly available and clearly
  mapped to protein samples.
- Matched transcriptomic data only if sample IDs can be safely aligned.

## Local File Placement

Place a local matrix in one of the following locations:

- `data/raw/cptac_luad/protein_abundance.csv`
- `data/raw/cptac_luad/protein_abundance.tsv`
- `data/processed/cptac_luad/protein_abundance.csv`

The Stage 5 script also accepts an explicit path:

```powershell
python scripts/stage5_cptac_validation.py --config configs/base.yaml --protein-matrix "D:\path\to\protein_abundance.csv"
```

## Accepted Matrix Formats

Preferred samples x proteins:

```text
sample_id,MKI67,TOP2A,CA9,VEGFA,...
CPTAC_001,0.52,1.18,-0.20,0.44,...
```

Alternative proteins x samples:

```text
gene_symbol,CPTAC_001,CPTAC_002,CPTAC_003
MKI67,0.52,-0.11,1.03
TOP2A,1.18,0.20,0.91
```

## Optional Metadata

If survival or risk-like groups are available, prepare a separate metadata file
with:

```text
sample_id,risk_score,risk_group,os_time_days,os_event
```

The current Stage 5 implementation does not force CPTAC survival modeling when
the schema is not curated. Unavailable matched data must remain marked
`manual_download_required` or `not_available`.

