# Stage 5B CPTAC-LUAD Download / Import Guide

Stage 5B requires a real CPTAC/PDC LUAD protein abundance matrix for formal
quantitative proteomic validation. Missing data must remain marked
`manual_download_required`; unavailable is not negative evidence.

Official PDC resources:

- PDC portal: https://pdc.cancer.gov/pdc/
- PDC open API: https://pdc.cancer.gov/pdc/publicapi-documentation

The Stage 5B inventory script can query PDC metadata and write candidate
remote files to:

- `outputs/tables/stage5b_pdc_luad_remote_file_candidates.csv`

This remote table is a download lead only. It is not a local abundance matrix
and must not be reported as completed CPTAC validation.

## Required Files

Place files under:

- `data/raw/cptac_luad/`

Required roles:

- protein abundance matrix
- clinical or survival metadata
- sample annotation or sample-to-case mapping

## PDC LUAD Proteome Leads

The following PDC proteome studies were observed by the PDC study catalog query
and are suitable starting points for manual review:

- `PDC000153`: CPTAC LUAD Discovery Study - Proteome
- `PDC000489`: CPTAC LUAD Confirmatory Study - Proteome
- `PDC000219`: Academia Sinica LUAD100-Proteome
- `PDC000434`: APOLLO LUAD - Proteome
- `PDC000563`: Academia Sinica LUAD ICPC-B - Proteome

For each study, prioritize report files with roles like:

- `protein_quantitation_candidate`, usually `*.tmt10.tsv` or `*.tmt11.tsv`
- `sample_annotation_candidate`, usually `*.sample.txt`
- supporting metadata such as `*.label.txt`

Do not use `summary.tsv` alone as a quantitative abundance matrix without
confirming the columns are quantitative abundance values rather than spectral
counts, peptide counts, or report metadata.

The import script can also copy local files into the raw directory:

```powershell
python scripts/stage5b_cptac_download_or_import.py --config configs/base.yaml `
  --protein-matrix "D:\path\to\protein_abundance.csv" `
  --clinical-metadata "D:\path\to\clinical.csv" `
  --sample-annotation "D:\path\to\sample_annotation.csv"
```

## Accepted Protein Matrix Formats

Preferred samples x proteins:

```text
sample_id,MKI67,TOP2A,CA9,VEGFA,...
CPTAC_001,0.12,1.44,-0.31,0.52,...
```

Alternative proteins x samples:

```text
gene_symbol,CPTAC_001,CPTAC_002,CPTAC_003
MKI67,0.12,-0.20,1.05
TOP2A,1.44,0.55,0.91
```

## Clinical Metadata

Recommended columns:

```text
sample_id,risk_score,risk_group,os_time_days,os_event,age,male,stage_numeric
```

If `risk_score` is not available, Stage 5B will fall back to `stage_numeric`
for direction checks when possible. If survival fields are absent, survival
analysis is reported as unavailable.

## Run Order

```powershell
python scripts/stage5b_cptac_download_or_import.py --config configs/base.yaml
python scripts/stage5b_cptac_preprocess.py --config configs/base.yaml
python scripts/stage5b_cptac_candidate_validation.py --config configs/base.yaml
python scripts/stage5b_cptac_survival_analysis.py --config configs/base.yaml
python scripts/stage5b_integrate_cptac_evidence.py --config configs/base.yaml
```

## Integrity Rules

- Do not use CPTAC protein data to retrain or tune the Stage 2D model.
- Do not use CPTAC results to revise Stage 4 signatures.
- Do not claim causal confirmation.
- Do not claim CPTAC validation unless a real compatible matrix is present.
