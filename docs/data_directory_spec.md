# Data Directory Specification

Use relative paths from the project root and `pathlib.Path` in Python code.

## Raw Data

Raw files are stored under `data/raw/` and are not committed.

Suggested future layout:

```text
data/raw/
  tcga_luad/
    clinical/
    survival/
    rnaseq/
    mutation/
    cnv/
    methylation/
    wsi/
  external/
    geo/
    cptac/
    icgc/
    hpa/
  single_cell/
```

## Processed Data

Processed files are stored under `data/processed/` and are not committed.

Suggested future layout:

```text
data/processed/
  tcga_luad/
    clinical_survival/
    rnaseq/
    multiomics/
    wsi_features/
  external_validation/
  single_cell_signatures/
```

## Metadata

Small metadata files are stored under `data/metadata/` and may be committed:

- `data_manifest_template.csv`
- `patient_manifest_template.csv`
- `stage0_directory_manifest.csv`
- Stage 1 patient-level modality matching tables

## Output Directories

- Tables: `outputs/tables/`
- Figures: `outputs/figures/`
- Checkpoints: `outputs/checkpoints/`
- Logs: `outputs/logs/`
- Human-readable reports: `outputs/reports/`
- Audit reports: `outputs/audit/<stage>/audit_report.md`
