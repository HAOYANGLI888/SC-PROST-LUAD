# TCGA-LUAD Manual Download Guide

Stage 1 audits public metadata only. It does not pretend that large data files
have been downloaded. Use this guide in later stages when analysis-ready files
are required.

## Automatic Metadata Audit

Run from PowerShell:

```powershell
cd SC-PROST-LUAD
conda activate gpu_py310
python scripts/stage1_audit_tcga_luad.py
```

The audit uses:

- GDC `cases` metadata for public TCGA-LUAD clinical case availability.
- GDC `files` metadata for RNA-seq, masked somatic MAF, gene-level CNV,
  methylation beta values, and diagnostic slide SVS availability.
- The public UCSC Xena TCGA Pan-Cancer Clinical Data Resource survival table
  for curated OS, DSS, and PFI endpoints.

The small TCGA-CDR text table is cached under:

```text
data/raw/tcga_luad/clinical/Survival_SupplementalTable_S1_20171025_xena_sp.tsv
```

Existing cache files are reused. Add `--refresh-survival-cache` only when a
fresh download is intentional.

## Large Files Deferred to Later Stages

Do not commit any downloaded file below `data/raw/`.

| Modality | Planned local directory | Later-stage action |
| --- | --- | --- |
| Clinical and survival | `data/raw/tcga_luad/clinical/` | Keep source table and derived clinical table |
| RNA-seq | `data/raw/tcga_luad/rnaseq/` | Download selected tumor gene-expression quantification files or an approved public matrix |
| Mutation MAF | `data/raw/tcga_luad/mutation/` | Download open masked somatic MAF files |
| Gene-level CNV | `data/raw/tcga_luad/cnv/` | Download selected gene-level copy-number files |
| Methylation | `data/raw/tcga_luad/methylation/` | Download beta-value files only when Stage 3 starts |
| Diagnostic WSI | `data/raw/tcga_luad/wsi/` | Download selected `.svs` slides only when Stage 6 starts |

## Download Options

For large GDC file sets, create a manifest from the selected Stage 1 cohort and
use the GDC Data Transfer Tool on Windows. The exact manifest generator belongs
to the stage that consumes each modality, so Stage 1 does not generate a
download-everything command.

GDC Portal:

```text
https://portal.gdc.cancer.gov/
```

GDC Data Transfer Tool documentation:

```text
https://gdc.cancer.gov/access-data/gdc-data-transfer-tool
```

## Local Manifest Template

Record manual files with:

```text
file_name,file_id,patient_id,modality,source_url,local_path,checksum,status,notes
```

## Important Limitations

- Raw sequencing reads and some protected data are not needed for the planned
  public-data workflow.
- WSI files are large. Metadata availability is not evidence that all slides
  have already downloaded successfully.
- Keep download scripts restartable: check file existence and checksums before
  fetching again.
