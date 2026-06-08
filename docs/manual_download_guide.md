# Manual Download Guide

Stage 0 does not download biomedical data. This guide records datasets that may require manual access, account approval, browser-based download, or source-specific terms.

## General Rules

- Do not place private hospital data in this project.
- Put raw manually downloaded files under `data/raw/<dataset>/<modality>/`.
- Record every manual file in `data/metadata/data_manifest_template.csv` or the Stage 1 manifest generated from it.
- Keep original filenames when practical.
- Do not commit raw data.

## Planned Datasets

| Dataset | Modality | Stage | Manual status |
| --- | --- | --- | --- |
| TCGA-LUAD | Clinical, survival, RNA-seq, mutation, CNV, methylation | 1/3 | To be checked by public API or documented if manual download is required |
| TCGA-LUAD | Diagnostic WSI | 6 | To be checked; if automated download is restricted, record browser/manual instructions |
| Public LUAD single-cell datasets | scRNA-seq | 4 | Dataset selection and access rules will be recorded in Stage 4 |
| CPTAC LUAD | Proteomics | 5 | Access path and local format will be recorded in Stage 5 |
| Human Protein Atlas | Protein expression/IHC | 5 | Public evidence tables/images will be linked or documented in Stage 5 |
| GEO/CPTAC/ICGC validation cohorts | External validation | 2/5 | Availability and expected formats will be documented per cohort |

## Placeholder Local Format

For datasets that cannot be downloaded automatically, create a subfolder with:

```text
README.md
manifest.csv
raw_files/
```

Minimum `manifest.csv` columns:

```text
file_name,modality,source,url_or_accession,download_date,checksum,notes
```
