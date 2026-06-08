# Stage 6A TCGA-LUAD WSI Download Guide

Stage 6A audits all public diagnostic SVS files but intentionally does not
download the full WSI cohort by default.

## 1. Build The Public Metadata Manifest

```powershell
conda activate gpu_py310
python scripts/stage6a_build_wsi_manifest.py --config configs/base.yaml
```

This writes:

- `data/metadata/stage6a_tcga_luad_wsi_manifest.tsv`
- `data/metadata/stage6a_tcga_luad_wsi_patient_slide_map.csv`
- `outputs/tables/stage6a_wsi_query_summary.csv`

## 2. Estimate A Balanced 20-Slide Smallset

```powershell
python scripts/stage6a_download_wsi_smallset.py --config configs/base.yaml --n-slides 20 --select-only
```

Review `data/metadata/stage6a_wsi_smallset_download_status.csv` before downloading.
The selected slide files may still require several GB.

## 3. Download Only The Smallset

Public GDC API with resume support:

```powershell
python scripts/stage6a_download_wsi_smallset.py --config configs/base.yaml --n-slides 20
```

Optional GDC Data Transfer Tool:

```powershell
python scripts/stage6a_download_wsi_smallset.py --config configs/base.yaml --n-slides 20 --gdc-client "C:\path\to\gdc-client.exe"
```

Install the Windows client from:
https://gdc.cancer.gov/access-data/gdc-data-transfer-tool

The script skips completed files, retains `.part` files for interrupted direct
downloads, records failures, and never reports failed files as downloaded.

## Full-Cohort Planning

The full diagnostic-slide set is not a Stage 6A default. Budget raw SVS storage,
derived embedding storage, backup capacity, download time, and GPU extraction
time before starting a full download.

