# Stage 2 TCGA-LUAD RNA-seq Manual Download Guide

Stage 2 needs one public, analysis-ready TCGA-LUAD tumor RNA-seq expression
matrix. The repository intentionally does not include large expression files.

## Required Local File

Default location:

```text
data/raw/tcga_luad/rnaseq/tcga_luad_tpm_matrix.csv
```

Accepted wide CSV format:

```text
patient_id,TP53,KRAS,EGFR,...
TCGA-05-4244-01A,12.4,8.2,1.7,...
TCGA-05-4249-01A,10.1,3.4,2.0,...
```

The first column may contain TCGA patient or sample barcodes. Code converts
them to the first 12 barcode characters. Gene-by-sample TSV layout is also
accepted and transposed automatically when sample barcodes are in columns.

Values should be TPM by default. The preparation script applies
`log2(TPM + 1)`. For an already transformed matrix, pass:

```powershell
python scripts/stage2_prepare_rnaseq_survival.py `
  --config configs/base.yaml `
  --rnaseq-matrix data/raw/tcga_luad/rnaseq/tcga_luad_log2_tpm.tsv `
  --input-scale log2_tpm
```

## Public Acquisition Options

Preferred reproducible route:

1. Use the Stage 1 matrix to define the eligible TCGA-LUAD patient cohort.
2. Obtain public tumor gene-expression quantification files from the GDC
   Portal or generate a selected-file manifest from GDC metadata.
3. Convert gene-level values into one patient-row TPM matrix.
4. Keep the source manifest and transformation notes under
   `data/metadata/`.

GDC Portal:

```text
https://portal.gdc.cancer.gov/
```

GDC Data Transfer Tool:

```text
https://gdc.cancer.gov/access-data/gdc-data-transfer-tool
```

An analysis-ready public matrix from UCSC Xena may also be used if its
normalization, cohort source, and download date are recorded.

## Windows Commands

Preferred public GDC STAR-counts route:

```powershell
cd SC-PROST-LUAD
conda activate gpu_py310
python scripts/stage2_build_gdc_rnaseq_manifest.py
python scripts/stage2_download_gdc_rnaseq.py --method direct-api
python scripts/stage2_build_rnaseq_tpm_matrix.py
python scripts/stage2_prepare_rnaseq_survival.py --config configs/base.yaml
python scripts/train_stage2_baselines.py --config configs/base.yaml --seed 42
python scripts/train_stage2_baselines.py --config configs/base.yaml --seed 3407
python scripts/train_stage2_baselines.py --config configs/base.yaml --seed 2026
python scripts/evaluate_stage2_models.py --config configs/base.yaml
```

The download helper also accepts the official Windows transfer client:

```powershell
python scripts/stage2_download_gdc_rnaseq.py `
  --manifest data/metadata/gdc_tcga_luad_rnaseq_star_counts_manifest.tsv `
  --gdc-client "C:\path\to\gdc-client.exe" `
  --method gdc-client
```

Both download methods reuse completed files. Direct API mode preserves partial
files with a `.part` suffix and resumes when the server supports byte ranges.

## Optional UCSC Xena Fallback

Keep Xena fallback files separate from the GDC primary analysis:

```powershell
python scripts/stage2_import_xena_rnaseq_matrix.py `
  --source-matrix C:\path\to\xena_matrix.tsv `
  --input-scale log2_tpm
```

The importer writes under `data/raw/tcga_luad/rnaseq/xena/` and records
`source = UCSC_Xena`. It never overwrites the GDC primary matrix.

## Leakage Rule

Do not globally select highly variable genes before splitting patients.
Stage 2 stores candidate genes during preparation, then fits low-expression
filtering, missingness filtering, variable-gene selection, imputation, and
standardization separately on each seed's training split.
