# Stage 4B LUAD scRNA-seq Manual Download Guide

## Preferred dataset

The preferred formal dataset is **GSE131907**, an official NCBI GEO lung cancer
single-cell dataset with author-provided cell annotations.

Official GEO page:

https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE131907

Official supplementary directory:

https://ftp.ncbi.nlm.nih.gov/geo/series/GSE131nnn/GSE131907/suppl/

Required files:

1. `GSE131907_Lung_Cancer_cell_annotation.txt.gz`
2. `GSE131907_Lung_Cancer_raw_UMI_matrix.rds.gz`

Place both files in:

```text
data/raw/scrna_luad/GSE131907/
```

The Python downloader supports HTTP range-based resume:

```powershell
conda activate gpu_py310
python scripts/stage4b_scrna_download_or_import.py --config configs/base.yaml
```

If the NCBI connection is interrupted, rerun the same command. Existing complete
files and completed range parts are retained.

## RDS conversion

The downloaded GSE131907 artifact has two gzip layers. The project validates
the outer stream and writes the inner gzip payload to:

```text
data/raw/scrna_luad/GSE131907/.stage4b_cache/
GSE131907_Lung_Cancer_raw_UMI_matrix.rds.inner.gz
```

The RDS is a dense R `data.frame` with 29,634 genes and 208,506 cells. Its
in-memory size is about 24.7 GB. Do not use `as.matrix()` on a 32 GB workstation:
the temporary dense copy plus sparse output can exhaust physical and virtual
memory. Use a workstation with at least 64 GB RAM (96 GB preferred), or perform
a chunked HDF5 conversion.

### High-memory Seurat route

On a sufficiently large workstation, install `Seurat`, `SeuratDisk`, and
`Matrix`, then run:

```r
library(Seurat)
library(SeuratDisk)
library(Matrix)

con <- gzfile(
  file.path(
    "data", "raw", "scrna_luad", "GSE131907", ".stage4b_cache",
    "GSE131907_Lung_Cancer_raw_UMI_matrix.rds.inner.gz"
  ),
  open = "rb"
)
raw_df <- readRDS(con)
close(con)

counts <- Matrix(as.matrix(raw_df), sparse = TRUE)
seurat_obj <- CreateSeuratObject(counts = counts, project = "GSE131907")
SaveH5Seurat(seurat_obj, filename = "GSE131907_raw.h5Seurat", overwrite = TRUE)
Convert(
  "GSE131907_raw.h5Seurat",
  dest = "h5ad",
  overwrite = TRUE
)
```

Move the resulting file to:

```text
data/processed/scrna_luad/scrna_luad_raw_or_converted.h5ad
```

Before using it, verify that the h5ad has exactly 208,506 observations and
29,634 variables and that its observation names match the official annotation.
This conversion does not authorize QC, UMAP, or biological analysis; those
remain a separate user-confirmed stage.

### Lower-memory route

Use R `hdf5r` or Bioconductor `HDF5Array` to write columns in chunks directly
to an on-disk HDF5-backed matrix. Avoid creating a second full dense matrix.
After conversion, use `zellkonverter::writeH5AD()` or Python `anndata` to create
the final h5ad. Keep gene names as variables and cell barcodes as observations.

## Other supported local inputs

The import script checks, in priority order:

1. `.h5ad`
2. complete GSE131907 raw RDS plus annotation
3. `.loom`
4. 10x `.h5`
5. a 10x directory containing `matrix.mtx`, `barcodes.tsv`, and `features.tsv`
6. a gene-by-cell CSV or TSV expression matrix

An alternative integrated NSCLC/LUAD h5ad is acceptable only when its public
source, LUAD subset, processing history, cell identifiers, patient/sample
metadata and original annotations are documented.

## Scientific integrity

- An incomplete file is not a formal dataset.
- `--small-test` creates isolated synthetic data for engineering tests only.
- No toy result may be reported as raw single-cell evidence.
- Stage 4B evaluates cellular context, not survival prediction.
- Stage 2D genes and Stage 4/5B signatures must remain fixed.
