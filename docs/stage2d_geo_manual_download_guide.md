# Stage 2D GEO Manual Download Guide

Stage 2D uses public GEO series matrices and platform annotations only. The
preferred Windows entry point downloads files automatically and skips completed
files:

```powershell
conda activate gpu_py310
python scripts/stage2d_download_geo_metadata.py --config configs/base.yaml
```

If NCBI download access is interrupted, download the following files in a web
browser or with another resumable downloader:

| Resource | Destination |
| --- | --- |
| [GSE31210 series matrix](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE31nnn/GSE31210/matrix/GSE31210_series_matrix.txt.gz) | `data/raw/geo/GSE31210/GSE31210_series_matrix.txt.gz` |
| [GSE50081 series matrix](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE50nnn/GSE50081/matrix/GSE50081_series_matrix.txt.gz) | `data/raw/geo/GSE50081/GSE50081_series_matrix.txt.gz` |
| [GSE72094 series matrix](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE72nnn/GSE72094/matrix/GSE72094_series_matrix.txt.gz) | `data/raw/geo/GSE72094/GSE72094_series_matrix.txt.gz` |
| [GSE68465 series matrix](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE68nnn/GSE68465/matrix/GSE68465_series_matrix.txt.gz) | `data/raw/geo/GSE68465/GSE68465_series_matrix.txt.gz` |
| [GPL570 annotation](https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/GPL570/annot/GPL570.annot.gz) | `data/raw/geo/platforms/GPL570.annot.gz` |
| [GPL96 annotation](https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/GPL96/annot/GPL96.annot.gz) | `data/raw/geo/platforms/GPL96.annot.gz` |
| [GPL15048 full platform text](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GPL15048&targ=self&form=text&view=full) | `data/raw/geo/platforms/GPL15048.txt` |

Then prepare each cohort:

```powershell
python scripts/stage2d_prepare_geo_cohort.py --config configs/base.yaml --cohort GSE31210
python scripts/stage2d_prepare_geo_cohort.py --config configs/base.yaml --cohort GSE50081
python scripts/stage2d_prepare_geo_cohort.py --config configs/base.yaml --cohort GSE72094
python scripts/stage2d_prepare_geo_cohort.py --config configs/base.yaml --cohort GSE68465
```

The formal workflow stops if OS time or status cannot be parsed, if annotations
do not overlap expression probes, or if more than 30% of the frozen TCGA genes
are unavailable. It does not silently claim a successful download or validation.

