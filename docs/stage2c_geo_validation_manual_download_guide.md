# Stage 2C GEO External Validation Manual Download Guide

Stage 2C prepares GEO import and frozen-model validation inputs. It does not
claim that external validation has been completed.

## Candidate cohorts

- [`GSE31210`](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE31210)
- [`GSE50081`](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE50081)
- [`GSE72094`](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE72094)
- [`GSE68465`](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE68465)

Review each GEO record and its linked publication before analysis. Confirm the
platform, LUAD inclusion criteria, sample identifiers, OS time unit, and event
encoding. Do not tune the TCGA model after inspecting external outcomes.

## Windows directory layout

For each cohort, create:

```text
data/raw/geo/GSE31210/
  expression_matrix.tsv
  probe_annotation.tsv
  clinical_survival.csv
```

The same structure applies to the other GSE accessions. A downloaded GEO
`*_series_matrix.txt` or `*_series_matrix.txt.gz` file may replace
`expression_matrix.tsv`.

Templates are generated under:

```text
data/metadata/geo_templates/
```

## Required formats

`expression_matrix.tsv` is probe-by-sample:

```text
probe_id    GSM000001    GSM000002
probe_001   8.1          7.4
```

`probe_annotation.tsv` maps probes to symbols:

```text
probe_id    gene_symbol
probe_001   TP53
```

`clinical_survival.csv` is sample-level OS:

```text
sample_id,OS_time,OS_status
GSM000001,730,1
```

Use OS days where possible. If source OS uses months or years, convert it
explicitly before formal scoring and record that conversion.

## Probe collapse and scaling

- Default multi-probe collapse: `mean`.
- Alternative: `--collapse-strategy max_variance`.
- Gene symbols are normalized to uppercase.
- Each external cohort is z-scored independently.
- External samples are never pooled with TCGA for scaling.
- Frozen TCGA genes and coefficients must not be reselected using GEO outcomes.

## Readiness audit

```powershell
cd SC-PROST-LUAD
conda activate gpu_py310
python scripts/stage2c_prepare_geo_validation.py --config configs/base.yaml
```

The audit writes:

```text
outputs/tables/stage2c_external_validation_readiness.csv
```

Missing OS fields, invalid status values, unmatched sample IDs, and missing
files are reported as blockers. They are not silently ignored.
