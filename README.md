# SC-PROST-LUAD

SC-PROST-LUAD is a staged, reproducible research codebase for externally
validated transcriptomic-clinical survival modeling and bounded biological
interpretation in lung adenocarcinoma (LUAD), using public datasets only.

The completed workflow includes TCGA-LUAD model development, locked validation
in four GEO cohorts, curated bulk cell-state interpretation, raw single-cell
cellular-context analysis, qualitative HPA evidence, exploratory CPTAC
proteomics, and an exploratory WSI feasibility analysis. The WSI pilot did not
show stable predictive improvement and is not presented as a successful main
model.

Aggregate publication results that do not contain patient-level records are
provided in `publication_results/tables/`. Raw data, large processed matrices,
whole-slide images, single-cell objects, model checkpoints, access credentials,
and patient-level predictions are intentionally excluded.

## Stage Roadmap

| Stage | Scope |
| --- | --- |
| 0 | Project structure, environment files, configs, README, AGENTS.md, directory standards, audit report, smoke test |
| 1 | TCGA-LUAD availability audit for clinical, survival, RNA-seq, mutation, CNV, methylation, WSI, plus patient-level matching table |
| 2 | Clinical plus RNA-seq survival baselines: Cox, LASSO-Cox, Random Survival Forest, DeepSurv |
| 3 | Mutation, CNV, methylation multi-omics survival modeling |
| 4 | Single-cell-guided cell-state signatures and bulk RNA-seq scoring |
| 5 | CPTAC proteomics or HPA expression/IHC validation |
| 6 | WSI download guidance, patch extraction, feature extraction, MIL/Transformer aggregation |
| 7 | Final multimodal deep survival model with cross-modal Transformer, modality dropout, Cox and discrete-time losses |
| 8 | Figures, tables, result reports, and manuscript skeleton |

## Repository Layout

```text
SC-PROST-LUAD/
  configs/                  YAML configuration files
  data/
    raw/                    Raw downloaded data, not tracked
    processed/              Processed analysis-ready data, not tracked
    metadata/               Data manifests and patient matching tables
  docs/                     Protocols, data guides, audit notes
  notebooks/                Exploratory notebooks only
  outputs/
    audit/                  Stage audit reports
    checkpoints/            Model checkpoints
    figures/                Generated figures
    logs/                   Runtime logs
    reports/                Human-readable stage reports
    tables/                 Generated tables
  scripts/                  Command-line entry scripts
  src/scprost_luad/         Python package
  tests/                    Smoke and unit tests
```

## Install

Preferred Windows/GPU conda environment:

```powershell
cd SC-PROST-LUAD
conda env create -f environment.yml
conda activate gpu_py310
python -m pip install -e .
```

If your machine has no NVIDIA GPU or the CUDA package is not available, create a CPU-first environment:

```powershell
conda create -n gpu_py310 python=3.10 -y
conda activate gpu_py310
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Stage 0 Smoke Test

Run the project skeleton smoke test:

```powershell
cd SC-PROST-LUAD
python scripts/run_stage0_smoke_test.py --root .
```

Expected outputs:

```text
data/metadata/stage0_directory_manifest.csv
outputs/logs/stage0_smoke_test.json
outputs/audit/stage0/audit_report.md
```

Run tests:

```powershell
python -m pytest tests -q
```

Every main script must support `--help`:

```powershell
python scripts/run_stage0_smoke_test.py --help
python -m scprost_luad.cli --help
```

## Data Policy

Large raw and processed biomedical files must stay under `data/raw/` and `data/processed/` and must not be committed. Download scripts in later stages must check whether files already exist and must resume or skip safely. Any dataset that requires manual access will be documented in `docs/manual_download_guide.md` with an expected local file format.

## Stage 1: TCGA-LUAD Data Availability Audit

Stage 1 queries public metadata only. It does not download expression matrices,
MAF files, methylation arrays, CNV matrices, or WSI images.

```powershell
cd SC-PROST-LUAD
python scripts/stage1_audit_tcga_luad.py --help
python scripts/stage1_audit_tcga_luad.py --dry-run
python scripts/stage1_audit_tcga_luad.py --small-test
python scripts/stage1_audit_tcga_luad.py
```

Use `--dry-run` to inspect the plan without network access. Use `--small-test`
for an offline smoke test with built-in miniature metadata. Run without either
flag to query live public GDC metadata and the public TCGA Pan-Cancer Clinical
Data Resource survival table.

## Stage 2: Clinical + RNA-seq Survival Baselines

Stage 2 uses the frozen Stage 1 clinical + usable OS + RNA-seq cohort. It does
not use WSI, mutation, CNV, methylation, single-cell signatures, or protein
data.

The repository does not currently include a real TCGA-LUAD RNA-seq matrix.
Follow `docs/stage2_rnaseq_manual_download_guide.md` before running the real
pipeline. The offline smoke workflow is:

```powershell
cd SC-PROST-LUAD
conda activate gpu_py310
python scripts/stage2_prepare_rnaseq_survival.py --config configs/base.yaml --small-test
python scripts/train_stage2_baselines.py --config configs/base.yaml --small-test
python scripts/evaluate_stage2_models.py --config configs/base.yaml --small-test
python -m pytest tests -q
```

Real preparation will fail with an actionable message until the expression
matrix is placed under `data/raw/tcga_luad/rnaseq/`:

```powershell
python scripts/stage2_prepare_rnaseq_survival.py --config configs/base.yaml
```

## Stage 2B: Real GDC STAR-counts Acquisition

Stage 2B builds the real TCGA-LUAD TPM matrix from public GDC STAR-counts files.
The matrix layout is patients x genes: the first column is `patient_id`, and
the remaining columns are version-stripped Ensembl gene IDs.

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

`stage2_download_gdc_rnaseq.py` also supports the official Windows GDC Data
Transfer Tool:

```powershell
python scripts/stage2_download_gdc_rnaseq.py `
  --manifest data/metadata/gdc_tcga_luad_rnaseq_star_counts_manifest.tsv `
  --gdc-client "C:\path\to\gdc-client.exe" `
  --method gdc-client
```

## Stage 2C: RNA-seq Robustness And GEO Readiness

Stage 2C remains limited to clinical + OS + RNA-seq. It adds leakage-safe
nested cross-validation, outer-fold OOF scores, RNA feature-space comparison,
and GEO import readiness checks.

```powershell
cd SC-PROST-LUAD
conda activate gpu_py310
python scripts/stage2c_compare_feature_spaces.py --config configs/base.yaml
python scripts/stage2c_nested_cv_rnaseq.py --config configs/base.yaml --seeds 42 3407 2026
python scripts/stage2c_generate_oof_risk_scores.py --config configs/base.yaml
python scripts/stage2c_prepare_geo_validation.py --config configs/base.yaml
python -m pytest tests -q
```

External GEO files are manual inputs. See
`docs/stage2c_geo_validation_manual_download_guide.md`.
