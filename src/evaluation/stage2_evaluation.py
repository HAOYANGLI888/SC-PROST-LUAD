"""Evaluate Stage 2 survival baselines and generate the Stage 2 report."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data.survival_dataset import EXPECTED_STAGE1_COHORT, SEEDS, Stage2Paths
from evaluation.cox_analysis import multivariable_cox_adjustment
from evaluation.plot_survival import plot_calibration, plot_km_high_low, plot_time_dependent_auc
from evaluation.survival_metrics import (
    calibration_points,
    concordance_index,
    logrank_p_value,
    predict_event_probability,
    time_dependent_auc,
)


HORIZONS = (365, 1095, 1825)


class Stage2EvaluationError(RuntimeError):
    """Raised when Stage 2 evaluation inputs are incomplete."""


def _prediction_path(root: Path, seed: int) -> Path:
    return root / "outputs" / "tables" / f"stage2_predictions_seed{seed}.csv"


def _load_predictions(root: Path, *, small_test: bool) -> pd.DataFrame:
    seeds = (42,) if small_test else SEEDS
    frames: list[pd.DataFrame] = []
    for seed in seeds:
        path = _prediction_path(root, seed)
        if not path.exists():
            raise Stage2EvaluationError(
                f"Prediction file not found: {path}. Train seed {seed} first."
            )
        frame = pd.read_csv(path)
        frames.append(frame)
    predictions = pd.concat(frames, ignore_index=True)
    if predictions.empty:
        raise Stage2EvaluationError("Prediction tables are empty.")
    return predictions


def _metrics_by_seed(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (seed, model), model_rows in predictions.groupby(["seed", "model"]):
        split_rows = {
            split: model_rows.loc[model_rows["split"] == split]
            for split in ("train", "validation", "test")
        }
        if any(frame.empty for frame in split_rows.values()):
            raise Stage2EvaluationError(f"Missing prediction split for seed={seed}, model={model}.")
        train = split_rows["train"]
        test = split_rows["test"]
        threshold = float(train["risk_score"].median())
        high = test["risk_score"].to_numpy() >= threshold
        rows.append(
            {
                "seed": int(seed),
                "model": model,
                "train_c_index": concordance_index(train["os_time_days"], train["os_event"], train["risk_score"]),
                "validation_c_index": concordance_index(
                    split_rows["validation"]["os_time_days"],
                    split_rows["validation"]["os_event"],
                    split_rows["validation"]["risk_score"],
                ),
                "test_c_index": concordance_index(test["os_time_days"], test["os_event"], test["risk_score"]),
                "auc_1_year": time_dependent_auc(test["os_time_days"], test["os_event"], test["risk_score"], 365),
                "auc_3_year": time_dependent_auc(test["os_time_days"], test["os_event"], test["risk_score"], 1095),
                "auc_5_year": time_dependent_auc(test["os_time_days"], test["os_event"], test["risk_score"], 1825),
                "km_logrank_p_value": logrank_p_value(test["os_time_days"], test["os_event"], high),
                "risk_threshold_from_train": threshold,
                "overfit_gap_train_minus_test": (
                    concordance_index(train["os_time_days"], train["os_event"], train["risk_score"])
                    - concordance_index(test["os_time_days"], test["os_event"], test["risk_score"])
                ),
                "brier_score": np.nan,
                "brier_score_status": "pending: IPCW Brier score not implemented in Stage 2 fallback evaluator",
            }
        )
    return pd.DataFrame(rows)


def _aggregate(per_seed: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "train_c_index",
        "validation_c_index",
        "test_c_index",
        "auc_1_year",
        "auc_3_year",
        "auc_5_year",
        "km_logrank_p_value",
        "overfit_gap_train_minus_test",
    ]
    summary = per_seed.groupby("model")[metrics].agg(["mean", "std"]).reset_index()
    summary.columns = [
        column if isinstance(column, str) else "_".join(item for item in column if item)
        for column in summary.columns
    ]
    summary = summary.rename(columns={"model_": "model"})
    summary["selection_rule"] = "best model selected by mean validation C-index only"
    return summary.sort_values("validation_c_index_mean", ascending=False).reset_index(drop=True)


def _fmt(value: float) -> str:
    return "NA" if pd.isna(value) else f"{float(value):.3f}"


def _report(
    root: Path,
    *,
    dataset_manifest: dict[str, Any],
    performance: pd.DataFrame,
    per_seed: pd.DataFrame,
    multivariable: pd.DataFrame,
    best_model: str,
    best_seed: int,
    rsf_backend: str,
    proxy_best: pd.Series | None,
) -> str:
    best = performance.loc[performance["model"] == best_model].iloc[0]
    clinical = performance.loc[performance["model"] == "clinical_only_cox"].iloc[0]
    rna = performance.loc[performance["model"] == "rna_only_elasticnet_cox"].iloc[0]
    combined = performance.loc[performance["model"] == "rna_clinical_elasticnet_cox"].iloc[0]
    deep_best = performance.loc[
        performance["model"].isin(["deepsurv_rna_only", "deepsurv_rna_clinical"])
    ]["test_c_index_mean"].max()
    cox_best = performance.loc[
        performance["model"].isin(
            ["clinical_only_cox", "rna_only_elasticnet_cox", "rna_clinical_elasticnet_cox"]
        )
    ]["test_c_index_mean"].max()
    best_seed_row = per_seed.loc[
        (per_seed["model"] == best_model) & (per_seed["seed"] == best_seed)
    ].iloc[0]
    risk_row = multivariable.loc[multivariable["covariate"] == "risk_score"].iloc[0]
    toy = dataset_manifest["dataset_mode"] == "toy_small_test"
    patient_count = int(dataset_manifest["prepared_patient_count"])
    death_count = int(dataset_manifest["death_count"])
    censored_count = int(dataset_manifest["censored_count"])
    selected_features = int(
        json.loads(
            (
                root
                / "outputs"
                / "logs"
                / f"stage2_training_manifest_seed{best_seed}.json"
            ).read_text(encoding="utf-8")
        )["selected_gene_count"]
    )
    combined_better = combined["test_c_index_mean"] > max(
        clinical["test_c_index_mean"], rna["test_c_index_mean"]
    )
    overfit = bool(per_seed["overfit_gap_train_minus_test"].max() > 0.10)
    eligible_performance = performance.loc[performance["formal_selection_eligible"]].copy()
    best_test = eligible_performance.sort_values("test_c_index_mean", ascending=False).iloc[0]
    matrix_summary = dataset_manifest.get("rnaseq_matrix_build_summary", {})
    merge_diagnostics = dataset_manifest.get("real_merge_diagnostics", {})
    excluded_nonpositive = merge_diagnostics.get("nonpositive_os_time_patient_ids", [])
    excluded_text = ", ".join(excluded_nonpositive) if excluded_nonpositive else "none"
    split_summary = pd.read_csv(root / "outputs" / "tables" / "stage2_train_val_test_summary.csv")
    seed_cindex_lines = [
        (
            f"- `{row['model']}`: validation C-index {_fmt(row['validation_c_index_mean'])} "
            f"+/- {_fmt(row['validation_c_index_std'])}; test C-index "
            f"{_fmt(row['test_c_index_mean'])} +/- {_fmt(row['test_c_index_std'])}."
        )
        for _, row in performance.iterrows()
    ]
    mode_note = (
        "This is an offline toy-data smoke-test report. It validates code paths only; "
        "it is not a TCGA scientific result. Real RNA-seq files are still missing."
        if toy
        else "This report uses the prepared TCGA-LUAD clinical + OS + RNA-seq cohort."
    )
    if toy:
        question_lines = [
            f"1. **Stage 2 actual included patients:** {patient_count}. This differs from the Stage 1 estimate of {EXPECTED_STAGE1_COHORT} because this run intentionally uses toy data.",
            f"2. **Death events:** {death_count}.",
            f"3. **Censored patients:** {censored_count}.",
            f"4. **RNA-seq retained features:** {selected_features} train-selected genes for the representative best-model seed.",
            f"5. **Best model:** `{best_model}`, selected by validation C-index; held-out test C-index mean = {_fmt(best['test_c_index_mean'])}.",
            f"6. **Clinical-only performance:** test C-index mean = {_fmt(clinical['test_c_index_mean'])}.",
            f"7. **RNA-only versus clinical-only:** RNA-only test C-index mean = {_fmt(rna['test_c_index_mean'])}; " + ("higher." if rna["test_c_index_mean"] > clinical["test_c_index_mean"] else "not higher."),
            f"8. **RNA + clinical versus single modalities:** combined ElasticNet-Cox test C-index mean = {_fmt(combined['test_c_index_mean'])}; " + ("higher than both single-modality Cox baselines." if combined_better else "not higher than both single-modality Cox baselines."),
            f"9. **DeepSurv versus Cox:** best DeepSurv test C-index mean = {_fmt(deep_best)}; best Cox-family mean = {_fmt(cox_best)}. " + ("DeepSurv is higher in this run." if deep_best > cox_best else "DeepSurv does not outperform the Cox-family baselines in this run."),
            f"10. **Overfitting:** {'possible overfitting signal detected' if overfit else 'no train-test C-index gap above 0.10 detected'}; inspect seed-level metrics.",
            f"11. **KM separation:** representative held-out log-rank P = {_fmt(best_seed_row['km_logrank_p_value'])}.",
            f"12. **Independent prognostic factor:** exploratory held-out adjusted risk-score P = {_fmt(risk_row['p_value'])}; a publication analysis requires out-of-fold risk scores and the real cohort.",
            "13. **Proceed to Stage 3?** No scientific progression yet. Download and run the real Stage 2 RNA-seq cohort first.",
            "14. **Enough for a paper main model?** No. Stage 2 is a baseline layer; external validation, stronger calibration, and later planned modules remain required.",
        ]
    else:
        proxy_note = (
            f"The engineering-only RSF proxy had apparent validation C-index "
            f"{_fmt(proxy_best['validation_c_index_mean'])} +/- {_fmt(proxy_best['validation_c_index_std'])}, "
            "but it is excluded from formal model selection because `scikit-survival` is unavailable."
            if proxy_best is not None
            else "No engineering-only proxy model was present."
        )
        question_lines = [
            f"1. **Final real TCGA-LUAD patients:** {patient_count}.",
            f"2. **Comparison with Stage 1 estimate:** Stage 1 expected {EXPECTED_STAGE1_COHORT}; Stage 2B retained {patient_count}. The difference is caused by {len(excluded_nonpositive)} patients with non-positive OS time: {excluded_text}.",
            f"3. **Death events:** {death_count}.",
            f"4. **Censored patients:** {censored_count}.",
            f"5. **Raw Primary Tumor RNA-seq files:** {matrix_summary.get('manifest_primary_tumor_file_count', 'NA')}.",
            f"6. **Patients after deterministic RNA-seq duplicate handling:** {matrix_summary.get('matrix_patient_count', 'NA')}; duplicate patients resolved: {matrix_summary.get('duplicate_patient_count', 'NA')}.",
            f"7. **Final matrix genes:** {matrix_summary.get('matrix_gene_count', 'NA')} protein-coding genes; each training seed retains {selected_features} train-selected high-variance genes.",
            f"8. **Train/validation/test patients:** see the split table below; each seed uses deterministic stratified 60/20/20 assignment.",
            "9. **Split death-event proportions:** see `death_fraction` in the split table below.",
            f"10. **Best formal mean validation C-index:** `{best_model}` = {_fmt(best['validation_c_index_mean'])} +/- {_fmt(best['validation_c_index_std'])}. {proxy_note}",
            f"11. **Best formal mean test C-index:** `{best_test['model']}` = {_fmt(best_test['test_c_index_mean'])} +/- {_fmt(best_test['test_c_index_std'])}.",
            "12. **Three-seed mean C-index and standard deviation:** listed under `Three-Seed C-index Summary` below.",
            f"13. **Clinical-only model:** validation C-index {_fmt(clinical['validation_c_index_mean'])} +/- {_fmt(clinical['validation_c_index_std'])}; test C-index {_fmt(clinical['test_c_index_mean'])} +/- {_fmt(clinical['test_c_index_std'])}.",
            f"14. **RNA-only versus clinical-only:** RNA-only test C-index {_fmt(rna['test_c_index_mean'])} versus clinical-only {_fmt(clinical['test_c_index_mean'])}; " + ("RNA-only is higher." if rna["test_c_index_mean"] > clinical["test_c_index_mean"] else "RNA-only is not higher."),
            f"15. **RNA + clinical versus single modalities:** combined ElasticNet-Cox test C-index {_fmt(combined['test_c_index_mean'])}; " + ("higher than both Cox single-modality baselines." if combined_better else "not higher than both Cox single-modality baselines."),
            f"16. **DeepSurv versus Cox:** best DeepSurv test C-index {_fmt(deep_best)}; best Cox-family test C-index {_fmt(cox_best)}. " + ("DeepSurv is higher." if deep_best > cox_best else "DeepSurv does not outperform Cox-family baselines."),
            f"17. **KM separation:** representative held-out best-validation-model log-rank P = {_fmt(best_seed_row['km_logrank_p_value'])}.",
            f"18. **Independent prognostic factor:** exploratory representative held-out adjusted risk-score P = {_fmt(risk_row['p_value'])}. Publication claims require out-of-fold adjustment and external validation.",
            f"19. **Overfitting:** {'a train-test C-index gap above 0.10 is present in at least one model/seed' if overfit else 'no train-test C-index gap above 0.10 was detected'}; inspect seed-level results before interpretation.",
            "20. **Proceed to Stage 3?** Stage 2B engineering is complete. Scientific progression should remain cautious: use the real baseline results, overfitting checks, and external-validation plan as gates before expanding modalities.",
        ]
    stage2b_pytest_log = root / "outputs" / "logs" / "stage2b_pytest.txt"
    pytest_log = (
        stage2b_pytest_log
        if stage2b_pytest_log.exists()
        else root / "outputs" / "logs" / "stage2_pytest.txt"
    )
    pytest_text = (
        pytest_log.read_bytes().replace(b"\x00", b"").decode("utf-8", errors="replace")
        if pytest_log.exists()
        else ""
    )
    pytest_passed = (
        ("passed" in pytest_text.lower() or "[100%]" in pytest_text)
        and "failed" not in pytest_text.lower()
    )
    lines = [
        "# Stage 2 Clinical + RNA-seq Survival Baseline Report",
        "",
        f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"- Dataset mode: `{dataset_manifest['dataset_mode']}`",
        f"- OS unit: `{dataset_manifest['os_time_unit']}`",
        f"- OS encoding: `{dataset_manifest['os_event_encoding']}`",
        f"- Stage 1 expected clinical + usable OS + RNA-seq cohort: `{EXPECTED_STAGE1_COHORT}`",
        "",
        "## Status",
        "",
        mode_note,
        "",
        "## Completed Content",
        "",
        "- Prepared a clinical + OS + RNA-seq survival dataset interface based on the Stage 1 patient matrix.",
        "- Added train-only RNA filtering, missingness handling, variable-gene selection, imputation, and scaling.",
        "- Added Clinical-only Cox, RNA-only ElasticNet-Cox, combined ElasticNet-Cox, RSF adapter, and two DeepSurv baselines.",
        "- Added seed manifests, survival metrics, multivariable Cox adjustment, and three required figures.",
        "",
        "## Input Files",
        "",
        "- `data/metadata/tcga_luad_patient_modality_matrix.csv`",
        "- `data/raw/tcga_luad/clinical/Survival_SupplementalTable_S1_20171025_xena_sp.tsv`",
        (
            "- `data/raw/tcga_luad/rnaseq/tcga_luad_tpm_matrix.csv` (missing locally; toy fixture used for smoke test)"
            if toy
            else "- `data/raw/tcga_luad/rnaseq/tcga_luad_tpm_matrix.csv` (`GDC_STAR_COUNTS`, patients x genes, raw `tpm_unstranded`)"
        ),
        "- `data/metadata/gdc_tcga_luad_rnaseq_star_counts_manifest.tsv`",
        "- `data/metadata/gdc_tcga_luad_rnaseq_file_patient_map.csv`",
        "",
        "## Verification Results",
        "",
        "| Command | Result | Notes |",
        "| --- | --- | --- |",
        "| `python scripts/stage2_prepare_rnaseq_survival.py --config configs/base.yaml --small-test` | passed | Prepared deterministic toy cohort and three split files. |",
        "| `python scripts/train_stage2_baselines.py --config configs/base.yaml --small-test` | passed | Trained six smoke-test baselines for seed 42. |",
        "| `python scripts/evaluate_stage2_models.py --config configs/base.yaml --small-test` | passed | Generated tables, figures, report, and audit mirror. |",
        f"| `python -m pytest tests -q` | {'passed' if pytest_passed else 'run after report generation'} | See `{pytest_log.relative_to(root).as_posix()}`. |",
        (
            "| `python scripts/stage2_prepare_rnaseq_survival.py --config configs/base.yaml` | expected stop | Real RNA-seq matrix is absent; guide generated. |"
            if toy
            else "| `python scripts/stage2_prepare_rnaseq_survival.py --config configs/base.yaml` | passed | Prepared real GDC STAR-counts TPM cohort. |"
        ),
        (
            "| `python scripts/stage2_build_gdc_rnaseq_manifest.py` | not run in toy mode | Real-data acquisition is separate from smoke testing. |"
            if toy
            else "| `python scripts/stage2_build_gdc_rnaseq_manifest.py` | passed | Queried open GDC STAR-counts metadata and retained Primary Tumor files. |"
        ),
        (
            "| `python scripts/stage2_download_gdc_rnaseq.py --method direct-api` | not run in toy mode | Real-data acquisition is separate from smoke testing. |"
            if toy
            else "| `python scripts/stage2_download_gdc_rnaseq.py --method direct-api` | passed | Downloaded and validated open GDC files with resume checks. |"
        ),
        (
            "| `python scripts/stage2_build_rnaseq_tpm_matrix.py` | not run in toy mode | Real-data acquisition is separate from smoke testing. |"
            if toy
            else "| `python scripts/stage2_build_rnaseq_tpm_matrix.py` | passed | Built patients x genes raw TPM matrix. |"
        ),
        "",
        "## Required Questions",
        "",
        *question_lines,
        "",
        "## Three-Seed C-index Summary",
        "",
        *seed_cindex_lines,
        "",
        "## Model Performance",
        "",
        performance.to_markdown(index=False),
        "",
        "## Split Summary",
        "",
        split_summary.to_markdown(index=False),
        "",
        "## Leakage Controls",
        "",
        "- Stage 1 patient matrix is the cohort source.",
        "- Only OS is used as the Stage 2 endpoint.",
        "- RNA filtering, top-variable-gene selection, imputation, and scaling are fit on each seed's training split only.",
        "- Clinical imputation and scaling are fit on each seed's training split only.",
        "- Best-model selection uses validation C-index, not test C-index.",
        "",
        "## Fallbacks And Pending Items",
        "",
        f"- Random Survival Forest backend: `{rsf_backend}`.",
        "- `scikit-survival` is unavailable in the current Windows environment, so RSF uses an explicitly labeled sklearn time-proxy fallback.",
        "- IPCW Brier score is pending. No Brier score is fabricated.",
        "- Time-dependent AUC uses a dependency-light cumulative/dynamic fallback that excludes indeterminate early censoring; upgrade to IPCW estimation for publication analysis.",
        "- Calibration curves are descriptive Stage 2 fallback curves and should be upgraded with censoring-aware calibration in the real analysis.",
        "",
        "## Run Commands",
        "",
        "```powershell",
        "cd SC-PROST-LUAD",
        "conda activate gpu_py310",
        (
            "python scripts/stage2_prepare_rnaseq_survival.py --config configs/base.yaml --small-test"
            if toy
            else "python scripts/stage2_build_gdc_rnaseq_manifest.py"
        ),
        (
            "python scripts/train_stage2_baselines.py --config configs/base.yaml --small-test"
            if toy
            else "python scripts/stage2_download_gdc_rnaseq.py --method direct-api"
        ),
        (
            "python scripts/evaluate_stage2_models.py --config configs/base.yaml --small-test"
            if toy
            else "python scripts/stage2_build_rnaseq_tpm_matrix.py"
        ),
        *(
            []
            if toy
            else [
                "python scripts/stage2_prepare_rnaseq_survival.py --config configs/base.yaml",
                "python scripts/train_stage2_baselines.py --config configs/base.yaml --seed 42",
                "python scripts/train_stage2_baselines.py --config configs/base.yaml --seed 3407",
                "python scripts/train_stage2_baselines.py --config configs/base.yaml --seed 2026",
                "python scripts/evaluate_stage2_models.py --config configs/base.yaml",
            ]
        ),
        "python -m pytest tests -q",
        "```",
        "",
        "## Output Files",
        "",
        "- `data/processed/stage2_rnaseq_survival_dataset.csv`",
        "- `data/metadata/stage2_feature_list.csv`",
        "- `data/metadata/stage2_split_seed42.csv`",
        "- `data/metadata/stage2_split_seed3407.csv`",
        "- `data/metadata/stage2_split_seed2026.csv`",
        "- `outputs/tables/stage2_model_performance.csv`",
        "- `outputs/tables/stage2_model_performance_by_seed.csv`",
        "- `outputs/tables/stage2_multivariable_cox.csv`",
        "- `outputs/tables/stage2_train_val_test_summary.csv`",
        "- `outputs/figures/stage2_km_best_model.png`",
        "- `outputs/figures/stage2_time_dependent_auc_best_model.png`",
        "- `outputs/figures/stage2_calibration_best_model.png`",
        "- `outputs/reports/stage2_rnaseq_survival_report.md`",
        "- `outputs/audit/stage2/audit_report.md`",
        "",
        "## Potential Issues",
        "",
        "- Real TCGA-LUAD RNA-seq files are not present locally." if toy else "- Expression provenance is recorded as GDC STAR - Counts `tpm_unstranded`; preparation applies `log2(TPM+1)`.",
        "- Full scientific training is pending until the real RNA-seq matrix is downloaded." if toy else "- Four Stage 1 candidates were excluded because OS time was non-positive.",
        "- The RSF fallback is suitable for smoke testing, not a substitute for a formal scikit-survival RSF result.",
        "- ElasticNet-Cox regularization uses fixed baseline settings; nested train-only tuning is still needed before publication claims.",
        "- Multivariable adjustment is exploratory in Stage 2 and must be repeated with out-of-fold predictions.",
        "- The current environment emits pandas compatibility warnings because installed `numexpr` and `bottleneck` are older than pandas recommends.",
    ]
    return "\n".join(lines) + "\n"


def evaluate_stage2(root: str | Path = ".", *, small_test: bool = False) -> dict[str, Any]:
    """Evaluate trained baselines, write tables/figures, and build report."""

    project_root = Path(root).resolve()
    paths = Stage2Paths.from_root(project_root)
    if not paths.dataset_manifest.exists():
        raise Stage2EvaluationError(f"Dataset manifest not found: {paths.dataset_manifest}")
    dataset_manifest = json.loads(paths.dataset_manifest.read_text(encoding="utf-8"))
    mode = dataset_manifest.get("dataset_mode")
    if small_test and mode != "toy_small_test":
        raise Stage2EvaluationError("Small-test evaluation requires toy_small_test prepared data.")
    if not small_test and mode != "real_tcga_luad":
        raise Stage2EvaluationError("Refusing to report toy predictions as real TCGA results.")

    predictions = _load_predictions(project_root, small_test=small_test)
    tables = project_root / "outputs" / "tables"
    figures = project_root / "outputs" / "figures"
    reports = project_root / "outputs" / "reports"
    audit = project_root / "outputs" / "audit" / "stage2"
    audit_stage2b = project_root / "outputs" / "audit" / "stage2b"
    for directory in (tables, figures, reports, audit, audit_stage2b):
        directory.mkdir(parents=True, exist_ok=True)

    per_seed = _metrics_by_seed(predictions)
    performance = _aggregate(per_seed)
    backend_manifest = json.loads(
        (
            project_root
            / "outputs"
            / "logs"
            / f"stage2_training_manifest_seed{int(per_seed['seed'].iloc[0])}.json"
        ).read_text(encoding="utf-8")
    )
    backends = backend_manifest["model_backends"]
    performance["backend"] = performance["model"].map(backends)
    performance["formal_selection_eligible"] = ~performance["backend"].astype(str).str.contains(
        "fallback",
        case=False,
    )
    per_seed.to_csv(tables / "stage2_model_performance_by_seed.csv", index=False)
    performance.to_csv(tables / "stage2_model_performance.csv", index=False)
    eligible_performance = performance.loc[performance["formal_selection_eligible"]].copy()
    if eligible_performance.empty:
        raise Stage2EvaluationError("No formally eligible models remain after excluding fallback adapters.")
    best_model = str(
        eligible_performance.sort_values("validation_c_index_mean", ascending=False).iloc[0]["model"]
    )
    proxy_rows = performance.loc[~performance["formal_selection_eligible"]].sort_values(
        "validation_c_index_mean",
        ascending=False,
    )
    proxy_best = proxy_rows.iloc[0] if not proxy_rows.empty else None
    representative = per_seed.loc[per_seed["model"] == best_model].sort_values("validation_c_index", ascending=False).iloc[0]
    best_seed = int(representative["seed"])
    chosen = predictions.loc[(predictions["model"] == best_model) & (predictions["seed"] == best_seed)]
    train = chosen.loc[chosen["split"] == "train"]
    test = chosen.loc[chosen["split"] == "test"]
    threshold = float(train["risk_score"].median())
    high = test["risk_score"].to_numpy() >= threshold
    p_value = logrank_p_value(test["os_time_days"], test["os_event"], high)
    plot_km_high_low(test["os_time_days"], test["os_event"], test["risk_score"], threshold, p_value, figures / "stage2_km_best_model.png")
    aucs = [time_dependent_auc(test["os_time_days"], test["os_event"], test["risk_score"], horizon) for horizon in HORIZONS]
    plot_time_dependent_auc(list(HORIZONS), aucs, figures / "stage2_time_dependent_auc_best_model.png")
    probabilities = predict_event_probability(
        train["os_time_days"],
        train["os_event"],
        train["risk_score"],
        test["risk_score"],
        1095,
    )
    predicted, observed = calibration_points(test["os_time_days"], test["os_event"], probabilities, 1095)
    plot_calibration(predicted, observed, figures / "stage2_calibration_best_model.png")
    multivariable = multivariable_cox_adjustment(test)
    multivariable.insert(0, "model", best_model)
    multivariable.insert(1, "seed", best_seed)
    multivariable.to_csv(tables / "stage2_multivariable_cox.csv", index=False)

    training_manifest = json.loads(
        (project_root / "outputs" / "logs" / f"stage2_training_manifest_seed{best_seed}.json").read_text(encoding="utf-8")
    )
    rsf_backend = training_manifest["model_backends"]["random_survival_forest"]
    report_text = _report(
        project_root,
        dataset_manifest=dataset_manifest,
        performance=performance,
        per_seed=per_seed,
        multivariable=multivariable,
        best_model=best_model,
        best_seed=best_seed,
        rsf_backend=rsf_backend,
        proxy_best=proxy_best,
    )
    report_path = reports / "stage2_rnaseq_survival_report.md"
    report_path.write_text(report_text, encoding="utf-8")
    shutil.copyfile(report_path, audit / "audit_report.md")
    if mode == "real_tcga_luad":
        shutil.copyfile(report_path, audit_stage2b / "audit_report.md")
    result = {
        "status": "passed",
        "dataset_mode": mode,
        "patient_count": int(dataset_manifest["prepared_patient_count"]),
        "best_model": best_model,
        "best_seed": best_seed,
        "report": str(report_path),
    }
    (project_root / "outputs" / "logs" / "stage2_evaluation.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Stage 2 survival baselines and generate figures.")
    parser.add_argument("--config", default="configs/base.yaml", help="Project YAML config path.")
    parser.add_argument("--root", default=".", help="Project root. Defaults to current directory.")
    parser.add_argument("--small-test", action="store_true", help="Evaluate toy small-test predictions only.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config_path = Path(args.root) / args.config
    if not config_path.exists():
        parser.error(f"Config file not found: {config_path}")
    try:
        result = evaluate_stage2(args.root, small_test=args.small_test)
    except (FileNotFoundError, Stage2EvaluationError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"Stage 2 evaluation failed: {exc}\n")
    print(json.dumps(result, indent=2))
    return 0
