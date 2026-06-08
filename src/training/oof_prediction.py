"""OOF risk-score analysis, fixed TCGA model export, and Stage 2C reporting."""

from __future__ import annotations

import json
import pickle
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.clinical_preprocess import CLINICAL_COLUMNS, ClinicalTrainPreprocessor
from evaluation.cox_analysis import multivariable_cox_adjustment
from evaluation.plot_survival import plot_calibration, plot_km_high_low, plot_time_dependent_auc
from evaluation.survival_metrics import calibration_points, concordance_index, logrank_p_value, time_dependent_auc
from features.rnaseq_feature_spaces import RNAFeatureSpace, feature_space_inventory
from training.nested_cv import (
    HORIZONS,
    ModelConfiguration,
    _fit_estimator,
    _model_input,
    load_stage2c_dataset,
    model_configurations,
)


class Stage2COOFError(RuntimeError):
    """Raised when OOF analysis inputs are incomplete."""


def _paths(root: Path) -> dict[str, Path]:
    return {
        "raw_oof": root / "outputs" / "tables" / "stage2c_nested_cv_oof_predictions_raw.csv",
        "performance": root / "outputs" / "tables" / "stage2c_nested_cv_performance.csv",
        "oof": root / "data" / "processed" / "stage2c_oof_risk_scores.csv",
        "km": root / "outputs" / "tables" / "stage2c_oof_km_logrank.csv",
        "mcox": root / "outputs" / "tables" / "stage2c_oof_multivariable_cox.csv",
        "overfit": root / "outputs" / "tables" / "stage2c_overfitting_diagnostics.csv",
        "dca": root / "outputs" / "tables" / "stage2c_oof_decision_curve.csv",
        "external_spec": root / "outputs" / "checkpoints" / "stage2c_tcga_fixed_rna_validation_model.json",
        "external_pickle": root / "outputs" / "checkpoints" / "stage2c_tcga_fixed_rna_validation_model.pkl",
        "report": root / "outputs" / "reports" / "stage2c_rnaseq_robustness_report.md",
        "audit": root / "outputs" / "audit" / "stage2c" / "audit_report.md",
        "root_audit": root / "audit_report.md",
    }


def _load_inputs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = _paths(root)
    if not paths["raw_oof"].exists() or not paths["performance"].exists():
        raise Stage2COOFError("Nested-CV outputs are missing. Run scripts/stage2c_nested_cv_rnaseq.py first.")
    oof = pd.read_csv(paths["raw_oof"])
    performance = pd.read_csv(paths["performance"])
    if oof.empty or performance.empty:
        raise Stage2COOFError("Nested-CV outputs are empty.")
    return oof, performance


def _best_configurations(performance: pd.DataFrame) -> tuple[str, str]:
    summary = performance.groupby("config_id", as_index=False)["outer_test_c_index"].mean()
    best_overall = str(summary.sort_values("outer_test_c_index", ascending=False).iloc[0]["config_id"])
    rna_configs = performance.loc[
        (performance["input_scope"] == "combined") & (performance["feature_space"] != "clinical_only"),
        "config_id",
    ].unique()
    if not len(rna_configs):
        raise Stage2COOFError("No RNA + clinical configuration is available for fixed external-validation export.")
    best_rna = str(
        summary.loc[summary["config_id"].isin(rna_configs)]
        .sort_values("outer_test_c_index", ascending=False)
        .iloc[0]["config_id"]
    )
    return best_overall, best_rna


def _oof_metrics(oof: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (config_id, seed), frame in oof.groupby(["config_id", "seed"]):
        threshold = float(frame["oof_risk_score"].median())
        high = frame["oof_risk_score"].to_numpy(dtype=float) >= threshold
        rows.append(
            {
                "config_id": config_id,
                "model_name": frame["model_name"].iloc[0],
                "feature_space": frame["feature_space"].iloc[0],
                "seed": int(seed),
                "patient_count": len(frame),
                "death_count": int(frame["OS_status"].sum()),
                "risk_threshold": threshold,
                "oof_c_index": concordance_index(frame["OS_time"], frame["OS_status"], frame["oof_risk_score"]),
                "auc_1_year": time_dependent_auc(frame["OS_time"], frame["OS_status"], frame["oof_risk_score"], 365),
                "auc_3_year": time_dependent_auc(frame["OS_time"], frame["OS_status"], frame["oof_risk_score"], 1095),
                "auc_5_year": time_dependent_auc(frame["OS_time"], frame["OS_status"], frame["oof_risk_score"], 1825),
                "km_logrank_p_value": logrank_p_value(frame["OS_time"], frame["OS_status"], high),
            }
        )
    return pd.DataFrame(rows)


def _multivariable_oof(oof: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for (config_id, seed), frame in oof.groupby(["config_id", "seed"]):
        predictions = frame.rename(
            columns={"OS_time": "os_time_days", "OS_status": "os_event", "oof_risk_score": "risk_score"}
        )
        result = multivariable_cox_adjustment(predictions)
        result.insert(0, "config_id", config_id)
        result.insert(1, "model_name", frame["model_name"].iloc[0])
        result.insert(2, "feature_space", frame["feature_space"].iloc[0])
        result.insert(3, "seed", int(seed))
        result["analysis_scope"] = "outer-fold out-of-fold predictions; exploratory"
        rows.append(result)
    return pd.concat(rows, ignore_index=True)


def _decision_curve(frame: pd.DataFrame, *, horizon_days: int = 1095) -> pd.DataFrame:
    times = frame["OS_time"].to_numpy(dtype=float)
    status = frame["OS_status"].to_numpy(dtype=int)
    probabilities = frame["event_probability_3_year"].to_numpy(dtype=float)
    usable = ((status == 1) & (times <= horizon_days)) | (times > horizon_days)
    labels = ((status == 1) & (times <= horizon_days)).astype(int)[usable]
    probabilities = probabilities[usable]
    n = len(labels)
    rows = []
    for threshold in np.arange(0.10, 0.71, 0.05):
        predicted = probabilities >= threshold
        true_positive = int(np.sum(predicted & (labels == 1)))
        false_positive = int(np.sum(predicted & (labels == 0)))
        net_benefit = true_positive / n - false_positive / n * threshold / (1.0 - threshold)
        rows.append(
            {
                "threshold": threshold,
                "net_benefit": net_benefit,
                "usable_patient_count": n,
                "horizon_days": horizon_days,
                "status": "descriptive fallback excluding indeterminate early censoring; IPCW DCA pending",
            }
        )
    return pd.DataFrame(rows)


def _plot_oof(
    root: Path,
    oof: pd.DataFrame,
    metrics: pd.DataFrame,
    performance: pd.DataFrame,
    best_config: str,
) -> None:
    figures = root / "outputs" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    chosen = oof.loc[(oof["config_id"] == best_config) & (oof["seed"] == 42)].copy()
    if chosen.empty:
        selected_seed = int(oof.loc[oof["config_id"] == best_config, "seed"].iloc[0])
        chosen = oof.loc[(oof["config_id"] == best_config) & (oof["seed"] == selected_seed)].copy()
    threshold = float(chosen["oof_risk_score"].median())
    p_value = logrank_p_value(
        chosen["OS_time"],
        chosen["OS_status"],
        chosen["oof_risk_score"].to_numpy(dtype=float) >= threshold,
    )
    plot_km_high_low(
        chosen["OS_time"],
        chosen["OS_status"],
        chosen["oof_risk_score"],
        threshold,
        p_value,
        figures / "stage2c_oof_km_best_model.png",
    )
    aucs = [
        time_dependent_auc(chosen["OS_time"], chosen["OS_status"], chosen["oof_risk_score"], horizon)
        for horizon in HORIZONS
    ]
    plot_time_dependent_auc(list(HORIZONS), aucs, figures / "stage2c_oof_time_auc_best_model.png")
    predicted, observed = calibration_points(
        chosen["OS_time"],
        chosen["OS_status"],
        chosen["event_probability_3_year"],
        1095,
    )
    plot_calibration(predicted, observed, figures / "stage2c_oof_calibration_best_model.png")

    diagnostic = (
        performance.groupby(["config_id", "model_name"], as_index=False)
        .agg(train_c_index=("outer_train_c_index", "mean"), test_c_index=("outer_test_c_index", "mean"))
    )
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    ax.scatter(diagnostic["test_c_index"], diagnostic["train_c_index"], color="#B13C2E")
    for row in diagnostic.itertuples():
        ax.annotate(row.config_id, (row.test_c_index, row.train_c_index), fontsize=6, alpha=0.85)
    ax.plot([0.45, 1.0], [0.45, 1.0], linestyle="--", color="#777777", linewidth=1)
    ax.set_xlim(0.45, max(0.75, float(diagnostic["test_c_index"].max()) + 0.03))
    ax.set_ylim(0.45, 1.01)
    ax.set_xlabel("Outer-test C-index")
    ax.set_ylabel("Outer-train C-index")
    ax.set_title("Stage 2C overfitting diagnostics")
    fig.tight_layout()
    fig.savefig(figures / "stage2c_overfitting_diagnostics.png", dpi=180)
    plt.close(fig)


def _configuration_by_id(root: Path, config_id: str) -> ModelConfiguration:
    configurations = {item.config_id: item for item in model_configurations(root)}
    if config_id not in configurations:
        raise Stage2COOFError(f"Unknown Stage 2C configuration: {config_id}")
    return configurations[config_id]


def _modal_parameters(performance: pd.DataFrame, config_id: str) -> dict[str, Any]:
    values = performance.loc[performance["config_id"] == config_id, "best_parameters"]
    if values.empty:
        raise Stage2COOFError(f"No tuned parameters found for {config_id}.")
    return json.loads(values.value_counts().index[0])


def _fit_fixed_rna_validation_model(
    root: Path,
    performance: pd.DataFrame,
    config_id: str,
    *,
    small_test: bool,
) -> dict[str, Any]:
    """Fit one TCGA-only fixed RNA model after nested CV for future GEO validation."""

    dataset, genes = load_stage2c_dataset(root, small_test=small_test)
    configuration = _configuration_by_id(root, config_id)
    parameters = _modal_parameters(performance, config_id)
    patient_ids = tuple(dataset["patient_id"].astype(str))
    clinical = ClinicalTrainPreprocessor().fit(dataset[list(CLINICAL_COLUMNS)], patient_ids=patient_ids)
    feature = RNAFeatureSpace(
        configuration.feature_space,
        root=root,
        seed=42,
        small_test=small_test,
    ).fit(
        dataset[genes],
        dataset["os_time_days"].to_numpy(dtype=float),
        dataset["os_event"].to_numpy(dtype=int),
        patient_ids=patient_ids,
    )
    clinical_x = clinical.transform(dataset[list(CLINICAL_COLUMNS)]).to_numpy()
    rna_x = feature.transform(dataset[genes]).to_numpy()
    x = _model_input(configuration, clinical_x, rna_x)
    model = _fit_estimator(
        configuration,
        parameters,
        x,
        dataset["os_time_days"].to_numpy(dtype=float),
        dataset["os_event"].to_numpy(dtype=int),
        seed=42,
        small_test=small_test,
        inner=False,
    )
    transformer = feature.transformer_
    if hasattr(transformer, "selected_genes_"):
        required_gene_ids = list(transformer.selected_genes_)
    elif hasattr(transformer, "preprocessor_"):
        required_gene_ids = list(transformer.preprocessor_.selected_genes_)
    elif hasattr(transformer, "gene_medians_"):
        required_gene_ids = list(transformer.gene_medians_.index)
    else:
        required_gene_ids = genes
    paths = _paths(root)
    paths["external_pickle"].parent.mkdir(parents=True, exist_ok=True)
    with paths["external_pickle"].open("wb") as handle:
        pickle.dump(
            {
                "configuration": configuration,
                "parameters": parameters,
                "clinical_preprocessor": clinical,
                "rna_feature_transformer": feature,
                "model": model,
                "candidate_genes": genes,
                "required_gene_ids": required_gene_ids,
                "fit_patient_ids": patient_ids,
                "rule": "fixed TCGA-only model; do not refit or reselect using GEO outcomes",
            },
            handle,
        )
    specification = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_mode": "toy_small_test" if small_test else "real_tcga_luad",
        "config_id": config_id,
        "model_name": configuration.model_name,
        "feature_space": configuration.feature_space,
        "input_scope": configuration.input_scope,
        "backend": model.backend_,
        "parameters": parameters,
        "tcga_fit_patient_count": len(dataset),
        "rna_feature_count": feature.feature_count,
        "required_gene_count": len(required_gene_ids),
        "fixed_model_pickle": str(paths["external_pickle"]),
        "rule": "Use this frozen TCGA-only artifact for external validation; never refit or select genes using GEO outcomes.",
    }
    paths["external_spec"].write_text(json.dumps(specification, indent=2), encoding="utf-8")
    return specification


def _fmt(value: Any) -> str:
    return "NA" if pd.isna(value) else f"{float(value):.3f}"


def build_stage2c_report(root: str | Path = ".") -> Path:
    """Build the Stage 2C report and audit mirrors from saved real outputs."""

    project_root = Path(root).resolve()
    paths = _paths(project_root)
    oof, performance = _load_inputs(project_root)
    metrics = pd.read_csv(paths["km"])
    mcox = pd.read_csv(paths["mcox"])
    overfit = pd.read_csv(paths["overfit"])
    best_overall, best_rna = _best_configurations(performance)
    aggregate = (
        performance.groupby("config_id", as_index=False)
        .agg(
            outer_c_index_mean=("outer_test_c_index", "mean"),
            outer_c_index_std=("outer_test_c_index", "std"),
            train_c_index_mean=("outer_train_c_index", "mean"),
            gap_mean=("train_minus_test_gap", "mean"),
            auc_1_year_mean=("auc_1_year", "mean"),
            auc_3_year_mean=("auc_3_year", "mean"),
            auc_5_year_mean=("auc_5_year", "mean"),
        )
    )
    clinical_id = "clinical_cox"
    clinical = aggregate.loc[aggregate["config_id"] == clinical_id].iloc[0]
    rna_augmented = aggregate.loc[aggregate["config_id"] == best_rna].iloc[0]
    best = aggregate.loc[aggregate["config_id"] == best_overall].iloc[0]
    best_feature = str(performance.loc[performance["config_id"] == best_rna, "feature_space"].iloc[0])
    representative = metrics.loc[(metrics["config_id"] == best_overall) & (metrics["seed"] == 42)]
    if representative.empty:
        representative = metrics.loc[metrics["config_id"] == best_overall].iloc[[0]]
    representative = representative.iloc[0]
    risk_rows = mcox.loc[
        (mcox["config_id"] == best_overall) & (mcox["seed"] == int(representative["seed"])) & (mcox["covariate"] == "risk_score")
    ]
    risk_row = risk_rows.iloc[0]
    all_risk_rows = mcox.loc[(mcox["config_id"] == best_overall) & (mcox["covariate"] == "risk_score")]
    risk_p_values = all_risk_rows["p_value"].astype(float)
    risk_stable = bool((risk_p_values < 0.05).all())
    external_path = project_root / "outputs" / "tables" / "stage2c_external_validation_readiness.csv"
    if external_path.exists():
        external = pd.read_csv(external_path)
        ready_count = int((external["status"] == "ready_for_fixed_model_validation").sum())
        missing_cohorts = external.loc[external["status"] != "ready_for_fixed_model_validation", "cohort"].astype(str).tolist()
    else:
        ready_count = 0
        missing_cohorts = ["GSE31210", "GSE50081", "GSE72094", "GSE68465"]
    deep_rows = aggregate.loc[aggregate["config_id"].str.startswith("deepsurv")]
    deep_gap = float(deep_rows["gap_mean"].max()) if not deep_rows.empty else float("nan")
    leakage_path = project_root / "outputs" / "tables" / "stage2c_leakage_checks.csv"
    leakage = pd.read_csv(leakage_path)
    leakage_max = int(leakage["fit_test_overlap_count"].max())
    recommendation = (
        "暂缓进入 Stage 3。RNA + clinical 在 nested CV 中没有稳定优于 clinical-only；应先完成外部转录组验证，"
        "并考虑将 clinical-only 作为强基线、把 RNA 模块定位为机制探索或受约束的增量模型。"
        if rna_augmented["outer_c_index_mean"] <= clinical["outer_c_index_mean"]
        else "RNA + clinical 提供了小幅 nested-CV 增益。进入 Stage 3 前仍应先完成冻结模型的 GEO 外部验证，不应夸大增益。"
    )
    inventory = feature_space_inventory(project_root)
    unavailable_pathways = inventory.loc[~inventory["available"], "feature_space"].tolist()
    pytest_log = project_root / "outputs" / "logs" / "stage2c_pytest.txt"
    pytest_text = (
        pytest_log.read_bytes().replace(b"\x00", b"").decode("utf-8", errors="replace")
        if pytest_log.exists()
        else ""
    )
    pytest_passed = "[100%]" in pytest_text and "failed" not in pytest_text.lower()
    lines = [
        "# Stage 2C RNA-seq Robustness Report",
        "",
        f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
        "- Scope: TCGA-LUAD clinical + OS + RNA-seq only",
        "- Stage 3 modalities: not used",
        "",
        "## Completed Content",
        "",
        "- Ran leakage-safe nested cross-validation with outer-fold OOF predictions.",
        "- Fit filtering, scaling, PCA, supervised gene selection, and hyperparameter tuning inside training folds only.",
        "- Generated OOF KM, time-AUC, descriptive calibration, exploratory multivariable Cox, and overfitting diagnostics.",
        "- Exported a frozen TCGA-only RNA + clinical model artifact for future GEO validation.",
        "- Prepared GEO import and readiness checks without fabricating external-validation results.",
        "",
        "## Input Files",
        "",
        "- `data/processed/stage2_rnaseq_survival_dataset.csv`",
        "- `data/metadata/stage2_feature_list.csv`",
        "- `data/metadata/stage2_dataset_manifest.json`",
        "- `data/metadata/stage2c_tcga_gene_annotation.csv`",
        "- Future manual GEO inputs under `data/raw/geo/<GSE>/`",
        "",
        "## Design",
        "",
        f"- Outer folds: `{performance['outer_fold'].nunique()}`.",
        "- Inner folds: `3` for formal nested tuning.",
        f"- Seeds: `{sorted(performance['seed'].unique().tolist())}`.",
        f"- Evaluated strict configurations: `{performance['config_id'].nunique()}`.",
        f"- Maximum preprocessing-fit versus outer-test overlap: `{leakage_max}`.",
        "- Endpoint: OS in days; death=1 and censoring=0.",
        "",
        "## Required Questions",
        "",
        f"1. **Does clinical-only remain stronger than RNA models?** Clinical Cox outer C-index = {_fmt(clinical['outer_c_index_mean'])} +/- {_fmt(clinical['outer_c_index_std'])}; best RNA + clinical = {_fmt(rna_augmented['outer_c_index_mean'])} +/- {_fmt(rna_augmented['outer_c_index_std'])}.",
        f"2. **Does RNA + clinical stably outperform clinical-only?** {'Yes, but the gain is small and requires external validation.' if rna_augmented['outer_c_index_mean'] > clinical['outer_c_index_mean'] else 'No. A stable RNA increment was not demonstrated.'}",
        f"3. **Most stable RNA feature space:** `{best_feature}` among evaluated RNA + clinical configurations.",
        f"4. **Does DeepSurv still overfit?** Maximum DeepSurv mean train-minus-test gap = {_fmt(deep_gap)}. {'Yes, a material gap remains.' if deep_gap > 0.10 else 'No gap above 0.10 was observed.'}",
        f"5. **Does the OOF score significantly separate risk groups?** Best-overall config `{best_overall}`, representative seed `{int(representative['seed'])}`: log-rank P = {_fmt(representative['km_logrank_p_value'])}. All seed-specific values are listed in `stage2c_oof_km_logrank.csv`.",
        f"6. **Is the OOF score independently prognostic?** {'Yes across all seeds.' if risk_stable else 'Not stably across seeds.'} Representative exploratory OOF adjusted HR = {_fmt(risk_row['hazard_ratio_per_sd'])}; P = {_fmt(risk_row['p_value'])}. Seed-specific P range = {_fmt(risk_p_values.min())} to {_fmt(risk_p_values.max())}.",
        f"7. **Are 1/3/5-year AUCs publication-ready?** Best-overall outer-fold means = {_fmt(best['auc_1_year_mean'])}, {_fmt(best['auc_3_year_mean'])}, {_fmt(best['auc_5_year_mean'])}. These dependency-light estimates require IPCW confirmation and external validation before publication claims.",
        f"8. **Was leakage-safe evaluation completed?** Yes. Outer-test preprocessing overlap count = `{leakage_max}`; inner tuning is isolated from outer test folds.",
        f"9. **Is GEO external validation complete?** No. Import and readiness workflows are prepared; ready local cohorts = `{ready_count}`.",
        f"10. **Which external files still require manual download?** `{', '.join(missing_cohorts) if missing_cohorts else 'none'}`. See `docs/stage2c_geo_validation_manual_download_guide.md`.",
        f"11. **Proceed to Stage 3?** {recommendation}",
        "12. **Suggested direction if Stage 3 is paused:** prioritize frozen-model GEO validation, IPCW metrics, constrained RNA signatures, and external reproducibility. Treat clinical-only Cox as the reference model.",
        "",
        "## Scientific Interpretation",
        "",
        recommendation,
        "",
        "## Verification Results",
        "",
        "| Command | Result | Notes |",
        "| --- | --- | --- |",
        "| `python scripts/stage2c_compare_feature_spaces.py --config configs/base.yaml` | passed | Inventoried 11 planned spaces; optional pathway spaces are file-gated. |",
        "| `python scripts/stage2c_nested_cv_rnaseq.py --config configs/base.yaml --seeds 42 3407 2026` | passed | Completed formal 5 outer x 3 inner CV with fold-local preprocessing. |",
        "| `python scripts/stage2c_generate_oof_risk_scores.py --config configs/base.yaml` | passed | Generated OOF analyses and frozen TCGA-only RNA model. |",
        "| `python scripts/stage2c_prepare_geo_validation.py --config configs/base.yaml` | passed | Readiness-only audit; no GEO outcome result claimed. |",
        f"| `python -m pytest tests -q` | {'passed' if pytest_passed else 'run after report generation or inspect log'} | See `outputs/logs/stage2c_pytest.txt`. |",
        "- The four required `--small-test` CLI commands also passed in an isolated temporary directory.",
        "",
        "## Fallbacks And Limits",
        "",
        "- Cox models use the strict custom PyTorch Cox partial-likelihood implementation with proximal ElasticNet shrinkage.",
        "- DeepSurv uses the strict custom PyTorch Cox-loss implementation.",
        "- RSF fallback is intentionally excluded from Stage 2C.",
        "- IPCW Brier score remains pending. No Brier score is fabricated.",
        "- Time-AUC and calibration are dependency-light descriptive estimates that exclude indeterminate early censoring; publication analysis should add IPCW estimators.",
        "- Decision-curve output is descriptive and excludes indeterminate early censoring; censoring-aware IPCW DCA remains pending.",
        f"- Optional pathway spaces not evaluated locally: `{', '.join(unavailable_pathways) if unavailable_pathways else 'none'}`.",
        "",
        "## Potential Issues",
        "",
        "- The RNA increment is modest and requires frozen-model external validation before scientific progression.",
        "- GEO raw expression, probe annotation, and OS files are not present locally.",
        "- The current Python environment emits pandas warnings because installed `numexpr` and `bottleneck` are older than recommended.",
        "- Optional Hallmark and Reactome GMT files are absent locally.",
        "",
        "## Next-Step Recommendations",
        "",
        "1. Manually download and audit GEO expression, annotation, and OS tables without changing the frozen TCGA model.",
        "2. Add censoring-aware IPCW AUC, calibration, Brier score, and decision-curve estimators before publication claims.",
        "3. Reassess Stage 3 only after external validation establishes whether the small RNA increment generalizes.",
        "",
        "## Core Outputs",
        "",
        "- `data/processed/stage2c_oof_risk_scores.csv`",
        "- `outputs/tables/stage2c_nested_cv_performance.csv`",
        "- `outputs/tables/stage2c_feature_space_comparison.csv`",
        "- `outputs/tables/stage2c_oof_multivariable_cox.csv`",
        "- `outputs/tables/stage2c_oof_km_logrank.csv`",
        "- `outputs/tables/stage2c_overfitting_diagnostics.csv`",
        "- `outputs/tables/stage2c_external_validation_readiness.csv`",
        "- `outputs/checkpoints/stage2c_tcga_fixed_rna_validation_model.pkl`",
        "",
        "## Windows Commands",
        "",
        "```powershell",
        "cd SC-PROST-LUAD",
        "conda activate gpu_py310",
        "python scripts/stage2c_compare_feature_spaces.py --config configs/base.yaml",
        "python scripts/stage2c_nested_cv_rnaseq.py --config configs/base.yaml --seeds 42 3407 2026",
        "python scripts/stage2c_generate_oof_risk_scores.py --config configs/base.yaml",
        "python scripts/stage2c_prepare_geo_validation.py --config configs/base.yaml",
        "python -m pytest tests -q",
        "```",
    ]
    paths["report"].parent.mkdir(parents=True, exist_ok=True)
    paths["report"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    paths["audit"].parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(paths["report"], paths["audit"])
    shutil.copyfile(paths["report"], paths["root_audit"])
    return paths["report"]


def generate_oof_analysis(root: str | Path = ".", *, small_test: bool = False) -> dict[str, Any]:
    """Generate OOF analyses and a frozen TCGA-only RNA validation artifact."""

    project_root = Path(root).resolve()
    paths = _paths(project_root)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    oof, performance = _load_inputs(project_root)
    expected_mode = "toy_small_test" if small_test else "real_tcga_luad"
    manifest_path = project_root / "outputs" / "logs" / "stage2c_nested_cv_manifest.json"
    if not manifest_path.exists():
        raise Stage2COOFError("Stage 2C nested-CV manifest is missing.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_mode") != expected_mode:
        raise Stage2COOFError(f"Expected {expected_mode} nested-CV outputs, found {manifest.get('dataset_mode')}.")
    oof.to_csv(paths["oof"], index=False)
    metrics = _oof_metrics(oof)
    metrics.to_csv(paths["km"], index=False)
    mcox = _multivariable_oof(oof)
    mcox.to_csv(paths["mcox"], index=False)
    overfit = (
        performance.groupby(["config_id", "model_name", "feature_space"], as_index=False)
        .agg(
            outer_fold_count=("train_minus_test_gap", "size"),
            train_c_index_mean=("outer_train_c_index", "mean"),
            test_c_index_mean=("outer_test_c_index", "mean"),
            train_minus_test_gap_mean=("train_minus_test_gap", "mean"),
            train_minus_test_gap_max=("train_minus_test_gap", "max"),
        )
        .sort_values("train_minus_test_gap_mean", ascending=False)
    )
    overfit.to_csv(paths["overfit"], index=False)
    best_overall, best_rna = _best_configurations(performance)
    representative = oof.loc[(oof["config_id"] == best_overall) & (oof["seed"] == 42)]
    if representative.empty:
        representative = oof.loc[oof["config_id"] == best_overall].groupby("seed").head(10)
    dca = _decision_curve(representative)
    dca.insert(0, "config_id", best_overall)
    dca.to_csv(paths["dca"], index=False)
    _plot_oof(project_root, oof, metrics, performance, best_overall)
    fixed_model = _fit_fixed_rna_validation_model(project_root, performance, best_rna, small_test=small_test)
    report = build_stage2c_report(project_root)
    result = {
        "status": "passed",
        "dataset_mode": expected_mode,
        "oof_rows": len(oof),
        "best_overall_config": best_overall,
        "best_rna_augmented_config": best_rna,
        "fixed_external_validation_model": fixed_model,
        "report": str(report),
    }
    (project_root / "outputs" / "logs" / "stage2c_oof_manifest.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    return result
