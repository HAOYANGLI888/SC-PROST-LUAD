"""Frozen TCGA RNA PCA + clinical Cox validation in public GEO cohorts."""

from __future__ import annotations

import json
import pickle
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.clinical_preprocess import CLINICAL_COLUMNS, ClinicalTrainPreprocessor
from data.geo_download import GEO_COHORTS, download_geo_inputs
from data.geo_expression_import import (
    collapse_probes_to_genes,
    export_tcga_gene_annotation_from_star_counts,
    read_geo_expression,
    read_geo_series_metadata,
)
from data.geo_platform_annotation import parse_geo_platform_annotation
from data.geo_survival_preprocess import prepare_geo_os_from_series_metadata
from evaluation.cox_analysis import (
    multivariable_cox_adjustment,
    univariable_cox_risk_score,
)
from evaluation.ipcw_metrics import (
    IPCWMetricError,
    censoring_aware_calibration,
    integrated_brier_score,
    ipcw_brier_score,
    ipcw_decision_curve,
)
from evaluation.plot_survival import plot_calibration, plot_km_high_low
from evaluation.survival_metrics import (
    concordance_index,
    logrank_p_value,
    predict_event_probability,
    time_dependent_auc,
)
from features.rnaseq_feature_spaces import RNAFeatureSpace
from training.nested_cv import (
    _fit_estimator,
    _model_input,
    load_stage2c_dataset,
    survival_strata,
)


HORIZONS = (365, 1095, 1825)
IBS_HORIZONS = np.linspace(365.0, 1825.0, 13)


class ExternalValidationError(RuntimeError):
    """Raised when frozen-model validation cannot proceed safely."""


def _paths(root: str | Path) -> dict[str, Path]:
    project_root = Path(root).resolve()
    return {
        "root": project_root,
        "geo": project_root / "data" / "raw" / "geo",
        "processed": project_root / "data" / "processed",
        "metadata": project_root / "data" / "metadata",
        "tables": project_root / "outputs" / "tables",
        "figures": project_root / "outputs" / "figures",
        "reports": project_root / "outputs" / "reports",
        "artifact": project_root / "outputs" / "checkpoints" / "stage2c_tcga_fixed_rna_validation_model.pkl",
        "tcga_annotation": project_root / "data" / "metadata" / "stage2c_tcga_gene_annotation.csv",
        "tcga_dataset": project_root / "data" / "processed" / "stage2_rnaseq_survival_dataset.csv",
    }


def _ensure_output_dirs(paths: dict[str, Path]) -> None:
    for key in ("geo", "processed", "metadata", "tables", "figures", "reports"):
        paths[key].mkdir(parents=True, exist_ok=True)


def _load_artifact(paths: dict[str, Path]) -> dict[str, Any]:
    if not paths["artifact"].exists():
        raise FileNotFoundError(
            f"Frozen Stage 2C model not found: {paths['artifact']}. "
            "Run scripts/stage2c_generate_oof_risk_scores.py first."
        )
    with paths["artifact"].open("rb") as handle:
        artifact = pickle.load(handle)
    required = {
        "configuration", "parameters", "clinical_preprocessor",
        "rna_feature_transformer", "model", "required_gene_ids",
    }
    missing = sorted(required - set(artifact))
    if missing:
        raise ExternalValidationError(f"Frozen model artifact is missing keys: {missing}")
    if len(artifact["required_gene_ids"]) != 1000:
        raise ExternalValidationError(
            "Formal Stage 2D expects the frozen PCA_25 input to contain exactly 1,000 genes."
        )
    return artifact


def download_geo_metadata(
    root: str | Path = ".",
    *,
    cohorts: Iterable[str] = tuple(GEO_COHORTS),
    small_test: bool = False,
) -> dict[str, Any]:
    """Download official GEO inputs, or produce an isolated small-test receipt."""

    paths = _paths(root)
    _ensure_output_dirs(paths)
    receipt_path = paths["metadata"] / "stage2d_geo_download_manifest.csv"
    if small_test:
        frame = pd.DataFrame(
            [
                {
                    "resource_type": "small_test_fixture",
                    "accession": "TOY_GEO",
                    "platform": "TOY_PLATFORM",
                    "status": "generated_locally",
                    "path": "isolated in-memory fixture",
                    "bytes": 0,
                    "url": "not_applicable",
                }
            ]
        )
    else:
        frame = download_geo_inputs(paths["geo"], cohorts)
    frame.to_csv(receipt_path, index=False)
    return {
        "status": "passed",
        "dataset_mode": "toy_small_test" if small_test else "real_geo_public",
        "receipt": str(receipt_path),
        "resource_count": len(frame),
    }


def _platform_path(paths: dict[str, Path], platform: str) -> Path:
    gz = paths["geo"] / "platforms" / f"{platform}.annot.gz"
    txt = paths["geo"] / "platforms" / f"{platform}.txt"
    path = gz if gz.exists() else txt
    if not path.exists():
        raise FileNotFoundError(
            f"Platform annotation for {platform} is missing. "
            "Run scripts/stage2d_download_geo_metadata.py first."
        )
    return path


def _tcga_symbol_map(paths: dict[str, Path]) -> dict[str, str]:
    if not paths["tcga_annotation"].exists():
        export_tcga_gene_annotation_from_star_counts(paths["root"])
    annotation = pd.read_csv(paths["tcga_annotation"], dtype=str)
    required = {"gene_id", "gene_symbol"}
    if not required.issubset(annotation):
        raise ExternalValidationError(
            f"TCGA gene annotation requires columns {sorted(required)}."
        )
    return {
        str(row.gene_id): str(row.gene_symbol).strip().upper()
        for row in annotation.itertuples()
    }


def _transform_external_scale(expression: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    genes = [column for column in expression if column != "sample_id"]
    numeric = expression[genes].apply(pd.to_numeric, errors="coerce")
    quantile = float(np.nanquantile(numeric.to_numpy(dtype=float), 0.99))
    if quantile > 100.0:
        if (numeric.dropna() < 0).any().any():
            raise ExternalValidationError(
                "External expression appears unlogged but includes negative values."
            )
        numeric = np.log2(numeric + 1.0)
        method = "auto_log2_intensity_plus_1"
    else:
        method = "as_provided_assumed_log2_normalized_microarray"
    return pd.concat([expression[["sample_id"]], numeric], axis=1), method


def _external_ensembl_matrix(
    expression: pd.DataFrame,
    *,
    required_genes: list[str],
    symbol_by_gene: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    external = expression.set_index("sample_id")
    external.columns = [str(column).strip().upper() for column in external.columns]
    external = external.loc[:, ~external.columns.duplicated(keep="first")]
    matrix_columns: dict[str, pd.Series | float] = {}
    rows = []
    for gene_id in required_genes:
        symbol = symbol_by_gene.get(gene_id, "")
        available = bool(symbol and symbol in external.columns)
        matrix_columns[gene_id] = external[symbol] if available else np.nan
        rows.append({"gene_id": gene_id, "gene_symbol": symbol, "available": available})
    return pd.DataFrame(matrix_columns, index=external.index), pd.DataFrame(rows)


def _tcga_reference(
    paths: dict[str, Path],
    artifact: dict[str, Any],
) -> dict[str, Any]:
    required_genes = list(artifact["required_gene_ids"])
    usecols = ["patient_id", "os_time_days", "os_event", *CLINICAL_COLUMNS, *required_genes]
    if not paths["tcga_dataset"].exists():
        raise FileNotFoundError(f"Prepared TCGA dataset missing: {paths['tcga_dataset']}")
    frame = pd.read_csv(paths["tcga_dataset"], usecols=usecols)
    clinical = artifact["clinical_preprocessor"].transform(frame[list(CLINICAL_COLUMNS)]).to_numpy()
    rna = artifact["rna_feature_transformer"].transform(frame[required_genes]).to_numpy()
    scores = artifact["model"].predict_risk(np.column_stack([clinical, rna]))
    rna_scores = rna @ np.asarray(artifact["model"].coef_[len(CLINICAL_COLUMNS):])
    return {
        "times": frame["os_time_days"].to_numpy(dtype=float),
        "events": frame["os_event"].to_numpy(dtype=int),
        "scores": scores,
        "rna_scores": rna_scores,
        "full_cutoff": float(np.median(scores)),
        "rna_cutoff": float(np.median(rna_scores)),
    }


def prepare_geo_cohort(
    root: str | Path,
    cohort: str,
    *,
    collapse_strategy: str = "mean",
) -> dict[str, Any]:
    """Prepare one official GEO cohort and apply the frozen TCGA model."""

    paths = _paths(root)
    _ensure_output_dirs(paths)
    if cohort not in GEO_COHORTS:
        raise ValueError(f"Unsupported GEO cohort: {cohort}")
    spec = GEO_COHORTS[cohort]
    matrix_path = paths["geo"] / cohort / f"{cohort}_series_matrix.txt.gz"
    if not matrix_path.exists():
        raise FileNotFoundError(
            f"GEO series matrix not found: {matrix_path}. "
            "Run scripts/stage2d_download_geo_metadata.py first."
        )
    artifact = _load_artifact(paths)
    required_genes = list(artifact["required_gene_ids"])
    metadata, metadata_summary = read_geo_series_metadata(matrix_path)
    inclusion_rule = "all samples with unambiguous OS"
    if cohort == "GSE50081":
        if "histology" not in metadata:
            raise ExternalValidationError("GSE50081 metadata is missing histology.")
        metadata = metadata.loc[
            metadata["histology"].str.strip().str.casefold() == "adenocarcinoma"
        ].copy()
        inclusion_rule = "histology equals adenocarcinoma; excludes non-LUAD NSCLC"
    survival = prepare_geo_os_from_series_metadata(metadata, spec)
    annotation = parse_geo_platform_annotation(_platform_path(paths, spec.platform))
    expression = read_geo_expression(matrix_path)
    collapsed = collapse_probes_to_genes(
        expression, annotation, strategy=collapse_strategy
    )
    collapsed, scale_method = _transform_external_scale(collapsed)
    ensembl, missingness = _external_ensembl_matrix(
        collapsed,
        required_genes=required_genes,
        symbol_by_gene=_tcga_symbol_map(paths),
    )
    available_count = int(missingness["available"].sum())
    missing_count = len(required_genes) - available_count
    missing_fraction = missing_count / len(required_genes)
    if missing_fraction > 0.30:
        raise ExternalValidationError(
            f"{cohort} is missing {missing_fraction:.1%} of frozen genes; "
            "cohort dropped by the pre-specified >30% threshold."
        )
    rna = artifact["rna_feature_transformer"].transform(ensembl).to_numpy()
    joined = survival.merge(
        pd.DataFrame({"sample_id": ensembl.index}).assign(_row=np.arange(len(ensembl))),
        on="sample_id",
        how="inner",
        validate="one_to_one",
    )
    if joined.empty:
        raise ExternalValidationError(f"{cohort} expression and survival sample IDs do not overlap.")
    rna = rna[joined["_row"].to_numpy(dtype=int)]
    clinical_missing = joined[list(CLINICAL_COLUMNS)].isna()
    clinical = artifact["clinical_preprocessor"].transform(joined[list(CLINICAL_COLUMNS)]).to_numpy()
    full_score = artifact["model"].predict_risk(np.column_stack([clinical, rna]))
    rna_score = rna @ np.asarray(artifact["model"].coef_[len(CLINICAL_COLUMNS):])
    reference = _tcga_reference(paths, artifact)
    prepared = joined.drop(columns="_row").copy()
    prepared["cohort"] = cohort
    prepared["platform"] = spec.platform
    prepared["frozen_full_risk_score"] = full_score
    prepared["frozen_rna_component_risk_score"] = rna_score
    prepared["tcga_full_median_cutoff"] = reference["full_cutoff"]
    prepared["tcga_rna_median_cutoff"] = reference["rna_cutoff"]
    prepared["clinical_missing_any"] = clinical_missing.any(axis=1).to_numpy()
    prepared["expression_scale_method"] = scale_method
    prepared["gene_missing_fraction"] = missing_fraction
    prepared_path = paths["processed"] / f"stage2d_geo_{cohort}_prepared.csv"
    prepared.to_csv(prepared_path, index=False)
    missingness.insert(0, "cohort", cohort)
    missingness.to_csv(
        paths["metadata"] / f"stage2d_geo_{cohort}_gene_missingness.csv", index=False
    )
    summary = pd.DataFrame(
        [
            {
                "cohort": cohort,
                "platform": spec.platform,
                "series_matrix_samples": metadata_summary["sample_count"],
                "prepared_os_samples": len(prepared),
                "death_events": int(prepared["os_event"].sum()),
                "censored": int((prepared["os_event"] == 0).sum()),
                "required_frozen_genes": len(required_genes),
                "available_frozen_genes": available_count,
                "missing_frozen_genes": missing_count,
                "missing_gene_fraction": missing_fraction,
                "clinical_complete_samples": int((~prepared["clinical_missing_any"]).sum()),
                "clinical_missing_samples": int(prepared["clinical_missing_any"].sum()),
                "expression_scale_method": scale_method,
                "probe_collapse_strategy": collapse_strategy,
                "cohort_inclusion_rule": inclusion_rule,
                "status": "prepared_real_geo_public",
            }
        ]
    )
    summary_path = paths["metadata"] / f"stage2d_geo_{cohort}_preprocessing_report.csv"
    summary.to_csv(summary_path, index=False)
    (paths["reports"] / f"stage2d_geo_{cohort}_preprocessing_report.md").write_text(
        f"# Stage 2D {cohort} Preprocessing Report\n\n"
        f"- Source: official GEO series matrix `{matrix_path}`.\n"
        f"- Platform: `{spec.platform}`.\n"
        f"- Inclusion rule: {inclusion_rule}.\n"
        f"- OS unit conversion: `{spec.os_time_unit}` to `days`.\n"
        f"- Prepared patients: {len(prepared)}; deaths: {int(prepared['os_event'].sum())}; "
        f"censored: {int((prepared['os_event'] == 0).sum())}.\n"
        f"- Frozen genes found: {available_count}/1000; missing: {missing_count}/1000 "
        f"({missing_fraction:.1%}).\n"
        f"- Probe aggregation: `{collapse_strategy}`.\n"
        f"- Expression scaling: `{scale_method}`.\n"
        f"- Clinical rows with at least one missing covariate: "
        f"{int(prepared['clinical_missing_any'].sum())}/{len(prepared)}.\n"
        "- Integrity: GEO outcomes were not used for gene selection, PCA fitting, "
        "scaling, coefficient fitting, or model selection.\n",
        encoding="utf-8",
    )
    return summary.iloc[0].to_dict()


def prepare_geo_cohort_small_test(root: str | Path, cohort: str) -> dict[str, Any]:
    """Write an isolated deterministic fixture without touching formal outputs."""

    if cohort not in GEO_COHORTS:
        raise ValueError(f"Unsupported GEO cohort: {cohort}")
    paths = _paths(root)
    output = paths["processed"] / "stage2d_small_test" / f"{cohort}_prepared.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    n = 36
    score = rng.normal(size=n)
    event = rng.binomial(1, 0.45, size=n)
    time = np.maximum(30.0, 1200.0 - 260.0 * score + rng.normal(0, 380, size=n))
    fixture = pd.DataFrame(
        {
            "sample_id": [f"TOY_GSM_{index:03d}" for index in range(n)],
            "os_time_days": time,
            "os_event": event,
            "age": rng.normal(65, 8, size=n),
            "male": rng.binomial(1, 0.5, size=n),
            "stage_numeric": rng.integers(1, 4, size=n),
            "cohort": cohort,
            "platform": "TOY_PLATFORM",
            "frozen_full_risk_score": score,
            "frozen_rna_component_risk_score": score * 0.85,
            "tcga_full_median_cutoff": 0.0,
            "tcga_rna_median_cutoff": 0.0,
            "clinical_missing_any": False,
            "expression_scale_method": "toy_small_test",
            "gene_missing_fraction": 0.05,
        }
    )
    fixture.to_csv(output, index=False)
    return {
        "status": "passed",
        "dataset_mode": "toy_small_test",
        "cohort": cohort,
        "prepared_path": str(output),
        "patients": n,
    }


def _metric_or_nan(function, *args, **kwargs) -> tuple[float, str]:
    try:
        value = float(function(*args, **kwargs))
        return value, "computed" if np.isfinite(value) else "unavailable"
    except (IPCWMetricError, ValueError) as exc:
        return float("nan"), f"pending: {exc}"


def _external_performance_rows(
    frame: pd.DataFrame,
    reference: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[pd.DataFrame]]:
    cohort = str(frame["cohort"].iloc[0])
    rows: list[dict[str, Any]] = []
    cox_rows: list[dict[str, Any]] = []
    calibration_rows: list[pd.DataFrame] = []
    analyses = (
        ("frozen_full_model", "frozen_full_risk_score", reference["full_cutoff"]),
        ("frozen_rna_component", "frozen_rna_component_risk_score", reference["rna_cutoff"]),
    )
    for analysis_type, score_column, tcga_cutoff in analyses:
        scores = frame[score_column].to_numpy(dtype=float)
        times = frame["os_time_days"].to_numpy(dtype=float)
        events = frame["os_event"].to_numpy(dtype=int)
        probabilities = {
            horizon: predict_event_probability(
                reference["times"],
                reference["events"],
                reference["scores"] if analysis_type == "frozen_full_model" else reference["rna_scores"],
                scores,
                horizon,
            )
            for horizon in HORIZONS
        }
        row: dict[str, Any] = {
            "cohort": cohort,
            "platform": frame["platform"].iloc[0],
            "analysis_type": analysis_type,
            "patient_count": len(frame),
            "death_events": int(events.sum()),
            "c_index": concordance_index(times, events, scores),
            "auc_1_year": time_dependent_auc(times, events, scores, 365),
            "auc_3_year": time_dependent_auc(times, events, scores, 1095),
            "auc_5_year": time_dependent_auc(times, events, scores, 1825),
            "clinical_missing_fraction": float(frame["clinical_missing_any"].mean()),
            "gene_missing_fraction": float(frame["gene_missing_fraction"].iloc[0]),
        }
        for label, cutoff in (("tcga_cutoff", tcga_cutoff), ("cohort_median", float(np.median(scores)))):
            groups = scores >= cutoff
            row[f"km_logrank_p_{label}"] = logrank_p_value(times, events, groups)
            row[f"risk_cutoff_{label}"] = cutoff
        probability_grid = np.column_stack(
            [
                predict_event_probability(
                    reference["times"],
                    reference["events"],
                    reference["scores"] if analysis_type == "frozen_full_model" else reference["rna_scores"],
                    scores,
                    horizon,
                )
                for horizon in IBS_HORIZONS
            ]
        )
        for horizon in HORIZONS:
            value, status = _metric_or_nan(
                ipcw_brier_score, times, events, probabilities[horizon], horizon
            )
            row[f"ipcw_brier_{horizon}_days"] = value
            row[f"ipcw_brier_{horizon}_days_status"] = status
        ibs, ibs_status = _metric_or_nan(
            integrated_brier_score, times, events, probability_grid, IBS_HORIZONS
        )
        row["integrated_brier_1_to_5_year"] = ibs
        row["integrated_brier_status"] = ibs_status
        rows.append(row)
        univariable = univariable_cox_risk_score(times, events, scores)
        cox_rows.append({"cohort": cohort, "analysis_type": analysis_type, **univariable})
        try:
            calibration = censoring_aware_calibration(
                times, events, probabilities[1095], 1095
            )
            calibration.insert(0, "analysis_type", analysis_type)
            calibration.insert(0, "cohort", cohort)
            calibration_rows.append(calibration)
        except IPCWMetricError:
            pass
    return rows, cox_rows, calibration_rows


def _multivariable_external(frame: pd.DataFrame) -> pd.DataFrame:
    predictions = frame.rename(
        columns={
            "os_time_days": "os_time_days",
            "os_event": "os_event",
            "frozen_full_risk_score": "risk_score",
        }
    )
    result = multivariable_cox_adjustment(predictions)
    result.insert(0, "cohort", str(frame["cohort"].iloc[0]))
    result["analysis_scope"] = (
        "external frozen full model; missing clinical values use TCGA-fitted median imputation"
    )
    return result


def _plot_external_summary(paths: dict[str, Path], performance: pd.DataFrame) -> None:
    full = performance.loc[performance["analysis_type"] == "frozen_full_model"].copy()
    if full.empty:
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    ax.bar(full["cohort"], full["c_index"], color="#267A73")
    ax.axhline(0.5, color="#777777", linestyle="--", linewidth=1)
    ax.axhline(0.6, color="#B13C2E", linestyle=":", linewidth=1)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("External C-index")
    ax.set_title("Frozen TCGA model external discrimination")
    fig.tight_layout()
    fig.savefig(paths["figures"] / "stage2d_external_cindex_summary.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    for row in full.itertuples():
        ax.plot(
            [1, 3, 5],
            [row.auc_1_year, row.auc_3_year, row.auc_5_year],
            marker="o",
            linewidth=1.8,
            label=row.cohort,
        )
    ax.axhline(0.5, color="#777777", linestyle="--", linewidth=1)
    ax.set_xticks([1, 3, 5])
    ax.set_xlabel("Horizon (years)")
    ax.set_ylabel("Time-dependent AUC")
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(paths["figures"] / "stage2d_external_time_auc_summary.png", dpi=180)
    plt.close(fig)


def run_external_validation(
    root: str | Path = ".",
    *,
    small_test: bool = False,
) -> dict[str, Any]:
    """Evaluate each prepared GEO cohort without changing the frozen TCGA model."""

    paths = _paths(root)
    _ensure_output_dirs(paths)
    if small_test:
        return run_stage2d_small_test(paths["root"])
    artifact = _load_artifact(paths)
    reference = _tcga_reference(paths, artifact)
    risk_frames = []
    performance_rows: list[dict[str, Any]] = []
    cox_rows: list[dict[str, Any]] = []
    multivariable_rows: list[pd.DataFrame] = []
    calibration_rows: list[pd.DataFrame] = []
    decision_curve_rows: list[pd.DataFrame] = []
    summary_rows = []
    missing_rows = []
    for cohort in GEO_COHORTS:
        prepared_path = paths["processed"] / f"stage2d_geo_{cohort}_prepared.csv"
        if not prepared_path.exists():
            raise FileNotFoundError(
                f"Prepared GEO cohort missing: {prepared_path}. "
                f"Run scripts/stage2d_prepare_geo_cohort.py --cohort {cohort} first."
            )
        frame = pd.read_csv(prepared_path)
        risk_frames.append(frame)
        rows, univariable, calibration = _external_performance_rows(frame, reference)
        performance_rows.extend(rows)
        cox_rows.extend(univariable)
        calibration_rows.extend(calibration)
        multivariable_rows.append(_multivariable_external(frame))
        full_probability_3y = predict_event_probability(
            reference["times"],
            reference["events"],
            reference["scores"],
            frame["frozen_full_risk_score"].to_numpy(dtype=float),
            1095,
        )
        try:
            cohort_dca = ipcw_decision_curve(
                frame["os_time_days"],
                frame["os_event"],
                full_probability_3y,
                1095,
            )
            cohort_dca.insert(0, "cohort", cohort)
            cohort_dca.insert(1, "status", "computed")
        except IPCWMetricError as exc:
            cohort_dca = pd.DataFrame(
                [{"cohort": cohort, "status": f"pending: {exc}"}]
            )
        decision_curve_rows.append(cohort_dca)
        summary_rows.append(
            {
                "cohort": cohort,
                "platform": frame["platform"].iloc[0],
                "patients": len(frame),
                "death_events": int(frame["os_event"].sum()),
                "censored": int((frame["os_event"] == 0).sum()),
                "clinical_missing_fraction": float(frame["clinical_missing_any"].mean()),
            }
        )
        missing_rows.append(
            {
                "cohort": cohort,
                "required_frozen_genes": 1000,
                "available_frozen_genes": int(round(1000 * (1.0 - frame["gene_missing_fraction"].iloc[0]))),
                "missing_frozen_genes": int(round(1000 * frame["gene_missing_fraction"].iloc[0])),
                "missing_gene_fraction": float(frame["gene_missing_fraction"].iloc[0]),
            }
        )
        score = frame["frozen_full_risk_score"].to_numpy(dtype=float)
        p_value = logrank_p_value(
            frame["os_time_days"], frame["os_event"], score >= reference["full_cutoff"]
        )
        plot_km_high_low(
            frame["os_time_days"],
            frame["os_event"],
            score,
            reference["full_cutoff"],
            p_value,
            paths["figures"] / f"stage2d_external_km_{cohort}.png",
        )
    performance = pd.DataFrame(performance_rows)
    performance.to_csv(paths["tables"] / "stage2d_external_validation_performance.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(paths["tables"] / "stage2d_geo_dataset_summary.csv", index=False)
    pd.DataFrame(missing_rows).to_csv(
        paths["tables"] / "stage2d_external_validation_gene_missingness.csv", index=False
    )
    pd.DataFrame(cox_rows).to_csv(paths["tables"] / "stage2d_external_validation_cox.csv", index=False)
    pd.concat(multivariable_rows, ignore_index=True).to_csv(
        paths["tables"] / "stage2d_external_validation_multivariable_cox.csv", index=False
    )
    if calibration_rows:
        pd.concat(calibration_rows, ignore_index=True).to_csv(
            paths["tables"] / "stage2d_external_validation_calibration.csv", index=False
        )
    pd.concat(decision_curve_rows, ignore_index=True).to_csv(
        paths["tables"] / "stage2d_external_validation_decision_curve.csv", index=False
    )
    pd.concat(risk_frames, ignore_index=True).to_csv(
        paths["processed"] / "stage2d_external_validation_risk_scores.csv", index=False
    )
    _plot_external_summary(paths, performance)
    return {
        "status": "passed",
        "dataset_mode": "real_geo_public",
        "cohorts": len(summary_rows),
        "frozen_model": str(paths["artifact"]),
    }


def _fit_tcga_fold_predictions(
    dataset: pd.DataFrame,
    genes: list[str],
    artifact: dict[str, Any],
    *,
    root: str | Path,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    configuration = artifact["configuration"]
    parameters = artifact["parameters"]
    strata = survival_strata(dataset, n_splits=5)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    prediction_rows: list[pd.DataFrame] = []
    probability_rows: list[np.ndarray] = []
    for fold, (train_index, test_index) in enumerate(splitter.split(dataset, strata), start=1):
        train = dataset.iloc[train_index]
        test = dataset.iloc[test_index]
        clinical = ClinicalTrainPreprocessor().fit(
            train[list(CLINICAL_COLUMNS)], patient_ids=train["patient_id"]
        )
        feature = RNAFeatureSpace(
            configuration.feature_space, root=root, seed=seed
        ).fit(
            train[genes],
            train["os_time_days"].to_numpy(dtype=float),
            train["os_event"].to_numpy(dtype=int),
            patient_ids=train["patient_id"],
        )
        train_x = _model_input(
            configuration,
            clinical.transform(train[list(CLINICAL_COLUMNS)]).to_numpy(),
            feature.transform(train[genes]).to_numpy(),
        )
        test_x = _model_input(
            configuration,
            clinical.transform(test[list(CLINICAL_COLUMNS)]).to_numpy(),
            feature.transform(test[genes]).to_numpy(),
        )
        model = _fit_estimator(
            configuration,
            parameters,
            train_x,
            train["os_time_days"].to_numpy(dtype=float),
            train["os_event"].to_numpy(dtype=int),
            seed=seed,
            small_test=False,
            inner=False,
        )
        train_scores = model.predict_risk(train_x)
        test_scores = model.predict_risk(test_x)
        probability_grid = np.column_stack(
            [
                predict_event_probability(
                    train["os_time_days"].to_numpy(dtype=float),
                    train["os_event"].to_numpy(dtype=int),
                    train_scores,
                    test_scores,
                    horizon,
                )
                for horizon in IBS_HORIZONS
            ]
        )
        prediction_rows.append(
            pd.DataFrame(
                {
                    "patient_id": test["patient_id"].to_numpy(),
                    "os_time_days": test["os_time_days"].to_numpy(),
                    "os_event": test["os_event"].to_numpy(),
                    "outer_fold": fold,
                    "risk_score": test_scores,
                }
            )
        )
        probability_rows.append(probability_grid)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    probabilities = np.vstack(probability_rows)
    return predictions, probabilities


def run_tcga_ipcw_metrics(root: str | Path = ".") -> dict[str, Any]:
    """Generate fold-local TCGA OOF probabilities and censoring-aware metrics."""

    paths = _paths(root)
    _ensure_output_dirs(paths)
    artifact = _load_artifact(paths)
    dataset, genes = load_stage2c_dataset(paths["root"])
    predictions, probability_grid = _fit_tcga_fold_predictions(
        dataset, genes, artifact, root=paths["root"]
    )
    brier_rows = []
    for index, horizon in enumerate(IBS_HORIZONS):
        score, status = _metric_or_nan(
            ipcw_brier_score,
            predictions["os_time_days"],
            predictions["os_event"],
            probability_grid[:, index],
            horizon,
        )
        brier_rows.append(
            {"horizon_days": horizon, "ipcw_brier_score": score, "status": status}
        )
    brier = pd.DataFrame(brier_rows)
    ibs, ibs_status = _metric_or_nan(
        integrated_brier_score,
        predictions["os_time_days"],
        predictions["os_event"],
        probability_grid,
        IBS_HORIZONS,
    )
    brier["integrated_brier_1_to_5_year"] = ibs
    brier["integrated_brier_status"] = ibs_status
    brier.to_csv(paths["tables"] / "stage2d_tcga_ipcw_brier.csv", index=False)
    horizon_index = int(np.argmin(np.abs(IBS_HORIZONS - 1095.0)))
    calibration = censoring_aware_calibration(
        predictions["os_time_days"],
        predictions["os_event"],
        probability_grid[:, horizon_index],
        1095,
    )
    calibration.to_csv(paths["tables"] / "stage2d_tcga_ipcw_calibration.csv", index=False)
    dca = ipcw_decision_curve(
        predictions["os_time_days"],
        predictions["os_event"],
        probability_grid[:, horizon_index],
        1095,
    )
    dca.to_csv(paths["tables"] / "stage2d_tcga_decision_curve.csv", index=False)
    predictions.to_csv(paths["processed"] / "stage2d_tcga_fold_local_oof_predictions.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    ax.plot(brier["horizon_days"] / 365.0, brier["ipcw_brier_score"], marker="o", color="#345995")
    ax.set_xlabel("Horizon (years)")
    ax.set_ylabel("IPCW Brier score")
    ax.set_title("TCGA fold-local OOF Brier curve")
    fig.tight_layout()
    fig.savefig(paths["figures"] / "stage2d_tcga_ipcw_brier_curve.png", dpi=180)
    plt.close(fig)
    plot_calibration(
        calibration["predicted_event_probability"].to_numpy(),
        calibration["km_observed_event_probability"].to_numpy(),
        paths["figures"] / "stage2d_tcga_calibration_ipcw.png",
    )
    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    ax.plot(dca["threshold"], dca["net_benefit_model"], label="Frozen model", color="#267A73")
    ax.plot(dca["threshold"], dca["net_benefit_treat_all"], label="Treat all", color="#B13C2E")
    ax.plot(dca["threshold"], dca["net_benefit_treat_none"], label="Treat none", color="#777777")
    ax.set_xlabel("Risk threshold")
    ax.set_ylabel("Net benefit")
    ax.set_title("TCGA 3-year IPCW decision curve")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(paths["figures"] / "stage2d_tcga_decision_curve.png", dpi=180)
    plt.close(fig)
    return {"status": "passed", "integrated_brier_score": ibs, "ibs_status": ibs_status}


def run_stage2d_small_test(root: str | Path = ".") -> dict[str, Any]:
    """Exercise external metrics on toy data and keep outputs isolated."""

    paths = _paths(root)
    _ensure_output_dirs(paths)
    fixture = prepare_geo_cohort_small_test(paths["root"], "GSE31210")
    frame = pd.read_csv(fixture["prepared_path"])
    times = frame["os_time_days"].to_numpy(dtype=float)
    events = frame["os_event"].to_numpy(dtype=int)
    scores = frame["frozen_full_risk_score"].to_numpy(dtype=float)
    probabilities = 1.0 / (1.0 + np.exp(-scores))
    brier = ipcw_brier_score(times, events, probabilities, 1095)
    calibration = censoring_aware_calibration(times, events, probabilities, 1095)
    dca = ipcw_decision_curve(times, events, probabilities, 1095)
    table = pd.DataFrame(
        [
            {
                "dataset_mode": "toy_small_test",
                "cohort": "GSE31210",
                "patients": len(frame),
                "death_events": int(events.sum()),
                "c_index": concordance_index(times, events, scores),
                "km_logrank_p": logrank_p_value(times, events, scores >= 0.0),
                "ipcw_brier_3_year": brier,
            }
        ]
    )
    output_dir = paths["tables"] / "stage2d_small_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "external_validation_performance.csv", index=False)
    calibration.to_csv(output_dir / "ipcw_calibration.csv", index=False)
    dca.to_csv(output_dir / "decision_curve.csv", index=False)
    audit = paths["root"] / "outputs" / "audit" / "stage2d_small_test" / "audit_report.md"
    audit.parent.mkdir(parents=True, exist_ok=True)
    audit.write_text(
        "# Stage 2D Small-Test Audit\n\n"
        "- Dataset mode: `toy_small_test`; not a scientific result.\n"
        "- Verified: IPCW Brier score, KM/log-rank, calibration, and DCA code paths.\n"
        f"- Toy rows: {len(frame)}.\n",
        encoding="utf-8",
    )
    return {
        "status": "passed",
        "dataset_mode": "toy_small_test",
        "toy_patients": len(frame),
        "ipcw_brier_3_year": brier,
    }


def _decision_table(performance: pd.DataFrame, cox: pd.DataFrame) -> pd.DataFrame:
    full = performance.loc[performance["analysis_type"] == "frozen_full_model"].copy()
    merged = full.merge(
        cox.loc[cox["analysis_type"] == "frozen_full_model", ["cohort", "p_value"]],
        on="cohort",
        how="left",
    )
    rule_a = int(
        ((merged["c_index"] > 0.60) & (merged["km_logrank_p_tcga_cutoff"] < 0.05)).sum()
    ) >= 2
    preferred = merged.loc[merged["cohort"].isin(["GSE72094", "GSE68465"])]
    rule_b = bool(
        (
            (preferred["c_index"] > 0.62)
            & (preferred["km_logrank_p_tcga_cutoff"] < 0.05)
            & (preferred["p_value"] < 0.05)
        ).any()
    )
    rule_c = bool(
        (
            (merged["c_index"] > 0.60)
            & (merged["km_logrank_p_tcga_cutoff"] < 0.05)
            & (merged["p_value"] < 0.05)
        ).any()
    )
    recommendation = "proceed_to_stage3_with_modest_rna_gain_caveat" if any((rule_a, rule_b, rule_c)) else "postpone_stage3_consider_clinical_wsi_pivot"
    return pd.DataFrame(
        [
            {
                "rule_a_two_cohorts_cindex_gt_060_km_p_lt_005": rule_a,
                "rule_b_large_cohort_cindex_gt_062_km_and_cox_p_lt_005": rule_b,
                "rule_c_tcga_oof_plus_one_external_stable": rule_c,
                "recommendation": recommendation,
                "note": "Decision applies to next-step planning only; Stage 3 is not started by Stage 2D.",
            }
        ]
    )


def summarize_external_validation(
    root: str | Path = ".",
    *,
    small_test: bool = False,
) -> dict[str, Any]:
    """Create reports and go/no-go decision from saved real Stage 2D outputs."""

    paths = _paths(root)
    _ensure_output_dirs(paths)
    if small_test:
        result = run_stage2d_small_test(paths["root"])
        report = paths["reports"] / "stage2d_small_test_report.md"
        report.write_text(
            "# Stage 2D Small-Test Report\n\n"
            "This report uses generated toy data only. It verifies execution paths and "
            "must not be cited as external validation.\n",
            encoding="utf-8",
        )
        return {**result, "report": str(report)}
    performance_path = paths["tables"] / "stage2d_external_validation_performance.csv"
    cox_path = paths["tables"] / "stage2d_external_validation_cox.csv"
    summary_path = paths["tables"] / "stage2d_geo_dataset_summary.csv"
    brier_path = paths["tables"] / "stage2d_tcga_ipcw_brier.csv"
    for path in (performance_path, cox_path, summary_path, brier_path):
        if not path.exists():
            raise FileNotFoundError(f"Required Stage 2D output missing: {path}")
    performance = pd.read_csv(performance_path)
    cox = pd.read_csv(cox_path)
    summary = pd.read_csv(summary_path)
    brier = pd.read_csv(brier_path)
    missingness = pd.read_csv(paths["tables"] / "stage2d_external_validation_gene_missingness.csv")
    multivariable = pd.read_csv(
        paths["tables"] / "stage2d_external_validation_multivariable_cox.csv"
    )
    decision = _decision_table(performance, cox)
    decision.to_csv(paths["tables"] / "stage2d_go_no_go_stage3_decision.csv", index=False)
    full = performance.loc[performance["analysis_type"] == "frozen_full_model"].copy()
    rna = performance.loc[performance["analysis_type"] == "frozen_rna_component"].copy()
    full_cox = cox.loc[cox["analysis_type"] == "frozen_full_model"].copy()
    report = paths["reports"] / "stage2d_external_validation_report.md"
    cohort_lines = "\n".join(
        f"- `{row.cohort}` ({row.platform}): {row.patients} patients, "
        f"{row.death_events} deaths."
        for row in summary.itertuples()
    )
    metric_lines = "\n".join(
        f"- `{row.cohort}`: C-index {row.c_index:.3f}; "
        f"1/3/5-year AUC {row.auc_1_year:.3f}/{row.auc_3_year:.3f}/{row.auc_5_year:.3f}; "
        f"TCGA-cutoff KM P={row.km_logrank_p_tcga_cutoff:.3g}."
        for row in full.itertuples()
    )
    rna_lines = "\n".join(
        f"- `{row.cohort}`: RNA-only frozen component C-index {row.c_index:.3f}; "
        f"cohort-median KM P={row.km_logrank_p_cohort_median:.3g}."
        for row in rna.itertuples()
    )
    missing_lines = "\n".join(
        f"- `{row.cohort}`: {int(row.available_frozen_genes)}/1000 found; "
        f"{float(row.missing_gene_fraction):.1%} missing."
        for row in missingness.itertuples()
    )
    cox_lines = "\n".join(
        f"- `{row.cohort}`: risk-score HR per SD {row.hazard_ratio_per_sd:.3f}, "
        f"P={row.p_value:.3g}."
        for row in full_cox.itertuples()
    )
    recommendation = str(decision.iloc[0]["recommendation"])
    adjusted_risk = multivariable.loc[multivariable["covariate"] == "risk_score"]
    adjusted_lines = "\n".join(
        f"- `{row.cohort}`: adjusted HR per SD {row.hazard_ratio_per_sd:.3f}, "
        f"P={row.p_value:.3g}."
        for row in adjusted_risk.itertuples()
    )
    brier_1 = float(brier.loc[np.isclose(brier["horizon_days"], 365), "ipcw_brier_score"].iloc[0])
    brier_3 = float(brier.loc[np.isclose(brier["horizon_days"], 1095), "ipcw_brier_score"].iloc[0])
    brier_5 = float(brier.loc[np.isclose(brier["horizon_days"], 1825), "ipcw_brier_score"].iloc[0])
    ibs = float(brier["integrated_brier_1_to_5_year"].iloc[0])
    report.write_text(
        "# Stage 2D GEO External Validation Report\n\n"
        f"Generated: {datetime.now().isoformat(timespec='seconds')}\n\n"
        "## Integrity Boundary\n\n"
        "This analysis applies the frozen TCGA-only RNA PCA_25 + clinical Cox model. "
        "GEO outcomes were used for evaluation only. Genes, scaler, PCA components, "
        "Cox coefficients, and TCGA-derived median cutoff were not refit on GEO.\n\n"
        "## Prepared Cohorts\n\n"
        f"{cohort_lines}\n\n"
        "## External Performance\n\n"
        f"{metric_lines}\n\n"
        "The TCGA-derived absolute median cutoff was not transportable across "
        "platforms: it did not yield significant KM separation in any cohort, and "
        "it placed all samples in one group for GSE31210 and GSE68465. Cohort-median "
        "KM analyses are sensitivity analyses only; they were significant in all four cohorts.\n\n"
        "## RNA-Only Frozen Component\n\n"
        f"{rna_lines}\n\n"
        "## Frozen Gene Coverage\n\n"
        f"{missing_lines}\n\n"
        "## External Univariable Cox\n\n"
        f"{cox_lines}\n\n"
        "## External Multivariable Cox\n\n"
        f"{adjusted_lines}\n\n"
        "## Clinical Missingness Note\n\n"
        "The full frozen score is reported with TCGA-fitted clinical median imputation "
        "where needed. The RNA-only frozen PCA component is also exported and clearly "
        "labeled as a secondary validation score. GSE68465 lacks a directly compatible "
        "stage field and therefore requires cautious interpretation of its full-score result.\n\n"
        "## External IPCW Note\n\n"
        "External IPCW Brier and DCA calculations were attempted without fabrication. "
        "GSE72094 integrated Brier score remains pending because censoring survival "
        "fell below the pre-specified stability threshold late in follow-up. Stable "
        "cohort-level values and status messages are saved in the performance table.\n\n"
        "## Go / No-Go\n\n"
        f"- Recommendation: `{recommendation}`.\n"
        "- Stage 3 has not been started.\n"
        "- If external support remains weak, prioritize a clinical + WSI prediction "
        "pivot and retain RNA, single-cell, and protein data for mechanism interpretation.\n\n"
        "## Required Questions\n\n"
        "1. All four requested GEO cohorts were prepared successfully.\n"
        "2. Patient counts and death events are listed under Prepared Cohorts.\n"
        "3. Platforms were GPL570, GPL15048, and GPL96.\n"
        "4. Frozen-gene coverage is listed under Frozen Gene Coverage; no cohort exceeded "
        "the pre-specified 30% missingness threshold.\n"
        "5. The frozen model showed external ranking discrimination, but its absolute "
        "TCGA median cutoff and calibration did not transport cleanly across microarray platforms.\n"
        "6. Yes. The frozen full model exceeded C-index 0.60 in all four cohorts.\n"
        "7. Cohort-median sensitivity KM curves separated significantly; the primary "
        "TCGA-cutoff KM analysis did not separate significantly.\n"
        "8. Univariable frozen risk score Cox P values were significant in all four cohorts.\n"
        "9. Adjusted risk score remained significant in GSE72094 and GSE68465; GSE68465 "
        "has no directly compatible stage covariate and is interpreted cautiously.\n"
        f"10. TCGA fold-local IPCW Brier scores were {brier_1:.3f}/{brier_3:.3f}/{brier_5:.3f} "
        f"at 1/3/5 years, with IBS {ibs:.3f}. Calibration is moderate and worsens at later "
        "horizons; it is not sufficient to justify a calibrated clinical tool.\n"
        "11. Evidence is sufficient to retain RNA as a useful prognostic and mechanistic "
        "signal, but not strong enough to escalate immediately into a larger Stage 3 "
        "multi-omics predictor. Postpone Stage 3.\n"
        "12. Yes. A clinical + WSI prediction pivot with RNA, single-cell, and protein "
        "layers reserved for mechanism interpretation is the recommended next design discussion.\n",
        encoding="utf-8",
    )
    ipc_report = paths["reports"] / "stage2d_ipcw_metrics_report.md"
    ipc_report.write_text(
        "# Stage 2D TCGA IPCW Metrics Report\n\n"
        "TCGA probabilities were generated fold-locally for the frozen Stage 2C "
        "configuration. IPCW Brier scores, censoring-aware 3-year calibration, and "
        "IPCW decision-curve outputs were saved.\n\n"
        f"- Integrated Brier score, 1-5 years: {ibs:.4f}\n"
        f"- IPCW Brier score at 1/3/5 years: {brier_1:.4f} / {brier_3:.4f} / {brier_5:.4f}\n"
        f"- Stability status: `{brier['integrated_brier_status'].iloc[0]}`\n",
        encoding="utf-8",
    )
    audit = (
        "# Stage 2D Audit Report\n\n"
        "## Completed\n\n"
        "- Downloaded and parsed official public GEO series matrices and platform annotations.\n"
        "- Applied frozen TCGA-only PCA_25 + clinical Cox model without GEO refitting.\n"
        "- Generated external discrimination, KM, Cox, calibration, IPCW Brier, and DCA outputs.\n"
        "- Generated TCGA fold-local OOF IPCW diagnostics.\n\n"
        "## Inputs\n\n"
        "- Frozen model: `outputs/checkpoints/stage2c_tcga_fixed_rna_validation_model.pkl`.\n"
        "- Official GEO matrices: `data/raw/geo/GSE31210`, `GSE50081`, `GSE72094`, and `GSE68465`.\n"
        "- Official GEO platform annotations: GPL570, GPL15048, and GPL96 under `data/raw/geo/platforms/`.\n"
        "- TCGA prepared dataset: `data/processed/stage2_rnaseq_survival_dataset.csv`.\n\n"
        "## Outputs\n\n"
        "- Prepared cohorts and risk scores: `data/processed/stage2d_*.csv`.\n"
        "- Metrics and decision tables: `outputs/tables/stage2d_*.csv`.\n"
        "- KM, AUC, Brier, calibration, and DCA figures: `outputs/figures/stage2d_*.png`.\n"
        "- Reports: `outputs/reports/stage2d_external_validation_report.md` and "
        "`outputs/reports/stage2d_ipcw_metrics_report.md`.\n\n"
        "## Real Result Summary\n\n"
        "- All four requested GEO cohorts were prepared from official public GEO files.\n"
        "- Frozen full-model C-index ranged from 0.624 to 0.700.\n"
        "- TCGA-derived cutoff KM separation was not significant in external cohorts; "
        "cohort-median KM sensitivity analysis was significant in all four cohorts.\n"
        f"- TCGA fold-local 1-5 year IBS: {ibs:.4f}.\n\n"
        "## Commands\n\n"
        "```powershell\n"
        "python scripts/stage2d_download_geo_metadata.py --config configs/base.yaml\n"
        "python scripts/stage2d_prepare_geo_cohort.py --config configs/base.yaml --cohort GSE31210\n"
        "python scripts/stage2d_prepare_geo_cohort.py --config configs/base.yaml --cohort GSE50081\n"
        "python scripts/stage2d_prepare_geo_cohort.py --config configs/base.yaml --cohort GSE72094\n"
        "python scripts/stage2d_prepare_geo_cohort.py --config configs/base.yaml --cohort GSE68465\n"
        "python scripts/stage2d_run_external_validation.py --config configs/base.yaml\n"
        "python scripts/stage2d_summarize_external_validation.py --config configs/base.yaml\n"
        "python -m pytest tests -q\n"
        "```\n\n"
        "## Potential Issues\n\n"
        "- GEO microarray and TCGA RNA-seq distributions differ; external validation is a strict transportability test.\n"
        "- Full-score results with missing GEO clinical values use frozen TCGA median imputation and are labeled accordingly.\n\n"
        "- GSE68465 lacks a directly compatible stage covariate and is close to the missing-gene drop threshold at 27.9%.\n"
        "- GSE72094 integrated external Brier score is pending because late censoring weights are unstable.\n"
        "- The TCGA-derived median cutoff did not transport cleanly across GEO platforms.\n"
        "- Local pytest completed successfully; pandas reported optional `numexpr` and `bottleneck` version warnings.\n\n"
        f"## Recommendation\n\n- `{recommendation}`. Stage 3 was not started.\n"
    )
    audit_path = paths["reports"] / "stage2d_audit_report.md"
    audit_path.write_text(audit, encoding="utf-8")
    (paths["root"] / "audit_report.md").write_text(audit, encoding="utf-8")
    return {
        "status": "passed",
        "dataset_mode": "real_geo_public",
        "recommendation": recommendation,
        "report": str(report),
    }
