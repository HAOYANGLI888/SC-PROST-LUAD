"""Leakage-safe nested cross-validation for Stage 2C clinical + RNA-seq models."""

from __future__ import annotations

import json
import inspect
from dataclasses import asdict, dataclass
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
from data.survival_dataset import SEEDS, Stage2Paths, make_toy_dataset
from evaluation.survival_metrics import (
    calibration_points,
    concordance_index,
    logrank_p_value,
    predict_event_probability,
    time_dependent_auc,
)
from features.rnaseq_feature_spaces import (
    CORE_FEATURE_SPACES,
    OPTIONAL_FEATURE_SPACES,
    RNAFeatureSpace,
    feature_space_inventory,
)
from models.cox_baseline import TorchCoxElasticNet
from models.deepsurv import DeepSurvEstimator


HORIZONS = (365, 1095, 1825)
METADATA_COLUMNS = {
    "dataset_mode",
    "patient_id",
    "os_time_days",
    "os_event",
    "age",
    "male",
    "stage_numeric",
    "gender",
    "stage_raw",
}


class Stage2CNestedCVError(RuntimeError):
    """Raised when nested CV cannot be completed safely."""


@dataclass(frozen=True)
class ModelConfiguration:
    """One Stage 2C model and RNA-representation pairing."""

    config_id: str
    model_name: str
    feature_space: str
    input_scope: str
    estimator: str


def model_configurations(root: str | Path = ".") -> list[ModelConfiguration]:
    """Return strict Stage 2C configurations, including ready pathways."""

    configurations = [
        ModelConfiguration("clinical_cox", "Clinical Cox", "clinical_only", "clinical", "cox"),
        ModelConfiguration("clinical_elasticnet_cox", "Clinical ElasticNet Cox", "clinical_only", "clinical", "cox"),
        ModelConfiguration("rna_elasticnet_raw_top500", "RNA ElasticNet Cox", "raw_high_variance_genes_top500", "rna", "cox"),
        ModelConfiguration("rna_clinical_elasticnet_raw_top500", "RNA + clinical ElasticNet Cox", "raw_high_variance_genes_top500", "combined", "cox"),
        ModelConfiguration("rna_clinical_elasticnet_raw_top1000", "RNA + clinical ElasticNet Cox", "raw_high_variance_genes_top1000", "combined", "cox"),
        ModelConfiguration("rna_clinical_elasticnet_raw_top3000", "RNA + clinical ElasticNet Cox", "raw_high_variance_genes_top3000", "combined", "cox"),
        ModelConfiguration("rna_pca25_clinical_cox", "RNA PCA + clinical Cox", "PCA_25", "combined", "cox"),
        ModelConfiguration("rna_pca50_clinical_cox", "RNA PCA + clinical Cox", "PCA_50", "combined", "cox"),
        ModelConfiguration("rna_pca100_clinical_cox", "RNA PCA + clinical Cox", "PCA_100", "combined", "cox"),
        ModelConfiguration("rna_clinical_elasticnet_selected", "RNA + clinical ElasticNet Cox", "ElasticNet_selected_genes", "combined", "cox"),
        ModelConfiguration(
            "rna_clinical_univariate_cox_selected",
            "RNA + clinical ElasticNet Cox",
            "univariate_cox_selected_genes_inside_inner_cv",
            "combined",
            "cox",
        ),
        ModelConfiguration("deepsurv_rna_raw_top500", "DeepSurv RNA only", "raw_high_variance_genes_top500", "rna", "deepsurv"),
        ModelConfiguration("deepsurv_rna_clinical_raw_top500", "DeepSurv RNA + clinical", "raw_high_variance_genes_top500", "combined", "deepsurv"),
    ]
    inventory = feature_space_inventory(root).set_index("feature_space")
    if bool(inventory.loc["pathway_Hallmark_scores", "available"]):
        configurations.append(
            ModelConfiguration("rna_hallmark_clinical_cox", "RNA pathway + clinical Cox", "pathway_Hallmark_scores", "combined", "cox")
        )
    if bool(inventory.loc["pathway_Reactome_scores", "available"]):
        configurations.append(
            ModelConfiguration("rna_reactome_clinical_cox", "RNA pathway + clinical Cox", "pathway_Reactome_scores", "combined", "cox")
        )
    return configurations


def _candidate_genes(dataset: pd.DataFrame, paths: Stage2Paths, *, small_test: bool) -> list[str]:
    if small_test:
        return [column for column in dataset.columns if column.startswith("GENE_")]
    if not paths.feature_list.exists():
        raise Stage2CNestedCVError(f"Stage 2 feature list not found: {paths.feature_list}")
    table = pd.read_csv(paths.feature_list)
    if "feature" not in table:
        raise Stage2CNestedCVError("Stage 2 feature list has no 'feature' column.")
    genes = [str(gene) for gene in table["feature"] if str(gene) in dataset.columns]
    if not genes:
        raise Stage2CNestedCVError("No RNA candidate genes overlap the prepared Stage 2 dataset.")
    return genes


def load_stage2c_dataset(
    root: str | Path = ".",
    *,
    small_test: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Load real Stage 2B data or an isolated deterministic toy cohort."""

    paths = Stage2Paths.from_root(root)
    if small_test:
        dataset = make_toy_dataset(n_patients=54, n_genes=80, seed=2026)
    else:
        if not paths.dataset.exists():
            raise Stage2CNestedCVError(
                f"Real prepared Stage 2 dataset not found: {paths.dataset}. Run Stage 2B preparation first."
            )
        dataset = pd.read_csv(paths.dataset)
        if dataset.empty or str(dataset["dataset_mode"].iloc[0]) != "real_tcga_luad":
            raise Stage2CNestedCVError("Refusing to run formal Stage 2C with toy or empty Stage 2 data.")
    required = {"patient_id", "os_time_days", "os_event", *CLINICAL_COLUMNS}
    missing = sorted(required - set(dataset.columns))
    if missing:
        raise Stage2CNestedCVError(f"Prepared dataset is missing columns: {missing}")
    if dataset["patient_id"].duplicated().any():
        raise Stage2CNestedCVError("Prepared Stage 2 dataset contains duplicate patient IDs.")
    dataset = dataset.reset_index(drop=True)
    genes = _candidate_genes(dataset, paths, small_test=small_test)
    return dataset, genes


def survival_strata(dataset: pd.DataFrame, *, n_splits: int) -> np.ndarray:
    """Stratify by event status and coarse observed-time quantiles when feasible."""

    status = dataset["os_event"].astype(int).astype(str)
    try:
        quantile = pd.qcut(dataset["os_time_days"], q=3, labels=False, duplicates="drop").astype(str)
        labels = status + "_q" + quantile
        if labels.value_counts().min() >= n_splits:
            return labels.to_numpy()
    except (TypeError, ValueError):
        pass
    if status.value_counts().min() < n_splits:
        raise Stage2CNestedCVError("Not enough events or censored samples for stratified folds.")
    return status.to_numpy()


def _parameter_grid(configuration: ModelConfiguration, *, small_test: bool) -> list[dict[str, Any]]:
    if configuration.estimator == "deepsurv":
        return [
            {"weight_decay": 1e-4, "dropout": 0.15},
            *([] if small_test else [{"weight_decay": 1e-3, "dropout": 0.25}]),
        ]
    if configuration.config_id == "clinical_cox" or configuration.model_name == "RNA PCA + clinical Cox":
        return [{"alpha": 0.0, "l1_ratio": 0.0}]
    return [
        {"alpha": 0.001, "l1_ratio": 0.70},
        *([] if small_test else [{"alpha": 0.01, "l1_ratio": 0.80}]),
    ]


def _fit_estimator(
    configuration: ModelConfiguration,
    parameters: dict[str, Any],
    x: np.ndarray,
    durations: np.ndarray,
    events: np.ndarray,
    *,
    seed: int,
    small_test: bool,
    inner: bool,
) -> TorchCoxElasticNet | DeepSurvEstimator:
    if configuration.estimator == "cox":
        return TorchCoxElasticNet(
            alpha=float(parameters["alpha"]),
            l1_ratio=float(parameters["l1_ratio"]),
            epochs=7 if small_test else (36 if inner else 65),
            learning_rate=0.03,
            seed=seed,
        ).fit(x, durations, events)
    return DeepSurvEstimator(
        hidden_dims=(24, 12) if small_test else (48, 24),
        dropout=float(parameters["dropout"]),
        weight_decay=float(parameters["weight_decay"]),
        epochs=6 if small_test else (16 if inner else 42),
        learning_rate=0.004,
        seed=seed,
    ).fit(x, durations, events)


def _stack(clinical: np.ndarray, rna: np.ndarray) -> np.ndarray:
    return np.concatenate([clinical, rna], axis=1)


def _model_input(configuration: ModelConfiguration, clinical: np.ndarray, rna: np.ndarray) -> np.ndarray:
    if configuration.input_scope == "clinical":
        return clinical
    if configuration.input_scope == "rna":
        return rna
    return _stack(clinical, rna)


@dataclass
class FoldArrays:
    """Preprocessed arrays fit exclusively on one fold's training IDs."""

    clinical_train: np.ndarray
    clinical_eval: np.ndarray
    rna_train: np.ndarray
    rna_eval: np.ndarray
    feature_count: int
    fit_patient_ids: tuple[str, ...]


def _fit_fold_arrays(
    dataset: pd.DataFrame,
    genes: list[str],
    train_indices: np.ndarray,
    eval_indices: np.ndarray,
    *,
    feature_space: str,
    root: Path,
    seed: int,
    small_test: bool,
) -> FoldArrays:
    train = dataset.iloc[train_indices]
    evaluation = dataset.iloc[eval_indices]
    train_ids = tuple(train["patient_id"].astype(str))
    clinical = ClinicalTrainPreprocessor().fit(train[list(CLINICAL_COLUMNS)], patient_ids=train_ids)
    clinical_train = clinical.transform(train[list(CLINICAL_COLUMNS)]).to_numpy()
    clinical_eval = clinical.transform(evaluation[list(CLINICAL_COLUMNS)]).to_numpy()
    feature = RNAFeatureSpace(feature_space, root=root, seed=seed, small_test=small_test).fit(
        train[genes],
        train["os_time_days"].to_numpy(dtype=float),
        train["os_event"].to_numpy(dtype=int),
        patient_ids=train_ids,
    )
    rna_train = feature.transform(train[genes]).to_numpy(dtype=np.float32)
    rna_eval = feature.transform(evaluation[genes]).to_numpy(dtype=np.float32)
    if set(feature.fit_patient_ids_) != set(train_ids):
        raise Stage2CNestedCVError("RNA feature transformer fit IDs differ from the current training fold.")
    return FoldArrays(
        clinical_train=clinical_train,
        clinical_eval=clinical_eval,
        rna_train=rna_train,
        rna_eval=rna_eval,
        feature_count=feature.feature_count,
        fit_patient_ids=train_ids,
    )


def _select_parameters(
    configuration: ModelConfiguration,
    dataset: pd.DataFrame,
    genes: list[str],
    outer_train_indices: np.ndarray,
    *,
    root: Path,
    seed: int,
    inner_splits: int,
    small_test: bool,
    feature_cache: dict[str, list[tuple[np.ndarray, np.ndarray, FoldArrays]]] | None = None,
) -> tuple[dict[str, Any], float]:
    outer_train = dataset.iloc[outer_train_indices].reset_index(drop=True)
    strata = survival_strata(outer_train, n_splits=inner_splits)
    splitter = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
    parameter_rows: list[tuple[dict[str, Any], float]] = []
    cache = {} if feature_cache is None else feature_cache
    fold_arrays = cache.get(configuration.feature_space)
    if fold_arrays is None:
        fold_arrays = []
        for inner_train_local, inner_val_local in splitter.split(outer_train, strata):
            arrays = _fit_fold_arrays(
                outer_train,
                genes,
                inner_train_local,
                inner_val_local,
                feature_space=configuration.feature_space,
                root=root,
                seed=seed,
                small_test=small_test,
            )
            fold_arrays.append((inner_train_local, inner_val_local, arrays))
        cache[configuration.feature_space] = fold_arrays
    for parameters in _parameter_grid(configuration, small_test=small_test):
        scores: list[float] = []
        for fold_index, (inner_train_local, inner_val_local, arrays) in enumerate(fold_arrays):
            train = outer_train.iloc[inner_train_local]
            validation = outer_train.iloc[inner_val_local]
            model = _fit_estimator(
                configuration,
                parameters,
                _model_input(configuration, arrays.clinical_train, arrays.rna_train),
                train["os_time_days"].to_numpy(dtype=float),
                train["os_event"].to_numpy(dtype=int),
                seed=seed + fold_index,
                small_test=small_test,
                inner=True,
            )
            risk = model.predict_risk(_model_input(configuration, arrays.clinical_eval, arrays.rna_eval))
            scores.append(concordance_index(validation["os_time_days"], validation["os_event"], risk))
        parameter_rows.append((parameters, float(np.nanmean(scores))))
    return max(parameter_rows, key=lambda item: item[1])


def _calibration_mae(
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_risk: np.ndarray,
    test_risk: np.ndarray,
) -> tuple[float, np.ndarray]:
    probabilities = predict_event_probability(
        train["os_time_days"],
        train["os_event"],
        train_risk,
        test_risk,
        1095,
    )
    predicted, observed = calibration_points(test["os_time_days"], test["os_event"], probabilities, 1095)
    mae = float(np.mean(np.abs(predicted - observed))) if len(predicted) else float("nan")
    return mae, probabilities


def _plot_nested_cv(performance: pd.DataFrame, figures: Path) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    order = (
        performance.groupby("config_id")["outer_test_c_index"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    labels = [item.replace("_", "\n") for item in order]
    values = [performance.loc[performance["config_id"] == item, "outer_test_c_index"].to_numpy() for item in order]
    fig, ax = plt.subplots(figsize=(max(9.0, len(order) * 0.75), 5.4))
    label_argument = "tick_labels" if "tick_labels" in inspect.signature(ax.boxplot).parameters else "labels"
    ax.boxplot(values, showfliers=False, **{label_argument: labels})
    ax.axhline(0.5, color="#777777", linestyle="--", linewidth=1)
    ax.set_ylabel("Outer-fold C-index")
    ax.set_title("Stage 2C nested cross-validation")
    ax.tick_params(axis="x", labelsize=7)
    fig.tight_layout()
    fig.savefig(figures / "stage2c_nested_cv_cindex_boxplot.png", dpi=180)
    plt.close(fig)

    comparison = performance.groupby("feature_space")["outer_test_c_index"].agg(["mean", "std"]).sort_values("mean")
    fig, ax = plt.subplots(figsize=(8.2, max(4.6, len(comparison) * 0.42)))
    ax.barh(comparison.index, comparison["mean"], xerr=comparison["std"].fillna(0.0), color="#267A73", alpha=0.9)
    ax.axvline(0.5, color="#777777", linestyle="--", linewidth=1)
    ax.set_xlabel("Outer-fold C-index, mean +/- SD")
    ax.set_title("RNA feature-space comparison")
    fig.tight_layout()
    fig.savefig(figures / "stage2c_feature_space_performance.png", dpi=180)
    plt.close(fig)


def _comparison_table(root: Path, performance: pd.DataFrame) -> pd.DataFrame:
    summary = (
        performance.groupby(["config_id", "model_name", "feature_space", "input_scope"], as_index=False)
        .agg(
            outer_fold_count=("outer_test_c_index", "size"),
            outer_c_index_mean=("outer_test_c_index", "mean"),
            outer_c_index_std=("outer_test_c_index", "std"),
            auc_1_year_mean=("auc_1_year", "mean"),
            auc_3_year_mean=("auc_3_year", "mean"),
            auc_5_year_mean=("auc_5_year", "mean"),
            km_logrank_p_median=("km_logrank_p_value", "median"),
            train_minus_test_gap_mean=("train_minus_test_gap", "mean"),
        )
    )
    summary["available"] = True
    summary["status"] = "evaluated"
    inventory = feature_space_inventory(root)
    represented = set(summary["feature_space"])
    missing_rows = inventory.loc[~inventory["feature_space"].isin(represented)].copy()
    if not missing_rows.empty:
        missing_rows = missing_rows.rename(columns={"reason": "status"})
        missing_rows["config_id"] = ""
        missing_rows["model_name"] = ""
        missing_rows["input_scope"] = ""
        for column in (
            "outer_fold_count",
            "outer_c_index_mean",
            "outer_c_index_std",
            "auc_1_year_mean",
            "auc_3_year_mean",
            "auc_5_year_mean",
            "km_logrank_p_median",
            "train_minus_test_gap_mean",
        ):
            missing_rows[column] = np.nan
        summary = pd.concat([summary, missing_rows[summary.columns]], ignore_index=True)
    return summary.sort_values(["available", "outer_c_index_mean"], ascending=[False, False], na_position="last")


def run_nested_cv(
    root: str | Path = ".",
    *,
    seeds: Iterable[int] = SEEDS,
    small_test: bool = False,
) -> dict[str, Any]:
    """Run Stage 2C nested CV and save outer-fold and OOF predictions."""

    project_root = Path(root).resolve()
    tables = project_root / "outputs" / "tables"
    figures = project_root / "outputs" / "figures"
    logs = project_root / "outputs" / "logs"
    for directory in (tables, figures, logs):
        directory.mkdir(parents=True, exist_ok=True)
    dataset, genes = load_stage2c_dataset(project_root, small_test=small_test)
    outer_splits = 2 if small_test else 5
    inner_splits = 2 if small_test else 3
    configurations = model_configurations(project_root)
    requested_seeds = tuple(int(seed) for seed in seeds)
    if not requested_seeds:
        raise Stage2CNestedCVError("At least one seed is required.")
    performance_rows: list[dict[str, Any]] = []
    oof_rows: list[pd.DataFrame] = []
    leakage_checks: list[dict[str, Any]] = []
    for seed in requested_seeds:
        strata = survival_strata(dataset, n_splits=outer_splits)
        splitter = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=seed)
        for outer_fold, (train_indices, test_indices) in enumerate(splitter.split(dataset, strata), start=1):
            train = dataset.iloc[train_indices]
            test = dataset.iloc[test_indices]
            inner_feature_cache: dict[str, list[tuple[np.ndarray, np.ndarray, FoldArrays]]] = {}
            outer_feature_cache: dict[str, FoldArrays] = {}
            for configuration in configurations:
                best_parameters, inner_c_index = _select_parameters(
                    configuration,
                    dataset,
                    genes,
                    train_indices,
                    root=project_root,
                    seed=seed + outer_fold,
                    inner_splits=inner_splits,
                    small_test=small_test,
                    feature_cache=inner_feature_cache,
                )
                arrays = outer_feature_cache.get(configuration.feature_space)
                if arrays is None:
                    arrays = _fit_fold_arrays(
                        dataset,
                        genes,
                        train_indices,
                        test_indices,
                        feature_space=configuration.feature_space,
                        root=project_root,
                        seed=seed + outer_fold,
                        small_test=small_test,
                    )
                    outer_feature_cache[configuration.feature_space] = arrays
                overlap_count = len(set(arrays.fit_patient_ids) & set(test["patient_id"].astype(str)))
                if overlap_count:
                    raise Stage2CNestedCVError("Outer-test patient IDs leaked into preprocessing fit IDs.")
                leakage_checks.append(
                    {
                        "seed": seed,
                        "outer_fold": outer_fold,
                        "config_id": configuration.config_id,
                        "fit_patient_count": len(arrays.fit_patient_ids),
                        "outer_test_patient_count": len(test),
                        "fit_test_overlap_count": overlap_count,
                    }
                )
                x_train = _model_input(configuration, arrays.clinical_train, arrays.rna_train)
                x_test = _model_input(configuration, arrays.clinical_eval, arrays.rna_eval)
                model = _fit_estimator(
                    configuration,
                    best_parameters,
                    x_train,
                    train["os_time_days"].to_numpy(dtype=float),
                    train["os_event"].to_numpy(dtype=int),
                    seed=seed + outer_fold,
                    small_test=small_test,
                    inner=False,
                )
                raw_train_risk = model.predict_risk(x_train)
                raw_test_risk = model.predict_risk(x_test)
                risk_mean = float(np.mean(raw_train_risk))
                risk_scale = float(np.std(raw_train_risk))
                risk_scale = risk_scale if risk_scale > 1e-8 else 1.0
                train_risk = (raw_train_risk - risk_mean) / risk_scale
                test_risk = (raw_test_risk - risk_mean) / risk_scale
                threshold = float(np.median(train_risk))
                calibration_mae, probabilities = _calibration_mae(train, test, train_risk, test_risk)
                train_c = concordance_index(train["os_time_days"], train["os_event"], train_risk)
                test_c = concordance_index(test["os_time_days"], test["os_event"], test_risk)
                performance_rows.append(
                    {
                        "seed": seed,
                        "outer_fold": outer_fold,
                        **asdict(configuration),
                        "backend": model.backend_,
                        "outer_train_patient_count": len(train),
                        "outer_test_patient_count": len(test),
                        "outer_train_event_count": int(train["os_event"].sum()),
                        "outer_test_event_count": int(test["os_event"].sum()),
                        "inner_best_c_index": inner_c_index,
                        "best_parameters": json.dumps(best_parameters, sort_keys=True),
                        "feature_count": arrays.feature_count,
                        "outer_train_c_index": train_c,
                        "outer_test_c_index": test_c,
                        "auc_1_year": time_dependent_auc(test["os_time_days"], test["os_event"], test_risk, 365),
                        "auc_3_year": time_dependent_auc(test["os_time_days"], test["os_event"], test_risk, 1095),
                        "auc_5_year": time_dependent_auc(test["os_time_days"], test["os_event"], test_risk, 1825),
                        "km_logrank_p_value": logrank_p_value(
                            test["os_time_days"],
                            test["os_event"],
                            test_risk >= threshold,
                        ),
                        "calibration_mae_3_year": calibration_mae,
                        "train_minus_test_gap": train_c - test_c,
                        "brier_score": np.nan,
                        "brier_score_status": "pending: IPCW Brier score not implemented",
                    }
                )
                oof_rows.append(
                    pd.DataFrame(
                        {
                            "patient_id": test["patient_id"].to_numpy(),
                            "OS_time": test["os_time_days"].to_numpy(),
                            "OS_status": test["os_event"].to_numpy(),
                            "age": test["age"].to_numpy(),
                            "male": test["male"].to_numpy(),
                            "stage_numeric": test["stage_numeric"].to_numpy(),
                            "model_name": configuration.model_name,
                            "config_id": configuration.config_id,
                            "feature_space": configuration.feature_space,
                            "outer_fold": outer_fold,
                            "seed": seed,
                            "oof_risk_score": test_risk,
                            "event_probability_3_year": probabilities,
                        }
                    )
                )
                print(
                    f"[stage2c] seed={seed} outer_fold={outer_fold}/{outer_splits} "
                    f"config={configuration.config_id} test_c_index={test_c:.4f}",
                    flush=True,
                )
    performance = pd.DataFrame(performance_rows)
    oof = pd.concat(oof_rows, ignore_index=True)
    leakage = pd.DataFrame(leakage_checks)
    performance.to_csv(tables / "stage2c_nested_cv_performance.csv", index=False)
    oof.to_csv(tables / "stage2c_nested_cv_oof_predictions_raw.csv", index=False)
    leakage.to_csv(tables / "stage2c_leakage_checks.csv", index=False)
    comparison = _comparison_table(project_root, performance)
    comparison.to_csv(tables / "stage2c_feature_space_comparison.csv", index=False)
    _plot_nested_cv(performance, figures)
    result = {
        "status": "passed",
        "dataset_mode": "toy_small_test" if small_test else "real_tcga_luad",
        "patient_count": len(dataset),
        "candidate_gene_count": len(genes),
        "seeds": list(requested_seeds),
        "outer_folds": outer_splits,
        "inner_folds": inner_splits,
        "evaluated_configuration_count": len(configurations),
        "outer_result_rows": len(performance),
        "oof_prediction_rows": len(oof),
        "maximum_fit_test_overlap_count": int(leakage["fit_test_overlap_count"].max()),
    }
    (logs / "stage2c_nested_cv_manifest.json").write_text(
        json.dumps({"generated_at": datetime.now().isoformat(timespec="seconds"), **result}, indent=2),
        encoding="utf-8",
    )
    return result
