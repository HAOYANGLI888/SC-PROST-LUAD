"""Cell-state association statistics and plots for Stage 4."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.scrna_signature import CellStateSignature
from evaluation.cox_analysis import multivariable_cox_adjustment, univariable_cox_risk_score
from evaluation.plot_survival import plot_calibration, plot_km_high_low
from evaluation.survival_metrics import (
    calibration_points,
    logrank_p_value,
    predict_event_probability,
    time_dependent_auc,
)


BAD_STATE_NAMES = {
    "emt_like_tumor_cells",
    "m2_like_macrophages",
    "treg_cells",
    "exhausted_cd8_t_cells",
    "caf",
    "hypoxia_tumor_cells",
    "proliferative_tumor_cells",
}
PROTECTIVE_STATE_NAMES = {
    "cytotoxic_cd8_t_cells",
    "nk_cells",
    "m1_like_macrophages",
    "dendritic_cells",
    "b_cells",
    "plasma_cells",
}


def bh_adjust(p_values: Iterable[float]) -> list[float]:
    """Benjamini-Hochberg adjusted P values."""

    values = np.asarray(list(p_values), dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return adjusted.tolist()
    order = np.argsort(values[finite])
    sorted_values = values[finite][order]
    ranks = np.arange(1, len(sorted_values) + 1)
    corrected = sorted_values * len(sorted_values) / ranks
    corrected = np.minimum.accumulate(corrected[::-1])[::-1]
    corrected = np.clip(corrected, 0.0, 1.0)
    finite_indices = np.where(finite)[0]
    adjusted[finite_indices[order]] = corrected
    return adjusted.tolist()


def load_tcga_stage2d_context(root: str | Path, *, small_test: bool = False) -> pd.DataFrame:
    """Load TCGA clinical OS covariates and Stage 2D fold-local risk scores."""

    project_root = Path(root).resolve()
    dataset_path = project_root / "data" / "processed" / "stage2_rnaseq_survival_dataset.csv"
    risk_path = project_root / "data" / "processed" / "stage2d_tcga_fold_local_oof_predictions.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Stage 2 RNA survival dataset missing: {dataset_path}")
    if not risk_path.exists():
        raise FileNotFoundError(f"Stage 2D TCGA OOF risk scores missing: {risk_path}")
    usecols = ["patient_id", "os_time_days", "os_event", "age", "male", "stage_numeric"]
    clinical = pd.read_csv(dataset_path, usecols=usecols)
    risk = pd.read_csv(risk_path)
    required = {"patient_id", "os_time_days", "os_event", "risk_score"}
    missing = sorted(required - set(risk.columns))
    if missing:
        raise ValueError(f"Stage 2D risk table is missing columns: {missing}")
    merged = clinical.merge(
        risk[["patient_id", "risk_score", "outer_fold"]],
        on="patient_id",
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("TCGA clinical and Stage 2D risk scores have no overlapping patient IDs.")
    if small_test:
        merged = merged.sort_values("patient_id").head(40).copy()
    merged["risk_group"] = np.where(
        merged["risk_score"] >= float(merged["risk_score"].median()),
        "high",
        "low",
    )
    return merged


def signature_names(signatures: list[CellStateSignature]) -> list[str]:
    return [signature.signature_name for signature in signatures]


def correlate_with_risk(
    frame: pd.DataFrame,
    score_columns: list[str],
    *,
    risk_col: str = "risk_score",
    dataset: str = "TCGA-LUAD",
) -> pd.DataFrame:
    """Spearman correlations between risk score and cell-state scores."""

    rows = []
    risk = pd.to_numeric(frame[risk_col], errors="coerce")
    for column in score_columns:
        score = pd.to_numeric(frame[column], errors="coerce")
        valid = risk.notna() & score.notna()
        if int(valid.sum()) < 5 or score[valid].nunique() < 2:
            rho, p_value = np.nan, np.nan
        else:
            rho, p_value = spearmanr(risk[valid], score[valid])
        rows.append(
            {
                "dataset": dataset,
                "signature_name": column,
                "n": int(valid.sum()),
                "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
                "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
                "direction": "positive" if np.isfinite(rho) and rho > 0 else "negative" if np.isfinite(rho) and rho < 0 else "unknown",
            }
        )
    result = pd.DataFrame(rows)
    result["q_value_bh"] = bh_adjust(result["p_value"])
    return result


def compare_risk_groups(
    frame: pd.DataFrame,
    score_columns: list[str],
    *,
    group_col: str = "risk_group",
    dataset: str = "TCGA-LUAD",
) -> pd.DataFrame:
    """Compare high-risk and low-risk cell-state scores."""

    rows = []
    high_mask = frame[group_col].astype(str).str.lower() == "high"
    for column in score_columns:
        high = pd.to_numeric(frame.loc[high_mask, column], errors="coerce").dropna()
        low = pd.to_numeric(frame.loc[~high_mask, column], errors="coerce").dropna()
        if len(high) < 3 or len(low) < 3:
            p_value = np.nan
        else:
            p_value = float(mannwhitneyu(high, low, alternative="two-sided").pvalue)
        high_median = float(high.median()) if len(high) else np.nan
        low_median = float(low.median()) if len(low) else np.nan
        rows.append(
            {
                "dataset": dataset,
                "signature_name": column,
                "n_high": int(len(high)),
                "n_low": int(len(low)),
                "mean_high": float(high.mean()) if len(high) else np.nan,
                "mean_low": float(low.mean()) if len(low) else np.nan,
                "median_high": high_median,
                "median_low": low_median,
                "median_difference_high_minus_low": high_median - low_median,
                "p_value": p_value,
                "high_risk_enriched": bool(np.isfinite(high_median - low_median) and high_median > low_median),
            }
        )
    result = pd.DataFrame(rows)
    result["q_value_bh"] = bh_adjust(result["p_value"])
    return result


def univariable_cell_state_cox(
    frame: pd.DataFrame,
    score_columns: list[str],
    *,
    time_col: str = "os_time_days",
    event_col: str = "os_event",
    dataset: str = "TCGA-LUAD",
) -> pd.DataFrame:
    """Univariable Cox analysis for each cell-state score."""

    rows = []
    for column in score_columns:
        subset = frame[[time_col, event_col, column]].dropna()
        row: dict[str, object] = {
            "dataset": dataset,
            "signature_name": column,
            "n": int(len(subset)),
            "events": int(subset[event_col].sum()) if len(subset) else 0,
        }
        try:
            result = univariable_cox_risk_score(
                subset[time_col].to_numpy(dtype=float),
                subset[event_col].to_numpy(dtype=int),
                subset[column].to_numpy(dtype=float),
            )
            row.update(result)
        except Exception as exc:  # pragma: no cover - audit fallback
            row.update(
                {
                    "coefficient_per_sd": np.nan,
                    "hazard_ratio_per_sd": np.nan,
                    "standard_error": np.nan,
                    "z_score": np.nan,
                    "p_value": np.nan,
                    "optimizer_success": False,
                    "error": str(exc),
                }
            )
        rows.append(row)
    result = pd.DataFrame(rows)
    result["q_value_bh"] = bh_adjust(result["p_value"])
    return result


def multivariable_cell_state_cox(
    frame: pd.DataFrame,
    score_columns: list[str],
    *,
    clinical_covariates: tuple[str, ...] = ("age", "male", "stage_numeric"),
    dataset: str = "TCGA-LUAD",
) -> pd.DataFrame:
    """Cox adjustment for each cell-state score plus clinical covariates."""

    rows = []
    for column in score_columns:
        predictions = frame[
            ["os_time_days", "os_event", column, *[cov for cov in clinical_covariates if cov in frame.columns]]
        ].rename(columns={column: "risk_score"})
        predictions = predictions.dropna(subset=["os_time_days", "os_event", "risk_score"])
        try:
            result = multivariable_cox_adjustment(
                predictions,
                clinical_covariates=tuple(cov for cov in clinical_covariates if cov in predictions.columns),
            )
            result.insert(0, "signature_name", column)
            result.insert(0, "dataset", dataset)
            rows.append(result)
        except Exception as exc:  # pragma: no cover - audit fallback
            rows.append(
                pd.DataFrame(
                    [
                        {
                            "dataset": dataset,
                            "signature_name": column,
                            "covariate": "risk_score",
                            "coefficient_per_sd": np.nan,
                            "hazard_ratio_per_sd": np.nan,
                            "standard_error": np.nan,
                            "z_score": np.nan,
                            "p_value": np.nan,
                            "analysis_scope": "cell-state score plus clinical covariates",
                            "optimizer_success": False,
                            "error": str(exc),
                        }
                    ]
                )
            )
    combined = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not combined.empty and "p_value" in combined:
        risk_mask = combined["covariate"].astype(str) == "risk_score"
        combined["q_value_bh"] = np.nan
        combined.loc[risk_mask, "q_value_bh"] = bh_adjust(combined.loc[risk_mask, "p_value"])
    return combined


def risk_score_survival_summary(
    frame: pd.DataFrame,
    *,
    risk_col: str = "risk_score",
    output_prefix: str | Path | None = None,
) -> dict[str, float]:
    """Summarize the Stage 2D risk score in the analyzed Stage 4 cohort."""

    risk = frame[risk_col].to_numpy(dtype=float)
    times = frame["os_time_days"].to_numpy(dtype=float)
    events = frame["os_event"].to_numpy(dtype=int)
    threshold = float(np.median(risk))
    groups = risk >= threshold
    summary = {
        "patients": int(len(frame)),
        "events": int(events.sum()),
        "censored": int((events == 0).sum()),
        "risk_median_cutoff": threshold,
        "km_logrank_p": logrank_p_value(times, events, groups),
        "auc_1_year": time_dependent_auc(times, events, risk, 365),
        "auc_3_year": time_dependent_auc(times, events, risk, 1095),
        "auc_5_year": time_dependent_auc(times, events, risk, 1825),
    }
    if output_prefix is not None:
        prefix = Path(output_prefix)
        plot_km_high_low(times, events, risk, threshold, summary["km_logrank_p"], prefix.with_name(prefix.name + "_km.png"))
        probabilities = predict_event_probability(times, events, risk, risk, 1095)
        pred, obs = calibration_points(times, events, probabilities, 1095, bins=4)
        plot_calibration(pred, obs, prefix.with_name(prefix.name + "_calibration.png"), horizon_days=1095)
    return summary


def external_consistency(
    tcga_correlation: pd.DataFrame,
    geo_correlation: pd.DataFrame,
    signatures: list[CellStateSignature],
) -> pd.DataFrame:
    """Summarize whether GEO risk-cell-state correlation directions match TCGA."""

    meta = {
        signature.signature_name: {
            "cell_state": signature.cell_state,
            "category": signature.category,
            "expected_risk_direction": signature.expected_risk_direction,
        }
        for signature in signatures
    }
    tcga = tcga_correlation.set_index("signature_name")
    rows = []
    for signature in signature_names(signatures):
        tcga_rho = float(tcga.loc[signature, "spearman_rho"]) if signature in tcga.index else np.nan
        tcga_sign = int(np.sign(tcga_rho)) if np.isfinite(tcga_rho) and tcga_rho != 0 else 0
        subset = geo_correlation.loc[geo_correlation["signature_name"] == signature].copy()
        valid = subset.loc[np.isfinite(subset["spearman_rho"])]
        matching = 0
        opposite = 0
        for rho in valid["spearman_rho"]:
            sign = int(np.sign(float(rho))) if float(rho) != 0 else 0
            if sign == 0 or tcga_sign == 0:
                continue
            if sign == tcga_sign:
                matching += 1
            else:
                opposite += 1
        consistency_label = (
            "consistent"
            if len(valid) and matching >= max(2, int(np.ceil(0.75 * len(valid))))
            else "mixed_or_unstable"
        )
        expected_direction = meta.get(signature, {}).get("expected_risk_direction", "unknown")
        if expected_direction == "higher_risk":
            expected_match = bool(tcga_sign > 0 and consistency_label == "consistent")
        elif expected_direction == "lower_risk":
            expected_match = bool(tcga_sign < 0 and consistency_label == "consistent")
        else:
            expected_match = np.nan
        rows.append(
            {
                "signature_name": signature,
                **meta.get(signature, {}),
                "tcga_spearman_rho": tcga_rho,
                "tcga_direction": "positive" if tcga_sign > 0 else "negative" if tcga_sign < 0 else "unknown",
                "geo_cohorts_evaluable": int(len(valid)),
                "geo_cohorts_direction_match_tcga": int(matching),
                "geo_cohorts_direction_opposite_tcga": int(opposite),
                "direction_consistency_fraction": matching / len(valid) if len(valid) else np.nan,
                "external_direction_consistency": consistency_label,
                "matches_predefined_risk_direction": expected_match,
            }
        )
    return pd.DataFrame(rows)


def plot_tcga_heatmap(frame: pd.DataFrame, score_columns: list[str], output_path: str | Path) -> None:
    ordered = frame.sort_values("risk_score")
    matrix = ordered[score_columns].apply(pd.to_numeric, errors="coerce")
    matrix = (matrix - matrix.mean(axis=0)) / matrix.std(axis=0, ddof=0).replace(0.0, 1.0)
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    image = ax.imshow(matrix.T.to_numpy(dtype=float), aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
    ax.set_yticks(np.arange(len(score_columns)), [name.replace("_", " ") for name in score_columns], fontsize=7)
    ax.set_xticks([])
    ax.set_xlabel("TCGA patients ordered by Stage 2D RNA risk score")
    ax.set_title("Cell-state signature scores")
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, label="z-score")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_group_boxplots(frame: pd.DataFrame, score_columns: list[str], output_path: str | Path) -> None:
    columns = score_columns
    fig, axes = plt.subplots(4, 4, figsize=(10.5, 8.5))
    axes = axes.ravel()
    for ax, column in zip(axes, columns):
        high = pd.to_numeric(frame.loc[frame["risk_group"] == "high", column], errors="coerce").dropna()
        low = pd.to_numeric(frame.loc[frame["risk_group"] == "low", column], errors="coerce").dropna()
        ax.boxplot([low, high], labels=["Low", "High"], patch_artist=True, widths=0.55)
        ax.set_title(column.replace("_", " "), fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        ax.grid(axis="y", alpha=0.2)
    for ax in axes[len(columns):]:
        ax.axis("off")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_correlation_bar(correlation: pd.DataFrame, output_path: str | Path) -> None:
    table = correlation.sort_values("spearman_rho")
    colors = ["#B13C2E" if value > 0 else "#267A73" for value in table["spearman_rho"].fillna(0)]
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.barh(table["signature_name"].str.replace("_", " "), table["spearman_rho"], color=colors)
    ax.axvline(0.0, color="#555555", linewidth=1)
    ax.set_xlabel("Spearman rho with Stage 2D RNA risk score")
    ax.set_title("Risk score and cell-state associations")
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_cox_forest(cox: pd.DataFrame, output_path: str | Path) -> None:
    table = cox.copy()
    if "covariate" in table:
        table = table.loc[table["covariate"].astype(str) == "risk_score"].copy()
    table = table.sort_values("hazard_ratio_per_sd")
    hr = pd.to_numeric(table["hazard_ratio_per_sd"], errors="coerce")
    se = pd.to_numeric(table["standard_error"], errors="coerce")
    coef = pd.to_numeric(table["coefficient_per_sd"], errors="coerce")
    low = np.exp(coef - 1.96 * se)
    high = np.exp(coef + 1.96 * se)
    y = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.errorbar(hr, y, xerr=[hr - low, high - hr], fmt="o", color="#345995", ecolor="#777777", capsize=2)
    ax.axvline(1.0, color="#555555", linestyle="--", linewidth=1)
    ax.set_yticks(y, table["signature_name"].str.replace("_", " "), fontsize=7)
    ax.set_xlabel("Hazard ratio per SD cell-state score")
    ax.set_title("Univariable cell-state Cox analysis")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_geo_consistency_heatmap(geo_correlation: pd.DataFrame, output_path: str | Path) -> None:
    if geo_correlation.empty:
        return
    pivot = geo_correlation.pivot_table(
        index="signature_name",
        columns="dataset",
        values="spearman_rho",
        aggfunc="mean",
    ).sort_index()
    fig, ax = plt.subplots(figsize=(7.8, 6.2))
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="RdBu_r", vmin=-0.6, vmax=0.6)
    ax.set_xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)), [idx.replace("_", " ") for idx in pivot.index], fontsize=7)
    ax.set_title("External risk-cell-state correlation consistency")
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, label="Spearman rho")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_mechanism_summary(
    tcga_correlation: pd.DataFrame,
    consistency: pd.DataFrame,
    output_path: str | Path,
) -> None:
    selected = tcga_correlation.loc[
        tcga_correlation["signature_name"].isin(BAD_STATE_NAMES | PROTECTIVE_STATE_NAMES)
    ].copy()
    if selected.empty:
        return
    selected = selected.merge(
        consistency[["signature_name", "direction_consistency_fraction"]],
        on="signature_name",
        how="left",
    )
    selected = selected.sort_values("spearman_rho")
    colors = [
        "#B13C2E" if name in BAD_STATE_NAMES else "#267A73"
        for name in selected["signature_name"]
    ]
    fig, ax = plt.subplots(figsize=(8.0, 5.8))
    ax.scatter(
        selected["spearman_rho"],
        selected["signature_name"].str.replace("_", " "),
        s=70 + 180 * selected["direction_consistency_fraction"].fillna(0.0),
        color=colors,
        alpha=0.85,
    )
    ax.axvline(0.0, color="#555555", linewidth=1)
    ax.set_xlabel("TCGA Spearman rho with RNA risk score")
    ax.set_title("Stage 4 mechanism summary")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
