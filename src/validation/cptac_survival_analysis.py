"""Stage 5B CPTAC protein survival analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from evaluation.cox_analysis import univariable_cox_risk_score
from validation.cptac_data_import import CPTACPaths
from validation.cptac_luad_preprocess import read_stage5_candidates


matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _bh(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(values.shape, np.nan)
    finite = np.isfinite(values)
    if not finite.any():
        return adjusted.tolist()
    order = np.argsort(values[finite])
    sorted_values = values[finite][order]
    ranks = np.arange(1, len(sorted_values) + 1)
    corrected = np.minimum.accumulate((sorted_values * len(sorted_values) / ranks)[::-1])[::-1]
    finite_idx = np.where(finite)[0]
    adjusted[finite_idx[order]] = np.clip(corrected, 0, 1)
    return adjusted.tolist()


def _load(paths: CPTACPaths) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    protein_path = paths.processed_dir / "cptac_luad_protein_matrix_processed.csv"
    clinical_path = paths.processed_dir / "cptac_luad_clinical_processed.csv"
    module_path = paths.tables_dir / "stage5b_cptac_protein_module_scores.csv"
    protein = pd.read_csv(protein_path) if protein_path.exists() else pd.DataFrame(columns=["sample_id"])
    clinical = pd.read_csv(clinical_path) if clinical_path.exists() else pd.DataFrame(columns=["sample_id"])
    modules = pd.read_csv(module_path) if module_path.exists() else pd.DataFrame()
    return protein, clinical, modules


def _cox_rows(frame: pd.DataFrame, variables: list[tuple[str, str]]) -> pd.DataFrame:
    rows = []
    usable_survival = {"os_time_days", "os_event"}.issubset(frame.columns)
    if not usable_survival or frame["os_time_days"].notna().sum() < 10 or frame["os_event"].sum() < 3:
        for name, layer in variables:
            rows.append(
                {
                    "feature_name": name,
                    "mechanism_layer": layer,
                    "cox_available": False,
                    "status": "survival_unavailable_or_too_few_events",
                    "hazard_ratio_per_sd": np.nan,
                    "ci95_lower": np.nan,
                    "ci95_upper": np.nan,
                    "p_value": np.nan,
                }
            )
        result = pd.DataFrame(rows)
        result["fdr_bh"] = np.nan
        return result
    for name, layer in variables:
        if name not in frame.columns:
            rows.append(
                {
                    "feature_name": name,
                    "mechanism_layer": layer,
                    "cox_available": False,
                    "status": "feature_unavailable",
                    "hazard_ratio_per_sd": np.nan,
                    "ci95_lower": np.nan,
                    "ci95_upper": np.nan,
                    "p_value": np.nan,
                }
            )
            continue
        subset = frame[["os_time_days", "os_event", name]].dropna()
        if len(subset) < 10 or subset["os_event"].sum() < 3 or subset[name].nunique() < 2:
            rows.append(
                {
                    "feature_name": name,
                    "mechanism_layer": layer,
                    "cox_available": False,
                    "status": "too_few_rows_events_or_variance",
                    "hazard_ratio_per_sd": np.nan,
                    "ci95_lower": np.nan,
                    "ci95_upper": np.nan,
                    "p_value": np.nan,
                }
            )
            continue
        try:
            cox = univariable_cox_risk_score(
                subset["os_time_days"].to_numpy(dtype=float),
                subset["os_event"].to_numpy(dtype=int),
                subset[name].to_numpy(dtype=float),
            )
            coef = float(cox["coefficient_per_sd"])
            se = float(cox["standard_error"])
            rows.append(
                {
                    "feature_name": name,
                    "mechanism_layer": layer,
                    "cox_available": True,
                    "status": "computed_exploratory",
                    "coefficient_per_sd": coef,
                    "hazard_ratio_per_sd": float(cox["hazard_ratio_per_sd"]),
                    "ci95_lower": float(np.exp(coef - 1.96 * se)),
                    "ci95_upper": float(np.exp(coef + 1.96 * se)),
                    "standard_error": se,
                    "p_value": float(cox["p_value"]),
                    "optimizer_success": bool(cox["optimizer_success"]),
                    "analysis_note": "Exploratory CPTAC protein Cox; not used for model training.",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "feature_name": name,
                    "mechanism_layer": layer,
                    "cox_available": False,
                    "status": f"cox_failed: {exc}",
                    "hazard_ratio_per_sd": np.nan,
                    "ci95_lower": np.nan,
                    "ci95_upper": np.nan,
                    "p_value": np.nan,
                }
            )
    result = pd.DataFrame(rows)
    result["fdr_bh"] = _bh(result["p_value"].tolist())
    return result


def run_cptac_survival_analysis(paths: CPTACPaths, *, small_test: bool = False) -> dict[str, object]:
    """Run exploratory survival analysis when CPTAC OS metadata are available."""

    candidates = read_stage5_candidates(paths, small_test=small_test)
    protein, clinical, modules = _load(paths)
    protein_out = paths.tables_dir / "stage5b_cptac_protein_survival_cox.csv"
    module_out = paths.tables_dir / "stage5b_cptac_protein_module_survival_cox.csv"
    figure_out = paths.figures_dir / "stage5b_cptac_protein_cox_forest.png"
    report_out = paths.reports_dir / "stage5b_cptac_survival_report.md"
    paths.tables_dir.mkdir(parents=True, exist_ok=True)
    paths.figures_dir.mkdir(parents=True, exist_ok=True)
    paths.reports_dir.mkdir(parents=True, exist_ok=True)

    merged = clinical.merge(protein, on="sample_id", how="inner") if not clinical.empty and not protein.empty else pd.DataFrame()
    variables = [
        (str(row.gene_symbol).upper(), str(row.mechanism_layer))
        for row in candidates.drop_duplicates("gene_symbol").itertuples(index=False)
    ]
    protein_cox = _cox_rows(merged, variables)
    protein_cox = protein_cox.rename(columns={"feature_name": "gene_symbol"})
    protein_cox.to_csv(protein_out, index=False)

    if not modules.empty and not clinical.empty:
        module_wide = modules.pivot_table(index="sample_id", columns="mechanism_layer", values="module_score").reset_index()
        module_frame = clinical.merge(module_wide, on="sample_id", how="inner")
        module_variables = [(column, column) for column in module_wide.columns if column != "sample_id"]
    else:
        module_frame = pd.DataFrame()
        module_variables = sorted({(str(layer), str(layer)) for layer in candidates["mechanism_layer"].dropna().unique()})
    module_cox = _cox_rows(module_frame, module_variables)
    module_cox = module_cox.rename(columns={"feature_name": "mechanism_layer_score"})
    module_cox.to_csv(module_out, index=False)
    _plot_cox_forest(protein_cox, figure_out)
    write_survival_report(protein_cox, module_cox, report_out, small_test=small_test)
    return {
        "status": "computed" if protein_cox["cox_available"].any() or module_cox["cox_available"].any() else "survival_unavailable",
        "protein_rows": len(protein_cox),
        "module_rows": len(module_cox),
    }


def _plot_cox_forest(cox: pd.DataFrame, output_path: Path) -> None:
    table = cox.loc[cox["cox_available"].astype(str).str.lower() == "true"].copy()
    if table.empty:
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        ax.text(0.5, 0.5, "CPTAC protein survival Cox unavailable", ha="center", va="center", fontsize=12)
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        return
    table = table.sort_values("hazard_ratio_per_sd").tail(20)
    y = np.arange(len(table))
    hr = table["hazard_ratio_per_sd"].to_numpy(dtype=float)
    low = table["ci95_lower"].to_numpy(dtype=float)
    high = table["ci95_upper"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, max(4.5, 0.25 * len(table))))
    ax.errorbar(hr, y, xerr=[hr - low, high - hr], fmt="o", color="#345995", ecolor="#777777")
    ax.axvline(1.0, color="#555555", linestyle="--", linewidth=1)
    ax.set_yticks(y, table["gene_symbol"], fontsize=7)
    ax.set_xlabel("Hazard ratio per SD")
    ax.set_title("Exploratory CPTAC protein Cox")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_survival_report(protein_cox: pd.DataFrame, module_cox: pd.DataFrame, output_path: Path, *, small_test: bool) -> None:
    computed = int(protein_cox["cox_available"].astype(str).str.lower().eq("true").sum())
    module_computed = int(module_cox["cox_available"].astype(str).str.lower().eq("true").sum())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "# Stage 5B CPTAC Survival Report\n\n"
        f"- Mode: {'toy small-test' if small_test else 'formal'}.\n"
        f"- Candidate protein Cox models computed: {computed}.\n"
        f"- Protein module Cox models computed: {module_computed}.\n"
        "- If no Cox models were computed, usable OS/PFI/RFS/DFS metadata were unavailable or insufficient.\n"
        "- Any computed CPTAC survival result is exploratory and was not used to train or tune Stage 2D.\n",
        encoding="utf-8",
    )

