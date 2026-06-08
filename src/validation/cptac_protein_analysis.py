"""Stage 5B CPTAC candidate protein and module analyses."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

from validation.cptac_data_import import CPTACPaths
from validation.cptac_luad_preprocess import read_stage5_candidates

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_processed(paths: CPTACPaths) -> tuple[pd.DataFrame, pd.DataFrame]:
    protein_path = paths.processed_dir / "cptac_luad_protein_matrix_processed.csv"
    clinical_path = paths.processed_dir / "cptac_luad_clinical_processed.csv"
    protein = pd.read_csv(protein_path) if protein_path.exists() else pd.DataFrame(columns=["sample_id"])
    clinical = pd.read_csv(clinical_path) if clinical_path.exists() else pd.DataFrame(columns=["sample_id"])
    return protein, clinical


def _placeholder(path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.text(0.5, 0.5, title, ha="center", va="center", fontsize=12)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _comparison(frame: pd.DataFrame, value_col: str, expected_direction: str) -> dict[str, object]:
    if value_col not in frame:
        return {"comparison_basis": "unavailable", "effect": np.nan, "p_value": np.nan, "consistent_with_stage4": "unavailable"}
    if "risk_group" in frame and frame["risk_group"].notna().sum() >= 6:
        high = frame.loc[frame["risk_group"].astype(str).str.lower() == "high", value_col].dropna()
        low = frame.loc[frame["risk_group"].astype(str).str.lower() == "low", value_col].dropna()
        if len(high) >= 3 and len(low) >= 3:
            effect = float(high.mean() - low.mean())
            p_value = float(mannwhitneyu(high, low, alternative="two-sided").pvalue)
            expected_positive = expected_direction == "higher_in_high_risk"
            consistent = (effect > 0 and expected_positive) or (effect < 0 and not expected_positive)
            return {
                "comparison_basis": "risk_group_high_minus_low",
                "effect": effect,
                "p_value": p_value,
                "consistent_with_stage4": "consistent" if consistent else "inconsistent",
            }
    if "risk_score" in frame and frame["risk_score"].notna().sum() >= 6:
        valid = frame[[value_col, "risk_score"]].dropna()
        if len(valid) >= 6 and valid[value_col].nunique() > 1:
            rho, p_value = spearmanr(valid["risk_score"], valid[value_col])
            expected_positive = expected_direction == "higher_in_high_risk"
            consistent = (rho > 0 and expected_positive) or (rho < 0 and not expected_positive)
            return {
                "comparison_basis": "risk_score_spearman",
                "effect": float(rho),
                "p_value": float(p_value),
                "consistent_with_stage4": "consistent" if consistent else "inconsistent",
            }
    if "stage_numeric" in frame and frame["stage_numeric"].notna().sum() >= 6:
        valid = frame[[value_col, "stage_numeric"]].dropna()
        if len(valid) >= 6 and valid[value_col].nunique() > 1 and valid["stage_numeric"].nunique() > 1:
            rho, p_value = spearmanr(valid["stage_numeric"], valid[value_col])
            expected_positive = expected_direction == "higher_in_high_risk"
            consistent = (rho > 0 and expected_positive) or (rho < 0 and not expected_positive)
            return {
                "comparison_basis": "clinical_stage_spearman_no_risk_score",
                "effect": float(rho),
                "p_value": float(p_value),
                "consistent_with_stage4": "consistent" if consistent else "inconsistent",
            }
    return {"comparison_basis": "unavailable", "effect": np.nan, "p_value": np.nan, "consistent_with_stage4": "unavailable"}


def run_candidate_protein_validation(paths: CPTACPaths, *, small_test: bool = False) -> dict[str, object]:
    """Analyze CPTAC abundance for Stage 5 candidate proteins."""

    candidates = read_stage5_candidates(paths, small_test=small_test)
    protein, clinical = _load_processed(paths)
    merged = protein.copy()
    if not clinical.empty and "sample_id" in clinical:
        merged = clinical.merge(protein, on="sample_id", how="inner")
    candidate_out = paths.tables_dir / "stage5b_cptac_candidate_protein_abundance.csv"
    module_out = paths.tables_dir / "stage5b_cptac_protein_module_scores.csv"
    consistency_out = paths.tables_dir / "stage5b_cptac_stage4_direction_consistency.csv"
    boxplot_out = paths.figures_dir / "stage5b_cptac_candidate_boxplots.png"
    heatmap_out = paths.figures_dir / "stage5b_cptac_module_score_heatmap.png"
    paths.tables_dir.mkdir(parents=True, exist_ok=True)
    paths.figures_dir.mkdir(parents=True, exist_ok=True)
    if protein.empty or len(protein.columns) <= 1:
        abundance = candidates[["gene_symbol", "mechanism_layer", "expected_direction"]].drop_duplicates().copy()
        abundance["cptac_available"] = False
        abundance["status"] = "unavailable_no_cptac_processed_matrix"
        modules = pd.DataFrame(columns=["sample_id", "mechanism_layer", "module_score", "n_available_proteins", "status"])
        consistency = abundance[["gene_symbol", "mechanism_layer", "expected_direction"]].copy()
        consistency["consistent_with_stage4"] = "unavailable"
        consistency["comparison_basis"] = "unavailable_no_cptac_processed_matrix"
        consistency["effect"] = np.nan
        consistency["p_value"] = np.nan
        abundance.to_csv(candidate_out, index=False)
        modules.to_csv(module_out, index=False)
        consistency.to_csv(consistency_out, index=False)
        _placeholder(boxplot_out, "CPTAC candidate protein abundance unavailable")
        _placeholder(heatmap_out, "CPTAC protein module scores unavailable")
        return {"status": "unavailable", "candidate_available": 0, "module_count": 0}

    abundance_rows = []
    consistency_rows = []
    for row in candidates.drop_duplicates("gene_symbol").itertuples(index=False):
        gene = str(row.gene_symbol).upper()
        available = gene in protein.columns
        abundance_row = {
            "gene_symbol": gene,
            "mechanism_layer": row.mechanism_layer,
            "expected_direction": row.expected_direction,
            "cptac_available": available,
            "sample_count": int(protein[gene].notna().sum()) if available else 0,
            "mean_abundance": float(protein[gene].mean()) if available else np.nan,
            "sd_abundance": float(protein[gene].std(ddof=0)) if available else np.nan,
            "missing_fraction": float(protein[gene].isna().mean()) if available else np.nan,
            "status": "available" if available else "unavailable_gene_not_in_cptac_matrix",
        }
        abundance_rows.append(abundance_row)
        comparison = _comparison(merged, gene, str(row.expected_direction)) if available else {
            "comparison_basis": "unavailable_gene_not_in_cptac_matrix",
            "effect": np.nan,
            "p_value": np.nan,
            "consistent_with_stage4": "unavailable",
        }
        consistency_rows.append(
            {
                "gene_symbol": gene,
                "mechanism_layer": row.mechanism_layer,
                "expected_direction": row.expected_direction,
                **comparison,
            }
        )
    abundance = pd.DataFrame(abundance_rows)
    consistency = pd.DataFrame(consistency_rows)

    module_frames = []
    for mechanism, group in candidates.groupby("mechanism_layer"):
        genes = [gene for gene in group["gene_symbol"].astype(str).str.upper().drop_duplicates() if gene in protein.columns]
        if not genes:
            continue
        module = protein[["sample_id", *genes]].copy()
        module["mechanism_layer"] = mechanism
        module["module_score"] = module[genes].mean(axis=1)
        module["n_available_proteins"] = len(genes)
        module["status"] = "available"
        module_frames.append(module[["sample_id", "mechanism_layer", "module_score", "n_available_proteins", "status"]])
    modules = pd.concat(module_frames, ignore_index=True) if module_frames else pd.DataFrame(columns=["sample_id", "mechanism_layer", "module_score", "n_available_proteins", "status"])

    abundance.to_csv(candidate_out, index=False)
    modules.to_csv(module_out, index=False)
    consistency.to_csv(consistency_out, index=False)
    _plot_candidate_boxplots(merged, abundance.loc[abundance["cptac_available"], "gene_symbol"].head(12).tolist(), boxplot_out)
    _plot_module_heatmap(modules, heatmap_out)
    return {
        "status": "analyzed",
        "candidate_available": int(abundance["cptac_available"].sum()),
        "module_count": int(modules["mechanism_layer"].nunique()) if not modules.empty else 0,
    }


def _plot_candidate_boxplots(frame: pd.DataFrame, genes: list[str], output_path: Path) -> None:
    if not genes or "risk_group" not in frame:
        _placeholder(output_path, "CPTAC candidate risk-group boxplots unavailable")
        return
    fig, axes = plt.subplots(3, 4, figsize=(11, 7.2))
    axes = axes.ravel()
    for ax, gene in zip(axes, genes):
        high = frame.loc[frame["risk_group"].astype(str).str.lower() == "high", gene].dropna()
        low = frame.loc[frame["risk_group"].astype(str).str.lower() == "low", gene].dropna()
        ax.boxplot([low, high], labels=["Low", "High"], widths=0.55)
        ax.set_title(gene, fontsize=9)
        ax.grid(axis="y", alpha=0.2)
    for ax in axes[len(genes):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_module_heatmap(modules: pd.DataFrame, output_path: Path) -> None:
    if modules.empty:
        _placeholder(output_path, "CPTAC protein module scores unavailable")
        return
    pivot = modules.pivot_table(index="mechanism_layer", columns="sample_id", values="module_score")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index, fontsize=8)
    ax.set_xticks([])
    ax.set_xlabel("CPTAC samples")
    ax.set_title("CPTAC candidate protein module scores")
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def integrate_stage5b_evidence(paths: CPTACPaths, *, small_test: bool = False) -> dict[str, object]:
    """Integrate Stage 4, Stage 5 HPA, and Stage 5B CPTAC evidence."""

    candidates = read_stage5_candidates(paths, small_test=small_test)
    stage5_integrated_path = (
        paths.root / "outputs" / "stage5_small_test" / "tables" / "integrated_protein_evidence.csv"
        if small_test and (paths.root / "outputs" / "stage5_small_test" / "tables" / "integrated_protein_evidence.csv").exists()
        else paths.root / "outputs" / "tables" / "stage5_integrated_protein_evidence.csv"
    )
    hpa = pd.read_csv(stage5_integrated_path) if stage5_integrated_path.exists() else pd.DataFrame()
    availability_path = paths.tables_dir / "stage5b_cptac_candidate_availability.csv"
    abundance_path = paths.tables_dir / "stage5b_cptac_candidate_protein_abundance.csv"
    consistency_path = paths.tables_dir / "stage5b_cptac_stage4_direction_consistency.csv"
    survival_path = paths.tables_dir / "stage5b_cptac_protein_survival_cox.csv"
    module_survival_path = paths.tables_dir / "stage5b_cptac_protein_module_survival_cox.csv"
    inventory_path = paths.tables_dir / "stage5b_cptac_data_inventory.csv"
    availability = pd.read_csv(availability_path) if availability_path.exists() else pd.DataFrame()
    abundance = pd.read_csv(abundance_path) if abundance_path.exists() else pd.DataFrame()
    consistency = pd.read_csv(consistency_path) if consistency_path.exists() else pd.DataFrame()
    survival = pd.read_csv(survival_path) if survival_path.exists() else pd.DataFrame()
    module_survival = pd.read_csv(module_survival_path) if module_survival_path.exists() else pd.DataFrame()
    inventory = pd.read_csv(inventory_path) if inventory_path.exists() else pd.DataFrame()
    if not availability.empty and "status" in availability.columns:
        availability = availability.rename(columns={"status": "cptac_availability_status"})
    if not consistency.empty and "p_value" in consistency.columns:
        consistency = consistency.rename(columns={"p_value": "stage5b_direction_p_value"})
    if not survival.empty and "p_value" in survival.columns:
        survival = survival.rename(columns={"p_value": "survival_p_value"})

    frame = candidates.copy()
    for table, columns in (
        (hpa, ["gene_symbol", "supported_by_HPA", "hpa_support_status", "evidence_level"]),
        (availability, ["gene_symbol", "cptac_available", "cptac_availability_status"]),
        (abundance, ["gene_symbol", "sample_count", "mean_abundance", "missing_fraction"]),
        (consistency, ["gene_symbol", "comparison_basis", "effect", "stage5b_direction_p_value", "consistent_with_stage4"]),
        (survival, ["gene_symbol", "cox_available", "hazard_ratio_per_sd", "ci95_lower", "ci95_upper", "survival_p_value", "fdr_bh"]),
    ):
        if table.empty or "gene_symbol" not in table:
            continue
        keep = [column for column in columns if column in table.columns]
        suffix = "_hpa" if "supported_by_HPA" in keep else "_cptac"
        frame = frame.merge(table[keep].drop_duplicates("gene_symbol"), on="gene_symbol", how="left", suffixes=("", suffix))
    if "supported_by_HPA" not in frame:
        frame["supported_by_HPA"] = False
    frame["supported_by_HPA"] = frame["supported_by_HPA"].fillna(False).astype(bool)
    if "cptac_available" not in frame:
        frame["cptac_available"] = False
    frame["cptac_available"] = frame["cptac_available"].fillna(False).astype(bool)
    consistency_values = (
        frame["consistent_with_stage4"]
        if "consistent_with_stage4" in frame.columns
        else pd.Series("", index=frame.index)
    )
    direction_p = pd.to_numeric(
        frame.get("stage5b_direction_p_value", pd.Series(index=frame.index, dtype=float)),
        errors="coerce",
    )
    frame["cptac_direction_consistent_nominal"] = (
        frame["cptac_available"] & consistency_values.fillna("").eq("consistent")
    )
    frame["supported_by_CPTAC_quantitative"] = (
        frame["cptac_direction_consistent_nominal"] & direction_p.le(0.05)
    )
    frame["supported_by_HPA_and_CPTAC"] = frame["supported_by_HPA"] & frame["supported_by_CPTAC_quantitative"]
    frame["stage5b_evidence_level"] = np.where(
        frame["supported_by_HPA_and_CPTAC"],
        "strong",
        np.where(frame["supported_by_HPA"], "moderate", np.where(frame["supported_by_CPTAC_quantitative"], "moderate", "unavailable")),
    )
    frame["cptac_interpretation_note"] = frame.apply(_stage5b_note, axis=1)

    integrated_out = paths.tables_dir / "stage5b_integrated_hpa_cptac_evidence.csv"
    heatmap_out = paths.figures_dir / "stage5b_integrated_hpa_cptac_heatmap.png"
    report_out = paths.reports_dir / "stage5b_cptac_validation_report.md"
    paths.tables_dir.mkdir(parents=True, exist_ok=True)
    paths.figures_dir.mkdir(parents=True, exist_ok=True)
    paths.reports_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(integrated_out, index=False)
    _plot_stage5b_heatmap(frame, heatmap_out)
    _write_stage5b_report(
        frame,
        inventory,
        availability,
        module_survival,
        report_out,
        small_test=small_test,
    )
    return {
        "status": "integrated",
        "genes": len(frame),
        "cptac_supported": int(frame["supported_by_CPTAC_quantitative"].sum()),
        "strong": int(frame["stage5b_evidence_level"].eq("strong").sum()),
        "integrated": integrated_out,
        "report": report_out,
    }


def _stage5b_note(row: pd.Series) -> str:
    if bool(row.get("supported_by_HPA_and_CPTAC", False)):
        return "HPA qualitative evidence plus CPTAC quantitative direction support; orthogonal support only."
    if bool(row.get("cptac_direction_consistent_nominal", False)):
        return "CPTAC direction is consistent but not nominally significant; not counted as quantitative support."
    if bool(row.get("supported_by_HPA", False)) and not bool(row.get("cptac_available", False)):
        return "HPA qualitative support only; CPTAC protein abundance unavailable in current run."
    if bool(row.get("cptac_available", False)) and row.get("consistent_with_stage4") == "consistent":
        return "CPTAC quantitative direction support available; HPA support absent or unavailable."
    if bool(row.get("cptac_available", False)):
        return "CPTAC protein detected but not directionally supportive under the pre-specified Stage 4 expectation."
    return "Protein evidence unavailable; do not interpret as negative."


def _plot_stage5b_heatmap(frame: pd.DataFrame, output_path: Path) -> None:
    matrix = frame[["supported_by_HPA", "cptac_available", "supported_by_CPTAC_quantitative", "supported_by_HPA_and_CPTAC"]].astype(int)
    ordered = frame.assign(_score=matrix.sum(axis=1)).sort_values(["mechanism_layer", "_score"], ascending=[True, False])
    matrix = ordered[["supported_by_HPA", "cptac_available", "supported_by_CPTAC_quantitative", "supported_by_HPA_and_CPTAC"]].astype(int)
    fig, ax = plt.subplots(figsize=(8.2, max(5, 0.22 * len(ordered))))
    image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="YlGn", vmin=0, vmax=1)
    ax.set_xticks(
        np.arange(4),
        ["HPA", "CPTAC available", "CPTAC direction", "Both"],
        rotation=25,
        ha="right",
    )
    ax.set_yticks(np.arange(len(ordered)), ordered["gene_symbol"], fontsize=7)
    ax.set_title("Integrated HPA/CPTAC evidence")
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _write_stage5b_report(
    frame: pd.DataFrame,
    inventory: pd.DataFrame,
    availability: pd.DataFrame,
    module_survival: pd.DataFrame,
    output_path: Path,
    *,
    small_test: bool,
) -> None:
    matrix_ok = bool(not inventory.empty and "protein_abundance_matrix" in set(inventory.get("role", [])))
    sample_count = 0
    protein_count = 0
    if not inventory.empty:
        protein_rows = inventory.loc[inventory.get("role", pd.Series(dtype=str)).eq("protein_abundance_matrix")]
        if not protein_rows.empty:
            sample_count = int(protein_rows["row_count"].dropna().iloc[0]) if protein_rows["row_count"].notna().any() else 0
            protein_count = int(protein_rows["column_count"].dropna().iloc[0]) - 1 if protein_rows["column_count"].notna().any() else 0
    available_count = int(frame["cptac_available"].sum())
    cptac_support = int(frame["supported_by_CPTAC_quantitative"].sum())
    both = int(frame["supported_by_HPA_and_CPTAC"].sum())
    hpa_only = frame.loc[frame["supported_by_HPA"] & ~frame["supported_by_CPTAC_quantitative"], "gene_symbol"].tolist()
    strong = int(frame["stage5b_evidence_level"].eq("strong").sum())
    layer_support = frame.loc[frame["supported_by_CPTAC_quantitative"]].groupby("mechanism_layer")["gene_symbol"].apply(list).to_dict()
    output_path.write_text(
        "# Stage 5B CPTAC-LUAD Quantitative Proteomic Validation Report\n\n"
        f"- Mode: {'toy small-test' if small_test else 'formal'}.\n"
        "- CPTAC evidence is orthogonal support, not causal confirmation.\n"
        "- Stage 2D model was not retrained and CPTAC was not used for model optimization.\n\n"
        "## Required Answers\n\n"
        f"1. CPTAC-LUAD protein abundance matrix acquired: `{matrix_ok}`.\n"
        f"2. CPTAC samples/proteins inventoried: `{sample_count}` samples, `{protein_count}` proteins.\n"
        f"3. Candidate proteins matched in CPTAC: `{available_count}/{len(frame)}`.\n"
        f"4. Mechanism layers with CPTAC quantitative support: `{layer_support}`.\n"
        f"5. Proliferation support: `{_layer_answer(layer_support, 'proliferation')}`.\n"
        f"6. Hypoxia support: `{_layer_answer(layer_support, 'hypoxia')}`.\n"
        f"7. EMT/CAF support: `EMT={_layer_answer(layer_support, 'emt_like_malignant_program')}; CAF={_layer_answer(layer_support, 'caf_matrix')}`.\n"
        f"8. Dendritic/B/plasma context support: `{_layer_answer(layer_support, 'dendritic_b_plasma_context')}`.\n"
        f"9. Protein abundance association with survival/stage: `{_association_answer(frame, module_survival)}`.\n"
        f"10. HPA-only candidates without CPTAC support: `{', '.join(hpa_only) if hpa_only else 'none'}`.\n"
        f"11. Candidates supported by both HPA and CPTAC: `{both}`.\n"
        f"12. Evidence upgraded from moderate to strong: `{strong}` candidates.\n"
        "13. CPTAC evidence is orthogonal support, not causal confirmation.\n\n"
        "## Integrity Notes\n\n"
        "- Unavailable CPTAC data are not negative evidence.\n"
        "- Small-test outputs are toy engineering checks only.\n",
        encoding="utf-8",
    )


def _layer_answer(layer_support: dict[str, list[str]], layer: str) -> str:
    genes = layer_support.get(layer, [])
    return ", ".join(genes) if genes else "not supported or unavailable"


def _association_answer(frame: pd.DataFrame, module_survival: pd.DataFrame) -> str:
    protein_fdr = pd.to_numeric(
        frame.get("fdr_bh", pd.Series(index=frame.index, dtype=float)),
        errors="coerce",
    )
    significant_proteins = frame.loc[protein_fdr.lt(0.05), "gene_symbol"].astype(str).tolist()
    significant_modules: list[str] = []
    if not module_survival.empty and "fdr_bh" in module_survival.columns:
        module_fdr = pd.to_numeric(module_survival["fdr_bh"], errors="coerce")
        name_column = (
            "mechanism_layer_score"
            if "mechanism_layer_score" in module_survival.columns
            else "mechanism_layer"
        )
        if name_column in module_survival.columns:
            significant_modules = (
                module_survival.loc[module_fdr.lt(0.05), name_column].astype(str).tolist()
            )
    stage_p = pd.to_numeric(
        frame.get("stage5b_direction_p_value", pd.Series(index=frame.index, dtype=float)),
        errors="coerce",
    )
    significant_stage = frame.loc[
        frame.get(
            "cptac_direction_consistent_nominal",
            pd.Series(False, index=frame.index),
        ).fillna(False)
        & stage_p.le(0.05),
        "gene_symbol",
    ].astype(str).tolist()
    if significant_proteins or significant_modules or significant_stage:
        return (
            f"stage-direction nominal P<0.05: {significant_stage or 'none'}; "
            f"protein survival FDR<0.05: {significant_proteins or 'none'}; "
            f"module survival FDR<0.05: {significant_modules or 'none'}"
        )
    return "unavailable"
