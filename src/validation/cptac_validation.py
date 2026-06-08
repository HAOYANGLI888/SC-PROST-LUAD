"""CPTAC/PDC protein abundance availability and optional validation."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class CPTACValidationError(RuntimeError):
    """Raised when a local CPTAC/PDC matrix cannot be parsed."""


def _paths(root: str | Path, *, small_test: bool = False) -> dict[str, Path]:
    project_root = Path(root).resolve()
    if small_test:
        base = project_root / "outputs" / "stage5_small_test"
        return {
            "root": project_root,
            "candidate": base / "tables" / "candidate_genes.csv",
            "availability": base / "tables" / "cptac_data_availability.csv",
            "abundance": base / "tables" / "cptac_candidate_protein_abundance.csv",
            "cox": base / "tables" / "cptac_protein_survival_cox.csv",
            "mrna_protein": base / "tables" / "mrna_protein_correlation.csv",
            "boxplot": base / "figures" / "cptac_protein_boxplots.png",
            "correlation_plot": base / "figures" / "mrna_protein_correlation.png",
            "report": base / "reports" / "cptac_validation_report.md",
        }
    return {
        "root": project_root,
        "candidate": project_root / "outputs" / "tables" / "stage5_candidate_genes.csv",
        "availability": project_root / "outputs" / "tables" / "stage5_cptac_data_availability.csv",
        "abundance": project_root / "outputs" / "tables" / "stage5_cptac_candidate_protein_abundance.csv",
        "cox": project_root / "outputs" / "tables" / "stage5_cptac_protein_survival_cox.csv",
        "mrna_protein": project_root / "outputs" / "tables" / "stage5_mrna_protein_correlation.csv",
        "boxplot": project_root / "outputs" / "figures" / "stage5_cptac_protein_boxplots.png",
        "correlation_plot": project_root / "outputs" / "figures" / "stage5_mrna_protein_correlation.png",
        "report": project_root / "outputs" / "reports" / "stage5_cptac_validation_report.md",
    }


def _candidate_frame(path: Path, *, small_test: bool = False) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Stage 5 candidate table missing: {path}. "
            "Run scripts/stage5_select_candidate_genes.py first."
        )
    frame = pd.read_csv(path, dtype=str)
    frame["gene_symbol"] = frame["gene_symbol"].astype(str).str.upper()
    return frame.head(8).copy() if small_test else frame


def find_local_cptac_matrix(root: str | Path, explicit_path: str | Path | None = None) -> Path | None:
    """Find a local CPTAC/PDC protein abundance matrix if available."""

    if explicit_path:
        path = Path(explicit_path)
        return path if path.exists() else None
    project_root = Path(root).resolve()
    candidates = []
    for base in (project_root / "data" / "raw", project_root / "data" / "processed"):
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            name = path.name.lower()
            if (
                any(token in name for token in ("cptac", "pdc", "protein", "proteome", "proteomic", "abundance"))
                and path.suffix.lower() in {".csv", ".tsv", ".txt"}
            ):
                candidates.append(path)
    return sorted(candidates, key=lambda p: (len(str(p)), str(p)))[0] if candidates else None


def _read_table(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    frame = pd.read_csv(path, sep=sep)
    if frame.empty or frame.shape[1] < 2:
        raise CPTACValidationError(f"Protein matrix is empty or too narrow: {path}")
    return frame


def parse_protein_matrix(path: str | Path) -> pd.DataFrame:
    """Parse a local protein matrix into sample_id x gene_symbol columns."""

    path = Path(path)
    frame = _read_table(path)
    lower = {column.lower(): column for column in frame.columns}
    sample_col = next((lower[key] for key in ("sample_id", "patient_id", "case_id") if key in lower), None)
    gene_col = next((lower[key] for key in ("gene_symbol", "gene", "protein", "protein_id") if key in lower), None)
    if sample_col is not None:
        result = frame.rename(columns={sample_col: "sample_id"}).copy()
        result["sample_id"] = result["sample_id"].astype(str)
        protein_cols = [column for column in result.columns if column != "sample_id"]
        result = result[["sample_id", *protein_cols]]
    elif gene_col is not None:
        sample_columns = [column for column in frame.columns if column != gene_col]
        transposed = frame.set_index(gene_col)[sample_columns].T.reset_index()
        transposed = transposed.rename(columns={"index": "sample_id"})
        result = transposed
    else:
        first = frame.columns[0]
        if frame[first].astype(str).str.match(r"^[A-Za-z0-9_.:-]+$").all():
            transposed = frame.set_index(first).T.reset_index()
            result = transposed.rename(columns={"index": "sample_id"})
        else:
            raise CPTACValidationError(
                "Could not infer sample or gene column in local CPTAC/PDC matrix."
            )
    result.columns = ["sample_id", *[str(column).upper().split(";")[0] for column in result.columns[1:]]]
    numeric = result.drop(columns="sample_id").apply(pd.to_numeric, errors="coerce")
    return pd.concat([result[["sample_id"]], numeric], axis=1)


def _toy_cptac(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(20260603)
    genes = candidates["gene_symbol"].drop_duplicates().head(8).tolist()
    n = 48
    risk = rng.normal(size=n)
    matrix = pd.DataFrame({"sample_id": [f"CPTAC_TOY_{i:03d}" for i in range(n)]})
    for gene in genes:
        direction = 0.6 if gene in {"MKI67", "TOP2A", "CA9", "VEGFA"} else -0.35
        matrix[gene] = rng.normal(size=n) + direction * risk
    metadata = pd.DataFrame(
        {
            "sample_id": matrix["sample_id"],
            "risk_score": risk,
            "risk_group": np.where(risk >= np.median(risk), "high", "low"),
            "os_time_days": np.maximum(60, 1300 - 120 * risk + rng.normal(0, 350, n)),
            "os_event": rng.binomial(1, 0.45, n),
        }
    )
    return matrix, metadata


def _unavailable_outputs(candidates: pd.DataFrame, paths: dict[str, Path], matrix_path: Path | None) -> None:
    availability = pd.DataFrame(
        [
            {
                "dataset": "CPTAC-LUAD/PDC",
                "local_matrix_detected": False,
                "matrix_path": str(matrix_path) if matrix_path else "",
                "status": "manual_download_required",
                "reason": "No local CPTAC/PDC protein abundance matrix was found.",
                "can_claim_cptac_validation": False,
            }
        ]
    )
    abundance = candidates[["gene_symbol", "mechanism_layer"]].drop_duplicates().copy()
    abundance["protein_abundance_available"] = False
    abundance["status"] = "not_available_manual_download_required"
    cox = abundance[["gene_symbol", "mechanism_layer"]].copy()
    cox["cox_available"] = False
    cox["status"] = "not_available_no_cptac_survival_data"
    mrna = abundance[["gene_symbol", "mechanism_layer"]].copy()
    mrna["mrna_protein_correlation_available"] = False
    mrna["status"] = "not_available_no_matched_mrna_protein_data"
    availability.to_csv(paths["availability"], index=False)
    abundance.to_csv(paths["abundance"], index=False)
    cox.to_csv(paths["cox"], index=False)
    mrna.to_csv(paths["mrna_protein"], index=False)
    _plot_placeholder(paths["boxplot"], "CPTAC/PDC protein matrix unavailable")
    _plot_placeholder(paths["correlation_plot"], "Matched mRNA-protein data unavailable")
    _write_cptac_report(paths["report"], availability, abundance, cox, mrna)


def _plot_placeholder(path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.text(0.5, 0.5, title, ha="center", va="center", fontsize=12)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _available_outputs(
    candidates: pd.DataFrame,
    protein: pd.DataFrame,
    metadata: pd.DataFrame | None,
    paths: dict[str, Path],
    *,
    matrix_path: Path | None,
) -> None:
    genes = candidates["gene_symbol"].drop_duplicates().tolist()
    available_genes = [gene for gene in genes if gene in protein.columns]
    availability = pd.DataFrame(
        [
            {
                "dataset": "CPTAC-LUAD/PDC local matrix",
                "local_matrix_detected": True,
                "matrix_path": str(matrix_path) if matrix_path else "toy_small_test",
                "candidate_genes": len(genes),
                "available_candidate_proteins": len(available_genes),
                "status": "local_matrix_available",
                "can_claim_cptac_validation": bool(available_genes),
            }
        ]
    )
    rows = []
    joined = protein.copy()
    if metadata is not None and "sample_id" in metadata:
        joined = metadata.merge(protein, on="sample_id", how="inner")
    for gene in genes:
        row = {
            "gene_symbol": gene,
            "mechanism_layer": candidates.loc[candidates["gene_symbol"] == gene, "mechanism_layer"].iloc[0],
            "protein_abundance_available": gene in protein.columns,
            "sample_count": int(protein[gene].notna().sum()) if gene in protein.columns else 0,
            "mean_abundance": float(protein[gene].mean()) if gene in protein.columns else np.nan,
        }
        if gene in joined.columns and "risk_group" in joined:
            high = joined.loc[joined["risk_group"].astype(str).str.lower() == "high", gene].dropna()
            low = joined.loc[joined["risk_group"].astype(str).str.lower() == "low", gene].dropna()
            row["high_vs_low_p_value"] = (
                float(mannwhitneyu(high, low).pvalue) if len(high) >= 3 and len(low) >= 3 else np.nan
            )
            row["mean_high_minus_low"] = float(high.mean() - low.mean()) if len(high) and len(low) else np.nan
        rows.append(row)
    abundance = pd.DataFrame(rows)
    cox = pd.DataFrame(
        [
            {
                "gene_symbol": gene,
                "mechanism_layer": candidates.loc[candidates["gene_symbol"] == gene, "mechanism_layer"].iloc[0],
                "cox_available": False,
                "status": "pending_survival_model_not_run_in_stage5_without_curated_cptac_survival_schema",
            }
            for gene in genes
        ]
    )
    corr_rows = []
    for gene in genes:
        row = {
            "gene_symbol": gene,
            "mechanism_layer": candidates.loc[candidates["gene_symbol"] == gene, "mechanism_layer"].iloc[0],
            "mrna_protein_correlation_available": False,
            "status": "not_available_no_matched_mrna_matrix",
        }
        if gene in joined.columns and "risk_score" in joined:
            valid = joined[[gene, "risk_score"]].dropna()
            if len(valid) >= 5 and valid[gene].nunique() > 1:
                rho, p_value = spearmanr(valid["risk_score"], valid[gene])
                row.update(
                    {
                        "protein_risk_spearman": float(rho),
                        "protein_risk_p_value": float(p_value),
                        "status": "protein_risk_correlation_computed_no_matched_mrna",
                    }
                )
        corr_rows.append(row)
    mrna = pd.DataFrame(corr_rows)
    availability.to_csv(paths["availability"], index=False)
    abundance.to_csv(paths["abundance"], index=False)
    cox.to_csv(paths["cox"], index=False)
    mrna.to_csv(paths["mrna_protein"], index=False)
    _plot_abundance(joined, available_genes[:8], paths["boxplot"])
    _plot_correlation(mrna, paths["correlation_plot"])
    _write_cptac_report(paths["report"], availability, abundance, cox, mrna)


def _plot_abundance(frame: pd.DataFrame, genes: list[str], path: Path) -> None:
    if not genes or "risk_group" not in frame:
        _plot_placeholder(path, "CPTAC protein abundance groups unavailable")
        return
    fig, axes = plt.subplots(2, 4, figsize=(10, 5.2))
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
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_correlation(correlation: pd.DataFrame, path: Path) -> None:
    if "protein_risk_spearman" not in correlation:
        _plot_placeholder(path, "CPTAC protein-risk correlation unavailable")
        return
    table = correlation.dropna(subset=["protein_risk_spearman"]).copy()
    if table.empty:
        _plot_placeholder(path, "CPTAC protein-risk correlation unavailable")
        return
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    colors = ["#B13C2E" if value > 0 else "#267A73" for value in table["protein_risk_spearman"]]
    ax.barh(table["gene_symbol"], table["protein_risk_spearman"], color=colors)
    ax.axvline(0, color="#555555", linewidth=1)
    ax.set_xlabel("Spearman rho with risk score")
    ax.set_title("CPTAC protein-risk associations")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_cptac_report(
    path: Path,
    availability: pd.DataFrame,
    abundance: pd.DataFrame,
    cox: pd.DataFrame,
    mrna: pd.DataFrame,
) -> None:
    status = str(availability["status"].iloc[0])
    available = int(abundance.get("protein_abundance_available", pd.Series(dtype=bool)).sum()) if not abundance.empty else 0
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Stage 5 CPTAC/PDC Validation Report\n\n"
        f"- Status: `{status}`.\n"
        f"- Candidate proteins with abundance available: `{available}`.\n"
        "- CPTAC/PDC evidence is orthogonal proteomic support only; it is not causal confirmation.\n"
        "- If status is `manual_download_required`, no CPTAC validation result should be claimed.\n\n"
        "## Tables\n\n"
        f"- Availability rows: {len(availability)}.\n"
        f"- Candidate abundance rows: {len(abundance)}.\n"
        f"- Survival Cox rows: {len(cox)}.\n"
        f"- mRNA/protein correlation rows: {len(mrna)}.\n",
        encoding="utf-8",
    )


def run_cptac_validation(
    root: str | Path = ".",
    *,
    protein_matrix: str | Path | None = None,
    small_test: bool = False,
) -> dict[str, Path | str | int]:
    """Run CPTAC/PDC availability check or optional local matrix analysis."""

    paths = _paths(root, small_test=small_test)
    candidates = _candidate_frame(paths["candidate"], small_test=small_test)
    for path in (
        paths["availability"],
        paths["abundance"],
        paths["cox"],
        paths["mrna_protein"],
        paths["boxplot"],
        paths["correlation_plot"],
        paths["report"],
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    if small_test and protein_matrix is None:
        protein, metadata = _toy_cptac(candidates)
        _available_outputs(candidates, protein, metadata, paths, matrix_path=None)
        status = "toy_small_test_available"
    else:
        matrix_path = find_local_cptac_matrix(paths["root"], protein_matrix)
        if matrix_path is None:
            _unavailable_outputs(candidates, paths, matrix_path)
            status = "manual_download_required"
        else:
            protein = parse_protein_matrix(matrix_path)
            _available_outputs(candidates, protein, None, paths, matrix_path=matrix_path)
            status = "local_matrix_available"
    return {
        "status": status,
        "availability": paths["availability"],
        "abundance": paths["abundance"],
        "report": paths["report"],
    }

