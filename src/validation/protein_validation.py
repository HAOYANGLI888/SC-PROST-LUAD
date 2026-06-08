"""Integrated Stage 5 protein/HPA/CPTAC evidence."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class ProteinValidationIntegrationError(RuntimeError):
    """Raised when integrated Stage 5 evidence cannot be built."""


def _paths(root: str | Path, *, small_test: bool = False) -> dict[str, Path]:
    project_root = Path(root).resolve()
    if small_test:
        base = project_root / "outputs" / "stage5_small_test"
        return {
            "root": project_root,
            "candidate": base / "tables" / "candidate_genes.csv",
            "hpa": base / "tables" / "hpa_protein_evidence.csv",
            "cptac": base / "tables" / "cptac_candidate_protein_abundance.csv",
            "availability": base / "tables" / "cptac_data_availability.csv",
            "integrated": base / "tables" / "integrated_protein_evidence.csv",
            "figure": base / "figures" / "integrated_protein_evidence_heatmap.png",
            "report": base / "reports" / "protein_validation_report.md",
        }
    return {
        "root": project_root,
        "candidate": project_root / "outputs" / "tables" / "stage5_candidate_genes.csv",
        "hpa": project_root / "outputs" / "tables" / "stage5_hpa_protein_evidence.csv",
        "cptac": project_root / "outputs" / "tables" / "stage5_cptac_candidate_protein_abundance.csv",
        "availability": project_root / "outputs" / "tables" / "stage5_cptac_data_availability.csv",
        "integrated": project_root / "outputs" / "tables" / "stage5_integrated_protein_evidence.csv",
        "figure": project_root / "outputs" / "figures" / "stage5_integrated_protein_evidence_heatmap.png",
        "report": project_root / "outputs" / "reports" / "stage5_protein_validation_report.md",
    }


def _read_required(paths: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing = [str(path) for path in (paths["candidate"], paths["hpa"], paths["cptac"], paths["availability"]) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Integrated Stage 5 evidence requires previous Stage 5 outputs: "
            + ", ".join(missing)
        )
    candidates = pd.read_csv(paths["candidate"])
    hpa = pd.read_csv(paths["hpa"])
    cptac = pd.read_csv(paths["cptac"])
    availability = pd.read_csv(paths["availability"])
    for frame in (candidates, hpa, cptac):
        if "gene_symbol" not in frame:
            raise ProteinValidationIntegrationError("Stage 5 evidence tables require gene_symbol.")
        frame["gene_symbol"] = frame["gene_symbol"].astype(str).str.upper()
    return candidates, hpa, cptac, availability


def _hpa_supported(status: object) -> bool:
    text = str(status or "").lower()
    return text != "not_available" and ("supported" in text or "hpa_record_available" in text)


def _cptac_supported(row: pd.Series, cptac_available: bool) -> bool:
    if not cptac_available:
        return False
    value = row.get("protein_abundance_available", False)
    return str(value).lower() in {"true", "1", "yes"} or bool(value is True)


def _evidence_level(hpa: bool, cptac: bool, hpa_record: bool, cptac_status: str) -> str:
    if hpa and cptac:
        return "strong"
    if hpa or cptac:
        return "moderate"
    if hpa_record or cptac_status == "local_matrix_available":
        return "weak"
    return "unavailable"


def integrate_protein_evidence(root: str | Path = ".", *, small_test: bool = False) -> pd.DataFrame:
    """Integrate HPA and CPTAC/PDC evidence for Stage 5 candidates."""

    paths = _paths(root, small_test=small_test)
    candidates, hpa, cptac, availability = _read_required(paths)
    cptac_status = str(availability.get("status", pd.Series(["unknown"])).iloc[0])
    cptac_available = str(availability.get("local_matrix_detected", pd.Series([False])).iloc[0]).lower() == "true"
    hpa_keep = [
        "gene_symbol",
        "hpa_support_status",
        "hpa_url",
        "hpa_pathology_url",
        "reliability_ih",
        "lung_adenocarcinoma_prognostic",
        "lung_adenocarcinoma_prognostic_type",
        "lung_adenocarcinoma_p_value",
        "lung_cancer_ihc_availability",
    ]
    hpa_keep = [column for column in hpa_keep if column in hpa.columns]
    cptac_keep = [
        "gene_symbol",
        "protein_abundance_available",
        "sample_count",
        "mean_abundance",
        "high_vs_low_p_value",
        "mean_high_minus_low",
        "status",
    ]
    cptac_keep = [column for column in cptac_keep if column in cptac.columns]
    integrated = candidates.merge(hpa[hpa_keep], on="gene_symbol", how="left")
    integrated = integrated.merge(cptac[cptac_keep], on="gene_symbol", how="left", suffixes=("", "_cptac"))
    supported_hpa = integrated["hpa_support_status"].map(_hpa_supported)
    hpa_record = integrated["hpa_support_status"].notna() & integrated["hpa_support_status"].ne("not_available")
    supported_cptac = integrated.apply(lambda row: _cptac_supported(row, cptac_available), axis=1)
    integrated["supported_by_HPA"] = supported_hpa
    integrated["supported_by_CPTAC"] = supported_cptac
    integrated["supported_by_both"] = supported_hpa & supported_cptac
    integrated["not_available"] = ~(supported_hpa | supported_cptac | hpa_record)
    integrated["inconsistent"] = False
    integrated["evidence_level"] = [
        _evidence_level(bool(h), bool(c), bool(hr), cptac_status)
        for h, c, hr in zip(supported_hpa, supported_cptac, hpa_record)
    ]
    integrated["interpretation_note"] = integrated.apply(_interpretation_note, axis=1)
    paths["integrated"].parent.mkdir(parents=True, exist_ok=True)
    integrated.to_csv(paths["integrated"], index=False)
    plot_integrated_heatmap(integrated, paths["figure"])
    write_integrated_report(integrated, availability, paths["report"], small_test=small_test)
    return integrated


def _interpretation_note(row: pd.Series) -> str:
    if bool(row.get("supported_by_both", False)):
        return "Supported by both HPA qualitative protein/IHC evidence and local CPTAC/PDC abundance."
    if bool(row.get("supported_by_HPA", False)):
        return "Supported by HPA qualitative protein/IHC evidence or linkable HPA record; not CPTAC-validated here."
    if bool(row.get("supported_by_CPTAC", False)):
        return "Supported by local CPTAC/PDC abundance; HPA support not available here."
    if bool(row.get("not_available", False)):
        return "Protein evidence unavailable in current local/automatic Stage 5 run; do not interpret as negative evidence."
    return "Weak or incomplete protein evidence; use only as supplementary context."


def plot_integrated_heatmap(integrated: pd.DataFrame, output_path: str | Path) -> None:
    """Plot evidence level by candidate gene."""

    level_score = {"unavailable": 0, "weak": 1, "moderate": 2, "strong": 3}
    table = integrated.copy()
    table["score"] = table["evidence_level"].map(level_score).fillna(0)
    table = table.sort_values(["mechanism_layer", "score"], ascending=[True, False])
    matrix = table[["supported_by_HPA", "supported_by_CPTAC", "supported_by_both"]].astype(int).to_numpy()
    fig, ax = plt.subplots(figsize=(7.5, max(4.5, 0.22 * len(table))))
    image = ax.imshow(matrix, aspect="auto", cmap="YlGn", vmin=0, vmax=1)
    ax.set_xticks([0, 1, 2], ["HPA", "CPTAC", "Both"])
    ax.set_yticks(np.arange(len(table)), table["gene_symbol"], fontsize=7)
    ax.set_title("Integrated protein evidence")
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, label="Support")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_integrated_report(
    integrated: pd.DataFrame,
    availability: pd.DataFrame,
    output_path: str | Path,
    *,
    small_test: bool = False,
) -> None:
    """Write final Stage 5 protein validation report."""

    cptac_status = str(availability.get("status", pd.Series(["unknown"])).iloc[0])
    hpa_supported = integrated.loc[integrated["supported_by_HPA"], "gene_symbol"].tolist()
    cptac_supported = integrated.loc[integrated["supported_by_CPTAC"], "gene_symbol"].tolist()
    both = integrated.loc[integrated["supported_by_both"], "gene_symbol"].tolist()
    unavailable = integrated.loc[integrated["evidence_level"].eq("unavailable"), "gene_symbol"].tolist()
    main_figure = integrated.loc[
        integrated["evidence_level"].isin(["strong", "moderate"])
        & integrated["mechanism_layer"].isin(["proliferation", "hypoxia", "emt_like_malignant_program", "caf_matrix"]),
        "gene_symbol",
    ].head(10).tolist()
    supplement = integrated.loc[~integrated["gene_symbol"].isin(main_figure), "gene_symbol"].tolist()
    layer_summary = (
        integrated.groupby("mechanism_layer")["evidence_level"]
        .value_counts()
        .unstack(fill_value=0)
        .reset_index()
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "# Stage 5 Protein / HPA / CPTAC Orthogonal Validation Report\n\n"
        f"- Mode: {'toy small-test' if small_test else 'formal'}.\n"
        "- Stage 2D risk model was not retrained.\n"
        "- Stage 4 signatures were not modified based on protein evidence.\n"
        "- Protein evidence is orthogonal support, not causal confirmation.\n\n"
        "## Required Answers\n\n"
        f"1. Mechanism layers with protein/IHC support: {', '.join(layer_summary['mechanism_layer'].tolist()) if hpa_supported or cptac_supported else 'none with current data'}.\n"
        f"2. Proliferation markers with HPA/CPTAC support: {', '.join(_genes_by_layer(integrated, 'proliferation')) or 'none'}.\n"
        f"3. Hypoxia markers with HPA/CPTAC support: {', '.join(_genes_by_layer(integrated, 'hypoxia')) or 'none'}.\n"
        f"4. EMT/CAF markers with HPA/CPTAC support: {', '.join(_genes_by_layer(integrated, 'emt_like_malignant_program') + _genes_by_layer(integrated, 'caf_matrix')) or 'none'}.\n"
        f"5. Candidates with only transcriptomic support or unavailable protein support: {', '.join(unavailable) if unavailable else 'none'}.\n"
        f"6. Proteins suitable for main figure: {', '.join(main_figure) if main_figure else 'none yet'}.\n"
        f"7. Proteins suitable for supplementary material: {', '.join(supplement[:20]) if supplement else 'none'}.\n"
        f"8. Manuscript skeleton readiness: {'yes, with protein evidence framed as orthogonal support' if hpa_supported or cptac_supported else 'not yet; protein evidence unavailable'}.\n"
        f"9. Additional data needed: raw scRNA would strengthen cell-state specificity; CPTAC status is `{cptac_status}`.\n"
        "10. Protein evidence is orthogonal support, not causal confirmation.\n\n"
        "## Evidence Summary\n\n"
        f"- HPA-supported/linkable genes: {len(hpa_supported)} ({', '.join(hpa_supported[:20])}).\n"
        f"- CPTAC-supported genes: {len(cptac_supported)} ({', '.join(cptac_supported[:20]) if cptac_supported else 'none'}).\n"
        f"- Supported by both: {len(both)} ({', '.join(both) if both else 'none'}).\n"
        f"- CPTAC/PDC status: `{cptac_status}`.\n\n"
        "## Integrity Notes\n\n"
        "- Unavailable protein evidence was not interpreted as negative evidence.\n"
        "- HPA qualitative IHC/link evidence was not treated as quantitative survival validation.\n"
        "- CPTAC validation should not be claimed unless a local compatible protein abundance matrix is available and analyzed.\n",
        encoding="utf-8",
    )


def _genes_by_layer(integrated: pd.DataFrame, layer: str) -> list[str]:
    return integrated.loc[
        (integrated["mechanism_layer"] == layer)
        & integrated["evidence_level"].isin(["strong", "moderate"]),
        "gene_symbol",
    ].tolist()


def run_integrated_protein_validation(
    root: str | Path = ".",
    *,
    small_test: bool = False,
) -> dict[str, Path | int]:
    """Build integrated Stage 5 evidence artifacts."""

    paths = _paths(root, small_test=small_test)
    integrated = integrate_protein_evidence(paths["root"], small_test=small_test)
    return {
        "genes": len(integrated),
        "integrated": paths["integrated"],
        "figure": paths["figure"],
        "report": paths["report"],
    }
