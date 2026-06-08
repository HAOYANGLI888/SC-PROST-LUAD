"""Human Protein Atlas orthogonal protein/IHC evidence for Stage 5."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd
import requests

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class HPAValidationError(RuntimeError):
    """Raised when HPA validation cannot proceed safely."""


HPA_BASE = "https://www.proteinatlas.org"
LUAD_KEY = "Cancer prognostics - Lung Adenocarcinoma (TCGA)"
LUAD_VALIDATION_KEY = "Cancer prognostics - Lung Adenocarcinoma (validation)"


def _paths(root: str | Path, *, small_test: bool = False) -> dict[str, Path]:
    project_root = Path(root).resolve()
    if small_test:
        base = project_root / "outputs" / "stage5_small_test"
        return {
            "root": project_root,
            "candidate": base / "tables" / "candidate_genes.csv",
            "evidence": base / "tables" / "hpa_protein_evidence.csv",
            "links": base / "tables" / "hpa_ihc_links.csv",
            "figure": base / "figures" / "hpa_evidence_summary.png",
            "report": base / "reports" / "hpa_validation_report.md",
            "cache": project_root / "data" / "metadata" / "stage5_small_test_hpa_cache",
        }
    return {
        "root": project_root,
        "candidate": project_root / "outputs" / "tables" / "stage5_candidate_genes.csv",
        "evidence": project_root / "outputs" / "tables" / "stage5_hpa_protein_evidence.csv",
        "links": project_root / "outputs" / "tables" / "stage5_hpa_ihc_links.csv",
        "figure": project_root / "outputs" / "figures" / "stage5_hpa_evidence_summary.png",
        "report": project_root / "outputs" / "reports" / "stage5_hpa_validation_report.md",
        "cache": project_root / "data" / "metadata" / "stage5_hpa_cache",
    }


def _read_candidates(path: Path, *, small_test: bool = False) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Stage 5 candidate table is missing: {path}. "
            "Run scripts/stage5_select_candidate_genes.py first."
        )
    frame = pd.read_csv(path, dtype=str)
    required = {"gene_symbol", "ensembl_id", "mechanism_layer"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise HPAValidationError(f"Candidate table is missing columns: {missing}")
    frame["gene_symbol"] = frame["gene_symbol"].astype(str).str.upper()
    if small_test:
        frame = frame.head(8).copy()
    return frame


def _json_cache_path(cache_dir: Path, ensembl_id: str, gene_symbol: str) -> Path:
    safe_symbol = "".join(ch for ch in gene_symbol if ch.isalnum() or ch in {"_", "-"})
    return cache_dir / f"{ensembl_id}_{safe_symbol}.json"


def fetch_hpa_json(
    ensembl_id: str,
    gene_symbol: str,
    *,
    cache_dir: str | Path,
    timeout: int = 30,
    retries: int = 2,
) -> tuple[dict[str, Any] | None, str, str]:
    """Fetch and cache one HPA JSON record."""

    if not ensembl_id or not str(ensembl_id).startswith("ENSG"):
        return None, "missing_ensembl_id", ""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    path = _json_cache_path(cache, ensembl_id, gene_symbol)
    url = f"{HPA_BASE}/{ensembl_id}.json"
    if path.exists() and path.stat().st_size > 0:
        try:
            return json.loads(path.read_text(encoding="utf-8")), "cached", url
        except json.JSONDecodeError:
            path.unlink(missing_ok=True)
    last_error = ""
    for attempt in range(retries + 1):
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code == 404:
                return None, "not_found", url
            response.raise_for_status()
            data = response.json()
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            return data, "downloaded", url
        except (requests.RequestException, json.JSONDecodeError) as exc:
            last_error = str(exc)
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    return None, f"request_failed: {last_error}", url


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return "; ".join(f"{k}:{v}" for k, v in value.items())
    if isinstance(value, list):
        return ";".join(str(item) for item in value)
    return str(value)


def _parse_hpa_record(candidate: pd.Series, data: dict[str, Any] | None, status: str, url: str) -> dict[str, Any]:
    gene = str(candidate["gene_symbol"]).upper()
    ensembl = str(candidate.get("ensembl_id", ""))
    page_url = f"{HPA_BASE}/{ensembl}-{gene}" if ensembl.startswith("ENSG") else f"{HPA_BASE}/search/{gene}"
    pathology_url = f"{page_url}/pathology/lung+cancer" if ensembl.startswith("ENSG") else page_url
    if data is None:
        return {
            "gene_symbol": gene,
            "ensembl_id": ensembl,
            "mechanism_layer": candidate.get("mechanism_layer", ""),
            "hpa_query_status": status,
            "hpa_url": page_url,
            "hpa_pathology_url": pathology_url,
            "protein_name": "",
            "hpa_evidence": "",
            "protein_evidence": "",
            "reliability_ih": "",
            "antibody_count": 0,
            "antibodies": "",
            "lung_cancer_ihc_availability": "unparsed_link_available" if ensembl.startswith("ENSG") else "unavailable",
            "lung_adenocarcinoma_prognostic": "",
            "lung_adenocarcinoma_prognostic_type": "",
            "lung_adenocarcinoma_p_value": "",
            "hpa_support_status": "not_available",
            "support_reason": status,
        }
    antibodies = data.get("Antibody") or []
    if isinstance(antibodies, str):
        antibodies = [antibodies]
    reliability = _stringify(data.get("Reliability (IH)"))
    hpa_evidence = _stringify(data.get("HPA evidence"))
    protein_evidence = _stringify(data.get("Evidence"))
    luad = data.get(LUAD_KEY) or {}
    luad_validation = data.get(LUAD_VALIDATION_KEY) or {}
    reliability_supported = reliability.lower() in {"enhanced", "supported", "approved", "supported;enhanced"}
    has_protein_record = bool(hpa_evidence or protein_evidence or antibodies)
    prognostic_supported = bool(luad.get("is_prognostic") or luad_validation.get("is_prognostic"))
    if has_protein_record and (reliability_supported or antibodies or prognostic_supported):
        support_status = "supported_or_linked_qualitative_hpa_evidence"
    elif has_protein_record:
        support_status = "hpa_record_available_uncertain_ih"
    else:
        support_status = "not_available"
    return {
        "gene_symbol": gene,
        "ensembl_id": ensembl,
        "mechanism_layer": candidate.get("mechanism_layer", ""),
        "hpa_query_status": status,
        "hpa_url": page_url,
        "hpa_pathology_url": pathology_url,
        "protein_name": data.get("Gene description", ""),
        "uniprot": _stringify(data.get("Uniprot")),
        "protein_class": _stringify(data.get("Protein class")),
        "hpa_evidence": hpa_evidence,
        "protein_evidence": protein_evidence,
        "reliability_ih": reliability,
        "antibody_count": len(antibodies),
        "antibodies": ";".join(antibodies),
        "rna_cancer_specificity": data.get("RNA cancer specificity", ""),
        "rna_cancer_distribution": data.get("RNA cancer distribution", ""),
        "protein_tissue_specificity": data.get("Protein tissue specificity", ""),
        "protein_tissue_distribution": data.get("Protein tissue distribution", ""),
        "tissue_expression_cluster": data.get("Tissue expression cluster", ""),
        "lung_tissue_cell_type_enrichment": ";".join(
            item for item in (data.get("RNA tissue cell type enrichment") or []) if "lung" in str(item).lower()
        ),
        "lung_cancer_ihc_availability": "unparsed_link_available" if ensembl.startswith("ENSG") else "unavailable",
        "staining_level": "not_structured_in_json",
        "staining_intensity": "not_structured_in_json",
        "staining_quantity": "not_structured_in_json",
        "tumor_vs_normal_qualitative_pattern": "manual_review_required",
        "lung_adenocarcinoma_prognostic": luad.get("prognostic", ""),
        "lung_adenocarcinoma_prognostic_type": luad.get("prognostic type", ""),
        "lung_adenocarcinoma_p_value": luad.get("p_val", ""),
        "lung_adenocarcinoma_validation_prognostic": luad_validation.get("prognostic", ""),
        "lung_adenocarcinoma_validation_type": luad_validation.get("prognostic type", ""),
        "lung_adenocarcinoma_validation_p_value": luad_validation.get("p_val", ""),
        "hpa_support_status": support_status,
        "support_reason": (
            "HPA JSON record and gene/pathology links available; IHC staining details require manual review "
            "because lung-cancer IHC image-level fields are not structured in this JSON endpoint."
        ),
    }


def run_hpa_validation(
    root: str | Path = ".",
    *,
    small_test: bool = False,
    timeout: int = 30,
) -> dict[str, Path | int]:
    """Run HPA evidence gathering for Stage 5 candidates."""

    paths = _paths(root, small_test=small_test)
    candidates = _read_candidates(paths["candidate"], small_test=small_test)
    rows = []
    for candidate in candidates.itertuples(index=False):
        series = pd.Series(candidate._asdict())
        data, status, url = fetch_hpa_json(
            str(series.get("ensembl_id", "")),
            str(series.get("gene_symbol", "")),
            cache_dir=paths["cache"],
            timeout=timeout,
        )
        rows.append(_parse_hpa_record(series, data, status, url))
    evidence = pd.DataFrame(rows)
    links = evidence[
        [
            "gene_symbol",
            "ensembl_id",
            "mechanism_layer",
            "hpa_url",
            "hpa_pathology_url",
            "lung_cancer_ihc_availability",
            "antibodies",
        ]
    ].copy()
    for path in (paths["evidence"], paths["links"], paths["figure"], paths["report"]):
        path.parent.mkdir(parents=True, exist_ok=True)
    evidence.to_csv(paths["evidence"], index=False)
    links.to_csv(paths["links"], index=False)
    plot_hpa_summary(evidence, paths["figure"])
    write_hpa_report(evidence, paths["report"], small_test=small_test)
    return {
        "candidate_count": len(candidates),
        "evidence": paths["evidence"],
        "links": paths["links"],
        "report": paths["report"],
    }


def plot_hpa_summary(evidence: pd.DataFrame, output_path: str | Path) -> None:
    """Plot HPA evidence support counts by mechanism."""

    frame = evidence.copy()
    status = frame["hpa_support_status"].fillna("").str.lower()
    frame["supported"] = (status.ne("not_available")) & (
        status.str.contains("supported") | status.str.contains("hpa_record_available")
    )
    summary = frame.groupby("mechanism_layer")["supported"].agg(["sum", "count"]).reset_index()
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.bar(summary["mechanism_layer"], summary["sum"], color="#267A73", label="HPA record/link support")
    ax.bar(
        summary["mechanism_layer"],
        summary["count"] - summary["sum"],
        bottom=summary["sum"],
        color="#CCCCCC",
        label="Unavailable",
    )
    ax.set_ylabel("Candidate proteins")
    ax.set_title("HPA qualitative protein/IHC evidence availability")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(frameon=False)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_hpa_report(evidence: pd.DataFrame, output_path: str | Path, *, small_test: bool = False) -> None:
    """Write an HPA validation report."""

    status = evidence["hpa_support_status"].fillna("").str.lower()
    supported = (status.ne("not_available")) & (
        status.str.contains("supported") | status.str.contains("hpa_record_available")
    )
    by_layer = evidence.assign(supported=supported).groupby("mechanism_layer")["supported"].sum()
    unavailable = evidence.loc[evidence["hpa_support_status"].eq("not_available"), "gene_symbol"].tolist()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "# Stage 5 HPA Validation Report\n\n"
        f"- Mode: {'toy small-test' if small_test else 'formal'}.\n"
        f"- Candidate proteins queried: {len(evidence)}.\n"
        f"- HPA records/link evidence available: {int(supported.sum())}/{len(evidence)}.\n"
        f"- Unavailable candidates: {', '.join(unavailable) if unavailable else 'none'}.\n\n"
        "## Mechanism-Layer HPA Support Counts\n\n"
        + "\n".join(f"- {layer}: {int(count)} supported/linkable candidates" for layer, count in by_layer.items())
        + "\n\n## Integrity Boundary\n\n"
        "- HPA evidence is qualitative orthogonal protein/IHC support, not causal confirmation.\n"
        "- Lung-cancer IHC image-level staining details were not treated as quantitative survival validation.\n"
        "- Links are provided for manual review when image-level fields are not structured in the JSON endpoint.\n",
        encoding="utf-8",
    )
