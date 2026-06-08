"""GEO external-validation readiness checks without outcome fabrication."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data.geo_expression_import import (
    export_tcga_gene_annotation_from_star_counts,
    missing_fixed_model_gene_fraction,
    prepare_geo_expression,
)
from data.geo_survival_preprocess import load_geo_os


GEO_COHORTS = ("GSE31210", "GSE50081", "GSE72094", "GSE68465")


def _write_templates(root: Path) -> None:
    templates = root / "data" / "metadata" / "geo_templates"
    templates.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"probe_id": ["probe_001"], "GSM000001": [1.25], "GSM000002": [2.50]}).to_csv(
        templates / "expression_matrix_template.tsv", sep="\t", index=False
    )
    pd.DataFrame({"probe_id": ["probe_001"], "gene_symbol": ["TP53"]}).to_csv(
        templates / "probe_annotation_template.tsv", sep="\t", index=False
    )
    pd.DataFrame({"sample_id": ["GSM000001"], "OS_time": [365], "OS_status": [1]}).to_csv(
        templates / "clinical_survival_template.csv", index=False
    )


def _existing(directory: Path, names: tuple[str, ...]) -> Path | None:
    return next((directory / name for name in names if (directory / name).exists()), None)


def _small_test_fixture(root: Path) -> None:
    directory = root / "data" / "raw" / "geo" / "GSE_TOY"
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "probe_id": ["p1", "p2", "p3", "p4"],
            "GSM1": [1.0, 2.0, 4.0, 8.0],
            "GSM2": [1.2, 2.4, 3.0, 7.0],
            "GSM3": [0.8, 1.8, 5.0, 9.0],
        }
    ).to_csv(directory / "expression_matrix.tsv", sep="\t", index=False)
    pd.DataFrame(
        {"probe_id": ["p1", "p2", "p3", "p4"], "gene_symbol": ["TSPAN6", "TSPAN6", "DPM1", "SCYL3"]}
    ).to_csv(directory / "probe_annotation.tsv", sep="\t", index=False)
    pd.DataFrame(
        {"sample_id": ["GSM1", "GSM2", "GSM3"], "OS_time": [365, 640, 900], "OS_status": [1, 0, 1]}
    ).to_csv(directory / "clinical_survival.csv", index=False)


def _small_test_annotation(root: Path) -> Path:
    output = root / "data" / "metadata" / "stage2c_tcga_gene_annotation.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "gene_id": ["ENSG00000000003", "ENSG00000000419", "ENSG00000000457"],
            "gene_symbol": ["TSPAN6", "DPM1", "SCYL3"],
            "gene_type": ["protein_coding", "protein_coding", "protein_coding"],
        }
    ).to_csv(output, index=False)
    return output


def prepare_geo_validation_readiness(
    root: str | Path = ".",
    *,
    collapse_strategy: str = "mean",
    small_test: bool = False,
) -> dict[str, Any]:
    """Prepare local GEO files when present and report manual-download gaps."""

    project_root = Path(root).resolve()
    _write_templates(project_root)
    try:
        annotation = export_tcga_gene_annotation_from_star_counts(project_root)
    except FileNotFoundError:
        if not small_test:
            raise
        annotation = _small_test_annotation(project_root)
    fixed_spec = project_root / "outputs" / "checkpoints" / "stage2c_tcga_fixed_rna_validation_model.json"
    fixed_pickle = project_root / "outputs" / "checkpoints" / "stage2c_tcga_fixed_rna_validation_model.pkl"
    if small_test:
        _small_test_fixture(project_root)
        cohorts = ("GSE_TOY",)
        required_genes = ["ENSG00000000003", "ENSG00000000419", "ENSG00000000457"]
    else:
        cohorts = GEO_COHORTS
        if not fixed_spec.exists() or not fixed_pickle.exists():
            raise FileNotFoundError("Frozen TCGA Stage 2C external-validation artifact is missing. Generate OOF scores first.")
        with fixed_pickle.open("rb") as handle:
            import pickle

            required_genes = list(pickle.load(handle)["required_gene_ids"])
    rows: list[dict[str, Any]] = []
    processed_dir = project_root / "data" / "processed" / "geo"
    processed_dir.mkdir(parents=True, exist_ok=True)
    for cohort in cohorts:
        directory = project_root / "data" / "raw" / "geo" / cohort
        expression_path = _existing(directory, ("expression_matrix.tsv", "expression_matrix.csv", f"{cohort}_series_matrix.txt", f"{cohort}_series_matrix.txt.gz"))
        annotation_path = _existing(directory, ("probe_annotation.tsv", "probe_annotation.csv"))
        survival_path = _existing(directory, ("clinical_survival.csv", "clinical_survival.tsv"))
        missing = [
            name
            for name, path in (
                ("expression_matrix", expression_path),
                ("probe_annotation", annotation_path),
                ("clinical_survival", survival_path),
            )
            if path is None
        ]
        if missing:
            rows.append(
                {
                    "cohort": cohort,
                    "status": "manual_download_required",
                    "missing_files": ";".join(missing),
                    "sample_count": np.nan,
                    "gene_count": np.nan,
                    "missing_fixed_model_gene_fraction": np.nan,
                    "external_validation_completed": False,
                    "notes": "Place user-downloaded GEO files under data/raw/geo/<GSE>/.",
                }
            )
            continue
        try:
            expression = prepare_geo_expression(expression_path, annotation_path, collapse_strategy=collapse_strategy)
            survival = load_geo_os(survival_path)
            merged = survival.merge(expression, on="sample_id", how="inner", validate="one_to_one")
            if merged.empty:
                raise ValueError("No sample IDs overlap expression and survival tables.")
            missing_fraction = missing_fixed_model_gene_fraction(expression, annotation, required_genes)
            merged.to_csv(processed_dir / f"{cohort}_validation_ready.csv", index=False)
            rows.append(
                {
                    "cohort": cohort,
                    "status": "ready_for_fixed_model_validation",
                    "missing_files": "",
                    "sample_count": len(merged),
                    "gene_count": expression.shape[1] - 1,
                    "missing_fixed_model_gene_fraction": missing_fraction,
                    "external_validation_completed": False,
                    "notes": "Imported and independently z-scored. Frozen-model scoring remains a separate future run.",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "cohort": cohort,
                    "status": "blocked_invalid_local_files",
                    "missing_files": "",
                    "sample_count": np.nan,
                    "gene_count": np.nan,
                    "missing_fixed_model_gene_fraction": np.nan,
                    "external_validation_completed": False,
                    "notes": str(exc),
                }
            )
    table = pd.DataFrame(rows)
    output = project_root / "outputs" / "tables" / "stage2c_external_validation_readiness.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    result = {
        "status": "passed",
        "mode": "toy_small_test" if small_test else "real_readiness_only",
        "cohorts_checked": list(cohorts),
        "ready_cohort_count": int((table["status"] == "ready_for_fixed_model_validation").sum()),
        "external_validation_completed": False,
        "readiness_table": str(output),
        "tcga_gene_annotation": str(annotation),
    }
    logs = project_root / "outputs" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "stage2c_geo_validation_readiness.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if not small_test and (project_root / "outputs" / "tables" / "stage2c_nested_cv_oof_predictions_raw.csv").exists():
        from training.oof_prediction import build_stage2c_report

        build_stage2c_report(project_root)
    return result
