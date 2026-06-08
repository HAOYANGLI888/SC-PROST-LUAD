"""Stage 2 clinical + RNA-seq survival dataset preparation."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data.clinical_preprocess import load_tcga_cdr_os
from data.rnaseq_preprocess import candidate_feature_table, load_rnaseq_matrix


SEEDS = (42, 3407, 2026)
EXPECTED_STAGE1_COHORT = 507
DEFAULT_RNASEQ_MATRIX = "data/raw/tcga_luad/rnaseq/tcga_luad_tpm_matrix.csv"
DEFAULT_SURVIVAL_TABLE = (
    "data/raw/tcga_luad/clinical/"
    "Survival_SupplementalTable_S1_20171025_xena_sp.tsv"
)


class SurvivalDatasetError(RuntimeError):
    """Raised when a Stage 2 dataset cannot be prepared safely."""


@dataclass(frozen=True)
class Stage2Paths:
    """Resolved Stage 2 inputs and outputs."""

    root: Path

    @classmethod
    def from_root(cls, root: str | Path) -> "Stage2Paths":
        return cls(Path(root).resolve())

    @property
    def stage1_matrix(self) -> Path:
        return self.root / "data" / "metadata" / "tcga_luad_patient_modality_matrix.csv"

    @property
    def survival_table(self) -> Path:
        return self.root / DEFAULT_SURVIVAL_TABLE

    @property
    def default_rnaseq_matrix(self) -> Path:
        return self.root / DEFAULT_RNASEQ_MATRIX

    @property
    def dataset(self) -> Path:
        return self.root / "data" / "processed" / "stage2_rnaseq_survival_dataset.csv"

    @property
    def feature_list(self) -> Path:
        return self.root / "data" / "metadata" / "stage2_feature_list.csv"

    @property
    def dataset_manifest(self) -> Path:
        return self.root / "data" / "metadata" / "stage2_dataset_manifest.json"

    @property
    def split_summary(self) -> Path:
        return self.root / "outputs" / "tables" / "stage2_train_val_test_summary.csv"

    def split_file(self, seed: int) -> Path:
        return self.root / "data" / "metadata" / f"stage2_split_seed{seed}.csv"

    def ensure_dirs(self) -> None:
        for directory in (
            self.dataset.parent,
            self.feature_list.parent,
            self.split_summary.parent,
            self.root / "outputs" / "logs",
        ):
            directory.mkdir(parents=True, exist_ok=True)


def load_stage1_eligible_patients(path: str | Path) -> pd.DataFrame:
    """Load the frozen Stage 1 clinical + OS + RNA-seq eligibility cohort."""

    matrix_path = Path(path)
    if not matrix_path.exists():
        raise FileNotFoundError(f"Stage 1 patient matrix not found: {matrix_path}")
    frame = pd.read_csv(matrix_path)
    required = {"patient_id", "clinical_available", "os_available", "rnaseq_available"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SurvivalDatasetError(f"Stage 1 patient matrix is missing columns: {missing}")
    for column in ("clinical_available", "os_available", "rnaseq_available"):
        frame[column] = frame[column].astype(str).str.lower().isin({"true", "1", "yes"})
    eligible = frame.loc[
        frame["clinical_available"] & frame["os_available"] & frame["rnaseq_available"],
        ["patient_id"],
    ].drop_duplicates()
    if eligible.empty:
        raise SurvivalDatasetError("Stage 1 matrix contains no clinical + OS + RNA-seq patients.")
    return eligible


def _split_once(dataset: pd.DataFrame, seed: int) -> pd.DataFrame:
    patient_ids = dataset["patient_id"].astype(str)
    events = dataset["os_event"].astype(int)
    train_ids, holdout_ids = train_test_split(
        patient_ids,
        test_size=0.40,
        random_state=seed,
        stratify=events,
    )
    holdout = dataset.set_index("patient_id").loc[holdout_ids]
    val_ids, test_ids = train_test_split(
        holdout.index.to_series(),
        test_size=0.50,
        random_state=seed,
        stratify=holdout["os_event"].astype(int),
    )
    assignments = pd.DataFrame(
        {
            "patient_id": list(train_ids) + list(val_ids) + list(test_ids),
            "split": ["train"] * len(train_ids) + ["validation"] * len(val_ids) + ["test"] * len(test_ids),
            "seed": seed,
        }
    )
    return assignments.sort_values(["split", "patient_id"]).reset_index(drop=True)


def save_splits(dataset: pd.DataFrame, paths: Stage2Paths) -> pd.DataFrame:
    """Create deterministic stratified 60/20/20 splits for all required seeds."""

    summaries: list[dict[str, Any]] = []
    for seed in SEEDS:
        assignments = _split_once(dataset, seed)
        assignments.to_csv(paths.split_file(seed), index=False)
        merged = assignments.merge(dataset[["patient_id", "os_event"]], on="patient_id", how="left", validate="one_to_one")
        for split_name in ("train", "validation", "test"):
            subset = merged.loc[merged["split"] == split_name]
            deaths = int(subset["os_event"].sum())
            summaries.append(
                {
                    "seed": seed,
                    "split": split_name,
                    "patient_count": len(subset),
                    "death_count": deaths,
                    "censored_count": len(subset) - deaths,
                    "death_fraction": deaths / len(subset),
                }
            )
    summary = pd.DataFrame(summaries)
    summary.to_csv(paths.split_summary, index=False)
    return summary


def make_toy_dataset(
    *,
    n_patients: int = 96,
    n_genes: int = 80,
    seed: int = 2026,
) -> pd.DataFrame:
    """Create a deterministic toy survival cohort for offline smoke testing."""

    rng = np.random.default_rng(seed)
    patient_ids = [f"TOY-{index:04d}" for index in range(1, n_patients + 1)]
    expression = rng.gamma(shape=2.0, scale=2.0, size=(n_patients, n_genes))
    log_expression = np.log2(expression + 1.0)
    age = rng.normal(66.0, 8.0, size=n_patients)
    male = rng.binomial(1, 0.52, size=n_patients).astype(float)
    stage = rng.choice([1.0, 2.0, 3.0, 4.0], size=n_patients, p=[0.35, 0.28, 0.25, 0.12])
    linear_risk = (
        0.75 * log_expression[:, 0]
        - 0.65 * log_expression[:, 1]
        + 0.45 * log_expression[:, 2]
        + 0.025 * (age - 65.0)
        + 0.30 * (stage - 1.0)
    )
    event_time = rng.exponential(scale=1700.0 / np.exp(linear_risk - np.mean(linear_risk)))
    censor_time = rng.exponential(scale=2400.0, size=n_patients)
    observed_time = np.maximum(np.minimum(event_time, censor_time), 1.0)
    event = (event_time <= censor_time).astype(int)
    genes = [f"GENE_{index:04d}" for index in range(1, n_genes + 1)]
    frame = pd.DataFrame(log_expression, columns=genes)
    frame.insert(0, "stage_numeric", stage)
    frame.insert(0, "male", male)
    frame.insert(0, "age", age)
    frame.insert(0, "os_event", event)
    frame.insert(0, "os_time_days", observed_time.round(3))
    frame.insert(0, "patient_id", patient_ids)
    frame.insert(0, "dataset_mode", "toy_small_test")
    return frame


def _real_dataset(
    paths: Stage2Paths,
    *,
    rnaseq_matrix: Path,
    input_scale: str,
) -> tuple[pd.DataFrame, int, dict[str, Any]]:
    eligible = load_stage1_eligible_patients(paths.stage1_matrix)
    clinical = load_tcga_cdr_os(paths.survival_table)
    expression = load_rnaseq_matrix(rnaseq_matrix, input_scale=input_scale)
    merged = (
        eligible.merge(clinical, on="patient_id", how="inner", validate="one_to_one")
        .merge(expression, on="patient_id", how="inner", validate="one_to_one")
    )
    merged["os_event"] = pd.to_numeric(merged["os_event"], errors="coerce")
    merged["os_time_days"] = pd.to_numeric(merged["os_time_days"], errors="coerce")
    missing_os = merged.loc[merged[["os_event", "os_time_days"]].isna().any(axis=1), "patient_id"].tolist()
    nonpositive_os = merged.loc[merged["os_time_days"].fillna(0) <= 0, "patient_id"].tolist()
    merged = merged.dropna(subset=["os_event", "os_time_days"])
    merged = merged.loc[merged["os_time_days"] > 0].copy()
    merged["os_event"] = merged["os_event"].astype(int)
    invalid_status = sorted(set(merged["os_event"]) - {0, 1})
    if invalid_status:
        raise SurvivalDatasetError(f"OS status must be 0/1. Found: {invalid_status}")
    if merged.empty:
        raise SurvivalDatasetError("Clinical + OS + RNA-seq merge returned no patients.")
    merged.insert(0, "dataset_mode", "real_tcga_luad")
    diagnostics = {
        "stage1_eligible_patient_count": len(eligible),
        "clinical_patient_count": len(clinical),
        "rnaseq_matrix_patient_count": len(expression),
        "merged_before_os_validation_count": len(
            eligible.merge(clinical, on="patient_id", how="inner").merge(
                expression[["patient_id"]],
                on="patient_id",
                how="inner",
            )
        ),
        "missing_os_patient_ids": missing_os,
        "nonpositive_os_time_patient_ids": nonpositive_os,
    }
    return merged, len(eligible), diagnostics


def prepare_stage2_dataset(
    root: str | Path = ".",
    *,
    small_test: bool = False,
    rnaseq_matrix: str | Path | None = None,
    input_scale: str = "tpm",
) -> dict[str, Any]:
    """Prepare Stage 2 data and deterministic split manifests."""

    paths = Stage2Paths.from_root(root)
    paths.ensure_dirs()
    if small_test:
        dataset = make_toy_dataset()
        eligible_count = EXPECTED_STAGE1_COHORT
        input_description = "built-in deterministic toy cohort"
        real_diagnostics: dict[str, Any] | None = None
    else:
        matrix_path = Path(rnaseq_matrix) if rnaseq_matrix else paths.default_rnaseq_matrix
        if not matrix_path.is_absolute():
            matrix_path = paths.root / matrix_path
        dataset, eligible_count, real_diagnostics = _real_dataset(
            paths,
            rnaseq_matrix=matrix_path,
            input_scale=input_scale,
        )
        input_description = str(matrix_path)

    gene_columns = [column for column in dataset.columns if column.startswith("GENE_") or column not in {
        "dataset_mode", "patient_id", "os_time_days", "os_event", "age", "male",
        "stage_numeric", "gender", "stage_raw",
    }]
    if not gene_columns:
        raise SurvivalDatasetError("Prepared Stage 2 dataset has no RNA gene columns.")
    dataset.to_csv(paths.dataset, index=False)
    candidate_feature_table(dataset[["patient_id", *gene_columns]]).to_csv(paths.feature_list, index=False)
    split_summary = save_splits(dataset, paths)
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_mode": str(dataset["dataset_mode"].iloc[0]),
        "input": input_description,
        "stage1_expected_clinical_os_rnaseq": EXPECTED_STAGE1_COHORT,
        "stage1_eligible_count_observed": eligible_count,
        "prepared_patient_count": len(dataset),
        "death_count": int(dataset["os_event"].sum()),
        "censored_count": int(len(dataset) - dataset["os_event"].sum()),
        "os_time_unit": "days",
        "os_event_encoding": "death=1,censored=0",
        "candidate_gene_count": len(gene_columns),
        "split_seeds": list(SEEDS),
        "train_val_test_ratio": "60/20/20",
    }
    matrix_build_summary = paths.root / "data" / "metadata" / "stage2_rnaseq_matrix_build_summary.csv"
    if not small_test and matrix_build_summary.exists():
        summary = pd.read_csv(matrix_build_summary)
        manifest["rnaseq_matrix_build_summary"] = {
            str(row["metric"]): row["value"]
            for row in summary.to_dict(orient="records")
        }
    if real_diagnostics is not None:
        manifest["real_merge_diagnostics"] = real_diagnostics
    paths.dataset_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"manifest": manifest, "split_summary": split_summary.to_dict(orient="records")}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare Stage 2 TCGA-LUAD clinical + OS + RNA-seq data.")
    parser.add_argument("--config", default="configs/base.yaml", help="Project YAML config path.")
    parser.add_argument("--root", default=".", help="Project root. Defaults to current directory.")
    parser.add_argument("--rnaseq-matrix", help=f"RNA-seq matrix path. Default: {DEFAULT_RNASEQ_MATRIX}")
    parser.add_argument("--input-scale", choices=["tpm", "log2_tpm"], default="tpm")
    parser.add_argument("--small-test", action="store_true", help="Prepare deterministic toy data without network access.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config_path = Path(args.root) / args.config
    if not config_path.exists():
        parser.error(f"Config file not found: {config_path}")
    try:
        result = prepare_stage2_dataset(
            args.root,
            small_test=args.small_test,
            rnaseq_matrix=args.rnaseq_matrix,
            input_scale=args.input_scale,
        )
    except (FileNotFoundError, SurvivalDatasetError, RuntimeError) as exc:
        parser.exit(1, f"Stage 2 preparation failed: {exc}\n")
    print(json.dumps(result["manifest"], indent=2))
    return 0
