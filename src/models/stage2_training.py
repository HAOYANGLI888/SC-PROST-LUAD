"""Leakage-safe Stage 2 baseline training orchestration."""

from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from data.clinical_preprocess import CLINICAL_COLUMNS, ClinicalTrainPreprocessor
from data.rnaseq_preprocess import RNATrainPreprocessor
from data.survival_dataset import SEEDS, Stage2Paths
from models.cox_baseline import RandomSurvivalForestAdapter, TorchCoxElasticNet
from models.deepsurv import DeepSurvEstimator


MODEL_NAMES = (
    "clinical_only_cox",
    "rna_only_elasticnet_cox",
    "rna_clinical_elasticnet_cox",
    "random_survival_forest",
    "deepsurv_rna_only",
    "deepsurv_rna_clinical",
)


class Stage2TrainingError(RuntimeError):
    """Raised when Stage 2 baseline training cannot proceed."""


def _load_dataset(paths: Stage2Paths, *, small_test: bool) -> tuple[pd.DataFrame, list[str]]:
    if not paths.dataset.exists():
        raise Stage2TrainingError(
            f"Prepared dataset not found: {paths.dataset}. "
            "Run scripts/stage2_prepare_rnaseq_survival.py first."
        )
    dataset = pd.read_csv(paths.dataset)
    required = {"dataset_mode", "patient_id", "os_time_days", "os_event", *CLINICAL_COLUMNS}
    missing = sorted(required - set(dataset.columns))
    if missing:
        raise Stage2TrainingError(f"Prepared dataset is missing columns: {missing}")
    mode = str(dataset["dataset_mode"].iloc[0])
    if small_test and mode != "toy_small_test":
        raise Stage2TrainingError("Small-test training requires a toy_small_test prepared dataset.")
    if not small_test and mode != "real_tcga_luad":
        raise Stage2TrainingError(
            "Refusing to treat toy outputs as real TCGA training. "
            "Run stage2_prepare_rnaseq_survival.py without --small-test after downloading RNA-seq."
        )
    if not paths.feature_list.exists():
        raise Stage2TrainingError(f"Feature list not found: {paths.feature_list}")
    feature_table = pd.read_csv(paths.feature_list)
    if "feature" not in feature_table:
        raise Stage2TrainingError("Stage 2 feature list has no 'feature' column.")
    genes = [gene for gene in feature_table["feature"].astype(str) if gene in dataset.columns]
    if not genes:
        raise Stage2TrainingError("Prepared dataset has no RNA candidate genes.")
    return dataset, genes


def _load_split(paths: Stage2Paths, seed: int, dataset: pd.DataFrame) -> pd.DataFrame:
    split_path = paths.split_file(seed)
    if not split_path.exists():
        raise Stage2TrainingError(f"Split file not found: {split_path}")
    split = pd.read_csv(split_path)
    required = {"patient_id", "split", "seed"}
    if not required.issubset(split.columns):
        raise Stage2TrainingError(f"Malformed split file: {split_path}")
    unknown = sorted(set(split["patient_id"]) - set(dataset["patient_id"]))
    if unknown:
        raise Stage2TrainingError(f"Split contains {len(unknown)} unknown patient IDs.")
    if set(split["split"]) != {"train", "validation", "test"}:
        raise Stage2TrainingError("Split file must contain train, validation, and test.")
    if split["patient_id"].duplicated().any():
        raise Stage2TrainingError("Split file contains duplicate patient IDs.")
    return split


def _stack(*arrays: np.ndarray) -> np.ndarray:
    return np.concatenate(arrays, axis=1)


def _subset(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    return frame.loc[frame["split"] == split].copy()


def _fit_models(
    train: pd.DataFrame,
    arrays: dict[str, dict[str, np.ndarray]],
    *,
    seed: int,
    small_test: bool,
) -> dict[str, Any]:
    epochs_linear = 70 if small_test else 180
    epochs_deep = 45 if small_test else 120
    forests = 60 if small_test else 160
    times = train["os_time_days"].to_numpy(dtype=float)
    events = train["os_event"].to_numpy(dtype=int)
    combined = _stack(arrays["train"]["clinical"], arrays["train"]["rna"])
    return {
        "clinical_only_cox": TorchCoxElasticNet(epochs=epochs_linear, seed=seed).fit(
            arrays["train"]["clinical"], times, events
        ),
        "rna_only_elasticnet_cox": TorchCoxElasticNet(
            alpha=2e-3,
            l1_ratio=0.70,
            epochs=epochs_linear,
            seed=seed,
        ).fit(arrays["train"]["rna"], times, events),
        "rna_clinical_elasticnet_cox": TorchCoxElasticNet(
            alpha=2e-3,
            l1_ratio=0.55,
            epochs=epochs_linear,
            seed=seed,
        ).fit(combined, times, events),
        "random_survival_forest": RandomSurvivalForestAdapter(
            n_estimators=forests,
            seed=seed,
        ).fit(combined, times, events),
        "deepsurv_rna_only": DeepSurvEstimator(
            epochs=epochs_deep,
            seed=seed,
        ).fit(arrays["train"]["rna"], times, events),
        "deepsurv_rna_clinical": DeepSurvEstimator(
            epochs=epochs_deep,
            seed=seed,
        ).fit(combined, times, events),
    }


def _features_for_model(model_name: str, arrays: dict[str, np.ndarray]) -> np.ndarray:
    if model_name == "clinical_only_cox":
        return arrays["clinical"]
    if model_name in {"rna_only_elasticnet_cox", "deepsurv_rna_only"}:
        return arrays["rna"]
    return _stack(arrays["clinical"], arrays["rna"])


def train_seed(
    root: str | Path = ".",
    *,
    seed: int = 42,
    small_test: bool = False,
) -> dict[str, Any]:
    """Train all Stage 2 baseline models for one frozen split seed."""

    if seed not in SEEDS:
        raise Stage2TrainingError(f"Seed must be one of {SEEDS}.")
    paths = Stage2Paths.from_root(root)
    dataset, candidate_genes = _load_dataset(paths, small_test=small_test)
    split = _load_split(paths, seed, dataset)
    cohort = dataset.merge(split[["patient_id", "split"]], on="patient_id", how="inner", validate="one_to_one")
    train = _subset(cohort, "train")
    if train["os_event"].sum() == 0:
        raise Stage2TrainingError("Training split contains no death events.")

    rna_preprocessor = RNATrainPreprocessor(
        top_variable_genes=60 if small_test else 3000,
    )
    clinical_preprocessor = ClinicalTrainPreprocessor()
    rna_preprocessor.fit(train[candidate_genes], patient_ids=train["patient_id"])
    clinical_preprocessor.fit(train[list(CLINICAL_COLUMNS)], patient_ids=train["patient_id"])

    split_frames = {
        name: _subset(cohort, name)
        for name in ("train", "validation", "test")
    }
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for name, frame in split_frames.items():
        arrays[name] = {
            "rna": rna_preprocessor.transform(frame[candidate_genes]).to_numpy(),
            "clinical": clinical_preprocessor.transform(frame[list(CLINICAL_COLUMNS)]).to_numpy(),
        }

    models = _fit_models(train, arrays, seed=seed, small_test=small_test)
    output_dir = paths.root / "outputs"
    checkpoints = output_dir / "checkpoints"
    tables = output_dir / "tables"
    logs = output_dir / "logs"
    checkpoints.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    prediction_rows: list[pd.DataFrame] = []
    backends: dict[str, str] = {}
    for model_name, model in models.items():
        backends[model_name] = model.backend_
        if isinstance(model, DeepSurvEstimator):
            torch.save(model.checkpoint(), checkpoints / f"stage2_{model_name}_seed{seed}.pt")
        else:
            with (checkpoints / f"stage2_{model_name}_seed{seed}.pkl").open("wb") as handle:
                pickle.dump(model, handle)
        for split_name, frame in split_frames.items():
            scores = model.predict_risk(_features_for_model(model_name, arrays[split_name]))
            prediction_rows.append(
                pd.DataFrame(
                    {
                        "patient_id": frame["patient_id"].to_numpy(),
                        "seed": seed,
                        "split": split_name,
                        "model": model_name,
                        "os_time_days": frame["os_time_days"].to_numpy(),
                        "os_event": frame["os_event"].to_numpy(),
                        "age": frame["age"].to_numpy(),
                        "male": frame["male"].to_numpy(),
                        "stage_numeric": frame["stage_numeric"].to_numpy(),
                        "risk_score": scores,
                    }
                )
            )

    predictions = pd.concat(prediction_rows, ignore_index=True)
    predictions.to_csv(tables / f"stage2_predictions_seed{seed}.csv", index=False)
    pd.DataFrame({"feature": rna_preprocessor.selected_genes_}).to_csv(
        paths.root / "data" / "metadata" / f"stage2_selected_features_seed{seed}.csv",
        index=False,
    )
    with (checkpoints / f"stage2_preprocessors_seed{seed}.pkl").open("wb") as handle:
        pickle.dump(
            {
                "rna": rna_preprocessor,
                "clinical": clinical_preprocessor,
            },
            handle,
        )
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_mode": str(dataset["dataset_mode"].iloc[0]),
        "seed": seed,
        "candidate_gene_count": len(candidate_genes),
        "selected_gene_count": len(rna_preprocessor.selected_genes_),
        "rna_preprocessor_fit_patient_ids": list(rna_preprocessor.fit_patient_ids_),
        "clinical_preprocessor_fit_patient_ids": list(clinical_preprocessor.fit_patient_ids_),
        "validation_patient_ids": split_frames["validation"]["patient_id"].tolist(),
        "test_patient_ids": split_frames["test"]["patient_id"].tolist(),
        "model_backends": backends,
        "no_leakage_rule": "all preprocessing fit on training split only",
    }
    manifest_path = logs / f"stage2_training_manifest_seed{seed}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train leakage-safe Stage 2 survival baselines.")
    parser.add_argument("--config", default="configs/base.yaml", help="Project YAML config path.")
    parser.add_argument("--root", default=".", help="Project root. Defaults to current directory.")
    parser.add_argument("--seed", type=int, choices=list(SEEDS), default=42)
    parser.add_argument("--small-test", action="store_true", help="Train on prepared toy_small_test data.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config_path = Path(args.root) / args.config
    if not config_path.exists():
        parser.error(f"Config file not found: {config_path}")
    try:
        result = train_seed(args.root, seed=args.seed, small_test=args.small_test)
    except (FileNotFoundError, Stage2TrainingError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"Stage 2 training failed: {exc}\n")
    print(json.dumps(result, indent=2))
    return 0
