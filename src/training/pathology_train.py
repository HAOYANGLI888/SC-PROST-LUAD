"""Stage 6A pathology MIL proof-of-concept training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models.clinical_pathology_fusion import ClinicalPathologyFusionSurvival
from models.pathology_attention_mil import AttentionMILSurvival
from models.pathology_gated_mil import GatedAttentionMILSurvival
from models.pathology_survival_heads import CoxSurvivalHead
from models.survival_losses import cox_ph_loss


@dataclass
class PathologyTrainingResult:
    predictions: pd.DataFrame
    attention: np.ndarray
    coordinates: np.ndarray
    backend: str


def _load_patient_bags(feature_summary_path: Path) -> tuple[pd.DataFrame, list[torch.Tensor], list[torch.Tensor]]:
    summary = pd.read_csv(feature_summary_path)
    summary = summary.loc[summary["feature_status"].isin(["extracted", "skipped_existing"])].copy()
    if summary.empty:
        raise RuntimeError(f"No extracted WSI feature bags found in {feature_summary_path}")
    rows = []
    bags = []
    coordinates = []
    for patient_id, patient_slides in summary.groupby("patient_id", sort=True):
        patient_bags = []
        patient_coordinates = []
        for row in patient_slides.to_dict("records"):
            artifact = torch.load(row["feature_path"], map_location="cpu", weights_only=False)
            patient_bags.append(artifact["features"].float())
            patient_coordinates.append(artifact["coordinates"].numpy())
        representative = patient_slides.sort_values("file_id").iloc[0].to_dict()
        representative["patient_id"] = patient_id
        representative["aggregated_slide_count"] = len(patient_slides)
        rows.append(representative)
        bags.append(torch.cat(patient_bags, dim=0))
        coordinates.append(np.vstack(patient_coordinates))
    return pd.DataFrame(rows), bags, coordinates


def _clinical_tensor(frame: pd.DataFrame) -> torch.Tensor:
    clinical = frame[["age", "male", "stage_numeric"]].apply(pd.to_numeric, errors="coerce")
    clinical = clinical.fillna(clinical.median()).fillna(0.0)
    values = clinical.to_numpy(dtype=np.float32)
    scales = values.std(axis=0)
    scales[scales == 0] = 1.0
    return torch.as_tensor((values - values.mean(axis=0)) / scales, dtype=torch.float32)


def _train(model, times: torch.Tensor, events: torch.Tensor, forward, *, epochs: int, seed: int) -> tuple[np.ndarray, list[torch.Tensor]]:
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.008, weight_decay=1e-4)
    attention: list[torch.Tensor] = []
    for _ in range(epochs):
        optimizer.zero_grad()
        scores, attention = forward()
        loss = cox_ph_loss(scores, times, events)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        scores, attention = forward()
    return scores.numpy(), attention


def train_pathology_models(
    root: str | Path = ".",
    *,
    small_test: bool = False,
    smallset: bool = False,
    epochs: int | None = None,
) -> PathologyTrainingResult:
    """Train Stage 6A proof-of-concept models; never claim formal validation."""

    if not (small_test or smallset):
        raise ValueError("Stage 6A training requires --small-test or --smallset.")
    root = Path(root).resolve()
    metadata = root / "data" / "metadata"
    summary_path = (
        metadata / "stage6a_small_test" / "wsi_feature_summary.csv"
        if small_test
        else metadata / "stage6a_wsi_feature_summary.csv"
    )
    frame, bags, coordinates = _load_patient_bags(summary_path)
    times = torch.as_tensor(frame["os_time_days"].to_numpy(dtype=np.float32))
    events = torch.as_tensor(frame["os_event"].to_numpy(dtype=np.float32))
    clinical = _clinical_tensor(frame)
    input_dim = int(bags[0].shape[1])
    epochs = epochs or (24 if small_test else 50)
    seed = 42
    rows = []

    clinical_model = CoxSurvivalHead(3)
    clinical_scores, _ = _train(
        clinical_model, times, events, lambda: (clinical_model(clinical), []), epochs=epochs, seed=seed
    )
    rows.extend({"patient_id": pid, "model_name": "clinical_only_cox", "risk_score": score} for pid, score in zip(frame["patient_id"], clinical_scores))

    attention_model = AttentionMILSurvival(input_dim)
    attention_scores, attention_values = _train(
        attention_model, times, events, lambda: attention_model(bags), epochs=epochs, seed=seed
    )
    rows.extend({"patient_id": pid, "model_name": "pathology_attention_mil_cox", "risk_score": score} for pid, score in zip(frame["patient_id"], attention_scores))

    gated_model = GatedAttentionMILSurvival(input_dim)
    gated_scores, _ = _train(gated_model, times, events, lambda: gated_model(bags), epochs=epochs, seed=seed)
    rows.extend({"patient_id": pid, "model_name": "pathology_gated_mil_cox", "risk_score": score} for pid, score in zip(frame["patient_id"], gated_scores))

    fusion_model = ClinicalPathologyFusionSurvival(input_dim)
    fusion_scores, _ = _train(
        fusion_model, times, events, lambda: fusion_model(bags, clinical), epochs=epochs, seed=seed
    )
    rows.extend({"patient_id": pid, "model_name": "clinical_pathology_fusion_cox", "risk_score": score} for pid, score in zip(frame["patient_id"], fusion_scores))

    predictions = pd.DataFrame(rows).merge(
        frame[["patient_id", "os_time_days", "os_event", "age", "male", "stage_numeric"]],
        on="patient_id",
        how="left",
        validate="many_to_one",
    )
    predictions["dataset_mode"] = "toy_small_test" if small_test else "real_smallset_pipeline_only"
    output = root / "data" / "processed" / ("stage6a_small_test" if small_test else "") / "pathology_mil_predictions.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output, index=False)
    checkpoint = root / "outputs" / "checkpoints" / ("stage6a_small_test" if small_test else "") / "pathology_attention_mil.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": attention_model.state_dict(), "input_dim": input_dim, "dataset_mode": predictions["dataset_mode"].iloc[0]}, checkpoint)
    return PathologyTrainingResult(
        predictions=predictions,
        attention=attention_values[0].detach().numpy(),
        coordinates=coordinates[0],
        backend="pytorch_attention_mil_cox_breslow",
    )
