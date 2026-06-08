"""Train Stage 6A WSI pilot MIL models with patient-level splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluation.pathology_pilot_metrics import overfitting_diagnostics, summarize_pilot_predictions
from models.clinical_pathology_fusion import ClinicalPathologyFusionSurvival
from models.pathology_attention_mil import AttentionMILSurvival
from models.pathology_gated_mil import GatedAttentionMILSurvival
from models.pathology_survival_heads import CoxSurvivalHead
from models.survival_losses import cox_ph_loss


class ClinicalPathologyAttentionFusion(torch.nn.Module):
    def __init__(self, pathology_dim: int, clinical_dim: int = 3, hidden_dim: int = 64) -> None:
        super().__init__()
        self.project = torch.nn.Sequential(torch.nn.Linear(pathology_dim, hidden_dim), torch.nn.Tanh())
        self.attention = torch.nn.Linear(hidden_dim, 1)
        self.head = CoxSurvivalHead(hidden_dim + clinical_dim)

    def encode_bag(self, bag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.project(bag)
        attention = torch.softmax(self.attention(hidden).reshape(-1), dim=0)
        return torch.sum(attention[:, None] * hidden, dim=0), attention

    def forward(self, bags: list[torch.Tensor], clinical: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        encoded = [self.encode_bag(bag) for bag in bags]
        pooled = torch.stack([item[0] for item in encoded])
        return self.head(torch.cat([pooled, clinical], dim=1)), [item[1] for item in encoded]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Stage 6A WSI pilot Cox-loss MIL models.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--root", default=".")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    return parser


def _load_pilot(root: Path, device: torch.device):
    summary = pd.read_csv(root / "outputs" / "tables" / "stage6a_wsi_pilot_feature_summary.csv")
    summary = summary.loc[summary["feature_status"].isin(["extracted", "skipped_existing"])].copy()
    if summary.empty:
        raise RuntimeError("No WSI pilot feature bags available.")
    rows, bags, coords = [], [], []
    for patient_id, group in summary.groupby("patient_id", sort=True):
        patient_bags, patient_coords = [], []
        for row in group.sort_values("file_id").to_dict("records"):
            artifact = torch.load(row["feature_path"], map_location="cpu", weights_only=False)
            patient_bags.append(artifact["features"].float())
            patient_coords.append(artifact["coordinates"].numpy())
        representative = group.sort_values("file_id").iloc[0].to_dict()
        representative["patient_id"] = patient_id
        rows.append(representative)
        bags.append(torch.cat(patient_bags, dim=0).to(device))
        coords.append(np.vstack(patient_coords))
    frame = pd.DataFrame(rows).reset_index(drop=True)
    return frame, bags, coords


def _split(frame: pd.DataFrame, seed: int) -> pd.Series:
    ids = np.arange(len(frame))
    events = frame["os_event"].astype(int).to_numpy()
    train_idx, temp_idx = train_test_split(ids, test_size=0.4, random_state=seed, stratify=events)
    temp_events = events[temp_idx]
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=seed, stratify=temp_events)
    split = pd.Series("train", index=frame.index, dtype=object)
    split.iloc[val_idx] = "validation"
    split.iloc[test_idx] = "test"
    return split


def _clinical(frame: pd.DataFrame, split: pd.Series, device: torch.device) -> torch.Tensor:
    values = frame[["age", "male", "stage_numeric"]].apply(pd.to_numeric, errors="coerce")
    train_values = values.loc[split == "train"]
    med = train_values.median().fillna(0.0)
    values = values.fillna(med).fillna(0.0).to_numpy(dtype=np.float32)
    train = values[split.to_numpy() == "train"]
    mean, std = train.mean(axis=0), train.std(axis=0)
    std[std == 0] = 1.0
    return torch.as_tensor((values - mean) / std, dtype=torch.float32, device=device)


def _subset(items, indices):
    return [items[int(index)] for index in indices]


def _train_model(model, forward_train, forward_all, times_train, events_train, *, epochs: int, seed: int):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=0.006, weight_decay=1e-4)
    attention = []
    for _ in range(epochs):
        opt.zero_grad()
        scores, attention = forward_train()
        loss = cox_ph_loss(scores, times_train, events_train)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        scores, attention = forward_all()
    return scores.detach().cpu().numpy(), attention


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable.")
    root = Path(args.root).resolve()
    device = torch.device(args.device)
    started = perf_counter()
    frame, bags, coords = _load_pilot(root, device)
    split = _split(frame, args.seed)
    frame["split"] = split
    split_table = root / "data" / "metadata" / "stage6a_wsi_pilot_split_seed42.csv"
    split_table.parent.mkdir(parents=True, exist_ok=True)
    frame[["patient_id", "file_id", "os_time_days", "os_event", "stage_numeric", "split"]].to_csv(split_table, index=False)
    clinical = _clinical(frame, split, device)
    times_all = torch.as_tensor(frame["os_time_days"].to_numpy(dtype=np.float32), device=device)
    events_all = torch.as_tensor(frame["os_event"].to_numpy(dtype=np.float32), device=device)
    train_idx = np.where(split.to_numpy() == "train")[0]
    input_dim = int(bags[0].shape[1])
    models = {
        "clinical_only_cox": CoxSurvivalHead(3).to(device),
        "pathology_attention_mil_cox": AttentionMILSurvival(input_dim).to(device),
        "pathology_gated_mil_cox": GatedAttentionMILSurvival(input_dim).to(device),
        "clinical_pathology_attention_fusion_cox": ClinicalPathologyAttentionFusion(input_dim).to(device),
        "clinical_pathology_gated_fusion_cox": ClinicalPathologyFusionSurvival(input_dim).to(device),
    }
    predictions = []
    attention_example = None
    checkpoint_dir = root / "outputs" / "checkpoints" / "stage6a_wsi_pilot"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        if name == "clinical_only_cox":
            scores, attention = _train_model(
                model,
                lambda m=model: (m(clinical[train_idx]), []),
                lambda m=model: (m(clinical), []),
                times_all[train_idx],
                events_all[train_idx],
                epochs=args.epochs,
                seed=args.seed,
            )
        elif "fusion" in name:
            scores, attention = _train_model(
                model,
                lambda m=model: m(_subset(bags, train_idx), clinical[train_idx]),
                lambda m=model: m(bags, clinical),
                times_all[train_idx],
                events_all[train_idx],
                epochs=args.epochs,
                seed=args.seed,
            )
        else:
            scores, attention = _train_model(
                model,
                lambda m=model: m(_subset(bags, train_idx)),
                lambda m=model: m(bags),
                times_all[train_idx],
                events_all[train_idx],
                epochs=args.epochs,
                seed=args.seed,
            )
        torch.save({"state_dict": model.state_dict(), "input_dim": input_dim, "model_name": name}, checkpoint_dir / f"{name}.pt")
        for patient_id, score, split_name, time, event in zip(frame["patient_id"], scores, frame["split"], frame["os_time_days"], frame["os_event"]):
            predictions.append({"patient_id": patient_id, "model_name": name, "risk_score": float(score), "split": split_name, "os_time_days": time, "os_event": event})
        if attention_example is None and attention:
            attention_example = (coords[0], attention[0].detach().cpu().numpy(), name)
    pred = pd.DataFrame(predictions).merge(
        frame[["patient_id", "age", "male", "stage_numeric"]], on="patient_id", how="left", validate="many_to_one"
    )
    pred["dataset_mode"] = "stage6a_wsi_gpu_pilot_feasibility"
    processed = root / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    pred.to_csv(processed / "stage6a_wsi_pilot_predictions.csv", index=False)
    if attention_example is not None:
        np.savez(processed / "stage6a_wsi_pilot_attention_example.npz", coordinates=attention_example[0], attention=attention_example[1], model_name=attention_example[2])
    performance = summarize_pilot_predictions(pred)
    diagnostics = overfitting_diagnostics(performance)
    tables = root / "outputs" / "tables"
    logs = root / "outputs" / "logs"
    tables.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    performance.to_csv(tables / "stage6a_wsi_pilot_model_performance.csv", index=False)
    diagnostics.to_csv(tables / "stage6a_wsi_pilot_overfitting_diagnostics.csv", index=False)
    runtime = perf_counter() - started
    log = {
        "status": "passed",
        "patients": int(frame["patient_id"].nunique()),
        "epochs": args.epochs,
        "device": str(device),
        "training_runtime_seconds": round(runtime, 4),
        "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 3) if device.type == "cuda" else 0.0,
    }
    (logs / "stage6a_wsi_pilot_training.log").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(json.dumps(log, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

