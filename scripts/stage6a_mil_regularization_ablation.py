"""Regularization ablation for Stage 6A MIL overfitting diagnostics."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from time import perf_counter

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluation.survival_metrics import concordance_index
from models.pathology_survival_heads import CoxSurvivalHead
from models.survival_losses import cox_ph_loss
from pathology.diagnostics_utils import load_patient_feature_frame, split_indices


class DropoutAttentionMIL(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.project = torch.nn.Sequential(torch.nn.Linear(input_dim, hidden_dim), torch.nn.Tanh(), torch.nn.Dropout(dropout))
        self.attention = torch.nn.Linear(hidden_dim, 1)
        self.head = CoxSurvivalHead(hidden_dim)

    def forward(self, bags: list[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]]:
        encoded = []
        attention_values = []
        for bag in bags:
            hidden = self.project(bag)
            attention = torch.softmax(self.attention(hidden).reshape(-1), dim=0)
            encoded.append(torch.sum(attention[:, None] * hidden, dim=0))
            attention_values.append(attention)
        return self.head(torch.stack(encoded)), attention_values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run validation-selected MIL regularization ablation on the fixed WSI pilot.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--root", default=".")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    return parser


def _load_bags(root: Path, device: torch.device):
    frame, _, _ = load_patient_feature_frame(root)
    summary = pd.read_csv(root / "outputs" / "tables" / "stage6a_wsi_pilot_feature_summary.csv")
    summary = summary.loc[summary["feature_status"].isin(["extracted", "skipped_existing"])].sort_values("patient_id")
    bags = []
    for row in summary.to_dict("records"):
        artifact = torch.load(row["feature_path"], map_location="cpu", weights_only=False)
        bags.append(artifact["features"].float().to(device))
    return frame.sort_values("patient_id").reset_index(drop=True), bags


def _subset(values, idx):
    return [values[int(index)] for index in idx]


def _risk(model, bags):
    model.eval()
    with torch.no_grad():
        scores, _ = model(bags)
    return scores.detach().cpu().numpy()


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable.")
    root = Path(args.root).resolve()
    resolved_device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    device = torch.device(resolved_device)
    frame, bags = _load_bags(root, device)
    idx = split_indices(frame)
    times = torch.as_tensor(frame["os_time_days"].to_numpy(dtype=np.float32), device=device)
    events = torch.as_tensor(frame["os_event"].to_numpy(dtype=np.float32), device=device)
    np_times = frame["os_time_days"].to_numpy(dtype=float)
    np_events = frame["os_event"].to_numpy(dtype=int)
    input_dim = int(bags[0].shape[1])
    grid = list(
        itertools.product(
            [5, 10, 20, 30],
            [0.25, 0.5, 0.7],
            [1e-4, 1e-3, 1e-2],
            [1e-4, 3e-4, 1e-3],
            [64, 128],
            [3, 5],
        )
    )
    rows = []
    best = None
    started = perf_counter()
    for combo_index, (epochs, dropout, weight_decay, learning_rate, hidden_dim, patience) in enumerate(grid, start=1):
        torch.manual_seed(42)
        model = DropoutAttentionMIL(input_dim, hidden_dim, dropout).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        best_state = None
        best_val_loss = float("inf")
        stale = 0
        epochs_run = 0
        for epoch in range(epochs):
            model.train()
            opt.zero_grad()
            train_scores, _ = model(_subset(bags, idx["train"]))
            train_loss = cox_ph_loss(train_scores, times[idx["train"]], events[idx["train"]])
            train_loss.backward()
            opt.step()
            epochs_run += 1
            model.eval()
            with torch.no_grad():
                val_scores, _ = model(_subset(bags, idx["validation"]))
                val_loss = float(cox_ph_loss(val_scores, times[idx["validation"]], events[idx["validation"]]).detach().cpu())
            if val_loss + 1e-6 < best_val_loss:
                best_val_loss = val_loss
                best_state = deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
            if stale >= patience:
                break
        if best_state is not None:
            model.load_state_dict(best_state)
        train_risk = _risk(model, _subset(bags, idx["train"]))
        val_risk = _risk(model, _subset(bags, idx["validation"]))
        train_c = concordance_index(np_times[idx["train"]], np_events[idx["train"]], train_risk)
        val_c = concordance_index(np_times[idx["validation"]], np_events[idx["validation"]], val_risk)
        row = {
            "combo_index": combo_index,
            "epochs": epochs,
            "dropout": dropout,
            "weight_decay": weight_decay,
            "learning_rate": learning_rate,
            "hidden_dim": hidden_dim,
            "early_stopping_patience": patience,
            "epochs_run": epochs_run,
            "train_c_index": train_c,
            "validation_c_index": val_c,
            "test_c_index": np.nan,
            "train_test_gap": np.nan,
            "test_evaluated": False,
        }
        rows.append(row)
        if best is None or val_c > best["validation_c_index"]:
            best = {**row, "state_dict": deepcopy(model.state_dict()), "model": model}
    assert best is not None
    best_model = DropoutAttentionMIL(input_dim, int(best["hidden_dim"]), float(best["dropout"])).to(device)
    best_model.load_state_dict(best["state_dict"])
    test_risk = _risk(best_model, _subset(bags, idx["test"]))
    test_c = concordance_index(np_times[idx["test"]], np_events[idx["test"]], test_risk)
    for row in rows:
        if row["combo_index"] == best["combo_index"]:
            row["test_c_index"] = test_c
            row["train_test_gap"] = row["train_c_index"] - test_c
            row["test_evaluated"] = True
    table = pd.DataFrame(rows)
    tables = root / "outputs" / "tables"
    figs = root / "outputs" / "figures"
    reports = root / "outputs" / "reports"
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    table.to_csv(tables / "stage6a_mil_regularization_ablation.csv", index=False)
    pivot = table.groupby(["dropout", "weight_decay"])["validation_c_index"].max().unstack()
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    image = ax.imshow(pivot.to_numpy(), cmap="viridis", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(pivot.columns)), [str(value) for value in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), [str(value) for value in pivot.index])
    ax.set_xlabel("Weight decay")
    ax.set_ylabel("Dropout")
    ax.set_title("Best validation C-index by regularization")
    fig.colorbar(image, ax=ax, label="Validation C-index")
    fig.tight_layout()
    fig.savefig(figs / "stage6a_mil_overfitting_heatmap.png", dpi=180)
    plt.close(fig)
    selected = table.loc[table["test_evaluated"]].iloc[0]
    report = (
        "# Stage 6A MIL Regularization Ablation Report\n\n"
        f"Generated: {datetime.now().isoformat(timespec='seconds')}\n\n"
        f"- Grid combinations evaluated on validation: {len(table)}.\n"
        f"- Test evaluated once for validation-selected combo: #{int(selected['combo_index'])}.\n"
        f"- Selected train/validation/test C-index: {selected['train_c_index']:.3f}/{selected['validation_c_index']:.3f}/{selected['test_c_index']:.3f}.\n"
        f"- Selected train-test gap: {selected['train_test_gap']:.3f}.\n"
        + (
            "- Overfitting remains severe after regularization; current WSI MIL mainline should be paused.\n"
            if selected["train_test_gap"] > 0.30 or selected["test_c_index"] < 0.58
            else "- Regularization reduced overfitting enough to justify cautious follow-up, but this is still pilot-only evidence.\n"
        )
        + f"- Runtime seconds: {perf_counter() - started:.1f}.\n"
    )
    (reports / "stage6a_mil_regularization_ablation_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"status": "passed", "combinations": len(table), "selected_combo": int(selected["combo_index"]), "selected_test_c": float(selected["test_c_index"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
