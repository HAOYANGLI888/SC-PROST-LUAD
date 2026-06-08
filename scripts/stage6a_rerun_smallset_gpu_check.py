"""Rerun the validated 20-slide smallset on CUDA for Stage 6A-GPU-Pilot."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluation.pathology_metrics import pathology_survival_metrics
from models.pathology_attention_mil import AttentionMILSurvival
from models.pathology_gated_mil import GatedAttentionMILSurvival
from models.survival_losses import cox_ph_loss
from pathology.patch_feature_extraction import extract_features_from_patch_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Re-extract/check 20-slide smallset ResNet50 features on CUDA and train 5-epoch MIL probes.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--root", default=".")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    return parser


def _load_bags(summary_path: Path, device: torch.device):
    summary = pd.read_csv(summary_path)
    summary = summary.loc[summary["feature_status"].isin(["extracted", "skipped_existing"])].copy()
    rows, bags, coords = [], [], []
    for _, row in summary.sort_values("patient_id").iterrows():
        artifact = torch.load(row["feature_path"], map_location="cpu", weights_only=False)
        rows.append(row.to_dict())
        bags.append(artifact["features"].float().to(device))
        coords.append(artifact["coordinates"].cpu().numpy())
    return pd.DataFrame(rows), bags, coords


def _train_probe(model, bags, times, events, epochs: int):
    opt = torch.optim.Adam(model.parameters(), lr=0.008, weight_decay=1e-4)
    attention = []
    for _ in range(epochs):
        opt.zero_grad()
        scores, attention = model(bags)
        loss = cox_ph_loss(scores, times, events)
        loss.backward()
        opt.step()
    with torch.no_grad():
        scores, attention = model(bags)
    return scores.detach().cpu().numpy(), attention


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; Stage 6A-GPU-Pilot smallset check stopped.")
    root = Path(args.root).resolve()
    patch_summary = root / "data" / "metadata" / "stage6a_wsi_patch_summary.csv"
    if not patch_summary.exists():
        raise SystemExit(f"Smallset patch summary missing: {patch_summary}")
    feature_root = root / "data" / "processed" / "wsi_smallset_gpu_features"
    started = perf_counter()
    torch.cuda.reset_peak_memory_stats()
    features = extract_features_from_patch_summary(
        patch_summary,
        root / "data" / "processed" / "wsi_patches",
        feature_root,
        backend="resnet50",
        device="cuda",
        batch_size=args.batch_size,
        allow_handcrafted_fallback=False,
    )
    features["used_cuda"] = True
    feature_runtime = perf_counter() - started
    feature_summary = root / "outputs" / "tables" / "stage6a_smallset_gpu_feature_summary.csv"
    feature_summary.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(feature_summary, index=False)
    device = torch.device("cuda")
    frame, bags, _ = _load_bags(feature_summary, device)
    times = torch.as_tensor(frame["os_time_days"].to_numpy(dtype="float32"), device=device)
    events = torch.as_tensor(frame["os_event"].to_numpy(dtype="float32"), device=device)
    input_dim = int(bags[0].shape[1])
    train_started = perf_counter()
    attention_scores, _ = _train_probe(AttentionMILSurvival(input_dim).to(device), bags, times, events, args.epochs)
    gated_scores, _ = _train_probe(GatedAttentionMILSurvival(input_dim).to(device), bags, times, events, args.epochs)
    train_runtime = perf_counter() - train_started
    rows = []
    for name, scores in (("smallset_attention_mil_cuda_probe", attention_scores), ("smallset_gated_mil_cuda_probe", gated_scores)):
        rows.append(
            {
                "model_name": name,
                **pathology_survival_metrics(frame["os_time_days"], frame["os_event"], scores),
                "patient_count": frame["patient_id"].nunique(),
                "feature_runtime_seconds": round(feature_runtime, 4),
                "training_runtime_seconds": round(train_runtime, 4),
                "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 3),
                "used_cuda": True,
                "interpretation": "20-slide pipeline verification only; not a scientific performance estimate",
            }
        )
    table = pd.DataFrame(rows)
    out = root / "outputs" / "tables" / "stage6a_smallset_gpu_check.csv"
    table.to_csv(out, index=False)
    report = (
        "# Stage 6A Smallset GPU Check\n\n"
        f"Generated: {datetime.now().isoformat(timespec='seconds')}\n\n"
        f"- CUDA available: `{torch.cuda.is_available()}`.\n"
        f"- GPU: `{torch.cuda.get_device_name(0)}`.\n"
        f"- Smallset slides with CUDA features: {int(features['feature_status'].isin(['extracted', 'skipped_existing']).sum())}/{len(features)}.\n"
        f"- Feature runtime: {feature_runtime:.2f} seconds.\n"
        f"- Training runtime for 5-epoch probes: {train_runtime:.2f} seconds.\n"
        f"- Peak GPU memory: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB.\n"
        "- This is pipeline verification only and must not be used as a scientific result.\n"
    )
    report_path = root / "outputs" / "reports" / "stage6a_smallset_gpu_check_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(json.dumps({"status": "passed", "slides_with_cuda_features": int(features['feature_status'].isin(['extracted', 'skipped_existing']).sum()), "report": str(report_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

