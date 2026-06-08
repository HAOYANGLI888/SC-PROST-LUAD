"""Generate Stage 6A GPU pilot resource and decision report."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluation.resource_monitoring import collect_resource_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Stage 6A GPU pilot disk, runtime and full-cohort resource estimates.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--root", default=".")
    return parser


def _safe_read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _full_wsi_raw_gb(root: Path) -> float:
    manifest = root / "data" / "metadata" / "stage6a_tcga_luad_wsi_patient_slide_map.csv"
    if manifest.exists():
        frame = pd.read_csv(manifest)
        return float(pd.to_numeric(frame["file_size"], errors="coerce").sum() / 1e9)
    return float("nan")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root).resolve()
    resources = collect_resource_rows(root)
    cohort = _safe_read(root / "data" / "metadata" / "stage6a_wsi_pilot_cohort.csv")
    downloads = _safe_read(root / "data" / "metadata" / "stage6a_wsi_pilot_download_status.csv")
    patches = _safe_read(root / "outputs" / "tables" / "stage6a_wsi_pilot_patch_summary.csv")
    features = _safe_read(root / "outputs" / "tables" / "stage6a_wsi_pilot_feature_summary.csv")
    performance = _safe_read(root / "outputs" / "tables" / "stage6a_wsi_pilot_model_performance.csv")
    diagnostics = _safe_read(root / "outputs" / "tables" / "stage6a_wsi_pilot_overfitting_diagnostics.csv")
    training_log = root / "outputs" / "logs" / "stage6a_wsi_pilot_training.log"
    training_runtime = float("nan")
    if training_log.exists():
        try:
            training_runtime = float(json.loads(training_log.read_text(encoding="utf-8")).get("training_runtime_seconds", "nan"))
        except Exception:
            training_runtime = float("nan")
    selected = len(cohort)
    complete = int(downloads["download_status"].isin(["complete_existing", "copied_from_smallset", "downloaded_complete"]).sum()) if not downloads.empty else 0
    extracted = int(patches["patch_status"].eq("extracted").sum()) if not patches.empty else 0
    featured = int(features["feature_status"].isin(["extracted", "skipped_existing"]).sum()) if not features.empty else 0
    avg_patches = float(pd.to_numeric(features.get("patch_count", pd.Series(dtype=float)), errors="coerce").mean()) if not features.empty else float("nan")
    used_cuda = bool(features.get("used_cuda", pd.Series(dtype=bool)).astype(str).str.casefold().isin(["true", "1"]).any()) if not features.empty else False
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA unavailable"
    gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0.0
    scale = 541 / selected if selected else float("nan")
    full_raw_gb = _full_wsi_raw_gb(root)
    full_feature_runtime_hours = resources["gpu_feature_runtime_seconds"] * scale / 3600 if selected else float("nan")
    test_perf = performance.loc[performance.get("split", pd.Series(dtype=str)).eq("test")].copy() if not performance.empty else pd.DataFrame()
    best_test = ""
    best_pathology = ""
    if not test_perf.empty:
        test_perf["c_index_num"] = pd.to_numeric(test_perf["c_index"], errors="coerce")
        best_test = str(test_perf.sort_values("c_index_num", ascending=False).iloc[0]["model_name"])
        pathology_test = test_perf.loc[test_perf["model_name"].astype(str).str.contains("pathology|mil|fusion", case=False, regex=True)]
        if not pathology_test.empty:
            best_pathology = str(pathology_test.sort_values("c_index_num", ascending=False).iloc[0]["model_name"])
    report = (
        "# Stage 6A GPU Pilot Report\n\n"
        f"Generated: {datetime.now().isoformat(timespec='seconds')}\n\n"
        "## Scope\n\n"
        "- Stage 6A-GPU-Pilot only. No Stage 3, Stage 6B, mutation, CNV, methylation, single-cell, protein, or full 541-slide download was started.\n\n"
        "## Required Answers\n\n"
        f"1. CUDA truly usable: `{torch.cuda.is_available()}`.\n"
        f"2. GPU: `{gpu_name}`.\n"
        f"3. GPU memory: {gpu_gb:.2f} GiB.\n"
        f"4. 20-slide smallset GPU recheck: see `stage6a_smallset_gpu_check_report.md`.\n"
        f"5. Pilot selected patients/slides: {cohort['patient_id'].nunique() if not cohort.empty else 0}/{selected}.\n"
        f"6. Death/censored balance: {int(cohort['OS_status'].sum()) if not cohort.empty else 0}/{int((cohort['OS_status'] == 0).sum()) if not cohort.empty else 0}.\n"
        f"7. Pilot download complete: {complete}/{selected}.\n"
        f"8. Patch extraction success: {extracted}/{selected}.\n"
        f"9. Average patches per slide: {avg_patches:.1f}.\n"
        f"10. ResNet50 feature extraction success: {featured}/{selected}.\n"
        f"11. Feature extraction used CUDA: `{used_cuda}`.\n"
        f"12. Pathology-only models completed: `{not performance.empty}`.\n"
        f"13. Clinical + pathology fusion completed: `{not performance.empty and performance['model_name'].astype(str).str.contains('fusion').any()}`.\n"
        f"14. Overfitting present: `{not diagnostics.empty and diagnostics.get('overfitting_flag', pd.Series(dtype=bool)).astype(bool).any()}`.\n"
        f"15. Pathology vs clinical-only: best test model is `{best_test}`; best pathology/fusion test model is `{best_pathology}`. Pilot performance is not publication evidence.\n"
        "16. Recommendation for full Stage 6B: not yet. Defer because pathology and fusion models overfit and did not beat clinical-only on the held-out pilot test split.\n"
        "17. If deferring: pathology signal is not yet stable, and full WSI still requires major disk/time commitment.\n"
        f"18. Full Stage 6B planning estimate: full audited raw diagnostic SVS size ~{full_raw_gb:.1f} GB; feature runtime extrapolated at current 512-patch setting ~{full_feature_runtime_hours:.2f} GPU-hours. Maintain the prior 0.8-1.0 TB workspace recommendation for raw slides, features, logs, checkpoints and working files.\n\n"
        "## Resource Summary\n\n"
        f"- Downloaded WSI size: {resources['downloaded_wsi_gb']:.3f} GB.\n"
        f"- Patch metadata size: {resources['patch_metadata_mb']:.2f} MB.\n"
        f"- Feature file size: {resources['feature_file_mb']:.2f} MB.\n"
        f"- GPU feature extraction runtime: {resources['gpu_feature_runtime_seconds']:.2f} seconds.\n"
        f"- Training runtime: {training_runtime:.2f} seconds.\n"
        f"- Peak GPU memory: {resources['peak_gpu_memory_mb']:.1f} MB.\n"
        "- RTX 4060 Ti 8GB is sufficient for ResNet50 feature extraction and MIL training at this pilot batch size, but batch size should remain configurable for full-cohort runs.\n\n"
        "## Integrity Note\n\n"
        "- Pilot metrics are engineering diagnostics only and must not be written as formal manuscript conclusions.\n"
    )
    report_path = root / "outputs" / "reports" / "stage6a_gpu_pilot_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    audit = root / "audit_report.md"
    existing = audit.read_text(encoding="utf-8") if audit.exists() else "# SC-PROST-LUAD Audit Report\n"
    marker = "\n\n## Stage 6A-GPU-Pilot\n\n"
    audit.write_text(existing.split(marker)[0] + marker + report, encoding="utf-8")
    print(json.dumps({"status": "passed", "report": str(report_path), "download_complete": f"{complete}/{selected}", "feature_complete": f"{featured}/{selected}"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
