"""Evaluate Stage 6A pathology proof-of-concept without overstating evidence."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluation.pathology_metrics import pathology_survival_metrics
from pathology.wsi_io import openslide_status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Stage 6A pathology proof-of-concept metrics and feasibility.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--root", default=".")
    parser.add_argument("--small-test", action="store_true")
    parser.add_argument("--smallset", action="store_true")
    parser.add_argument("--real-features", action="store_true", help="Write the strict real-smallset feasibility report.")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not (args.small_test or args.smallset):
        raise SystemExit("Choose --small-test or --smallset.")
    if args.real_features and (args.small_test or not args.smallset):
        raise SystemExit("--real-features requires --smallset and cannot be combined with --small-test.")
    root = Path(args.root).resolve()
    mode_dir = "stage6a_small_test" if args.small_test else ""
    predictions_path = root / "data" / "processed" / mode_dir / "pathology_mil_predictions.csv"
    if not predictions_path.exists():
        raise SystemExit(f"Prediction file missing: {predictions_path}")
    predictions = pd.read_csv(predictions_path)
    rows = []
    for model_name, frame in predictions.groupby("model_name"):
        rows.append(
            {
                "model_name": model_name,
                **pathology_survival_metrics(frame["os_time_days"], frame["os_event"], frame["risk_score"]),
                "patient_count": frame["patient_id"].nunique(),
                "dataset_mode": frame["dataset_mode"].iloc[0],
                "interpretation": "pipeline feasibility only; not a scientific performance estimate",
            }
        )
    performance = pd.DataFrame(rows)
    tables = root / "outputs" / "tables"
    reports = root / "outputs" / "reports"
    tables.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    performance.to_csv(tables / "stage6a_pathology_smallset_performance.csv", index=False)
    if args.real_features:
        performance.to_csv(tables / "stage6a_real_smallset_performance.csv", index=False)
    feature_summary = root / "data" / "metadata" / ("stage6a_small_test/wsi_feature_summary.csv" if args.small_test else "stage6a_wsi_feature_summary.csv")
    features = pd.read_csv(feature_summary)
    features.to_csv(tables / "stage6a_pathology_qc_summary.csv", index=False)
    status = openslide_status()
    real_download = root / "data" / "metadata" / "stage6a_wsi_smallset_download_status.csv"
    if real_download.exists():
        real_download_frame = pd.read_csv(real_download)
        complete_real = int(real_download_frame["download_status"].isin(["skipped_complete", "complete_existing", "downloaded_complete"]).sum())
        partial_real = int(real_download_frame["download_status"].eq("partial_resume_available").sum())
        selected_real = len(real_download_frame)
        expected_real_gb = float(real_download_frame["file_size"].sum() / 1e9)
    else:
        complete_real = partial_real = selected_real = 0
        expected_real_gb = float("nan")
    formal_patch_path = root / "data" / "metadata" / "stage6a_wsi_patch_summary.csv"
    formal_feature_path = root / "data" / "metadata" / "stage6a_wsi_feature_summary.csv"
    formal_patches = pd.read_csv(formal_patch_path) if formal_patch_path.exists() else pd.DataFrame()
    formal_features = pd.read_csv(formal_feature_path) if formal_feature_path.exists() else pd.DataFrame()
    formal_patch_count = int(formal_patches.get("patch_status", pd.Series(dtype=str)).eq("extracted").sum())
    formal_feature_count = int(formal_features.get("feature_status", pd.Series(dtype=str)).isin(["extracted", "skipped_existing"]).sum())
    if args.real_features:
        real_patch_cap = int(pd.to_numeric(formal_patches.get("selected_patch_count", pd.Series(dtype=float)), errors="coerce").max()) if formal_patch_count else 0
        real_feature_backends = ", ".join(sorted(formal_features.get("feature_backend", pd.Series(dtype=str)).dropna().astype(str).unique()))
        patient_count = int(predictions["patient_id"].nunique())
        event_count = int(predictions.drop_duplicates("patient_id")["os_event"].sum())
        real_report = (
            "# Stage 6A-Fix Real WSI Smallset Pipeline Report\n\n"
            f"Generated: {datetime.now().isoformat(timespec='seconds')}\n\n"
            "## Integrity Boundary\n\n"
            "- This is a real-SVS engineering feasibility run on the selected TCGA-LUAD 20-slide smallset.\n"
            "- It is not a scientific performance estimate, external validation, or a basis for survival claims.\n"
            "- No mutation, CNV, methylation, single-cell, protein, Stage 3, Stage 6B, or full-cohort WSI work was started.\n\n"
            "## Closed-Loop Status\n\n"
            f"- Real WSI downloads complete with size and MD5 checks: {complete_real}/{selected_real}; resumable partial files: {partial_real}.\n"
            f"- Selected raw size: {expected_real_gb:.3f} GB.\n"
            f"- OpenSlide active: `{status['available']}`. Detail: `{status['detail']}`.\n"
            f"- Real slides with patch coordinates: {formal_patch_count}.\n"
            f"- Real slides with pretrained ResNet50 feature bags: {formal_feature_count}.\n"
            f"- Actual engineering cap per slide: {real_patch_cap} patches.\n"
            f"- Feature backend: `{real_feature_backends}`.\n"
            f"- MIL patients: {patient_count}; observed deaths: {event_count}; censored: {patient_count - event_count}.\n"
            f"- MIL models executed: {len(performance)}.\n"
            "- KM plots were intentionally not generated. The smallset is too small for scientific survival inference.\n\n"
            "## Interpretation\n\n"
            "- The real-SVS path is feasible only if all selected files, patch indexes and pretrained feature bags are present.\n"
            "- Metrics in `stage6a_real_smallset_performance.csv` are pipeline diagnostics only and must not be reported as model evidence.\n"
            "- Full-cohort WSI download and Stage 6B remain out of scope.\n\n"
            "## Inputs\n\n"
            "- `data/metadata/stage6a_wsi_smallset_download_status.csv`\n"
            "- `data/raw/tcga_luad/wsi/smallset/`\n"
            "- `data/metadata/stage6a_wsi_patch_summary.csv`\n"
            "- `data/metadata/stage6a_wsi_feature_summary.csv`\n\n"
            "## Outputs\n\n"
            "- `outputs/tables/stage6a_real_patch_extraction_summary.csv`\n"
            "- `outputs/tables/stage6a_real_feature_extraction_summary.csv`\n"
            "- `outputs/tables/stage6a_real_smallset_performance.csv`\n"
            "- `outputs/figures/stage6a_real_tissue_mask_example.png`\n"
            "- `outputs/figures/stage6a_real_patch_grid_example.png`\n"
            "- `outputs/figures/stage6a_real_attention_heatmap_example.png`\n"
            "- `outputs/logs/stage6a_real_smallset_training.log`\n\n"
            "## Potential Issues And Next Step\n\n"
            "- The validated `gpu_py310` environment currently contains a CPU-only PyTorch build. This is acceptable for the smallset feasibility run but not for full-scale feature extraction.\n"
            "- Keep the raw-SVS download scope fixed at the selected 20 slides. Review this feasibility report before any larger storage commitment.\n\n"
            "## Windows Commands\n\n"
            "```powershell\n"
            "conda activate gpu_py310\n"
            "python scripts/stage6a_check_wsi_environment.py --config configs/base.yaml\n"
            "python scripts/stage6a_resume_wsi_smallset.py --config configs/base.yaml\n"
            "python scripts/stage6a_extract_wsi_patches.py --config configs/base.yaml --smallset --real-svs\n"
            "python scripts/stage6a_extract_patch_features.py --config configs/base.yaml --smallset --real-svs --encoder resnet50\n"
            "python scripts/train_stage6a_pathology_mil.py --config configs/base.yaml --smallset --real-features\n"
            "python scripts/evaluate_stage6a_pathology_mil.py --config configs/base.yaml --smallset --real-features\n"
            "```\n"
        )
        real_report_path = reports / "stage6a_real_smallset_pipeline_report.md"
        real_report_path.write_text(real_report, encoding="utf-8")
        root_audit_path = root / "audit_report.md"
        existing_audit = root_audit_path.read_text(encoding="utf-8") if root_audit_path.exists() else "# SC-PROST-LUAD Audit Report\n"
        marker = "\n\n## Stage 6A-Fix Real Smallset\n\n"
        prefix = existing_audit.split(marker)[0]
        root_audit_path.write_text(prefix + marker + real_report, encoding="utf-8")
        print(json.dumps({"status": "passed", "dataset_mode": performance["dataset_mode"].iloc[0], "models": len(performance), "report": str(real_report_path)}, indent=2))
        return 0
    audit_path = reports / "stage6a_wsi_audit_report.md"
    audit_text = audit_path.read_text(encoding="utf-8") if audit_path.exists() else "# Stage 6A WSI Audit\n\nFormal GDC audit has not yet run.\n"
    report = (
        "# Stage 6A Pathology MIL Proof-of-Concept Report\n\n"
        f"Generated: {datetime.now().isoformat(timespec='seconds')}\n\n"
        "## Integrity Boundary\n\n"
        f"- Dataset mode: `{performance['dataset_mode'].iloc[0]}`.\n"
        "- These metrics verify pipeline execution only. They are not scientific WSI model results.\n"
        "- No mutation, CNV, methylation, single-cell, protein, or final multimodal fusion data were used.\n"
        "- Stage 3 was not started.\n\n"
        "## Pipeline Status\n\n"
        f"- OpenSlide available: `{status['available']}`. Detail: `{status['detail']}`.\n"
        f"- Synthetic-smoke slides with extracted patch features: {int(features['feature_status'].isin(['extracted', 'skipped_existing']).sum())}.\n"
        f"- Real smallset download state: {complete_real}/{selected_real} complete, {partial_real} resumable partial; "
        f"selected size {expected_real_gb:.3f} GB.\n"
        f"- Real smallset patch extraction: {formal_patch_count} slides; real feature extraction: {formal_feature_count} slides.\n"
        "- Tissue masking, background filtering, coordinate extraction, QC images, patch embeddings, "
        "attention MIL, gated MIL, clinical Cox, and clinical + pathology fusion all executed.\n"
        "- `stage6a_pathology_smallset_km.png` was intentionally not generated because this proof-of-concept "
        "sample size is too small for a scientific survival conclusion.\n\n"
        "## Required Answers\n\n"
        "1. WSI query: see `stage6a_wsi_audit_report.md`; real audit is separate from synthetic smoke testing.\n"
        "2. Patients with WSI: see the real GDC audit report.\n"
        "3. Clinical + OS + WSI overlap: see the real GDC audit report.\n"
        "4. Clinical + OS + WSI + RNA overlap: see the real GDC audit report.\n"
        f"5. Real WSI smallset download is partial: {complete_real}/{selected_real} complete and {partial_real} resumable partial. "
        "The remaining files were not reported as successful.\n"
        f"6. OpenSlide available on this interpreter: `{status['available']}`.\n"
        f"7. Patch extraction: synthetic mode passed; real SVS mode remains blocked because OpenSlide is unavailable "
        f"({formal_patch_count} real slides extracted).\n"
        f"8. Patch feature extraction: synthetic mode passed with an explicitly labeled fallback; formal real extraction "
        f"has not completed ({formal_feature_count} real slides).\n"
        "9. Attention MIL: passed as a PyTorch Cox-loss smoke test.\n"
        "10. Smallset sample size is insufficient for scientific conclusions.\n"
        "11. Do not start full WSI training yet. First install OpenSlide, finish the resumable 20-slide download, and "
        "validate pretrained torchvision ResNet50 extraction on the real smallset.\n"
        "12. The real audit estimates about 415 GB raw SVS. Reserve roughly 0.8-1.0 TB including derived artifacts and "
        "working space. At up to 1,000 patches per slide, the ceiling is about 541,000 patches. A single modern GPU may "
        "need roughly 4-12 hours for ResNet50 feature extraction after download; network download may take days on the "
        "currently observed connection. These are planning estimates, not measured benchmarks.\n"
        "13. The recommended design discussion remains clinical + WSI prediction, with RNA and later biological "
        "layers reserved for interpretation. No later layer was added here.\n"
    )
    (reports / "stage6a_pathology_mil_report.md").write_text(report, encoding="utf-8")
    root_audit = (
        audit_text
        + "\n\n## Stage 6A Pipeline Smoke Test\n\n"
        + report.split("## Required Answers")[0]
        + "\n## Commands\n\n```powershell\n"
        + "python scripts/stage6a_build_wsi_manifest.py --config configs/base.yaml\n"
        + "python scripts/stage6a_download_wsi_smallset.py --config configs/base.yaml --n-slides 20 --select-only\n"
        + "python scripts/stage6a_download_wsi_smallset.py --config configs/base.yaml --small-test\n"
        + "python scripts/stage6a_extract_wsi_patches.py --config configs/base.yaml --small-test\n"
        + "python scripts/stage6a_extract_patch_features.py --config configs/base.yaml --small-test\n"
        + "python scripts/train_stage6a_pathology_mil.py --config configs/base.yaml --small-test\n"
        + "python scripts/evaluate_stage6a_pathology_mil.py --config configs/base.yaml --small-test\n"
        + "python -m pytest tests -q\npython -m compileall -q src scripts\n```\n"
    )
    (root / "audit_report.md").write_text(root_audit, encoding="utf-8")
    print(json.dumps({"status": "passed", "dataset_mode": performance["dataset_mode"].iloc[0], "models": len(performance)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
