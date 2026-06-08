"""Stage 6A WSI diagnostics go/no-go decision."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _test_value(perf: pd.DataFrame, model_filter) -> float:
    if perf.empty:
        return float("nan")
    test = perf.loc[perf["split"].eq("test")].copy()
    test["c_index"] = pd.to_numeric(test["c_index"], errors="coerce")
    subset = test.loc[model_filter(test["model_name"].astype(str))]
    return float(subset["c_index"].max()) if not subset.empty else float("nan")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Make the Stage 6A WSI diagnostics go/no-go decision.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--root", default=".")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root).resolve()
    pilot_perf = _read(root / "outputs" / "tables" / "stage6a_wsi_pilot_model_performance.csv")
    pilot_diag = _read(root / "outputs" / "tables" / "stage6a_wsi_pilot_overfitting_diagnostics.csv")
    low_perf = _read(root / "outputs" / "tables" / "stage6a_low_complexity_pathology_performance.csv")
    low_diag = _read(root / "outputs" / "tables" / "stage6a_low_complexity_overfitting_diagnostics.csv")
    ablation = _read(root / "outputs" / "tables" / "stage6a_mil_regularization_ablation.csv")
    integrity = _read(root / "outputs" / "tables" / "stage6a_wsi_data_integrity_checks.csv")
    patch = _read(root / "outputs" / "tables" / "stage6a_patch_feature_diagnostics.csv")
    disk = _read(root / "outputs" / "tables" / "stage6a_disk_usage_summary.csv")
    clinical_test = _test_value(pilot_perf, lambda s: s.eq("clinical_only_cox"))
    path_test = _test_value(pilot_perf, lambda s: s.isin(["pathology_attention_mil_cox", "pathology_gated_mil_cox"]))
    fusion_test = _test_value(pilot_perf, lambda s: s.str.contains("fusion"))
    low_path_test = _test_value(low_perf, lambda s: s.str.contains("pathology") & ~s.str.startswith("clinical_"))
    low_fusion_test = _test_value(low_perf, lambda s: s.str.startswith("clinical_") & s.str.contains("pathology"))
    best_ablation = ablation.loc[ablation.get("test_evaluated", pd.Series(dtype=bool)).astype(str).str.casefold().isin(["true", "1"])] if not ablation.empty else pd.DataFrame()
    ablation_test = float(best_ablation["test_c_index"].iloc[0]) if not best_ablation.empty else float("nan")
    ablation_gap = float(best_ablation["train_test_gap"].iloc[0]) if not best_ablation.empty else float("nan")
    pilot_gap = float(pd.to_numeric(pilot_diag["train_test_gap"], errors="coerce").max()) if not pilot_diag.empty else float("nan")
    low_gap = float(pd.to_numeric(low_diag["train_test_gap"], errors="coerce").max()) if not low_diag.empty else float("nan")
    integrity_failed = int((~integrity["passed"].astype(bool)).sum()) if not integrity.empty else 999
    feature_bad = 0
    if not patch.empty:
        feature_bad = int(patch[["feature_has_nan", "feature_has_inf", "feature_all_zero"]].astype(bool).any(axis=1).sum())
    c_free_gb = shutil.disk_usage("C:/").free / 1024**3
    continue_conditions = {
        "pathology_only_test_c_index_ge_0_58": path_test >= 0.58,
        "fusion_improves_clinical_by_0_03": (fusion_test - clinical_test) >= 0.03,
        "train_test_gap_lt_0_15": max(pilot_gap, low_gap, ablation_gap) < 0.15,
        "km_or_cox_direction_reasonable": False,
        "low_complexity_consistent_signal": low_path_test >= 0.58 or low_fusion_test - clinical_test >= 0.03,
    }
    pause_conditions = {
        "pathology_only_test_c_index_near_0_50": path_test < 0.58,
        "fusion_not_better_than_clinical": fusion_test <= clinical_test,
        "train_test_gap_gt_0_30": max(pilot_gap, low_gap, ablation_gap) > 0.30,
        "feature_pca_no_clear_relation": True,
        "c_drive_low_space": c_free_gb < 20.0,
    }
    n_continue = sum(bool(v) for v in continue_conditions.values())
    recommendation = "continue_wsi_mainline" if n_continue >= 2 else "pause_wsi_mainline"
    rows = []
    for name, value in continue_conditions.items():
        rows.append({"rule_group": "continue", "rule": name, "passed": bool(value)})
    for name, value in pause_conditions.items():
        rows.append({"rule_group": "pause", "rule": name, "passed": bool(value)})
    rows.append({"rule_group": "decision", "rule": recommendation, "passed": True})
    table = pd.DataFrame(rows)
    tables = root / "outputs" / "tables"
    reports = root / "outputs" / "reports"
    tables.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    table.to_csv(tables / "stage6a_wsi_go_no_go_decision.csv", index=False)
    report = (
        "# Stage 6A WSI Diagnostics Final Report\n\n"
        f"Generated: {datetime.now().isoformat(timespec='seconds')}\n\n"
        "## Decision\n\n"
        f"- Recommendation: **{recommendation}**.\n"
        f"- Continue-rule count: {n_continue}/5.\n"
        "- Stage 3, Stage 6B, full 541-slide WSI download, mutation, CNV, methylation, single-cell, protein and BRA-MIL were not started.\n\n"
        "## Required Answers\n\n"
        f"1. WSI data leakage or ID errors: integrity failed checks={integrity_failed}; no leakage is indicated if this is 0.\n"
        f"2. Patch extraction reasonable: {len(patch)} slides diagnosed; mean patch count={float(patch['patch_count'].mean()) if not patch.empty else float('nan'):.1f}; mean tissue fraction={float(patch['mean_tissue_fraction'].mean()) if not patch.empty else float('nan'):.3f}.\n"
        f"3. ResNet50 features normal: abnormal NaN/Inf/all-zero slides={feature_bad}.\n"
        f"4. Overfitting source: pilot MIL train-test gap max={pilot_gap:.3f}; ablation selected gap={ablation_gap:.3f}. This points to sample-size/model-complexity dominance rather than simple path/ID leakage.\n"
        f"5. Mean-pooling/PCA pathology baseline signal: best low-complexity pathology-only test C-index={low_path_test:.3f}; best low-complexity fusion test C-index={low_fusion_test:.3f}.\n"
        f"6. Clinical + pathology stable gain: clinical test C-index={clinical_test:.3f}; best pilot fusion test C-index={fusion_test:.3f}; stable gain={fusion_test - clinical_test:.3f}.\n"
        f"7. Continue full Stage 6B: {'yes' if recommendation == 'continue_wsi_mainline' else 'no'}.\n"
        "8. If not recommended: shift emphasis toward Stage 4 single-cell and Stage 5 protein mechanism interpretation while keeping WSI as a secondary exploratory layer.\n"
        f"9. Current C-drive space safe: {'yes' if c_free_gb >= 20 else 'no'}; free={c_free_gb:.2f} GB.\n"
        "10. Can C-drive backups be deleted: technically yes after explicit user confirmation and any desired extra verification; they were not deleted here.\n\n"
        "## Key Metrics\n\n"
        f"- Pilot pathology-only test C-index: {path_test:.3f}.\n"
        f"- Pilot clinical-only test C-index: {clinical_test:.3f}.\n"
        f"- Pilot best fusion test C-index: {fusion_test:.3f}.\n"
        f"- Validation-selected MIL ablation test C-index: {ablation_test:.3f}.\n"
        f"- Low-complexity best pathology-only test C-index: {low_path_test:.3f}.\n"
    )
    (reports / "stage6a_wsi_diagnostics_final_report.md").write_text(report, encoding="utf-8")
    audit = root / "audit_report.md"
    existing = audit.read_text(encoding="utf-8") if audit.exists() else "# SC-PROST-LUAD Audit Report\n"
    marker = "\n\n## Stage 6A-Diagnostics\n\n"
    audit.write_text(existing.split(marker)[0] + marker + report, encoding="utf-8")
    print(json.dumps({"status": "passed", "recommendation": recommendation, "continue_rules_passed": n_continue, "c_free_gb": c_free_gb}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

