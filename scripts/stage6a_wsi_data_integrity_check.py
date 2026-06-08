"""Check WSI pilot IDs, leakage and ResNet50 feature tensor integrity."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pathology.diagnostics_utils import load_feature_artifact, load_pilot_tables


def _add(rows, check_name: str, passed: bool, value, detail: str = "") -> None:
    rows.append({"check_name": check_name, "passed": bool(passed), "value": value, "detail": detail})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit Stage 6A WSI pilot data integrity and feature tensors.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--root", default=".")
    parser.add_argument("--similarity-threshold", type=float, default=0.999)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root).resolve()
    tables = load_pilot_tables(root)
    cohort, split, features = tables["cohort"], tables["split"], tables["feature"]
    ok_features = features.loc[features["feature_status"].isin(["extracted", "skipped_existing"])].copy()
    rows = []
    _add(rows, "cohort_patient_id_unique", cohort["patient_id"].is_unique, cohort["patient_id"].nunique())
    _add(rows, "cohort_slide_id_unique", cohort["slide_id"].is_unique, cohort["slide_id"].nunique())
    _add(rows, "os_time_positive", bool((cohort["OS_time"].astype(float) > 0).all()), float(cohort["OS_time"].min()))
    _add(rows, "os_status_binary", set(cohort["OS_status"].astype(int).unique()).issubset({0, 1}), sorted(cohort["OS_status"].astype(int).unique()))
    split_sets = {name: set(split.loc[split["split"] == name, "patient_id"]) for name in ("train", "validation", "test")}
    overlap_count = sum(len(split_sets[a] & split_sets[b]) for a, b in itertools.combinations(split_sets, 2))
    _add(rows, "train_val_test_patient_overlap_zero", overlap_count == 0, overlap_count)
    feature_patients = set(ok_features["patient_id"])
    _add(rows, "clinical_feature_patient_match", set(cohort["patient_id"]) == feature_patients, f"{len(feature_patients)}/{len(cohort)}")
    _add(rows, "feature_file_one_to_one", ok_features["file_id"].is_unique and ok_features["feature_path"].is_unique, len(ok_features))
    event_rates = split.merge(cohort[["patient_id", "OS_status"]], on="patient_id", how="left").groupby("split")["OS_status"].mean().to_dict()
    _add(rows, "event_rates_balanced_by_split", max(event_rates.values()) - min(event_rates.values()) < 0.05, event_rates)
    vectors = []
    hashes = []
    bad = {"nan": 0, "inf": 0, "all_zero": 0, "bad_dim": 0}
    shapes = []
    for row in ok_features.to_dict("records"):
        artifact = load_feature_artifact(row["feature_path"])
        tensor = artifact["features"].float()
        shapes.append(tuple(tensor.shape))
        if tensor.ndim != 2 or tensor.shape[1] != 2048:
            bad["bad_dim"] += 1
        if torch.isnan(tensor).any():
            bad["nan"] += 1
        if torch.isinf(tensor).any():
            bad["inf"] += 1
        if torch.all(tensor == 0):
            bad["all_zero"] += 1
        arr = tensor.mean(dim=0).numpy().astype(np.float32)
        vectors.append(arr)
        hashes.append(hashlib.md5(arr.tobytes()).hexdigest())
    _add(rows, "feature_shape_second_dim_2048", bad["bad_dim"] == 0, sorted(set(shapes))[:5], "Patch count may vary; feature dimension must be 2048.")
    _add(rows, "feature_no_nan", bad["nan"] == 0, bad["nan"])
    _add(rows, "feature_no_inf", bad["inf"] == 0, bad["inf"])
    _add(rows, "feature_not_all_zero", bad["all_zero"] == 0, bad["all_zero"])
    duplicate_hash_count = len(hashes) - len(set(hashes))
    _add(rows, "no_duplicate_mean_feature_hash", duplicate_hash_count == 0, duplicate_hash_count)
    matrix = np.vstack(vectors)
    sim = cosine_similarity(matrix)
    np.fill_diagonal(sim, -1.0)
    max_sim = float(sim.max()) if sim.size else float("nan")
    abnormal_pairs = int((sim > args.similarity_threshold).sum() / 2)
    _add(rows, "no_abnormally_similar_mean_features", abnormal_pairs == 0, abnormal_pairs, f"max cosine similarity={max_sim:.6f}")
    table = pd.DataFrame(rows)
    out_table = root / "outputs" / "tables" / "stage6a_wsi_data_integrity_checks.csv"
    out_report = root / "outputs" / "reports" / "stage6a_wsi_data_integrity_report.md"
    out_table.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_table, index=False)
    failed = table.loc[~table["passed"]]
    report = (
        "# Stage 6A WSI Data Integrity Report\n\n"
        f"Generated: {datetime.now().isoformat(timespec='seconds')}\n\n"
        f"- Checks passed: {int(table['passed'].sum())}/{len(table)}.\n"
        f"- Failed checks: {', '.join(failed['check_name']) if not failed.empty else 'none'}.\n"
        f"- Event rates by split: {event_rates}.\n"
        f"- Feature max mean-vector cosine similarity: {max_sim:.6f}.\n\n"
        "## Interpretation\n\n"
        + ("- No evidence of patient/slide leakage, label mismatch, NaN/Inf/all-zero tensors, or duplicate feature vectors was detected.\n" if failed.empty else "- Review failed checks before using WSI pilot results.\n")
    )
    out_report.write_text(report, encoding="utf-8")
    print(json.dumps({"status": "passed" if failed.empty else "warning", "failed_checks": failed["check_name"].tolist()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

