"""Stage 6A 100-slide WSI pilot selection and safe resumable download."""

from __future__ import annotations

import hashlib
import shutil
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

from data.clinical_preprocess import load_tcga_cdr_os
from data.wsi_manifest import GDC_API, USER_AGENT, WSIManifestError, WSIPaths


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _resource_aware_stage_cover(frame: pd.DataFrame, n: int) -> pd.DataFrame:
    frame = frame.copy()
    frame["stage_bucket"] = frame["stage_numeric"].fillna(99).astype(float)
    covered = []
    for _, bucket in frame.groupby("stage_bucket", sort=True):
        covered.append(bucket.sort_values(["file_size", "patient_id", "file_id"]).iloc[0])
    selected = pd.DataFrame(covered).drop_duplicates("file_id")
    remaining = frame.loc[~frame["file_id"].isin(selected["file_id"])].sort_values(["file_size", "stage_bucket", "patient_id", "file_id"])
    selected = pd.concat([selected, remaining.head(max(n - len(selected), 0))], ignore_index=True)
    return selected.head(n).drop(columns=["stage_bucket"], errors="ignore")


def select_wsi_pilot_cohort(root: str | Path, *, n_slides: int = 100, seed: int = 42) -> pd.DataFrame:
    """Select one diagnostic SVS per patient for a resource-conscious GPU pilot."""

    paths = WSIPaths.from_root(root)
    if not paths.patient_slide_map.exists():
        raise FileNotFoundError(f"WSI patient-slide map missing: {paths.patient_slide_map}")
    survival_path = paths.root / "data" / "raw" / "tcga_luad" / "clinical" / "Survival_SupplementalTable_S1_20171025_xena_sp.tsv"
    clinical = load_tcga_cdr_os(survival_path)
    slides = pd.read_csv(paths.patient_slide_map)
    preferred = slides.loc[slides["preferred_slide_for_smallset"].astype(str).str.casefold().isin({"true", "1"})].copy()
    eligible = preferred.merge(
        clinical[["patient_id", "os_time_days", "os_event", "age", "male", "stage_numeric"]],
        on="patient_id",
        how="inner",
        validate="one_to_one",
    ).dropna(subset=["os_time_days", "os_event"])
    eligible = eligible.loc[eligible["os_time_days"] > 0].copy()
    if eligible.empty:
        raise WSIManifestError("No WSI patients overlap usable OS.")
    eligible["os_event"] = eligible["os_event"].astype(int)
    target_events = min(n_slides // 2, int(eligible["os_event"].sum()))
    target_censored = min(n_slides - target_events, int((eligible["os_event"] == 0).sum()))
    selected = pd.concat(
        [
            _resource_aware_stage_cover(eligible.loc[eligible["os_event"] == 1], target_events),
            _resource_aware_stage_cover(eligible.loc[eligible["os_event"] == 0], target_censored),
        ],
        ignore_index=True,
    )
    if len(selected) < n_slides:
        remaining = eligible.loc[~eligible["file_id"].isin(selected["file_id"])]
        selected = pd.concat([selected, remaining.sort_values(["file_size", "patient_id"]).head(n_slides - len(selected))], ignore_index=True)
    selected = selected.drop_duplicates("patient_id").head(n_slides)
    selected = selected.sort_values(["os_event", "stage_numeric", "file_size", "patient_id"]).reset_index(drop=True)
    selected["slide_id"] = selected["file_name"].map(lambda value: Path(str(value)).stem)
    selected["OS_time"] = selected["os_time_days"]
    selected["OS_status"] = selected["os_event"]
    selected["stage"] = selected["stage_numeric"]
    selected["download_url"] = selected["file_id"].map(lambda file_id: f"{GDC_API}/data/{file_id}")
    selected["expected_size"] = selected["file_size"].astype(int)
    selected["md5"] = selected["md5sum"]
    selected["selection_note"] = (
        "Deterministic event-balanced and stage-covering feasibility pilot; file_size is used only as a resource-aware tie-breaker."
    )
    out = paths.metadata / "stage6a_wsi_pilot_cohort.csv"
    paths.metadata.mkdir(parents=True, exist_ok=True)
    selected.to_csv(out, index=False)
    summary = pd.DataFrame(
        [
            {
                "requested_slides": n_slides,
                "selected_slides": len(selected),
                "selected_patients": selected["patient_id"].nunique(),
                "death_events": int(selected["OS_status"].sum()),
                "censored": int((selected["OS_status"] == 0).sum()),
                "expected_size_gb": float(selected["expected_size"].sum() / 1e9),
                "seed": seed,
            }
        ]
    )
    paths.tables.mkdir(parents=True, exist_ok=True)
    summary.to_csv(paths.tables / "stage6a_wsi_pilot_cohort_summary.csv", index=False)
    return selected


def _smallset_source(root: Path, row: dict[str, Any]) -> Path | None:
    source = root / "data" / "raw" / "tcga_luad" / "wsi" / "smallset" / str(row["file_id"]) / str(row["file_name"])
    if source.exists() and source.stat().st_size == int(row["expected_size"]):
        return source
    return None


def _target(root: Path, row: dict[str, Any]) -> Path:
    return root / "data" / "raw" / "tcga_luad" / "wsi_pilot" / str(row["file_id"]) / str(row["file_name"])


def _refresh_row(root: Path, row: dict[str, Any]) -> dict[str, Any]:
    target = _target(root, row)
    partial = target.with_suffix(target.suffix + ".part")
    expected = int(row["expected_size"])
    expected_md5 = str(row.get("md5") or row.get("md5sum") or "")
    local_size = target.stat().st_size if target.exists() else partial.stat().st_size if partial.exists() else 0
    md5_observed = ""
    md5_status = "not_checked"
    error = ""
    if target.exists() and target.stat().st_size == expected:
        md5_observed = _md5(target)
        md5_status = "passed" if (not expected_md5 or md5_observed == expected_md5) else "failed"
        status = "complete_existing" if md5_status == "passed" else "failed_md5_mismatch"
        if md5_status == "failed":
            error = f"expected {expected_md5}, observed {md5_observed}"
    elif partial.exists():
        status = "partial_resume_available"
    elif target.exists():
        status = "incomplete_target_file"
        error = f"expected {expected}, observed {target.stat().st_size}"
    else:
        status = "selected_not_downloaded"
    return {
        **row,
        "local_path": str(target),
        "download_status": status,
        "local_size": int(local_size),
        "md5_status": md5_status,
        "observed_md5": md5_observed,
        "error_message": error,
    }


def _download_one(root: Path, row: dict[str, Any], *, retries: int, timeout: int) -> dict[str, Any]:
    target = _target(root, row)
    target.parent.mkdir(parents=True, exist_ok=True)
    source = _smallset_source(root, row)
    if source is not None and not target.exists():
        shutil.copy2(source, target)
        refreshed = _refresh_row(root, row)
        refreshed["download_status"] = "copied_from_smallset" if refreshed["md5_status"] == "passed" else refreshed["download_status"]
        return refreshed
    refreshed = _refresh_row(root, row)
    if refreshed["download_status"] == "complete_existing":
        return refreshed
    expected = int(row["expected_size"])
    expected_md5 = str(row.get("md5") or row.get("md5sum") or "")
    partial = target.with_suffix(target.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            request = urllib.request.Request(
                f"{GDC_API}/data/{row['file_id']}",
                headers={"User-Agent": USER_AGENT, **({"Range": f"bytes={offset}-"} if offset else {})},
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                append = offset > 0 and getattr(response, "status", None) == 206
                with partial.open("ab" if append else "wb") as output:
                    shutil.copyfileobj(response, output, length=1024 * 1024)
            if partial.stat().st_size != expected:
                raise IOError(f"size mismatch {partial.stat().st_size}/{expected}")
            if expected_md5 and _md5(partial) != expected_md5:
                raise IOError("MD5 mismatch")
            partial.replace(target)
            result = _refresh_row(root, row)
            result["download_status"] = "downloaded_complete"
            return result
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(attempt)
    result = _refresh_row(root, row)
    result["download_status"] = "failed"
    result["error_message"] = str(last_error)
    return result


def refresh_wsi_pilot_download_status(root: str | Path) -> pd.DataFrame:
    root = Path(root).resolve()
    cohort_path = root / "data" / "metadata" / "stage6a_wsi_pilot_cohort.csv"
    status_path = root / "data" / "metadata" / "stage6a_wsi_pilot_download_status.csv"
    if not cohort_path.exists():
        raise FileNotFoundError(f"Pilot cohort missing: {cohort_path}")
    cohort = pd.read_csv(cohort_path)
    frame = pd.DataFrame([_refresh_row(root, row) for row in cohort.to_dict("records")])
    frame.to_csv(status_path, index=False)
    return frame


def download_wsi_pilot(root: str | Path, *, workers: int = 8, retries: int = 4, timeout: int = 240, min_free_gb: float = 2.0) -> pd.DataFrame:
    root = Path(root).resolve()
    status_path = root / "data" / "metadata" / "stage6a_wsi_pilot_download_status.csv"
    frame = refresh_wsi_pilot_download_status(root)
    remaining_bytes = int((frame["expected_size"] - frame["local_size"]).clip(lower=0).sum())
    free_bytes = shutil.disk_usage(root).free
    if free_bytes < remaining_bytes + int(min_free_gb * 1024**3):
        raise RuntimeError(
            f"Insufficient disk for pilot download: free={free_bytes / 1e9:.2f} GB, "
            f"remaining={remaining_bytes / 1e9:.2f} GB, reserve={min_free_gb:.1f} GB."
        )
    pending = frame.loc[~frame["download_status"].isin(["complete_existing", "copied_from_smallset", "downloaded_complete"])].copy()
    rows = {str(row["file_id"]): row for row in frame.to_dict("records")}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_download_one, root, row, retries=retries, timeout=timeout): str(row["file_id"])
            for row in pending.to_dict("records")
        }
        for future in as_completed(futures):
            file_id = futures[future]
            try:
                rows[file_id] = future.result()
            except Exception as exc:
                rows[file_id] = {**rows[file_id], "download_status": "failed", "error_message": f"{type(exc).__name__}: {exc}"}
            pd.DataFrame(rows.values()).sort_values(["OS_status", "stage", "expected_size", "patient_id"]).to_csv(status_path, index=False)
    final = refresh_wsi_pilot_download_status(root)
    summary = pd.DataFrame(
        [
            {
                "selected_slides": len(final),
                "complete_slides": int(final["download_status"].isin(["complete_existing"]).sum()),
                "partial_slides": int(final["download_status"].eq("partial_resume_available").sum()),
                "failed_slides": int(final["download_status"].str.startswith("failed", na=False).sum()),
                "expected_gb": float(final["expected_size"].sum() / 1e9),
                "local_gb": float(final["local_size"].sum() / 1e9),
            }
        ]
    )
    (root / "outputs" / "tables").mkdir(parents=True, exist_ok=True)
    summary.to_csv(root / "outputs" / "tables" / "stage6a_wsi_pilot_download_summary.csv", index=False)
    return final
