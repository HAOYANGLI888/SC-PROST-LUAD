"""Balanced Stage 6A WSI smallset selection and resumable public download."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data.clinical_preprocess import load_tcga_cdr_os
from data.wsi_manifest import GDC_API, USER_AGENT, WSIManifestError, WSIPaths
from pathology.wsi_io import create_synthetic_slide


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def select_smallset(root: str | Path, *, n_slides: int = 20, seed: int = 42) -> pd.DataFrame:
    """Select one small diagnostic slide per patient with approximate event balance."""

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
        raise WSIManifestError("No diagnostic WSI patients overlap usable OS.")
    eligible["os_event"] = eligible["os_event"].astype(int)
    rng = np.random.default_rng(seed)
    eligible["_random"] = rng.random(len(eligible))
    eligible = eligible.sort_values(["os_event", "file_size", "_random", "patient_id"])
    target_events = min(n_slides // 2, int((eligible["os_event"] == 1).sum()))
    target_censored = min(n_slides - target_events, int((eligible["os_event"] == 0).sum()))
    selected = pd.concat(
        [
            eligible.loc[eligible["os_event"] == 1].head(target_events),
            eligible.loc[eligible["os_event"] == 0].head(target_censored),
        ],
        ignore_index=True,
    )
    if len(selected) < n_slides:
        remaining = eligible.loc[~eligible["file_id"].isin(selected["file_id"])]
        selected = pd.concat([selected, remaining.head(n_slides - len(selected))], ignore_index=True)
    return selected.drop(columns="_random").sort_values(["os_event", "patient_id"]).reset_index(drop=True)


def _download_one(row: dict[str, Any], download_dir: Path, *, retries: int = 3, timeout: int = 180) -> dict[str, Any]:
    target = download_dir / str(row["file_id"]) / str(row["file_name"])
    target.parent.mkdir(parents=True, exist_ok=True)
    expected_size = int(row["file_size"])
    expected_md5 = str(row["md5sum"])
    if target.exists() and target.stat().st_size == expected_size and (not expected_md5 or _md5(target) == expected_md5):
        return {**row, "local_path": str(target), "download_status": "skipped_complete", "downloaded_bytes": expected_size}
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
            if partial.stat().st_size != expected_size:
                raise IOError(f"size mismatch {partial.stat().st_size}/{expected_size}")
            if expected_md5 and _md5(partial) != expected_md5:
                raise IOError("MD5 mismatch")
            partial.replace(target)
            return {**row, "local_path": str(target), "download_status": "downloaded", "downloaded_bytes": expected_size}
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(attempt)
    return {**row, "local_path": str(target), "download_status": "failed", "downloaded_bytes": 0, "error": str(last_error)}


def _refresh_resume_row(row: dict[str, Any], download_dir: Path) -> dict[str, Any]:
    target = download_dir / str(row["file_id"]) / str(row["file_name"])
    partial = target.with_suffix(target.suffix + ".part")
    expected_size = int(row["file_size"])
    expected_md5 = str(row.get("md5sum", "") or "")
    local_size = target.stat().st_size if target.exists() else partial.stat().st_size if partial.exists() else 0
    actual_md5 = ""
    error_message = ""
    if target.exists() and target.stat().st_size == expected_size:
        actual_md5 = _md5(target)
        if expected_md5 and actual_md5 != expected_md5:
            status = "failed_md5_mismatch"
            error_message = f"MD5 mismatch: expected {expected_md5}, observed {actual_md5}"
        else:
            status = "complete_existing"
    elif partial.exists():
        status = "partial_resume_available"
    elif target.exists():
        status = "incomplete_target_file"
        error_message = f"size mismatch: expected {expected_size}, observed {target.stat().st_size}"
    else:
        status = "selected_not_downloaded"
    return {
        **row,
        "slide_id": str(row.get("slide_id") or Path(str(row["file_name"])).stem),
        "expected_size": expected_size,
        "local_size": int(local_size),
        "md5": actual_md5 or expected_md5,
        "local_path": str(target),
        "download_status": status,
        "error_message": error_message,
        "downloaded_bytes": int(local_size),
    }


def refresh_smallset_download_status(root: str | Path) -> pd.DataFrame:
    """Refresh the persisted strict 20-slide selection without changing membership."""

    paths = WSIPaths.from_root(root)
    status_path = paths.metadata / "stage6a_wsi_smallset_download_status.csv"
    if not status_path.exists():
        raise FileNotFoundError(
            f"Smallset selection missing: {status_path}. Run stage6a_download_wsi_smallset.py --select-only first."
        )
    current = pd.read_csv(status_path)
    download_dir = paths.root / "data" / "raw" / "tcga_luad" / "wsi" / "smallset"
    refreshed = pd.DataFrame([_refresh_resume_row(row, download_dir) for row in current.to_dict("records")])
    refreshed.to_csv(status_path, index=False)
    return refreshed


def resume_smallset_download(
    root: str | Path,
    *,
    workers: int = 4,
    retries: int = 3,
    timeout: int = 180,
) -> pd.DataFrame:
    """Resume only the persisted Stage 6A smallset and save status after each file."""

    if workers < 1:
        raise ValueError("workers must be at least 1.")
    paths = WSIPaths.from_root(root)
    status_path = paths.metadata / "stage6a_wsi_smallset_download_status.csv"
    download_dir = paths.root / "data" / "raw" / "tcga_luad" / "wsi" / "smallset"
    frame = refresh_smallset_download_status(root)
    pending = frame.loc[~frame["download_status"].eq("complete_existing")].copy()
    by_file = {str(row["file_id"]): row for row in frame.to_dict("records")}
    if pending.empty:
        return frame
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_download_one, row, download_dir, retries=retries, timeout=timeout): str(row["file_id"])
            for row in pending.to_dict("records")
        }
        for future in as_completed(futures):
            file_id = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {**by_file[file_id], "download_status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            refreshed = _refresh_resume_row(result, download_dir)
            if refreshed["download_status"] == "complete_existing" and result.get("download_status") == "downloaded":
                refreshed["download_status"] = "downloaded_complete"
            if result.get("error"):
                refreshed["error_message"] = str(result["error"])
            by_file[file_id] = refreshed
            pd.DataFrame(by_file.values()).sort_values(["os_event", "patient_id"]).to_csv(status_path, index=False)
    return refresh_smallset_download_status(root)


def _run_gdc_client(client: Path, manifest: Path, destination: Path) -> None:
    if not client.exists():
        raise FileNotFoundError(f"gdc-client.exe not found: {client}")
    result = subprocess.run([str(client), "download", "-m", str(manifest), "-d", str(destination)], check=False)
    if result.returncode != 0:
        raise WSIManifestError(f"gdc-client exited with code {result.returncode}.")


def prepare_synthetic_smallset(root: str | Path, *, n_slides: int = 20) -> pd.DataFrame:
    """Create deterministic local TIFF slides for a dependency-light smoke test."""

    root = Path(root).resolve()
    raw = root / "data" / "raw" / "stage6a_small_test" / "slides"
    raw.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(n_slides):
        slide_id = f"TOY-SLIDE-{index:03d}"
        path = raw / f"{slide_id}.tif"
        create_synthetic_slide(path, seed=2026 + index)
        rows.append(
            {
                "file_id": slide_id,
                "file_name": path.name,
                "patient_id": f"TOY-PATIENT-{index:03d}",
                "os_time_days": float(240 + 55 * index),
                "os_event": int(index % 2 == 0),
                "age": float(52 + index % 18),
                "male": float(index % 2),
                "stage_numeric": float(1 + index % 3),
                "local_path": str(path),
                "download_status": "synthetic_created",
                "downloaded_bytes": path.stat().st_size,
            }
        )
    frame = pd.DataFrame(rows)
    output = root / "data" / "metadata" / "stage6a_small_test" / "wsi_smallset_download_status.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame


def download_smallset(
    root: str | Path,
    *,
    n_slides: int = 20,
    gdc_client: str | Path | None = None,
    select_only: bool = False,
    small_test: bool = False,
) -> dict[str, Any]:
    """Select and optionally download a balanced small diagnostic WSI set."""

    if small_test:
        frame = prepare_synthetic_smallset(root, n_slides=n_slides)
        return {"status": "passed", "dataset_mode": "toy_small_test", "slides": len(frame)}
    paths = WSIPaths.from_root(root)
    paths.ensure_dirs()
    selected = select_smallset(root, n_slides=n_slides)
    download_dir = paths.root / "data" / "raw" / "tcga_luad" / "wsi" / "smallset"
    status_path = paths.metadata / "stage6a_wsi_smallset_download_status.csv"
    if select_only:
        local_paths = []
        statuses = []
        bytes_present = []
        for row in selected.to_dict("records"):
            target = download_dir / str(row["file_id"]) / str(row["file_name"])
            partial = target.with_suffix(target.suffix + ".part")
            expected_size = int(row["file_size"])
            if target.exists() and target.stat().st_size == expected_size:
                status = "skipped_complete"
                present = expected_size
            elif partial.exists():
                status = "partial_resume_available"
                present = partial.stat().st_size
            else:
                status = "selected_not_downloaded"
                present = 0
            local_paths.append(str(target))
            statuses.append(status)
            bytes_present.append(present)
        selected["local_path"] = local_paths
        selected["download_status"] = statuses
        selected["downloaded_bytes"] = bytes_present
        selected.to_csv(status_path, index=False)
        return {
            "status": "passed",
            "dataset_mode": "real_gdc_metadata_select_only",
            "slides": len(selected),
            "expected_gb": float(selected["file_size"].sum() / 1e9),
            "complete_slides": int((selected["download_status"] == "skipped_complete").sum()),
            "partial_slides": int((selected["download_status"] == "partial_resume_available").sum()),
        }
    if gdc_client:
        manifest = paths.metadata / "stage6a_wsi_smallset_manifest.tsv"
        selected[["file_id", "file_name", "md5sum", "file_size", "state"]].rename(
            columns={"file_id": "id", "file_name": "filename", "md5sum": "md5", "file_size": "size"}
        ).to_csv(manifest, sep="\t", index=False)
        _run_gdc_client(Path(gdc_client), manifest, download_dir)
    rows = [_download_one(row, download_dir) for row in selected.to_dict("records")]
    frame = pd.DataFrame(rows)
    frame.to_csv(status_path, index=False)
    return {
        "status": "passed" if (frame["download_status"] != "failed").all() else "partial_failure",
        "dataset_mode": "real_gdc_smallset",
        "slides": len(frame),
        "downloaded": int(frame["download_status"].isin(["downloaded", "skipped_complete"]).sum()),
    }
