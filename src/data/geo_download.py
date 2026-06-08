"""Download public GEO series matrices and platform annotations from NCBI."""

from __future__ import annotations

import shutil
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class GEOCohortSpec:
    accession: str
    platform: str
    os_time_field: str
    os_status_field: str
    os_time_unit: str
    age_field: str | None
    sex_field: str | None
    stage_field: str | None

    @property
    def matrix_url(self) -> str:
        bucket = f"GSE{int(self.accession[3:]) // 1000}nnn"
        return (
            f"https://ftp.ncbi.nlm.nih.gov/geo/series/{bucket}/"
            f"{self.accession}/matrix/{self.accession}_series_matrix.txt.gz"
        )


GEO_COHORTS: dict[str, GEOCohortSpec] = {
    "GSE31210": GEOCohortSpec(
        "GSE31210", "GPL570", "days_before_death_censor", "death", "days",
        "age_years", "gender", "pathological_stage",
    ),
    "GSE50081": GEOCohortSpec(
        "GSE50081", "GPL570", "survival_time", "status", "years",
        "age", "sex", "stage",
    ),
    "GSE72094": GEOCohortSpec(
        "GSE72094", "GPL15048", "survival_time_in_days", "vital_status", "days",
        "age_at_diagnosis", "gender", "stage",
    ),
    "GSE68465": GEOCohortSpec(
        "GSE68465", "GPL96", "months_to_last_contact_or_death", "vital_status",
        "months", "age", "sex", None,
    ),
}


PLATFORM_URLS: dict[str, str] = {
    "GPL570": "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/GPL570/annot/GPL570.annot.gz",
    "GPL96": "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/GPL96/annot/GPL96.annot.gz",
    "GPL15048": (
        "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?"
        "acc=GPL15048&targ=self&form=text&view=full"
    ),
}


def _remote_size(url: str, timeout: int) -> int | None:
    try:
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = response.headers.get("Content-Length")
            return int(value) if value else None
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return None


def download_with_resume(
    url: str,
    destination: Path,
    *,
    timeout: int = 60,
    retries: int = 3,
) -> dict[str, object]:
    """Download one file, skipping completed files and resuming partial files."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    remote_size = _remote_size(url, timeout)
    if destination.exists() and (
        remote_size is None or destination.stat().st_size == remote_size
    ):
        return {
            "status": "already_present",
            "path": str(destination),
            "bytes": destination.stat().st_size,
            "url": url,
        }
    partial = destination.with_suffix(destination.suffix + ".part")
    if destination.exists():
        destination.replace(partial)
    if remote_size is not None and remote_size > 8 * 1024 * 1024 and not partial.exists():
        _download_segmented(
            url,
            destination,
            remote_size=remote_size,
            timeout=timeout,
            retries=retries,
        )
        return {
            "status": "downloaded",
            "path": str(destination),
            "bytes": destination.stat().st_size,
            "url": url,
        }
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(
            url, headers={"Range": f"bytes={offset}-"} if offset else {}
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                append = offset > 0 and getattr(response, "status", None) == 206
                with partial.open("ab" if append else "wb") as output:
                    shutil.copyfileobj(response, output)
            if remote_size is not None and partial.stat().st_size != remote_size:
                raise IOError(
                    f"Incomplete download: {partial.stat().st_size}/{remote_size} bytes"
                )
            partial.replace(destination)
            return {
                "status": "downloaded",
                "path": str(destination),
                "bytes": destination.stat().st_size,
                "url": url,
            }
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"Failed to download {url}: {last_error}") from last_error


def _download_segment(
    url: str,
    destination: Path,
    start: int,
    end: int,
    *,
    timeout: int,
    retries: int,
) -> None:
    expected_size = end - start + 1
    if destination.exists() and destination.stat().st_size == expected_size:
        return
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if getattr(response, "status", None) != 206:
                    raise IOError(f"Server did not honor byte range {start}-{end}.")
                with destination.open("wb") as output:
                    shutil.copyfileobj(response, output, length=1024 * 1024)
            if destination.stat().st_size != expected_size:
                raise IOError(
                    f"Incomplete segment {start}-{end}: {destination.stat().st_size} bytes"
                )
            return
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"Failed segment {start}-{end} for {url}: {last_error}") from last_error


def _download_segmented(
    url: str,
    destination: Path,
    *,
    remote_size: int,
    timeout: int,
    retries: int,
    chunk_size: int = 4 * 1024 * 1024,
    workers: int = 8,
) -> None:
    """Download independently resumable byte ranges and merge after validation."""

    parts_dir = destination.with_suffix(destination.suffix + ".parts")
    parts_dir.mkdir(parents=True, exist_ok=True)
    segments = []
    for index, start in enumerate(range(0, remote_size, chunk_size)):
        end = min(start + chunk_size - 1, remote_size - 1)
        segments.append((parts_dir / f"{index:05d}.part", start, end))
    with ThreadPoolExecutor(max_workers=min(workers, len(segments))) as executor:
        futures = [
            executor.submit(
                _download_segment,
                url,
                path,
                start,
                end,
                timeout=timeout,
                retries=retries,
            )
            for path, start, end in segments
        ]
        for future in as_completed(futures):
            future.result()
    partial = destination.with_suffix(destination.suffix + ".part")
    with partial.open("wb") as output:
        for path, _, _ in segments:
            with path.open("rb") as handle:
                shutil.copyfileobj(handle, output, length=1024 * 1024)
    if partial.stat().st_size != remote_size:
        raise IOError(
            f"Merged download has {partial.stat().st_size}/{remote_size} bytes: {url}"
        )
    partial.replace(destination)
    for path, _, _ in segments:
        path.unlink()
    parts_dir.rmdir()


def download_geo_inputs(
    root: Path,
    accessions: Iterable[str],
    *,
    timeout: int = 60,
    retries: int = 3,
) -> pd.DataFrame:
    """Download requested GEO series matrices and unique platform annotations."""

    records: list[dict[str, object]] = []
    platforms: set[str] = set()
    for accession in accessions:
        if accession not in GEO_COHORTS:
            raise ValueError(f"Unsupported GEO cohort: {accession}")
        spec = GEO_COHORTS[accession]
        platforms.add(spec.platform)
        result = download_with_resume(
            spec.matrix_url,
            Path(root) / accession / f"{accession}_series_matrix.txt.gz",
            timeout=timeout,
            retries=retries,
        )
        records.append(
            {"resource_type": "series_matrix", "accession": accession,
             "platform": spec.platform, **result}
        )
    for platform in sorted(platforms):
        url = PLATFORM_URLS[platform]
        suffix = ".annot.gz" if url.endswith(".gz") else ".txt"
        result = download_with_resume(
            url,
            Path(root) / "platforms" / f"{platform}{suffix}",
            timeout=timeout,
            retries=retries,
        )
        records.append(
            {"resource_type": "platform_annotation", "accession": platform,
             "platform": platform, **result}
        )
    return pd.DataFrame.from_records(records)
