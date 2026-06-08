"""Raw single-cell data discovery, download, conversion, and import utilities."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread


GSE131907_ACCESSION = "GSE131907"
GSE131907_BASE_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE131nnn/GSE131907/suppl"
)
GSE131907_FILES = {
    "annotation": {
        "name": "GSE131907_Lung_Cancer_cell_annotation.txt.gz",
        "url": f"{GSE131907_BASE_URL}/GSE131907_Lung_Cancer_cell_annotation.txt.gz",
        "size": 1_886_187,
    },
    "raw_rds": {
        "name": "GSE131907_Lung_Cancer_raw_UMI_matrix.rds.gz",
        "url": f"{GSE131907_BASE_URL}/GSE131907_Lung_Cancer_raw_UMI_matrix.rds.gz",
        "size": 633_500_069,
    },
}
SUPPORTED_SUFFIXES = {".h5ad", ".h5", ".rds", ".mtx", ".csv", ".tsv", ".loom", ".gz"}


class SCRNAImportError(RuntimeError):
    """Raised when raw single-cell data cannot be imported without guessing."""


def _rscript_command() -> list[str]:
    """Return a subprocess-safe Rscript command on Windows and POSIX."""

    executable = shutil.which("Rscript") or shutil.which("Rscript.cmd")
    if not executable:
        raise FileNotFoundError(
            "Rscript was not found on PATH. Install R or add its bin directory "
            "(or the Rscript.cmd wrapper) to PATH."
        )
    if executable.lower().endswith((".cmd", ".bat")):
        return ["cmd.exe", "/d", "/c", executable]
    return [executable]


def sanitize_anndata_strings(data):
    """Convert pandas Arrow/string extension arrays to h5py-compatible objects."""

    for frame in (data.obs, data.var):
        for column in frame.columns:
            dtype_name = str(frame[column].dtype).lower()
            if dtype_name == "str" or "string" in dtype_name or "arrowstring" in type(frame[column].array).__name__.lower():
                frame[column] = frame[column].astype(str).astype(object)
        frame.index = pd.Index(
            frame.index.astype(str).to_numpy(dtype=object),
            dtype=object,
            name=frame.index.name,
        )
    return data


@dataclass(frozen=True)
class SCRNAPaths:
    root: Path
    raw_dir: Path
    processed_dir: Path
    dataset_dir: Path
    annotation: Path
    raw_rds: Path
    imported_h5ad: Path
    raw_or_converted_h5ad: Path


def scrna_paths(root: str | Path = ".", *, small_test: bool = False) -> SCRNAPaths:
    project_root = Path(root).resolve()
    raw_dir = project_root / "data" / "raw" / "scrna_luad"
    processed_dir = project_root / "data" / "processed" / "scrna_luad"
    if small_test:
        raw_dir = raw_dir / "stage4b_small_test"
        processed_dir = processed_dir / "stage4b_small_test"
    dataset_dir = raw_dir / GSE131907_ACCESSION
    return SCRNAPaths(
        root=project_root,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        dataset_dir=dataset_dir,
        annotation=dataset_dir / GSE131907_FILES["annotation"]["name"],
        raw_rds=dataset_dir / GSE131907_FILES["raw_rds"]["name"],
        imported_h5ad=processed_dir / "scrna_luad_imported.h5ad",
        raw_or_converted_h5ad=processed_dir / "scrna_luad_raw_or_converted.h5ad",
    )


def discover_scrna_files(raw_dir: str | Path) -> list[Path]:
    directory = Path(raw_dir)
    if not directory.exists():
        return []
    files = []
    for path in directory.rglob("*"):
        if not path.is_file():
            continue
        suffixes = {suffix.lower() for suffix in path.suffixes}
        if suffixes & SUPPORTED_SUFFIXES:
            files.append(path)
    return sorted(files)


def _annotation_summary(path: Path) -> dict[str, object]:
    frame = pd.read_csv(path, sep="\t", compression="infer")
    return {
        "number_of_cells": int(len(frame)),
        "number_of_genes": np.nan,
        "has_cell_metadata": True,
        "has_cell_type_annotation": any(
            column in frame.columns
            for column in ("Cell_type", "Cell_type.refined", "Cell_subtype", "cell_type")
        ),
        "has_sample_annotation": any(
            column in frame.columns for column in ("Sample", "sample", "patient_id")
        ),
        "read_status": "readable_annotation",
    }


def _h5ad_summary(path: Path) -> dict[str, object]:
    try:
        import anndata as ad

        data = ad.read_h5ad(path, backed="r")
        columns = set(data.obs.columns)
        summary = {
            "number_of_cells": int(data.n_obs),
            "number_of_genes": int(data.n_vars),
            "has_cell_metadata": bool(len(columns)),
            "has_cell_type_annotation": bool(
                columns
                & {
                    "cell_type",
                    "Cell_type",
                    "Cell_type.refined",
                    "Cell_subtype",
                    "major_cell_type",
                }
            ),
            "has_sample_annotation": bool(
                columns & {"sample", "Sample", "patient_id", "donor_id"}
            ),
            "read_status": "readable_h5ad",
        }
        data.file.close()
        return summary
    except Exception as exc:
        return {
            "number_of_cells": np.nan,
            "number_of_genes": np.nan,
            "has_cell_metadata": False,
            "has_cell_type_annotation": False,
            "has_sample_annotation": False,
            "read_status": f"h5ad_read_error: {exc}",
        }


def build_data_inventory(raw_dir: str | Path) -> pd.DataFrame:
    rows = []
    for path in discover_scrna_files(raw_dir):
        lower_name = path.name.lower()
        summary = {
            "number_of_cells": np.nan,
            "number_of_genes": np.nan,
            "has_cell_metadata": False,
            "has_cell_type_annotation": False,
            "has_sample_annotation": False,
            "read_status": "not_inspected",
        }
        if lower_name.endswith(".h5ad"):
            summary = _h5ad_summary(path)
        elif "annotation" in lower_name and lower_name.endswith((".txt.gz", ".tsv.gz")):
            try:
                summary = _annotation_summary(path)
            except Exception as exc:
                summary["read_status"] = f"annotation_read_error: {exc}"
        elif lower_name.endswith(".rds.gz"):
            expected = (
                int(GSE131907_FILES["raw_rds"]["size"])
                if path.name == GSE131907_FILES["raw_rds"]["name"]
                else None
            )
            if expected and path.stat().st_size != expected:
                summary["read_status"] = (
                    f"incomplete_download:{path.stat().st_size}/{expected}_bytes"
                )
            else:
                summary["read_status"] = "rds_requires_R_conversion"
        elif lower_name.endswith(".rds"):
            summary["read_status"] = "rds_requires_R_conversion"
        elif lower_name.endswith(".mtx") or lower_name.endswith(".mtx.gz"):
            summary["read_status"] = "10x_matrix_candidate"
        rows.append(
            {
                "file_path": str(path.resolve()),
                "file_size_bytes": int(path.stat().st_size),
                "file_size_mb": round(path.stat().st_size / 1024**2, 3),
                "format": "".join(path.suffixes).lower() or "unknown",
                **summary,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "file_path",
            "file_size_bytes",
            "file_size_mb",
            "format",
            "number_of_cells",
            "number_of_genes",
            "has_cell_metadata",
            "has_cell_type_annotation",
            "has_sample_annotation",
            "read_status",
        ],
    )


def _run_curl_range(url: str, part: Path, start: int, end: int) -> None:
    expected = end - start + 1
    if part.exists() and part.stat().st_size == expected:
        return
    part.parent.mkdir(parents=True, exist_ok=True)
    partial = part.with_suffix(part.suffix + ".download")
    segment = part.with_suffix(part.suffix + ".segment")
    if segment.exists() and segment.stat().st_size:
        with partial.open("ab") as output, segment.open("rb") as source:
            shutil.copyfileobj(source, output, length=4 * 1024 * 1024)
        segment.unlink(missing_ok=True)
    current = partial.stat().st_size if partial.exists() else 0
    if current > expected:
        partial.unlink()
        current = 0
    if current == expected:
        partial.replace(part)
        return
    segment.unlink(missing_ok=True)
    request_start = start + current
    command = [
        "curl.exe",
        "-L",
        "--fail",
        "--retry",
        "10",
        "--retry-delay",
        "3",
        "--range",
        f"{request_start}-{end}",
        "--output",
        str(segment),
        url,
    ]
    result = subprocess.run(command, check=False)
    if segment.exists() and segment.stat().st_size:
        with partial.open("ab") as output, segment.open("rb") as source:
            shutil.copyfileobj(source, output, length=4 * 1024 * 1024)
        segment.unlink(missing_ok=True)
    if result.returncode != 0:
        raise SCRNAImportError(
            f"curl range download failed for bytes {request_start}-{end} "
            f"(code {result.returncode}); partial bytes were retained."
        )
    if not partial.exists() or partial.stat().st_size != expected:
        actual = partial.stat().st_size if partial.exists() else 0
        raise SCRNAImportError(
            f"Range {start}-{end} has {actual} bytes; expected {expected}."
        )
    partial.replace(part)


def download_with_resume(
    url: str,
    destination: str | Path,
    *,
    expected_size: int,
    workers: int = 8,
    chunk_size: int = 8 * 1024 * 1024,
) -> Path:
    """Download a large HTTP file with range-based resumable parts."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size == expected_size:
        return destination
    if destination.exists() and destination.stat().st_size > expected_size:
        raise SCRNAImportError(f"Existing file is larger than expected: {destination}")

    prefix_size = destination.stat().st_size if destination.exists() else 0
    parts_dir = destination.with_name(destination.name + ".parts")
    parts_dir.mkdir(parents=True, exist_ok=True)
    ranges = []
    start = prefix_size
    while start < expected_size:
        end = min(start + chunk_size - 1, expected_size - 1)
        part = parts_dir / f"{start:012d}-{end:012d}.part"
        ranges.append((start, end, part))
        start = end + 1

    errors = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(_run_curl_range, url, part, start, end): (start, end)
            for start, end, part in ranges
        }
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                errors.append(f"{futures[future]}: {exc}")
    if errors:
        raise SCRNAImportError(
            "Download incomplete; rerun to resume. First error: " + errors[0]
        )

    mode = "ab" if prefix_size else "wb"
    with destination.open(mode) as output:
        for _, _, part in ranges:
            with part.open("rb") as source:
                shutil.copyfileobj(source, output, length=16 * 1024 * 1024)
    if destination.stat().st_size != expected_size:
        raise SCRNAImportError(
            f"Downloaded size mismatch: {destination.stat().st_size} != {expected_size}"
        )
    shutil.rmtree(parts_dir, ignore_errors=True)
    return destination


def download_gse131907(paths: SCRNAPaths, *, workers: int = 8) -> list[Path]:
    paths.dataset_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for key, destination in (
        ("annotation", paths.annotation),
        ("raw_rds", paths.raw_rds),
    ):
        metadata = GSE131907_FILES[key]
        downloaded.append(
            download_with_resume(
                metadata["url"],
                destination,
                expected_size=int(metadata["size"]),
                workers=1 if key == "annotation" else workers,
                chunk_size=2 * 1024 * 1024 if key == "annotation" else 8 * 1024 * 1024,
            )
        )
    return downloaded


def file_md5(path: str | Path, *, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _prepare_rds_payload(path: str | Path) -> tuple[Path, bool, int]:
    """Expose the RDS payload and validate any outer gzip wrapper."""

    source = Path(path).resolve()
    with source.open("rb") as handle:
        raw_magic = handle.read(2)
    if raw_magic != b"\x1f\x8b":
        return source, False, 0

    with gzip.open(source, "rb") as handle:
        inner_magic = handle.read(2)
    if inner_magic != b"\x1f\x8b":
        return source, True, 1

    cache_dir = source.parent / ".stage4b_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = cache_dir / f"{source.stem}.inner.gz"
    if not payload.exists():
        temporary = payload.with_suffix(payload.suffix + ".download")
        temporary.unlink(missing_ok=True)
        with gzip.open(source, "rb") as input_handle, temporary.open("wb") as output_handle:
            shutil.copyfileobj(input_handle, output_handle, length=16 * 1024 * 1024)
        temporary.replace(payload)
    return payload, True, 2


def inspect_rds_with_r(path: str | Path) -> dict[str, object]:
    """Read the compressed RDS in R and report class, dimensions, and names."""

    payload, compressed, compression_layers = _prepare_rds_payload(path)
    r_code = r"""
args <- commandArgs(trailingOnly=TRUE)
if (args[[2]] == "TRUE") {
  con <- gzfile(args[[1]], open="rb")
} else {
  con <- file(args[[1]], open="rb")
}
obj <- readRDS(con)
close(con)
cat("class=", paste(class(obj), collapse=";"), "\n", sep="")
d <- dim(obj)
if (is.null(d)) {
  cat("nrow=NA\nncol=NA\n")
} else {
  cat("nrow=", d[[1]], "\n", sep="")
  cat("ncol=", d[[2]], "\n", sep="")
}
cat("has_rownames=", !is.null(rownames(obj)), "\n", sep="")
cat("has_colnames=", !is.null(colnames(obj)), "\n", sep="")
cat("object_size_bytes=", as.numeric(object.size(obj)), "\n", sep="")
if (is.data.frame(obj) && ncol(obj) > 0 && nrow(obj) > 0) {
  sample_columns <- unique(round(seq(1, ncol(obj), length.out=min(100, ncol(obj)))))
  sample_nnz <- sum(vapply(
    obj[sample_columns],
    function(values) sum(!is.na(values) & values != 0),
    numeric(1)
  ))
  sample_entries <- nrow(obj) * length(sample_columns)
  cat("sampled_columns=", length(sample_columns), "\n", sep="")
  cat("sampled_nonzero_fraction=", sample_nnz / sample_entries, "\n", sep="")
  cat("first_column_class=", paste(class(obj[[1]]), collapse=";"), "\n", sep="")
}
"""
    with tempfile.TemporaryDirectory(prefix="stage4b_rds_inspect_") as temp:
        script = Path(temp) / "inspect.R"
        script.write_text(r_code, encoding="utf-8")
        result = subprocess.run(
            [
                *_rscript_command(),
                str(script),
                str(payload),
                "TRUE" if compressed else "FALSE",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    values: dict[str, object] = {
        "rds_readable": result.returncode == 0,
        "rds_error": result.stderr.strip()[-2000:] if result.returncode else "",
        "compression_layers": compression_layers,
        "rds_payload_path": str(payload),
    }
    for line in result.stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    for key in ("nrow", "ncol", "object_size_bytes", "sampled_columns"):
        value = values.get(key)
        values[key] = int(value) if str(value).isdigit() else np.nan
    if "sampled_nonzero_fraction" in values:
        values["sampled_nonzero_fraction"] = float(values["sampled_nonzero_fraction"])
    return values


def annotation_matrix_match(
    annotation_path: str | Path,
    barcodes_path: str | Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    annotation = _read_annotation(Path(annotation_path))
    barcodes = (
        pd.read_csv(barcodes_path, sep="\t", header=None)[0]
        .astype(str)
        .drop_duplicates()
    )
    matrix_set = set(barcodes)
    annotation_set = set(annotation.index.astype(str))
    matched = matrix_set & annotation_set
    matrix_only = matrix_set - annotation_set
    annotation_only = annotation_set - matrix_set
    rows = [
        {
            "check": "matrix_cells",
            "value": len(matrix_set),
            "status": "observed",
        },
        {
            "check": "annotation_cells",
            "value": len(annotation_set),
            "status": "observed",
        },
        {
            "check": "matched_cells",
            "value": len(matched),
            "status": "pass" if len(matched) >= 0.95 * len(matrix_set) else "review",
        },
        {
            "check": "matrix_only_cells",
            "value": len(matrix_only),
            "status": "pass" if not matrix_only else "review",
        },
        {
            "check": "annotation_only_cells",
            "value": len(annotation_only),
            "status": "pass" if not annotation_only else "review",
        },
        {
            "check": "matrix_match_fraction",
            "value": len(matched) / len(matrix_set) if matrix_set else np.nan,
            "status": "pass"
            if matrix_set and len(matched) >= 0.95 * len(matrix_set)
            else "review",
        },
    ]
    details = {
        "matrix_cells": len(matrix_set),
        "annotation_cells": len(annotation_set),
        "matched_cells": len(matched),
        "matrix_only_cells": len(matrix_only),
        "annotation_only_cells": len(annotation_only),
        "matrix_match_fraction": len(matched) / len(matrix_set) if matrix_set else np.nan,
    }
    return pd.DataFrame(rows), details


def export_rds_dimnames_with_r(
    path: str | Path,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Export gene and cell names without copying the expression matrix."""

    payload, compressed, _ = _prepare_rds_payload(path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    features_path = output_dir / "features.tsv"
    barcodes_path = output_dir / "barcodes.tsv"
    if features_path.exists() and barcodes_path.exists():
        return features_path, barcodes_path
    r_code = r"""
args <- commandArgs(trailingOnly=TRUE)
input <- args[[1]]
output <- args[[2]]
compressed <- args[[3]] == "TRUE"
if (compressed) {
  con <- gzfile(input, open="rb")
} else {
  con <- file(input, open="rb")
}
obj <- readRDS(con)
close(con)
if (is.null(rownames(obj)) || is.null(colnames(obj))) {
  stop("RDS object does not contain both row names and column names.")
}
dir.create(output, recursive=TRUE, showWarnings=FALSE)
write.table(
  rownames(obj), file.path(output, "features.tsv"),
  quote=FALSE, row.names=FALSE, col.names=FALSE, sep="\t"
)
write.table(
  colnames(obj), file.path(output, "barcodes.tsv"),
  quote=FALSE, row.names=FALSE, col.names=FALSE, sep="\t"
)
"""
    with tempfile.TemporaryDirectory(prefix="stage4b_rds_dimnames_") as temp:
        script = Path(temp) / "export_dimnames.R"
        script.write_text(r_code, encoding="utf-8")
        result = subprocess.run(
            [
                *_rscript_command(),
                str(script),
                str(payload),
                str(output_dir),
                "TRUE" if compressed else "FALSE",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        raise SCRNAImportError(
            "R could not export RDS dimnames: "
            + (result.stderr.strip() or result.stdout.strip())
        )
    return features_path, barcodes_path


def _chunk_metadata_files(work_dir: Path) -> list[Path]:
    return sorted((work_dir / "chunks").glob("chunk_*.done.tsv"))


def export_rds_sparse_chunks_with_r(
    path: str | Path,
    work_dir: str | Path,
    *,
    chunk_size: int = 256,
    log_path: str | Path | None = None,
) -> list[Path]:
    """Export cell-major sparse arrays in resumable chunks without a full copy."""

    payload, compressed, _ = _prepare_rds_payload(path)
    work_dir = Path(work_dir).resolve()
    chunks_dir = work_dir / "chunks"
    temp_dir = work_dir / "r_temp"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    script_path = work_dir / "export_sparse_chunks.R"
    script_path.write_text(
        r"""
args <- commandArgs(trailingOnly=TRUE)
input <- args[[1]]
output <- args[[2]]
chunk_size <- as.integer(args[[3]])
compressed <- args[[4]] == "TRUE"
dir.create(output, recursive=TRUE, showWarnings=FALSE)

if (compressed) {
  con <- gzfile(input, open="rb")
} else {
  con <- file(input, open="rb")
}
obj <- readRDS(con)
close(con)
if (!is.data.frame(obj) && !is.matrix(obj)) {
  stop(paste("Expected a data.frame or matrix; observed", paste(class(obj), collapse=",")))
}
if (is.null(rownames(obj)) || is.null(colnames(obj))) {
  stop("RDS matrix lacks row or column names.")
}

write_binary <- function(values, path) {
  temp_path <- paste0(path, ".incomplete")
  if (file.exists(temp_path)) file.remove(temp_path)
  handle <- file(temp_path, open="wb")
  writeBin(as.integer(values), handle, size=4L, endian="little")
  close(handle)
  if (file.exists(path)) file.remove(path)
  if (!file.rename(temp_path, path)) stop(paste("Could not finalize", path))
}

n_genes <- nrow(obj)
n_cells <- ncol(obj)
n_chunks <- ceiling(n_cells / chunk_size)
cat("RDS_LOADED", n_genes, n_cells, as.numeric(object.size(obj)), "\n")
flush.console()

for (chunk_id in seq_len(n_chunks)) {
  start <- (chunk_id - 1L) * chunk_size + 1L
  end <- min(chunk_id * chunk_size, n_cells)
  prefix <- file.path(output, sprintf("chunk_%05d", chunk_id))
  data_path <- paste0(prefix, ".data.bin")
  indices_path <- paste0(prefix, ".indices.bin")
  indptr_path <- paste0(prefix, ".indptr.bin")
  done_path <- paste0(prefix, ".done.tsv")
  if (
    file.exists(done_path) && file.exists(data_path) &&
    file.exists(indices_path) && file.exists(indptr_path)
  ) {
    if (chunk_id %% 25L == 0L || chunk_id == n_chunks) {
      cat("SKIP", chunk_id, n_chunks, start, end, "\n")
      flush.console()
    }
    next
  }

  matrix_chunk <- as.matrix(obj[, start:end, drop=FALSE])
  if (anyNA(matrix_chunk)) stop(paste("NA values in chunk", chunk_id))
  if (any(!is.finite(matrix_chunk))) stop(paste("Inf values in chunk", chunk_id))
  if (any(matrix_chunk < 0L)) stop(paste("Negative counts in chunk", chunk_id))
  positions <- which(matrix_chunk != 0L, arr.ind=TRUE)
  if (length(positions) == 0L) {
    values <- integer(0)
    indices <- integer(0)
    counts <- integer(ncol(matrix_chunk))
    observed_min <- 0L
    observed_max <- 0L
  } else {
    values <- as.integer(matrix_chunk[positions])
    indices <- as.integer(positions[, 1L] - 1L)
    counts <- tabulate(positions[, 2L], nbins=ncol(matrix_chunk))
    observed_min <- min(values)
    observed_max <- max(values)
  }
  indptr <- c(0, cumsum(counts))
  if (tail(indptr, 1L) > .Machine$integer.max) {
    stop(paste("Chunk nnz exceeds int32 in chunk", chunk_id))
  }
  write_binary(values, data_path)
  write_binary(indices, indices_path)
  write_binary(indptr, indptr_path)

  metadata <- data.frame(
    chunk_id=chunk_id,
    start_cell_1based=start,
    end_cell_1based=end,
    n_cells=end - start + 1L,
    n_genes=n_genes,
    nnz=length(values),
    min_value=observed_min,
    max_value=observed_max,
    nan_count=0L,
    inf_count=0L,
    negative_count=0L
  )
  temp_done <- paste0(done_path, ".incomplete")
  write.table(
    metadata, temp_done, sep="\t", quote=FALSE,
    row.names=FALSE, col.names=TRUE
  )
  if (file.exists(done_path)) file.remove(done_path)
  if (!file.rename(temp_done, done_path)) {
    stop(paste("Could not finalize metadata for chunk", chunk_id))
  }
  rm(matrix_chunk, positions, values, indices, counts, indptr)
  if (chunk_id %% 10L == 0L || chunk_id == n_chunks) {
    cat("DONE", chunk_id, n_chunks, start, end, "\n")
    flush.console()
  }
  if (chunk_id %% 25L == 0L) gc(verbose=FALSE)
}
cat("EXPORT_COMPLETE", n_genes, n_cells, n_chunks, "\n")
""",
        encoding="utf-8",
    )
    log_path = Path(log_path or (work_dir / "r_chunk_export.log")).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["TMP"] = str(temp_dir)
    environment["TEMP"] = str(temp_dir)
    environment["TMPDIR"] = str(temp_dir)
    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write(
            f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting/resuming R export\n"
        )
        log_handle.flush()
        result = subprocess.run(
            [
                *_rscript_command(),
                str(script_path),
                str(payload),
                str(chunks_dir),
                str(int(chunk_size)),
                "TRUE" if compressed else "FALSE",
            ],
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
            env=environment,
        )
    if result.returncode != 0:
        raise SCRNAImportError(
            f"R sparse chunk export failed with code {result.returncode}. "
            f"See {log_path}."
        )
    metadata_files = _chunk_metadata_files(work_dir)
    if not metadata_files:
        raise SCRNAImportError("R export completed without chunk metadata.")
    return metadata_files


def _read_chunk_manifest(work_dir: Path) -> pd.DataFrame:
    frames = [pd.read_csv(path, sep="\t") for path in _chunk_metadata_files(work_dir)]
    if not frames:
        raise SCRNAImportError("No completed sparse chunks were found.")
    manifest = pd.concat(frames, ignore_index=True).sort_values("chunk_id")
    manifest = manifest.reset_index(drop=True)
    expected_start = 1
    for row in manifest.itertuples(index=False):
        if int(row.start_cell_1based) != expected_start:
            raise SCRNAImportError(
                f"Sparse chunks are not contiguous at cell {expected_start}."
            )
        expected_start = int(row.end_cell_1based) + 1
        prefix = work_dir / "chunks" / f"chunk_{int(row.chunk_id):05d}"
        expected_sizes = {
            prefix.with_suffix(".data.bin"): int(row.nnz) * 4,
            prefix.with_suffix(".indices.bin"): int(row.nnz) * 4,
            prefix.with_suffix(".indptr.bin"): (int(row.n_cells) + 1) * 4,
        }
        for path, expected_size in expected_sizes.items():
            if not path.exists() or path.stat().st_size != expected_size:
                actual = path.stat().st_size if path.exists() else -1
                raise SCRNAImportError(
                    f"Chunk file size mismatch for {path}: {actual} != {expected_size}."
                )
    return manifest


def _gse131907_obs(annotation_path: Path, barcodes: pd.Index) -> pd.DataFrame:
    annotation = _read_annotation(annotation_path)
    obs = annotation.reindex(barcodes).copy()
    if obs.isna().all(axis=1).any():
        missing = int(obs.isna().all(axis=1).sum())
        raise SCRNAImportError(f"{missing} cell barcodes lack official annotation.")
    obs.insert(0, "cell_barcode", barcodes.astype(str))
    if "Sample" in obs.columns:
        obs["sample_id"] = obs["Sample"].astype(str)
        patient_number = obs["Sample"].astype(str).str.extract(r"(\d+)$", expand=False)
        obs["patient_id"] = "P" + patient_number.fillna("unknown").str.zfill(2)
    obs.index = pd.Index(barcodes.astype(str), name="cell_barcode_index")
    return obs


def assemble_sparse_chunks_to_h5ad(
    work_dir: str | Path,
    annotation_path: str | Path,
    features_path: str | Path,
    barcodes_path: str | Path,
    destination: str | Path,
) -> dict[str, object]:
    """Assemble validated raw chunks into an AnnData CSR file atomically."""

    import anndata as ad
    import h5py

    work_dir = Path(work_dir).resolve()
    destination = Path(destination).resolve()
    incomplete = destination.with_name(destination.name + ".incomplete")
    manifest = _read_chunk_manifest(work_dir)
    n_obs = int(manifest["n_cells"].sum())
    n_vars_values = manifest["n_genes"].astype(int).unique()
    if len(n_vars_values) != 1:
        raise SCRNAImportError("Sparse chunks disagree on gene count.")
    n_vars = int(n_vars_values[0])
    total_nnz = int(manifest["nnz"].sum())
    if total_nnz > np.iinfo(np.int32).max:
        raise SCRNAImportError("Total nnz exceeds the current int32 CSR limit.")
    features = pd.read_csv(features_path, sep="\t", header=None)[0].astype(str)
    barcodes = pd.read_csv(barcodes_path, sep="\t", header=None)[0].astype(str)
    if len(features) != n_vars or len(barcodes) != n_obs:
        raise SCRNAImportError(
            f"Dimnames mismatch: {len(barcodes)} cells/{len(features)} genes "
            f"versus {n_obs}/{n_vars} sparse dimensions."
        )
    obs = _gse131907_obs(Path(annotation_path), pd.Index(barcodes))
    var = pd.DataFrame(index=pd.Index(features, name="gene_symbol"))
    skeleton = ad.AnnData(
        X=sparse.csr_matrix((n_obs, n_vars), dtype=np.int32),
        obs=obs,
        var=var,
    )
    skeleton.var_names_make_unique()
    skeleton.uns["dataset_accession"] = GSE131907_ACCESSION
    skeleton.uns["data_source"] = "NCBI_GEO_official_raw_UMI_RDS"
    skeleton.uns["annotation_source"] = "GSE131907 official cell annotation"
    skeleton.uns["conversion"] = "resumable_R_chunks_to_AnnData_CSR"
    destination.parent.mkdir(parents=True, exist_ok=True)
    incomplete.unlink(missing_ok=True)
    sanitize_anndata_strings(skeleton).write_h5ad(
        incomplete, compression="gzip", compression_opts=4
    )
    del skeleton, obs, var

    offset = 0
    row_offset = 0
    try:
        with h5py.File(incomplete, "r+") as handle:
            data_set = handle["X/data"]
            indices_set = handle["X/indices"]
            indptr_set = handle["X/indptr"]
            data_set.resize((total_nnz,))
            indices_set.resize((total_nnz,))
            indptr_set.resize((n_obs + 1,))
            indptr_set[0] = 0
            handle["X"].attrs["shape"] = np.asarray([n_obs, n_vars], dtype=np.int64)
            for row in manifest.itertuples(index=False):
                prefix = work_dir / "chunks" / f"chunk_{int(row.chunk_id):05d}"
                values = np.fromfile(prefix.with_suffix(".data.bin"), dtype="<i4")
                indices = np.fromfile(prefix.with_suffix(".indices.bin"), dtype="<i4")
                local_indptr = np.fromfile(
                    prefix.with_suffix(".indptr.bin"), dtype="<i4"
                )
                if (
                    len(values) != int(row.nnz)
                    or len(indices) != int(row.nnz)
                    or len(local_indptr) != int(row.n_cells) + 1
                    or int(local_indptr[-1]) != int(row.nnz)
                ):
                    raise SCRNAImportError(
                        f"Chunk {row.chunk_id} failed CSR array validation."
                    )
                next_offset = offset + len(values)
                data_set[offset:next_offset] = values
                indices_set[offset:next_offset] = indices
                next_row = row_offset + int(row.n_cells)
                indptr_set[row_offset + 1 : next_row + 1] = (
                    local_indptr[1:].astype(np.int64) + offset
                ).astype(np.int32)
                offset = next_offset
                row_offset = next_row
            if offset != total_nnz or row_offset != n_obs:
                raise SCRNAImportError("Final CSR offsets do not match manifest totals.")
            handle.flush()
        destination.unlink(missing_ok=True)
        incomplete.replace(destination)
    except Exception:
        failed = destination.with_name(destination.name + ".failed_incomplete")
        failed.unlink(missing_ok=True)
        if incomplete.exists():
            incomplete.replace(failed)
        raise
    return {
        "h5ad_path": str(destination),
        "n_obs": n_obs,
        "n_vars": n_vars,
        "nnz": total_nnz,
        "density": total_nnz / (n_obs * n_vars),
        "chunk_count": len(manifest),
    }


def validate_gse131907_h5ad(
    h5ad_path: str | Path,
    annotation_path: str | Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Validate structure, metadata, barcodes, and finite raw counts."""

    import anndata as ad
    import h5py

    h5ad_path = Path(h5ad_path).resolve()
    rows: list[dict[str, object]] = []
    data = ad.read_h5ad(h5ad_path, backed="r")
    try:
        n_obs, n_vars = data.shape
        obs_columns = list(data.obs.columns)
        obs_names = pd.Index(data.obs_names.astype(str))
        var_names = pd.Index(data.var_names.astype(str))
        x_exists = data.X is not None
        sample_fields = [name for name in ("Sample", "sample_id") if name in obs_columns]
        patient_fields = [name for name in ("patient_id",) if name in obs_columns]
        cell_type_fields = [
            name
            for name in ("Cell_type", "Cell_type.refined", "Cell_subtype")
            if name in obs_columns
        ]
        sample_count = (
            int(data.obs[sample_fields[0]].astype(str).nunique())
            if sample_fields
            else 0
        )
        patient_count = (
            int(data.obs[patient_fields[0]].astype(str).nunique())
            if patient_fields
            else 0
        )
        cell_barcode_present = "cell_barcode" in obs_columns
        sample_missing = (
            int(data.obs[sample_fields[0]].isna().sum()) if sample_fields else n_obs
        )
        patient_missing = (
            int(data.obs[patient_fields[0]].isna().sum())
            if patient_fields
            else n_obs
        )
        cell_type_missing = {
            field: int(data.obs[field].isna().sum()) for field in cell_type_fields
        }
    finally:
        data.file.close()
    annotation = _read_annotation(Path(annotation_path))
    annotation_names = pd.Index(annotation.index.astype(str))
    matched = len(set(obs_names) & set(annotation_names))
    matrix_only = len(set(obs_names) - set(annotation_names))
    annotation_only = len(set(annotation_names) - set(obs_names))
    nonfinite = 0
    negative = 0
    value_min = None
    value_max = None
    nnz = 0
    with h5py.File(h5ad_path, "r") as handle:
        values = handle["X/data"]
        nnz = int(len(values))
        step = 8_000_000
        for start in range(0, nnz, step):
            block = values[start : min(start + step, nnz)]
            nonfinite += int((~np.isfinite(block)).sum())
            negative += int((block < 0).sum())
            if len(block):
                block_min = int(block.min())
                block_max = int(block.max())
                value_min = block_min if value_min is None else min(value_min, block_min)
                value_max = block_max if value_max is None else max(value_max, block_max)
    checks = [
        ("h5ad_readable", True, True, "pass"),
        ("n_obs", n_obs, 208506, "pass" if n_obs == 208506 else "fail"),
        ("n_vars", n_vars, "10000-60000", "pass" if 10000 <= n_vars <= 60000 else "fail"),
        ("X_exists", x_exists, True, "pass" if x_exists else "fail"),
        ("X_nnz", nnz, ">0", "pass" if nnz > 0 else "fail"),
        ("X_nonfinite_values", nonfinite, 0, "pass" if nonfinite == 0 else "fail"),
        ("X_negative_values", negative, 0, "pass" if negative == 0 else "fail"),
        ("obs_cell_barcode_field", cell_barcode_present, True, "pass" if cell_barcode_present else "fail"),
        ("obs_names_unique", obs_names.is_unique, True, "pass" if obs_names.is_unique else "fail"),
        ("var_names_unique", var_names.is_unique, True, "pass" if var_names.is_unique else "fail"),
        ("barcode_matches", matched, 208506, "pass" if matched == 208506 else "fail"),
        ("matrix_only_barcodes", matrix_only, 0, "pass" if matrix_only == 0 else "fail"),
        ("annotation_only_barcodes", annotation_only, 0, "pass" if annotation_only == 0 else "fail"),
        ("sample_metadata", ",".join(sample_fields), "present", "pass" if sample_fields else "fail"),
        ("sample_metadata_missing", sample_missing, 0, "pass" if sample_missing == 0 else "fail"),
        ("patient_metadata", ",".join(patient_fields), "present", "pass" if patient_fields else "fail"),
        ("patient_metadata_missing", patient_missing, 0, "pass" if patient_missing == 0 else "fail"),
        ("cell_type_metadata", ",".join(cell_type_fields), "present", "pass" if cell_type_fields else "fail"),
        (
            "cell_type_metadata_missing",
            json.dumps(cell_type_missing, sort_keys=True),
            "source missingness retained",
            "pass" if cell_type_fields else "fail",
        ),
    ]
    for check, observed, expected, status in checks:
        rows.append(
            {
                "check": check,
                "observed": observed,
                "expected": expected,
                "status": status,
            }
        )
    details = {
        "h5ad_path": str(h5ad_path),
        "file_size_bytes": h5ad_path.stat().st_size,
        "n_obs": n_obs,
        "n_vars": n_vars,
        "nnz": nnz,
        "density": nnz / (n_obs * n_vars),
        "value_min": value_min,
        "value_max": value_max,
        "nonfinite_values": nonfinite,
        "negative_values": negative,
        "matched_barcodes": matched,
        "matrix_only_barcodes": matrix_only,
        "annotation_only_barcodes": annotation_only,
        "sample_fields": ",".join(sample_fields),
        "patient_fields": ",".join(patient_fields),
        "cell_type_fields": ",".join(cell_type_fields),
        "sample_count": sample_count,
        "patient_count": patient_count,
        "sample_metadata_missing": sample_missing,
        "patient_metadata_missing": patient_missing,
        "cell_type_metadata_missing": json.dumps(cell_type_missing, sort_keys=True),
        "all_checks_pass": all(row["status"] == "pass" for row in rows),
    }
    return pd.DataFrame(rows), details


def convert_gse131907_to_h5ad_chunked(
    paths: SCRNAPaths,
    *,
    work_dir: str | Path,
    chunk_size: int = 256,
    log_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Run resumable R chunk export, atomic h5ad assembly, and validation."""

    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    dimnames_dir = paths.processed_dir / "gse131907_dimnames"
    features_path, barcodes_path = export_rds_dimnames_with_r(
        paths.raw_rds, dimnames_dir
    )
    if not paths.raw_or_converted_h5ad.exists():
        export_rds_sparse_chunks_with_r(
            paths.raw_rds,
            work_dir,
            chunk_size=chunk_size,
            log_path=log_path,
        )
        assembly = assemble_sparse_chunks_to_h5ad(
            work_dir,
            paths.annotation,
            features_path,
            barcodes_path,
            paths.raw_or_converted_h5ad,
        )
    else:
        assembly = {"h5ad_path": str(paths.raw_or_converted_h5ad)}
    checks, details = validate_gse131907_h5ad(
        paths.raw_or_converted_h5ad, paths.annotation
    )
    details.update(assembly)
    if not details["all_checks_pass"]:
        failed = paths.raw_or_converted_h5ad.with_name(
            paths.raw_or_converted_h5ad.name + ".failed_integrity"
        )
        failed.unlink(missing_ok=True)
        paths.raw_or_converted_h5ad.replace(failed)
        raise SCRNAImportError(
            f"Generated h5ad failed integrity checks and was moved to {failed}."
        )
    return checks, details


def _rds_to_matrix_market(rds_gz: Path, output_dir: Path) -> tuple[Path, Path, Path]:
    """Use base R and Matrix to convert an R sparse matrix to Matrix Market."""

    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = output_dir / "matrix.mtx"
    features_path = output_dir / "features.tsv"
    barcodes_path = output_dir / "barcodes.tsv"
    if matrix_path.exists() and features_path.exists() and barcodes_path.exists():
        return matrix_path, features_path, barcodes_path

    payload, compressed, _ = _prepare_rds_payload(rds_gz)
    r_code = r"""
args <- commandArgs(trailingOnly=TRUE)
input <- args[[1]]
output <- args[[2]]
compressed <- args[[3]] == "TRUE"
suppressPackageStartupMessages(library(Matrix))
if (compressed) {
  con <- gzfile(input, open="rb")
} else {
  con <- file(input, open="rb")
}
obj <- readRDS(con)
close(con)
if (inherits(obj, "Seurat")) stop("Seurat object requires Seurat conversion.")
if (inherits(obj, "SingleCellExperiment")) {
  if (!requireNamespace("SummarizedExperiment", quietly=TRUE)) {
    stop("SingleCellExperiment requires SummarizedExperiment.")
  }
  obj <- SummarizedExperiment::assay(obj, "counts")
}
if (!inherits(obj, "Matrix") && !is.matrix(obj)) {
  stop(paste("Unsupported RDS object class:", paste(class(obj), collapse=",")))
}
obj <- as(obj, "dgCMatrix")
dir.create(output, recursive=TRUE, showWarnings=FALSE)
Matrix::writeMM(obj, file.path(output, "matrix.mtx"))
write.table(rownames(obj), file.path(output, "features.tsv"),
            quote=FALSE, row.names=FALSE, col.names=FALSE, sep="\t")
write.table(colnames(obj), file.path(output, "barcodes.tsv"),
            quote=FALSE, row.names=FALSE, col.names=FALSE, sep="\t")
cat(nrow(obj), ncol(obj), class(obj), "\n")
"""
    with tempfile.TemporaryDirectory(prefix="stage4b_rds_") as temp:
        script = Path(temp) / "convert_rds.R"
        script.write_text(r_code, encoding="utf-8")
        result = subprocess.run(
            [
                *_rscript_command(),
                str(script),
                str(payload),
                str(output_dir),
                "TRUE" if compressed else "FALSE",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        raise SCRNAImportError(
            "RDS conversion failed. "
            f"stdout={result.stdout[-1000:]} stderr={result.stderr[-2000:]}"
        )
    return matrix_path, features_path, barcodes_path


def _read_annotation(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"GSE131907 annotation is missing: {path}")
    frame = pd.read_csv(path, sep="\t", compression="infer")
    if "Index" not in frame.columns:
        raise SCRNAImportError("GSE131907 annotation lacks the Index cell identifier.")
    frame["Index"] = frame["Index"].astype(str)
    frame = frame.drop_duplicates("Index", keep="first").set_index("Index")
    return frame


def import_gse131907(paths: SCRNAPaths) -> Path:
    """Convert the official raw UMI RDS and join official cell annotations."""

    import anndata as ad

    if paths.raw_or_converted_h5ad.exists():
        return paths.raw_or_converted_h5ad
    if not paths.raw_rds.exists() or paths.raw_rds.stat().st_size != GSE131907_FILES["raw_rds"]["size"]:
        raise SCRNAImportError("Official GSE131907 raw UMI RDS is incomplete.")
    converted = paths.processed_dir / "gse131907_matrix_market"
    matrix_path, features_path, barcodes_path = _rds_to_matrix_market(
        paths.raw_rds, converted
    )
    matrix = mmread(matrix_path).tocsr()
    features = pd.read_csv(features_path, sep="\t", header=None)[0].astype(str)
    barcodes = pd.read_csv(barcodes_path, sep="\t", header=None)[0].astype(str)
    annotation = _read_annotation(paths.annotation)

    if matrix.shape == (len(features), len(barcodes)):
        matrix = matrix.transpose().tocsr()
    elif matrix.shape != (len(barcodes), len(features)):
        raise SCRNAImportError(
            f"Matrix shape {matrix.shape} is incompatible with "
            f"{len(features)} features and {len(barcodes)} barcodes."
        )
    obs = annotation.reindex(barcodes.to_numpy()).copy()
    obs.index = barcodes.to_numpy()
    matched = int(obs["Sample"].notna().sum()) if "Sample" in obs else 0
    if matched < int(0.95 * len(obs)):
        raise SCRNAImportError(
            f"Only {matched}/{len(obs)} matrix cells matched official annotations."
        )
    var = pd.DataFrame(index=pd.Index(features.to_numpy(), name="gene_symbol"))
    data = ad.AnnData(X=matrix, obs=obs, var=var)
    data.var_names_make_unique()
    data.uns["dataset_accession"] = GSE131907_ACCESSION
    data.uns["data_source"] = "NCBI_GEO_official_raw_UMI_RDS"
    data.uns["annotation_source"] = "GSE131907 official cell annotation"
    paths.processed_dir.mkdir(parents=True, exist_ok=True)
    sanitize_anndata_strings(data).write_h5ad(
        paths.raw_or_converted_h5ad, compression="gzip"
    )
    if not paths.imported_h5ad.exists():
        shutil.copy2(paths.raw_or_converted_h5ad, paths.imported_h5ad)
    return paths.raw_or_converted_h5ad


def import_best_available(paths: SCRNAPaths) -> tuple[Path, str]:
    """Import the highest-priority readable local single-cell format."""

    import anndata as ad
    import scanpy as sc

    if paths.imported_h5ad.exists():
        return paths.imported_h5ad, "existing_imported_h5ad"
    candidates = discover_scrna_files(paths.raw_dir)
    h5ad_files = [path for path in candidates if path.name.lower().endswith(".h5ad")]
    if h5ad_files:
        data = ad.read_h5ad(h5ad_files[0])
        paths.processed_dir.mkdir(parents=True, exist_ok=True)
        sanitize_anndata_strings(data).write_h5ad(paths.imported_h5ad, compression="gzip")
        return paths.imported_h5ad, "local_h5ad"
    if (
        paths.raw_rds.exists()
        and paths.raw_rds.stat().st_size == GSE131907_FILES["raw_rds"]["size"]
        and paths.annotation.exists()
    ):
        return import_gse131907(paths), "GSE131907_official_raw_RDS"

    loom_files = [path for path in candidates if path.suffix.lower() == ".loom"]
    h5_files = [
        path
        for path in candidates
        if path.suffix.lower() == ".h5" and not path.name.lower().endswith(".h5ad")
    ]
    matrix_files = [
        path for path in candidates if path.name.lower() in {"matrix.mtx", "matrix.mtx.gz"}
    ]
    tabular_files = [
        path
        for path in candidates
        if path.suffix.lower() in {".csv", ".tsv"}
        and "annotation" not in path.name.lower()
    ]
    if loom_files:
        data = sc.read_loom(loom_files[0])
        source = "local_loom"
    elif h5_files:
        data = sc.read_10x_h5(h5_files[0])
        source = "local_10x_h5"
    elif matrix_files:
        data = sc.read_10x_mtx(matrix_files[0].parent, var_names="gene_symbols")
        source = "local_10x_mtx"
    elif tabular_files:
        separator = "\t" if tabular_files[0].suffix.lower() == ".tsv" else ","
        frame = pd.read_csv(tabular_files[0], sep=separator, index_col=0)
        data = ad.AnnData(
            X=sparse.csr_matrix(frame.to_numpy(dtype=np.float32).T),
            obs=pd.DataFrame(index=frame.columns.astype(str)),
            var=pd.DataFrame(index=frame.index.astype(str)),
        )
        source = "local_tabular_expression"
    else:
        raise SCRNAImportError(
            "No complete readable raw scRNA input is available. "
            "See docs/stage4b_scrna_manual_download_guide.md."
        )

    if paths.annotation.exists():
        annotation = _read_annotation(paths.annotation)
        overlap = data.obs_names.intersection(annotation.index)
        if len(overlap) >= int(0.8 * data.n_obs):
            data.obs = data.obs.join(annotation, how="left")
    data.uns["data_source"] = source
    paths.processed_dir.mkdir(parents=True, exist_ok=True)
    sanitize_anndata_strings(data).write_h5ad(paths.imported_h5ad, compression="gzip")
    return paths.imported_h5ad, source


def create_small_test_h5ad(paths: SCRNAPaths, *, seed: int = 42) -> Path:
    """Create an isolated toy dataset for engineering smoke tests only."""

    import anndata as ad

    rng = np.random.default_rng(seed)
    cell_types = [
        "Malignant epithelial",
        "Epithelial",
        "CD8 T cells",
        "CD4 T cells",
        "NK cells",
        "B cells",
        "Plasma cells",
        "Macrophages/monocytes",
        "Dendritic cells",
        "Fibroblasts/CAF",
        "Endothelial cells",
        "Mast cells",
    ]
    marker_map = {
        "Malignant epithelial": ["EPCAM", "KRT8", "KRT18", "KRT19", "LDHA", "MKI67", "CDK1", "TOP2A"],
        "Epithelial": ["EPCAM", "KRT8", "KRT18", "KRT19"],
        "CD8 T cells": ["CD3D", "CD3E", "TRAC", "CD8A", "CD8B"],
        "CD4 T cells": ["CD3D", "CD3E", "TRAC", "CD4", "IL7R"],
        "NK cells": ["NKG7", "GNLY", "KLRD1"],
        "B cells": ["MS4A1", "CD79A", "CD79B"],
        "Plasma cells": ["MZB1", "JCHAIN", "XBP1"],
        "Macrophages/monocytes": ["LST1", "C1QA", "C1QB", "CD68"],
        "Dendritic cells": ["CD74", "HLA-DRA", "HLA-DRB1", "FCER1A"],
        "Fibroblasts/CAF": ["COL1A1", "COL1A2", "COL3A1", "ACTA2", "FAP", "PDGFRB", "TAGLN", "FN1", "VIM"],
        "Endothelial cells": ["PECAM1", "VWF", "KDR"],
        "Mast cells": ["TPSAB1", "TPSB2", "KIT"],
    }
    program_genes = [
        "CA9", "SLC2A1", "VEGFA", "BNIP3", "EGLN3", "PCNA", "MCM2", "MCM5",
        "CCNB1", "ZEB1", "ZEB2", "SNAI1", "SNAI2", "CDH2", "ITGA5",
    ]
    genes = sorted(
        set(program_genes)
        | {gene for values in marker_map.values() for gene in values}
        | {f"GENE{i:03d}" for i in range(80)}
        | {"MT-CO1", "MT-ND1"}
    )
    labels = np.repeat(cell_types, 20)
    counts = rng.poisson(0.2, size=(len(labels), len(genes))).astype(np.float32)
    gene_index = {gene: index for index, gene in enumerate(genes)}
    for row, label in enumerate(labels):
        for gene in marker_map[label]:
            counts[row, gene_index[gene]] += rng.poisson(5) + 1
        if label == "Malignant epithelial":
            for gene in ["CA9", "SLC2A1", "VEGFA", "BNIP3", "EGLN3", "PCNA", "MCM2", "MCM5", "CCNB1"]:
                counts[row, gene_index[gene]] += rng.poisson(3) + 1
    obs = pd.DataFrame(
        {
            "Sample": np.repeat(["TOY_A", "TOY_B", "TOY_C"], len(labels) // 3),
            "Sample_Origin": "toy",
            "Cell_type.refined": labels,
            "Cell_subtype": labels,
            "analysis_cell_type": labels,
            "source_annotation_status": "toy_engineering_only",
        },
        index=[f"TOY_CELL_{index:04d}" for index in range(len(labels))],
    )
    data = ad.AnnData(
        X=sparse.csr_matrix(counts),
        obs=obs,
        var=pd.DataFrame(index=genes),
    )
    data.uns["dataset_accession"] = "TOY_STAGE4B_SMALL_TEST"
    data.uns["data_source"] = "synthetic_engineering_smoke_test_not_scientific"
    paths.processed_dir.mkdir(parents=True, exist_ok=True)
    sanitize_anndata_strings(data).write_h5ad(paths.imported_h5ad, compression="gzip")
    return paths.imported_h5ad
