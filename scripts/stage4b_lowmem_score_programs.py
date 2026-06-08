"""Score fixed Stage 4 programs with chunked mean log1p(CPM)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.scrna_raw_import import scrna_paths  # noqa: E402
from validation.scrna_chunked_scoring import score_fixed_programs  # noqa: E402
from visualization.scrna_lowmem_plots import plot_program_heatmap  # noqa: E402


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", default="configs/base.yaml")
    result.add_argument("--block-size", type=int, default=2048)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    source = scrna_paths(ROOT).raw_or_converted_h5ad
    celltype, sample, patient, missingness = score_fixed_programs(
        source, block_size=args.block_size
    )
    table_dir = ROOT / "outputs" / "tables"
    figure_dir = ROOT / "outputs" / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "celltype": table_dir / "stage4b_lowmem_celltype_program_scores.csv",
        "sample": table_dir / "stage4b_lowmem_sample_program_scores.csv",
        "patient": table_dir / "stage4b_lowmem_patient_program_scores.csv",
        "missing": table_dir / "stage4b_lowmem_signature_missingness.csv",
        "figure": figure_dir / "stage4b_lowmem_program_score_heatmap.png",
    }
    celltype.to_csv(paths["celltype"], index=False)
    sample.to_csv(paths["sample"], index=False)
    patient.to_csv(paths["patient"], index=False)
    missingness.to_csv(paths["missing"], index=False)
    plot_program_heatmap(celltype, paths["figure"])
    print(f"Stage 4B-LowMem program scores complete: {paths['celltype']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
