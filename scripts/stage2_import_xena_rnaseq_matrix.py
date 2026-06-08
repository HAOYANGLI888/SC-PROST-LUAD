"""Import an explicitly labeled UCSC Xena RNA-seq fallback matrix."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.rnaseq_preprocess import load_rnaseq_matrix


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Import a UCSC Xena RNA-seq fallback matrix into a separate labeled path. "
            "This does not overwrite the GDC primary-analysis matrix."
        ),
    )
    parser.add_argument("--root", default=".", help="Project root. Defaults to current directory.")
    parser.add_argument("--source-matrix", required=True, help="Downloaded UCSC Xena matrix path.")
    parser.add_argument("--input-scale", choices=["tpm", "log2_tpm"], required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root).resolve()
    source = Path(args.source_matrix)
    if not source.is_absolute():
        source = root / source
    if not source.exists():
        print(f"Xena source matrix not found: {source}", file=sys.stderr)
        return 1
    try:
        parsed = load_rnaseq_matrix(source, input_scale=args.input_scale)
    except Exception as exc:
        print(f"Xena fallback validation failed: {exc}", file=sys.stderr)
        return 1
    destination_dir = root / "data" / "raw" / "tcga_luad" / "rnaseq" / "xena"
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"tcga_luad_xena_{args.input_scale}_matrix{source.suffix.lower()}"
    shutil.copyfile(source, destination)
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": "UCSC_Xena",
        "analysis_role": "fallback_only_not_mixed_with_GDC_primary_analysis",
        "input_scale": args.input_scale,
        "parsed_patient_count": len(parsed),
        "parsed_gene_count": len(parsed.columns) - 1,
        "imported_matrix": str(destination),
        "stage2_command": (
            f"python scripts/stage2_prepare_rnaseq_survival.py --config configs/base.yaml "
            f"--rnaseq-matrix \"{destination}\" --input-scale {args.input_scale}"
        ),
    }
    summary_path = root / "data" / "metadata" / "stage2_xena_fallback_import_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
