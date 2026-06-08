"""Build Stage 4 single-cell-guided cell-state signatures."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.scrna_signature import (  # noqa: E402
    SCRNASignatureError,
    build_signatures_from_source,
    save_signature_artifacts,
)


def _paths(root: Path, small_test: bool) -> dict[str, Path]:
    if small_test:
        return {
            "table": root / "outputs" / "tables" / "stage4_small_test" / "cell_state_signature_definitions.csv",
            "json": root / "data" / "metadata" / "stage4_small_test_cell_state_signatures.json",
            "audit": root / "outputs" / "audit" / "stage4_small_test" / "audit_report.md",
        }
    return {
        "table": root / "outputs" / "tables" / "stage4_cell_state_signature_definitions.csv",
        "json": root / "data" / "metadata" / "stage4_cell_state_signatures.json",
        "audit": root / "audit_report.md",
    }


def _write_audit(path: Path, *, mode: str, source_mode: str, table: Path, signature_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        f"\n## Stage 4 Signature Build\n\n"
        f"- Generated: {datetime.now().isoformat(timespec='seconds')}.\n"
        f"- Mode: `{mode}`.\n"
        f"- Signature source: `{source_mode}`.\n"
        f"- Signature count: `{signature_count}`.\n"
        f"- Output table: `{table}`.\n"
        "- Integrity: signatures are not derived from survival outcomes.\n"
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml", help="Project YAML config path.")
    parser.add_argument("--scrna-source", default=None, help="Optional local marker CSV/TSV or h5ad file.")
    parser.add_argument("--cell-type-col", default="cell_type", help="Cell-type column for local scRNA marker input.")
    parser.add_argument("--gene-col", default="gene_symbol", help="Gene-symbol column for local marker input.")
    parser.add_argument("--score-col", default=None, help="Optional marker ranking column for local marker input.")
    parser.add_argument("--top-n", type=int, default=50, help="Top markers per local cell state.")
    parser.add_argument("--small-test", action="store_true", help="Run an isolated small-test signature build.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    root = ROOT
    paths = _paths(root, args.small_test)
    try:
        source = None if args.small_test else args.scrna_source
        signatures, source_mode = build_signatures_from_source(
            source,
            cell_type_col=args.cell_type_col,
            gene_col=args.gene_col,
            score_col=args.score_col,
            top_n=args.top_n,
        )
        frame = save_signature_artifacts(signatures, paths["table"], paths["json"])
        _write_audit(
            paths["audit"],
            mode="toy_small_test" if args.small_test else "formal",
            source_mode=source_mode,
            table=paths["table"],
            signature_count=len(frame),
        )
    except (FileNotFoundError, SCRNASignatureError, ValueError) as exc:
        parser.exit(2, f"Stage 4 signature build failed: {exc}\n")
    print(f"Stage 4 signature build complete: {len(frame)} signatures -> {paths['table']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

