"""Inventory or import CPTAC/PDC LUAD files for Stage 5B."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from validation.cptac_data_import import (  # noqa: E402
    build_cptac_inventory,
    create_small_test_raw_files,
    download_pdc_luad_study,
    import_local_files,
    query_pdc_luad_proteome_file_candidates,
    stage5b_paths,
    write_inventory_report,
)


def _audit(small_test: bool) -> Path:
    return ROOT / "outputs" / "audit" / "stage5b_small_test" / "audit_report.md" if small_test else ROOT / "audit_report.md"


def _append_audit(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml", help="Project YAML config path.")
    parser.add_argument("--protein-matrix", default=None, help="Optional local protein abundance matrix to import.")
    parser.add_argument("--clinical-metadata", default=None, help="Optional local clinical/survival metadata to import.")
    parser.add_argument("--sample-annotation", default=None, help="Optional local sample annotation to import.")
    parser.add_argument(
        "--pdc-study-ids",
        nargs="*",
        default=None,
        help="Optional PDC study IDs to query for candidate report files.",
    )
    parser.add_argument("--pdc-timeout", type=int, default=30, help="PDC GraphQL timeout in seconds.")
    parser.add_argument("--skip-pdc-query", action="store_true", help="Skip remote PDC candidate-file query.")
    parser.add_argument(
        "--download-pdc-study",
        default=None,
        help="Download a PDC LUAD study through the official GraphQL API, for example PDC000153.",
    )
    parser.add_argument(
        "--pdc-data-type",
        default="log2_ratio",
        help="PDC quantDataMatrix data type used with --download-pdc-study.",
    )
    parser.add_argument("--force-download", action="store_true", help="Overwrite existing PDC API download outputs.")
    parser.add_argument("--small-test", action="store_true", help="Run isolated toy small-test inventory.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    paths = stage5b_paths(ROOT, small_test=args.small_test)
    if args.small_test:
        create_small_test_raw_files(paths)
    else:
        import_local_files(
            paths,
            protein_matrix=args.protein_matrix,
            clinical_metadata=args.clinical_metadata,
            sample_annotation=args.sample_annotation,
        )
        if args.download_pdc_study:
            download_result = download_pdc_luad_study(
                paths,
                pdc_study_id=args.download_pdc_study,
                data_type=args.pdc_data_type,
                timeout=max(args.pdc_timeout, 60),
                force=args.force_download,
            )
            print(
                "PDC study download complete: "
                f"{args.download_pdc_study}; "
                f"samples={download_result.get('sample_count', 0)}; "
                f"genes={download_result.get('gene_count', 0)}"
            )
    inventory = build_cptac_inventory(paths)
    table = paths.tables_dir / "stage5b_cptac_data_inventory.csv"
    remote_table = paths.tables_dir / "stage5b_pdc_luad_remote_file_candidates.csv"
    report = paths.reports_dir / "stage5b_cptac_data_inventory_report.md"
    paths.tables_dir.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(table, index=False)
    remote_candidates = None
    if not args.small_test and not args.skip_pdc_query:
        try:
            remote_candidates = query_pdc_luad_proteome_file_candidates(
                study_ids=args.pdc_study_ids,
                timeout=args.pdc_timeout,
            )
        except Exception as exc:
            remote_candidates = None
            print(f"Warning: PDC remote query failed; local inventory still completed. {exc}")
    if remote_candidates is not None:
        remote_candidates.to_csv(remote_table, index=False)
    write_inventory_report(inventory, report, small_test=args.small_test, remote_candidates=remote_candidates)
    _append_audit(
        _audit(args.small_test),
        "\n## Stage 5B CPTAC Data Inventory\n\n"
        f"- Generated: {datetime.now().isoformat(timespec='seconds')}.\n"
        f"- Mode: `{'toy_small_test' if args.small_test else 'formal'}`.\n"
        f"- Inventory table: `{table}`.\n"
        f"- PDC remote candidate table: `{remote_table if remote_candidates is not None else 'not generated'}`.\n"
        f"- Report: `{report}`.\n"
        "- Integrity: no formal toy CPTAC data were generated.\n",
    )
    print(f"Stage 5B CPTAC inventory complete: {table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
