"""Inventory Stage 2C RNA feature spaces before nested CV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.geo_expression_import import export_tcga_gene_annotation_from_star_counts
from features.rnaseq_feature_spaces import feature_space_inventory
from training.nested_cv import model_configurations


def compare_feature_spaces(root: str | Path = ".", *, small_test: bool = False) -> dict[str, object]:
    project_root = Path(root).resolve()
    try:
        export_tcga_gene_annotation_from_star_counts(project_root)
    except FileNotFoundError:
        if not small_test:
            raise
    inventory = feature_space_inventory(project_root)
    inventory["status"] = inventory.apply(
        lambda row: "available_for_nested_cv" if bool(row["available"]) else f"unavailable: {row['reason']}",
        axis=1,
    )
    inventory["outer_c_index_mean"] = ""
    inventory["outer_c_index_std"] = ""
    output = project_root / "outputs" / "tables" / "stage2c_feature_space_comparison.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(output, index=False)
    return {
        "status": "passed",
        "mode": "toy_small_test" if small_test else "real_inventory",
        "feature_space_count": len(inventory),
        "available_feature_space_count": int(inventory["available"].sum()),
        "planned_model_configuration_count": len(model_configurations(project_root)),
        "output": str(output),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inventory Stage 2C RNA feature spaces and optional pathway resources.")
    parser.add_argument("--config", default="configs/base.yaml", help="Project YAML config path.")
    parser.add_argument("--root", default=".", help="Project root. Defaults to current directory.")
    parser.add_argument("--small-test", action="store_true", help="Allow an offline inventory smoke test.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not (Path(args.root) / args.config).exists():
        parser.error(f"Config file not found: {Path(args.root) / args.config}")
    try:
        result = compare_feature_spaces(args.root, small_test=args.small_test)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"Stage 2C feature-space inventory failed: {exc}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
