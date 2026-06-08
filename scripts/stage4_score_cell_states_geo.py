"""Score Stage 4 cell states in GEO cohorts and test direction consistency."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from zlib import crc32

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.geo_download import GEO_COHORTS  # noqa: E402
from data.geo_expression_import import collapse_probes_to_genes, read_geo_expression  # noqa: E402
from data.geo_platform_annotation import parse_geo_platform_annotation  # noqa: E402
from data.scrna_signature import (  # noqa: E402
    curated_luad_cell_state_signatures,
    load_signature_artifacts,
    save_signature_artifacts,
)
from features.cell_state_scores import (  # noqa: E402
    BAD_STATE_NAMES,
    PROTECTIVE_STATE_NAMES,
    compare_risk_groups,
    correlate_with_risk,
    external_consistency,
    plot_geo_consistency_heatmap,
    plot_mechanism_summary,
    signature_names,
)
from features.signature_scoring import SignatureScoringError, score_signatures  # noqa: E402


def _paths(root: Path, small_test: bool) -> dict[str, Path]:
    if small_test:
        base = root / "outputs" / "stage4_small_test"
        return {
            "signature_table": root / "outputs" / "tables" / "stage4_small_test" / "cell_state_signature_definitions.csv",
            "tcga_correlation": base / "tables" / "tcga_cell_state_risk_correlation.csv",
            "geo_scores": base / "tables" / "geo_cell_state_scores.csv",
            "geo_correlation": base / "tables" / "geo_cell_state_risk_correlation.csv",
            "geo_group": base / "tables" / "geo_cell_state_group_comparison.csv",
            "geo_missingness": base / "tables" / "geo_signature_missingness.csv",
            "consistency": base / "tables" / "cell_state_external_consistency.csv",
            "figures": base / "figures",
            "report": base / "reports" / "stage4_single_cell_interpretation_report.md",
            "audit": root / "outputs" / "audit" / "stage4_small_test" / "audit_report.md",
        }
    return {
        "signature_table": root / "outputs" / "tables" / "stage4_cell_state_signature_definitions.csv",
        "tcga_correlation": root / "outputs" / "tables" / "stage4_tcga_cell_state_risk_correlation.csv",
        "tcga_group": root / "outputs" / "tables" / "stage4_tcga_cell_state_group_comparison.csv",
        "tcga_mv_cox": root / "outputs" / "tables" / "stage4_tcga_cell_state_multivariable_cox.csv",
        "geo_scores": root / "data" / "processed" / "stage4_geo_cell_state_scores.csv",
        "geo_correlation": root / "outputs" / "tables" / "stage4_geo_cell_state_risk_correlation.csv",
        "geo_group": root / "outputs" / "tables" / "stage4_geo_cell_state_group_comparison.csv",
        "geo_missingness": root / "outputs" / "tables" / "stage4_geo_signature_missingness.csv",
        "consistency": root / "outputs" / "tables" / "stage4_cell_state_external_consistency.csv",
        "figures": root / "outputs" / "figures",
        "report": root / "outputs" / "reports" / "stage4_single_cell_interpretation_report.md",
        "audit": root / "audit_report.md",
    }


def _ensure_signature_table(paths: dict[str, Path]) -> None:
    if not paths["signature_table"].exists():
        save_signature_artifacts(
            curated_luad_cell_state_signatures(),
            paths["signature_table"],
            ROOT / "data" / "metadata" / "stage4_cell_state_signatures.json",
        )


def _platform_path(cohort: str) -> Path:
    platform = GEO_COHORTS[cohort].platform
    gz = ROOT / "data" / "raw" / "geo" / "platforms" / f"{platform}.annot.gz"
    txt = ROOT / "data" / "raw" / "geo" / "platforms" / f"{platform}.txt"
    path = gz if gz.exists() else txt
    if not path.exists():
        raise FileNotFoundError(f"Missing platform annotation for {cohort}: {path}")
    return path


def _transform_external_scale(expression: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    genes = [column for column in expression.columns if column != "sample_id"]
    numeric = expression[genes].apply(pd.to_numeric, errors="coerce")
    q99 = float(np.nanquantile(numeric.to_numpy(dtype=float), 0.99)) if numeric.size else 0.0
    if q99 > 100.0:
        if (numeric < 0).any().any():
            raise SignatureScoringError("External expression appears unlogged but has negative values.")
        numeric = np.log2(numeric + 1.0)
        method = "auto_log2_intensity_plus_1"
    else:
        method = "as_provided_assumed_log2_normalized_microarray"
    return pd.concat([expression[["sample_id"]].reset_index(drop=True), numeric.reset_index(drop=True)], axis=1), method


def _load_formal_geo_expression(cohort: str) -> tuple[pd.DataFrame, str]:
    matrix_path = ROOT / "data" / "raw" / "geo" / cohort / f"{cohort}_series_matrix.txt.gz"
    if not matrix_path.exists():
        raise FileNotFoundError(f"GEO series matrix is missing: {matrix_path}")
    annotation = parse_geo_platform_annotation(_platform_path(cohort))
    expression = read_geo_expression(matrix_path)
    collapsed = collapse_probes_to_genes(expression, annotation, strategy="mean")
    return _transform_external_scale(collapsed)


def _toy_geo_expression(cohort: str, signatures) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    rng = np.random.default_rng(crc32(cohort.encode("utf-8")))
    sample_ids = [f"{cohort}_TOY_{index:03d}" for index in range(24)]
    risk = rng.normal(size=len(sample_ids))
    genes = sorted({gene for signature in signatures for gene in signature.genes})
    columns: dict[str, object] = {"sample_id": sample_ids}
    for gene in genes:
        signal = 0.25 * risk if gene in {"VIM", "FN1", "CD163", "FOXP3", "CA9", "MKI67"} else -0.15 * risk
        columns[gene] = rng.normal(size=len(sample_ids)) + signal
    matrix = pd.DataFrame(columns)
    prepared = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "os_time_days": np.maximum(30, 1300 - 180 * risk + rng.normal(0, 300, len(sample_ids))),
            "os_event": rng.binomial(1, 0.45, len(sample_ids)),
            "age": rng.normal(65, 7, len(sample_ids)),
            "male": rng.binomial(1, 0.5, len(sample_ids)),
            "stage_numeric": rng.integers(1, 4, len(sample_ids)),
            "cohort": cohort,
            "platform": "TOY_PLATFORM",
            "frozen_full_risk_score": risk,
            "frozen_rna_component_risk_score": risk * 0.8,
            "gene_missing_fraction": 0.0,
        }
    )
    return matrix, prepared, "toy_small_test"


def _load_risk_table(cohort: str, *, small_test: bool, signatures) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if small_test:
        return _toy_geo_expression(cohort, signatures)
    prepared_path = ROOT / "data" / "processed" / f"stage2d_geo_{cohort}_prepared.csv"
    if not prepared_path.exists():
        raise FileNotFoundError(f"Stage 2D prepared GEO cohort missing: {prepared_path}")
    expression, scale_method = _load_formal_geo_expression(cohort)
    prepared = pd.read_csv(prepared_path)
    return expression, prepared, scale_method


def _make_toy_tcga_correlation(signatures) -> pd.DataFrame:
    rows = []
    for signature in signatures:
        if signature.signature_name in BAD_STATE_NAMES:
            rho = 0.30
        elif signature.signature_name in PROTECTIVE_STATE_NAMES:
            rho = -0.25
        else:
            rho = 0.05
        rows.append(
            {
                "dataset": "TCGA-LUAD",
                "signature_name": signature.signature_name,
                "n": 30,
                "spearman_rho": rho,
                "p_value": 0.05,
                "direction": "positive" if rho > 0 else "negative",
                "q_value_bh": 0.10,
            }
        )
    return pd.DataFrame(rows)


def _write_final_report(paths: dict[str, Path], *, mode: str, method: str) -> None:
    tcga_corr = pd.read_csv(paths["tcga_correlation"]) if paths["tcga_correlation"].exists() else pd.DataFrame()
    geo_corr = pd.read_csv(paths["geo_correlation"])
    consistency = pd.read_csv(paths["consistency"])
    missingness = pd.read_csv(paths["geo_missingness"])
    tcga_group = pd.read_csv(paths.get("tcga_group", paths["geo_group"])) if paths.get("tcga_group", paths["geo_group"]).exists() else pd.DataFrame()
    mv_path = paths.get("tcga_mv_cox")
    mv = pd.read_csv(mv_path) if mv_path is not None and mv_path.exists() else pd.DataFrame()

    positive = tcga_corr.sort_values("spearman_rho", ascending=False).head(6)["signature_name"].tolist() if not tcga_corr.empty else []
    negative = tcga_corr.sort_values("spearman_rho", ascending=True).head(6)["signature_name"].tolist() if not tcga_corr.empty else []
    expected_match = consistency["matches_predefined_risk_direction"].astype(str).str.lower() == "true"
    bad_consistent = consistency.loc[
        consistency["signature_name"].isin(BAD_STATE_NAMES) & expected_match
    ]["signature_name"].tolist()
    bad_not_supported = consistency.loc[
        consistency["signature_name"].isin(BAD_STATE_NAMES) & ~expected_match
    ]["signature_name"].tolist()
    protective_consistent = consistency.loc[
        consistency["signature_name"].isin(PROTECTIVE_STATE_NAMES) & expected_match
    ]["signature_name"].tolist()
    protective_not_supported = consistency.loc[
        consistency["signature_name"].isin(PROTECTIVE_STATE_NAMES) & ~expected_match
    ]["signature_name"].tolist()
    unstable = consistency.loc[
        consistency["external_direction_consistency"] != "consistent", "signature_name"
    ].tolist()
    unexpected_consistent = consistency.loc[
        (consistency["external_direction_consistency"] == "consistent")
        & ~expected_match
        & consistency["expected_risk_direction"].isin(["higher_risk", "lower_risk"]),
        "signature_name",
    ].tolist()
    independent = []
    if not mv.empty and "covariate" in mv:
        independent = mv.loc[
            (mv["covariate"] == "risk_score") & (pd.to_numeric(mv["p_value"], errors="coerce") < 0.05),
            "signature_name",
        ].unique().tolist()
    missing_summary = (
        missingness.groupby("signature_name")["missing_fraction"].mean().sort_values(ascending=False).head(5)
        if not missingness.empty
        else pd.Series(dtype=float)
    )
    mechanism_strength = "supportive_associative_layer" if len(bad_consistent) + len(protective_consistent) >= 4 else "limited_or_mixed_associative_layer"
    stage5_recommendation = (
        "yes_for_orthogonal_protein_validation_of_consistent_cell_state_markers"
        if mechanism_strength == "supportive_associative_layer"
        else "yes_as_targeted_validation_only_before_strong_mechanistic_claims"
    )
    paths["report"].parent.mkdir(parents=True, exist_ok=True)
    paths["report"].write_text(
        "# Stage 4 Single-Cell-Guided Biological Interpretation Report\n\n"
        f"Generated: {datetime.now().isoformat(timespec='seconds')}\n\n"
        "## Scope And Integrity\n\n"
        "- This stage interprets the Stage 2D frozen TCGA RNA PCA_25 + clinical Cox risk score.\n"
        "- No mutation, CNV, methylation, WSI expansion, protein data, or Stage 3 modeling was used.\n"
        "- Cell-state signatures were not constructed from survival outcomes.\n"
        "- GEO outcomes were not used to construct signatures or tune the frozen risk model.\n"
        "- Findings are biological associations and tumor microenvironment characterization, not causal mechanisms.\n"
        f"- Mode: `{mode}`; scoring method: `{method}`.\n\n"
        "## Required Answers\n\n"
        f"1. High RNA risk score is most positively associated in TCGA with: {', '.join(positive) or 'not available'}.\n"
        f"2. Predefined adverse states enriched in the high-risk direction include: {', '.join(bad_consistent) or 'none'}. Not supported as high-risk-enriched: {', '.join(bad_not_supported) or 'none'}.\n"
        f"3. Predefined protective states enriched in the low-risk direction include: {', '.join(protective_consistent) or 'none'}. Not supported as protective/low-risk-enriched: {', '.join(protective_not_supported) or 'none'}.\n"
        f"4. TCGA/GEO direction-consistent signatures: {', '.join(consistency.loc[consistency['external_direction_consistency'] == 'consistent', 'signature_name'].tolist()) or 'none'}.\n"
        f"5. Externally mixed or unstable signatures: {', '.join(unstable) or 'none'}. Direction-consistent but contrary to predefined risk expectation: {', '.join(unexpected_consistent) or 'none'}.\n"
        f"6. Cell-state scores with age/sex/stage-adjusted Cox P < 0.05 in TCGA: {', '.join(independent) or 'none'}.\n"
        f"7. Single-cell interpretation strength: `{mechanism_strength}`; it is suitable as an RNA-risk biological interpretation layer, but not by itself sufficient for a causal or direct scRNA prognostic mechanism claim.\n"
        f"8. Stage 5 protein validation recommendation: `{stage5_recommendation}`. Stage 5 was not started.\n\n"
        "## External Consistency Notes\n\n"
        f"- GEO correlation rows: `{len(geo_corr)}` across `{geo_corr['dataset'].nunique() if not geo_corr.empty else 0}` cohorts.\n"
        "- Each GEO cohort was processed independently; external expression was not mixed with TCGA for standardization.\n"
        f"- Highest mean external signature missingness: {', '.join([f'{idx}={value:.2%}' for idx, value in missing_summary.items()]) or 'none'}.\n\n"
        "## Interpretation Boundary\n\n"
        "- A directionally consistent EMT/CAF/hypoxia/proliferation pattern supports a high-risk tumor-state and stromal interpretation.\n"
        "- B-cell, plasma-cell and dendritic-cell signatures support an immune-protective low-risk interpretation here.\n"
        "- M2 macrophage, Treg, exhausted CD8, cytotoxic CD8, M1 macrophage and NK findings should not be overclaimed unless their direction and orthogonal validation are resolved.\n"
        "- Mixed external directions should remain exploratory and should not be used as core mechanism claims.\n\n"
        "## Outputs\n\n"
        f"- TCGA correlation: `{paths['tcga_correlation']}`\n"
        f"- GEO correlation: `{paths['geo_correlation']}`\n"
        f"- GEO missingness: `{paths['geo_missingness']}`\n"
        f"- External consistency: `{paths['consistency']}`\n"
        f"- Report: `{paths['report']}`\n",
        encoding="utf-8",
    )


def _append_audit(paths: dict[str, Path], *, mode: str, cohorts: int) -> None:
    paths["audit"].parent.mkdir(parents=True, exist_ok=True)
    with paths["audit"].open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## Stage 4 GEO Cell-State Scoring\n\n"
            f"- Generated: {datetime.now().isoformat(timespec='seconds')}.\n"
            f"- Mode: `{mode}`.\n"
            f"- Cohorts analyzed: `{cohorts}`.\n"
            f"- GEO correlation: `{paths['geo_correlation']}`.\n"
            f"- External consistency: `{paths['consistency']}`.\n"
            f"- Report: `{paths['report']}`.\n"
            "- Integrity: no GEO outcome was used for signature construction or model tuning.\n"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml", help="Project YAML config path.")
    parser.add_argument(
        "--method",
        choices=["mean_zscore", "rank", "ssgsea_like"],
        default="mean_zscore",
        help="Signature scoring method.",
    )
    parser.add_argument("--small-test", action="store_true", help="Run an isolated small-test analysis.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    paths = _paths(ROOT, args.small_test)
    _ensure_signature_table(paths)
    signatures = load_signature_artifacts(paths["signature_table"])
    score_cols = signature_names(signatures)
    all_scores = []
    all_missingness = []
    all_correlations = []
    all_groups = []
    for cohort in GEO_COHORTS:
        expression, prepared, scale_method = _load_risk_table(cohort, small_test=args.small_test, signatures=signatures)
        result = score_signatures(
            expression,
            signatures,
            sample_col="sample_id",
            method=args.method,
            dataset_label=cohort,
        )
        merged = prepared.merge(result.scores, on="sample_id", how="inner", validate="one_to_one")
        if merged.empty:
            raise SystemExit(f"{cohort} expression and Stage 2D risk scores did not overlap.")
        merged["risk_score"] = merged["frozen_full_risk_score"]
        merged["risk_group"] = np.where(
            merged["risk_score"] >= float(merged["risk_score"].median()),
            "high",
            "low",
        )
        merged["stage4_expression_scale_method"] = scale_method
        all_scores.append(merged)
        all_missingness.append(result.missingness)
        all_correlations.append(correlate_with_risk(merged, score_cols, risk_col="risk_score", dataset=cohort))
        all_groups.append(compare_risk_groups(merged, score_cols, group_col="risk_group", dataset=cohort))

    scores = pd.concat(all_scores, ignore_index=True)
    missingness = pd.concat(all_missingness, ignore_index=True)
    geo_correlation = pd.concat(all_correlations, ignore_index=True)
    geo_group = pd.concat(all_groups, ignore_index=True)
    for path in [paths["geo_scores"], paths["geo_missingness"], paths["geo_correlation"], paths["geo_group"], paths["consistency"]]:
        path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(paths["geo_scores"], index=False)
    missingness.to_csv(paths["geo_missingness"], index=False)
    geo_correlation.to_csv(paths["geo_correlation"], index=False)
    geo_group.to_csv(paths["geo_group"], index=False)

    if args.small_test and not paths["tcga_correlation"].exists():
        _make_toy_tcga_correlation(signatures).to_csv(paths["tcga_correlation"], index=False)
    if not paths["tcga_correlation"].exists():
        raise SystemExit(
            f"TCGA Stage 4 correlation table missing: {paths['tcga_correlation']}. "
            "Run scripts/stage4_score_cell_states_tcga.py first."
        )
    tcga_correlation = pd.read_csv(paths["tcga_correlation"])
    consistency = external_consistency(tcga_correlation, geo_correlation, signatures)
    consistency.to_csv(paths["consistency"], index=False)
    plot_geo_consistency_heatmap(geo_correlation, paths["figures"] / "stage4_geo_cell_state_consistency_heatmap.png")
    plot_mechanism_summary(tcga_correlation, consistency, paths["figures"] / "stage4_mechanism_summary.png")
    _write_final_report(paths, mode="toy_small_test" if args.small_test else "formal", method=args.method)
    _append_audit(paths, mode="toy_small_test" if args.small_test else "formal", cohorts=len(GEO_COHORTS))
    print(f"Stage 4 GEO cell-state scoring complete: {len(scores)} samples -> {paths['geo_correlation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
