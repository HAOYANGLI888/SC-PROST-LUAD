"""Score Stage 4 cell states in TCGA-LUAD and relate them to Stage 2D risk."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

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
    load_tcga_stage2d_context,
    multivariable_cell_state_cox,
    plot_cox_forest,
    plot_correlation_bar,
    plot_group_boxplots,
    plot_tcga_heatmap,
    risk_score_survival_summary,
    signature_names,
    univariable_cell_state_cox,
)
from features.signature_scoring import load_tcga_symbol_expression, score_signatures  # noqa: E402


def _paths(root: Path, small_test: bool) -> dict[str, Path]:
    if small_test:
        base = root / "outputs" / "stage4_small_test"
        return {
            "signature_table": root / "outputs" / "tables" / "stage4_small_test" / "cell_state_signature_definitions.csv",
            "scores": base / "tables" / "tcga_cell_state_scores.csv",
            "missingness": base / "tables" / "tcga_signature_missingness.csv",
            "correlation": base / "tables" / "tcga_cell_state_risk_correlation.csv",
            "group": base / "tables" / "tcga_cell_state_group_comparison.csv",
            "cox": base / "tables" / "tcga_cell_state_cox.csv",
            "mv_cox": base / "tables" / "tcga_cell_state_multivariable_cox.csv",
            "risk_summary": base / "tables" / "tcga_risk_score_summary.csv",
            "figures": base / "figures",
            "report": base / "reports" / "stage4_single_cell_interpretation_report.md",
            "audit": root / "outputs" / "audit" / "stage4_small_test" / "audit_report.md",
        }
    return {
        "signature_table": root / "outputs" / "tables" / "stage4_cell_state_signature_definitions.csv",
        "scores": root / "outputs" / "tables" / "stage4_tcga_cell_state_scores.csv",
        "missingness": root / "data" / "metadata" / "stage4_tcga_signature_missingness.csv",
        "correlation": root / "outputs" / "tables" / "stage4_tcga_cell_state_risk_correlation.csv",
        "group": root / "outputs" / "tables" / "stage4_tcga_cell_state_group_comparison.csv",
        "cox": root / "outputs" / "tables" / "stage4_tcga_cell_state_cox.csv",
        "mv_cox": root / "outputs" / "tables" / "stage4_tcga_cell_state_multivariable_cox.csv",
        "risk_summary": root / "outputs" / "tables" / "stage4_tcga_risk_score_summary.csv",
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


def _write_report(
    paths: dict[str, Path],
    *,
    mode: str,
    method: str,
    expression_summary: dict[str, object],
    analyzed_patients: int,
) -> None:
    corr = pd.read_csv(paths["correlation"])
    group = pd.read_csv(paths["group"])
    cox = pd.read_csv(paths["cox"])
    mv = pd.read_csv(paths["mv_cox"])
    risk = pd.read_csv(paths["risk_summary"]).iloc[0].to_dict()
    bad = corr.loc[corr["signature_name"].isin(BAD_STATE_NAMES)].sort_values("spearman_rho", ascending=False)
    protective = corr.loc[corr["signature_name"].isin(PROTECTIVE_STATE_NAMES)].sort_values("spearman_rho")
    independent = mv.loc[(mv["covariate"] == "risk_score") & (mv["p_value"] < 0.05)]
    top_positive = corr.sort_values("spearman_rho", ascending=False).head(5)
    top_negative = corr.sort_values("spearman_rho", ascending=True).head(5)
    paths["report"].parent.mkdir(parents=True, exist_ok=True)
    paths["report"].write_text(
        "# Stage 4 Single-Cell-Guided Interpretation Report\n\n"
        f"Generated: {datetime.now().isoformat(timespec='seconds')}\n\n"
        "## Scope\n\n"
        "- Stage: 4, biological interpretation of the Stage 2D frozen RNA PCA_25 + clinical risk score.\n"
        "- No Stage 3 omics, WSI expansion, mutation, CNV, methylation, protein data, or survival training with scRNA was performed.\n"
        f"- Mode: `{mode}`; signature scoring method: `{method}`.\n"
        f"- TCGA expression scale: `{expression_summary.get('expression_scale_method')}`.\n"
        f"- TCGA patients analyzed: `{analyzed_patients}`.\n\n"
        "## TCGA Risk Score Context\n\n"
        f"- OS events: `{int(risk['events'])}`; censored: `{int(risk['censored'])}`.\n"
        f"- Risk-score KM log-rank P by median split: `{risk['km_logrank_p']:.4g}`.\n"
        f"- Risk-score AUC at 1/3/5 years: `{risk['auc_1_year']:.3f}`, `{risk['auc_3_year']:.3f}`, `{risk['auc_5_year']:.3f}`.\n\n"
        "## Main TCGA Associations\n\n"
        f"- Top positive risk-associated states: {', '.join(top_positive['signature_name'].tolist())}.\n"
        f"- Top negative risk-associated states: {', '.join(top_negative['signature_name'].tolist())}.\n"
        f"- Higher-risk adverse-state candidates with positive correlation: {', '.join(bad.loc[bad['spearman_rho'] > 0, 'signature_name'].tolist()) or 'none'}.\n"
        f"- Protective-state candidates with negative correlation: {', '.join(protective.loc[protective['spearman_rho'] < 0, 'signature_name'].tolist()) or 'none'}.\n"
        f"- Cell-state scores with multivariable Cox P < 0.05 after age/sex/stage adjustment: {', '.join(independent['signature_name'].unique().tolist()) or 'none'}.\n\n"
        "## Required Answers So Far\n\n"
        "1. High RNA risk score is associated with the states listed in the TCGA correlation table; this report should be finalized after GEO consistency is computed.\n"
        "2. EMT-like tumor, M2 macrophage, Treg, exhausted CD8, CAF, hypoxia and proliferation enrichment should be judged from `stage4_tcga_cell_state_group_comparison.csv` and the GEO consistency table.\n"
        "3. Cytotoxic CD8/NK protective direction is considered supportive only if negative in TCGA and directionally consistent in GEO.\n"
        "4. GEO consistency is pending until `scripts/stage4_score_cell_states_geo.py` is run.\n"
        "5. TCGA-only findings are not causal and are not direct single-cell survival validation.\n"
        "6. Cell-state independent prognostic value is summarized in `stage4_tcga_cell_state_multivariable_cox.csv`.\n"
        "7. Single-cell interpretation is not yet sufficient as a full mechanism core until external direction consistency is reviewed.\n"
        "8. Stage 5 protein validation is not started here; recommendation will be finalized by the GEO script.\n\n"
        "## Outputs\n\n"
        f"- Scores: `{paths['scores']}`\n"
        f"- Correlation: `{paths['correlation']}`\n"
        f"- Group comparison: `{paths['group']}`\n"
        f"- Cox: `{paths['cox']}`\n"
        f"- Multivariable Cox: `{paths['mv_cox']}`\n",
        encoding="utf-8",
    )


def _append_audit(paths: dict[str, Path], *, mode: str, patients: int) -> None:
    paths["audit"].parent.mkdir(parents=True, exist_ok=True)
    with paths["audit"].open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## Stage 4 TCGA Cell-State Scoring\n\n"
            f"- Generated: {datetime.now().isoformat(timespec='seconds')}.\n"
            f"- Mode: `{mode}`.\n"
            f"- Patients analyzed: `{patients}`.\n"
            f"- Scores: `{paths['scores']}`.\n"
            f"- Report: `{paths['report']}`.\n"
            "- Integrity: cell-state scores were not trained with survival outcomes.\n"
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
    expression, expression_summary = load_tcga_symbol_expression(ROOT, small_test=args.small_test)
    result = score_signatures(
        expression,
        signatures,
        sample_col="patient_id",
        method=args.method,
        dataset_label="TCGA-LUAD",
    )
    context = load_tcga_stage2d_context(ROOT, small_test=False)
    merged = context.merge(result.scores, on="patient_id", how="inner", validate="one_to_one")
    if args.small_test:
        merged = merged.sort_values("patient_id").head(30).copy()
        merged["risk_group"] = ["high" if value >= merged["risk_score"].median() else "low" for value in merged["risk_score"]]
    if len(merged) < 10:
        raise SystemExit("Stage 4 TCGA analysis has fewer than 10 overlapping patients.")
    score_cols = signature_names(signatures)
    for path in [paths["scores"], paths["missingness"], paths["correlation"], paths["group"], paths["cox"], paths["mv_cox"], paths["risk_summary"]]:
        path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(paths["scores"], index=False)
    result.missingness.to_csv(paths["missingness"], index=False)
    correlation = correlate_with_risk(merged, score_cols, dataset="TCGA-LUAD")
    group = compare_risk_groups(merged, score_cols, dataset="TCGA-LUAD")
    cox = univariable_cell_state_cox(merged, score_cols, dataset="TCGA-LUAD")
    mv_cox = multivariable_cell_state_cox(merged, score_cols, dataset="TCGA-LUAD")
    correlation.to_csv(paths["correlation"], index=False)
    group.to_csv(paths["group"], index=False)
    cox.to_csv(paths["cox"], index=False)
    mv_cox.to_csv(paths["mv_cox"], index=False)
    risk_summary = risk_score_survival_summary(merged)
    pd.DataFrame([risk_summary]).to_csv(paths["risk_summary"], index=False)
    figs = paths["figures"]
    plot_tcga_heatmap(merged, score_cols, figs / "stage4_tcga_cell_state_heatmap.png")
    plot_group_boxplots(merged, score_cols, figs / "stage4_tcga_cell_state_boxplots.png")
    plot_correlation_bar(correlation, figs / "stage4_tcga_risk_cell_state_correlation.png")
    plot_cox_forest(cox, figs / "stage4_tcga_cell_state_cox_forest.png")
    _write_report(
        paths,
        mode="toy_small_test" if args.small_test else "formal",
        method=args.method,
        expression_summary=expression_summary,
        analyzed_patients=len(merged),
    )
    _append_audit(paths, mode="toy_small_test" if args.small_test else "formal", patients=len(merged))
    print(f"Stage 4 TCGA cell-state scoring complete: {len(merged)} patients -> {paths['scores']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

