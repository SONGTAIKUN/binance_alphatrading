#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from crypto_stat_arb.config import ensure_dir, load_config
from run_p2_family_b_mean_reversion import centered_rank_by_timestamp
from run_p6_feature_expansion import (
    LABEL_COLS,
    PRIMARY_LABEL,
    build_candidate_diagnostics,
    build_ic_diagnostics,
    build_p3d_weight_correlations,
    build_p6_features,
    build_signal_turnover,
    plot_family_counts,
    plot_ic_bar,
    plot_turnover_corr,
    resolve_config_path,
    select_retained_features,
    write_feature_panel,
)
from run_residual_alphagpt_search import build_candidate_scores, prepare_alpha_panel


EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "baseline_locked_v1"
DEFAULT_CONFIG = EXPERIMENT_ROOT / "configs" / "baseline_locked_v1.yaml"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "outputs" / "p6b_derivatives_feature_expansion"
SLOW_LABELS = EXPERIMENT_ROOT / "outputs" / "p1_slow_labels" / "slow_residual_labels.csv.gz"
P2_FEATURES = EXPERIMENT_ROOT / "outputs" / "p2_manual_slow_alpha_grid" / "manual_slow_feature_panel.csv.gz"
DERIVATIVES_PANEL = EXPERIMENT_ROOT / "inputs" / "binance_derivatives" / "derivatives_hourly_panel.csv.gz"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--derivatives-panel", default=str(DERIVATIVES_PANEL))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--diagnostic-horizon", type=int, default=72)
    parser.add_argument("--max-abs-p3d-corr", type=float, default=0.35)
    parser.add_argument("--min-primary-ic", type=float, default=0.005)
    parser.add_argument("--min-positive-fold-rate", type=float, default=0.52)
    parser.add_argument("--min-q25-fold-ic", type=float, default=-0.03)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    output_dir = ensure_dir(args.output_dir)
    horizon = int(args.diagnostic_horizon)

    panel_path = resolve_config_path(config_path, cfg["inputs"]["panel"])
    selections_path = resolve_config_path(config_path, cfg["inputs"]["formula_selections"])
    selections = pd.read_csv(selections_path, parse_dates=["train_start", "train_end", "test_start", "test_end"])

    print("[1/8] Loading residual panel, labels, P2 features, and Binance derivatives panel", flush=True)
    panel = pd.read_csv(panel_path, parse_dates=["timestamp"])
    panel = prepare_alpha_panel(panel, [48, 72, 96, 168], execution_delay_bars=1)
    build_candidate_scores(panel)
    labels = pd.read_csv(SLOW_LABELS, parse_dates=["timestamp"], usecols=["timestamp", "symbol", *LABEL_COLS])
    p2_features = pd.read_csv(P2_FEATURES, parse_dates=["timestamp"])
    derivatives = pd.read_csv(args.derivatives_panel, parse_dates=["timestamp"])

    panel = panel.merge(labels, on=["timestamp", "symbol"], how="left")
    panel = panel.merge(p2_features, on=["timestamp", "symbol"], how="left", suffixes=("", "_p2"))
    panel = panel.merge(derivatives, on=["timestamp", "symbol"], how="left")

    print("[2/8] Building C/D/E/F/G OHLCV candidates", flush=True)
    candidates = build_p6_features(panel)

    print("[3/8] Building H/I/J/K Binance derivatives candidates", flush=True)
    candidates.extend(build_derivatives_candidates(panel))
    candidate_meta = pd.DataFrame(candidates)
    candidate_meta.to_csv(output_dir / "candidate_formulas.csv", index=False)
    write_feature_panel_extended(panel, candidates, output_dir / "p6b_feature_panel.csv.gz")
    write_derivatives_coverage(panel, output_dir / "derivatives_coverage.csv")

    print("[4/8] Computing walk-forward OOS IC and path-aware IC", flush=True)
    ic_summary, fold_ic = build_ic_diagnostics(panel, selections, candidates, horizon)
    ic_summary.to_csv(output_dir / "ic_summary.csv", index=False)
    fold_ic.to_csv(output_dir / "fold_ic.csv", index=False)

    print("[5/8] Computing signal turnover and P3d correlation diagnostics", flush=True)
    turnover = build_signal_turnover(panel, selections, candidates, horizon)
    p3d_corr = build_p3d_weight_correlations(panel, candidates)
    diagnostics = build_candidate_diagnostics(candidate_meta, ic_summary, turnover, p3d_corr)
    diagnostics.to_csv(output_dir / "candidate_diagnostics.csv", index=False)

    print("[6/8] Applying retention rules", flush=True)
    retained, watchlist = select_retained_features(
        diagnostics=diagnostics,
        max_abs_p3d_corr=float(args.max_abs_p3d_corr),
        min_primary_ic=float(args.min_primary_ic),
        min_positive_fold_rate=float(args.min_positive_fold_rate),
        min_q25_fold_ic=float(args.min_q25_fold_ic),
    )
    retained.to_csv(output_dir / "retained_features.csv", index=False)
    watchlist.to_csv(output_dir / "watchlist_features.csv", index=False)
    diagnostics[diagnostics["family"].str.startswith(("H_", "I_", "J_", "K_"))].to_csv(
        output_dir / "derivatives_candidate_diagnostics.csv", index=False
    )

    print("[7/8] Writing plots", flush=True)
    plot_ic_bar(output_dir, diagnostics)
    plot_turnover_corr(output_dir, diagnostics, retained)
    plot_family_counts(output_dir, diagnostics, retained)
    plot_derivative_family_ic(output_dir, diagnostics)

    print("[8/8] Writing report and config stage", flush=True)
    write_report(output_dir, diagnostics, retained, watchlist, args)
    update_config_stage(config_path)
    print(f"Done: {output_dir}", flush=True)


def build_derivatives_candidates(panel: pd.DataFrame) -> list[dict[str, Any]]:
    panel.sort_values(["symbol", "timestamp"], inplace=True)

    raw_cols = [
        "deriv_funding_rate_ewm_72",
        "deriv_funding_rate_ewm_168",
        "deriv_funding_accel_24_168",
        "deriv_funding_annualized_ewm_168",
        "deriv_abs_funding_stress_168",
        "deriv_premium_index_ewm_72",
        "deriv_premium_index_ewm_168",
        "deriv_premium_accel_24_168",
        "deriv_mark_index_basis_ewm_72",
        "deriv_mark_index_basis_ewm_168",
        "deriv_abs_basis_stress_168",
        "deriv_log_oi_value_ewm_72",
        "deriv_log_oi_value_ewm_336",
        "deriv_oi_growth_72_336",
        "deriv_account_long_short_ratio_ewm_168",
        "deriv_top_count_long_short_ratio_ewm_168",
        "deriv_top_sum_long_short_ratio_ewm_168",
        "deriv_taker_long_short_ratio_ewm_24",
        "deriv_taker_pressure_24",
        "deriv_crowding_pressure_72",
    ]
    for col in raw_cols:
        if col in panel:
            panel[f"cs_{col}"] = centered_rank_by_timestamp(panel, col)

    panel["p6b_H01_funding_contra_168"] = -panel["cs_deriv_funding_rate_ewm_168"]
    panel["p6b_H02_funding_accel_fade"] = -panel["cs_deriv_funding_accel_24_168"]
    panel["p6b_H03_slow_mom_funding_penalty"] = (
        panel["cs_res_mom_336_24"] - 0.45 * panel["cs_deriv_funding_rate_ewm_168"]
    )
    panel["p6b_H04_reversion_funding_stress_guard"] = (
        panel["alpha_B_soft_combo"] - 0.35 * panel["cs_deriv_abs_funding_stress_168"]
    )
    panel["p6b_H05_carry_unwind_contra"] = -(
        0.60 * panel["cs_deriv_funding_rate_ewm_168"] + 0.40 * panel["cs_deriv_funding_annualized_ewm_168"]
    )

    panel["p6b_I01_premium_contra_168"] = -panel["cs_deriv_premium_index_ewm_168"]
    panel["p6b_I02_basis_contra_168"] = -panel["cs_deriv_mark_index_basis_ewm_168"]
    panel["p6b_I03_premium_accel_fade"] = -panel["cs_deriv_premium_accel_24_168"]
    panel["p6b_I04_basis_stress_reversion_guard"] = (
        panel["alpha_B_soft_combo"] - 0.35 * panel["cs_deriv_abs_basis_stress_168"]
    )
    panel["p6b_I05_mom_basis_quality"] = (
        panel["cs_res_mom_336_24"]
        - 0.25 * panel["cs_deriv_abs_basis_stress_168"]
        - 0.25 * panel["cs_deriv_premium_index_ewm_168"]
    )

    panel["p6b_J01_oi_growth_mom_quality"] = (
        panel["cs_res_mom_336_24"] + 0.35 * panel["cs_deriv_oi_growth_72_336"]
    )
    panel["p6b_J02_oi_growth_crowding_fade"] = -(
        0.55 * panel["cs_deriv_oi_growth_72_336"] + 0.45 * panel["cs_deriv_crowding_pressure_72"]
    )
    panel["p6b_J03_account_longs_contra"] = -panel["cs_deriv_account_long_short_ratio_ewm_168"]
    panel["p6b_J04_top_trader_contra"] = -(
        0.50 * panel["cs_deriv_top_count_long_short_ratio_ewm_168"]
        + 0.50 * panel["cs_deriv_top_sum_long_short_ratio_ewm_168"]
    )
    panel["p6b_J05_taker_flow_contra"] = -panel["cs_deriv_taker_pressure_24"]

    panel["p6b_K01_deriv_crowding_contra"] = -(
        0.35 * panel["cs_deriv_funding_rate_ewm_168"]
        + 0.25 * panel["cs_deriv_premium_index_ewm_168"]
        + 0.20 * panel["cs_deriv_top_sum_long_short_ratio_ewm_168"]
        + 0.20 * panel["cs_deriv_taker_pressure_24"]
    )
    panel["p6b_K02_uncrowded_slow_mom"] = (
        panel["cs_res_mom_336_24"] + 0.35 * panel["p6b_K01_deriv_crowding_contra"]
    )
    panel["p6b_K03_deriv_low_stress_mom"] = (
        panel["cs_res_mom_336_24"]
        - 0.25 * panel["cs_deriv_abs_funding_stress_168"]
        - 0.25 * panel["cs_deriv_abs_basis_stress_168"]
    )
    panel["p6b_K04_deriv_reversion_quality"] = (
        0.60 * panel["alpha_B_soft_combo"]
        + 0.25 * panel["p6b_K01_deriv_crowding_contra"]
        - 0.15 * panel["cs_deriv_abs_basis_stress_168"]
    )
    panel["p6b_K05_deriv_p3d_diversifier"] = (
        0.40 * panel["p6_E03_basket_diversifier_mom"]
        + 0.35 * panel["p6b_K01_deriv_crowding_contra"]
        + 0.25 * panel["p6_D04_low_vov_momentum"]
    )

    specs = [
        ("H01_funding_contra_168", "H_funding_carry", "p6b_H01_funding_contra_168", "-funding_rate_ewm_168"),
        ("H02_funding_accel_fade", "H_funding_carry", "p6b_H02_funding_accel_fade", "-(funding_ewm_24 - funding_ewm_168)"),
        ("H03_slow_mom_funding_penalty", "H_funding_carry", "p6b_H03_slow_mom_funding_penalty", "res_mom_336_24 - funding_rate_ewm_168"),
        ("H04_reversion_funding_stress_guard", "H_funding_carry", "p6b_H04_reversion_funding_stress_guard", "B_soft_combo - abs(funding_z_168)"),
        ("H05_carry_unwind_contra", "H_funding_carry", "p6b_H05_carry_unwind_contra", "-funding_carry_168"),
        ("I01_premium_contra_168", "I_basis_premium", "p6b_I01_premium_contra_168", "-premium_index_ewm_168"),
        ("I02_basis_contra_168", "I_basis_premium", "p6b_I02_basis_contra_168", "-mark_index_basis_ewm_168"),
        ("I03_premium_accel_fade", "I_basis_premium", "p6b_I03_premium_accel_fade", "-(premium_ewm_24 - premium_ewm_168)"),
        ("I04_basis_stress_reversion_guard", "I_basis_premium", "p6b_I04_basis_stress_reversion_guard", "B_soft_combo - abs(basis_z_168)"),
        ("I05_mom_basis_quality", "I_basis_premium", "p6b_I05_mom_basis_quality", "res_mom_336_24 - basis_stress - premium"),
        ("J01_oi_growth_mom_quality", "J_oi_crowding", "p6b_J01_oi_growth_mom_quality", "res_mom_336_24 + oi_growth"),
        ("J02_oi_growth_crowding_fade", "J_oi_crowding", "p6b_J02_oi_growth_crowding_fade", "-oi_growth - crowding_pressure"),
        ("J03_account_longs_contra", "J_oi_crowding", "p6b_J03_account_longs_contra", "-account_long_short_ratio_168"),
        ("J04_top_trader_contra", "J_oi_crowding", "p6b_J04_top_trader_contra", "-top_trader_long_short_ratio_168"),
        ("J05_taker_flow_contra", "J_oi_crowding", "p6b_J05_taker_flow_contra", "-taker_long_short_ratio_24"),
        ("K01_deriv_crowding_contra", "K_derivatives_combo", "p6b_K01_deriv_crowding_contra", "-funding/premium/top/taker crowding"),
        ("K02_uncrowded_slow_mom", "K_derivatives_combo", "p6b_K02_uncrowded_slow_mom", "res_mom_336_24 + deriv_crowding_contra"),
        ("K03_deriv_low_stress_mom", "K_derivatives_combo", "p6b_K03_deriv_low_stress_mom", "res_mom_336_24 - funding/basis stress"),
        ("K04_deriv_reversion_quality", "K_derivatives_combo", "p6b_K04_deriv_reversion_quality", "B_soft + deriv crowding contra - basis stress"),
        ("K05_deriv_p3d_diversifier", "K_derivatives_combo", "p6b_K05_deriv_p3d_diversifier", "E03 + deriv crowding contra + D04"),
    ]
    return [
        {"name": name, "family": family, "score_col": score_col, "formula": formula, "features": score_col}
        for name, family, score_col, formula in specs
    ]


def write_feature_panel_extended(panel: pd.DataFrame, candidates: list[dict[str, Any]], output_path: Path) -> None:
    write_feature_panel(panel, candidates, output_path)


def write_derivatives_coverage(panel: pd.DataFrame, output_path: Path) -> None:
    cols = [col for col in panel.columns if col.startswith("deriv_")]
    warmup_start = pd.Timestamp("2024-07-01", tz="UTC")
    rows = []
    for col in cols:
        rows.append(
            {
                "column": col,
                "coverage": float(panel[col].notna().mean()),
                "coverage_oos_like_after_warmup": float(panel.loc[panel["timestamp"] >= warmup_start, col].notna().mean()),
            }
        )
    pd.DataFrame(rows).sort_values("coverage", ascending=False).to_csv(output_path, index=False)


def plot_derivative_family_ic(output_dir: Path, diagnostics: pd.DataFrame) -> None:
    deriv = diagnostics[diagnostics["family"].str.startswith(("H_", "I_", "J_", "K_"))].copy()
    if deriv.empty:
        return
    top = deriv.sort_values("primary_mean_ic").tail(20)
    plt.figure(figsize=(12, 8))
    colors = np.where(top["primary_mean_ic"] >= 0.0, "#386641", "#bc4749")
    plt.barh(top["name"], top["primary_mean_ic"], color=colors)
    plt.axvline(0.0, color="black", linewidth=0.8, alpha=0.6)
    plt.title("P6b derivatives candidate IC vs path slow low-turnover label")
    plt.xlabel("Mean OOS cross-sectional Spearman IC")
    plt.grid(True, axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "derivative_top_primary_ic.png", dpi=180)
    plt.close()


def write_report(
    output_dir: Path,
    diagnostics: pd.DataFrame,
    retained: pd.DataFrame,
    watchlist: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    top = diagnostics.sort_values("diagnostic_score", ascending=False).head(15)
    deriv_top = diagnostics[diagnostics["family"].str.startswith(("H_", "I_", "J_", "K_"))].sort_values(
        "diagnostic_score", ascending=False
    ).head(15)
    retained_names = set(retained["name"].astype(str).tolist()) if not retained.empty else set()
    summary_frame = diagnostics.copy()
    summary_frame["retained_flag"] = summary_frame["name"].astype(str).isin(retained_names)
    family_summary = (
        summary_frame.groupby("family")
        .agg(
            candidates=("name", "count"),
            retained=("retained_flag", "sum"),
            mean_primary_ic=("primary_mean_ic", "mean"),
            best_primary_ic=("primary_mean_ic", "max"),
            mean_turnover=("signal_turnover_72h", "mean"),
            mean_abs_p3d_corr=("mean_abs_p3d_weight_corr", "mean"),
        )
        .reset_index()
    )

    report = [
        "# P6b Binance Derivatives Feature Expansion",
        "",
        "This run extends P6 beyond OHLCV-only features with Binance USD-M futures public data.",
        "",
        "Ex-ante rule: funding, mark/index/premium, OI, long-short ratios, and taker-flow fields are shifted by 1 hour before feature construction.",
        "",
        "## Inputs",
        "",
        f"- Derivatives panel: `{args.derivatives_panel}`",
        f"- Diagnostic rebalance horizon: `{args.diagnostic_horizon}h`",
        f"- Primary label: `{PRIMARY_LABEL}`",
        "",
        "## Retained Features",
        "",
    ]
    if retained.empty:
        report.append("No features passed all retention rules.")
    else:
        report.append(df_to_markdown(retained[["name", "family", "formula", "primary_mean_ic", "primary_q25_fold_ic", "primary_positive_fold_ic_rate", "signal_turnover_72h", "mean_abs_p3d_weight_corr"]]))

    report.extend(["", "## Top Diagnostics", "", df_to_markdown(top[["name", "family", "primary_mean_ic", "primary_q25_fold_ic", "primary_positive_fold_ic_rate", "signal_turnover_72h", "mean_abs_p3d_weight_corr", "diagnostic_score"]])])
    report.extend(["", "## Top Derivatives Candidates", ""])
    if deriv_top.empty:
        report.append("No derivatives candidates available.")
    else:
        report.append(df_to_markdown(deriv_top[["name", "family", "primary_mean_ic", "primary_q25_fold_ic", "primary_positive_fold_ic_rate", "signal_turnover_72h", "mean_abs_p3d_weight_corr", "diagnostic_score"]]))

    report.extend(["", "## Family Summary", "", df_to_markdown(family_summary), "", "## Files", ""])
    for name in [
        "candidate_formulas.csv",
        "p6b_feature_panel.csv.gz",
        "derivatives_coverage.csv",
        "ic_summary.csv",
        "fold_ic.csv",
        "candidate_diagnostics.csv",
        "derivatives_candidate_diagnostics.csv",
        "retained_features.csv",
        "watchlist_features.csv",
        "top_primary_ic.png",
        "derivative_top_primary_ic.png",
        "turnover_vs_p3d_corr.png",
        "family_retention_counts.png",
    ]:
        report.append(f"- `{name}`")
    output_dir.joinpath("REPORT.md").write_text("\n".join(report), encoding="utf-8")


def df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    frame = df.copy()
    for col in frame.columns:
        if pd.api.types.is_float_dtype(frame[col]):
            frame[col] = frame[col].map(lambda x: "" if pd.isna(x) else f"{x:.6g}")
        else:
            frame[col] = frame[col].map(lambda x: "" if pd.isna(x) else str(x))
    headers = [str(col) for col in frame.columns]
    rows = frame.astype(str).values.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |")
    return "\n".join(lines)


def update_config_stage(config_path: Path) -> None:
    import yaml

    cfg = load_config(config_path)
    stages = cfg.setdefault("stages", {})
    stages["p6b_derivatives_feature_expansion"] = {
        "script": "../../../scripts/run_p6b_derivatives_feature_expansion.py",
        "output_dir": "../outputs/p6b_derivatives_feature_expansion",
        "report": "../outputs/p6b_derivatives_feature_expansion/REPORT.md",
        "feature_panel": "../outputs/p6b_derivatives_feature_expansion/p6b_feature_panel.csv.gz",
        "candidate_formulas": "../outputs/p6b_derivatives_feature_expansion/candidate_formulas.csv",
        "ic_summary": "../outputs/p6b_derivatives_feature_expansion/ic_summary.csv",
        "candidate_diagnostics": "../outputs/p6b_derivatives_feature_expansion/candidate_diagnostics.csv",
        "retained_features": "../outputs/p6b_derivatives_feature_expansion/retained_features.csv",
        "derivatives_coverage": "../outputs/p6b_derivatives_feature_expansion/derivatives_coverage.csv",
    }
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False), encoding="utf-8")


if __name__ == "__main__":
    main()
