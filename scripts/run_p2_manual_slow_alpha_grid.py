#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from crypto_stat_arb.config import ensure_dir, load_config
from run_p0_baseline_lock_review import (
    build_asset_summary,
    build_cost_sensitivity,
    build_regime_frame,
    build_regime_summary,
    build_side_summary,
)
from run_p2_family_a_momentum import build_family_a_features
from run_p2_family_b_mean_reversion import (
    build_baseline_comparison,
    build_family_b_features,
    build_ic_summary,
    centered_rank_by_timestamp,
    load_baseline_summary,
    make_locked_risk_config,
    make_portfolio_config,
    run_candidate_replay,
    write_run_result,
)
from run_residual_alphagpt_search import build_candidate_scores, prepare_alpha_panel
from run_turnover_experiments import to_jsonable


EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "baseline_locked_v1"
DEFAULT_CONFIG = EXPERIMENT_ROOT / "configs" / "baseline_locked_v1.yaml"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "outputs" / "p2_manual_slow_alpha_grid"
SLOW_LABELS = EXPERIMENT_ROOT / "outputs" / "p1_slow_labels" / "slow_residual_labels.csv.gz"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--horizons", type=int, nargs="+", default=[48, 72, 96, 168])
    parser.add_argument("--cost-bps", type=float, nargs="+", default=[0, 4, 6, 8, 10, 12, 15, 20, 30])
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    output_dir = ensure_dir(args.output_dir)
    horizons = sorted(set(int(h) for h in args.horizons))
    cost_bps = [float(x) for x in args.cost_bps]

    panel_path = resolve_config_path(config_path, cfg["inputs"]["panel"])
    selections_path = resolve_config_path(config_path, cfg["inputs"]["formula_selections"])
    if not SLOW_LABELS.exists():
        raise FileNotFoundError(f"P1 slow label file not found: {SLOW_LABELS}")

    print("[1/8] Loading residual panel, P1 labels, and frozen walk-forward folds", flush=True)
    panel = pd.read_csv(panel_path, parse_dates=["timestamp"])
    panel = prepare_alpha_panel(panel, sorted(set(horizons + [48, 72, 96, 168])), execution_delay_bars=1)
    build_candidate_scores(panel)
    labels = pd.read_csv(
        SLOW_LABELS,
        parse_dates=["timestamp"],
        usecols=[
            "timestamp",
            "symbol",
            "label_slow_medium",
            "label_slow_low_turnover",
            "label_path_slow_medium",
            "label_path_slow_low_turnover",
            "label_horizon_sign_consistency",
            "label_path_horizon_sign_consistency",
        ],
    )
    panel = panel.merge(labels, on=["timestamp", "symbol"], how="left")
    selections = pd.read_csv(selections_path, parse_dates=["train_start", "train_end", "test_start", "test_end"])

    print("[2/8] Building unified OHLCV-only slow residual feature panel", flush=True)
    candidates: list[dict[str, Any]] = []

    family_a = build_family_a_features(panel)
    for item in family_a:
        candidates.append({**item, "family": "A_residual_momentum", "use_regime_gate": False})

    family_b = build_family_b_features(panel)
    for item in family_b:
        candidates.append({**item, "family": "B_stationary_mean_reversion", "use_regime_gate": True})

    family_c = build_manual_slow_hybrid_features(panel)
    candidates.extend(family_c)

    candidate_meta = pd.DataFrame(candidates)
    candidate_meta.to_csv(output_dir / "candidate_formulas.csv", index=False)
    write_feature_panel(panel, candidates, output_dir / "manual_slow_feature_panel.csv.gz")

    print("[3/8] Running unified strict-neutral walk-forward grid", flush=True)
    risk_cfg = make_locked_risk_config(cfg)
    summary_rows: list[dict[str, Any]] = []
    result_index: dict[str, dict[str, Any]] = {}
    runs_dir = ensure_dir(output_dir / "runs")
    total_runs = len(horizons) * len(candidates)
    run_count = 0
    for horizon in horizons:
        pf_cfg = make_portfolio_config(cfg, horizon)
        for candidate in candidates:
            run_count += 1
            key = f"h{horizon:03d}_{candidate['name']}"
            print(f"      [{run_count}/{total_runs}] {key}", flush=True)
            result = run_candidate_replay(
                panel=panel,
                selections=selections,
                pf_cfg=pf_cfg,
                wf_cfg=None,
                risk_cfg=risk_cfg,
                score_col=str(candidate["score_col"]),
                formula=str(candidate["formula"]),
                use_regime_gate=bool(candidate["use_regime_gate"]),
            )
            result_index[key] = result
            write_run_result(ensure_dir(runs_dir / key), result)
            summary_rows.append({"run": key, "horizon": horizon, **candidate, **result["metrics"]})

    summary = pd.DataFrame(summary_rows)
    corr_summary = build_locked_baseline_correlations(result_index, summary)
    summary = summary.merge(corr_summary, on="run", how="left")
    summary.to_csv(output_dir / "candidate_summary.csv", index=False)
    summary.sort_values(["total_return", "sharpe"], ascending=[False, False]).to_csv(
        output_dir / "candidate_summary_by_net.csv", index=False
    )
    summary.sort_values(["gross_total_return_before_cost", "gross_max_drawdown"], ascending=[False, False]).to_csv(
        output_dir / "candidate_summary_by_gross.csv", index=False
    )
    summary.sort_values(["net_corr_to_locked_baseline", "total_return"], ascending=[True, False]).to_csv(
        output_dir / "candidate_summary_by_low_corr.csv", index=False
    )

    print("[4/8] Computing unified OOS IC against P1 slow labels", flush=True)
    ic_summary = build_ic_summary(panel, selections, candidates)
    ic_summary.to_csv(output_dir / "ic_summary.csv", index=False)
    ic_summary.sort_values(["label", "mean_ic"], ascending=[True, False]).to_csv(
        output_dir / "ic_summary_by_label.csv", index=False
    )

    print("[5/8] Building baseline comparison and best-run diagnostics", flush=True)
    baseline = load_baseline_summary(EXPERIMENT_ROOT / "baseline_lock_review" / "horizon_summary.csv")
    comparison = build_baseline_comparison(summary, baseline)
    comparison.to_csv(output_dir / "baseline_comparison.csv", index=False)

    best_key = str(summary.sort_values(["total_return", "sharpe"], ascending=[False, False]).iloc[0]["run"])
    best_result = result_index[best_key]
    write_run_result(ensure_dir(output_dir / "best_run"), best_result)
    best_horizon = int(summary.loc[summary["run"] == best_key, "horizon"].iloc[0])
    best_equity = best_result["equity"].sort_values("timestamp")
    best_positions = best_result["positions"].sort_values(["timestamp", "symbol"])
    regime_summary = build_regime_summary(best_equity, build_regime_frame(panel), best_horizon)
    side_summary = build_side_summary(best_positions)
    asset_summary = build_asset_summary(best_positions)
    regime_summary.to_csv(output_dir / "best_regime_summary.csv", index=False)
    side_summary.to_csv(output_dir / "best_side_contribution.csv", index=False)
    asset_summary.to_csv(output_dir / "best_asset_contribution.csv", index=False)

    print("[6/8] Running top-run cost sensitivity", flush=True)
    top_keys = unique_ordered(
        list(summary.sort_values(["total_return", "sharpe"], ascending=[False, False])["run"].head(5))
        + list(summary.sort_values(["gross_total_return_before_cost", "gross_max_drawdown"], ascending=[False, False])["run"].head(5))
    )
    cost_summary = build_cost_sensitivity({key: result_index[key] for key in top_keys}, cost_bps)
    cost_summary.to_csv(output_dir / "top_run_cost_sensitivity.csv", index=False)

    print("[7/8] Writing plots", flush=True)
    plot_summary(output_dir, summary)
    plot_best_curves(output_dir, best_equity, best_key)
    plot_top_curves(output_dir, result_index, top_keys[:8])
    plot_ic_summary(output_dir, ic_summary)
    plot_family_summary(output_dir, summary)
    plot_cost_sensitivity(output_dir, cost_summary)
    plot_asset_contribution(output_dir, asset_summary)

    print("[8/8] Writing report", flush=True)
    write_report(
        output_dir=output_dir,
        summary=summary,
        ic_summary=ic_summary,
        comparison=comparison,
        best_key=best_key,
        regime_summary=regime_summary,
        side_summary=side_summary,
        asset_summary=asset_summary,
        cost_summary=cost_summary,
        top_keys=top_keys,
    )
    print(f"Done: {output_dir}", flush=True)


def resolve_config_path(config_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def build_manual_slow_hybrid_features(panel: pd.DataFrame) -> list[dict[str, Any]]:
    required = [
        "alpha_A_res_trend_quality",
        "alpha_B_seed_combo",
        "alpha_B_soft_combo",
        "cs_res_mom_168_24",
        "cs_res_mom_336_24",
        "cs_res_high_mom_168",
        "cs_res_high_mom_336",
        "cs_res_ma_slope_48_168",
        "cs_vol_of_vol_res_168",
        "cs_volume_share_trend_7d_30d",
        "cs_res_z_168",
        "cs_res_z_336",
        "cs_ou_speed_168",
        "cs_beta_stability_168",
        "cs_liquidity_quality",
        "pct_ou_speed_168",
        "pct_ou_speed_336",
        "pct_beta_stability_168",
        "pct_liquidity_quality",
        "stationarity_gate_168",
        "stationarity_gate_336",
    ]
    missing = [col for col in required if col not in panel.columns]
    if missing:
        raise KeyError(f"Missing required slow feature columns: {missing}")

    panel["alpha_C_slow_mom_low_vov"] = (
        panel["cs_res_mom_336_24"] - 0.50 * panel["cs_vol_of_vol_res_168"] + 0.25 * panel["cs_liquidity_quality"]
    )
    panel["alpha_C_liq_trend_mom"] = (
        panel["cs_res_mom_168_24"] + 0.50 * panel["cs_volume_share_trend_7d_30d"] + 0.20 * panel["cs_liquidity_quality"]
    )
    panel["alpha_C_stationary_revert_168"] = (
        -panel["cs_res_z_168"] * panel["stationarity_gate_168"] * panel["pct_beta_stability_168"].fillna(0.0)
    )
    panel["alpha_C_soft_revert_liq"] = panel["alpha_B_soft_combo"] + 0.20 * panel["cs_liquidity_quality"]
    panel["alpha_C_A_B_seed_blend"] = 0.50 * panel["alpha_A_res_trend_quality"] + 0.50 * panel["alpha_B_seed_combo"]
    panel["alpha_C_A_B_soft_blend"] = 0.40 * panel["alpha_A_res_trend_quality"] + 0.60 * panel["alpha_B_soft_combo"]
    panel["alpha_C_mom_when_stationary"] = panel["cs_res_mom_168_24"] * panel["stationarity_gate_168"]
    panel["alpha_C_revert_low_vov"] = (
        -panel["cs_res_z_168"] * panel["stationarity_gate_168"] - 0.30 * panel["cs_vol_of_vol_res_168"]
    )
    panel["alpha_C_quality_committee"] = (
        0.35 * panel["alpha_A_res_trend_quality"]
        + 0.35 * panel["alpha_B_soft_combo"]
        + 0.30 * panel["cs_liquidity_quality"]
        - 0.20 * panel["cs_vol_of_vol_res_168"]
    )
    panel["alpha_C_low_turnover_trend"] = (
        0.50 * panel["cs_res_mom_336_24"]
        + 0.30 * panel["cs_res_ma_slope_48_168"]
        - 0.20 * panel["cs_vol_of_vol_res_168"]
    )
    panel["alpha_C_high_mom_liq"] = (
        panel["cs_res_high_mom_336"] + 0.30 * panel["cs_liquidity_quality"] - 0.20 * panel["cs_vol_of_vol_res_168"]
    )
    panel["alpha_C_ou_confirmed_mom"] = (
        panel["cs_res_mom_336_24"] * panel["pct_ou_speed_336"].fillna(0.0)
        + 0.25 * panel["cs_beta_stability_168"]
    )

    specs: list[tuple[str, str, str, list[str], str, bool]] = [
        (
            "C01_slow_mom_low_vov",
            "alpha_C_slow_mom_low_vov",
            "rank_cs(res_mom_336_24)-0.50*rank_cs(vol_of_vol_res_168)+0.25*rank_cs(liquidity_quality)",
            ["res_mom_336_24", "vol_of_vol_res_168", "liquidity_quality"],
            "C_slow_manual_hybrid",
            False,
        ),
        (
            "C02_liq_trend_mom",
            "alpha_C_liq_trend_mom",
            "rank_cs(res_mom_168_24)+0.50*rank_cs(volume_share_trend_7d_30d)+0.20*rank_cs(liquidity_quality)",
            ["res_mom_168_24", "volume_share_trend_7d_30d", "liquidity_quality"],
            "C_slow_manual_hybrid",
            False,
        ),
        (
            "C03_stationary_revert_168",
            "alpha_C_stationary_revert_168",
            "-rank_cs(res_z_168)*stationarity_gate_168*pct_rank(beta_stability_168)",
            ["res_z_168", "stationarity_gate_168", "beta_stability_168"],
            "C_stationary_reversion",
            True,
        ),
        (
            "C04_soft_revert_liq",
            "alpha_C_soft_revert_liq",
            "alpha_B_soft_combo+0.20*rank_cs(liquidity_quality)",
            ["alpha_B_soft_combo", "liquidity_quality"],
            "C_stationary_reversion",
            True,
        ),
        (
            "C05_A_B_seed_blend",
            "alpha_C_A_B_seed_blend",
            "0.50*alpha_A_res_trend_quality+0.50*alpha_B_seed_combo",
            ["alpha_A_res_trend_quality", "alpha_B_seed_combo"],
            "C_committee_blend",
            True,
        ),
        (
            "C06_A_B_soft_blend",
            "alpha_C_A_B_soft_blend",
            "0.40*alpha_A_res_trend_quality+0.60*alpha_B_soft_combo",
            ["alpha_A_res_trend_quality", "alpha_B_soft_combo"],
            "C_committee_blend",
            True,
        ),
        (
            "C07_mom_when_stationary",
            "alpha_C_mom_when_stationary",
            "rank_cs(res_mom_168_24)*stationarity_gate_168",
            ["res_mom_168_24", "stationarity_gate_168"],
            "C_slow_manual_hybrid",
            True,
        ),
        (
            "C08_revert_low_vov",
            "alpha_C_revert_low_vov",
            "-rank_cs(res_z_168)*stationarity_gate_168-0.30*rank_cs(vol_of_vol_res_168)",
            ["res_z_168", "stationarity_gate_168", "vol_of_vol_res_168"],
            "C_stationary_reversion",
            True,
        ),
        (
            "C09_quality_committee",
            "alpha_C_quality_committee",
            "0.35*A_trend_quality+0.35*B_soft_combo+0.30*liquidity_quality-0.20*vol_of_vol",
            ["alpha_A_res_trend_quality", "alpha_B_soft_combo", "liquidity_quality", "vol_of_vol_res_168"],
            "C_committee_blend",
            True,
        ),
        (
            "C10_low_turnover_trend",
            "alpha_C_low_turnover_trend",
            "0.50*rank_cs(res_mom_336_24)+0.30*rank_cs(res_ma_slope_48_168)-0.20*rank_cs(vol_of_vol_res_168)",
            ["res_mom_336_24", "res_ma_slope_48_168", "vol_of_vol_res_168"],
            "C_slow_manual_hybrid",
            False,
        ),
        (
            "C11_high_mom_liq",
            "alpha_C_high_mom_liq",
            "rank_cs(res_high_mom_336)+0.30*rank_cs(liquidity_quality)-0.20*rank_cs(vol_of_vol_res_168)",
            ["res_high_mom_336", "liquidity_quality", "vol_of_vol_res_168"],
            "C_slow_manual_hybrid",
            False,
        ),
        (
            "C12_ou_confirmed_mom",
            "alpha_C_ou_confirmed_mom",
            "rank_cs(res_mom_336_24)*pct_rank(ou_speed_336)+0.25*rank_cs(beta_stability_168)",
            ["res_mom_336_24", "ou_speed_336", "beta_stability_168"],
            "C_slow_manual_hybrid",
            True,
        ),
    ]

    candidates = []
    for name, score_col, formula, features, family, use_regime_gate in specs:
        candidates.append(
            {
                "name": name,
                "score_col": score_col,
                "formula": formula,
                "features": "|".join(features),
                "family": family,
                "use_regime_gate": use_regime_gate,
            }
        )
    return candidates


def write_feature_panel(panel: pd.DataFrame, candidates: list[dict[str, Any]], output_path: Path) -> None:
    base_cols = {
        "timestamp",
        "symbol",
        "family_b_regime_gate",
        "market_vol_14d",
        "residual_dispersion_7d",
    }
    feature_cols = set(base_cols)
    for candidate in candidates:
        feature_cols.add(str(candidate["score_col"]))
        for feature in str(candidate["features"]).split("|"):
            if feature:
                feature_cols.add(feature)
                feature_cols.add(f"cs_{feature}")
                feature_cols.add(f"pct_{feature}")
    existing = ["timestamp", "symbol"] + sorted(col for col in feature_cols if col in panel.columns and col not in {"timestamp", "symbol"})
    panel[existing].to_csv(output_path, index=False)


def build_locked_baseline_correlations(
    result_index: dict[str, dict[str, Any]], summary: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    horizon_by_run = summary.set_index("run")["horizon"].to_dict()
    for run, result in result_index.items():
        horizon = int(horizon_by_run[run])
        base_path = EXPERIMENT_ROOT / "baseline_lock_review" / "horizon" / f"h{horizon:03d}_tau0" / "equity_curve.csv"
        row: dict[str, Any] = {"run": run}
        if not base_path.exists() or result["equity"].empty:
            row.update(
                {
                    "net_corr_to_locked_baseline": np.nan,
                    "gross_corr_to_locked_baseline": np.nan,
                    "aligned_periods_with_locked_baseline": 0,
                }
            )
            rows.append(row)
            continue
        base = pd.read_csv(base_path, parse_dates=["timestamp"]).sort_values("timestamp")
        equity = result["equity"].sort_values("timestamp")
        merged = base[["timestamp", "period_return", "gross_period_return"]].rename(
            columns={"period_return": "baseline_net", "gross_period_return": "baseline_gross"}
        ).merge(
            equity[["timestamp", "period_return", "gross_period_return"]].rename(
                columns={"period_return": "candidate_net", "gross_period_return": "candidate_gross"}
            ),
            on="timestamp",
            how="inner",
        )
        row["aligned_periods_with_locked_baseline"] = int(len(merged))
        row["net_corr_to_locked_baseline"] = (
            float(merged["baseline_net"].corr(merged["candidate_net"])) if len(merged) > 3 else np.nan
        )
        row["gross_corr_to_locked_baseline"] = (
            float(merged["baseline_gross"].corr(merged["candidate_gross"])) if len(merged) > 3 else np.nan
        )
        rows.append(row)
    return pd.DataFrame(rows)


def unique_ordered(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def plot_summary(output_dir: Path, summary: pd.DataFrame) -> None:
    top = summary.sort_values(["total_return", "sharpe"], ascending=[False, False]).head(18).sort_values("total_return")
    plt.figure(figsize=(12, 8))
    colors = np.where(top["total_return"] >= 0.0, "#2a9d8f", "#c44536")
    plt.barh(top["run"], top["total_return"], color=colors)
    plt.axvline(0.0, color="black", linewidth=0.8, alpha=0.6)
    plt.title("P2 manual slow alpha grid: top runs by net return")
    plt.xlabel("Net cumulative return")
    plt.gca().xaxis.set_major_formatter(PercentFormatter(1.0))
    plt.tight_layout()
    plt.savefig(output_dir / "manual_grid_top_net_returns.png", dpi=180)
    plt.close()


def plot_best_curves(output_dir: Path, equity: pd.DataFrame, best_key: str) -> None:
    if equity.empty:
        return
    frame = equity.sort_values("timestamp")
    net = (1.0 + frame["period_return"]).cumprod() - 1.0
    gross = (1.0 + frame["gross_period_return"]).cumprod() - 1.0
    plt.figure(figsize=(12, 6.5))
    plt.plot(frame["timestamp"], gross, label="gross before cost", linewidth=2)
    plt.plot(frame["timestamp"], net, label="net after 6bps cost", linewidth=2)
    plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    plt.title(f"Best P2 manual slow alpha run: {best_key}")
    plt.ylabel("Cumulative return")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "best_net_gross_curve.png", dpi=180)
    plt.close()


def plot_top_curves(output_dir: Path, result_index: dict[str, dict[str, Any]], top_keys: list[str]) -> None:
    plt.figure(figsize=(12, 6.5))
    for key in top_keys:
        equity = result_index[key]["equity"].sort_values("timestamp")
        if equity.empty:
            continue
        net = (1.0 + equity["period_return"]).cumprod() - 1.0
        plt.plot(equity["timestamp"], net, label=key, linewidth=1.6)
    plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    plt.title("P2 manual grid selected top net curves")
    plt.ylabel("Cumulative return")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(output_dir / "top_net_curves.png", dpi=180)
    plt.close()


def plot_ic_summary(output_dir: Path, ic_summary: pd.DataFrame) -> None:
    target = ic_summary[ic_summary["label"] == "label_path_slow_low_turnover"].copy()
    if target.empty:
        return
    target = target.sort_values("mean_ic").tail(18)
    plt.figure(figsize=(12, 8))
    colors = np.where(target["mean_ic"] >= 0.0, "#2a9d8f", "#c44536")
    plt.barh(target["name"], target["mean_ic"], color=colors)
    plt.axvline(0.0, color="black", linewidth=0.8, alpha=0.6)
    plt.title("P2 manual grid OOS IC vs path slow low-turnover label")
    plt.xlabel("Mean cross-sectional Spearman IC")
    plt.grid(True, axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "ic_path_slow_low_turnover.png", dpi=180)
    plt.close()


def plot_family_summary(output_dir: Path, summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    grouped = (
        summary.groupby(["family", "horizon"])
        .agg(
            best_net=("total_return", "max"),
            median_net=("total_return", "median"),
            best_gross=("gross_total_return_before_cost", "max"),
            median_turnover=("mean_turnover", "median"),
        )
        .reset_index()
    )
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    for family, group in grouped.groupby("family"):
        group = group.sort_values("horizon")
        axes[0].plot(group["horizon"], group["best_net"], marker="o", label=family)
        axes[1].plot(group["horizon"], group["median_turnover"], marker="o", label=family)
    axes[0].set_title("Best net return by family and horizon")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].grid(True, alpha=0.25)
    axes[1].set_title("Median turnover by family and horizon")
    axes[1].set_xlabel("horizon hours")
    axes[1].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "family_horizon_summary.png", dpi=180)
    plt.close(fig)
    grouped.to_csv(output_dir / "family_horizon_summary.csv", index=False)


def plot_cost_sensitivity(output_dir: Path, cost_summary: pd.DataFrame) -> None:
    if cost_summary.empty:
        return
    plt.figure(figsize=(12, 6.5))
    for name, group in cost_summary.groupby("experiment"):
        group = group.sort_values("one_way_cost_bps")
        plt.plot(group["one_way_cost_bps"], group["total_return"], marker="o", label=name)
    plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    plt.title("P2 manual grid top-run cost sensitivity")
    plt.xlabel("one-way cost bps")
    plt.ylabel("Net cumulative return")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(output_dir / "top_run_cost_sensitivity.png", dpi=180)
    plt.close()


def plot_asset_contribution(output_dir: Path, asset_summary: pd.DataFrame) -> None:
    if asset_summary.empty:
        return
    frame = asset_summary.sort_values("gross_contribution_sum")
    plt.figure(figsize=(11, 7))
    colors = np.where(frame["gross_contribution_sum"] >= 0.0, "#2a9d8f", "#c44536")
    plt.barh(frame["symbol"], frame["gross_contribution_sum"], color=colors)
    plt.axvline(0.0, color="black", linewidth=0.8, alpha=0.6)
    plt.title("Best P2 run gross contribution by asset")
    plt.xlabel("Gross contribution sum")
    plt.gca().xaxis.set_major_formatter(PercentFormatter(1.0))
    plt.tight_layout()
    plt.savefig(output_dir / "best_asset_contribution.png", dpi=180)
    plt.close()


def write_report(
    output_dir: Path,
    summary: pd.DataFrame,
    ic_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    best_key: str,
    regime_summary: pd.DataFrame,
    side_summary: pd.DataFrame,
    asset_summary: pd.DataFrame,
    cost_summary: pd.DataFrame,
    top_keys: list[str],
) -> None:
    display_cols = [
        "run",
        "family",
        "horizon",
        "name",
        "total_return",
        "gross_total_return_before_cost",
        "sharpe",
        "max_drawdown",
        "gross_max_drawdown",
        "mean_turnover",
        "cost_drag",
        "positive_fold_rate",
        "mean_regime_gross_scale",
        "low_regime_gross_scale_rate",
        "net_corr_to_locked_baseline",
        "mean_abs_beta_market",
        "max_abs_weight",
    ]
    comp_cols = [
        "run",
        "family",
        "horizon",
        "name",
        "total_return",
        "baseline_total_return",
        "delta_total_return",
        "gross_total_return_before_cost",
        "baseline_gross_total_return_before_cost",
        "delta_gross_total_return_before_cost",
        "mean_turnover",
        "baseline_mean_turnover",
        "delta_mean_turnover",
        "net_corr_to_locked_baseline",
    ]
    ic_focus = ic_summary[
        ic_summary["label"].isin(["label_slow_low_turnover", "label_path_slow_low_turnover"])
    ].sort_values(["label", "mean_ic"], ascending=[True, False])
    cost_pivot = (
        cost_summary.pivot_table(index="one_way_cost_bps", columns="experiment", values="total_return")
        .reset_index()
        .sort_values("one_way_cost_bps")
    )
    family_summary = pd.read_csv(output_dir / "family_horizon_summary.csv") if (output_dir / "family_horizon_summary.csv").exists() else pd.DataFrame()
    lines = [
        "# P2 Manual Slow Alpha Grid",
        "",
        "## Design",
        "",
        "- Unified OHLCV-only slow residual feature panel.",
        "- Includes Family A residual momentum/high-momentum, Family B stationary mean reversion, and additional C-family manual slow/hybrid candidates.",
        "- All candidates are evaluated with the same strict-neutral allocator, same walk-forward folds, same fee/slippage, same labels, and same diagnostics.",
        "- Family B and stationarity-driven C candidates use the same ex-ante regime gross gate; momentum-only candidates do not.",
        "- Locked-baseline correlation is computed against the same-horizon locked baseline curve.",
        "",
        "## Best Run",
        "",
        f"- best_run: `{best_key}`",
        "",
        "## Candidate Summary",
        "",
        "```text",
        summary.sort_values(["total_return", "sharpe"], ascending=[False, False])[
            [col for col in display_cols if col in summary.columns]
        ].head(40).to_string(index=False),
        "```",
        "",
        "## Family / Horizon Summary",
        "",
        "```text",
        family_summary.to_string(index=False),
        "```",
        "",
        "## Baseline Comparison",
        "",
        "```text",
        comparison[[col for col in comp_cols if col in comparison.columns]].head(40).to_string(index=False),
        "```",
        "",
        "## OOS IC Focus",
        "",
        "```text",
        ic_focus.head(80).to_string(index=False),
        "```",
        "",
        "## Top-Run Cost Sensitivity",
        "",
        f"Top-run set: `{', '.join(top_keys)}`",
        "",
        "```text",
        cost_pivot.to_string(index=False),
        "```",
        "",
        "## Best Run Regime Summary",
        "",
        "```text",
        regime_summary.to_string(index=False),
        "```",
        "",
        "## Best Run Long/Short Contribution",
        "",
        "```text",
        side_summary.to_string(index=False),
        "```",
        "",
        "## Best Run Asset Contribution",
        "",
        "```text",
        asset_summary.to_string(index=False),
        "```",
        "",
        "## Files",
        "",
        "- `manual_slow_feature_panel.csv.gz`: unified OHLCV-only slow residual features and alpha scores.",
        "- `candidate_formulas.csv`: candidate definitions.",
        "- `candidate_summary.csv`: all portfolio metrics and locked-baseline correlations.",
        "- `candidate_summary_by_net.csv`, `candidate_summary_by_gross.csv`, `candidate_summary_by_low_corr.csv`: ranking views.",
        "- `ic_summary.csv`: OOS IC against P1 slow labels.",
        "- `baseline_comparison.csv`: deltas versus same-horizon locked baseline.",
        "- `family_horizon_summary.csv`: family-level horizon rollup.",
        "- `runs/`: raw folds/equity/positions per run.",
        "- `best_run/`: raw details for the best run.",
        "- plots: `manual_grid_top_net_returns.png`, `best_net_gross_curve.png`, `top_net_curves.png`, `ic_path_slow_low_turnover.png`, `family_horizon_summary.png`, `top_run_cost_sensitivity.png`, `best_asset_contribution.png`.",
        "",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
