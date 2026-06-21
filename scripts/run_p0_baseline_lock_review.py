#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
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
from crypto_stat_arb.portfolio import PortfolioConfig
from crypto_stat_arb.walk_forward import WalkForwardConfig
from run_delay_smoothing_experiments import DelayConfig, run_delay_replay
from run_residual_alphagpt_search import build_candidate_scores, prepare_alpha_panel
from run_risk_control_experiments import RiskControlConfig
from run_turnover_experiments import to_jsonable


BASE_PANEL = PROJECT_ROOT / "outputs" / "baseline_residual_24h" / "asset_residual_panel.csv"
SELECTIONS = PROJECT_ROOT / "outputs" / "residual_alphagpt_search" / "h048_minhold5d" / "committee_top3" / "formula_selections.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "p0_baseline_lock_review"
LOCKED_BASELINE = "cap12p5_shortconfirm_hedged_clip_gross"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "baseline.yaml"))
    parser.add_argument("--panel", default=str(BASE_PANEL))
    parser.add_argument("--selections", default=str(SELECTIONS))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--horizons", type=int, nargs="+", default=[24, 48, 72, 96, 168])
    parser.add_argument("--tau-days", type=float, nargs="+", default=[0, 1, 2, 3, 5, 7, 10, 14, 21, 30])
    parser.add_argument("--cost-bps", type=float, nargs="+", default=[0, 2, 4, 6, 8, 10, 12, 15, 20, 30])
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    cfg = load_config(args.config)

    horizons = sorted(set(int(h) for h in args.horizons))
    tau_values = [float(x) for x in args.tau_days]
    cost_bps = [float(x) for x in args.cost_bps]

    print("[1/7] Loading panel, selected committees, and candidate scores", flush=True)
    panel = pd.read_csv(args.panel, parse_dates=["timestamp"])
    panel = prepare_alpha_panel(panel, horizons, int(cfg["labels"]["execution_delay_bars"]))
    build_candidate_scores(panel)
    selections = pd.read_csv(args.selections, parse_dates=["train_start", "train_end", "test_start", "test_end"])

    wf_cfg = WalkForwardConfig(
        train_days=int(cfg["walk_forward"]["train_days"]),
        test_days=int(cfg["walk_forward"]["test_days"]),
        step_days=int(cfg["walk_forward"]["step_days"]),
        mode=str(cfg["walk_forward"].get("mode", "rolling")),
    )
    risk_cfg = locked_risk_config()

    lock_meta = {
        "baseline_name": LOCKED_BASELINE,
        "panel": str(Path(args.panel).resolve()),
        "selections": str(Path(args.selections).resolve()),
        "horizons": horizons,
        "tau_days": tau_values,
        "cost_bps": cost_bps,
        "risk_config": asdict(risk_cfg),
        "portfolio_defaults": cfg["portfolio"],
        "label_defaults": cfg["labels"],
        "walk_forward_defaults": cfg["walk_forward"],
    }
    write_json(output_dir / "BASELINE_LOCK.json", lock_meta)

    print("[2/7] Running horizon stress tests with tau=0", flush=True)
    horizon_rows: list[dict[str, Any]] = []
    horizon_results: dict[int, dict[str, Any]] = {}
    for horizon in horizons:
        print(f"      horizon={horizon}h", flush=True)
        pf_cfg = make_portfolio_config(cfg, horizon)
        result = run_delay_replay(panel, selections, pf_cfg, wf_cfg, risk_cfg, DelayConfig("tau_0", 0.0))
        exp_dir = ensure_dir(output_dir / "horizon" / f"h{horizon:03d}_tau0")
        write_replay_result(exp_dir, result)
        horizon_results[horizon] = result
        horizon_rows.append({"horizon": horizon, "min_hold_rebalances": pf_cfg.min_hold_rebalances, **result["metrics"]})
    horizon_summary = pd.DataFrame(horizon_rows).sort_values("horizon")
    horizon_summary.to_csv(output_dir / "horizon_summary.csv", index=False)

    base_horizon = 48
    if base_horizon not in horizon_results:
        raise ValueError("P0 lock expects horizon 48 to be included.")
    locked_result = horizon_results[base_horizon]

    print("[3/7] Running delay/tau sweep for locked 48h baseline", flush=True)
    tau_rows: list[dict[str, Any]] = []
    tau_results: dict[str, dict[str, Any]] = {}
    pf_48 = make_portfolio_config(cfg, base_horizon)
    for tau in tau_values:
        name = f"tau_{format_number(tau)}"
        print(f"      {name}", flush=True)
        result = locked_result if abs(tau) < 1e-12 else run_delay_replay(
            panel, selections, pf_48, wf_cfg, risk_cfg, DelayConfig(name, tau)
        )
        exp_dir = ensure_dir(output_dir / "tau_sweep" / name)
        write_replay_result(exp_dir, result)
        tau_results[name] = result
        tau_rows.append({"experiment": name, "tau_days": tau, **result["metrics"]})
    tau_summary = pd.DataFrame(tau_rows).sort_values("tau_days")
    tau_summary.to_csv(output_dir / "tau_summary.csv", index=False)
    tau_summary.sort_values("total_return", ascending=False).to_csv(output_dir / "tau_summary_by_net.csv", index=False)
    tau_summary.sort_values("mean_turnover").to_csv(output_dir / "tau_summary_by_turnover.csv", index=False)

    print("[4/7] Running cost sensitivity", flush=True)
    cost_summary = build_cost_sensitivity(tau_results, cost_bps)
    cost_summary.to_csv(output_dir / "cost_sensitivity_by_tau.csv", index=False)
    locked_cost = cost_summary[cost_summary["experiment"] == "tau_0"].copy()
    locked_cost.to_csv(output_dir / "locked_baseline_cost_sensitivity.csv", index=False)

    print("[5/7] Building regime, side, and asset contribution reviews", flush=True)
    regime_frame = build_regime_frame(panel)
    regime_frame.to_csv(output_dir / "regime_frame.csv", index=False)
    locked_equity = locked_result["equity"].sort_values("timestamp").copy()
    locked_positions = locked_result["positions"].sort_values(["timestamp", "symbol"]).copy()
    regime_summary = build_regime_summary(locked_equity, regime_frame, pf_48.rebalance_hours)
    regime_summary.to_csv(output_dir / "regime_summary.csv", index=False)
    side_summary = build_side_summary(locked_positions)
    side_summary.to_csv(output_dir / "side_contribution.csv", index=False)
    asset_summary = build_asset_summary(locked_positions)
    asset_summary.to_csv(output_dir / "asset_contribution.csv", index=False)
    formula_counts = build_formula_counts(selections)
    formula_counts.to_csv(output_dir / "selected_formula_counts.csv", index=False)

    print("[6/7] Writing plots", flush=True)
    plot_locked_curves(output_dir, locked_equity)
    plot_horizon_summary(output_dir, horizon_summary)
    plot_tau_summary(output_dir, tau_summary)
    plot_cost_sensitivity(output_dir, cost_summary)
    plot_regime_summary(output_dir, regime_summary)
    plot_asset_contribution(output_dir, asset_summary)

    print("[7/7] Writing report", flush=True)
    write_report(
        output_dir=output_dir,
        lock_meta=lock_meta,
        horizon_summary=horizon_summary,
        tau_summary=tau_summary,
        cost_summary=cost_summary,
        regime_summary=regime_summary,
        side_summary=side_summary,
        asset_summary=asset_summary,
        formula_counts=formula_counts,
    )
    print(f"Done: {output_dir}", flush=True)


def locked_risk_config() -> RiskControlConfig:
    return RiskControlConfig(
        name=LOCKED_BASELINE,
        description="12.5% cap + short confirmation + all-symbol constrained hedge projection, gross limited to clipped target.",
        max_symbol_weight=0.125,
        short_resid_confirm_rank=0.35,
        allocation_method="hedged_constrained_projection",
        optimizer_target_gross_mode="clip_gross",
        hedge_penalty=25.0,
    )


def make_portfolio_config(cfg: dict[str, Any], horizon: int) -> PortfolioConfig:
    return PortfolioConfig(
        rebalance_hours=horizon,
        long_k=int(cfg["portfolio"]["long_k"]),
        short_k=int(cfg["portfolio"]["short_k"]),
        gross_exposure=float(cfg["portfolio"]["gross_exposure"]),
        one_way_fee_bps=float(cfg["portfolio"]["one_way_fee_bps"]),
        one_way_slippage_bps=float(cfg["portfolio"]["one_way_slippage_bps"]),
        neutralize_exposures=list(cfg["portfolio"]["neutralize_exposures"]),
        primary_horizon=horizon,
        execution_delay_bars=int(cfg["labels"]["execution_delay_bars"]),
        use_hysteresis=True,
        min_hold_rebalances=max(1, math.ceil(5 * 24 / horizon)),
    )


def write_replay_result(output_dir: Path, result: dict[str, Any]) -> None:
    result["folds"].to_csv(output_dir / "walk_forward_folds.csv", index=False)
    result["equity"].to_csv(output_dir / "equity_curve.csv", index=False)
    result["positions"].to_csv(output_dir / "positions.csv", index=False)
    write_json(output_dir / "metrics.json", result["metrics"])


def build_cost_sensitivity(results: dict[str, dict[str, Any]], cost_bps: list[float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, result in results.items():
        equity = result["equity"].sort_values("timestamp").copy()
        if equity.empty:
            continue
        tau_days = float(equity["delay_tau_days"].dropna().iloc[0]) if "delay_tau_days" in equity else np.nan
        for bps in cost_bps:
            scenario = equity["gross_period_return"].astype(float) - equity["turnover"].astype(float) * bps / 10_000.0
            curve = (1.0 + scenario).cumprod()
            drawdown = curve / curve.cummax() - 1.0
            rows.append(
                {
                    "experiment": name,
                    "tau_days": tau_days,
                    "one_way_cost_bps": bps,
                    "total_return": float(curve.iloc[-1] - 1.0),
                    "max_drawdown": float(drawdown.min()),
                    "mean_period_return": float(scenario.mean()),
                    "positive_period_rate": float((scenario > 0).mean()),
                    "cost_drag_sum": float((equity["turnover"].astype(float) * bps / 10_000.0).sum()),
                }
            )
    return pd.DataFrame(rows).sort_values(["one_way_cost_bps", "tau_days"]).reset_index(drop=True)


def build_regime_frame(panel: pd.DataFrame) -> pd.DataFrame:
    ordered = panel.sort_values(["timestamp", "symbol"]).copy()
    market_ret = ordered.groupby("timestamp")["log_ret"].mean()
    residual_dispersion = ordered.groupby("timestamp")["residual_log_ret"].std()
    total_quote_volume = ordered.groupby("timestamp")["quote_volume"].sum()

    volume_share = ordered["quote_volume"] / ordered.groupby("timestamp")["quote_volume"].transform("sum")
    volume_concentration = (volume_share**2).groupby(ordered["timestamp"]).sum()

    frame = pd.DataFrame(
        {
            "timestamp": market_ret.index,
            "market_ret": market_ret.to_numpy(),
            "residual_dispersion": residual_dispersion.reindex(market_ret.index).to_numpy(),
            "total_quote_volume": total_quote_volume.reindex(market_ret.index).to_numpy(),
            "volume_concentration": volume_concentration.reindex(market_ret.index).to_numpy(),
        }
    ).sort_values("timestamp")

    frame["market_ret_7d"] = frame["market_ret"].rolling(168, min_periods=48).sum()
    frame["market_ret_30d"] = frame["market_ret"].rolling(720, min_periods=168).sum()
    frame["market_vol_14d"] = frame["market_ret"].rolling(336, min_periods=72).std(ddof=0)
    frame["residual_dispersion_7d"] = frame["residual_dispersion"].rolling(168, min_periods=48).mean()
    frame["log_liquidity_7d"] = np.log(frame["total_quote_volume"].replace(0.0, np.nan)).rolling(168, min_periods=48).mean()
    frame["volume_concentration_7d"] = frame["volume_concentration"].rolling(168, min_periods=48).mean()

    frame["market_trend"] = np.where(frame["market_ret_30d"] >= 0.0, "up_30d", "down_30d")
    frame["crash_state"] = np.where(frame["market_ret_7d"] <= -0.05, "crash_7d", "normal")
    frame["market_vol_bucket"] = tertile_bucket(frame["market_vol_14d"], "vol")
    frame["residual_dispersion_bucket"] = tertile_bucket(frame["residual_dispersion_7d"], "resid_disp")
    frame["liquidity_bucket"] = tertile_bucket(frame["log_liquidity_7d"], "liq")
    frame["volume_concentration_bucket"] = tertile_bucket(frame["volume_concentration_7d"], "vconc")
    return frame


def tertile_bucket(series: pd.Series, prefix: str) -> pd.Series:
    out = pd.Series("unknown", index=series.index, dtype=object)
    valid = series.dropna()
    if valid.empty:
        return out
    q1 = valid.quantile(1 / 3)
    q2 = valid.quantile(2 / 3)
    out.loc[series <= q1] = f"low_{prefix}"
    out.loc[(series > q1) & (series <= q2)] = f"mid_{prefix}"
    out.loc[series > q2] = f"high_{prefix}"
    return out


def build_regime_summary(equity: pd.DataFrame, regime_frame: pd.DataFrame, rebalance_hours: int) -> pd.DataFrame:
    if equity.empty:
        return pd.DataFrame()
    joined = equity.merge(regime_frame, on="timestamp", how="left")
    rows: list[dict[str, Any]] = []
    regime_cols = [
        "market_trend",
        "crash_state",
        "market_vol_bucket",
        "residual_dispersion_bucket",
        "liquidity_bucket",
        "volume_concentration_bucket",
    ]
    for col in regime_cols:
        for bucket, group in joined.groupby(col, dropna=False):
            if len(group) < 3:
                continue
            rows.append({"regime_type": col, "regime": str(bucket), **summarize_periods(group, rebalance_hours)})
    return pd.DataFrame(rows).sort_values(["regime_type", "total_return"], ascending=[True, False])


def summarize_periods(frame: pd.DataFrame, rebalance_hours: int) -> dict[str, float]:
    period = frame["period_return"].astype(float)
    gross = frame["gross_period_return"].astype(float)
    curve = (1.0 + period).cumprod()
    gross_curve = (1.0 + gross).cumprod()
    drawdown = curve / curve.cummax() - 1.0
    periods_per_year = 365.25 * 24.0 / rebalance_hours
    sharpe = np.nan
    if period.std(ddof=0) > 0:
        sharpe = float(period.mean() / period.std(ddof=0) * np.sqrt(periods_per_year))
    return {
        "period_count": int(len(frame)),
        "total_return": float(curve.iloc[-1] - 1.0),
        "gross_total_return": float(gross_curve.iloc[-1] - 1.0),
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "cost_drag_sum": float(frame["cost_return"].sum()),
        "mean_turnover": float(frame["turnover"].mean()),
        "mean_gross_exposure": float(frame["gross_exposure"].mean()),
        "positive_period_rate": float((period > 0).mean()),
        "mean_rank_ic": float(frame["rank_ic"].mean()),
    }


def build_side_summary(positions: pd.DataFrame) -> pd.DataFrame:
    if positions.empty:
        return pd.DataFrame()
    frame = positions.copy()
    frame["side"] = np.where(frame["weight"] > 0.0, "long", "short")
    rows = []
    for side, group in frame.groupby("side"):
        rows.append(
            {
                "side": side,
                "gross_contribution_sum": float(group["contribution"].sum()),
                "mean_contribution": float(group["contribution"].mean()),
                "positive_position_rate": float((group["contribution"] > 0.0).mean()),
                "mean_abs_weight": float(group["weight"].abs().mean()),
                "max_abs_weight": float(group["weight"].abs().max()),
                "position_count": int(len(group)),
                "symbol_count": int(group["symbol"].nunique()),
            }
        )
    out = pd.DataFrame(rows)
    total_abs = out["gross_contribution_sum"].abs().sum()
    out["abs_contribution_share"] = out["gross_contribution_sum"].abs() / total_abs if total_abs > 0 else np.nan
    return out.sort_values("gross_contribution_sum", ascending=False)


def build_asset_summary(positions: pd.DataFrame) -> pd.DataFrame:
    if positions.empty:
        return pd.DataFrame()
    frame = positions.copy()
    frame["long_contribution"] = np.where(frame["weight"] > 0.0, frame["contribution"], 0.0)
    frame["short_contribution"] = np.where(frame["weight"] < 0.0, frame["contribution"], 0.0)
    grouped = frame.groupby("symbol")
    out = grouped.agg(
        gross_contribution_sum=("contribution", "sum"),
        long_contribution_sum=("long_contribution", "sum"),
        short_contribution_sum=("short_contribution", "sum"),
        mean_contribution=("contribution", "mean"),
        positive_position_rate=("contribution", lambda x: float((x > 0.0).mean())),
        mean_abs_weight=("weight", lambda x: float(x.abs().mean())),
        max_abs_weight=("weight", lambda x: float(x.abs().max())),
        position_count=("symbol", "size"),
    ).reset_index()
    total_abs = out["gross_contribution_sum"].abs().sum()
    out["abs_contribution_share"] = out["gross_contribution_sum"].abs() / total_abs if total_abs > 0 else np.nan
    return out.sort_values("gross_contribution_sum", ascending=False).reset_index(drop=True)


def build_formula_counts(selections: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in selections.iterrows():
        formulas = [item.strip() for item in str(row["selected_formulas"]).split("|")]
        score_cols = [item.strip() for item in str(row["selected_score_cols"]).split("|")]
        for formula, score_col in zip(formulas, score_cols, strict=False):
            rows.append({"formula": formula, "score_col": score_col, "fold": int(row["fold"])})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return (
        frame.groupby(["formula", "score_col"])
        .agg(selected_folds=("fold", "nunique"))
        .reset_index()
        .sort_values("selected_folds", ascending=False)
    )


def plot_locked_curves(output_dir: Path, equity: pd.DataFrame) -> None:
    if equity.empty:
        return
    frame = equity.sort_values("timestamp")
    net = (1.0 + frame["period_return"]).cumprod() - 1.0
    gross = (1.0 + frame["gross_period_return"]).cumprod() - 1.0
    plt.figure(figsize=(12, 6.5))
    plt.plot(frame["timestamp"], gross, label="gross before cost", linewidth=2.0)
    plt.plot(frame["timestamp"], net, label="net after 6bps cost", linewidth=2.0)
    plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    plt.title(f"Locked baseline: {LOCKED_BASELINE}")
    plt.ylabel("Cumulative return")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "locked_baseline_net_gross_curve.png", dpi=180)
    plt.close()


def plot_horizon_summary(output_dir: Path, summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    metrics = [
        ("gross_total_return_before_cost", "Gross return"),
        ("total_return", "Net return"),
        ("max_drawdown", "Max drawdown"),
        ("mean_turnover", "Mean turnover"),
    ]
    for ax, (col, title) in zip(axes.ravel(), metrics, strict=True):
        ax.plot(summary["horizon"], summary[col], marker="o", linewidth=2)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        if col != "mean_turnover":
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1, 0].set_xlabel("horizon hours")
    axes[1, 1].set_xlabel("horizon hours")
    fig.tight_layout()
    fig.savefig(output_dir / "horizon_summary.png", dpi=180)
    plt.close(fig)


def plot_tau_summary(output_dir: Path, summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    metrics = [
        ("gross_total_return_before_cost", "Gross return"),
        ("total_return", "Net return"),
        ("mean_turnover", "Mean turnover"),
        ("max_drawdown", "Max drawdown"),
    ]
    for ax, (col, title) in zip(axes.ravel(), metrics, strict=True):
        ax.plot(summary["tau_days"], summary[col], marker="o", linewidth=2)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        if col != "mean_turnover":
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1, 0].set_xlabel("tau days")
    axes[1, 1].set_xlabel("tau days")
    fig.tight_layout()
    fig.savefig(output_dir / "tau_summary.png", dpi=180)
    plt.close(fig)


def plot_cost_sensitivity(output_dir: Path, summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    selected = {"tau_0", "tau_1", "tau_3", "tau_5", "tau_10"}
    plt.figure(figsize=(12, 6.5))
    for name, group in summary.groupby("experiment"):
        if name not in selected:
            continue
        group = group.sort_values("one_way_cost_bps")
        plt.plot(group["one_way_cost_bps"], group["total_return"], marker="o", label=name)
    plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    plt.title("Cost sensitivity by delay tau")
    plt.xlabel("one-way cost bps")
    plt.ylabel("Net cumulative return")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "cost_sensitivity_by_tau.png", dpi=180)
    plt.close()


def plot_regime_summary(output_dir: Path, summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    frame = summary.copy()
    frame["label"] = frame["regime_type"] + ": " + frame["regime"]
    frame = frame.sort_values("total_return")
    plt.figure(figsize=(12, max(6, 0.33 * len(frame))))
    colors = np.where(frame["total_return"] >= 0.0, "#2a9d8f", "#c44536")
    plt.barh(frame["label"], frame["total_return"], color=colors)
    plt.axvline(0.0, color="black", linewidth=0.8, alpha=0.6)
    plt.title("Locked baseline net return by regime bucket")
    plt.xlabel("Net return")
    plt.gca().xaxis.set_major_formatter(PercentFormatter(1.0))
    plt.tight_layout()
    plt.savefig(output_dir / "regime_net_return.png", dpi=180)
    plt.close()


def plot_asset_contribution(output_dir: Path, summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    frame = summary.sort_values("gross_contribution_sum")
    plt.figure(figsize=(11, 7))
    colors = np.where(frame["gross_contribution_sum"] >= 0.0, "#2a9d8f", "#c44536")
    plt.barh(frame["symbol"], frame["gross_contribution_sum"], color=colors)
    plt.axvline(0.0, color="black", linewidth=0.8, alpha=0.6)
    plt.title("Locked baseline gross contribution by asset")
    plt.xlabel("Gross contribution sum")
    plt.gca().xaxis.set_major_formatter(PercentFormatter(1.0))
    plt.tight_layout()
    plt.savefig(output_dir / "asset_gross_contribution.png", dpi=180)
    plt.close()


def write_report(
    output_dir: Path,
    lock_meta: dict[str, Any],
    horizon_summary: pd.DataFrame,
    tau_summary: pd.DataFrame,
    cost_summary: pd.DataFrame,
    regime_summary: pd.DataFrame,
    side_summary: pd.DataFrame,
    asset_summary: pd.DataFrame,
    formula_counts: pd.DataFrame,
) -> None:
    metric_cols = [
        "total_return",
        "gross_total_return_before_cost",
        "sharpe",
        "max_drawdown",
        "gross_max_drawdown",
        "mean_turnover",
        "cost_drag",
        "mean_gross_exposure",
        "positive_fold_rate",
        "mean_abs_beta_market",
        "max_abs_beta_market",
        "max_abs_weight",
    ]
    horizon_cols = ["horizon", "min_hold_rebalances", *metric_cols]
    tau_cols = ["experiment", "tau_days", *metric_cols, "turnover_reduction", "gross_exposure_retention"]
    cost_pivot = (
        cost_summary.pivot_table(index="one_way_cost_bps", columns="experiment", values="total_return")
        .reset_index()
        .sort_values("one_way_cost_bps")
    )
    locked_row = horizon_summary[horizon_summary["horizon"] == 48].iloc[0].to_dict()
    best_tau_by_cost = (
        cost_summary.sort_values(["one_way_cost_bps", "total_return"], ascending=[True, False])
        .groupby("one_way_cost_bps")
        .head(1)
        .reset_index(drop=True)
    )

    lines = [
        "# P0 Baseline Lock Review",
        "",
        "## Locked Baseline",
        "",
        f"- baseline: `{LOCKED_BASELINE}`",
        "- signal: prior 48h/5d AlphaGPT committee selections are frozen per walk-forward fold.",
        "- allocator: 12.5% max symbol weight, short residual confirmation, all-symbol constrained hedge projection.",
        "- default decision horizon: 48h rebalance, 5d minimum hold, tau=0 unless the tau sweep states otherwise.",
        "- purpose: use this as the clean comparison baseline for all later slow-alpha experiments.",
        "",
        "## 48h Locked Metrics",
        "",
        "```text",
        pd.DataFrame([locked_row])[[col for col in horizon_cols if col in horizon_summary.columns]].to_string(index=False),
        "```",
        "",
        "## Horizon Stress",
        "",
        "Same frozen formulas and allocator, tau=0.",
        "",
        "```text",
        horizon_summary[[col for col in horizon_cols if col in horizon_summary.columns]].to_string(index=False),
        "```",
        "",
        "## Delay/Tau Sweep",
        "",
        "Same 48h locked baseline with exponential target-weight delay.",
        "",
        "```text",
        tau_summary[[col for col in tau_cols if col in tau_summary.columns]].to_string(index=False),
        "```",
        "",
        "## Cost Sensitivity",
        "",
        "Net return by one-way cost bps and tau.",
        "",
        "```text",
        cost_pivot.to_string(index=False),
        "```",
        "",
        "Best tau by cost level:",
        "",
        "```text",
        best_tau_by_cost[["one_way_cost_bps", "experiment", "tau_days", "total_return", "max_drawdown", "cost_drag_sum"]].to_string(
            index=False
        ),
        "```",
        "",
        "## Regime Review",
        "",
        "Regime buckets are diagnostics only; strategy decisions do not use them.",
        "",
        "```text",
        regime_summary.to_string(index=False),
        "```",
        "",
        "## Long/Short Contribution",
        "",
        "```text",
        side_summary.to_string(index=False),
        "```",
        "",
        "## Asset Contribution",
        "",
        "```text",
        asset_summary.to_string(index=False),
        "```",
        "",
        "## Selected Formula Counts",
        "",
        "```text",
        formula_counts.to_string(index=False),
        "```",
        "",
        "## Files",
        "",
        "- `BASELINE_LOCK.json`: frozen baseline metadata.",
        "- `horizon_summary.csv`: horizon stress table.",
        "- `tau_summary.csv`: delay/tau table.",
        "- `cost_sensitivity_by_tau.csv`: tau x cost table.",
        "- `regime_summary.csv`: regime diagnostics.",
        "- `side_contribution.csv`: long/short contribution.",
        "- `asset_contribution.csv`: symbol contribution.",
        "- `locked_baseline_net_gross_curve.png`: locked net/gross curve.",
        "- `horizon_summary.png`, `tau_summary.png`, `cost_sensitivity_by_tau.png`, `regime_net_return.png`, `asset_gross_contribution.png`: plots.",
        "",
        "## Raw Lock Metadata",
        "",
        "```json",
        json.dumps(to_jsonable(lock_meta), indent=2, ensure_ascii=False),
        "```",
        "",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(to_jsonable(data), fh, indent=2, ensure_ascii=False)


def format_number(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return str(value).replace(".", "p")


if __name__ == "__main__":
    main()
