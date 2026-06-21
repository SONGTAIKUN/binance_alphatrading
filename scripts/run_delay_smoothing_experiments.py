#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
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
from crypto_stat_arb.portfolio import (
    PortfolioConfig,
    portfolio_exposures,
    select_target_sides,
    sides_to_weights,
    spearman_ic,
    summarize_equity_curve,
    update_state,
)
from crypto_stat_arb.walk_forward import WalkForwardConfig
from run_residual_alphagpt_search import (
    aggregate_oos_metrics,
    build_candidate_scores,
    committee_score,
    prepare_alpha_panel,
)
from run_risk_control_experiments import (
    RiskControlConfig,
    add_gross_metrics,
    add_risk_event_metrics,
    apply_short_residual_confirmation,
    build_weights,
)
from run_turnover_experiments import to_jsonable


BASE_PANEL = PROJECT_ROOT / "outputs" / "baseline_residual_24h" / "asset_residual_panel.csv"
SELECTIONS = PROJECT_ROOT / "outputs" / "residual_alphagpt_search" / "h048_minhold5d" / "committee_top3" / "formula_selections.csv"
SCORE_COL = "_score_committee_delay"


@dataclass(frozen=True)
class DelayConfig:
    name: str
    tau_days: float


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "baseline.yaml"))
    parser.add_argument("--panel", default=str(BASE_PANEL))
    parser.add_argument("--selections", default=str(SELECTIONS))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "delay_smoothing_experiments_48h5d"))
    parser.add_argument("--tau-days", type=float, nargs="+", default=[0, 1, 2, 3, 5, 7, 10, 14, 21, 30])
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    cfg = load_config(args.config)

    print("[1/4] Loading panel and selected committees", flush=True)
    panel = pd.read_csv(args.panel, parse_dates=["timestamp"])
    panel = prepare_alpha_panel(panel, horizons=[48], execution_delay_bars=int(cfg["labels"]["execution_delay_bars"]))
    build_candidate_scores(panel)
    selections = pd.read_csv(args.selections, parse_dates=["train_start", "train_end", "test_start", "test_end"])

    pf_cfg = PortfolioConfig(
        rebalance_hours=48,
        long_k=int(cfg["portfolio"]["long_k"]),
        short_k=int(cfg["portfolio"]["short_k"]),
        gross_exposure=float(cfg["portfolio"]["gross_exposure"]),
        one_way_fee_bps=float(cfg["portfolio"]["one_way_fee_bps"]),
        one_way_slippage_bps=float(cfg["portfolio"]["one_way_slippage_bps"]),
        neutralize_exposures=list(cfg["portfolio"]["neutralize_exposures"]),
        primary_horizon=48,
        execution_delay_bars=int(cfg["labels"]["execution_delay_bars"]),
        use_hysteresis=True,
        min_hold_rebalances=math.ceil(5 * 24 / 48),
    )
    wf_cfg = WalkForwardConfig(
        train_days=int(cfg["walk_forward"]["train_days"]),
        test_days=int(cfg["walk_forward"]["test_days"]),
        step_days=int(cfg["walk_forward"]["step_days"]),
        mode=str(cfg["walk_forward"].get("mode", "rolling")),
    )
    risk_cfg = RiskControlConfig(
        name="cap12p5_shortconfirm_hedged_clip_gross",
        description="12.5% cap + short confirmation + all-symbol constrained hedge projection.",
        max_symbol_weight=0.125,
        short_resid_confirm_rank=0.35,
        allocation_method="hedged_constrained_projection",
        optimizer_target_gross_mode="clip_gross",
        hedge_penalty=25.0,
    )

    delays = [DelayConfig(name=f"tau_{format_tau(value)}", tau_days=float(value)) for value in args.tau_days]

    print("[2/4] Running delay smoothing sweep", flush=True)
    summary_rows: list[dict[str, Any]] = []
    for delay in delays:
        print(f"      {delay.name}", flush=True)
        result = run_delay_replay(panel, selections, pf_cfg, wf_cfg, risk_cfg, delay)
        exp_dir = ensure_dir(output_dir / delay.name)
        write_result(exp_dir, result, delay)
        summary_rows.append({"experiment": delay.name, "tau_days": delay.tau_days, **result["metrics"]})

    print("[3/4] Writing summary and plots", flush=True)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "summary.csv", index=False)
    summary.sort_values("total_return", ascending=False).to_csv(output_dir / "summary_by_net.csv", index=False)
    summary.sort_values("mean_turnover").to_csv(output_dir / "summary_by_turnover.csv", index=False)
    write_cost_sensitivity(output_dir, delays)
    write_review(output_dir, summary)
    plot_sweep(output_dir, summary)
    plot_curves(output_dir, delays)

    print("[4/4] Done", flush=True)
    print(f"Done: {output_dir}", flush=True)


def run_delay_replay(
    panel: pd.DataFrame,
    selections: pd.DataFrame,
    pf_cfg: PortfolioConfig,
    wf_cfg: WalkForwardConfig,
    risk_cfg: RiskControlConfig,
    delay_cfg: DelayConfig,
) -> dict[str, Any]:
    fold_rows: list[dict[str, Any]] = []
    equity_parts: list[pd.DataFrame] = []
    position_parts: list[pd.DataFrame] = []

    for _, selection in selections.iterrows():
        selected_cols = [part.strip() for part in str(selection["selected_score_cols"]).split("|")]
        selected_formulas = str(selection["selected_formulas"])
        panel[SCORE_COL] = committee_score(panel, selected_cols)
        equity, positions, metrics = run_delay_backtest(
            panel=panel,
            start=selection["test_start"],
            end=selection["test_end"],
            cfg=pf_cfg,
            risk_cfg=risk_cfg,
            delay_cfg=delay_cfg,
            score_col=SCORE_COL,
            selected_score_cols=selected_cols,
            selected_formulas=selected_formulas,
        )

        fold_id = int(selection["fold"])
        if not equity.empty:
            equity = equity.copy()
            equity["fold"] = fold_id
            equity["selected_formulas"] = selected_formulas
            positions = positions.copy()
            positions["fold"] = fold_id
            positions["selected_formulas"] = selected_formulas
            equity_parts.append(equity)
            position_parts.append(positions)
        fold_rows.append({"fold": fold_id, **selection.to_dict(), **metrics})

    fold_summary = pd.DataFrame(fold_rows)
    equity_curve = pd.concat(equity_parts, ignore_index=True) if equity_parts else pd.DataFrame()
    positions = pd.concat(position_parts, ignore_index=True) if position_parts else pd.DataFrame()
    metrics = aggregate_oos_metrics(equity_curve, pf_cfg.rebalance_hours)
    metrics["fold_count"] = int(len(selections))
    if not fold_summary.empty and "total_return" in fold_summary:
        metrics["positive_fold_rate"] = float((fold_summary["total_return"] > 0).mean())
    add_gross_metrics(equity_curve, metrics)
    add_risk_event_metrics(positions, metrics)
    add_delay_metrics(equity_curve, metrics)
    return {"folds": fold_summary, "equity": equity_curve, "positions": positions, "metrics": metrics}


def run_delay_backtest(
    panel: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cfg: PortfolioConfig,
    risk_cfg: RiskControlConfig,
    delay_cfg: DelayConfig,
    score_col: str,
    selected_score_cols: list[str],
    selected_formulas: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    label_col = f"label_resid_ret_h{cfg.primary_horizon}"
    ret_col = f"fwd_simple_ret_h{cfg.primary_horizon}"
    symbols = sorted(panel["symbol"].unique())
    mask = (panel["timestamp"] >= start) & (panel["timestamp"] < end)
    timestamps = np.array(sorted(panel.loc[mask, "timestamp"].unique()))
    timestamps = timestamps[:: cfg.rebalance_hours]

    prev_weights = pd.Series(0.0, index=symbols)
    current_state: dict[str, tuple[int, int]] = {}
    equity = 1.0
    rows_out: list[dict[str, Any]] = []
    pos_out: list[dict[str, Any]] = []
    retention = delay_retention(delay_cfg.tau_days, cfg.rebalance_hours)
    new_target_weight = 1.0 - retention

    for ts in timestamps:
        rows = panel[panel["timestamp"] == ts].copy()
        rows = rows[np.isfinite(rows[score_col]) & np.isfinite(rows[ret_col]) & np.isfinite(rows[label_col])]
        if len(rows) < cfg.long_k + cfg.short_k:
            continue

        rows["score_rank"] = rows[score_col].rank(pct=True, method="average")
        rows["ema_rank"] = rows["score_ema_025"].rank(pct=True, method="average")
        rows["base_rank"] = rows["score_baseline"].rank(pct=True, method="average")
        rows = rows.sort_values(score_col, ascending=False)
        rows_by_symbol = rows.set_index("symbol")

        target_sides = select_target_sides(rows, current_state, cfg, score_col)
        target_sides = apply_short_residual_confirmation(target_sides, rows_by_symbol, risk_cfg)
        if not any(side > 0 for side in target_sides.values()) or not any(side < 0 for side in target_sides.values()):
            continue

        state_sides = target_sides.copy()
        selected = rows_by_symbol.reindex(list(target_sides)).reset_index()
        raw_weights = sides_to_weights(target_sides, cfg.gross_exposure)
        target_weights, optimizer_info = build_weights(raw_weights, selected, rows_by_symbol, cfg, risk_cfg)
        target_weights = target_weights[target_weights.abs() > 1e-12]
        if target_weights.abs().sum() <= 0:
            continue

        target_full_weights = pd.Series(0.0, index=symbols)
        target_full_weights.loc[target_weights.index] = target_weights
        full_weights = smooth_target_weights(prev_weights, target_full_weights, retention)
        full_weights = full_weights[full_weights.abs() > 1e-12].reindex(symbols).fillna(0.0)
        target_turnover = float((target_full_weights - prev_weights).abs().sum())
        turnover = float((full_weights - prev_weights).abs().sum())
        cost = turnover * (cfg.one_way_fee_bps + cfg.one_way_slippage_bps) / 10_000.0

        active_weights = full_weights[full_weights.abs() > 1e-12]
        active_rows = rows_by_symbol.reindex(active_weights.index)
        gross_period_ret = float((active_weights * active_rows[ret_col]).sum())
        period_ret = gross_period_ret - cost
        equity *= 1.0 + period_ret

        rank_ic = spearman_ic(rows[score_col], rows[label_col])
        long_labels = active_rows.loc[active_weights[active_weights > 0].index, label_col]
        short_labels = active_rows.loc[active_weights[active_weights < 0].index, label_col]
        top_bottom = float(long_labels.mean() - short_labels.mean()) if len(long_labels) and len(short_labels) else np.nan
        exposures = portfolio_exposures(active_weights, active_rows, cfg.neutralize_exposures)

        rows_out.append(
            {
                "timestamp": pd.Timestamp(ts),
                "exit_timestamp": pd.Timestamp(ts) + pd.Timedelta(hours=cfg.execution_delay_bars + cfg.primary_horizon),
                "period_return": period_ret,
                "gross_period_return": gross_period_ret,
                "cost_return": cost,
                "equity": equity,
                "turnover": turnover,
                "target_turnover": target_turnover,
                "turnover_saved": target_turnover - turnover,
                "gross_exposure": float(active_weights.abs().sum()),
                "target_gross_exposure": float(target_full_weights.abs().sum()),
                "rank_ic": rank_ic,
                "top_bottom_label_spread": top_bottom,
                "abs_beta_btc": abs(exposures.get("beta_btc", 0.0)),
                "abs_beta_eth": abs(exposures.get("beta_eth", 0.0)),
                "abs_beta_market": abs(exposures.get("beta_market", 0.0)),
                "symbol_concentration": float((active_weights**2).sum()),
                "max_abs_weight": float(active_weights.abs().max()),
                "delay_tau_days": delay_cfg.tau_days,
                "delay_retention": retention,
                "delay_new_target_weight": new_target_weight,
                **optimizer_info,
                "target_applied": 1.0,
                "selected_formulas": selected_formulas,
            }
        )

        for symbol, weight in active_weights.items():
            pos_out.append(
                {
                    "timestamp": pd.Timestamp(ts),
                    "exit_timestamp": pd.Timestamp(ts)
                    + pd.Timedelta(hours=cfg.execution_delay_bars + cfg.primary_horizon),
                    "symbol": symbol,
                    "weight": float(weight),
                    "target_weight": float(target_full_weights.loc[symbol]),
                    "score": float(rows_by_symbol.loc[symbol, score_col]),
                    "ema_rank": float(rows_by_symbol.loc[symbol, "ema_rank"]),
                    "base_rank": float(rows_by_symbol.loc[symbol, "base_rank"]),
                    "fwd_simple_ret": float(rows_by_symbol.loc[symbol, ret_col]),
                    "label_resid_ret": float(rows_by_symbol.loc[symbol, label_col]),
                    "contribution": float(weight * rows_by_symbol.loc[symbol, ret_col]),
                    "selected_score_cols": " | ".join(selected_score_cols),
                }
            )

        prev_weights = full_weights
        current_state = update_state(current_state, state_sides, True)

    equity_curve = pd.DataFrame(rows_out)
    positions = pd.DataFrame(pos_out)
    metrics = summarize_equity_curve(equity_curve, cfg.rebalance_hours)
    return equity_curve, positions, metrics


def delay_retention(tau_days: float, rebalance_hours: int) -> float:
    if tau_days <= 0:
        return 0.0
    return float(np.exp(-rebalance_hours / (tau_days * 24.0)))


def smooth_target_weights(prev_weights: pd.Series, target_weights: pd.Series, retention: float) -> pd.Series:
    if retention <= 0:
        return target_weights.copy()
    return retention * prev_weights + (1.0 - retention) * target_weights


def add_delay_metrics(equity_curve: pd.DataFrame, metrics: dict[str, float]) -> None:
    if equity_curve.empty:
        return
    for col in ["target_turnover", "turnover_saved", "target_gross_exposure"]:
        if col in equity_curve:
            metrics[f"mean_{col}"] = float(equity_curve[col].mean())
            metrics[f"total_{col}"] = float(equity_curve[col].sum())
    if "turnover" in equity_curve and "target_turnover" in equity_curve:
        target_total = float(equity_curve["target_turnover"].sum())
        actual_total = float(equity_curve["turnover"].sum())
        metrics["turnover_reduction"] = 1.0 - actual_total / target_total if target_total > 0 else np.nan
    if "gross_exposure" in equity_curve and "target_gross_exposure" in equity_curve:
        target_gross = float(equity_curve["target_gross_exposure"].mean())
        actual_gross = float(equity_curve["gross_exposure"].mean())
        metrics["gross_exposure_retention"] = actual_gross / target_gross if target_gross > 0 else np.nan


def write_result(output_dir: Path, result: dict[str, Any], delay_cfg: DelayConfig) -> None:
    result["folds"].to_csv(output_dir / "walk_forward_folds.csv", index=False)
    result["equity"].to_csv(output_dir / "equity_curve.csv", index=False)
    result["positions"].to_csv(output_dir / "positions.csv", index=False)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(to_jsonable(result["metrics"]), fh, indent=2, ensure_ascii=False)
    (output_dir / "README.md").write_text(
        f"# {delay_cfg.name}\n\nExponential target-weight smoothing tau_days={delay_cfg.tau_days}.\n",
        encoding="utf-8",
    )


def write_review(output_dir: Path, summary: pd.DataFrame) -> None:
    display_cols = [
        "experiment",
        "tau_days",
        "total_return",
        "gross_total_return_before_cost",
        "gross_max_drawdown",
        "sharpe",
        "max_drawdown",
        "mean_turnover",
        "turnover_reduction",
        "cost_drag",
        "mean_gross_exposure",
        "gross_exposure_retention",
        "positive_fold_rate",
    ]
    for col in display_cols:
        if col not in summary.columns:
            summary[col] = np.nan
    lines = [
        "# Delay Smoothing Sweep",
        "",
        "```text",
        summary[display_cols].to_string(index=False),
        "```",
        "",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def write_cost_sensitivity(output_dir: Path, delays: list[DelayConfig]) -> None:
    rows: list[dict[str, float | str]] = []
    for delay in delays:
        path = output_dir / delay.name / "equity_curve.csv"
        if not path.exists():
            continue
        equity = pd.read_csv(path, parse_dates=["timestamp"]).sort_values("timestamp")
        if equity.empty:
            continue
        gross_returns = equity["gross_period_return"].astype(float)
        turnover = equity["turnover"].astype(float)
        for bps in [0, 4, 6, 8, 10, 12, 15, 20]:
            scenario_returns = gross_returns - turnover * bps / 10_000.0
            curve = (1.0 + scenario_returns).cumprod()
            drawdown = curve / curve.cummax() - 1.0
            rows.append(
                {
                    "experiment": delay.name,
                    "tau_days": delay.tau_days,
                    "one_way_cost_bps": bps,
                    "total_return": float(curve.iloc[-1] - 1.0),
                    "max_drawdown": float(drawdown.min()),
                    "cost_drag_sum": float((turnover * bps / 10_000.0).sum()),
                }
            )
    pd.DataFrame(rows).to_csv(output_dir / "delay_cost_bps_sensitivity.csv", index=False)


def plot_sweep(output_dir: Path, summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    ordered = summary.sort_values("tau_days")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    plot_metric(axes[0, 0], ordered, "total_return", "Net return")
    plot_metric(axes[0, 1], ordered, "gross_total_return_before_cost", "Gross before cost")
    plot_metric(axes[1, 0], ordered, "mean_turnover", "Mean turnover / rebalance")
    plot_metric(axes[1, 1], ordered, "cost_drag", "Cost drag")
    axes[1, 0].set_xlabel("tau days")
    axes[1, 1].set_xlabel("tau days")
    fig.tight_layout()
    fig.savefig(output_dir / "delay_sweep_metrics.png", dpi=180)
    plt.close(fig)


def plot_metric(ax: Any, frame: pd.DataFrame, col: str, title: str) -> None:
    ax.plot(frame["tau_days"], frame[col], marker="o", linewidth=2)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    if col != "mean_turnover":
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))


def plot_curves(output_dir: Path, delays: list[DelayConfig]) -> None:
    selected = {"tau_0", "tau_2", "tau_5", "tau_10", "tau_21"}
    plt.figure(figsize=(12, 6.5))
    for delay in delays:
        if delay.name not in selected:
            continue
        path = output_dir / delay.name / "equity_curve.csv"
        if not path.exists():
            continue
        equity = pd.read_csv(path, parse_dates=["timestamp"]).sort_values("timestamp")
        if equity.empty:
            continue
        curve = (1.0 + equity["period_return"]).cumprod() - 1.0
        plt.plot(equity["timestamp"], curve, label=delay.name, linewidth=2 if delay.name in {"tau_0", "tau_5"} else 1.4)
    plt.axhline(0.0, color="black", linewidth=0.9, alpha=0.6)
    plt.title("Delay smoothing net curves")
    plt.ylabel("Cumulative return")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "delay_net_curves.png", dpi=180)
    plt.close()


def format_tau(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return str(value).replace(".", "p")


if __name__ == "__main__":
    main()
