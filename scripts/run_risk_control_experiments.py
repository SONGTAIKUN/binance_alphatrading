#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from crypto_stat_arb.config import ensure_dir, load_config
from crypto_stat_arb.features import add_forward_labels
from crypto_stat_arb.portfolio import (
    PortfolioConfig,
    constrained_neutralize_weights,
    constrained_project_weights,
    neutralize_weights,
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
from run_turnover_experiments import to_jsonable


BASE_PANEL = PROJECT_ROOT / "outputs" / "baseline_residual_24h" / "asset_residual_panel.csv"
SELECTIONS = PROJECT_ROOT / "outputs" / "residual_alphagpt_search" / "h048_minhold5d" / "committee_top3" / "formula_selections.csv"
OLD_METRICS = PROJECT_ROOT / "outputs" / "residual_alphagpt_search" / "h048_minhold5d" / "committee_top3" / "metrics.json"
SCORE_COL = "_score_committee_risk"


@dataclass(frozen=True)
class RiskControlConfig:
    name: str
    description: str
    max_symbol_weight: float | None = None
    short_resid_confirm_rank: float | None = None
    disagreement_scale: float | None = None
    snapback_threshold: float | None = None
    snapback_gross_scale: float = 1.0
    emergency_stop_pct: float | None = None
    allocation_method: str = "neutralize_clip"
    optimizer_target_gross_mode: str = "capacity"
    hedge_penalty: float = 25.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "baseline.yaml"))
    parser.add_argument("--panel", default=str(BASE_PANEL))
    parser.add_argument("--selections", default=str(SELECTIONS))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "risk_control_experiments_48h5d_committee"))
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    cfg = load_config(args.config)

    print("[1/4] Loading panel and selected committees", flush=True)
    panel = pd.read_csv(args.panel, parse_dates=["timestamp"])
    panel = prepare_alpha_panel(panel, horizons=[48], execution_delay_bars=int(cfg["labels"]["execution_delay_bars"]))
    build_candidate_scores(panel)
    add_market_regime_features(panel)
    selections = pd.read_csv(args.selections, parse_dates=["train_start", "train_end", "test_start", "test_end"])
    log_close_wide = panel.pivot(index="timestamp", columns="symbol", values="log_close").sort_index()

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

    experiments = [
        RiskControlConfig("baseline_replay", "Replay current 48h/5d committee without extra risk controls."),
        RiskControlConfig("cap_12p5", "Post-neutralization cap: abs(weight) <= 12.5%.", max_symbol_weight=0.125),
        RiskControlConfig("cap_10", "Post-neutralization cap: abs(weight) <= 10%.", max_symbol_weight=0.10),
        RiskControlConfig(
            "short_resid_confirm_35",
            "Block shorts unless EMA/residual score rank is in bottom 35%.",
            short_resid_confirm_rank=0.35,
        ),
        RiskControlConfig(
            "disagreement_half",
            "Halve positions where committee side conflicts with EMA/residual side.",
            disagreement_scale=0.50,
        ),
        RiskControlConfig(
            "snapback_half",
            "Halve gross exposure when trailing 7d market return is below -5%.",
            snapback_threshold=-0.05,
            snapback_gross_scale=0.50,
        ),
        RiskControlConfig(
            "emergency_stop_20",
            "Hourly close emergency stop: close a symbol if adverse move reaches 20% inside the 48h hold.",
            emergency_stop_pct=0.20,
        ),
        RiskControlConfig(
            "cap12p5_shortconfirm",
            "Combine 12.5% symbol cap and short residual confirmation.",
            max_symbol_weight=0.125,
            short_resid_confirm_rank=0.35,
        ),
        RiskControlConfig(
            "cap12p5_shortconfirm_opt_same_gross",
            "12.5% cap + short confirmation with constrained optimizer targeting the clipped gross exposure.",
            max_symbol_weight=0.125,
            short_resid_confirm_rank=0.35,
            allocation_method="constrained_optimizer",
            optimizer_target_gross_mode="clip_gross",
        ),
        RiskControlConfig(
            "cap12p5_shortconfirm_opt_capacity",
            "12.5% cap + short confirmation with constrained optimizer using the highest feasible capped gross target.",
            max_symbol_weight=0.125,
            short_resid_confirm_rank=0.35,
            allocation_method="constrained_optimizer",
            optimizer_target_gross_mode="capacity",
        ),
        RiskControlConfig(
            "cap12p5_shortconfirm_hedged_clip_gross",
            "12.5% cap + short confirmation with all-symbol constrained projection and gross limited to clipped baseline gross.",
            max_symbol_weight=0.125,
            short_resid_confirm_rank=0.35,
            allocation_method="hedged_constrained_projection",
            optimizer_target_gross_mode="clip_gross",
            hedge_penalty=25.0,
        ),
        RiskControlConfig(
            "cap12p5_shortconfirm_hedged_full_gross",
            "12.5% cap + short confirmation with all-symbol constrained projection and gross limited to 1.0.",
            max_symbol_weight=0.125,
            short_resid_confirm_rank=0.35,
            allocation_method="hedged_constrained_projection",
            optimizer_target_gross_mode="full_gross",
            hedge_penalty=25.0,
        ),
        RiskControlConfig(
            "cap10_shortconfirm_opt_capacity",
            "10% cap + short confirmation with constrained optimizer using the highest feasible capped gross target.",
            max_symbol_weight=0.10,
            short_resid_confirm_rank=0.35,
            allocation_method="constrained_optimizer",
            optimizer_target_gross_mode="capacity",
        ),
        RiskControlConfig(
            "cap12p5_disagreement",
            "Combine 12.5% symbol cap and disagreement half-size.",
            max_symbol_weight=0.125,
            disagreement_scale=0.50,
        ),
        RiskControlConfig(
            "combined_core",
            "12.5% cap + short confirmation + disagreement half-size.",
            max_symbol_weight=0.125,
            short_resid_confirm_rank=0.35,
            disagreement_scale=0.50,
        ),
        RiskControlConfig(
            "combined_all",
            "12.5% cap + short confirmation + disagreement + snapback gross half + emergency stop.",
            max_symbol_weight=0.125,
            short_resid_confirm_rank=0.35,
            disagreement_scale=0.50,
            snapback_threshold=-0.05,
            snapback_gross_scale=0.50,
            emergency_stop_pct=0.20,
        ),
    ]

    print("[2/4] Running risk-control experiments", flush=True)
    summary_rows: list[dict[str, Any]] = []
    for experiment in experiments:
        print(f"      {experiment.name}", flush=True)
        result = run_replay(panel, selections, pf_cfg, wf_cfg, experiment, log_close_wide)
        exp_dir = ensure_dir(output_dir / experiment.name)
        write_result(exp_dir, result, experiment)
        summary_rows.append({"experiment": experiment.name, "description": experiment.description, **result["metrics"]})

    print("[3/4] Writing summary and diagnostics", flush=True)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "summary.csv", index=False)
    compare_to_existing(summary, output_dir)
    write_review(output_dir, summary)

    print("[4/4] Done", flush=True)
    print(f"Done: {output_dir}", flush=True)


def add_market_regime_features(panel: pd.DataFrame) -> None:
    market = panel.groupby("timestamp")["log_ret"].mean().sort_index()
    trailing = market.rolling(168, min_periods=48).sum()
    panel["market_trailing_ret_7d"] = panel["timestamp"].map(trailing)


def run_replay(
    panel: pd.DataFrame,
    selections: pd.DataFrame,
    pf_cfg: PortfolioConfig,
    wf_cfg: WalkForwardConfig,
    risk_cfg: RiskControlConfig,
    log_close_wide: pd.DataFrame,
) -> dict[str, Any]:
    fold_rows: list[dict[str, Any]] = []
    equity_parts: list[pd.DataFrame] = []
    position_parts: list[pd.DataFrame] = []

    for _, selection in selections.iterrows():
        selected_cols = [part.strip() for part in str(selection["selected_score_cols"]).split("|")]
        selected_formulas = str(selection["selected_formulas"])
        panel[SCORE_COL] = committee_score(panel, selected_cols)

        equity, positions, metrics = run_risk_backtest(
            panel=panel,
            start=selection["test_start"],
            end=selection["test_end"],
            cfg=pf_cfg,
            risk_cfg=risk_cfg,
            score_col=SCORE_COL,
            selected_score_cols=selected_cols,
            selected_formulas=selected_formulas,
            log_close_wide=log_close_wide,
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
    metrics = aggregate_oos_metrics(equity_curve, wf_cfg.step_days * 24 if False else pf_cfg.rebalance_hours)
    metrics["fold_count"] = int(len(selections))
    if not fold_summary.empty:
        metrics["positive_fold_rate"] = float((fold_summary["total_return"] > 0).mean())
    add_gross_metrics(equity_curve, metrics)
    add_risk_event_metrics(positions, metrics)
    return {"folds": fold_summary, "equity": equity_curve, "positions": positions, "metrics": metrics}


def run_risk_backtest(
    panel: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cfg: PortfolioConfig,
    risk_cfg: RiskControlConfig,
    score_col: str,
    selected_score_cols: list[str],
    selected_formulas: str,
    log_close_wide: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    label_col = f"label_resid_ret_h{cfg.primary_horizon}"
    ret_col = f"fwd_simple_ret_h{cfg.primary_horizon}"
    symbols = sorted(panel["symbol"].unique())
    timestamps = np.array(sorted(panel.loc[(panel["timestamp"] >= start) & (panel["timestamp"] < end), "timestamp"].unique()))
    timestamps = timestamps[:: cfg.rebalance_hours]

    prev_weights = pd.Series(0.0, index=symbols)
    current_state: dict[str, tuple[int, int]] = {}
    equity = 1.0
    rows_out: list[dict[str, Any]] = []
    pos_out: list[dict[str, Any]] = []

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
        weights, optimizer_info = build_weights(raw_weights, selected, rows_by_symbol, cfg, risk_cfg)
        weights = weights[weights.abs() > 1e-12]
        if weights.abs().sum() <= 0:
            if risk_cfg.allocation_method == "constrained_optimizer":
                target_full_weights = pd.Series(0.0, index=symbols)
                turnover = float((target_full_weights - prev_weights).abs().sum())
                cost = turnover * (cfg.one_way_fee_bps + cfg.one_way_slippage_bps) / 10_000.0
                equity *= 1.0 - cost
                rank_ic = spearman_ic(rows[score_col], rows[label_col])
                rows_out.append(
                    {
                        "timestamp": pd.Timestamp(ts),
                        "exit_timestamp": pd.Timestamp(ts)
                        + pd.Timedelta(hours=cfg.execution_delay_bars + cfg.primary_horizon),
                        "period_return": -cost,
                        "gross_period_return": 0.0,
                        "cost_return": cost,
                        "base_rebalance_cost_return": cost,
                        "emergency_extra_cost_return": 0.0,
                        "equity": equity,
                        "turnover": turnover,
                        "gross_exposure": 0.0,
                        "rank_ic": rank_ic,
                        "top_bottom_label_spread": np.nan,
                        "abs_beta_btc": 0.0,
                        "abs_beta_eth": 0.0,
                        "abs_beta_market": 0.0,
                        "symbol_concentration": 0.0,
                        "max_abs_weight": 0.0,
                        "stopped_weight": 0.0,
                        "stopped_count": 0,
                        **optimizer_info,
                        "target_applied": 1.0,
                        "selected_formulas": selected_formulas,
                    }
                )
                prev_weights = target_full_weights
                current_state = update_state(current_state, {}, True)
            continue

        target_full_weights = pd.Series(0.0, index=symbols)
        target_full_weights.loc[weights.index] = weights
        turnover = float((target_full_weights - prev_weights).abs().sum())
        cost = turnover * (cfg.one_way_fee_bps + cfg.one_way_slippage_bps) / 10_000.0

        active_rows = rows_by_symbol.reindex(weights.index)
        fwd_returns, stopped = adjusted_forward_returns(
            weights=weights,
            active_rows=active_rows,
            timestamp=pd.Timestamp(ts),
            cfg=cfg,
            risk_cfg=risk_cfg,
            log_close_wide=log_close_wide,
            ret_col=ret_col,
        )
        emergency_extra_cost = stopped.abs().sum() * (cfg.one_way_fee_bps + cfg.one_way_slippage_bps) / 10_000.0
        gross_period_ret = float((weights * fwd_returns).sum())
        period_ret = gross_period_ret - cost - emergency_extra_cost
        equity *= 1.0 + period_ret

        rank_ic = spearman_ic(rows[score_col], rows[label_col])
        long_labels = active_rows.loc[weights[weights > 0].index, label_col]
        short_labels = active_rows.loc[weights[weights < 0].index, label_col]
        top_bottom = float(long_labels.mean() - short_labels.mean())
        exposures = portfolio_exposures(weights, active_rows, cfg.neutralize_exposures)

        rows_out.append(
            {
                "timestamp": pd.Timestamp(ts),
                "exit_timestamp": pd.Timestamp(ts) + pd.Timedelta(hours=cfg.execution_delay_bars + cfg.primary_horizon),
                "period_return": period_ret,
                "gross_period_return": gross_period_ret,
                "cost_return": cost + emergency_extra_cost,
                "base_rebalance_cost_return": cost,
                "emergency_extra_cost_return": emergency_extra_cost,
                "equity": equity,
                "turnover": turnover,
                "gross_exposure": float(weights.abs().sum()),
                "rank_ic": rank_ic,
                "top_bottom_label_spread": top_bottom,
                "abs_beta_btc": abs(exposures.get("beta_btc", 0.0)),
                "abs_beta_eth": abs(exposures.get("beta_eth", 0.0)),
                "abs_beta_market": abs(exposures.get("beta_market", 0.0)),
                "symbol_concentration": float((weights**2).sum()),
                "max_abs_weight": float(weights.abs().max()),
                "stopped_weight": float(stopped.abs().sum()),
                "stopped_count": int((stopped.abs() > 0).sum()),
                **optimizer_info,
                "target_applied": 1.0,
                "selected_formulas": selected_formulas,
            }
        )

        for symbol, weight in weights.items():
            pos_out.append(
                {
                    "timestamp": pd.Timestamp(ts),
                    "exit_timestamp": pd.Timestamp(ts) + pd.Timedelta(hours=cfg.execution_delay_bars + cfg.primary_horizon),
                    "symbol": symbol,
                    "weight": float(weight),
                    "score": float(rows_by_symbol.loc[symbol, score_col]),
                    "ema_rank": float(rows_by_symbol.loc[symbol, "ema_rank"]),
                    "base_rank": float(rows_by_symbol.loc[symbol, "base_rank"]),
                    "fwd_simple_ret": float(fwd_returns.loc[symbol]),
                    "raw_fwd_simple_ret": float(active_rows.loc[symbol, ret_col]),
                    "label_resid_ret": float(active_rows.loc[symbol, label_col]),
                    "contribution": float(weight * fwd_returns.loc[symbol]),
                    "stopped_weight": float(stopped.loc[symbol]),
                    **optimizer_info,
                    "selected_score_cols": " | ".join(selected_score_cols),
                }
            )

        prev_weights = target_full_weights
        current_state = update_state(current_state, state_sides, True)

    equity_curve = pd.DataFrame(rows_out)
    positions = pd.DataFrame(pos_out)
    metrics = summarize_equity_curve(equity_curve, cfg.rebalance_hours)
    return equity_curve, positions, metrics


def build_weights(
    raw_weights: pd.Series,
    selected: pd.DataFrame,
    rows_by_symbol: pd.DataFrame,
    cfg: PortfolioConfig,
    risk_cfg: RiskControlConfig,
) -> tuple[pd.Series, dict[str, float]]:
    if risk_cfg.allocation_method == "constrained_optimizer":
        target_gross = None
        if risk_cfg.optimizer_target_gross_mode == "clip_gross":
            clipped = neutralize_weights(raw_weights, selected, cfg.neutralize_exposures, cfg.gross_exposure)
            clipped = apply_disagreement_cap(clipped, rows_by_symbol, risk_cfg)
            clipped = apply_snapback_filter(clipped, rows_by_symbol, risk_cfg)
            clipped = apply_symbol_cap(clipped, risk_cfg)
            target_gross = float(clipped.abs().sum())
        weights, info = constrained_neutralize_weights(
            raw_weights=raw_weights,
            selected=selected,
            exposure_cols=cfg.neutralize_exposures,
            gross_exposure=cfg.gross_exposure,
            max_abs_weight=float(risk_cfg.max_symbol_weight or cfg.gross_exposure),
            target_gross=target_gross,
            return_info=True,
        )
        return weights, info

    if risk_cfg.allocation_method == "hedged_constrained_projection":
        clipped = neutralize_weights(raw_weights, selected, cfg.neutralize_exposures, cfg.gross_exposure)
        clipped = apply_disagreement_cap(clipped, rows_by_symbol, risk_cfg)
        clipped = apply_snapback_filter(clipped, rows_by_symbol, risk_cfg)
        clipped = apply_symbol_cap(clipped, risk_cfg)
        gross_limit = cfg.gross_exposure
        if risk_cfg.optimizer_target_gross_mode == "clip_gross":
            gross_limit = float(clipped.abs().sum())
        fixed_signs = {symbol: int(np.sign(weight)) for symbol, weight in raw_weights.items() if abs(weight) > 0}
        target = pd.Series(0.0, index=rows_by_symbol.index)
        target.loc[clipped.index] = clipped
        weights, info = constrained_project_weights(
            target_weights=target,
            universe=rows_by_symbol.reset_index(),
            exposure_cols=cfg.neutralize_exposures,
            max_abs_weight=float(risk_cfg.max_symbol_weight or cfg.gross_exposure),
            gross_limit=gross_limit,
            fixed_signs=fixed_signs,
            hedge_penalty=risk_cfg.hedge_penalty,
            return_info=True,
        )
        return weights, info

    weights = neutralize_weights(raw_weights, selected, cfg.neutralize_exposures, cfg.gross_exposure)
    weights = apply_disagreement_cap(weights, rows_by_symbol, risk_cfg)
    weights = apply_snapback_filter(weights, rows_by_symbol, risk_cfg)
    weights = apply_symbol_cap(weights, risk_cfg)
    info = {
        "optimizer_success": np.nan,
        "optimizer_target_gross": np.nan,
        "optimizer_actual_gross": np.nan,
        "optimizer_exposure_error": np.nan,
        "optimizer_attempts": np.nan,
    }
    return weights, info


def apply_short_residual_confirmation(
    target_sides: dict[str, int],
    rows_by_symbol: pd.DataFrame,
    risk_cfg: RiskControlConfig,
) -> dict[str, int]:
    if risk_cfg.short_resid_confirm_rank is None:
        return target_sides
    out: dict[str, int] = {}
    for symbol, side in target_sides.items():
        if side < 0 and rows_by_symbol.loc[symbol, "ema_rank"] > risk_cfg.short_resid_confirm_rank:
            continue
        out[symbol] = side
    return out


def apply_disagreement_cap(weights: pd.Series, rows_by_symbol: pd.DataFrame, risk_cfg: RiskControlConfig) -> pd.Series:
    if risk_cfg.disagreement_scale is None:
        return weights
    out = weights.copy()
    for symbol, weight in out.items():
        ema_rank = rows_by_symbol.loc[symbol, "ema_rank"]
        conflict = (weight > 0 and ema_rank < 0.5) or (weight < 0 and ema_rank > 0.5)
        if conflict:
            out.loc[symbol] = weight * risk_cfg.disagreement_scale
    return out


def apply_snapback_filter(weights: pd.Series, rows_by_symbol: pd.DataFrame, risk_cfg: RiskControlConfig) -> pd.Series:
    if risk_cfg.snapback_threshold is None:
        return weights
    trailing = rows_by_symbol["market_trailing_ret_7d"].dropna()
    if trailing.empty:
        return weights
    if float(trailing.iloc[0]) <= risk_cfg.snapback_threshold:
        return weights * risk_cfg.snapback_gross_scale
    return weights


def apply_symbol_cap(weights: pd.Series, risk_cfg: RiskControlConfig) -> pd.Series:
    if risk_cfg.max_symbol_weight is None:
        return weights
    return weights.clip(lower=-risk_cfg.max_symbol_weight, upper=risk_cfg.max_symbol_weight)


def adjusted_forward_returns(
    weights: pd.Series,
    active_rows: pd.DataFrame,
    timestamp: pd.Timestamp,
    cfg: PortfolioConfig,
    risk_cfg: RiskControlConfig,
    log_close_wide: pd.DataFrame,
    ret_col: str,
) -> tuple[pd.Series, pd.Series]:
    fwd = active_rows[ret_col].astype(float).copy()
    stopped = pd.Series(0.0, index=weights.index)
    if risk_cfg.emergency_stop_pct is None:
        return fwd, stopped

    entry_time = timestamp + pd.Timedelta(hours=cfg.execution_delay_bars)
    exit_time = timestamp + pd.Timedelta(hours=cfg.execution_delay_bars + cfg.primary_horizon)
    if entry_time not in log_close_wide.index:
        return fwd, stopped
    path = log_close_wide.loc[(log_close_wide.index >= entry_time) & (log_close_wide.index <= exit_time)]
    if path.empty:
        return fwd, stopped
    entry = log_close_wide.loc[entry_time]
    simple_path = np.exp(path[weights.index].subtract(entry[weights.index], axis=1)) - 1.0
    for symbol, weight in weights.items():
        series = simple_path[symbol].dropna()
        if series.empty:
            continue
        if weight > 0:
            breach = series[series <= -risk_cfg.emergency_stop_pct]
        else:
            breach = series[series >= risk_cfg.emergency_stop_pct]
        if not breach.empty:
            fwd.loc[symbol] = float(breach.iloc[0])
            stopped.loc[symbol] = abs(float(weight))
    return fwd, stopped


def add_gross_metrics(equity_curve: pd.DataFrame, metrics: dict[str, float]) -> None:
    if equity_curve.empty:
        return
    curve = equity_curve.sort_values("timestamp").copy()
    gross_equity = (1.0 + curve["gross_period_return"]).cumprod()
    gross_dd = gross_equity / gross_equity.cummax() - 1.0
    metrics["gross_max_drawdown"] = float(gross_dd.min())
    metrics["mean_gross_exposure"] = float(curve["gross_exposure"].mean())
    metrics["mean_max_abs_weight"] = float(curve["max_abs_weight"].mean()) if "max_abs_weight" in curve else np.nan
    metrics["max_abs_weight"] = float(curve["max_abs_weight"].max()) if "max_abs_weight" in curve else np.nan
    metrics["total_stopped_weight"] = float(curve["stopped_weight"].sum()) if "stopped_weight" in curve else 0.0
    metrics["stopped_period_rate"] = float((curve["stopped_count"] > 0).mean()) if "stopped_count" in curve else 0.0
    for col in ["abs_beta_btc", "abs_beta_eth", "abs_beta_market"]:
        if col in curve:
            metrics[f"max_{col}"] = float(curve[col].max())
    for col in ["optimizer_success", "optimizer_exposure_error", "optimizer_target_gross", "optimizer_actual_gross"]:
        if col in curve:
            metrics[f"mean_{col}"] = float(curve[col].mean())
            metrics[f"max_{col}"] = float(curve[col].max())


def add_risk_event_metrics(positions: pd.DataFrame, metrics: dict[str, float]) -> None:
    if positions.empty:
        return
    fil = positions[positions["symbol"] == "FILUSDT"].copy()
    if not fil.empty:
        metrics["fil_total_contribution"] = float(fil["contribution"].sum())
        target = fil[fil["timestamp"] == pd.Timestamp("2025-11-05 08:00:00+00:00")]
        if not target.empty:
            metrics["fil_20251105_contribution"] = float(target["contribution"].sum())
            metrics["fil_20251105_weight"] = float(target["weight"].sum())


def write_result(output_dir: Path, result: dict[str, Any], risk_cfg: RiskControlConfig) -> None:
    result["folds"].to_csv(output_dir / "walk_forward_folds.csv", index=False)
    result["equity"].to_csv(output_dir / "equity_curve.csv", index=False)
    result["positions"].to_csv(output_dir / "positions.csv", index=False)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(to_jsonable(result["metrics"]), fh, indent=2, ensure_ascii=False)
    (output_dir / "README.md").write_text(f"# {risk_cfg.name}\n\n{risk_cfg.description}\n", encoding="utf-8")


def compare_to_existing(summary: pd.DataFrame, output_dir: Path) -> None:
    if not OLD_METRICS.exists():
        return
    old = json.loads(OLD_METRICS.read_text(encoding="utf-8"))
    baseline = summary[summary["experiment"] == "baseline_replay"].iloc[0].to_dict()
    comparison = pd.DataFrame(
        [
            {"source": "previous_committee", **old},
            {"source": "baseline_replay", **{k: v for k, v in baseline.items() if k in old or k in ["experiment"]}},
        ]
    )
    comparison.to_csv(output_dir / "baseline_replay_check.csv", index=False)


def write_review(output_dir: Path, summary: pd.DataFrame) -> None:
    display_cols = [
        "experiment",
        "total_return",
        "gross_total_return_before_cost",
        "gross_max_drawdown",
        "sharpe",
        "max_drawdown",
        "mean_gross_exposure",
        "mean_turnover",
        "cost_drag",
        "max_abs_weight",
        "mean_abs_beta_market",
        "max_abs_beta_market",
        "mean_optimizer_success",
        "mean_optimizer_exposure_error",
        "fil_20251105_contribution",
        "positive_fold_rate",
    ]
    for col in display_cols:
        if col not in summary.columns:
            summary[col] = np.nan
    summary_sorted = summary.sort_values(["gross_max_drawdown", "gross_total_return_before_cost"], ascending=[False, False])
    best_gross_dd = summary_sorted.iloc[0].to_dict()
    best_return = summary.sort_values("total_return", ascending=False).iloc[0].to_dict()
    lines = [
        "# 48h/5d Committee Risk-Control Experiments",
        "",
        "## Design",
        "",
        "- Formula selections are fixed from the prior walk-forward AlphaGPT committee run.",
        "- Each experiment replays OOS folds with the same 48h horizon and 5d minimum hold.",
        "- Emergency stop uses hourly close path inside the 48h holding window.",
        "",
        "## Summary",
        "",
        "```text",
        summary[display_cols].to_string(index=False),
        "```",
        "",
        "## Best Gross Drawdown Control",
        "",
        f"- experiment: {best_gross_dd['experiment']}",
        f"- gross_total_return_before_cost: {best_gross_dd['gross_total_return_before_cost']}",
        f"- gross_max_drawdown: {best_gross_dd['gross_max_drawdown']}",
        f"- total_return: {best_gross_dd['total_return']}",
        f"- max_abs_weight: {best_gross_dd.get('max_abs_weight')}",
        "",
        "## Best Net Return",
        "",
        f"- experiment: {best_return['experiment']}",
        f"- total_return: {best_return['total_return']}",
        f"- gross_total_return_before_cost: {best_return['gross_total_return_before_cost']}",
        f"- gross_max_drawdown: {best_return.get('gross_max_drawdown')}",
        "",
    ]
    (output_dir / "REVIEW.md").write_text("\n".join(lines), encoding="utf-8")
    summary.to_csv(output_dir / "summary_unsorted.csv", index=False)
    summary.sort_values("total_return", ascending=False).to_csv(output_dir / "summary_by_return.csv", index=False)
    summary_sorted.to_csv(output_dir / "summary_by_gross_drawdown.csv", index=False)


if __name__ == "__main__":
    main()
