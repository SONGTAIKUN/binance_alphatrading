#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
from crypto_stat_arb.portfolio import portfolio_exposures, spearman_ic, update_state
from run_p3_committee_guard_scan import CommitteeSpec, build_committee_specs, side_fraction_weights
from run_p3d_p4_gated_rerank_alphagpt import (
    CANDIDATE_RANK_COL,
    P4_SCORE_COL,
    SCORE_COL,
    add_delay_metrics,
    add_gross_metrics,
    add_risk_event_metrics,
    aggregate_oos_metrics,
    build_weights,
    decision_end_for_closed_positions,
    load_panel as load_p3d_panel,
    make_portfolio_config,
    make_risk_template,
    select_p4_gated_target_sides,
)
from run_p4_nested_walkforward_selection import summarize_nested_result
from run_p18_factor_decay_walkforward import (
    DEFAULT_ASSET_PANEL,
    DEFAULT_DERIVATIVES_PANEL,
    DEFAULT_MANUAL_PANEL,
    DEFAULT_P6B_PANEL,
    EXPERIMENT_ROOT,
    centered_rank_matrix,
    load_factor_library,
)
from run_turnover_experiments import to_jsonable


LONG_ROOT = EXPERIMENT_ROOT / "outputs" / "long_history_nested_oos"
DEFAULT_CONFIG = EXPERIMENT_ROOT / "configs" / "baseline_locked_v1.yaml"
DEFAULT_FOLDS = LONG_ROOT / "inputs" / "long_formula_selections.csv"
DEFAULT_P18B = EXPERIMENT_ROOT / "outputs" / "p18b_factor_selector_oos_ic"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "outputs" / "p18c_factor_selector_portfolio"
DEFAULT_P15C_RUN = LONG_ROOT / "p15c_nested" / "nested_extreme_cluster_run"
DEFAULT_COST_BPS = [0, 6, 10, 15, 20, 30]
P18C_SCORE = "_p18c_fold_local_factor_score"


def main() -> None:
    parser = argparse.ArgumentParser(description="P18c fold-local factor selector portfolio backtest.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--asset-panel", default=str(DEFAULT_ASSET_PANEL))
    parser.add_argument("--manual-panel", default=str(DEFAULT_MANUAL_PANEL))
    parser.add_argument("--p6b-panel", default=str(DEFAULT_P6B_PANEL))
    parser.add_argument("--derivatives-panel", default=str(DEFAULT_DERIVATIVES_PANEL))
    parser.add_argument("--folds", default=str(DEFAULT_FOLDS))
    parser.add_argument("--p18b-dir", default=str(DEFAULT_P18B))
    parser.add_argument("--p15c-run-dir", default=str(DEFAULT_P15C_RUN))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--horizon", type=int, default=72)
    parser.add_argument("--long-fraction", type=float, default=0.50)
    parser.add_argument("--short-confirm-rank", type=float, default=0.35)
    parser.add_argument("--cost-bps", type=float, nargs="+", default=DEFAULT_COST_BPS)
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    run_dir = ensure_dir(output_dir / "nested_p18c_run")
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    horizon = int(args.horizon)

    print("[1/6] Loading folds and P18b selected factors", flush=True)
    folds = pd.read_csv(args.folds, parse_dates=["train_start", "train_end", "test_start", "test_end"]).sort_values("fold")
    selected = pd.read_csv(Path(args.p18b_dir) / "selected_factors_by_fold.csv", parse_dates=["test_start"])
    selected = selected[selected["horizon"] == horizon].copy()
    if selected.empty:
        raise RuntimeError(f"No P18b selected factors found for horizon={horizon}.")
    selected.to_csv(output_dir / "selected_factors_horizon.csv", index=False)

    print("[2/6] Building fold-local P18c score panel", flush=True)
    score_panel = build_p18c_score_panel(
        selected=selected,
        folds=folds,
        asset_panel=Path(args.asset_panel),
        manual_panel=Path(args.manual_panel),
        p6b_panel=Path(args.p6b_panel),
        derivatives_panel=Path(args.derivatives_panel),
    )
    score_panel.to_csv(output_dir / "p18c_score_panel.csv.gz", index=False)

    print("[3/6] Loading strict-neutral allocator panel", flush=True)
    base_panel = load_p3d_panel(config_path, Path(args.asset_panel), Path(args.manual_panel), horizon)
    add_p18c_entry_fields(base_panel)

    print("[4/6] Replaying P18c on test windows only", flush=True)
    pf_cfg = make_portfolio_config(cfg, horizon)
    risk_template = make_risk_template(cfg)
    spec = CommitteeSpec(
        name="p18c_fold_local_factor_selector",
        w_a07=0.0,
        w_b11=0.0,
        w_c09=0.0,
        long_fraction=float(args.long_fraction),
        short_guard="p18c_score_gate",
        short_resid_confirm_rank=float(args.short_confirm_rank),
    )
    result = replay_p18c_tests(base_panel, score_panel, folds, pf_cfg, risk_template, spec)
    write_run_result(run_dir, result)

    print("[5/6] Comparing with long-history P15c", flush=True)
    p15c = load_existing_result(Path(args.p15c_run_dir))
    comparison = build_comparison({"p18c": result, "p15c": p15c}, horizon)
    comparison.to_csv(output_dir / "comparison_summary.csv", index=False)
    cost = build_cost_sensitivity({"p18c": result, "p15c": p15c}, [float(x) for x in args.cost_bps])
    cost.to_csv(output_dir / "cost_sensitivity.csv", index=False)
    segments = build_segment_comparison({"p18c": result, "p15c": p15c}, [float(x) for x in args.cost_bps])
    segments.to_csv(output_dir / "segment_cost_summary.csv", index=False)
    fold_delta = build_fold_delta(result, p15c)
    fold_delta.to_csv(output_dir / "fold_delta_vs_p15c.csv", index=False)
    asset = build_asset_contribution(result["positions"])
    asset.to_csv(output_dir / "p18c_asset_contribution.csv", index=False)

    print("[6/6] Writing plots and report", flush=True)
    plot_curves(output_dir, {"p18c": result, "p15c": p15c}, [6.0, 15.0])
    plot_fold_delta(output_dir, fold_delta)
    write_report(output_dir, comparison, cost, segments, fold_delta, asset, selected)
    metadata = {
        "horizon": horizon,
        "long_fraction": float(args.long_fraction),
        "short_confirm_rank": float(args.short_confirm_rank),
        "score_source": str(Path(args.p18b_dir) / "selected_factors_by_fold.csv"),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Done: {output_dir}", flush=True)


def build_p18c_score_panel(
    selected: pd.DataFrame,
    folds: pd.DataFrame,
    asset_panel: Path,
    manual_panel: Path,
    p6b_panel: Path,
    derivatives_panel: Path,
) -> pd.DataFrame:
    factor_panel, _ = load_factor_library(
        asset_panel=asset_panel,
        manual_panel=manual_panel,
        p6b_panel=p6b_panel,
        derivatives_panel=derivatives_panel,
        max_factors=0,
    )
    needed_factors = sorted(set(selected["factor"]))
    rank_matrices = {factor: centered_rank_matrix(factor_panel, factor, min_symbols=8) for factor in needed_factors}
    fold_by_id = folds.set_index("fold")
    score_parts: list[pd.DataFrame] = []
    for fold_id, group in selected.groupby("fold", sort=True):
        if int(fold_id) not in fold_by_id.index:
            continue
        fold = fold_by_id.loc[int(fold_id)]
        start = pd.Timestamp(fold["test_start"])
        end = decision_end_for_closed_positions(pd.Timestamp(fold["test_end"]), 72)
        combined = None
        total_weight = 0.0
        for _, row in group.iterrows():
            factor = str(row["factor"])
            if factor not in rank_matrices:
                continue
            orientation = float(row["orientation"])
            weight = max(float(row["selection_score"]), 0.01)
            block = rank_matrices[factor].loc[start:end] * orientation
            combined = block * weight if combined is None else combined.add(block * weight, fill_value=0.0)
            total_weight += weight
        if combined is None or total_weight <= 0:
            continue
        combined = combined / total_weight
        long_score = combined.stack(dropna=True).rename(P18C_SCORE).reset_index()
        long_score["fold"] = int(fold_id)
        long_score["selected_factor_count"] = int(len(group))
        long_score["selected_factors"] = "|".join(group["raw_name"].astype(str).tolist())
        score_parts.append(long_score)
    if not score_parts:
        raise RuntimeError("No P18c scores were generated.")
    score = pd.concat(score_parts, ignore_index=True)
    return score


def add_p18c_entry_fields(panel: pd.DataFrame) -> None:
    panel[P18C_SCORE] = np.nan
    panel[P4_SCORE_COL] = np.nan
    panel[SCORE_COL] = np.nan
    panel[CANDIDATE_RANK_COL] = np.nan


def replay_p18c_tests(
    panel: pd.DataFrame,
    score_panel: pd.DataFrame,
    folds: pd.DataFrame,
    pf_cfg: Any,
    risk_template: Any,
    spec: CommitteeSpec,
) -> dict[str, Any]:
    equity_parts: list[pd.DataFrame] = []
    position_parts: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    score_by_fold = {int(k): v.copy() for k, v in score_panel.groupby("fold", sort=False)}
    for _, fold in folds.sort_values("fold").iterrows():
        fold_id = int(fold["fold"])
        if fold_id not in score_by_fold:
            continue
        fold_scores = score_by_fold[fold_id][["timestamp", "symbol", P18C_SCORE, "selected_factor_count", "selected_factors"]]
        fold_panel = panel.drop(columns=[P18C_SCORE, P4_SCORE_COL, SCORE_COL, CANDIDATE_RANK_COL], errors="ignore").merge(
            fold_scores,
            on=["timestamp", "symbol"],
            how="left",
        )
        fold_panel[P4_SCORE_COL] = fold_panel[P18C_SCORE]
        fold_panel[SCORE_COL] = fold_panel[P18C_SCORE]
        fold_panel[CANDIDATE_RANK_COL] = (
            fold_panel.groupby("timestamp", sort=False)[P18C_SCORE].rank(pct=True, method="average") * 2.0 - 1.0
        )
        risk_cfg = risk_template.__class__(
            **{
                **risk_template.__dict__,
                "name": f"p18c_{fold_id}",
                "short_resid_confirm_rank": spec.short_resid_confirm_rank,
            }
        )
        start = pd.Timestamp(fold["test_start"])
        end = decision_end_for_closed_positions(pd.Timestamp(fold["test_end"]), pf_cfg.primary_horizon)
        equity, positions, metrics = run_p18c_backtest(
            panel=fold_panel,
            start=start,
            end=end,
            cfg=pf_cfg,
            risk_cfg=risk_cfg,
            spec=spec,
        )
        add_gross_metrics(equity, metrics)
        add_risk_event_metrics(positions, metrics)
        add_delay_metrics(equity, metrics)
        fold_rows.append(
            {
                "fold": fold_id,
                "train_start": fold["train_start"],
                "train_end": fold["train_end"],
                "test_start": fold["test_start"],
                "test_end": fold["test_end"],
                "test_decision_end": end,
                "selected_factor_count": int(fold_scores["selected_factor_count"].max()),
                **metrics,
            }
        )
        if not equity.empty:
            equity = equity.copy()
            equity["fold"] = fold_id
            positions = positions.copy()
            positions["fold"] = fold_id
            equity_parts.append(equity)
            position_parts.append(positions)
    equity_curve = pd.concat(equity_parts, ignore_index=True) if equity_parts else pd.DataFrame()
    positions = pd.concat(position_parts, ignore_index=True) if position_parts else pd.DataFrame()
    folds_out = pd.DataFrame(fold_rows)
    metrics = summarize_nested_result({"equity": equity_curve, "positions": positions, "folds": folds_out}, pf_cfg.rebalance_hours)
    return {"equity": equity_curve, "positions": positions, "folds": folds_out, "metrics": metrics}


def run_p18c_backtest(
    panel: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cfg: Any,
    risk_cfg: Any,
    spec: CommitteeSpec,
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

    for ts in timestamps:
        rows = panel[panel["timestamp"] == ts].copy()
        rows = rows[
            np.isfinite(rows[P18C_SCORE])
            & np.isfinite(rows[ret_col])
            & np.isfinite(rows[label_col])
            & np.isfinite(rows["score_ema_025"])
            & np.isfinite(rows["score_baseline"])
        ]
        if len(rows) < cfg.long_k + cfg.short_k:
            continue
        rows["score_rank"] = rows[P18C_SCORE].rank(pct=True, method="average")
        rows["p4_rank"] = rows["score_rank"]
        rows["ema_rank"] = rows["score_ema_025"].rank(pct=True, method="average")
        rows["base_rank"] = rows["score_baseline"].rank(pct=True, method="average")
        rows = rows.sort_values(P18C_SCORE, ascending=False)
        rows_by_symbol = rows.set_index("symbol")
        target_sides = select_p4_gated_target_sides(rows, current_state, cfg, P18C_SCORE)
        if not any(side > 0 for side in target_sides.values()) or not any(side < 0 for side in target_sides.values()):
            continue
        selected = rows_by_symbol.reindex(list(target_sides)).reset_index()
        raw_weights = side_fraction_weights(target_sides, cfg.gross_exposure, spec.long_fraction)
        target_weights, optimizer_info = build_weights(raw_weights, selected, rows_by_symbol, cfg, risk_cfg)
        target_weights = target_weights[target_weights.abs() > 1e-12]
        if target_weights.empty:
            continue
        target_full_weights = pd.Series(0.0, index=symbols)
        target_full_weights.loc[target_weights.index] = target_weights
        turnover = float((target_full_weights - prev_weights).abs().sum())
        cost = turnover * (cfg.one_way_fee_bps + cfg.one_way_slippage_bps) / 10_000.0
        active_rows = rows_by_symbol.reindex(target_weights.index)
        gross_period_ret = float((target_weights * active_rows[ret_col]).sum())
        period_ret = gross_period_ret - cost
        equity *= 1.0 + period_ret
        rank_ic = spearman_ic(rows[P18C_SCORE], rows[label_col])
        long_labels = active_rows.loc[target_weights[target_weights > 0].index, label_col]
        short_labels = active_rows.loc[target_weights[target_weights < 0].index, label_col]
        top_bottom = float(long_labels.mean() - short_labels.mean()) if len(long_labels) and len(short_labels) else np.nan
        exposures = portfolio_exposures(target_weights, active_rows, cfg.neutralize_exposures)
        rows_out.append(
            {
                "timestamp": pd.Timestamp(ts),
                "exit_timestamp": pd.Timestamp(ts) + pd.Timedelta(hours=cfg.execution_delay_bars + cfg.primary_horizon),
                "period_return": period_ret,
                "gross_period_return": gross_period_ret,
                "cost_return": cost,
                "equity": equity,
                "turnover": turnover,
                "target_turnover": turnover,
                "turnover_saved": 0.0,
                "gross_exposure": float(target_weights.abs().sum()),
                "target_gross_exposure": float(target_full_weights.abs().sum()),
                "rank_ic": rank_ic,
                "top_bottom_label_spread": top_bottom,
                "abs_beta_btc": abs(exposures.get("beta_btc", 0.0)),
                "abs_beta_eth": abs(exposures.get("beta_eth", 0.0)),
                "abs_beta_market": abs(exposures.get("beta_market", 0.0)),
                "symbol_concentration": float((target_weights**2).sum()),
                "max_abs_weight": float(target_weights.abs().max()),
                "delay_tau_days": 0.0,
                "delay_retention": 0.0,
                "delay_new_target_weight": 1.0,
                "short_guard_removed": 0.0,
                **optimizer_info,
                "target_applied": 1.0,
                "selected_formulas": "P18c fold-local factor selector",
            }
        )
        for symbol, weight in target_weights.items():
            pos_out.append(
                {
                    "timestamp": pd.Timestamp(ts),
                    "exit_timestamp": pd.Timestamp(ts) + pd.Timedelta(hours=cfg.execution_delay_bars + cfg.primary_horizon),
                    "symbol": symbol,
                    "weight": float(weight),
                    "target_weight": float(target_full_weights.loc[symbol]),
                    "score": float(rows_by_symbol.loc[symbol, P18C_SCORE]),
                    "p4_score": float(rows_by_symbol.loc[symbol, P18C_SCORE]),
                    "candidate_rank": float(rows_by_symbol.loc[symbol, CANDIDATE_RANK_COL])
                    if CANDIDATE_RANK_COL in rows_by_symbol
                    else np.nan,
                    "p4_rank": float(rows_by_symbol.loc[symbol, "p4_rank"]),
                    "ema_rank": float(rows_by_symbol.loc[symbol, "ema_rank"]),
                    "base_rank": float(rows_by_symbol.loc[symbol, "base_rank"]),
                    "rank_a07": float(rows_by_symbol.loc[symbol, "rank_a07"]),
                    "rank_b11": float(rows_by_symbol.loc[symbol, "rank_b11"]),
                    "rank_c09": float(rows_by_symbol.loc[symbol, "rank_c09"]),
                    "fwd_simple_ret": float(active_rows.loc[symbol, ret_col]),
                    "label_resid_ret": float(active_rows.loc[symbol, label_col]),
                    "contribution": float(weight * active_rows.loc[symbol, ret_col]),
                    "selected_score_cols": "p18c_fold_local_factor_score",
                }
            )
        prev_weights = target_full_weights
        current_state = update_state(current_state, target_sides, True)
    equity_curve = pd.DataFrame(rows_out)
    positions = pd.DataFrame(pos_out)
    metrics = aggregate_oos_metrics(equity_curve, cfg.rebalance_hours)
    return equity_curve, positions, metrics


def write_run_result(output_dir: Path, result: dict[str, Any]) -> None:
    ensure_dir(output_dir)
    result["folds"].to_csv(output_dir / "walk_forward_folds.csv", index=False)
    result["equity"].to_csv(output_dir / "equity_curve.csv", index=False)
    result["positions"].to_csv(output_dir / "positions.csv", index=False)
    (output_dir / "metrics.json").write_text(json.dumps(to_jsonable(result["metrics"]), indent=2), encoding="utf-8")


def load_existing_result(run_dir: Path) -> dict[str, Any]:
    return {
        "folds": pd.read_csv(run_dir / "walk_forward_folds.csv", parse_dates=["train_start", "train_end", "test_start", "test_end", "test_decision_end"]),
        "equity": pd.read_csv(run_dir / "equity_curve.csv", parse_dates=["timestamp", "exit_timestamp"]),
        "positions": pd.read_csv(run_dir / "positions.csv", parse_dates=["timestamp", "exit_timestamp"]),
        "metrics": json.loads((run_dir / "metrics.json").read_text(encoding="utf-8")),
    }


def build_comparison(results: dict[str, dict[str, Any]], horizon: int) -> pd.DataFrame:
    rows = []
    for name, result in results.items():
        row = {"stage": name, **result["metrics"]}
        row.update(cost_adjusted_from_equity(result["equity"], [15.0, 20.0, 30.0]))
        rows.append(row)
    return pd.DataFrame(rows)


def cost_adjusted_from_equity(equity: pd.DataFrame, cost_bps: list[float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for bps in cost_bps:
        adjusted = equity.copy()
        gross = adjusted["gross_period_return"] if "gross_period_return" in adjusted else adjusted["period_return"] + adjusted["cost_return"]
        adjusted["period_return"] = gross - adjusted["turnover"] * bps / 10_000.0
        metrics = aggregate_oos_metrics(adjusted, 72)
        out[f"total_return_{int(bps)}bps"] = metrics.get("total_return", np.nan)
        out[f"sharpe_{int(bps)}bps"] = metrics.get("sharpe", np.nan)
        out[f"max_drawdown_{int(bps)}bps"] = metrics.get("max_drawdown", np.nan)
    return out


def build_cost_sensitivity(results: dict[str, dict[str, Any]], cost_bps: list[float]) -> pd.DataFrame:
    rows = []
    for name, result in results.items():
        equity = result["equity"]
        for bps in cost_bps:
            adjusted = equity.copy()
            gross = adjusted["gross_period_return"] if "gross_period_return" in adjusted else adjusted["period_return"] + adjusted["cost_return"]
            adjusted["period_return"] = gross - adjusted["turnover"] * bps / 10_000.0
            metrics = aggregate_oos_metrics(adjusted, 72)
            gross_metrics = aggregate_oos_metrics(
                adjusted.assign(period_return=gross),
                72,
            )
            rows.append(
                {
                    "stage": name,
                    "one_way_cost_bps": float(bps),
                    "total_return": metrics.get("total_return", np.nan),
                    "gross_total_return": gross_metrics.get("total_return", np.nan),
                    "sharpe": metrics.get("sharpe", np.nan),
                    "max_drawdown": metrics.get("max_drawdown", np.nan),
                    "gross_max_drawdown": gross_metrics.get("max_drawdown", np.nan),
                    "mean_turnover": metrics.get("mean_turnover", np.nan),
                    "total_turnover": metrics.get("total_turnover", np.nan),
                    "positive_period_rate": metrics.get("positive_period_rate", np.nan),
                }
            )
    return pd.DataFrame(rows)


def build_segment_comparison(results: dict[str, dict[str, Any]], cost_bps: list[float]) -> pd.DataFrame:
    cut = pd.Timestamp("2024-07-01", tz="UTC")
    rows = []
    for name, result in results.items():
        equity = result["equity"].copy()
        segments = {
            "full": equity,
            "pre_original_2023_2024H1": equity[equity["timestamp"] < cut],
            "original_window_2024_07_plus": equity[equity["timestamp"] >= cut],
        }
        for segment_name, seg in segments.items():
            if seg.empty:
                continue
            for bps in cost_bps:
                gross = seg["gross_period_return"] if "gross_period_return" in seg else seg["period_return"] + seg["cost_return"]
                adjusted = seg.assign(period_return=gross - seg["turnover"] * bps / 10_000.0)
                metrics = aggregate_oos_metrics(adjusted, 72)
                gross_metrics = aggregate_oos_metrics(seg.assign(period_return=gross), 72)
                rows.append(
                    {
                        "stage": name,
                        "segment": segment_name,
                        "one_way_cost_bps": float(bps),
                        "periods": metrics.get("periods", np.nan),
                        "total_return": metrics.get("total_return", np.nan),
                        "gross_total_return": gross_metrics.get("total_return", np.nan),
                        "sharpe": metrics.get("sharpe", np.nan),
                        "max_drawdown": metrics.get("max_drawdown", np.nan),
                        "gross_max_drawdown": gross_metrics.get("max_drawdown", np.nan),
                        "mean_turnover": metrics.get("mean_turnover", np.nan),
                        "total_turnover": metrics.get("total_turnover", np.nan),
                    }
                )
    return pd.DataFrame(rows)


def build_fold_delta(p18c: dict[str, Any], p15c: dict[str, Any]) -> pd.DataFrame:
    left = p18c["folds"][["fold", "test_start", "total_return", "gross_total_return_before_cost", "sharpe", "max_drawdown", "mean_turnover"]].copy()
    right = p15c["folds"][["fold", "total_return", "gross_total_return_before_cost", "sharpe", "max_drawdown", "mean_turnover"]].copy()
    merged = left.merge(right, on="fold", suffixes=("_p18c", "_p15c"))
    for col in ["total_return", "gross_total_return_before_cost", "sharpe", "max_drawdown", "mean_turnover"]:
        merged[f"delta_{col}"] = merged[f"{col}_p18c"] - merged[f"{col}_p15c"]
    return merged


def build_asset_contribution(positions: pd.DataFrame) -> pd.DataFrame:
    if positions.empty:
        return pd.DataFrame()
    grouped = positions.groupby("symbol", sort=True)
    out = grouped.agg(
        gross_contribution=("contribution", "sum"),
        avg_abs_weight=("weight", lambda s: float(s.abs().mean())),
        max_abs_weight=("weight", lambda s: float(s.abs().max())),
        observations=("weight", "count"),
    ).reset_index()
    total_abs = out["gross_contribution"].abs().sum()
    out["abs_contribution_share"] = out["gross_contribution"].abs() / total_abs if total_abs else np.nan
    return out.sort_values("gross_contribution")


def breakeven_cost(cost: pd.DataFrame, stage: str) -> float:
    sub = cost[cost["stage"] == stage].sort_values("one_way_cost_bps")
    xs = sub["one_way_cost_bps"].to_numpy(dtype=float)
    ys = sub["total_return"].to_numpy(dtype=float)
    for idx in range(len(xs) - 1):
        if (ys[idx] >= 0 >= ys[idx + 1]) or (ys[idx] <= 0 <= ys[idx + 1]):
            return float(xs[idx] + (0.0 - ys[idx]) * (xs[idx + 1] - xs[idx]) / (ys[idx + 1] - ys[idx]))
    return np.nan


def plot_curves(output_dir: Path, results: dict[str, dict[str, Any]], cost_bps: list[float]) -> None:
    for bps in cost_bps:
        fig, ax = plt.subplots(figsize=(14, 7))
        for name, result in results.items():
            equity = result["equity"].sort_values("timestamp")
            gross = equity["gross_period_return"] if "gross_period_return" in equity else equity["period_return"] + equity["cost_return"]
            ret = gross - equity["turnover"] * bps / 10_000.0
            curve = (1.0 + ret).cumprod() - 1.0
            ax.plot(equity["timestamp"], curve, label=name, linewidth=2.0 if name == "p18c" else 1.5)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_title(f"P18c vs P15c cumulative return, {bps:g}bps one-way cost")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Cumulative return")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"p18c_vs_p15c_pnl_{int(bps)}bps.png", dpi=160)
        plt.close(fig)


def plot_fold_delta(output_dir: Path, fold_delta: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(fold_delta["fold"], fold_delta["delta_total_return"])
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("P18c fold net return delta vs P15c")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Delta return")
    fig.tight_layout()
    fig.savefig(output_dir / "fold_delta_vs_p15c.png", dpi=160)
    plt.close(fig)


def write_report(
    output_dir: Path,
    comparison: pd.DataFrame,
    cost: pd.DataFrame,
    segments: pd.DataFrame,
    fold_delta: pd.DataFrame,
    asset: pd.DataFrame,
    selected: pd.DataFrame,
) -> None:
    p18_cost = cost[cost["stage"] == "p18c"]
    report = [
        "# P18c Fold-Local Factor Selector Portfolio",
        "",
        "P18c uses P18b fold-local selected factors as the portfolio score, then runs the strict-neutral allocator on test windows only.",
        "",
        f"- P18c estimated breakeven one-way cost: {breakeven_cost(cost, 'p18c'):.2f}bps",
        f"- P15c estimated breakeven one-way cost: {breakeven_cost(cost, 'p15c'):.2f}bps",
        "",
        "## Comparison",
        "",
        "```text",
        comparison.to_string(index=False),
        "```",
        "",
        "## P18c Cost Sensitivity",
        "",
        "```text",
        p18_cost.to_string(index=False),
        "```",
        "",
        "## Segment Summary",
        "",
        "```text",
        segments[segments["one_way_cost_bps"].isin([6.0, 15.0, 20.0, 30.0])].to_string(index=False),
        "```",
        "",
        "## Worst Fold Deltas vs P15c",
        "",
        "```text",
        fold_delta.sort_values("delta_total_return").head(10).to_string(index=False),
        "```",
        "",
        "## Common Selected Factors",
        "",
        "```text",
        selected["raw_name"].value_counts().head(40).to_string(),
        "```",
        "",
        "## Asset Contribution",
        "",
        "```text",
        asset.to_string(index=False),
        "```",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
