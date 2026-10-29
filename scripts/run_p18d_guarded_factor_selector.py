#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
from crypto_stat_arb.portfolio import portfolio_exposures, spearman_ic, update_state
from run_p3_committee_guard_scan import CommitteeSpec, side_fraction_weights
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
from run_p18c_factor_selector_portfolio import (
    DEFAULT_P15C_RUN,
    P18C_SCORE,
    breakeven_cost,
    build_asset_contribution,
    build_cost_sensitivity,
    build_segment_comparison,
    load_existing_result,
    plot_fold_delta,
    write_run_result,
)
from run_turnover_experiments import to_jsonable


LONG_ROOT = EXPERIMENT_ROOT / "outputs" / "long_history_nested_oos"
DEFAULT_CONFIG = EXPERIMENT_ROOT / "configs" / "baseline_locked_v1.yaml"
DEFAULT_FOLDS = LONG_ROOT / "inputs" / "long_formula_selections.csv"
DEFAULT_P18B = EXPERIMENT_ROOT / "outputs" / "p18b_factor_selector_oos_ic"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "outputs" / "p18d_guarded_factor_selector"
DEFAULT_P15C_SELECTED = LONG_ROOT / "p15c_nested" / "selected_extreme_cluster_by_fold.csv"
DEFAULT_P15C_VALIDATION = LONG_ROOT / "p15c_nested" / "validation_extreme_cluster_metrics.csv"
DEFAULT_COST_BPS = [0, 6, 10, 15, 20, 30]


@dataclass(frozen=True)
class GuardSpec:
    name: str
    momentum_scale: float
    weak_factor_scale: float
    disagreement_scale: float
    regime_gross_scale: float
    long_fraction: float = 0.50
    short_confirm_rank: float = 0.35


def main() -> None:
    parser = argparse.ArgumentParser(description="P18d: P18c score with validation-selected regime/tail guards.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--asset-panel", default=str(DEFAULT_ASSET_PANEL))
    parser.add_argument("--manual-panel", default=str(DEFAULT_MANUAL_PANEL))
    parser.add_argument("--p6b-panel", default=str(DEFAULT_P6B_PANEL))
    parser.add_argument("--derivatives-panel", default=str(DEFAULT_DERIVATIVES_PANEL))
    parser.add_argument("--folds", default=str(DEFAULT_FOLDS))
    parser.add_argument("--p18b-dir", default=str(DEFAULT_P18B))
    parser.add_argument("--p15c-run-dir", default=str(DEFAULT_P15C_RUN))
    parser.add_argument("--p15c-selected", default=str(DEFAULT_P15C_SELECTED))
    parser.add_argument("--p15c-validation", default=str(DEFAULT_P15C_VALIDATION))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--horizon", type=int, default=72)
    parser.add_argument("--validation-days", type=int, default=90)
    parser.add_argument("--cost-bps", type=float, nargs="+", default=DEFAULT_COST_BPS)
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    run_dir = ensure_dir(output_dir / "nested_p18d_run")
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    horizon = int(args.horizon)

    print("[1/7] Loading folds, selected factors, and P15c references", flush=True)
    folds = pd.read_csv(args.folds, parse_dates=["train_start", "train_end", "test_start", "test_end"]).sort_values("fold")
    selected_factors = pd.read_csv(Path(args.p18b_dir) / "selected_factors_by_fold.csv", parse_dates=["test_start"])
    selected_factors = selected_factors[selected_factors["horizon"] == horizon].copy()
    p15c_result = load_existing_result(Path(args.p15c_run_dir))
    p15c_validation = load_selected_p15c_validation(Path(args.p15c_selected), Path(args.p15c_validation))

    print("[2/7] Loading factor ranks and allocator panel", flush=True)
    factor_panel, _ = load_factor_library(
        asset_panel=Path(args.asset_panel),
        manual_panel=Path(args.manual_panel),
        p6b_panel=Path(args.p6b_panel),
        derivatives_panel=Path(args.derivatives_panel),
        max_factors=0,
    )
    needed_factors = sorted(set(selected_factors["factor"]))
    rank_matrices = {factor: centered_rank_matrix(factor_panel, factor, min_symbols=8) for factor in needed_factors}
    base_panel = load_p3d_panel(config_path, Path(args.asset_panel), Path(args.manual_panel), horizon)
    add_regime_features(base_panel)

    print("[3/7] Validating P18d guard candidates fold-locally", flush=True)
    pf_cfg = make_portfolio_config(cfg, horizon)
    risk_template = make_risk_template(cfg)
    guard_specs = build_guard_specs()
    pd.DataFrame([spec.__dict__ for spec in guard_specs]).to_csv(output_dir / "guard_grid.csv", index=False)
    validation, selected = run_guard_validation(
        base_panel=base_panel,
        folds=folds,
        selected_factors=selected_factors,
        rank_matrices=rank_matrices,
        p15c_validation=p15c_validation,
        pf_cfg=pf_cfg,
        risk_template=risk_template,
        guard_specs=guard_specs,
        validation_days=int(args.validation_days),
    )
    validation.to_csv(output_dir / "validation_guard_metrics.csv", index=False)
    selected.to_csv(output_dir / "selected_guard_by_fold.csv", index=False)

    print("[4/7] Replaying selected P18d/P15c fallback on test windows", flush=True)
    result = replay_selected_guards(
        base_panel=base_panel,
        folds=folds,
        selected_factors=selected_factors,
        rank_matrices=rank_matrices,
        selected=selected,
        p15c_result=p15c_result,
        pf_cfg=pf_cfg,
        risk_template=risk_template,
    )
    write_run_result(run_dir, result)

    print("[5/7] Building diagnostics", flush=True)
    comparison = build_comparison({"p18d": result, "p18c": load_existing_result(EXPERIMENT_ROOT / "outputs" / "p18c_factor_selector_portfolio" / "nested_p18c_run"), "p15c": p15c_result})
    comparison.to_csv(output_dir / "comparison_summary.csv", index=False)
    cost = build_cost_sensitivity({"p18d": result, "p18c": load_existing_result(EXPERIMENT_ROOT / "outputs" / "p18c_factor_selector_portfolio" / "nested_p18c_run"), "p15c": p15c_result}, [float(x) for x in args.cost_bps])
    cost.to_csv(output_dir / "cost_sensitivity.csv", index=False)
    segments = build_segment_comparison({"p18d": result, "p18c": load_existing_result(EXPERIMENT_ROOT / "outputs" / "p18c_factor_selector_portfolio" / "nested_p18c_run"), "p15c": p15c_result}, [float(x) for x in args.cost_bps])
    segments.to_csv(output_dir / "segment_cost_summary.csv", index=False)
    fold_delta = build_fold_delta(result, p15c_result)
    fold_delta.to_csv(output_dir / "fold_delta_vs_p15c.csv", index=False)
    asset = build_asset_contribution(result["positions"])
    asset.to_csv(output_dir / "p18d_asset_contribution.csv", index=False)

    print("[6/7] Writing plots", flush=True)
    plot_curves(output_dir, {"p18d": result, "p18c": load_existing_result(EXPERIMENT_ROOT / "outputs" / "p18c_factor_selector_portfolio" / "nested_p18c_run"), "p15c": p15c_result}, [6.0, 15.0, 20.0])
    plot_fold_delta(output_dir, fold_delta)

    print("[7/7] Writing report", flush=True)
    write_report(output_dir, comparison, cost, segments, fold_delta, asset, selected, validation)
    metadata = {
        "horizon": horizon,
        "validation_days": int(args.validation_days),
        "hard_accept": "gross, 20bps return, drawdown, turnover, and positive-period filters versus selected P15c validation reference",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Done: {output_dir}", flush=True)


def build_guard_specs() -> list[GuardSpec]:
    specs: list[GuardSpec] = []
    for momentum_scale in [0.35, 0.50, 0.65]:
        for weak_factor_scale in [0.40, 0.60]:
            for disagreement_scale in [0.35, 0.50]:
                for regime_scale in [0.35, 0.50, 0.70]:
                    specs.append(
                        GuardSpec(
                            name=(
                                f"p18d_mom{int(momentum_scale*100)}_weak{int(weak_factor_scale*100)}_"
                                f"dis{int(disagreement_scale*100)}_reg{int(regime_scale*100)}"
                            ),
                            momentum_scale=momentum_scale,
                            weak_factor_scale=weak_factor_scale,
                            disagreement_scale=disagreement_scale,
                            regime_gross_scale=regime_scale,
                        )
                    )
    return specs


def load_selected_p15c_validation(selected_path: Path, validation_path: Path) -> pd.DataFrame:
    selected = pd.read_csv(selected_path)
    validation = pd.read_csv(validation_path)
    merged = selected[["fold", "selected_extreme_spec"]].merge(
        validation,
        left_on=["fold", "selected_extreme_spec"],
        right_on=["fold", "extreme_spec"],
        how="left",
    )
    if merged["validation_total_return"].isna().any():
        raise RuntimeError("Could not map every selected P15c fold to validation metrics.")
    return merged


def add_regime_features(panel: pd.DataFrame) -> None:
    market = panel.groupby("timestamp")["log_ret"].mean().sort_index()
    market_ret_72 = market.rolling(72, min_periods=24).sum()
    market_vol_14d = market.rolling(336, min_periods=72).std(ddof=0)
    dispersion = panel.groupby("timestamp")["residual_log_ret"].std(ddof=0).sort_index()
    residual_disp_7d = dispersion.rolling(168, min_periods=48).mean()
    vol_threshold = market_vol_14d.rolling(4320, min_periods=720).quantile(0.80).shift(1)
    disp_threshold = residual_disp_7d.rolling(4320, min_periods=720).quantile(0.80).shift(1)
    ret_threshold = market_ret_72.abs().rolling(4320, min_periods=720).quantile(0.80).shift(1)
    high_risk = ((market_vol_14d > vol_threshold) & (residual_disp_7d > disp_threshold)) | (market_ret_72.abs() > ret_threshold)
    panel["_p18d_market_ret_72h"] = panel["timestamp"].map(market_ret_72)
    panel["_p18d_market_vol_14d"] = panel["timestamp"].map(market_vol_14d)
    panel["_p18d_residual_disp_7d"] = panel["timestamp"].map(residual_disp_7d)
    panel["_p18d_high_risk_regime"] = panel["timestamp"].map(high_risk).fillna(False).astype(bool)


def run_guard_validation(
    base_panel: pd.DataFrame,
    folds: pd.DataFrame,
    selected_factors: pd.DataFrame,
    rank_matrices: dict[str, pd.DataFrame],
    p15c_validation: pd.DataFrame,
    pf_cfg: Any,
    risk_template: Any,
    guard_specs: list[GuardSpec],
    validation_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_by_fold = {int(k): v.copy() for k, v in selected_factors.groupby("fold", sort=False)}
    p15c_by_fold = p15c_validation.set_index("fold")
    rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for _, fold in folds.sort_values("fold").iterrows():
        fold_id = int(fold["fold"])
        print(f"      validation fold {fold_id}/{len(folds)}", flush=True)
        if fold_id not in selected_by_fold:
            continue
        validation_start = max(pd.Timestamp(fold["train_start"]), pd.Timestamp(fold["train_end"]) - pd.Timedelta(days=validation_days))
        validation_end = min(
            decision_end_for_closed_positions(validation_start + pd.Timedelta(days=validation_days), pf_cfg.primary_horizon),
            pd.Timestamp(fold["train_end"]),
        )
        p15c_ref = p15c_by_fold.loc[fold_id]
        fold_group = selected_by_fold[fold_id]
        for spec in guard_specs:
            score_panel = build_guarded_score_panel(fold_group, rank_matrices, validation_start, validation_end, fold_id, spec)
            fold_panel = attach_score(base_panel, score_panel)
            risk_cfg = risk_template.__class__(
                **{
                    **risk_template.__dict__,
                    "name": f"p18d_validation_{fold_id}_{spec.name}",
                    "short_resid_confirm_rank": spec.short_confirm_rank,
                    "disagreement_scale": spec.disagreement_scale,
                }
            )
            equity, positions, metrics = run_guarded_backtest(
                panel=fold_panel,
                start=validation_start,
                end=validation_end,
                cfg=pf_cfg,
                risk_cfg=risk_cfg,
                spec=spec,
            )
            enrich_metrics(equity, positions, metrics)
            hard = hard_accept(metrics, p15c_ref)
            rows.append(
                {
                    "fold": fold_id,
                    "guard_spec": spec.name,
                    **spec.__dict__,
                    "validation_start": validation_start,
                    "validation_decision_end": validation_end,
                    **{f"validation_{k}": v for k, v in metrics.items() if is_scalar(v)},
                    **{f"p15c_validation_{k.replace('validation_', '')}": v for k, v in p15c_ref.items() if str(k).startswith("validation_") and is_scalar(v)},
                    "hard_accept": hard,
                    "selection_score": selection_score(metrics, p15c_ref),
                }
            )
        fold_df = pd.DataFrame([row for row in rows if row["fold"] == fold_id])
        accepted = fold_df[fold_df["hard_accept"].astype(bool)].copy()
        if accepted.empty:
            selected_rows.append(
                {
                    "fold": fold_id,
                    "selected_guard_spec": "__P15C_FALLBACK__",
                    "fallback_to_p15c": True,
                    "validation_selection_score": float(p15c_ref["validation_total_return"]),
                    "validation_total_return": float(p15c_ref["validation_total_return"]),
                    "validation_gross_total_return_before_cost": float(p15c_ref["validation_gross_total_return_before_cost"]),
                    "validation_total_return_20bps": float(p15c_ref["validation_total_return_20bps"]),
                    "validation_sharpe": float(p15c_ref["validation_sharpe"]),
                    "validation_max_drawdown": float(p15c_ref["validation_max_drawdown"]),
                }
            )
        else:
            winner = accepted.sort_values(["selection_score", "validation_total_return_20bps"], ascending=False).iloc[0]
            selected_rows.append(
                {
                    "fold": fold_id,
                    "selected_guard_spec": winner["guard_spec"],
                    "fallback_to_p15c": False,
                    "momentum_scale": float(winner["momentum_scale"]),
                    "weak_factor_scale": float(winner["weak_factor_scale"]),
                    "disagreement_scale": float(winner["disagreement_scale"]),
                    "regime_gross_scale": float(winner["regime_gross_scale"]),
                    "validation_selection_score": float(winner["selection_score"]),
                    "validation_total_return": float(winner["validation_total_return"]),
                    "validation_gross_total_return_before_cost": float(winner["validation_gross_total_return_before_cost"]),
                    "validation_total_return_20bps": float(winner["validation_total_return_20bps"]),
                    "validation_sharpe": float(winner["validation_sharpe"]),
                    "validation_max_drawdown": float(winner["validation_max_drawdown"]),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(selected_rows)


def build_guarded_score_panel(
    fold_group: pd.DataFrame,
    rank_matrices: dict[str, pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
    fold_id: int,
    spec: GuardSpec,
) -> pd.DataFrame:
    combined = None
    total_weight = 0.0
    for _, row in fold_group.iterrows():
        factor = str(row["factor"])
        if factor not in rank_matrices:
            continue
        weight = adjusted_factor_weight(row, spec)
        if weight <= 0:
            continue
        block = rank_matrices[factor].loc[start:end] * float(row["orientation"])
        combined = block * weight if combined is None else combined.add(block * weight, fill_value=0.0)
        total_weight += weight
    if combined is None or total_weight <= 0:
        return pd.DataFrame(columns=["timestamp", "symbol", P18C_SCORE, "fold", "selected_factor_count", "selected_factors"])
    combined = combined / total_weight
    out = combined.stack(dropna=True).rename(P18C_SCORE).reset_index()
    out["fold"] = fold_id
    out["selected_factor_count"] = int(len(fold_group))
    out["selected_factors"] = "|".join(fold_group["raw_name"].astype(str).tolist())
    return out


def adjusted_factor_weight(row: pd.Series, spec: GuardSpec) -> float:
    weight = max(float(row["selection_score"]), 0.01)
    name = str(row["raw_name"]).lower()
    momentum_tokens = ["high_mom", "res_mom", "_mom", "momentum", "trend", "ma_slope", "uptrend"]
    if any(token in name for token in momentum_tokens):
        weight *= spec.momentum_scale
    if float(row.get("validation_decay_delta", 0.0)) < -0.01 or float(row.get("validation_ic_q25", 0.0)) < 0.0:
        weight *= spec.weak_factor_scale
    return weight


def attach_score(base_panel: pd.DataFrame, score_panel: pd.DataFrame) -> pd.DataFrame:
    panel = base_panel.drop(columns=[P18C_SCORE, P4_SCORE_COL, SCORE_COL, CANDIDATE_RANK_COL], errors="ignore").merge(
        score_panel[["timestamp", "symbol", P18C_SCORE]],
        on=["timestamp", "symbol"],
        how="left",
    )
    panel[P4_SCORE_COL] = panel[P18C_SCORE]
    panel[SCORE_COL] = panel[P18C_SCORE]
    panel[CANDIDATE_RANK_COL] = panel.groupby("timestamp", sort=False)[P18C_SCORE].rank(pct=True, method="average") * 2.0 - 1.0
    return panel


def run_guarded_backtest(
    panel: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cfg: Any,
    risk_cfg: Any,
    spec: GuardSpec,
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

    committee = CommitteeSpec(
        name=spec.name,
        w_a07=0.0,
        w_b11=0.0,
        w_c09=0.0,
        long_fraction=spec.long_fraction,
        short_guard="p18d",
        short_resid_confirm_rank=spec.short_confirm_rank,
    )
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
        raw_weights = side_fraction_weights(target_sides, cfg.gross_exposure, committee.long_fraction)
        target_weights, optimizer_info = build_weights(raw_weights, selected, rows_by_symbol, cfg, risk_cfg)
        target_weights = target_weights[target_weights.abs() > 1e-12]
        high_risk = bool(rows["_p18d_high_risk_regime"].iloc[0]) if "_p18d_high_risk_regime" in rows else False
        if high_risk:
            target_weights = target_weights * spec.regime_gross_scale
        if target_weights.empty or target_weights.abs().sum() <= 0:
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
                "p18d_high_risk_regime": float(high_risk),
                "regime_gross_scale": float(spec.regime_gross_scale if high_risk else 1.0),
                **optimizer_info,
                "target_applied": 1.0,
                "selected_formulas": spec.name,
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
                    "fwd_simple_ret": float(active_rows.loc[symbol, ret_col]),
                    "label_resid_ret": float(active_rows.loc[symbol, label_col]),
                    "contribution": float(weight * active_rows.loc[symbol, ret_col]),
                    "selected_score_cols": "p18d_guarded_factor_score",
                }
            )
        prev_weights = target_full_weights
        current_state = update_state(current_state, target_sides, True)
    equity_curve = pd.DataFrame(rows_out)
    positions = pd.DataFrame(pos_out)
    metrics = aggregate_oos_metrics(equity_curve, cfg.rebalance_hours)
    return equity_curve, positions, metrics


def enrich_metrics(equity: pd.DataFrame, positions: pd.DataFrame, metrics: dict[str, Any]) -> None:
    add_gross_metrics(equity, metrics)
    add_risk_event_metrics(positions, metrics)
    add_delay_metrics(equity, metrics)
    metrics.update(cost_adjusted_metrics(equity, [20.0, 30.0]))
    metrics["mean_high_risk_regime"] = float(equity["p18d_high_risk_regime"].mean()) if "p18d_high_risk_regime" in equity else np.nan
    metrics["mean_regime_gross_scale"] = float(equity["regime_gross_scale"].mean()) if "regime_gross_scale" in equity else np.nan


def cost_adjusted_metrics(equity: pd.DataFrame, cost_bps: list[float]) -> dict[str, float]:
    out: dict[str, float] = {}
    if equity.empty:
        return {f"total_return_{int(bps)}bps": np.nan for bps in cost_bps}
    for bps in cost_bps:
        gross = equity["gross_period_return"]
        adjusted = equity.assign(period_return=gross - equity["turnover"] * bps / 10_000.0)
        metrics = aggregate_oos_metrics(adjusted, 72)
        out[f"total_return_{int(bps)}bps"] = metrics.get("total_return", np.nan)
        out[f"sharpe_{int(bps)}bps"] = metrics.get("sharpe", np.nan)
        out[f"max_drawdown_{int(bps)}bps"] = metrics.get("max_drawdown", np.nan)
    return out


def hard_accept(metrics: dict[str, Any], p15c_ref: pd.Series) -> bool:
    if not metrics or not np.isfinite(metrics.get("total_return", np.nan)):
        return False
    p15_dd = float(p15c_ref["validation_max_drawdown"])
    return bool(
        (float(metrics.get("gross_total_return_before_cost", -999.0)) > float(p15c_ref["validation_gross_total_return_before_cost"]))
        and (float(metrics.get("total_return_20bps", -999.0)) > float(p15c_ref["validation_total_return_20bps"]))
        and (float(metrics.get("max_drawdown", -999.0)) >= 1.15 * p15_dd)
        and (float(metrics.get("mean_turnover", 999.0)) <= 1.05 * float(p15c_ref["validation_mean_turnover"]))
        and (float(metrics.get("positive_period_rate", 0.0)) >= float(p15c_ref["validation_positive_period_rate"]) - 0.05)
    )


def selection_score(metrics: dict[str, Any], p15c_ref: pd.Series) -> float:
    return float(
        100.0 * (metrics.get("total_return", -1.0) - float(p15c_ref["validation_total_return"]))
        + 120.0 * (metrics.get("total_return_20bps", -1.0) - float(p15c_ref["validation_total_return_20bps"]))
        + 80.0
        * (
            metrics.get("gross_total_return_before_cost", -1.0)
            - float(p15c_ref["validation_gross_total_return_before_cost"])
        )
        + 1.0 * (metrics.get("sharpe", -5.0) - float(p15c_ref["validation_sharpe"]))
        - 4.0 * max(0.0, metrics.get("mean_turnover", 1.0) - float(p15c_ref["validation_mean_turnover"]))
    )


def replay_selected_guards(
    base_panel: pd.DataFrame,
    folds: pd.DataFrame,
    selected_factors: pd.DataFrame,
    rank_matrices: dict[str, pd.DataFrame],
    selected: pd.DataFrame,
    p15c_result: dict[str, Any],
    pf_cfg: Any,
    risk_template: Any,
) -> dict[str, Any]:
    guard_by_name = {spec.name: spec for spec in build_guard_specs()}
    selected_by_fold = selected.set_index("fold")
    factor_by_fold = {int(k): v.copy() for k, v in selected_factors.groupby("fold", sort=False)}
    p15c_equity = {int(k): v.copy() for k, v in p15c_result["equity"].groupby("fold", sort=False)}
    p15c_positions = {int(k): v.copy() for k, v in p15c_result["positions"].groupby("fold", sort=False)}
    p15c_folds = p15c_result["folds"].set_index("fold")
    equity_parts: list[pd.DataFrame] = []
    position_parts: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    for _, fold in folds.sort_values("fold").iterrows():
        fold_id = int(fold["fold"])
        row = selected_by_fold.loc[fold_id]
        if bool(row["fallback_to_p15c"]):
            equity = p15c_equity.get(fold_id, pd.DataFrame()).copy()
            positions = p15c_positions.get(fold_id, pd.DataFrame()).copy()
            fold_metric = p15c_folds.loc[fold_id].to_dict()
            fold_metric.update({"selected_guard_spec": "__P15C_FALLBACK__", "fallback_to_p15c": True})
        else:
            spec = guard_by_name[str(row["selected_guard_spec"])]
            start = pd.Timestamp(fold["test_start"])
            end = decision_end_for_closed_positions(pd.Timestamp(fold["test_end"]), pf_cfg.primary_horizon)
            score_panel = build_guarded_score_panel(factor_by_fold[fold_id], rank_matrices, start, end, fold_id, spec)
            fold_panel = attach_score(base_panel, score_panel)
            risk_cfg = risk_template.__class__(
                **{
                    **risk_template.__dict__,
                    "name": f"p18d_test_{fold_id}_{spec.name}",
                    "short_resid_confirm_rank": spec.short_confirm_rank,
                    "disagreement_scale": spec.disagreement_scale,
                }
            )
            equity, positions, metrics = run_guarded_backtest(fold_panel, start, end, pf_cfg, risk_cfg, spec)
            enrich_metrics(equity, positions, metrics)
            fold_metric = {
                "fold": fold_id,
                "train_start": fold["train_start"],
                "train_end": fold["train_end"],
                "test_start": fold["test_start"],
                "test_end": fold["test_end"],
                "test_decision_end": end,
                "selected_guard_spec": spec.name,
                "fallback_to_p15c": False,
                **metrics,
            }
        if not equity.empty:
            equity = equity.copy()
            equity["fold"] = fold_id
            equity["selected_guard_spec"] = fold_metric["selected_guard_spec"]
            equity["fallback_to_p15c"] = bool(fold_metric["fallback_to_p15c"])
            equity_parts.append(equity)
        if not positions.empty:
            positions = positions.copy()
            positions["fold"] = fold_id
            positions["selected_guard_spec"] = fold_metric["selected_guard_spec"]
            positions["fallback_to_p15c"] = bool(fold_metric["fallback_to_p15c"])
            position_parts.append(positions)
        fold_rows.append(fold_metric)
    equity_curve = pd.concat(equity_parts, ignore_index=True) if equity_parts else pd.DataFrame()
    positions = pd.concat(position_parts, ignore_index=True) if position_parts else pd.DataFrame()
    folds_out = pd.DataFrame(fold_rows)
    metrics = summarize_nested_result({"equity": equity_curve, "positions": positions, "folds": folds_out}, pf_cfg.rebalance_hours)
    return {"equity": equity_curve, "positions": positions, "folds": folds_out, "metrics": metrics}


def build_comparison(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame([{"stage": name, **result["metrics"]} for name, result in results.items()])


def build_fold_delta(p18d: dict[str, Any], p15c: dict[str, Any]) -> pd.DataFrame:
    left = p18d["folds"][["fold", "test_start", "selected_guard_spec", "fallback_to_p15c", "total_return", "gross_total_return_before_cost", "sharpe", "max_drawdown", "mean_turnover"]].copy()
    right = p15c["folds"][["fold", "total_return", "gross_total_return_before_cost", "sharpe", "max_drawdown", "mean_turnover"]].copy()
    merged = left.merge(right, on="fold", suffixes=("_p18d", "_p15c"))
    for col in ["total_return", "gross_total_return_before_cost", "sharpe", "max_drawdown", "mean_turnover"]:
        merged[f"delta_{col}"] = merged[f"{col}_p18d"] - merged[f"{col}_p15c"]
    return merged


def plot_curves(output_dir: Path, results: dict[str, dict[str, Any]], cost_bps: list[float]) -> None:
    for bps in cost_bps:
        fig, ax = plt.subplots(figsize=(14, 7))
        for name, result in results.items():
            equity = result["equity"].sort_values("timestamp")
            gross = equity["gross_period_return"] if "gross_period_return" in equity else equity["period_return"] + equity["cost_return"]
            ret = gross - equity["turnover"] * bps / 10_000.0
            curve = (1.0 + ret).cumprod() - 1.0
            ax.plot(equity["timestamp"], curve, label=name, linewidth=2.2 if name == "p18d" else 1.5)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_title(f"P18d vs P18c/P15c cumulative return, {bps:g}bps one-way cost")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Cumulative return")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"p18d_vs_refs_pnl_{int(bps)}bps.png", dpi=160)
        plt.close(fig)


def write_report(
    output_dir: Path,
    comparison: pd.DataFrame,
    cost: pd.DataFrame,
    segments: pd.DataFrame,
    fold_delta: pd.DataFrame,
    asset: pd.DataFrame,
    selected: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    accept_rate = 1.0 - float(selected["fallback_to_p15c"].mean()) if not selected.empty else np.nan
    report = [
        "# P18d Guarded Factor Selector",
        "",
        f"- P18d accepted fold rate: {accept_rate:.2%}",
        f"- P18d breakeven one-way cost: {breakeven_cost(cost, 'p18d'):.2f}bps",
        f"- P18c breakeven one-way cost: {breakeven_cost(cost, 'p18c'):.2f}bps",
        f"- P15c breakeven one-way cost: {breakeven_cost(cost, 'p15c'):.2f}bps",
        "",
        "## Comparison",
        "",
        "```text",
        comparison.to_string(index=False),
        "```",
        "",
        "## Cost Sensitivity",
        "",
        "```text",
        cost.to_string(index=False),
        "```",
        "",
        "## Segment Summary",
        "",
        "```text",
        segments[segments["one_way_cost_bps"].isin([6.0, 15.0, 20.0, 30.0])].to_string(index=False),
        "```",
        "",
        "## Selected Guard Counts",
        "",
        "```text",
        selected["selected_guard_spec"].value_counts().to_string(),
        "```",
        "",
        "## Worst Fold Deltas vs P15c",
        "",
        "```text",
        fold_delta.sort_values("delta_total_return").head(12).to_string(index=False),
        "```",
        "",
        "## Asset Contribution",
        "",
        "```text",
        asset.to_string(index=False),
        "```",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")


def is_scalar(value: Any) -> bool:
    return isinstance(value, (str, bool, int, float, np.integer, np.floating, pd.Timestamp)) or value is None


if __name__ == "__main__":
    main()
