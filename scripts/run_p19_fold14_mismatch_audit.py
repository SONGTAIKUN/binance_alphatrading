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
from run_p3_committee_guard_scan import CommitteeSpec, side_fraction_weights
from run_p3d_p4_gated_rerank_alphagpt import (
    CANDIDATE_RANK_COL,
    P4_SCORE_COL,
    SCORE_COL,
    build_weights,
    decision_end_for_closed_positions,
    load_panel as load_p3d_panel,
    make_portfolio_config,
    make_risk_template,
    select_p4_gated_target_sides,
)
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
    load_existing_result,
    run_p18c_backtest,
)
from run_p18d_guarded_factor_selector import (
    DEFAULT_FOLDS,
    DEFAULT_P15C_SELECTED,
    DEFAULT_P15C_VALIDATION,
    DEFAULT_P18B,
    GuardSpec,
    add_regime_features,
    attach_score,
    build_guard_specs,
    build_guarded_score_panel,
    enrich_metrics,
    hard_accept,
    load_selected_p15c_validation,
    run_guarded_backtest,
    selection_score,
)


DEFAULT_CONFIG = EXPERIMENT_ROOT / "configs" / "baseline_locked_v1.yaml"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "outputs" / "p19_fold14_mismatch_audit"
DEFAULT_P18C_RUN = EXPERIMENT_ROOT / "outputs" / "p18c_factor_selector_portfolio" / "nested_p18c_run"
DEFAULT_P18D_RUN = EXPERIMENT_ROOT / "outputs" / "p18d_guarded_factor_selector" / "nested_p18d_run"
DEFAULT_P18D_VALIDATION = EXPERIMENT_ROOT / "outputs" / "p18d_guarded_factor_selector" / "validation_guard_metrics.csv"
DEFAULT_HORIZON = 72
DEFAULT_FOLD = 14
DEFAULT_COST_BPS = 6.0
ID_COLS = ["timestamp", "symbol"]


def main() -> None:
    parser = argparse.ArgumentParser(description="P19 fold14 validation-test mismatch audit.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--asset-panel", default=str(DEFAULT_ASSET_PANEL))
    parser.add_argument("--manual-panel", default=str(DEFAULT_MANUAL_PANEL))
    parser.add_argument("--p6b-panel", default=str(DEFAULT_P6B_PANEL))
    parser.add_argument("--derivatives-panel", default=str(DEFAULT_DERIVATIVES_PANEL))
    parser.add_argument("--folds", default=str(DEFAULT_FOLDS))
    parser.add_argument("--p18b-dir", default=str(DEFAULT_P18B))
    parser.add_argument("--p18d-validation", default=str(DEFAULT_P18D_VALIDATION))
    parser.add_argument("--p15c-selected", default=str(DEFAULT_P15C_SELECTED))
    parser.add_argument("--p15c-validation", default=str(DEFAULT_P15C_VALIDATION))
    parser.add_argument("--p15c-run-dir", default=str(DEFAULT_P15C_RUN))
    parser.add_argument("--p18c-run-dir", default=str(DEFAULT_P18C_RUN))
    parser.add_argument("--p18d-run-dir", default=str(DEFAULT_P18D_RUN))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--fold", type=int, default=DEFAULT_FOLD)
    parser.add_argument("--validation-days", type=int, default=90)
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    config_path = Path(args.config).resolve()
    cfg_raw = load_config(config_path)
    horizon = int(args.horizon)
    fold_id = int(args.fold)

    print("[1/8] Loading folds, existing runs, and selected factors", flush=True)
    folds = pd.read_csv(args.folds, parse_dates=["train_start", "train_end", "test_start", "test_end"]).sort_values("fold")
    fold = folds.loc[folds["fold"].astype(int).eq(fold_id)].iloc[0]
    selected_factors = pd.read_csv(Path(args.p18b_dir) / "selected_factors_by_fold.csv", parse_dates=["test_start"])
    selected_factors = selected_factors[(selected_factors["horizon"].astype(int) == horizon) & (selected_factors["fold"].astype(int) == fold_id)].copy()
    if selected_factors.empty:
        raise RuntimeError(f"No selected P18b factors for fold={fold_id}, horizon={horizon}.")
    selected_factors.to_csv(output_dir / "fold14_selected_factors.csv", index=False)

    p15c_result = load_existing_result(Path(args.p15c_run_dir))
    p18c_result = load_existing_result(Path(args.p18c_run_dir))
    p18d_result = load_existing_result(Path(args.p18d_run_dir))
    p15c_validation = load_selected_p15c_validation(Path(args.p15c_selected), Path(args.p15c_validation))
    p18d_validation = pd.read_csv(args.p18d_validation)

    validation_start, validation_end, test_start, test_end = fold_windows(fold, horizon, int(args.validation_days))
    selected_guard = selected_p18d_guard(p18d_result, fold_id)
    selected_spec = {spec.name: spec for spec in build_guard_specs()}[selected_guard]

    print("[2/8] Loading factor ranks and allocator panel", flush=True)
    factor_panel, _ = load_factor_library(
        asset_panel=Path(args.asset_panel),
        manual_panel=Path(args.manual_panel),
        p6b_panel=Path(args.p6b_panel),
        derivatives_panel=Path(args.derivatives_panel),
        max_factors=0,
    )
    needed_factors = sorted(set(selected_factors["factor"]))
    rank_matrices = {factor: centered_rank_matrix(factor_panel, factor, min_symbols=8) for factor in needed_factors}
    label_rank = centered_rank_matrix(factor_panel, f"label_resid_ret_h{horizon}", min_symbols=8)

    base_panel = load_p3d_panel(config_path, Path(args.asset_panel), Path(args.manual_panel), horizon)
    add_regime_features(base_panel)
    pf_cfg = make_portfolio_config(cfg_raw, horizon)
    risk_template = make_risk_template(cfg_raw)

    print("[3/8] P19a replaying fold14 validation/test configs", flush=True)
    replay = build_config_replay(
        base_panel=base_panel,
        selected_factors=selected_factors,
        rank_matrices=rank_matrices,
        p15c_result=p15c_result,
        p15c_validation=p15c_validation,
        p18c_result=p18c_result,
        p18d_result=p18d_result,
        p18d_validation=p18d_validation,
        pf_cfg=pf_cfg,
        risk_template=risk_template,
        validation_start=validation_start,
        validation_end=validation_end,
        test_start=test_start,
        test_end=test_end,
        fold_id=fold_id,
        selected_guard=selected_guard,
    )
    replay.to_csv(output_dir / "p19a_config_replay.csv", index=False)

    print("[4/8] P19b building regime shift diagnostics", flush=True)
    regime, regime_ts = build_regime_shift(
        base_panel=base_panel,
        derivatives_panel=Path(args.derivatives_panel),
        validation_start=validation_start,
        validation_end=validation_end,
        test_start=test_start,
        test_end=test_end,
    )
    regime.to_csv(output_dir / "p19b_regime_shift.csv", index=False)
    regime_ts.to_csv(output_dir / "p19b_regime_timeseries.csv", index=False)

    print("[5/8] P19c computing selected factor IC/inversion diagnostics", flush=True)
    factor_ic, factor_ts = build_factor_ic_diagnostics(
        selected_factors=selected_factors,
        rank_matrices=rank_matrices,
        label_rank=label_rank,
        validation_start=validation_start,
        validation_end=validation_end,
        test_start=test_start,
        test_end=test_end,
        selected_spec=selected_spec,
    )
    factor_ic.to_csv(output_dir / "p19c_factor_ic_decay.csv", index=False)
    factor_ts.to_csv(output_dir / "p19c_factor_ic_timeseries.csv", index=False)

    print("[6/8] P19d attributing P18d vs P15c position differences", flush=True)
    position_diff, symbol_delta, category_delta, timestamp_delta = build_position_diff(
        p18d_result=p18d_result,
        p15c_result=p15c_result,
        fold_id=fold_id,
    )
    position_diff.to_csv(output_dir / "p19d_position_diff.csv", index=False)
    symbol_delta.to_csv(output_dir / "p19d_symbol_delta.csv", index=False)
    category_delta.to_csv(output_dir / "p19d_category_delta.csv", index=False)
    timestamp_delta.to_csv(output_dir / "p19d_timestamp_delta.csv", index=False)

    print("[7/8] P19e auditing allocator projection", flush=True)
    alloc_snap, alloc_summary = build_allocator_audit(
        base_panel=base_panel,
        selected_factors=selected_factors,
        rank_matrices=rank_matrices,
        pf_cfg=pf_cfg,
        risk_template=risk_template,
        selected_spec=selected_spec,
        validation_start=validation_start,
        validation_end=validation_end,
        test_start=test_start,
        test_end=test_end,
        fold_id=fold_id,
    )
    alloc_snap.to_csv(output_dir / "p19e_allocator_snapshot.csv", index=False)
    alloc_summary.to_csv(output_dir / "p19e_allocator_summary.csv", index=False)

    print("[8/8] Writing plots and report", flush=True)
    plot_outputs(output_dir, replay, regime_ts, factor_ts, timestamp_delta)
    write_report(
        output_dir=output_dir,
        fold=fold,
        selected_guard=selected_guard,
        replay=replay,
        regime=regime,
        factor_ic=factor_ic,
        symbol_delta=symbol_delta,
        category_delta=category_delta,
        timestamp_delta=timestamp_delta,
        alloc_summary=alloc_summary,
        validation_start=validation_start,
        validation_end=validation_end,
        test_start=test_start,
        test_end=test_end,
    )
    metadata = {
        "fold": fold_id,
        "horizon": horizon,
        "selected_guard": selected_guard,
        "validation_start": str(validation_start),
        "validation_end": str(validation_end),
        "test_start": str(test_start),
        "test_end": str(test_end),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Done: {output_dir}", flush=True)


def fold_windows(fold: pd.Series, horizon: int, validation_days: int) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    validation_start = max(pd.Timestamp(fold["train_start"]), pd.Timestamp(fold["train_end"]) - pd.Timedelta(days=validation_days))
    validation_end = min(
        decision_end_for_closed_positions(validation_start + pd.Timedelta(days=validation_days), horizon),
        pd.Timestamp(fold["train_end"]),
    )
    test_start = pd.Timestamp(fold["test_start"])
    test_end = decision_end_for_closed_positions(pd.Timestamp(fold["test_end"]), horizon)
    return validation_start, validation_end, test_start, test_end


def selected_p18d_guard(p18d_result: dict[str, Any], fold_id: int) -> str:
    folds = p18d_result["folds"].copy()
    row = folds.loc[folds["fold"].astype(float).eq(float(fold_id))].iloc[0]
    guard = str(row["selected_guard_spec"])
    if guard == "__P15C_FALLBACK__":
        raise RuntimeError(f"Fold {fold_id} is not a P18d accepted fold.")
    return guard


def raw_score_panel(
    fold_group: pd.DataFrame,
    rank_matrices: dict[str, pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
    fold_id: int,
) -> pd.DataFrame:
    combined = None
    total_weight = 0.0
    for _, row in fold_group.iterrows():
        factor = str(row["factor"])
        if factor not in rank_matrices:
            continue
        weight = max(float(row["selection_score"]), 0.01)
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


def p18c_metrics_for_window(
    base_panel: pd.DataFrame,
    selected_factors: pd.DataFrame,
    rank_matrices: dict[str, pd.DataFrame],
    pf_cfg: Any,
    risk_template: Any,
    start: pd.Timestamp,
    end: pd.Timestamp,
    fold_id: int,
) -> dict[str, Any]:
    score = raw_score_panel(selected_factors, rank_matrices, start, end, fold_id)
    panel = attach_score(base_panel, score)
    spec = CommitteeSpec(
        name="p18c_raw_fold14",
        w_a07=0.0,
        w_b11=0.0,
        w_c09=0.0,
        long_fraction=0.50,
        short_guard="p18c_score_gate",
        short_resid_confirm_rank=0.35,
    )
    risk_cfg = risk_template.__class__(
        **{
            **risk_template.__dict__,
            "name": f"p19_p18c_raw_{fold_id}",
            "short_resid_confirm_rank": spec.short_resid_confirm_rank,
        }
    )
    equity, positions, metrics = run_p18c_backtest(panel, start, end, pf_cfg, risk_cfg, spec)
    enrich_metrics(equity, positions, metrics)
    return {"equity": equity, "positions": positions, "metrics": metrics}


def guarded_metrics_for_window(
    base_panel: pd.DataFrame,
    selected_factors: pd.DataFrame,
    rank_matrices: dict[str, pd.DataFrame],
    pf_cfg: Any,
    risk_template: Any,
    start: pd.Timestamp,
    end: pd.Timestamp,
    fold_id: int,
    spec: GuardSpec,
) -> dict[str, Any]:
    score = build_guarded_score_panel(selected_factors, rank_matrices, start, end, fold_id, spec)
    panel = attach_score(base_panel, score)
    risk_cfg = risk_template.__class__(
        **{
            **risk_template.__dict__,
            "name": f"p19_guarded_{fold_id}_{spec.name}",
            "short_resid_confirm_rank": spec.short_confirm_rank,
            "disagreement_scale": spec.disagreement_scale,
        }
    )
    equity, positions, metrics = run_guarded_backtest(panel, start, end, pf_cfg, risk_cfg, spec)
    enrich_metrics(equity, positions, metrics)
    return {"equity": equity, "positions": positions, "metrics": metrics}


def build_config_replay(
    base_panel: pd.DataFrame,
    selected_factors: pd.DataFrame,
    rank_matrices: dict[str, pd.DataFrame],
    p15c_result: dict[str, Any],
    p15c_validation: pd.DataFrame,
    p18c_result: dict[str, Any],
    p18d_result: dict[str, Any],
    p18d_validation: pd.DataFrame,
    pf_cfg: Any,
    risk_template: Any,
    validation_start: pd.Timestamp,
    validation_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    fold_id: int,
    selected_guard: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    p15c_val = p15c_validation.loc[p15c_validation["fold"].astype(int).eq(fold_id)].iloc[0]
    p15c_test = p15c_result["folds"].loc[p15c_result["folds"]["fold"].astype(float).eq(float(fold_id))].iloc[0]
    p18c_test = p18c_result["folds"].loc[p18c_result["folds"]["fold"].astype(float).eq(float(fold_id))].iloc[0]
    p18d_test = p18d_result["folds"].loc[p18d_result["folds"]["fold"].astype(float).eq(float(fold_id))].iloc[0]

    rows.append(metric_row("p15c_selected", "validation", "__P15C_REF__", p15c_validation_metrics(p15c_val), np.nan, np.nan, False, True))
    rows.append(metric_row("p15c_selected", "test", "__P15C_REF__", p15c_test.to_dict(), np.nan, np.nan, False, True))
    p18c_val = p18c_metrics_for_window(base_panel, selected_factors, rank_matrices, pf_cfg, risk_template, validation_start, validation_end, fold_id)
    rows.append(metric_row("p18c_raw", "validation", "p18c_raw", p18c_val["metrics"], np.nan, np.nan, False, False))
    rows.append(metric_row("p18c_raw", "test", "p18c_raw", p18c_test.to_dict(), np.nan, np.nan, False, False))
    rows.append(metric_row("p18d_selected", "test", selected_guard, p18d_test.to_dict(), np.nan, np.nan, False, False))

    p15c_ref = p15c_validation.loc[p15c_validation["fold"].astype(int).eq(fold_id)].iloc[0]
    validation_fold = p18d_validation.loc[p18d_validation["fold"].astype(int).eq(fold_id)].copy()
    for spec in build_guard_specs():
        val_row = validation_fold.loc[validation_fold["guard_spec"].eq(spec.name)]
        if not val_row.empty:
            val_metrics = unprefix_metrics(val_row.iloc[0], "validation_")
            val_hard = bool(val_row.iloc[0].get("hard_accept", False))
            val_score = float(val_row.iloc[0].get("selection_score", np.nan))
        else:
            val_metrics = guarded_metrics_for_window(
                base_panel, selected_factors, rank_matrices, pf_cfg, risk_template, validation_start, validation_end, fold_id, spec
            )["metrics"]
            val_hard = hard_accept(val_metrics, p15c_ref)
            val_score = selection_score(val_metrics, p15c_ref)
        test_metrics = guarded_metrics_for_window(
            base_panel, selected_factors, rank_matrices, pf_cfg, risk_template, test_start, test_end, fold_id, spec
        )["metrics"]
        rows.append(metric_row("p18d_guard_grid", "validation", spec.name, val_metrics, val_score, np.nan, val_hard, spec.name == selected_guard))
        rows.append(metric_row("p18d_guard_grid", "test", spec.name, test_metrics, np.nan, selection_score(test_metrics, p15c_ref), False, spec.name == selected_guard))
    out = pd.DataFrame(rows)
    pivot = out.pivot_table(index=["stage", "config"], columns="window", values="total_return", aggfunc="first")
    if {"validation", "test"}.issubset(pivot.columns):
        delta = (pivot["test"] - pivot["validation"]).rename("test_minus_validation_total_return")
        out = out.merge(delta.reset_index(), on=["stage", "config"], how="left")
    return out.sort_values(["window", "stage", "config"])


def metric_row(
    stage: str,
    window: str,
    config: str,
    metrics: dict[str, Any],
    validation_selection_score: float,
    test_selection_score: float,
    hard_accept_flag: bool,
    selected_flag: bool,
) -> dict[str, Any]:
    cols = [
        "periods",
        "total_return",
        "gross_total_return_before_cost",
        "sharpe",
        "max_drawdown",
        "gross_max_drawdown",
        "mean_turnover",
        "total_turnover",
        "cost_drag",
        "positive_period_rate",
        "positive_gross_period_rate",
        "mean_rank_ic",
        "mean_top_bottom_label_spread",
        "total_return_20bps",
        "sharpe_20bps",
        "max_drawdown_20bps",
        "total_return_30bps",
        "sharpe_30bps",
        "max_drawdown_30bps",
        "mean_high_risk_regime",
        "mean_regime_gross_scale",
    ]
    row = {
        "stage": stage,
        "window": window,
        "config": config,
        "validation_selection_score": validation_selection_score,
        "test_selection_score_vs_p15c_ref": test_selection_score,
        "hard_accept_on_validation": hard_accept_flag,
        "selected_by_p18d": selected_flag,
    }
    for col in cols:
        row[col] = metrics.get(col, np.nan)
    return row


def p15c_validation_metrics(row: pd.Series) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        if str(key).startswith("validation_"):
            out[str(key).replace("validation_", "", 1)] = value
    return out


def unprefix_metrics(row: pd.Series, prefix: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        key = str(key)
        if key.startswith(prefix):
            out[key.replace(prefix, "", 1)] = value
    return out


def build_regime_shift(
    base_panel: pd.DataFrame,
    derivatives_panel: Path,
    validation_start: pd.Timestamp,
    validation_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = base_panel.copy()
    panel["timestamp"] = pd.to_datetime(panel["timestamp"], utc=True)
    by_ts = panel.groupby("timestamp", sort=True)
    market = by_ts.agg(
        market_ret_1h=("log_ret", "mean"),
        residual_dispersion=("residual_log_ret", "std"),
        beta_btc_abs=("beta_btc", lambda x: float(np.nanmean(np.abs(x)))),
        beta_eth_abs=("beta_eth", lambda x: float(np.nanmean(np.abs(x)))),
        beta_market_abs=("beta_market", lambda x: float(np.nanmean(np.abs(x)))),
        gross_score_disp=("score_baseline", "std"),
        p18d_high_risk=("_p18d_high_risk_regime", "max"),
    )
    market["market_ret_24h"] = market["market_ret_1h"].rolling(24, min_periods=8).sum()
    market["market_ret_72h"] = market["market_ret_1h"].rolling(72, min_periods=24).sum()
    market["market_vol_14d"] = market["market_ret_1h"].rolling(336, min_periods=72).std(ddof=0)
    market["residual_disp_7d"] = market["residual_dispersion"].rolling(168, min_periods=48).mean()
    corr = rolling_mean_pairwise_corr(panel[["timestamp", "symbol", "residual_log_ret"]])
    market = market.join(corr.rename("cross_asset_resid_corr"), how="left")

    deriv = load_derivative_regime_series(derivatives_panel)
    market = market.join(deriv, how="left")
    market = market.reset_index()
    regime_cols = [c for c in market.columns if c != "timestamp"]
    ts = market.sort_values("timestamp")
    full_stats = ts[regime_cols].replace([np.inf, -np.inf], np.nan)
    rows: list[dict[str, Any]] = []
    for col in regime_cols:
        validation_values = ts.loc[(ts["timestamp"] >= validation_start) & (ts["timestamp"] < validation_end), col]
        test_values = ts.loc[(ts["timestamp"] >= test_start) & (ts["timestamp"] < test_end), col]
        baseline_values = full_stats[col].dropna()
        validation_mean = float(validation_values.mean()) if validation_values.notna().any() else np.nan
        test_mean = float(test_values.mean()) if test_values.notna().any() else np.nan
        percentile = float((baseline_values <= test_mean).mean()) if np.isfinite(test_mean) and len(baseline_values) else np.nan
        rows.append(
            {
                "feature": col,
                "validation_mean": validation_mean,
                "test_mean": test_mean,
                "test_minus_validation": test_mean - validation_mean if np.isfinite(test_mean) and np.isfinite(validation_mean) else np.nan,
                "test_percentile_full_sample": percentile,
                "validation_std": float(validation_values.std(ddof=0)) if validation_values.notna().any() else np.nan,
                "test_std": float(test_values.std(ddof=0)) if test_values.notna().any() else np.nan,
                "validation_nonnull": int(validation_values.notna().sum()),
                "test_nonnull": int(test_values.notna().sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("test_minus_validation", key=lambda s: s.abs(), ascending=False), ts


def rolling_mean_pairwise_corr(frame: pd.DataFrame, window: int = 168) -> pd.Series:
    wide = frame.pivot(index="timestamp", columns="symbol", values="residual_log_ret").sort_index()
    returns = []
    values = wide.to_numpy(dtype=float)
    idx = wide.index
    for i in range(len(idx)):
        start = max(0, i - window + 1)
        block = values[start : i + 1]
        if block.shape[0] < 48:
            returns.append(np.nan)
            continue
        corr = pd.DataFrame(block).corr().to_numpy(dtype=float)
        tri = corr[np.triu_indices_from(corr, k=1)]
        returns.append(float(np.nanmean(tri)) if np.isfinite(tri).any() else np.nan)
    return pd.Series(returns, index=idx)


def load_derivative_regime_series(path: Path) -> pd.DataFrame:
    cols = [
        "timestamp",
        "symbol",
        "deriv_funding_rate_ewm_168",
        "deriv_funding_rate_delta_24",
        "deriv_mark_index_basis_ewm_168",
        "deriv_mark_index_basis_delta_24",
        "deriv_premium_index_ewm_168",
        "deriv_premium_index_delta_24",
        "deriv_log_oi_value_delta_24",
        "deriv_log_oi_value_delta_168",
        "deriv_taker_long_short_ratio_ewm_24",
        "deriv_account_long_short_ratio_ewm_72",
        "deriv_top_sum_long_short_ratio_ewm_72",
        "deriv_crowding_pressure_72",
        "deriv_taker_pressure_24",
        "deriv_oi_growth_72_336",
        "deriv_abs_funding_stress_168",
        "deriv_abs_basis_stress_168",
    ]
    deriv = pd.read_csv(path, usecols=lambda c: c in cols, parse_dates=["timestamp"])
    deriv["timestamp"] = pd.to_datetime(deriv["timestamp"], utc=True)
    numeric_cols = [c for c in deriv.columns if c not in ID_COLS]
    grouped = deriv.groupby("timestamp", sort=True)
    parts = []
    for col in numeric_cols:
        parts.append(grouped[col].median().rename(f"{col}_median"))
        parts.append(grouped[col].std(ddof=0).rename(f"{col}_disp"))
    return pd.concat(parts, axis=1)


def build_factor_ic_diagnostics(
    selected_factors: pd.DataFrame,
    rank_matrices: dict[str, pd.DataFrame],
    label_rank: pd.DataFrame,
    validation_start: pd.Timestamp,
    validation_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    selected_spec: GuardSpec,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    ts_rows: list[dict[str, Any]] = []
    for _, row in selected_factors.iterrows():
        factor = str(row["factor"])
        if factor not in rank_matrices:
            continue
        oriented = rank_matrices[factor] * float(row["orientation"])
        val_ts = factor_ic_timeseries(oriented, label_rank, validation_start, validation_end)
        test_ts = factor_ic_timeseries(oriented, label_rank, test_start, test_end)
        val_spread = factor_spread_timeseries(oriented, label_rank, validation_start, validation_end)
        test_spread = factor_spread_timeseries(oriented, label_rank, test_start, test_end)
        for window, series in [("validation", val_ts), ("test", test_ts)]:
            for ts, value in series.items():
                ts_rows.append({"factor": factor, "raw_name": row["raw_name"], "window": window, "timestamp": ts, "ic": value})
        adjusted_weight = adjusted_weight_for_report(row, selected_spec)
        rows.append(
            {
                "factor": factor,
                "raw_name": row["raw_name"],
                "source": row["source"],
                "orientation": float(row["orientation"]),
                "selection_score": float(row["selection_score"]),
                "p18d_adjusted_weight": adjusted_weight,
                "weight_scale": adjusted_weight / max(float(row["selection_score"]), 0.01),
                "is_momentum_like": is_momentum_like(str(row["raw_name"])),
                "is_weak_or_decaying": is_weak_or_decaying(row),
                "original_validation_oriented_ic_mean": float(row["validation_oriented_ic_mean"]),
                "original_validation_ic_q25": float(row["validation_ic_q25"]),
                "original_validation_decay_delta": float(row["validation_decay_delta"]),
                "validation_ic_mean_recomputed": safe_mean(val_ts),
                "test_ic_mean": safe_mean(test_ts),
                "test_minus_validation_ic": safe_mean(test_ts) - safe_mean(val_ts),
                "validation_ic_q25_recomputed": safe_quantile(val_ts, 0.25),
                "test_ic_q25": safe_quantile(test_ts, 0.25),
                "validation_positive_rate": safe_positive_rate(val_ts),
                "test_positive_rate": safe_positive_rate(test_ts),
                "validation_top_bottom_spread": safe_mean(val_spread),
                "test_top_bottom_spread": safe_mean(test_spread),
                "spread_inverted": bool(np.isfinite(safe_mean(test_spread)) and safe_mean(test_spread) < 0.0),
            }
        )
    return (
        pd.DataFrame(rows).sort_values("test_minus_validation_ic"),
        pd.DataFrame(ts_rows).sort_values(["factor", "window", "timestamp"]),
    )


def factor_ic_timeseries(score: pd.DataFrame, label_rank: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    idx = score.loc[start:end].index
    values = []
    timestamps = []
    for ts in idx:
        if ts >= end:
            continue
        joined = pd.concat([score.loc[ts].rename("score"), label_rank.loc[ts].rename("label")], axis=1).dropna()
        if len(joined) < 8:
            continue
        values.append(float(joined["score"].corr(joined["label"], method="spearman")))
        timestamps.append(ts)
    return pd.Series(values, index=timestamps, dtype=float)


def factor_spread_timeseries(score: pd.DataFrame, label_rank: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    idx = score.loc[start:end].index
    values = []
    timestamps = []
    for ts in idx:
        if ts >= end:
            continue
        joined = pd.concat([score.loc[ts].rename("score"), label_rank.loc[ts].rename("label")], axis=1).dropna()
        if len(joined) < 8:
            continue
        q_long = joined["score"].quantile(0.80)
        q_short = joined["score"].quantile(0.20)
        spread = joined.loc[joined["score"] >= q_long, "label"].mean() - joined.loc[joined["score"] <= q_short, "label"].mean()
        values.append(float(spread))
        timestamps.append(ts)
    return pd.Series(values, index=timestamps, dtype=float)


def adjusted_weight_for_report(row: pd.Series, spec: GuardSpec) -> float:
    weight = max(float(row["selection_score"]), 0.01)
    if is_momentum_like(str(row["raw_name"])):
        weight *= spec.momentum_scale
    if is_weak_or_decaying(row):
        weight *= spec.weak_factor_scale
    return float(weight)


def is_momentum_like(name: str) -> bool:
    name = name.lower()
    return any(token in name for token in ["high_mom", "res_mom", "_mom", "momentum", "trend", "ma_slope", "uptrend"])


def is_weak_or_decaying(row: pd.Series) -> bool:
    return float(row.get("validation_decay_delta", 0.0)) < -0.01 or float(row.get("validation_ic_q25", 0.0)) < 0.0


def safe_mean(series: pd.Series) -> float:
    return float(series.mean()) if len(series) and series.notna().any() else np.nan


def safe_quantile(series: pd.Series, q: float) -> float:
    return float(series.quantile(q)) if len(series) and series.notna().any() else np.nan


def safe_positive_rate(series: pd.Series) -> float:
    return float((series > 0).mean()) if len(series) and series.notna().any() else np.nan


def build_position_diff(
    p18d_result: dict[str, Any],
    p15c_result: dict[str, Any],
    fold_id: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    p18d = p18d_result["positions"].loc[p18d_result["positions"]["fold"].astype(float).eq(float(fold_id))].copy()
    p15c = p15c_result["positions"].loc[p15c_result["positions"]["fold"].astype(float).eq(float(fold_id))].copy()
    keep = ["timestamp", "symbol", "weight", "contribution", "fwd_simple_ret", "label_resid_ret", "score"]
    left = p18d[[c for c in keep if c in p18d.columns]].rename(
        columns={c: f"{c}_p18d" for c in keep if c not in ["timestamp", "symbol"]}
    )
    right = p15c[[c for c in keep if c in p15c.columns]].rename(
        columns={c: f"{c}_p15c" for c in keep if c not in ["timestamp", "symbol"]}
    )
    merged = left.merge(right, on=["timestamp", "symbol"], how="outer")
    for col in ["weight", "contribution", "fwd_simple_ret", "label_resid_ret", "score"]:
        for suffix in ["p18d", "p15c"]:
            key = f"{col}_{suffix}"
            if key not in merged:
                merged[key] = np.nan
    merged["weight_p18d"] = merged["weight_p18d"].fillna(0.0)
    merged["weight_p15c"] = merged["weight_p15c"].fillna(0.0)
    merged["contribution_p18d"] = merged["contribution_p18d"].fillna(0.0)
    merged["contribution_p15c"] = merged["contribution_p15c"].fillna(0.0)
    merged["fwd_simple_ret"] = merged["fwd_simple_ret_p18d"].fillna(merged["fwd_simple_ret_p15c"])
    merged["delta_weight"] = merged["weight_p18d"] - merged["weight_p15c"]
    merged["delta_contribution"] = merged["contribution_p18d"] - merged["contribution_p15c"]
    merged["delta_weight_contribution"] = merged["delta_weight"] * merged["fwd_simple_ret"]
    merged["p18d_side"] = np.sign(merged["weight_p18d"])
    merged["p15c_side"] = np.sign(merged["weight_p15c"])
    merged["diff_category"] = np.select(
        [
            (merged["weight_p18d"] != 0) & (merged["weight_p15c"] == 0),
            (merged["weight_p18d"] == 0) & (merged["weight_p15c"] != 0),
            (merged["weight_p18d"] * merged["weight_p15c"] < 0),
            (merged["weight_p18d"] * merged["weight_p15c"] > 0) & (merged["delta_weight"].abs() > 1e-9),
        ],
        ["p18d_only", "p15c_only", "sign_flip", "resized_same_side"],
        default="same_or_zero",
    )
    symbol_delta = (
        merged.groupby("symbol", sort=False)
        .agg(
            delta_contribution=("delta_contribution", "sum"),
            delta_weight_contribution=("delta_weight_contribution", "sum"),
            p18d_contribution=("contribution_p18d", "sum"),
            p15c_contribution=("contribution_p15c", "sum"),
            avg_abs_delta_weight=("delta_weight", lambda x: float(np.nanmean(np.abs(x)))),
            max_abs_delta_weight=("delta_weight", lambda x: float(np.nanmax(np.abs(x)))),
            observations=("timestamp", "count"),
        )
        .reset_index()
        .sort_values("delta_contribution")
    )
    category_delta = (
        merged.groupby("diff_category", sort=False)
        .agg(
            delta_contribution=("delta_contribution", "sum"),
            delta_weight_contribution=("delta_weight_contribution", "sum"),
            abs_delta_weight=("delta_weight", lambda x: float(np.nansum(np.abs(x)))),
            rows=("timestamp", "count"),
        )
        .reset_index()
        .sort_values("delta_contribution")
    )
    timestamp_delta = (
        merged.groupby("timestamp", sort=True)
        .agg(
            delta_contribution=("delta_contribution", "sum"),
            p18d_contribution=("contribution_p18d", "sum"),
            p15c_contribution=("contribution_p15c", "sum"),
            gross_delta_turnover_proxy=("delta_weight", lambda x: float(np.nansum(np.abs(x)))),
        )
        .reset_index()
    )
    return merged.sort_values(["timestamp", "delta_contribution"]), symbol_delta, category_delta, timestamp_delta


def build_allocator_audit(
    base_panel: pd.DataFrame,
    selected_factors: pd.DataFrame,
    rank_matrices: dict[str, pd.DataFrame],
    pf_cfg: Any,
    risk_template: Any,
    selected_spec: GuardSpec,
    validation_start: pd.Timestamp,
    validation_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    fold_id: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts = []
    for window, start, end in [
        ("validation", validation_start, validation_end),
        ("test", test_start, test_end),
    ]:
        parts.append(
            allocator_snapshots_for_window(
                base_panel, selected_factors, rank_matrices, pf_cfg, risk_template, selected_spec, start, end, fold_id, window
            )
        )
    snap = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if snap.empty:
        return snap, pd.DataFrame()
    summary = (
        snap.groupby("window", sort=False)
        .agg(
            rows=("timestamp", "count"),
            mean_raw_gross=("raw_abs_weight", "sum"),
            mean_final_gross=("final_abs_weight", "sum"),
            projection_delta_contribution=("projection_delta_contribution", "sum"),
            raw_contribution=("raw_contribution", "sum"),
            final_contribution=("final_contribution", "sum"),
            mean_abs_weight_change=("weight_change", lambda x: float(np.nanmean(np.abs(x)))),
            max_abs_weight_change=("weight_change", lambda x: float(np.nanmax(np.abs(x)))),
        )
        .reset_index()
    )
    by_symbol = (
        snap.groupby(["window", "symbol"], sort=False)
        .agg(
            projection_delta_contribution=("projection_delta_contribution", "sum"),
            raw_contribution=("raw_contribution", "sum"),
            final_contribution=("final_contribution", "sum"),
            avg_abs_weight_change=("weight_change", lambda x: float(np.nanmean(np.abs(x)))),
        )
        .reset_index()
    )
    by_symbol.to_csv(DEFAULT_OUTPUT / "p19e_allocator_symbol_summary.csv", index=False)
    return snap, summary


def allocator_snapshots_for_window(
    base_panel: pd.DataFrame,
    selected_factors: pd.DataFrame,
    rank_matrices: dict[str, pd.DataFrame],
    pf_cfg: Any,
    risk_template: Any,
    selected_spec: GuardSpec,
    start: pd.Timestamp,
    end: pd.Timestamp,
    fold_id: int,
    window: str,
) -> pd.DataFrame:
    score = build_guarded_score_panel(selected_factors, rank_matrices, start, end, fold_id, selected_spec)
    panel = attach_score(base_panel, score)
    ret_col = f"fwd_simple_ret_h{pf_cfg.primary_horizon}"
    label_col = f"label_resid_ret_h{pf_cfg.primary_horizon}"
    symbols = sorted(panel["symbol"].unique())
    mask = (panel["timestamp"] >= start) & (panel["timestamp"] < end)
    timestamps = np.array(sorted(panel.loc[mask, "timestamp"].unique()))
    timestamps = timestamps[:: pf_cfg.rebalance_hours]
    current_state: dict[str, tuple[int, int]] = {}
    risk_cfg = risk_template.__class__(
        **{
            **risk_template.__dict__,
            "name": f"p19_alloc_{window}_{fold_id}_{selected_spec.name}",
            "short_resid_confirm_rank": selected_spec.short_confirm_rank,
            "disagreement_scale": selected_spec.disagreement_scale,
        }
    )
    committee = CommitteeSpec(
        name=selected_spec.name,
        w_a07=0.0,
        w_b11=0.0,
        w_c09=0.0,
        long_fraction=selected_spec.long_fraction,
        short_guard="p18d",
        short_resid_confirm_rank=selected_spec.short_confirm_rank,
    )
    out: list[dict[str, Any]] = []
    for ts in timestamps:
        rows = panel[panel["timestamp"].eq(ts)].copy()
        rows = rows[
            np.isfinite(rows[P18C_SCORE])
            & np.isfinite(rows[ret_col])
            & np.isfinite(rows[label_col])
            & np.isfinite(rows["score_ema_025"])
            & np.isfinite(rows["score_baseline"])
        ]
        if len(rows) < pf_cfg.long_k + pf_cfg.short_k:
            continue
        rows["score_rank"] = rows[P18C_SCORE].rank(pct=True, method="average")
        rows["p4_rank"] = rows["score_rank"]
        rows["ema_rank"] = rows["score_ema_025"].rank(pct=True, method="average")
        rows["base_rank"] = rows["score_baseline"].rank(pct=True, method="average")
        rows = rows.sort_values(P18C_SCORE, ascending=False)
        rows_by_symbol = rows.set_index("symbol")
        target_sides = select_p4_gated_target_sides(rows, current_state, pf_cfg, P18C_SCORE)
        if not any(side > 0 for side in target_sides.values()) or not any(side < 0 for side in target_sides.values()):
            continue
        selected = rows_by_symbol.reindex(list(target_sides)).reset_index()
        raw_weights = side_fraction_weights(target_sides, pf_cfg.gross_exposure, committee.long_fraction)
        target_weights, optimizer_info = build_weights(raw_weights, selected, rows_by_symbol, pf_cfg, risk_cfg)
        target_weights = target_weights[target_weights.abs() > 1e-12]
        high_risk = bool(rows["_p18d_high_risk_regime"].iloc[0])
        final_weights = target_weights * (selected_spec.regime_gross_scale if high_risk else 1.0)
        all_symbols = sorted(set(raw_weights.index) | set(final_weights.index))
        raw_full = pd.Series(0.0, index=symbols)
        opt_full = pd.Series(0.0, index=symbols)
        final_full = pd.Series(0.0, index=symbols)
        raw_full.loc[raw_weights.index] = raw_weights
        opt_full.loc[target_weights.index] = target_weights
        final_full.loc[final_weights.index] = final_weights
        raw_exp = portfolio_exposures(raw_weights, rows_by_symbol, pf_cfg.neutralize_exposures)
        opt_exp = portfolio_exposures(target_weights, rows_by_symbol, pf_cfg.neutralize_exposures)
        final_exp = portfolio_exposures(final_weights, rows_by_symbol, pf_cfg.neutralize_exposures)
        for symbol in all_symbols:
            fwd = float(rows_by_symbol.loc[symbol, ret_col])
            raw_w = float(raw_full.loc[symbol])
            opt_w = float(opt_full.loc[symbol])
            final_w = float(final_full.loc[symbol])
            out.append(
                {
                    "window": window,
                    "timestamp": pd.Timestamp(ts),
                    "symbol": symbol,
                    "score": float(rows_by_symbol.loc[symbol, P18C_SCORE]),
                    "fwd_simple_ret": fwd,
                    "raw_weight": raw_w,
                    "optimized_weight": opt_w,
                    "final_weight": final_w,
                    "raw_abs_weight": abs(raw_w),
                    "final_abs_weight": abs(final_w),
                    "weight_change": final_w - raw_w,
                    "raw_contribution": raw_w * fwd,
                    "optimized_contribution": opt_w * fwd,
                    "final_contribution": final_w * fwd,
                    "projection_delta_contribution": (final_w - raw_w) * fwd,
                    "high_risk_regime": high_risk,
                    "regime_gross_scale": selected_spec.regime_gross_scale if high_risk else 1.0,
                    "raw_abs_beta_btc": abs(raw_exp.get("beta_btc", np.nan)),
                    "raw_abs_beta_eth": abs(raw_exp.get("beta_eth", np.nan)),
                    "raw_abs_beta_market": abs(raw_exp.get("beta_market", np.nan)),
                    "opt_abs_beta_btc": abs(opt_exp.get("beta_btc", np.nan)),
                    "opt_abs_beta_eth": abs(opt_exp.get("beta_eth", np.nan)),
                    "opt_abs_beta_market": abs(opt_exp.get("beta_market", np.nan)),
                    "final_abs_beta_btc": abs(final_exp.get("beta_btc", np.nan)),
                    "final_abs_beta_eth": abs(final_exp.get("beta_eth", np.nan)),
                    "final_abs_beta_market": abs(final_exp.get("beta_market", np.nan)),
                    **{f"optimizer_{k}": v for k, v in optimizer_info.items()},
                }
            )
        current_state = update_state(current_state, target_sides, True)
    return pd.DataFrame(out)


def plot_outputs(output_dir: Path, replay: pd.DataFrame, regime_ts: pd.DataFrame, factor_ts: pd.DataFrame, timestamp_delta: pd.DataFrame) -> None:
    test = replay[replay["window"].eq("test")].copy()
    subset = test[test["stage"].isin(["p15c_selected", "p18c_raw", "p18d_guard_grid"])]
    subset = subset[(subset["stage"] != "p18d_guard_grid") | subset["selected_by_p18d"]]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(subset["stage"] + "\n" + subset["config"].astype(str).str.slice(0, 24), subset["total_return"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("Fold14 test return: P15c vs P18c raw vs selected P18d")
    fig.tight_layout()
    fig.savefig(output_dir / "p19a_fold14_test_return_bar.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    for ax, col in zip(axes, ["market_ret_72h", "market_vol_14d", "residual_disp_7d"]):
        if col in regime_ts:
            ax.plot(regime_ts["timestamp"], regime_ts[col], label=col)
            ax.axhline(0, color="black", linewidth=0.6)
            ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / "p19b_regime_timeseries.png", dpi=160)
    plt.close(fig)

    if not factor_ts.empty:
        pivot = factor_ts.pivot_table(index="timestamp", columns="raw_name", values="ic", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(13, 6))
        pivot.rolling(24, min_periods=4).mean().plot(ax=ax, legend=False, linewidth=1.0)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title("Fold14 selected factor IC rolling mean")
        fig.tight_layout()
        fig.savefig(output_dir / "p19c_factor_ic_rolling.png", dpi=160)
        plt.close(fig)

    if not timestamp_delta.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        ordered = timestamp_delta.sort_values("timestamp")
        ax.plot(pd.to_datetime(ordered["timestamp"]), ordered["delta_contribution"].cumsum(), label="cum P18d-P15c gross delta")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.legend()
        ax.set_title("Fold14 cumulative contribution delta")
        fig.tight_layout()
        fig.savefig(output_dir / "p19d_cumulative_delta.png", dpi=160)
        plt.close(fig)


def write_report(
    output_dir: Path,
    fold: pd.Series,
    selected_guard: str,
    replay: pd.DataFrame,
    regime: pd.DataFrame,
    factor_ic: pd.DataFrame,
    symbol_delta: pd.DataFrame,
    category_delta: pd.DataFrame,
    timestamp_delta: pd.DataFrame,
    alloc_summary: pd.DataFrame,
    validation_start: pd.Timestamp,
    validation_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> None:
    selected_test = replay[(replay["window"].eq("test")) & (replay["config"].eq(selected_guard))].head(1)
    p15c_test = replay[(replay["window"].eq("test")) & (replay["stage"].eq("p15c_selected"))].head(1)
    accepted_test = replay[(replay["window"].eq("test")) & (replay["stage"].eq("p18d_guard_grid"))].copy()
    accepted_test["beats_p15c"] = accepted_test["total_return"] > float(p15c_test["total_return"].iloc[0])
    top_test = accepted_test.sort_values("total_return", ascending=False).head(10)
    bottom_test = accepted_test.sort_values("total_return").head(10)
    report = [
        "# P19 Fold14 Mismatch Audit",
        "",
        "## Scope",
        "",
        f"- fold: {int(fold['fold'])}",
        f"- train_end: {fold['train_end']}",
        f"- validation window: {validation_start} to {validation_end}",
        f"- test window: {test_start} to {test_end}",
        f"- selected P18d guard: `{selected_guard}`",
        "",
        "## P19a Config Replay",
        "",
        "```text",
        replay[(replay["window"].eq("test")) & (replay["stage"].isin(["p15c_selected", "p18c_raw", "p18d_selected"]))]
        .sort_values(["stage", "config"])
        .to_string(index=False),
        "```",
        "",
        f"- guard configs beating P15c on fold14 test: {int(accepted_test['beats_p15c'].sum())}/{len(accepted_test)}",
        "",
        "### Top Test Guard Configs",
        "",
        "```text",
        top_test[["config", "total_return", "gross_total_return_before_cost", "sharpe", "max_drawdown", "mean_turnover", "selected_by_p18d"]].to_string(index=False),
        "```",
        "",
        "### Worst Test Guard Configs",
        "",
        "```text",
        bottom_test[["config", "total_return", "gross_total_return_before_cost", "sharpe", "max_drawdown", "mean_turnover", "selected_by_p18d"]].to_string(index=False),
        "```",
        "",
        "## P19b Largest Regime Shifts",
        "",
        "```text",
        regime.head(20).to_string(index=False),
        "```",
        "",
        "## P19c Factor IC Decay",
        "",
        "```text",
        factor_ic[[
            "raw_name",
            "source",
            "selection_score",
            "weight_scale",
            "is_momentum_like",
            "is_weak_or_decaying",
            "validation_ic_mean_recomputed",
            "test_ic_mean",
            "test_minus_validation_ic",
            "validation_top_bottom_spread",
            "test_top_bottom_spread",
            "spread_inverted",
        ]].to_string(index=False),
        "```",
        "",
        "## P19d Symbol Delta P18d minus P15c",
        "",
        "```text",
        symbol_delta.to_string(index=False),
        "```",
        "",
        "### Category Delta",
        "",
        "```text",
        category_delta.to_string(index=False),
        "```",
        "",
        "### Timestamp Delta",
        "",
        "```text",
        timestamp_delta.to_string(index=False),
        "```",
        "",
        "## P19e Allocator Projection Summary",
        "",
        "```text",
        alloc_summary.to_string(index=False),
        "```",
        "",
        "## Initial Read",
        "",
        initial_read(replay, factor_ic, symbol_delta, category_delta, alloc_summary),
    ]
    (output_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")


def initial_read(
    replay: pd.DataFrame,
    factor_ic: pd.DataFrame,
    symbol_delta: pd.DataFrame,
    category_delta: pd.DataFrame,
    alloc_summary: pd.DataFrame,
) -> str:
    p15c = replay[(replay["window"].eq("test")) & (replay["stage"].eq("p15c_selected"))].iloc[0]
    selected = replay[(replay["window"].eq("test")) & (replay["stage"].eq("p18d_selected"))].iloc[0]
    guard_test = replay[(replay["window"].eq("test")) & (replay["stage"].eq("p18d_guard_grid"))]
    beats = int((guard_test["total_return"] > float(p15c["total_return"])).sum())
    inverted = int(factor_ic["spread_inverted"].sum()) if "spread_inverted" in factor_ic else 0
    worst_symbol = symbol_delta.head(1)
    worst_cat = category_delta.head(1)
    lines = [
        f"- selected P18d test delta vs P15c: {float(selected['total_return']) - float(p15c['total_return']):.4%}.",
        f"- guard grid test configs beating P15c: {beats}/{len(guard_test)}.",
        f"- selected factor spread inversions in test: {inverted}/{len(factor_ic)}.",
    ]
    if not worst_symbol.empty:
        lines.append(
            f"- worst symbol delta: {worst_symbol.iloc[0]['symbol']} {float(worst_symbol.iloc[0]['delta_contribution']):.4%}."
        )
    if not worst_cat.empty:
        lines.append(
            f"- worst position-diff category: {worst_cat.iloc[0]['diff_category']} {float(worst_cat.iloc[0]['delta_contribution']):.4%}."
        )
    if not alloc_summary.empty:
        test_alloc = alloc_summary[alloc_summary["window"].eq("test")]
        if not test_alloc.empty:
            lines.append(
                f"- allocator projection delta contribution on test: {float(test_alloc.iloc[0]['projection_delta_contribution']):.4%}."
            )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
