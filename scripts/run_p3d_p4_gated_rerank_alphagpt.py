#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import warnings
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

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

from crypto_stat_arb.config import ensure_dir, load_config
from crypto_stat_arb.portfolio import portfolio_exposures, spearman_ic, update_state
from run_p0_baseline_lock_review import (
    build_asset_summary,
    build_cost_sensitivity,
    build_regime_frame,
    build_regime_summary,
    build_side_summary,
)
from run_p2_family_b_mean_reversion import build_baseline_comparison, load_baseline_summary, make_portfolio_config
from run_p3_committee_guard_scan import (
    COMPONENTS,
    CommitteeSpec,
    apply_short_guard,
    build_committee_specs,
    make_risk_template,
    run_committee_backtest,
    side_fraction_weights,
)
from run_p4_nested_walkforward_selection import decision_end_for_closed_positions, summarize_nested_result
from run_p5_alphagpt_slow_search import (
    DummyDelay,
    FormulaSpec,
    cost_adjusted_metrics,
    evaluate_node,
    generate_formula_specs,
    load_panel as load_slow_panel,
    q25_segment_sharpe,
)
from run_residual_alphagpt_search import aggregate_oos_metrics, prepare_alpha_panel
from run_risk_control_experiments import add_gross_metrics, add_risk_event_metrics, build_weights
from run_delay_smoothing_experiments import add_delay_metrics
from run_turnover_experiments import to_jsonable


EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "baseline_locked_v1"
DEFAULT_CONFIG = EXPERIMENT_ROOT / "configs" / "baseline_locked_v1.yaml"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "outputs" / "p3d_p4_gated_rerank_alphagpt"
MANUAL_FEATURE_PANEL = EXPERIMENT_ROOT / "outputs" / "p2_manual_slow_alpha_grid" / "manual_slow_feature_panel.csv.gz"
P4_SELECTED = EXPERIMENT_ROOT / "outputs" / "p4_nested_walkforward_selection" / "selected_specs_by_fold.csv"
P4_SUMMARY = EXPERIMENT_ROOT / "outputs" / "p4_nested_walkforward_selection" / "summary.csv"
P4_EQUITY = EXPERIMENT_ROOT / "outputs" / "p4_nested_walkforward_selection" / "nested_test_run" / "equity_curve.csv"

SCORE_COL = "_p3d_p4_gated_rerank_score"
P4_SCORE_COL = "_p3d_p4_score"
CANDIDATE_RANK_COL = "_p3d_candidate_rank"
EFFECTIVE_CANDIDATE_WEIGHT_COL = "_p3d_effective_candidate_weight"
HIGH_RISK_REGIME_COL = "_p3d_high_risk_regime"
P4_ONLY_FORMULA = "__P4_ONLY__"

SIDE_AWARE_GRID = [
    {"candidate_weight_long": 0.2, "candidate_weight_short": 0.2, "high_risk_candidate_cap": 0.1},
    {"candidate_weight_long": 0.3, "candidate_weight_short": 0.3, "high_risk_candidate_cap": 0.15},
    {"candidate_weight_long": 0.4, "candidate_weight_short": 0.4, "high_risk_candidate_cap": 0.2},
]

_WORKER_PANEL: pd.DataFrame | None = None
_WORKER_PF_CFG: Any = None
_WORKER_RISK_TEMPLATE: Any = None
_WORKER_FORMULAS: dict[str, FormulaSpec] = {}
_WORKER_SPECS: dict[str, CommitteeSpec] = {}
_WORKER_HORIZON: int = 72


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--horizon", type=int, default=72)
    parser.add_argument("--formulas", type=int, default=180)
    parser.add_argument("--seed", type=int, default=2417)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--inner-validation-days", type=int, default=90)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--chunksize", type=int, default=8)
    parser.add_argument("--cost-bps", type=float, nargs="+", default=[0, 4, 6, 8, 10, 12, 15, 20, 30])
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    output_dir = ensure_dir(args.output_dir)
    horizon = int(args.horizon)
    workers = max(1, int(args.workers))

    panel_path = resolve_config_path(config_path, cfg["inputs"]["panel"])
    selections_path = resolve_config_path(config_path, cfg["inputs"]["formula_selections"])
    selections = pd.read_csv(selections_path, parse_dates=["train_start", "train_end", "test_start", "test_end"])
    p4_selected = pd.read_csv(P4_SELECTED, parse_dates=["train_start", "train_end", "test_start", "test_end"])
    formulas = build_incremental_formula_universe(int(args.formulas), int(args.seed), int(args.max_depth))
    pd.DataFrame([item.to_row() for item in formulas]).to_csv(output_dir / "candidate_formulas.csv", index=False)

    tasks = build_validation_tasks(
        selections=selections,
        p4_selected=p4_selected,
        formulas=formulas,
        horizon=horizon,
        validation_days=int(args.inner_validation_days),
    )
    print(
        f"[1/9] P3d P4-gated rerank search: {len(selections)} folds x {len(formulas)} formulas x "
        f"{len(SIDE_AWARE_GRID)} rerank grids = {len(tasks)} validation tasks; workers={workers}",
        flush=True,
    )
    worker_args = (
        str(config_path),
        str(panel_path),
        str(MANUAL_FEATURE_PANEL),
        horizon,
        {item.name: item for item in formulas},
    )
    validation_rows: list[dict[str, Any]] = []
    if workers == 1:
        init_worker(*worker_args)
        for idx, task in enumerate(tasks, start=1):
            if idx == 1 or idx % 250 == 0 or idx == len(tasks):
                print(f"      validation task {idx}/{len(tasks)}", flush=True)
            validation_rows.append(run_validation_task(task))
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers, initializer=init_worker, initargs=worker_args) as pool:
            for idx, row in enumerate(pool.imap_unordered(run_validation_task, tasks, chunksize=max(1, int(args.chunksize))), start=1):
                validation_rows.append(row)
                if idx == 1 or idx % 250 == 0 or idx == len(tasks):
                    print(f"      validation task {idx}/{len(tasks)}", flush=True)

    validation = pd.DataFrame(validation_rows)

    print("[2/9] Running P4 validation baselines for hard-accept comparisons", flush=True)
    p4_validation = run_p4_validation_baselines(
        config_path=config_path,
        panel_path=panel_path,
        feature_path=MANUAL_FEATURE_PANEL,
        selections=selections,
        p4_selected=p4_selected,
        horizon=horizon,
    )
    p4_validation.to_csv(output_dir / "validation_p4_metrics.csv", index=False)
    validation = annotate_hard_accept(validation, p4_validation)
    validation.to_csv(output_dir / "validation_incremental_metrics.csv", index=False)

    print("[3/9] Selecting best accepted candidate per fold, otherwise falling back to P4", flush=True)
    selected = select_by_fold(validation, p4_validation, selections, p4_selected, horizon)
    selected.to_csv(output_dir / "selected_incremental_by_fold.csv", index=False)

    print("[4/9] Replaying selected P4-gated rerank formulas on test windows only", flush=True)
    panel = load_panel(config_path, panel_path, MANUAL_FEATURE_PANEL, horizon)
    pf_cfg = make_portfolio_config(cfg, horizon)
    risk_template = make_risk_template(cfg)
    spec_by_name = {spec.name: spec for spec in build_committee_specs()}
    formula_by_name = {item.name: item for item in formulas}
    result = replay_selected_tests(panel, selections, selected, pf_cfg, risk_template, spec_by_name, formula_by_name)
    write_run_result(ensure_dir(output_dir / "nested_incremental_run"), result)

    print("[5/9] Summarizing and comparing against P4", flush=True)
    metrics = summarize_nested_result(result, horizon)
    summary = pd.DataFrame([{**metrics, "run": "p3d_p4_gated_rerank_alphagpt", "horizon": horizon}])
    summary = add_reference_correlations(summary, result)
    summary.to_csv(output_dir / "summary.csv", index=False)
    baseline = load_baseline_summary(EXPERIMENT_ROOT / "baseline_lock_review" / "horizon_summary.csv")
    comparison = build_baseline_comparison(summary.copy(), baseline)
    comparison = add_p4_comparison(comparison)
    comparison.to_csv(output_dir / "baseline_comparison.csv", index=False)

    print("[6/9] Cost, regime, side, and asset diagnostics", flush=True)
    cost_summary = build_cost_sensitivity({"p3d_p4_gated_rerank_alphagpt": result}, [float(x) for x in args.cost_bps])
    cost_summary.to_csv(output_dir / "cost_sensitivity.csv", index=False)
    equity = result["equity"].sort_values("timestamp")
    positions = result["positions"].sort_values(["timestamp", "symbol"])
    regime_summary = build_regime_summary(equity, build_regime_frame(panel), horizon)
    side_summary = build_side_summary(positions)
    asset_summary = build_asset_summary(positions)
    regime_summary.to_csv(output_dir / "regime_summary.csv", index=False)
    side_summary.to_csv(output_dir / "side_contribution.csv", index=False)
    asset_summary.to_csv(output_dir / "asset_contribution.csv", index=False)

    print("[7/9] Writing plots", flush=True)
    plot_curves(output_dir, result, horizon)
    plot_cost_sensitivity(output_dir, cost_summary)
    plot_selected_counts(output_dir, selected)
    plot_validation_vs_test(output_dir, result["folds"])
    plot_asset_contribution(output_dir, asset_summary)

    print("[8/9] Writing report", flush=True)
    write_report(
        output_dir=output_dir,
        summary=summary,
        comparison=comparison,
        selected=selected,
        validation=validation,
        cost_summary=cost_summary,
        regime_summary=regime_summary,
        side_summary=side_summary,
        asset_summary=asset_summary,
        formula_count=len(formulas),
        workers=workers,
        horizon=horizon,
        validation_days=int(args.inner_validation_days),
    )
    update_config_stage(config_path)

    print("[9/9] Done", flush=True)
    print(f"Done: {output_dir}", flush=True)


def resolve_config_path(config_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def build_incremental_formula_universe(count: int, seed: int, max_depth: int) -> list[FormulaSpec]:
    raw = generate_formula_specs(count=max(count * 5, 360), seed=seed, max_depth=max_depth)
    filtered = [item for item in raw if formula_is_incremental_candidate(item)]
    seeds = [item for item in raw if item.source == "seed" and formula_is_directional(item)]
    merged: list[FormulaSpec] = []
    seen: set[str] = set()
    for item in seeds + filtered:
        if item.formula in seen:
            continue
        merged.append(item)
        seen.add(item.formula)
        if len(merged) >= count:
            break
    if len(merged) < min(30, count):
        raise RuntimeError(f"Too few incremental formulas after filtering: {len(merged)}")
    return merged


def formula_is_directional(item: FormulaSpec) -> bool:
    text = " ".join([item.formula, *item.primitives])
    tokens = [
        "res_mom",
        "res_z",
        "res_trend_quality",
        "res_high_mom",
        "res_ma_slope",
        "A07",
        "B11",
        "B12",
        "C09",
    ]
    return any(token in text for token in tokens)


def formula_is_incremental_candidate(item: FormulaSpec) -> bool:
    if not formula_is_directional(item):
        return False
    if "DIV_SAFE" in item.formula:
        return False
    if item.complexity > 14:
        return False
    non_directional = [
        "liquidity_quality",
        "stationarity_gate",
        "vol_of_vol",
        "beta_stability",
        "ou_speed",
        "regime_gate",
    ]
    text = " ".join(item.primitives)
    if all(token in text for token in item.primitives) and not formula_is_directional(item):
        return False
    pure_bad = (
        len(item.primitives) <= 2
        and not any(token in text for token in ["res_mom", "res_z", "res_trend_quality", "res_high_mom", "A07", "B11", "B12", "C09"])
        and any(token in text for token in non_directional)
    )
    return not pure_bad


def build_validation_tasks(
    selections: pd.DataFrame,
    p4_selected: pd.DataFrame,
    formulas: list[FormulaSpec],
    horizon: int,
    validation_days: int,
) -> list[dict[str, Any]]:
    p4_by_fold = p4_selected.set_index("fold")
    tasks = []
    for _, fold in selections.sort_values("fold").iterrows():
        fold_id = int(fold["fold"])
        p4 = p4_by_fold.loc[fold_id]
        train_start = pd.Timestamp(fold["train_start"])
        train_end = pd.Timestamp(fold["train_end"])
        validation_start = max(train_start + (train_end - train_start) / 2, train_end - pd.Timedelta(days=validation_days))
        validation_end = decision_end_for_closed_positions(train_end, horizon)
        for formula in formulas:
            for grid in SIDE_AWARE_GRID:
                tasks.append(
                    {
                        "fold": fold_id,
                        "formula_name": formula.name,
                        "p4_spec": str(p4["selected_spec"]),
                        "candidate_weight_long": float(grid["candidate_weight_long"]),
                        "candidate_weight_short": float(grid["candidate_weight_short"]),
                        "high_risk_candidate_cap": float(grid["high_risk_candidate_cap"]),
                        "validation_start": validation_start,
                        "validation_decision_end": validation_end,
                        "train_end": train_end,
                    }
                )
    return tasks


def init_worker(
    config_path_str: str,
    panel_path_str: str,
    feature_path_str: str,
    horizon: int,
    formulas: dict[str, FormulaSpec],
) -> None:
    global _WORKER_PANEL, _WORKER_PF_CFG, _WORKER_RISK_TEMPLATE, _WORKER_FORMULAS, _WORKER_SPECS, _WORKER_HORIZON
    config_path = Path(config_path_str)
    cfg = load_config(config_path)
    _WORKER_PANEL = load_panel(config_path, Path(panel_path_str), Path(feature_path_str), horizon)
    _WORKER_PANEL[SCORE_COL] = np.nan
    _WORKER_PANEL[P4_SCORE_COL] = np.nan
    _WORKER_PANEL[CANDIDATE_RANK_COL] = np.nan
    _WORKER_PANEL[EFFECTIVE_CANDIDATE_WEIGHT_COL] = np.nan
    _WORKER_PF_CFG = make_portfolio_config(cfg, horizon)
    _WORKER_RISK_TEMPLATE = make_risk_template(cfg)
    _WORKER_FORMULAS = formulas
    _WORKER_SPECS = {spec.name: spec for spec in build_committee_specs()}
    _WORKER_HORIZON = horizon


def load_panel(config_path: Path, panel_path: Path, feature_path: Path, horizon: int) -> pd.DataFrame:
    panel = load_slow_panel(config_path, panel_path, feature_path, horizon)
    for key, col in COMPONENTS.items():
        panel[f"rank_{key.lower()}"] = centered_rank_by_timestamp(panel, col)
    add_ex_ante_high_risk_regime(panel)
    return panel


def add_ex_ante_high_risk_regime(panel: pd.DataFrame) -> None:
    by_ts = (
        panel.groupby("timestamp", sort=True)
        .agg(
            market_vol_14d=("market_vol_14d", "mean"),
            residual_dispersion_7d=("residual_dispersion_7d", "mean"),
        )
        .sort_index()
    )
    high_flags = []
    for col in ["market_vol_14d", "residual_dispersion_7d"]:
        values = by_ts[col].astype(float)
        expanding_q = values.expanding(min_periods=168).quantile(2 / 3).shift(1)
        rolling_q = values.rolling(720, min_periods=168).quantile(2 / 3).shift(1)
        threshold = expanding_q.fillna(rolling_q)
        high_flags.append(values.gt(threshold).fillna(False))
    high_risk = high_flags[0] | high_flags[1]
    panel[HIGH_RISK_REGIME_COL] = panel["timestamp"].map(high_risk).fillna(False).astype(bool)


def centered_rank_by_timestamp(frame: pd.DataFrame, value_col: str) -> pd.Series:
    grouped = frame.groupby("timestamp", sort=False)[value_col]
    rank = grouped.rank(method="average")
    count = grouped.transform("count")
    centered = 2.0 * (rank - 1.0) / (count - 1.0) - 1.0
    return centered.where(count > 1)


def run_validation_task(task: dict[str, Any]) -> dict[str, Any]:
    if _WORKER_PANEL is None or _WORKER_PF_CFG is None or _WORKER_RISK_TEMPLATE is None:
        raise RuntimeError("Worker state was not initialized.")
    panel = _WORKER_PANEL
    formula = _WORKER_FORMULAS[str(task["formula_name"])]
    spec = _WORKER_SPECS[str(task["p4_spec"])]
    panel[P4_SCORE_COL] = p4_score(panel, spec)
    panel[CANDIDATE_RANK_COL] = evaluate_node(formula.node, panel).groupby(panel["timestamp"], sort=False).rank(pct=True, method="average")
    panel[CANDIDATE_RANK_COL] = 2.0 * panel[CANDIDATE_RANK_COL] - 1.0
    apply_constrained_incremental_score(
        panel,
        candidate_weight_long=float(task["candidate_weight_long"]),
        candidate_weight_short=float(task["candidate_weight_short"]),
        high_risk_candidate_cap=float(task["high_risk_candidate_cap"]),
    )
    risk_cfg = _WORKER_RISK_TEMPLATE.__class__(
        **{
            **_WORKER_RISK_TEMPLATE.__dict__,
            "name": f"p3d_{spec.name}",
            "short_resid_confirm_rank": spec.short_resid_confirm_rank,
        }
    )
    formula_text = (
        f"P4_gated_rerank({spec.name},{formula.name},"
        f"w={task['candidate_weight_long']:.2f},"
        f"high_cap={task['high_risk_candidate_cap']:.2f}): "
        f"{formula.formula}"
    )
    equity, positions, metrics = run_p4_gated_rerank_backtest(
        panel=panel,
        start=pd.Timestamp(task["validation_start"]),
        end=pd.Timestamp(task["validation_decision_end"]),
        cfg=_WORKER_PF_CFG,
        risk_cfg=risk_cfg,
        spec=spec,
        score_col=SCORE_COL,
        selected_formulas=formula_text,
    )
    enrich_incremental_metrics(equity, positions, metrics, formula, panel, task)
    return {
        "fold": int(task["fold"]),
        "formula_name": formula.name,
        "formula": formula.formula,
        "source": formula.source,
        "complexity": formula.complexity,
        "primitives": "|".join(formula.primitives),
        "p4_spec": spec.name,
        "candidate_weight_long": float(task["candidate_weight_long"]),
        "candidate_weight_short": float(task["candidate_weight_short"]),
        "high_risk_candidate_cap": float(task["high_risk_candidate_cap"]),
        "validation_start": pd.Timestamp(task["validation_start"]),
        "validation_decision_end": pd.Timestamp(task["validation_decision_end"]),
        "validation_fitness": incremental_fitness(metrics, formula.complexity),
        **{f"validation_{key}": value for key, value in metrics.items() if is_scalar(value)},
    }


def p4_score(panel: pd.DataFrame, spec: CommitteeSpec) -> pd.Series:
    return spec.w_a07 * panel["rank_a07"] + spec.w_b11 * panel["rank_b11"] + spec.w_c09 * panel["rank_c09"]


def apply_constrained_incremental_score(
    panel: pd.DataFrame,
    candidate_weight_long: float,
    candidate_weight_short: float,
    high_risk_candidate_cap: float,
) -> None:
    candidate = panel[CANDIDATE_RANK_COL].astype(float)
    p4 = panel[P4_SCORE_COL].astype(float)
    side_weight = np.where(candidate >= 0.0, candidate_weight_long, candidate_weight_short)
    high_risk = panel[HIGH_RISK_REGIME_COL].fillna(False).to_numpy(dtype=bool)
    effective_weight = np.where(high_risk, np.minimum(side_weight, high_risk_candidate_cap), side_weight)
    panel[EFFECTIVE_CANDIDATE_WEIGHT_COL] = effective_weight
    panel[SCORE_COL] = (1.0 - effective_weight) * p4 + effective_weight * candidate


def enrich_incremental_metrics(
    equity: pd.DataFrame,
    positions: pd.DataFrame,
    metrics: dict[str, float],
    formula: FormulaSpec,
    panel: pd.DataFrame,
    task: dict[str, Any],
) -> None:
    add_gross_metrics(equity, metrics)
    add_risk_event_metrics(positions, metrics)
    add_delay_metrics(equity, metrics)
    metrics.update(cost_adjusted_metrics(equity, _WORKER_HORIZON, [20.0, 30.0]))
    metrics["q25_segment_sharpe"] = q25_segment_sharpe(equity, _WORKER_HORIZON)
    metrics["long_short_balance"] = long_short_balance(positions)
    metrics["short_abs_contribution_share"] = short_abs_contribution_share(positions)
    mask = (panel["timestamp"] >= pd.Timestamp(task["validation_start"])) & (
        panel["timestamp"] < pd.Timestamp(task["validation_decision_end"])
    )
    valid = panel.loc[mask, [P4_SCORE_COL, CANDIDATE_RANK_COL]].replace([np.inf, -np.inf], np.nan).dropna()
    metrics["candidate_score_corr_to_p4"] = float(valid[P4_SCORE_COL].corr(valid[CANDIDATE_RANK_COL])) if len(valid) > 20 else 0.0
    scoped = panel.loc[mask]
    metrics["mean_effective_candidate_weight"] = float(scoped[EFFECTIVE_CANDIDATE_WEIGHT_COL].mean())
    metrics["high_risk_regime_rate"] = float(scoped.groupby("timestamp")[HIGH_RISK_REGIME_COL].first().mean())
    metrics["mean_short_guard_removed"] = float(equity["short_guard_removed"].mean()) if "short_guard_removed" in equity else np.nan


def incremental_fitness(metrics: dict[str, float], complexity: int) -> float:
    periods = float(metrics.get("periods", 0.0) or 0.0)
    if periods < 6:
        return -1000.0 + periods
    return (
        100.0 * float(metrics.get("total_return", 0.0) or 0.0)
        + 2.0 * float(metrics.get("sharpe_20bps", 0.0) or 0.0)
        + 1.5 * float(metrics.get("sharpe_30bps", 0.0) or 0.0)
        - 35.0 * abs(float(metrics.get("max_drawdown", 0.0) or 0.0))
        - 5.0 * float(metrics.get("mean_turnover", 1.0) or 1.0)
        + 1.2 * float(metrics.get("q25_segment_sharpe", 0.0) or 0.0)
        + 4.0 * float(metrics.get("long_short_balance", 0.0) or 0.0)
        - 8.0 * max(0.0, float(metrics.get("short_abs_contribution_share", 0.0) or 0.0) - 0.65)
        - 1.5 * abs(float(metrics.get("candidate_score_corr_to_p4", 0.0) or 0.0))
        - 0.10 * float(complexity)
    )


def long_short_balance(positions: pd.DataFrame) -> float:
    if positions.empty or "contribution" not in positions:
        return 0.0
    long_contrib = float(positions.loc[positions["weight"] > 0, "contribution"].sum())
    short_contrib = float(positions.loc[positions["weight"] < 0, "contribution"].sum())
    denom = abs(long_contrib) + abs(short_contrib)
    if denom <= 1e-12:
        return 0.0
    return float(1.0 - abs(abs(long_contrib) - abs(short_contrib)) / denom)


def short_abs_contribution_share(positions: pd.DataFrame) -> float:
    if positions.empty or "contribution" not in positions:
        return 0.0
    long_contrib = float(positions.loc[positions["weight"] > 0, "contribution"].sum())
    short_contrib = float(positions.loc[positions["weight"] < 0, "contribution"].sum())
    denom = abs(long_contrib) + abs(short_contrib)
    if denom <= 1e-12:
        return 0.0
    return float(abs(short_contrib) / denom)


def run_p4_gated_rerank_backtest(
    panel: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cfg: Any,
    risk_cfg: Any,
    spec: CommitteeSpec,
    score_col: str,
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

    for ts in timestamps:
        rows = panel[panel["timestamp"] == ts].copy()
        rows = rows[
            np.isfinite(rows[score_col])
            & np.isfinite(rows[P4_SCORE_COL])
            & np.isfinite(rows[ret_col])
            & np.isfinite(rows[label_col])
        ]
        if len(rows) < cfg.long_k + cfg.short_k:
            continue

        rows["score_rank"] = rows[score_col].rank(pct=True, method="average")
        rows["p4_rank"] = rows[P4_SCORE_COL].rank(pct=True, method="average")
        rows["ema_rank"] = rows["score_ema_025"].rank(pct=True, method="average")
        rows["base_rank"] = rows["score_baseline"].rank(pct=True, method="average")
        rows = rows.sort_values(score_col, ascending=False)
        rows_by_symbol = rows.set_index("symbol")

        target_sides = select_p4_gated_target_sides(rows, current_state, cfg, score_col)
        before_guard = len([side for side in target_sides.values() if side < 0])
        target_sides = apply_short_guard(target_sides, rows_by_symbol, spec)
        after_guard = len([side for side in target_sides.values() if side < 0])
        if not any(side > 0 for side in target_sides.values()) or not any(side < 0 for side in target_sides.values()):
            continue

        state_sides = target_sides.copy()
        selected = rows_by_symbol.reindex(list(target_sides)).reset_index()
        raw_weights = side_fraction_weights(target_sides, cfg.gross_exposure, spec.long_fraction)
        target_weights, optimizer_info = build_weights(raw_weights, selected, rows_by_symbol, cfg, risk_cfg)
        target_weights = target_weights[target_weights.abs() > 1e-12]
        if target_weights.abs().sum() <= 0:
            continue

        target_full_weights = pd.Series(0.0, index=symbols)
        target_full_weights.loc[target_weights.index] = target_weights
        turnover = float((target_full_weights - prev_weights).abs().sum())
        cost = turnover * (cfg.one_way_fee_bps + cfg.one_way_slippage_bps) / 10_000.0

        active_rows = rows_by_symbol.reindex(target_weights.index)
        gross_period_ret = float((target_weights * active_rows[ret_col]).sum())
        period_ret = gross_period_ret - cost
        equity *= 1.0 + period_ret

        rank_ic = spearman_ic(rows[score_col], rows[label_col])
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
                "short_guard_removed": float(before_guard - after_guard),
                **optimizer_info,
                "target_applied": 1.0,
                "selected_formulas": selected_formulas,
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
                    "score": float(rows_by_symbol.loc[symbol, score_col]),
                    "p4_score": float(rows_by_symbol.loc[symbol, P4_SCORE_COL]),
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
                    "selected_score_cols": "p4_score | candidate_rerank_score",
                }
            )

        prev_weights = target_full_weights
        current_state = update_state(current_state, state_sides, True)

    equity_curve = pd.DataFrame(rows_out)
    positions = pd.DataFrame(pos_out)
    metrics = aggregate_oos_metrics(equity_curve, cfg.rebalance_hours)
    return equity_curve, positions, metrics


def select_p4_gated_target_sides(
    rows: pd.DataFrame,
    current_state: dict[str, tuple[int, int]],
    cfg: Any,
    rerank_score_col: str,
) -> dict[str, int]:
    rows_by_symbol = rows.set_index("symbol")
    valid_symbols = set(rows_by_symbol.index)
    forced_longs = [
        symbol
        for symbol, (side, age) in current_state.items()
        if side > 0 and age < cfg.min_hold_rebalances and symbol in valid_symbols
    ]
    forced_shorts = [
        symbol
        for symbol, (side, age) in current_state.items()
        if side < 0 and age < cfg.min_hold_rebalances and symbol in valid_symbols
    ]

    longs = dedupe_keep_order(forced_longs)
    shorts = dedupe_keep_order(forced_shorts)

    if cfg.use_hysteresis:
        held_longs = [
            symbol
            for symbol, (side, _) in current_state.items()
            if side > 0
            and symbol in valid_symbols
            and rows_by_symbol.loc[symbol, "p4_rank"] >= cfg.long_exit_rank
            and symbol not in longs
        ]
        held_shorts = [
            symbol
            for symbol, (side, _) in current_state.items()
            if side < 0
            and symbol in valid_symbols
            and rows_by_symbol.loc[symbol, "p4_rank"] <= cfg.short_exit_rank
            and symbol not in shorts
        ]
        longs.extend(sort_symbols_by_score(held_longs, rows_by_symbol, rerank_score_col, ascending=False))
        shorts.extend(sort_symbols_by_score(held_shorts, rows_by_symbol, rerank_score_col, ascending=True))
        excluded = set(longs) | set(shorts)
        long_pool = rows[(rows["p4_rank"] >= cfg.long_entry_rank) & ~rows["symbol"].isin(excluded)].sort_values(
            rerank_score_col, ascending=False
        )
        short_pool = rows[(rows["p4_rank"] <= cfg.short_entry_rank) & ~rows["symbol"].isin(excluded)].sort_values(
            rerank_score_col, ascending=True
        )
    else:
        excluded = set(longs) | set(shorts)
        long_pool = rows[(rows["p4_rank"] >= cfg.long_entry_rank) & ~rows["symbol"].isin(excluded)].sort_values(
            rerank_score_col, ascending=False
        )
        short_pool = rows[(rows["p4_rank"] <= cfg.short_entry_rank) & ~rows["symbol"].isin(excluded)].sort_values(
            rerank_score_col, ascending=True
        )

    for symbol in long_pool["symbol"]:
        if len(longs) >= cfg.long_k:
            break
        if symbol not in shorts:
            longs.append(symbol)

    for symbol in short_pool["symbol"]:
        if len(shorts) >= cfg.short_k:
            break
        if symbol not in longs:
            shorts.append(symbol)

    longs = sort_symbols_by_score(longs, rows_by_symbol, rerank_score_col, ascending=False)[: cfg.long_k]
    shorts = sort_symbols_by_score(shorts, rows_by_symbol, rerank_score_col, ascending=True)[: cfg.short_k]
    sides = {symbol: 1 for symbol in longs}
    sides.update({symbol: -1 for symbol in shorts if symbol not in sides})
    return sides


def dedupe_keep_order(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item in seen:
            continue
        out.append(item)
        seen.add(item)
    return out


def sort_symbols_by_score(symbols: list[str], rows_by_symbol: pd.DataFrame, score_col: str, ascending: bool) -> list[str]:
    if not symbols:
        return []
    return list(rows_by_symbol.loc[symbols, score_col].sort_values(ascending=ascending).index)


def run_p4_validation_baselines(
    config_path: Path,
    panel_path: Path,
    feature_path: Path,
    selections: pd.DataFrame,
    p4_selected: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    cfg = load_config(config_path)
    panel = load_panel(config_path, panel_path, feature_path, horizon)
    pf_cfg = make_portfolio_config(cfg, horizon)
    risk_template = make_risk_template(cfg)
    spec_by_name = {spec.name: spec for spec in build_committee_specs()}
    p4_by_fold = p4_selected.set_index("fold")
    rows = []
    panel[P4_SCORE_COL] = np.nan
    for _, fold in selections.sort_values("fold").iterrows():
        fold_id = int(fold["fold"])
        p4 = p4_by_fold.loc[fold_id]
        spec = spec_by_name[str(p4["selected_spec"])]
        train_start = pd.Timestamp(fold["train_start"])
        train_end = pd.Timestamp(fold["train_end"])
        validation_start = max(train_start + (train_end - train_start) / 2, train_end - pd.Timedelta(days=90))
        validation_end = decision_end_for_closed_positions(train_end, horizon)
        panel[P4_SCORE_COL] = p4_score(panel, spec)
        risk_cfg = risk_template.__class__(
            **{
                **risk_template.__dict__,
                "name": f"p3d_p4_validation_{spec.name}",
                "short_resid_confirm_rank": spec.short_resid_confirm_rank,
            }
        )
        equity, positions, metrics = run_committee_backtest(
            panel=panel,
            start=validation_start,
            end=validation_end,
            cfg=pf_cfg,
            risk_cfg=risk_cfg,
            spec=spec,
            score_col=P4_SCORE_COL,
            selected_formulas=f"P4({spec.name})",
        )
        add_gross_metrics(equity, metrics)
        add_risk_event_metrics(positions, metrics)
        add_delay_metrics(equity, metrics)
        metrics.update(cost_adjusted_metrics(equity, horizon, [20.0, 30.0]))
        metrics["q25_segment_sharpe"] = q25_segment_sharpe(equity, horizon)
        metrics["long_short_balance"] = long_short_balance(positions)
        metrics["short_abs_contribution_share"] = short_abs_contribution_share(positions)
        rows.append(
            {
                "fold": fold_id,
                "p4_spec": spec.name,
                "validation_start": validation_start,
                "validation_decision_end": validation_end,
                **{f"p4_validation_{key}": value for key, value in metrics.items() if is_scalar(value)},
            }
        )
    return pd.DataFrame(rows)


def annotate_hard_accept(validation: pd.DataFrame, p4_validation: pd.DataFrame) -> pd.DataFrame:
    out = validation.merge(p4_validation.drop(columns=["p4_spec"], errors="ignore"), on="fold", how="left")
    p4_turnover_limit = np.maximum(
        out["p4_validation_mean_turnover"].astype(float) * 1.10,
        out["p4_validation_mean_turnover"].astype(float) + 0.015,
    )
    out["hard_accept"] = (
        (out["validation_total_return"] > out["p4_validation_total_return"])
        & (out["validation_sharpe"] > out["p4_validation_sharpe"])
        & (out["validation_total_return_20bps"] > out["p4_validation_total_return_20bps"])
        & (out["validation_max_drawdown"] >= 1.20 * out["p4_validation_max_drawdown"])
        & (out["validation_short_abs_contribution_share"] <= 0.70)
        & (out["validation_mean_turnover"] <= p4_turnover_limit)
    )
    return out


def select_by_fold(
    validation: pd.DataFrame,
    p4_validation: pd.DataFrame,
    selections: pd.DataFrame,
    p4_selected: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    p4_by_fold = p4_selected.set_index("fold")
    p4_validation_by_fold = p4_validation.set_index("fold")
    rows = []
    sorted_validation = validation.sort_values(
        ["fold", "validation_fitness", "validation_total_return", "validation_sharpe"],
        ascending=[True, False, False, False],
    )
    for _, fold in selections.sort_values("fold").iterrows():
        fold_id = int(fold["fold"])
        p4 = p4_by_fold.loc[fold_id]
        candidates = sorted_validation.loc[(sorted_validation["fold"] == fold_id) & (sorted_validation["hard_accept"])]
        if candidates.empty:
            p4_metrics = p4_validation_by_fold.loc[fold_id].to_dict()
            rows.append(
                {
                    "fold": fold_id,
                    "train_start": fold["train_start"],
                    "train_end": fold["train_end"],
                    "test_start": fold["test_start"],
                    "test_end": fold["test_end"],
                    "test_decision_end": decision_end_for_closed_positions(fold["test_end"], horizon),
                    "p4_spec": p4["selected_spec"],
                    "formula_name": P4_ONLY_FORMULA,
                    "formula": "P4 fallback: no candidate passed validation hard-accept rules",
                    "candidate_weight_long": 0.0,
                    "candidate_weight_short": 0.0,
                    "high_risk_candidate_cap": 0.0,
                    "validation_fitness": p4_metrics.get("p4_validation_total_return", np.nan),
                    "validation_total_return": p4_metrics.get("p4_validation_total_return", np.nan),
                    "validation_gross_return": p4_metrics.get("p4_validation_gross_total_return_before_cost", np.nan),
                    "validation_sharpe": p4_metrics.get("p4_validation_sharpe", np.nan),
                    "validation_max_drawdown": p4_metrics.get("p4_validation_max_drawdown", np.nan),
                    "validation_mean_turnover": p4_metrics.get("p4_validation_mean_turnover", np.nan),
                    "validation_short_abs_contribution_share": p4_metrics.get("p4_validation_short_abs_contribution_share", np.nan),
                    "validation_candidate_score_corr_to_p4": 1.0,
                    "hard_accept": False,
                    "fallback_to_p4": True,
                }
            )
            continue
        winner = candidates.iloc[0].to_dict()
        rows.append(
            {
                "fold": fold_id,
                "train_start": fold["train_start"],
                "train_end": fold["train_end"],
                "test_start": fold["test_start"],
                "test_end": fold["test_end"],
                "test_decision_end": decision_end_for_closed_positions(fold["test_end"], horizon),
                "p4_spec": p4["selected_spec"],
                "formula_name": winner["formula_name"],
                "formula": winner["formula"],
                "candidate_weight_long": winner["candidate_weight_long"],
                "candidate_weight_short": winner["candidate_weight_short"],
                "high_risk_candidate_cap": winner["high_risk_candidate_cap"],
                "validation_fitness": winner["validation_fitness"],
                "validation_total_return": winner.get("validation_total_return", np.nan),
                "validation_gross_return": winner.get("validation_gross_total_return_before_cost", np.nan),
                "validation_sharpe": winner.get("validation_sharpe", np.nan),
                "validation_max_drawdown": winner.get("validation_max_drawdown", np.nan),
                "validation_mean_turnover": winner.get("validation_mean_turnover", np.nan),
                "validation_short_abs_contribution_share": winner.get("validation_short_abs_contribution_share", np.nan),
                "validation_candidate_score_corr_to_p4": winner.get("validation_candidate_score_corr_to_p4", np.nan),
                "hard_accept": True,
                "fallback_to_p4": False,
            }
        )
    return pd.DataFrame(rows)


def replay_selected_tests(
    panel: pd.DataFrame,
    selections: pd.DataFrame,
    selected: pd.DataFrame,
    pf_cfg: Any,
    risk_template: Any,
    spec_by_name: dict[str, CommitteeSpec],
    formula_by_name: dict[str, FormulaSpec],
) -> dict[str, Any]:
    fold_rows = []
    equity_parts = []
    position_parts = []
    selected_by_fold = selected.set_index("fold")
    panel[SCORE_COL] = np.nan
    panel[P4_SCORE_COL] = np.nan
    panel[CANDIDATE_RANK_COL] = np.nan
    panel[EFFECTIVE_CANDIDATE_WEIGHT_COL] = np.nan
    for _, fold in selections.sort_values("fold").iterrows():
        fold_id = int(fold["fold"])
        row = selected_by_fold.loc[fold_id]
        spec = spec_by_name[str(row["p4_spec"])]
        panel[P4_SCORE_COL] = p4_score(panel, spec)
        if str(row["formula_name"]) == P4_ONLY_FORMULA:
            formula = None
            panel[SCORE_COL] = panel[P4_SCORE_COL]
            panel[EFFECTIVE_CANDIDATE_WEIGHT_COL] = 0.0
            formula_text = f"P4({spec.name}) fallback"
        else:
            formula = formula_by_name[str(row["formula_name"])]
            panel[CANDIDATE_RANK_COL] = evaluate_node(formula.node, panel).groupby(panel["timestamp"], sort=False).rank(pct=True, method="average")
            panel[CANDIDATE_RANK_COL] = 2.0 * panel[CANDIDATE_RANK_COL] - 1.0
            apply_constrained_incremental_score(
                panel,
                candidate_weight_long=float(row["candidate_weight_long"]),
                candidate_weight_short=float(row["candidate_weight_short"]),
                high_risk_candidate_cap=float(row["high_risk_candidate_cap"]),
            )
            formula_text = (
                f"P4_gated_rerank({spec.name},{row['formula_name']},"
                f"w={float(row['candidate_weight_long']):.2f},"
                f"high_cap={float(row['high_risk_candidate_cap']):.2f}): {formula.formula}"
            )
        risk_cfg = risk_template.__class__(
            **{
                **risk_template.__dict__,
                "name": f"p3d_{spec.name}",
                "short_resid_confirm_rank": spec.short_resid_confirm_rank,
            }
        )
        equity, positions, metrics = run_p4_gated_rerank_backtest(
            panel=panel,
            start=fold["test_start"],
            end=decision_end_for_closed_positions(fold["test_end"], pf_cfg.primary_horizon),
            cfg=pf_cfg,
            risk_cfg=risk_cfg,
            spec=spec,
            score_col=SCORE_COL,
            selected_formulas=formula_text,
        )
        add_gross_metrics(equity, metrics)
        add_risk_event_metrics(positions, metrics)
        add_delay_metrics(equity, metrics)
        metrics["long_short_balance"] = long_short_balance(positions)
        metrics["short_abs_contribution_share"] = short_abs_contribution_share(positions)
        metrics["mean_short_guard_removed"] = float(equity["short_guard_removed"].mean()) if "short_guard_removed" in equity else np.nan
        fold_rows.append(
            {
                "fold": fold_id,
                "train_start": fold["train_start"],
                "train_end": fold["train_end"],
                "test_start": fold["test_start"],
                "test_end": fold["test_end"],
                "test_decision_end": decision_end_for_closed_positions(fold["test_end"], pf_cfg.primary_horizon),
                "p4_spec": spec.name,
                "formula_name": str(row["formula_name"]),
                "formula": formula_text,
                "candidate_weight_long": float(row["candidate_weight_long"]),
                "candidate_weight_short": float(row["candidate_weight_short"]),
                "high_risk_candidate_cap": float(row["high_risk_candidate_cap"]),
                "hard_accept": bool(row.get("hard_accept", False)),
                "fallback_to_p4": bool(row.get("fallback_to_p4", False)),
                "validation_fitness": float(row["validation_fitness"]),
                **metrics,
            }
        )
        if not equity.empty:
            equity = equity.copy()
            equity["fold"] = fold_id
            equity["p4_spec"] = spec.name
            equity["formula_name"] = str(row["formula_name"])
            equity["formula"] = formula_text
            positions = positions.copy()
            positions["fold"] = fold_id
            positions["p4_spec"] = spec.name
            positions["formula_name"] = str(row["formula_name"])
            positions["formula"] = formula_text
            equity_parts.append(equity)
            position_parts.append(positions)
    equity_curve = pd.concat(equity_parts, ignore_index=True) if equity_parts else pd.DataFrame()
    positions = pd.concat(position_parts, ignore_index=True) if position_parts else pd.DataFrame()
    folds = pd.DataFrame(fold_rows)
    metrics = summarize_nested_result({"equity": equity_curve, "positions": positions, "folds": folds}, pf_cfg.rebalance_hours)
    return {"folds": folds, "equity": equity_curve, "positions": positions, "metrics": metrics}


def write_run_result(output_dir: Path, result: dict[str, Any]) -> None:
    result["folds"].to_csv(output_dir / "walk_forward_folds.csv", index=False)
    result["equity"].to_csv(output_dir / "equity_curve.csv", index=False)
    result["positions"].to_csv(output_dir / "positions.csv", index=False)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(to_jsonable(result["metrics"]), fh, indent=2, ensure_ascii=False)


def add_reference_correlations(summary: pd.DataFrame, result: dict[str, Any]) -> pd.DataFrame:
    equity = result["equity"].sort_values("timestamp")
    rows = [{"run": "p3d_p4_gated_rerank_alphagpt", "net_corr_to_p4_nested": corr_to_reference(equity, load_p4_equity())}]
    return summary.merge(pd.DataFrame(rows), on="run", how="left")


def corr_to_reference(equity: pd.DataFrame, reference: pd.DataFrame) -> float:
    if equity.empty or reference.empty:
        return np.nan
    merged = reference[["timestamp", "period_return"]].rename(columns={"period_return": "reference_return"}).merge(
        equity[["timestamp", "period_return"]].rename(columns={"period_return": "candidate_return"}),
        on="timestamp",
        how="inner",
    )
    if len(merged) < 5:
        return np.nan
    return float(merged["reference_return"].corr(merged["candidate_return"]))


def load_p4_equity() -> pd.DataFrame:
    if not P4_EQUITY.exists():
        return pd.DataFrame()
    return pd.read_csv(P4_EQUITY, parse_dates=["timestamp"]).sort_values("timestamp")


def add_p4_comparison(comparison: pd.DataFrame) -> pd.DataFrame:
    if not P4_SUMMARY.exists():
        return comparison
    p4 = pd.read_csv(P4_SUMMARY)
    if p4.empty:
        return comparison
    p4_row = p4.iloc[0].to_dict()
    out = comparison.copy()
    for col in [
        "total_return",
        "gross_total_return_before_cost",
        "sharpe",
        "max_drawdown",
        "gross_max_drawdown",
        "mean_turnover",
        "cost_drag",
    ]:
        if col in out and col in p4_row:
            out[f"p4_{col}"] = p4_row[col]
            out[f"delta_vs_p4_{col}"] = out[col] - float(p4_row[col])
    return out


def plot_curves(output_dir: Path, result: dict[str, Any], horizon: int) -> None:
    equity = result["equity"].sort_values("timestamp")
    if equity.empty:
        return
    p4 = load_p4_equity()
    fig, axes = plt.subplots(2, 1, figsize=(13, 8.5), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0]})
    gross = (1.0 + equity["gross_period_return"]).cumprod() - 1.0
    net = (1.0 + equity["period_return"]).cumprod() - 1.0
    axes[0].plot(equity["timestamp"], gross, label="P3d gross before cost", linewidth=2.0)
    axes[0].plot(equity["timestamp"], net, label="P3d net after 6bps", linewidth=2.0)
    if not p4.empty:
        p4_net = (1.0 + p4["period_return"]).cumprod() - 1.0
        axes[0].plot(p4["timestamp"], p4_net, label="P4 nested net", linewidth=1.6, alpha=0.75, color="#444444")
    axes[0].axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    axes[0].set_title(f"P3d P4-gated rerank AlphaGPT nested test ({horizon}h)")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)
    curve = (1.0 + equity["period_return"]).cumprod()
    dd = curve / curve.cummax() - 1.0
    axes[1].plot(equity["timestamp"], dd, label="P3d drawdown", linewidth=1.8)
    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "nested_incremental_curves.png", dpi=180)
    plt.close(fig)


def plot_cost_sensitivity(output_dir: Path, cost_summary: pd.DataFrame) -> None:
    if cost_summary.empty:
        return
    plt.figure(figsize=(12, 6.5))
    for name, group in cost_summary.groupby("experiment"):
        group = group.sort_values("one_way_cost_bps")
        plt.plot(group["one_way_cost_bps"], group["total_return"], marker="o", label=name)
    plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    plt.title("P3d cost sensitivity")
    plt.xlabel("one-way cost bps")
    plt.ylabel("Net cumulative return")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "cost_sensitivity.png", dpi=180)
    plt.close()


def plot_selected_counts(output_dir: Path, selected: pd.DataFrame) -> None:
    if selected.empty:
        return
    counts = selected["formula_name"].value_counts().head(18).sort_values()
    plt.figure(figsize=(11, 7))
    plt.barh(counts.index, counts.values, color="#2a9d8f")
    plt.title("P3d selected incremental formula counts")
    plt.xlabel("Fold count")
    plt.tight_layout()
    plt.savefig(output_dir / "selected_formula_counts.png", dpi=180)
    plt.close()
    mix = selected.groupby(["candidate_weight_long", "candidate_weight_short", "high_risk_candidate_cap"]).size().reset_index(name="fold_count")
    mix["label"] = mix.apply(
        lambda r: f"L{r['candidate_weight_long']:.1f}/S{r['candidate_weight_short']:.1f}/H{r['high_risk_candidate_cap']:.2f}",
        axis=1,
    )
    plt.figure(figsize=(8, 5))
    plt.bar(mix["label"], mix["fold_count"], color="#457b9d")
    plt.title("P3d selected rerank weights")
    plt.xlabel("candidate long/short/high-risk cap")
    plt.ylabel("Fold count")
    plt.tight_layout()
    plt.savefig(output_dir / "selected_side_regime_weights.png", dpi=180)
    plt.close()


def plot_validation_vs_test(output_dir: Path, folds: pd.DataFrame) -> None:
    if folds.empty:
        return
    plt.figure(figsize=(8, 6))
    plt.scatter(folds["validation_fitness"], folds["total_return"], alpha=0.8)
    plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    plt.title("P3d selected validation fitness vs test fold return")
    plt.xlabel("validation fitness")
    plt.ylabel("test fold return")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "validation_vs_test_return.png", dpi=180)
    plt.close()


def plot_asset_contribution(output_dir: Path, asset_summary: pd.DataFrame) -> None:
    if asset_summary.empty:
        return
    frame = asset_summary.sort_values("gross_contribution_sum")
    plt.figure(figsize=(11, 7))
    colors = np.where(frame["gross_contribution_sum"] >= 0.0, "#2a9d8f", "#c44536")
    plt.barh(frame["symbol"], frame["gross_contribution_sum"], color=colors)
    plt.axvline(0.0, color="black", linewidth=0.8, alpha=0.6)
    plt.title("P3d gross contribution by asset")
    plt.xlabel("Gross contribution sum")
    plt.gca().xaxis.set_major_formatter(PercentFormatter(1.0))
    plt.tight_layout()
    plt.savefig(output_dir / "asset_contribution.png", dpi=180)
    plt.close()


def write_report(
    output_dir: Path,
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    selected: pd.DataFrame,
    validation: pd.DataFrame,
    cost_summary: pd.DataFrame,
    regime_summary: pd.DataFrame,
    side_summary: pd.DataFrame,
    asset_summary: pd.DataFrame,
    formula_count: int,
    workers: int,
    horizon: int,
    validation_days: int,
) -> None:
    summary_cols = [
        "run",
        "horizon",
        "total_return",
        "gross_total_return_before_cost",
        "sharpe",
        "max_drawdown",
        "gross_max_drawdown",
        "mean_turnover",
        "cost_drag",
        "positive_fold_rate",
        "net_corr_to_p4_nested",
    ]
    summary_cols = [col for col in summary_cols if col in summary]
    comparison_cols = [
        "run",
        "total_return",
        "p4_total_return",
        "delta_vs_p4_total_return",
        "gross_total_return_before_cost",
        "p4_gross_total_return_before_cost",
        "delta_vs_p4_gross_total_return_before_cost",
        "sharpe",
        "p4_sharpe",
        "delta_vs_p4_sharpe",
        "max_drawdown",
        "p4_max_drawdown",
        "delta_vs_p4_max_drawdown",
        "mean_turnover",
        "p4_mean_turnover",
        "delta_vs_p4_mean_turnover",
    ]
    comparison_cols = [col for col in comparison_cols if col in comparison]
    selected_cols = [
        "fold",
        "p4_spec",
        "formula_name",
        "candidate_weight_long",
        "candidate_weight_short",
        "high_risk_candidate_cap",
        "hard_accept",
        "fallback_to_p4",
        "validation_fitness",
        "validation_total_return",
        "validation_short_abs_contribution_share",
        "validation_candidate_score_corr_to_p4",
        "formula",
    ]
    selected_cols = [col for col in selected_cols if col in selected]
    cost_display = cost_summary[cost_summary["one_way_cost_bps"].isin([0, 6, 10, 15, 20, 30])] if not cost_summary.empty else cost_summary
    top_validation = validation.sort_values("validation_fitness", ascending=False).head(25)
    top_validation_cols = [
        "fold",
        "formula_name",
        "candidate_weight_long",
        "candidate_weight_short",
        "high_risk_candidate_cap",
        "hard_accept",
        "validation_fitness",
        "validation_total_return",
        "validation_sharpe",
        "validation_short_abs_contribution_share",
        "validation_candidate_score_corr_to_p4",
    ]
    top_validation_cols = [col for col in top_validation_cols if col in top_validation]
    lines = [
        "# P3d P4-Gated Rerank AlphaGPT",
        "",
        "## Design",
        "",
        f"- Horizon fixed at `{horizon}h`.",
        f"- Fixed P4 baseline score per fold from `p4_nested_walkforward_selection/selected_specs_by_fold.csv`.",
        f"- Incremental candidate formulas: `{formula_count}` directional slow formulas.",
        "- Formula filter: requires residual-direction primitives and excludes `DIV_SAFE` formulas for this run.",
        "- P4 score defines the long-eligible and short-eligible pools; candidate can only rerank inside those pools.",
        "- Candidate rerank weights: `0.2/0.3/0.4` with equal long/short blend inside P4-approved side pools.",
        "- Regime-aware cap: high market-vol or high residual-dispersion timestamps cap candidate weight at `0.10/0.15/0.20`.",
        "- Hard accept on validation: net, Sharpe, 20bps return must beat P4; maxDD, turnover, and short-share must remain controlled.",
        f"- Each outer fold selects candidate+rerank weight using only the last `{validation_days}` days inside its prior training window.",
        "- If no candidate passes hard accept, that fold falls back to P4-only.",
        "- Test windows are clipped by `horizon + execution_delay`, same as P4.",
        f"- Validation backtests ran with `{workers}` worker processes.",
        "",
        "## Nested Test Summary",
        "",
        "```text",
        summary[summary_cols].to_string(index=False),
        "```",
        "",
        "## Comparison To P4",
        "",
        "```text",
        comparison[comparison_cols].to_string(index=False) if comparison_cols else comparison.to_string(index=False),
        "```",
        "",
        "## Selected Incremental Formulas",
        "",
        "```text",
        selected[selected_cols].to_string(index=False),
        "```",
        "",
        "## Top Validation Observations",
        "",
        "```text",
        top_validation[top_validation_cols].to_string(index=False),
        "```",
        "",
        "## Cost Sensitivity",
        "",
        "```text",
        cost_display.to_string(index=False),
        "```",
        "",
        "## Regime Summary",
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
        "## Files",
        "",
        "- `candidate_formulas.csv`: filtered incremental formula universe.",
        "- `validation_p4_metrics.csv`: P4 validation baseline metrics used by hard accept.",
        "- `validation_incremental_metrics.csv`: validation metrics for every fold/formula/rerank grid.",
        "- `selected_incremental_by_fold.csv`: selected candidate+grid per fold, including P4-only fallbacks.",
        "- `nested_incremental_run/`: stitched test-only run.",
        "- `summary.csv`, `baseline_comparison.csv`, `cost_sensitivity.csv`: headline tables.",
        "- `nested_incremental_curves.png`: P3d vs P4 curve.",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def update_config_stage(config_path: Path) -> None:
    text = config_path.read_text(encoding="utf-8")
    if "p3d_p4_gated_rerank_alphagpt:" in text:
        return
    marker = "committee_candidates:\n"
    insert = """\n  p3d_p4_gated_rerank_alphagpt:\n    script: ../../../scripts/run_p3d_p4_gated_rerank_alphagpt.py\n    output_dir: ../outputs/p3d_p4_gated_rerank_alphagpt\n    report: ../outputs/p3d_p4_gated_rerank_alphagpt/REPORT.md\n    candidate_formulas: ../outputs/p3d_p4_gated_rerank_alphagpt/candidate_formulas.csv\n    validation_p4_metrics: ../outputs/p3d_p4_gated_rerank_alphagpt/validation_p4_metrics.csv\n    validation_incremental_metrics: ../outputs/p3d_p4_gated_rerank_alphagpt/validation_incremental_metrics.csv\n    selected_incremental_by_fold: ../outputs/p3d_p4_gated_rerank_alphagpt/selected_incremental_by_fold.csv\n    summary: ../outputs/p3d_p4_gated_rerank_alphagpt/summary.csv\n    baseline_comparison: ../outputs/p3d_p4_gated_rerank_alphagpt/baseline_comparison.csv\n\n"""
    if marker in text:
        text = text.replace(marker, insert + marker, 1)
        config_path.write_text(text, encoding="utf-8")


def is_scalar(value: Any) -> bool:
    return isinstance(value, (int, float, str, bool, np.integer, np.floating)) or value is None


if __name__ == "__main__":
    main()
