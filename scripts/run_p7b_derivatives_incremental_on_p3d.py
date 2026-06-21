#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.ticker import PercentFormatter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from crypto_stat_arb.config import ensure_dir, load_config
import run_p3d_p4_gated_rerank_alphagpt as p3d
import run_p7_derivatives_gated_rerank_alphagpt as p7
from run_p5_alphagpt_slow_search import FormulaSpec, evaluate_node as evaluate_old_node
from run_turnover_experiments import to_jsonable


EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "baseline_locked_v1"
DEFAULT_CONFIG = EXPERIMENT_ROOT / "configs" / "baseline_locked_v1.yaml"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "outputs" / "p7b_derivatives_incremental_on_p3d"
P3D_SELECTED = EXPERIMENT_ROOT / "outputs" / "p3d_p4_gated_rerank_alphagpt" / "selected_incremental_by_fold.csv"
P3D_SUMMARY = EXPERIMENT_ROOT / "outputs" / "p3d_p4_gated_rerank_alphagpt" / "summary.csv"
P3D_EQUITY = EXPERIMENT_ROOT / "outputs" / "p3d_p4_gated_rerank_alphagpt" / "nested_incremental_run" / "equity_curve.csv"
BASE_SCORE_COL = "_p7b_p3d_base_score"
DERIV_CANDIDATE_COL = "_p7b_deriv_candidate_rank"

P7B_GRID = [
    {"candidate_weight_long": 0.05, "candidate_weight_short": 0.05, "high_risk_candidate_cap": 0.03},
    {"candidate_weight_long": 0.10, "candidate_weight_short": 0.10, "high_risk_candidate_cap": 0.05},
    {"candidate_weight_long": 0.20, "candidate_weight_short": 0.20, "high_risk_candidate_cap": 0.10},
    {"candidate_weight_long": 0.30, "candidate_weight_short": 0.30, "high_risk_candidate_cap": 0.15},
]

_PANEL: pd.DataFrame | None = None
_PF_CFG: Any = None
_RISK_TEMPLATE: Any = None
_DERIV_FORMULAS: dict[str, FormulaSpec] = {}
_P3D_FORMULAS: dict[str, FormulaSpec] = {}
_SPECS: dict[str, Any] = {}
_P3D_SELECTED: pd.DataFrame | None = None
_HORIZON: int = 72


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--horizon", type=int, default=72)
    parser.add_argument("--formulas", type=int, default=160)
    parser.add_argument("--seed", type=int, default=2607)
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
    panel_path = p3d.resolve_config_path(config_path, cfg["inputs"]["panel"])
    selections_path = p3d.resolve_config_path(config_path, cfg["inputs"]["formula_selections"])
    selections = pd.read_csv(selections_path, parse_dates=["train_start", "train_end", "test_start", "test_end"])
    p3d_selected = pd.read_csv(P3D_SELECTED, parse_dates=["train_start", "train_end", "test_start", "test_end", "test_decision_end"])
    retained = p7.load_retained_features()
    deriv_formulas = p7.build_p7_formula_universe(retained, int(args.formulas), int(args.seed), int(args.max_depth))
    p3d_formulas = p3d.build_incremental_formula_universe(180, 2417, 3)
    pd.DataFrame([item.to_row() for item in deriv_formulas]).to_csv(output_dir / "candidate_formulas.csv", index=False)

    tasks = build_validation_tasks(selections, p3d_selected, deriv_formulas, horizon, int(args.inner_validation_days))
    print(
        f"[1/9] P7b derivatives incremental-on-P3d: {len(selections)} folds x {len(deriv_formulas)} formulas x "
        f"{len(P7B_GRID)} grids = {len(tasks)} validation tasks; workers={workers}",
        flush=True,
    )
    worker_args = (
        str(config_path),
        str(panel_path),
        str(p3d.MANUAL_FEATURE_PANEL),
        horizon,
        {item.name: item for item in deriv_formulas},
        {item.name: item for item in p3d_formulas},
        p3d_selected,
    )
    rows = []
    if workers == 1:
        init_worker(*worker_args)
        for idx, task in enumerate(tasks, start=1):
            if idx == 1 or idx % 250 == 0 or idx == len(tasks):
                print(f"      validation task {idx}/{len(tasks)}", flush=True)
            rows.append(run_validation_task(task))
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers, initializer=init_worker, initargs=worker_args) as pool:
            for idx, row in enumerate(pool.imap_unordered(run_validation_task, tasks, chunksize=max(1, int(args.chunksize))), start=1):
                rows.append(row)
                if idx == 1 or idx % 250 == 0 or idx == len(tasks):
                    print(f"      validation task {idx}/{len(tasks)}", flush=True)
    validation = pd.DataFrame(rows)

    print("[2/9] Running P3d validation baselines for hard-accept comparisons", flush=True)
    p3d_validation = run_p3d_validation_baselines(config_path, panel_path, p3d.MANUAL_FEATURE_PANEL, selections, p3d_selected, p3d_formulas, horizon)
    p3d_validation.to_csv(output_dir / "validation_p3d_metrics.csv", index=False)
    validation = annotate_hard_accept(validation, p3d_validation)
    validation.to_csv(output_dir / "validation_incremental_metrics.csv", index=False)

    print("[3/9] Selecting best accepted derivative increment per fold, otherwise keeping P3d", flush=True)
    selected = select_by_fold(validation, p3d_validation, selections, p3d_selected, horizon)
    selected.to_csv(output_dir / "selected_incremental_by_fold.csv", index=False)

    print("[4/9] Replaying selected P7b increments on test windows only", flush=True)
    panel = p7.load_panel_p7(config_path, panel_path, p3d.MANUAL_FEATURE_PANEL, horizon)
    pf_cfg = p3d.make_portfolio_config(cfg, horizon)
    risk_template = p3d.make_risk_template(cfg)
    spec_by_name = {spec.name: spec for spec in p3d.build_committee_specs()}
    result = replay_selected_tests(panel, selections, selected, pf_cfg, risk_template, spec_by_name, {item.name: item for item in deriv_formulas}, {item.name: item for item in p3d_formulas})
    p3d.write_run_result(ensure_dir(output_dir / "nested_incremental_run"), result)

    print("[5/9] Summarizing and comparing against P3d", flush=True)
    metrics = p3d.summarize_nested_result(result, horizon)
    summary = pd.DataFrame([{**metrics, "run": "p7b_derivatives_incremental_on_p3d", "horizon": horizon}])
    summary["net_corr_to_p3d"] = p7.corr_to_reference(result["equity"], p7.load_reference_equity(P3D_EQUITY))
    summary["net_corr_to_p4_nested"] = p7.corr_to_reference(result["equity"], p3d.load_p4_equity())
    summary.to_csv(output_dir / "summary.csv", index=False)
    comparison = p7.add_p3d_comparison(summary.copy())
    comparison.to_csv(output_dir / "baseline_comparison.csv", index=False)

    print("[6/9] Cost, regime, side, and asset diagnostics", flush=True)
    cost_summary = p3d.build_cost_sensitivity({"p7b_derivatives_incremental_on_p3d": result}, [float(x) for x in args.cost_bps])
    cost_summary = p7.add_p3d_cost_comparison(cost_summary)
    cost_summary.to_csv(output_dir / "cost_sensitivity.csv", index=False)
    equity = result["equity"].sort_values("timestamp")
    positions = result["positions"].sort_values(["timestamp", "symbol"])
    regime_summary = p3d.build_regime_summary(equity, p3d.build_regime_frame(panel), horizon)
    side_summary = p3d.build_side_summary(positions)
    asset_summary = p3d.build_asset_summary(positions)
    regime_summary.to_csv(output_dir / "regime_summary.csv", index=False)
    side_summary.to_csv(output_dir / "side_contribution.csv", index=False)
    asset_summary.to_csv(output_dir / "asset_contribution.csv", index=False)
    usage_selected = selected.rename(columns={"deriv_formula_name": "formula_name"}).copy()
    usage_selected["fallback_to_p4"] = usage_selected["fallback_to_p3d"]
    primitive_usage = p7.build_primitive_usage(usage_selected, deriv_formulas, retained)
    primitive_usage.to_csv(output_dir / "primitive_usage.csv", index=False)

    print("[7/9] Writing plots", flush=True)
    p7.plot_curves(output_dir, result, horizon)
    p3d.plot_cost_sensitivity(output_dir, cost_summary)
    p3d.plot_selected_counts(output_dir, selected.rename(columns={"deriv_formula_name": "formula_name"}))
    p3d.plot_validation_vs_test(output_dir, result["folds"])
    p3d.plot_asset_contribution(output_dir, asset_summary)
    p7.plot_primitive_usage(output_dir, primitive_usage)

    print("[8/9] Writing report", flush=True)
    write_report(output_dir, summary, comparison, selected, validation, cost_summary, side_summary, asset_summary, primitive_usage, len(deriv_formulas), workers, horizon)
    update_config_stage(config_path)
    print("[9/9] Done", flush=True)
    print(f"Done: {output_dir}", flush=True)


def init_worker(
    config_path_str: str,
    panel_path_str: str,
    feature_path_str: str,
    horizon: int,
    deriv_formulas: dict[str, FormulaSpec],
    p3d_formulas: dict[str, FormulaSpec],
    p3d_selected: pd.DataFrame,
) -> None:
    global _PANEL, _PF_CFG, _RISK_TEMPLATE, _DERIV_FORMULAS, _P3D_FORMULAS, _SPECS, _P3D_SELECTED, _HORIZON
    config_path = Path(config_path_str)
    cfg = load_config(config_path)
    _PANEL = p7.load_panel_p7(config_path, Path(panel_path_str), Path(feature_path_str), horizon)
    for col in [p3d.SCORE_COL, p3d.P4_SCORE_COL, p3d.CANDIDATE_RANK_COL, p3d.EFFECTIVE_CANDIDATE_WEIGHT_COL, BASE_SCORE_COL, DERIV_CANDIDATE_COL]:
        _PANEL[col] = np.nan
    _PF_CFG = p3d.make_portfolio_config(cfg, horizon)
    _RISK_TEMPLATE = p3d.make_risk_template(cfg)
    _DERIV_FORMULAS = deriv_formulas
    _P3D_FORMULAS = p3d_formulas
    _SPECS = {spec.name: spec for spec in p3d.build_committee_specs()}
    _P3D_SELECTED = p3d_selected.set_index("fold")
    _HORIZON = horizon


def build_validation_tasks(
    selections: pd.DataFrame,
    p3d_selected: pd.DataFrame,
    formulas: list[FormulaSpec],
    horizon: int,
    validation_days: int,
) -> list[dict[str, Any]]:
    tasks = []
    by_fold = p3d_selected.set_index("fold")
    for _, fold in selections.sort_values("fold").iterrows():
        fold_id = int(fold["fold"])
        base = by_fold.loc[fold_id]
        train_start = pd.Timestamp(fold["train_start"])
        train_end = pd.Timestamp(fold["train_end"])
        validation_start = max(train_start + (train_end - train_start) / 2, train_end - pd.Timedelta(days=validation_days))
        validation_end = p3d.decision_end_for_closed_positions(train_end, horizon)
        for formula in formulas:
            for grid in P7B_GRID:
                tasks.append(
                    {
                        "fold": fold_id,
                        "deriv_formula_name": formula.name,
                        "p3d_formula_name": str(base["formula_name"]),
                        "p4_spec": str(base["p4_spec"]),
                        "candidate_weight_long": float(grid["candidate_weight_long"]),
                        "candidate_weight_short": float(grid["candidate_weight_short"]),
                        "high_risk_candidate_cap": float(grid["high_risk_candidate_cap"]),
                        "validation_start": validation_start,
                        "validation_decision_end": validation_end,
                        "train_end": train_end,
                    }
                )
    return tasks


def compute_base_score(panel: pd.DataFrame, base_row: pd.Series, spec: Any, p3d_formulas: dict[str, FormulaSpec]) -> None:
    panel[p3d.P4_SCORE_COL] = p3d.p4_score(panel, spec)
    base_formula_name = str(base_row.get("formula_name", base_row.get("p3d_formula_name")))
    if base_formula_name == p3d.P4_ONLY_FORMULA:
        panel[BASE_SCORE_COL] = panel[p3d.P4_SCORE_COL]
        return
    formula = p3d_formulas[base_formula_name]
    base_rank = evaluate_old_node(formula.node, panel).groupby(panel["timestamp"], sort=False).rank(pct=True, method="average")
    panel[p3d.CANDIDATE_RANK_COL] = 2.0 * base_rank - 1.0
    p3d_weight_long = float(base_row.get("p3d_candidate_weight_long", base_row.get("candidate_weight_long")))
    p3d_weight_short = float(base_row.get("p3d_candidate_weight_short", base_row.get("candidate_weight_short")))
    p3d_high_risk_cap = float(base_row.get("p3d_high_risk_candidate_cap", base_row.get("high_risk_candidate_cap")))
    p3d.apply_constrained_incremental_score(
        panel,
        candidate_weight_long=p3d_weight_long,
        candidate_weight_short=p3d_weight_short,
        high_risk_candidate_cap=p3d_high_risk_cap,
    )
    panel[BASE_SCORE_COL] = panel[p3d.SCORE_COL]


def apply_derivative_increment(panel: pd.DataFrame, formula: FormulaSpec, candidate_weight_long: float, candidate_weight_short: float, high_risk_candidate_cap: float) -> None:
    deriv_rank = p7.evaluate_node_p7(formula.node, panel).groupby(panel["timestamp"], sort=False).rank(pct=True, method="average")
    deriv_rank = 2.0 * deriv_rank - 1.0
    panel[DERIV_CANDIDATE_COL] = deriv_rank
    panel[p3d.CANDIDATE_RANK_COL] = deriv_rank
    side_weight = np.where(deriv_rank >= 0.0, candidate_weight_long, candidate_weight_short)
    high_risk = panel[p3d.HIGH_RISK_REGIME_COL].fillna(False).to_numpy(dtype=bool)
    effective = np.where(high_risk, np.minimum(side_weight, high_risk_candidate_cap), side_weight)
    panel[p3d.EFFECTIVE_CANDIDATE_WEIGHT_COL] = effective
    panel[p3d.SCORE_COL] = (1.0 - effective) * panel[BASE_SCORE_COL].astype(float) + effective * deriv_rank.astype(float)


def run_validation_task(task: dict[str, Any]) -> dict[str, Any]:
    if _PANEL is None or _PF_CFG is None or _RISK_TEMPLATE is None or _P3D_SELECTED is None:
        raise RuntimeError("Worker not initialized")
    panel = _PANEL
    spec = _SPECS[str(task["p4_spec"])]
    base_row = _P3D_SELECTED.loc[int(task["fold"])]
    deriv_formula = _DERIV_FORMULAS[str(task["deriv_formula_name"])]
    compute_base_score(panel, base_row, spec, _P3D_FORMULAS)
    apply_derivative_increment(
        panel,
        deriv_formula,
        float(task["candidate_weight_long"]),
        float(task["candidate_weight_short"]),
        float(task["high_risk_candidate_cap"]),
    )
    risk_cfg = _RISK_TEMPLATE.__class__(
        **{
            **_RISK_TEMPLATE.__dict__,
            "name": f"p7b_{spec.name}",
            "short_resid_confirm_rank": spec.short_resid_confirm_rank,
        }
    )
    formula_text = f"P3d_base + deriv({deriv_formula.name},w={task['candidate_weight_long']:.2f}): {deriv_formula.formula}"
    equity, positions, metrics = p3d.run_p4_gated_rerank_backtest(
        panel=panel,
        start=pd.Timestamp(task["validation_start"]),
        end=pd.Timestamp(task["validation_decision_end"]),
        cfg=_PF_CFG,
        risk_cfg=risk_cfg,
        spec=spec,
        score_col=p3d.SCORE_COL,
        selected_formulas=formula_text,
    )
    enrich_metrics(equity, positions, metrics, deriv_formula, panel, task)
    return {
        "fold": int(task["fold"]),
        "p4_spec": spec.name,
        "p3d_formula_name": str(task["p3d_formula_name"]),
        "deriv_formula_name": deriv_formula.name,
        "deriv_formula": deriv_formula.formula,
        "source": deriv_formula.source,
        "complexity": deriv_formula.complexity,
        "primitives": "|".join(deriv_formula.primitives),
        "candidate_weight_long": float(task["candidate_weight_long"]),
        "candidate_weight_short": float(task["candidate_weight_short"]),
        "high_risk_candidate_cap": float(task["high_risk_candidate_cap"]),
        "validation_start": pd.Timestamp(task["validation_start"]),
        "validation_decision_end": pd.Timestamp(task["validation_decision_end"]),
        "validation_fitness": p3d.incremental_fitness(metrics, deriv_formula.complexity),
        **{f"validation_{key}": value for key, value in metrics.items() if p3d.is_scalar(value)},
    }


def enrich_metrics(equity: pd.DataFrame, positions: pd.DataFrame, metrics: dict[str, float], formula: FormulaSpec, panel: pd.DataFrame, task: dict[str, Any]) -> None:
    p3d.add_gross_metrics(equity, metrics)
    p3d.add_risk_event_metrics(positions, metrics)
    p3d.add_delay_metrics(equity, metrics)
    metrics.update(p3d.cost_adjusted_metrics(equity, _HORIZON, [20.0, 30.0]))
    metrics["q25_segment_sharpe"] = p3d.q25_segment_sharpe(equity, _HORIZON)
    metrics["long_short_balance"] = p3d.long_short_balance(positions)
    metrics["short_abs_contribution_share"] = p3d.short_abs_contribution_share(positions)
    mask = (panel["timestamp"] >= pd.Timestamp(task["validation_start"])) & (panel["timestamp"] < pd.Timestamp(task["validation_decision_end"]))
    valid = panel.loc[mask, [BASE_SCORE_COL, DERIV_CANDIDATE_COL]].replace([np.inf, -np.inf], np.nan).dropna()
    metrics["candidate_score_corr_to_p3d"] = float(valid[BASE_SCORE_COL].corr(valid[DERIV_CANDIDATE_COL])) if len(valid) > 20 else 0.0
    metrics["mean_effective_candidate_weight"] = float(panel.loc[mask, p3d.EFFECTIVE_CANDIDATE_WEIGHT_COL].mean())
    metrics["high_risk_regime_rate"] = float(panel.loc[mask].groupby("timestamp")[p3d.HIGH_RISK_REGIME_COL].first().mean())
    metrics["mean_short_guard_removed"] = float(equity["short_guard_removed"].mean()) if "short_guard_removed" in equity else np.nan


def run_p3d_validation_baselines(
    config_path: Path,
    panel_path: Path,
    feature_path: Path,
    selections: pd.DataFrame,
    p3d_selected: pd.DataFrame,
    p3d_formulas: list[FormulaSpec],
    horizon: int,
) -> pd.DataFrame:
    cfg = load_config(config_path)
    panel = p7.load_panel_p7(config_path, panel_path, feature_path, horizon)
    pf_cfg = p3d.make_portfolio_config(cfg, horizon)
    risk_template = p3d.make_risk_template(cfg)
    spec_by_name = {spec.name: spec for spec in p3d.build_committee_specs()}
    formula_by_name = {item.name: item for item in p3d_formulas}
    base_by_fold = p3d_selected.set_index("fold")
    rows = []
    for _, fold in selections.sort_values("fold").iterrows():
        fold_id = int(fold["fold"])
        base = base_by_fold.loc[fold_id]
        spec = spec_by_name[str(base["p4_spec"])]
        train_start = pd.Timestamp(fold["train_start"])
        train_end = pd.Timestamp(fold["train_end"])
        validation_start = max(train_start + (train_end - train_start) / 2, train_end - pd.Timedelta(days=90))
        validation_end = p3d.decision_end_for_closed_positions(train_end, horizon)
        compute_base_score(panel, base, spec, formula_by_name)
        panel[p3d.SCORE_COL] = panel[BASE_SCORE_COL]
        risk_cfg = risk_template.__class__(
            **{
                **risk_template.__dict__,
                "name": f"p7b_p3d_validation_{spec.name}",
                "short_resid_confirm_rank": spec.short_resid_confirm_rank,
            }
        )
        equity, positions, metrics = p3d.run_p4_gated_rerank_backtest(
            panel=panel,
            start=validation_start,
            end=validation_end,
            cfg=pf_cfg,
            risk_cfg=risk_cfg,
            spec=spec,
            score_col=p3d.SCORE_COL,
            selected_formulas=f"P3d_base({base['formula_name']})",
        )
        p3d.add_gross_metrics(equity, metrics)
        p3d.add_risk_event_metrics(positions, metrics)
        p3d.add_delay_metrics(equity, metrics)
        metrics.update(p3d.cost_adjusted_metrics(equity, horizon, [20.0, 30.0]))
        metrics["q25_segment_sharpe"] = p3d.q25_segment_sharpe(equity, horizon)
        metrics["long_short_balance"] = p3d.long_short_balance(positions)
        metrics["short_abs_contribution_share"] = p3d.short_abs_contribution_share(positions)
        rows.append({"fold": fold_id, "p4_spec": spec.name, "p3d_formula_name": str(base["formula_name"]), **{f"p3d_validation_{k}": v for k, v in metrics.items() if p3d.is_scalar(v)}})
    return pd.DataFrame(rows)


def annotate_hard_accept(validation: pd.DataFrame, p3d_validation: pd.DataFrame) -> pd.DataFrame:
    out = validation.merge(p3d_validation.drop(columns=["p4_spec", "p3d_formula_name"], errors="ignore"), on="fold", how="left")
    turnover_limit = np.maximum(
        out["p3d_validation_mean_turnover"].astype(float) * 1.05,
        out["p3d_validation_mean_turnover"].astype(float) + 0.010,
    )
    out["hard_accept"] = (
        (out["validation_total_return"] > out["p3d_validation_total_return"])
        & (out["validation_sharpe"] > out["p3d_validation_sharpe"])
        & (out["validation_total_return_20bps"] > out["p3d_validation_total_return_20bps"])
        & (out["validation_max_drawdown"] >= 1.15 * out["p3d_validation_max_drawdown"])
        & (out["validation_short_abs_contribution_share"] <= 0.70)
        & (out["validation_mean_turnover"] <= turnover_limit)
    )
    return out


def select_by_fold(validation: pd.DataFrame, p3d_validation: pd.DataFrame, selections: pd.DataFrame, p3d_selected: pd.DataFrame, horizon: int) -> pd.DataFrame:
    base_by_fold = p3d_selected.set_index("fold")
    val_by_fold = p3d_validation.set_index("fold")
    rows = []
    sorted_validation = validation.sort_values(["fold", "validation_fitness", "validation_total_return", "validation_sharpe"], ascending=[True, False, False, False])
    for _, fold in selections.sort_values("fold").iterrows():
        fold_id = int(fold["fold"])
        base = base_by_fold.loc[fold_id]
        candidates = sorted_validation[(sorted_validation["fold"] == fold_id) & (sorted_validation["hard_accept"])]
        if candidates.empty:
            p3d_metrics = val_by_fold.loc[fold_id].to_dict()
            rows.append(
                {
                    "fold": fold_id,
                    "train_start": fold["train_start"],
                    "train_end": fold["train_end"],
                    "test_start": fold["test_start"],
                    "test_end": fold["test_end"],
                    "test_decision_end": p3d.decision_end_for_closed_positions(fold["test_end"], horizon),
                    "p4_spec": base["p4_spec"],
                    "p3d_formula_name": base["formula_name"],
                    "p3d_candidate_weight_long": base["candidate_weight_long"],
                    "p3d_candidate_weight_short": base["candidate_weight_short"],
                    "p3d_high_risk_candidate_cap": base["high_risk_candidate_cap"],
                    "deriv_formula_name": p3d.P4_ONLY_FORMULA,
                    "deriv_formula": "P3d fallback: no derivative increment passed validation hard-accept rules",
                    "candidate_weight_long": 0.0,
                    "candidate_weight_short": 0.0,
                    "high_risk_candidate_cap": 0.0,
                    "validation_fitness": p3d_metrics.get("p3d_validation_total_return", np.nan),
                    "validation_total_return": p3d_metrics.get("p3d_validation_total_return", np.nan),
                    "validation_sharpe": p3d_metrics.get("p3d_validation_sharpe", np.nan),
                    "validation_max_drawdown": p3d_metrics.get("p3d_validation_max_drawdown", np.nan),
                    "validation_mean_turnover": p3d_metrics.get("p3d_validation_mean_turnover", np.nan),
                    "validation_short_abs_contribution_share": p3d_metrics.get("p3d_validation_short_abs_contribution_share", np.nan),
                    "hard_accept": False,
                    "fallback_to_p3d": True,
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
                "test_decision_end": p3d.decision_end_for_closed_positions(fold["test_end"], horizon),
                "p4_spec": base["p4_spec"],
                "p3d_formula_name": base["formula_name"],
                "p3d_candidate_weight_long": base["candidate_weight_long"],
                "p3d_candidate_weight_short": base["candidate_weight_short"],
                "p3d_high_risk_candidate_cap": base["high_risk_candidate_cap"],
                "deriv_formula_name": winner["deriv_formula_name"],
                "deriv_formula": winner["deriv_formula"],
                "candidate_weight_long": winner["candidate_weight_long"],
                "candidate_weight_short": winner["candidate_weight_short"],
                "high_risk_candidate_cap": winner["high_risk_candidate_cap"],
                "validation_fitness": winner["validation_fitness"],
                "validation_total_return": winner.get("validation_total_return", np.nan),
                "validation_sharpe": winner.get("validation_sharpe", np.nan),
                "validation_max_drawdown": winner.get("validation_max_drawdown", np.nan),
                "validation_mean_turnover": winner.get("validation_mean_turnover", np.nan),
                "validation_short_abs_contribution_share": winner.get("validation_short_abs_contribution_share", np.nan),
                "validation_candidate_score_corr_to_p3d": winner.get("validation_candidate_score_corr_to_p3d", np.nan),
                "hard_accept": True,
                "fallback_to_p3d": False,
            }
        )
    return pd.DataFrame(rows)


def replay_selected_tests(
    panel: pd.DataFrame,
    selections: pd.DataFrame,
    selected: pd.DataFrame,
    pf_cfg: Any,
    risk_template: Any,
    spec_by_name: dict[str, Any],
    deriv_formulas: dict[str, FormulaSpec],
    p3d_formulas: dict[str, FormulaSpec],
) -> dict[str, Any]:
    fold_rows = []
    equity_parts = []
    position_parts = []
    selected_by_fold = selected.set_index("fold")
    for _, fold in selections.sort_values("fold").iterrows():
        fold_id = int(fold["fold"])
        row = selected_by_fold.loc[fold_id]
        spec = spec_by_name[str(row["p4_spec"])]
        compute_base_score(panel, row, spec, p3d_formulas)
        if str(row["deriv_formula_name"]) == p3d.P4_ONLY_FORMULA:
            panel[p3d.SCORE_COL] = panel[BASE_SCORE_COL]
            panel[p3d.CANDIDATE_RANK_COL] = np.nan
            formula_text = f"P3d_base({row['p3d_formula_name']}) fallback"
        else:
            formula = deriv_formulas[str(row["deriv_formula_name"])]
            apply_derivative_increment(panel, formula, float(row["candidate_weight_long"]), float(row["candidate_weight_short"]), float(row["high_risk_candidate_cap"]))
            formula_text = f"P3d_base({row['p3d_formula_name']}) + deriv({row['deriv_formula_name']},w={float(row['candidate_weight_long']):.2f}): {formula.formula}"
        risk_cfg = risk_template.__class__(
            **{
                **risk_template.__dict__,
                "name": f"p7b_{spec.name}",
                "short_resid_confirm_rank": spec.short_resid_confirm_rank,
            }
        )
        equity, positions, metrics = p3d.run_p4_gated_rerank_backtest(
            panel=panel,
            start=fold["test_start"],
            end=p3d.decision_end_for_closed_positions(fold["test_end"], pf_cfg.primary_horizon),
            cfg=pf_cfg,
            risk_cfg=risk_cfg,
            spec=spec,
            score_col=p3d.SCORE_COL,
            selected_formulas=formula_text,
        )
        p3d.add_gross_metrics(equity, metrics)
        p3d.add_risk_event_metrics(positions, metrics)
        p3d.add_delay_metrics(equity, metrics)
        metrics["long_short_balance"] = p3d.long_short_balance(positions)
        metrics["short_abs_contribution_share"] = p3d.short_abs_contribution_share(positions)
        fold_rows.append(
            {
                "fold": fold_id,
                "train_start": fold["train_start"],
                "train_end": fold["train_end"],
                "test_start": fold["test_start"],
                "test_end": fold["test_end"],
                "test_decision_end": p3d.decision_end_for_closed_positions(fold["test_end"], pf_cfg.primary_horizon),
                "p4_spec": spec.name,
                "p3d_formula_name": str(row["p3d_formula_name"]),
                "deriv_formula_name": str(row["deriv_formula_name"]),
                "formula": formula_text,
                "candidate_weight_long": float(row["candidate_weight_long"]),
                "candidate_weight_short": float(row["candidate_weight_short"]),
                "high_risk_candidate_cap": float(row["high_risk_candidate_cap"]),
                "hard_accept": bool(row.get("hard_accept", False)),
                "fallback_to_p3d": bool(row.get("fallback_to_p3d", False)),
                "validation_fitness": float(row["validation_fitness"]),
                **metrics,
            }
        )
        if not equity.empty:
            equity = equity.copy()
            positions = positions.copy()
            equity["fold"] = fold_id
            equity["p4_spec"] = spec.name
            equity["formula_name"] = str(row["deriv_formula_name"])
            equity["formula"] = formula_text
            positions["fold"] = fold_id
            positions["p4_spec"] = spec.name
            positions["formula_name"] = str(row["deriv_formula_name"])
            positions["formula"] = formula_text
            equity_parts.append(equity)
            position_parts.append(positions)
    equity_curve = pd.concat(equity_parts, ignore_index=True) if equity_parts else pd.DataFrame()
    positions = pd.concat(position_parts, ignore_index=True) if position_parts else pd.DataFrame()
    folds = pd.DataFrame(fold_rows)
    metrics = p3d.summarize_nested_result({"equity": equity_curve, "positions": positions, "folds": folds}, pf_cfg.rebalance_hours)
    return {"folds": folds, "equity": equity_curve, "positions": positions, "metrics": metrics}


def write_report(output_dir: Path, summary: pd.DataFrame, comparison: pd.DataFrame, selected: pd.DataFrame, validation: pd.DataFrame, cost_summary: pd.DataFrame, side_summary: pd.DataFrame, asset_summary: pd.DataFrame, primitive_usage: pd.DataFrame, formula_count: int, workers: int, horizon: int) -> None:
    lines = [
        "# P7b Derivatives Incremental On P3d",
        "",
        "## Design",
        "",
        f"- Horizon fixed at `{horizon}h`.",
        "- Base score is the already-selected P3d formula/weight for each fold.",
        "- P6b derivatives formulas can only add a small rerank increment inside the same P4-gated long/short pools.",
        "- Hard accept is against P3d validation, not P4.",
        f"- Derivative formulas: `{formula_count}`; workers: `{workers}`.",
        "",
        "## Summary",
        "```text",
        summary.to_string(index=False),
        "```",
        "",
        "## Comparison",
        "```text",
        comparison.to_string(index=False),
        "```",
        "",
        "## Selected",
        "```text",
        selected.to_string(index=False),
        "```",
        "",
        "## Primitive Usage",
        "```text",
        primitive_usage.to_string(index=False),
        "```",
        "",
        "## Cost Sensitivity",
        "```text",
        cost_summary.to_string(index=False),
        "```",
        "",
        "## Side Contribution",
        "```text",
        side_summary.to_string(index=False),
        "```",
        "",
        "## Asset Contribution",
        "```text",
        asset_summary.to_string(index=False),
        "```",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def update_config_stage(config_path: Path) -> None:
    cfg = load_config(config_path)
    stages = cfg.setdefault("stages", {})
    stages["p7b_derivatives_incremental_on_p3d"] = {
        "script": "../../../scripts/run_p7b_derivatives_incremental_on_p3d.py",
        "output_dir": "../outputs/p7b_derivatives_incremental_on_p3d",
        "report": "../outputs/p7b_derivatives_incremental_on_p3d/REPORT.md",
        "summary": "../outputs/p7b_derivatives_incremental_on_p3d/summary.csv",
        "baseline_comparison": "../outputs/p7b_derivatives_incremental_on_p3d/baseline_comparison.csv",
        "cost_sensitivity": "../outputs/p7b_derivatives_incremental_on_p3d/cost_sensitivity.csv",
        "selected_incremental_by_fold": "../outputs/p7b_derivatives_incremental_on_p3d/selected_incremental_by_fold.csv",
        "primitive_usage": "../outputs/p7b_derivatives_incremental_on_p3d/primitive_usage.csv",
    }
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False), encoding="utf-8")


if __name__ == "__main__":
    main()
