#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
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
from run_p3_committee_guard_scan import apply_short_guard, side_fraction_weights
import run_p3d_p4_gated_rerank_alphagpt as p3d
import run_p7_derivatives_gated_rerank_alphagpt as p7
import run_p9_clean_derivatives_baseline as p9
import run_p10_strategy_conditioned_regime_gate as p10
import run_p10d_nested_allocator_short_guard as p10d
from run_turnover_experiments import to_jsonable


EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "baseline_locked_v1"
DEFAULT_CONFIG = EXPERIMENT_ROOT / "configs" / "baseline_locked_v1.yaml"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "outputs" / "p13_turnover_no_trade_zone"
P9_DIR = EXPERIMENT_ROOT / "outputs" / "p9_clean_baseline_package" / "nested_incremental_run"
P9_SELECTED = EXPERIMENT_ROOT / "outputs" / "p9_clean_baseline_package" / "selected_incremental_by_fold.csv"
NO_OVERLAY = "__P9_NO_TURNOVER_OVERLAY__"


@dataclass(frozen=True)
class ExecutionSpec:
    name: str
    description: str
    family: str
    long_exit_rank: float | None = None
    short_exit_rank: float | None = None
    cost_gate_buffer: float = 0.0
    partial_kappa: float = 1.0
    adaptive_partial: bool = False
    min_kappa: float = 0.35
    max_kappa: float = 1.0
    turnover_budget: float | None = None
    initial_full_target: bool = True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--horizon", type=int, default=72)
    parser.add_argument("--inner-validation-days", type=int, default=90)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--cost-bps", type=float, nargs="+", default=[0, 4, 6, 8, 10, 12, 15, 20, 30])
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    output_dir = ensure_dir(args.output_dir)
    run_dir = ensure_dir(output_dir / "nested_turnover_run")
    horizon = int(args.horizon)
    cost_grid = [float(x) for x in args.cost_bps]

    print("[1/9] Loading P9 references and formula universes", flush=True)
    p9_result = p9.load_existing_result(P9_DIR)
    selections_path = p3d.resolve_config_path(config_path, cfg["inputs"]["formula_selections"])
    selections = pd.read_csv(selections_path, parse_dates=["train_start", "train_end", "test_start", "test_end"])
    selected_p9 = pd.read_csv(P9_SELECTED, parse_dates=["train_start", "train_end", "test_start", "test_end", "test_decision_end"])
    panel_path = p3d.resolve_config_path(config_path, cfg["inputs"]["panel"])

    retained = p7.load_retained_features()
    raw_deriv_formulas = p7.build_p7_formula_universe(retained, 160, 2607, 3)
    deriv_formulas = {item.name: item for item in p9.filter_p9_formulas(raw_deriv_formulas, exclude_k01=True)}
    p3d_formulas = {item.name: item for item in p3d.build_incremental_formula_universe(180, 2417, 3)}
    spec_by_name = {spec.name: spec for spec in p3d.build_committee_specs()}

    print("[2/9] Loading P7/P9 panel", flush=True)
    panel = p7.load_panel_p7(config_path, panel_path, p3d.MANUAL_FEATURE_PANEL, horizon)
    pf_cfg = p3d.make_portfolio_config(cfg, horizon)
    risk_template = p3d.make_risk_template(cfg)

    print("[3/9] Building turnover overlay grid", flush=True)
    overlay_specs = build_execution_specs()
    pd.DataFrame([spec.__dict__ for spec in overlay_specs]).to_csv(output_dir / "turnover_overlay_grid.csv", index=False)

    print("[4/9] Running fold-local validation search", flush=True)
    validation, p9_validation = run_validation_search(
        panel=panel,
        selections=selections,
        selected_p9=selected_p9,
        pf_cfg=pf_cfg,
        risk_template=risk_template,
        spec_by_name=spec_by_name,
        deriv_formulas=deriv_formulas,
        p3d_formulas=p3d_formulas,
        overlay_specs=overlay_specs,
        validation_days=int(args.inner_validation_days),
    )
    validation.to_csv(output_dir / "validation_turnover_metrics.csv", index=False)
    p9_validation.to_csv(output_dir / "validation_p9_turnover_baseline.csv", index=False)

    print("[5/9] Selecting execution overlay per fold without test data", flush=True)
    selected = select_overlay_by_fold(validation, p9_validation, selections)
    selected.to_csv(output_dir / "selected_turnover_overlay_by_fold.csv", index=False)

    print("[6/9] Replaying selected overlays on test windows only", flush=True)
    nested = replay_selected_overlays(
        panel=panel,
        selections=selections,
        selected=selected,
        selected_p9=selected_p9,
        pf_cfg=pf_cfg,
        risk_template=risk_template,
        spec_by_name=spec_by_name,
        deriv_formulas=deriv_formulas,
        p3d_formulas=p3d_formulas,
        overlay_specs=overlay_specs,
    )
    write_result(run_dir, nested)

    print("      running fixed matched overlay diagnostics", flush=True)
    fixed_results = run_fixed_overlay_diagnostics(
        panel=panel,
        selections=selections,
        selected_p9=selected_p9,
        pf_cfg=pf_cfg,
        risk_template=risk_template,
        spec_by_name=spec_by_name,
        deriv_formulas=deriv_formulas,
        p3d_formulas=p3d_formulas,
        overlay_specs=overlay_specs,
    )
    fixed_results_with_ref = {"p9_reference": p9_result, **fixed_results}
    fixed_summary = p10.build_summary(fixed_results_with_ref, horizon)
    fixed_summary.to_csv(output_dir / "fixed_overlay_summary.csv", index=False)
    fixed_cost = p10.build_cost_summary(fixed_results_with_ref, cost_grid)
    fixed_cost.to_csv(output_dir / "fixed_overlay_cost_sensitivity.csv", index=False)

    print("[7/9] Building diagnostics", flush=True)
    results = {
        "p13_turnover_overlay": nested,
        "p9_reference": p9_result,
    }
    summary = p10.build_summary(results, horizon)
    summary.to_csv(output_dir / "variant_summary.csv", index=False)
    cost_summary = p10.build_cost_summary(results, cost_grid)
    cost_summary.to_csv(output_dir / "cost_sensitivity.csv", index=False)
    fold_delta = p9.build_fold_delta(nested, p9_result, "p13_turnover_overlay", "p9")
    fold_delta.to_csv(output_dir / "fold_delta_vs_p9.csv", index=False)
    bootstrap = p9.bootstrap_fold_checks(fold_delta, "p13_turnover_overlay_minus_p9", int(args.bootstrap_samples))
    bootstrap.to_csv(output_dir / "bootstrap_checks.csv", index=False)
    start_vs_p9 = p9.build_start_fold_sensitivity(nested, p9_result, "p13_turnover_overlay", "p9", horizon)
    start_vs_p9.to_csv(output_dir / "start_fold_sensitivity_vs_p9.csv", index=False)
    dropout_vs_p9 = p9.build_symbol_dropout_checks(nested, p9_result, "p13_turnover_overlay", "p9", horizon)
    dropout_vs_p9.to_csv(output_dir / "symbol_dropout_vs_p9.csv", index=False)
    execution_attr = build_execution_attribution(nested)
    execution_attr.to_csv(output_dir / "execution_attribution.csv", index=False)
    reality = build_reality_summary(summary, cost_summary, bootstrap, start_vs_p9, dropout_vs_p9)
    reality.to_csv(output_dir / "reality_summary.csv", index=False)

    print("[8/9] Writing plots", flush=True)
    plot_curves(output_dir, results)
    plot_cost(output_dir, cost_summary)
    plot_selected(output_dir, selected)
    plot_fold_delta(output_dir, fold_delta)
    plot_execution_attr(output_dir, execution_attr)

    print("[9/9] Writing report and config metadata", flush=True)
    write_report(
        output_dir=output_dir,
        summary=summary,
        cost_summary=cost_summary,
        selected=selected,
        validation=validation,
        p9_validation=p9_validation,
        fold_delta=fold_delta,
        bootstrap=bootstrap,
        start_vs_p9=start_vs_p9,
        dropout_vs_p9=dropout_vs_p9,
        execution_attr=execution_attr,
        fixed_summary=fixed_summary,
        fixed_cost=fixed_cost,
        reality=reality,
        horizon=horizon,
        validation_days=int(args.inner_validation_days),
    )
    update_config_stage(config_path)
    print("[done] P13 complete", flush=True)
    print(f"Done: {output_dir}", flush=True)


def build_execution_specs() -> list[ExecutionSpec]:
    return [
        ExecutionSpec(
            name="p13_hysteresis_loose_exit50",
            description="Keep P9 target construction but widen exit no-trade zone to the cross-sectional midpoint.",
            family="hysteresis",
            long_exit_rank=0.50,
            short_exit_rank=0.50,
        ),
        ExecutionSpec(
            name="p13_hysteresis_loose_exit45_55",
            description="Very loose exit zone: long can remain until rank <45%, short can remain until rank >55%.",
            family="hysteresis",
            long_exit_rank=0.45,
            short_exit_rank=0.55,
        ),
        ExecutionSpec(
            name="p13_cost_gate_k1p0",
            description="Skip the rebalance if train-calibrated expected score benefit is not above 1.0x estimated cost.",
            family="cost_gate",
            cost_gate_buffer=1.0,
        ),
        ExecutionSpec(
            name="p13_cost_gate_k1p5",
            description="Skip the rebalance if train-calibrated expected score benefit is not above 1.5x estimated cost.",
            family="cost_gate",
            cost_gate_buffer=1.5,
        ),
        ExecutionSpec(
            name="p13_cost_gate_k2p0",
            description="Skip the rebalance if train-calibrated expected score benefit is not above 2.0x estimated cost.",
            family="cost_gate",
            cost_gate_buffer=2.0,
        ),
        ExecutionSpec(
            name="p13_partial_k50",
            description="Move 50% of the distance from current holdings to target weights.",
            family="partial",
            partial_kappa=0.50,
        ),
        ExecutionSpec(
            name="p13_partial_k65",
            description="Move 65% of the distance from current holdings to target weights.",
            family="partial",
            partial_kappa=0.65,
        ),
        ExecutionSpec(
            name="p13_partial_k80",
            description="Move 80% of the distance from current holdings to target weights.",
            family="partial",
            partial_kappa=0.80,
        ),
        ExecutionSpec(
            name="p13_budget_025",
            description="Cap each rebalance's realized turnover at 0.25 by uniformly scaling target delta.",
            family="turnover_budget",
            turnover_budget=0.25,
        ),
        ExecutionSpec(
            name="p13_budget_030",
            description="Cap each rebalance's realized turnover at 0.30 by uniformly scaling target delta.",
            family="turnover_budget",
            turnover_budget=0.30,
        ),
        ExecutionSpec(
            name="p13_budget_035",
            description="Cap each rebalance's realized turnover at 0.35 by uniformly scaling target delta.",
            family="turnover_budget",
            turnover_budget=0.35,
        ),
        ExecutionSpec(
            name="p13_costgate1p5_partial75",
            description="Cost gate plus 75% partial rebalance.",
            family="combo",
            cost_gate_buffer=1.5,
            partial_kappa=0.75,
        ),
        ExecutionSpec(
            name="p13_costgate1p5_budget030",
            description="Cost gate plus 0.30 turnover budget.",
            family="combo",
            cost_gate_buffer=1.5,
            turnover_budget=0.30,
        ),
        ExecutionSpec(
            name="p13_all_k1p5_partial75_budget030",
            description="Cost gate, 75% partial rebalance, and 0.30 turnover budget.",
            family="all",
            cost_gate_buffer=1.5,
            partial_kappa=0.75,
            turnover_budget=0.30,
        ),
        ExecutionSpec(
            name="p13_adaptive_partial_budget030",
            description="Adaptive partial rebalance with 0.30 turnover budget; kappa rises when score edge dominates cost.",
            family="all",
            adaptive_partial=True,
            min_kappa=0.35,
            max_kappa=0.90,
            turnover_budget=0.30,
            cost_gate_buffer=1.0,
        ),
    ]


def run_validation_search(
    panel: pd.DataFrame,
    selections: pd.DataFrame,
    selected_p9: pd.DataFrame,
    pf_cfg: Any,
    risk_template: Any,
    spec_by_name: dict[str, Any],
    deriv_formulas: dict[str, Any],
    p3d_formulas: dict[str, Any],
    overlay_specs: list[ExecutionSpec],
    validation_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validation_rows: list[dict[str, Any]] = []
    p9_rows: list[dict[str, Any]] = []
    selected_by_fold = selected_p9.set_index("fold")
    no_overlay = ExecutionSpec(NO_OVERLAY, "Original P9 execution.", "baseline")
    for _, fold in selections.sort_values("fold").iterrows():
        fold_id = int(fold["fold"])
        base = selected_by_fold.loc[fold_id]
        committee = spec_by_name[str(base["p4_spec"])]
        formula_text = p10d.prepare_fold_score(panel, base, committee, deriv_formulas, p3d_formulas)
        risk_cfg = make_fold_risk(risk_template, committee, f"p13_validation_{fold_id}")
        validation_start = max(pd.Timestamp(fold["train_start"]), pd.Timestamp(fold["train_end"]) - pd.Timedelta(days=validation_days))
        validation_end = p3d.decision_end_for_closed_positions(validation_start + pd.Timedelta(days=validation_days), pf_cfg.primary_horizon)
        validation_end = min(validation_end, pd.Timestamp(fold["train_end"]))
        coef = calibrate_score_coef(panel, pd.Timestamp(fold["train_start"]), validation_start, p3d.SCORE_COL, pf_cfg.primary_horizon)
        p9_equity, p9_positions, p9_metrics = run_turnover_overlay_backtest(
            panel=panel,
            start=validation_start,
            end=validation_end,
            cfg=pf_cfg,
            risk_cfg=risk_cfg,
            committee=committee,
            execution_spec=no_overlay,
            score_col=p3d.SCORE_COL,
            selected_formulas=formula_text,
            score_to_return_coef=coef,
        )
        enrich_metrics(p9_equity, p9_positions, p9_metrics, pf_cfg.primary_horizon)
        p9_rows.append(
            {
                "fold": fold_id,
                "validation_start": validation_start,
                "validation_decision_end": validation_end,
                **{f"p9_validation_{key}": value for key, value in p9_metrics.items() if p3d.is_scalar(value)},
            }
        )
        for spec in overlay_specs:
            candidate_pf = apply_execution_spec_to_cfg(pf_cfg, spec)
            equity, positions, metrics = run_turnover_overlay_backtest(
                panel=panel,
                start=validation_start,
                end=validation_end,
                cfg=candidate_pf,
                risk_cfg=risk_cfg,
                committee=committee,
                execution_spec=spec,
                score_col=p3d.SCORE_COL,
                selected_formulas=formula_text,
                score_to_return_coef=coef,
            )
            enrich_metrics(equity, positions, metrics, pf_cfg.primary_horizon)
            validation_rows.append(
                {
                    "fold": fold_id,
                    "overlay_name": spec.name,
                    "family": spec.family,
                    "validation_start": validation_start,
                    "validation_decision_end": validation_end,
                    "score_to_return_coef": coef,
                    **spec.__dict__,
                    **{f"validation_{key}": value for key, value in metrics.items() if p3d.is_scalar(value)},
                }
            )
    validation = pd.DataFrame(validation_rows)
    p9_validation = pd.DataFrame(p9_rows)
    validation = annotate_validation(validation, p9_validation)
    return validation, p9_validation


def calibrate_score_coef(panel: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, score_col: str, horizon: int) -> float:
    label_col = f"label_resid_ret_h{horizon}"
    mask = (panel["timestamp"] >= start) & (panel["timestamp"] < end)
    scoped = panel.loc[mask, ["timestamp", score_col, label_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(scoped) < 200:
        return 0.0
    x = scoped[score_col] - scoped.groupby("timestamp")[score_col].transform("mean")
    y = scoped[label_col] - scoped.groupby("timestamp")[label_col].transform("mean")
    denom = float(np.dot(x, x))
    if denom <= 1e-12:
        return 0.0
    slope = float(np.dot(x, y) / denom)
    return float(np.clip(max(0.0, slope), 0.0, 0.05))


def apply_execution_spec_to_cfg(cfg: Any, spec: ExecutionSpec) -> Any:
    updates: dict[str, Any] = {}
    if spec.long_exit_rank is not None:
        updates["long_exit_rank"] = float(spec.long_exit_rank)
    if spec.short_exit_rank is not None:
        updates["short_exit_rank"] = float(spec.short_exit_rank)
    return replace(cfg, **updates) if updates else cfg


def annotate_validation(validation: pd.DataFrame, p9_validation: pd.DataFrame) -> pd.DataFrame:
    out = validation.merge(p9_validation.drop(columns=["validation_start", "validation_decision_end"], errors="ignore"), on="fold", how="left")
    out["delta_validation_total_return"] = out["validation_total_return"] - out["p9_validation_total_return"]
    out["delta_validation_sharpe"] = out["validation_sharpe"] - out["p9_validation_sharpe"]
    out["delta_validation_max_drawdown"] = out["validation_max_drawdown"] - out["p9_validation_max_drawdown"]
    out["delta_validation_return_20bps"] = out["validation_total_return_20bps"] - out["p9_validation_total_return_20bps"]
    out["delta_validation_return_30bps"] = out["validation_total_return_30bps"] - out["p9_validation_total_return_30bps"]
    out["validation_turnover_reduction_vs_p9"] = 1.0 - out["validation_mean_turnover"] / out["p9_validation_mean_turnover"].replace(0.0, np.nan)
    out["validation_gross_retention_vs_p9"] = out["validation_gross_total_return_before_cost"] / out[
        "p9_validation_gross_total_return_before_cost"
    ].replace(0.0, np.nan)
    out["hard_accept"] = (
        (out["delta_validation_return_20bps"] > 0.0)
        & (out["delta_validation_return_30bps"] > 0.0)
        & (out["delta_validation_total_return"] >= -0.0025)
        & (out["delta_validation_sharpe"] >= -0.10)
        & (out["validation_max_drawdown"] >= 1.15 * out["p9_validation_max_drawdown"])
        & (out["validation_mean_turnover"] <= out["p9_validation_mean_turnover"] * 0.95)
        & (out["validation_gross_retention_vs_p9"] >= 0.75)
        & (out["validation_mean_abs_beta_market"] <= 1e-8)
    )
    out["selection_score"] = (
        100.0 * out["delta_validation_return_30bps"].fillna(-1.0)
        + 80.0 * out["delta_validation_return_20bps"].fillna(-1.0)
        + 40.0 * out["delta_validation_total_return"].fillna(-1.0)
        + 2.0 * out["delta_validation_sharpe"].fillna(-5.0)
        + 6.0 * out["validation_turnover_reduction_vs_p9"].fillna(0.0)
        + 1.5 * out["validation_gross_retention_vs_p9"].clip(-1.0, 2.0).fillna(0.0)
        - 1.0 * out.get("validation_missed_positive_target_return", 0.0)
    )
    return out


def select_overlay_by_fold(validation: pd.DataFrame, p9_validation: pd.DataFrame, selections: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    validation_by_fold = {int(k): v.copy() for k, v in validation.groupby("fold")}
    p9_by_fold = p9_validation.set_index("fold")
    for _, fold in selections.sort_values("fold").iterrows():
        fold_id = int(fold["fold"])
        candidates = validation_by_fold.get(fold_id, pd.DataFrame())
        accepted = candidates[candidates["hard_accept"].astype(bool)].copy() if not candidates.empty else pd.DataFrame()
        if accepted.empty:
            p9_row = p9_by_fold.loc[fold_id]
            rows.append(
                {
                    "fold": fold_id,
                    "selected_overlay": NO_OVERLAY,
                    "family": "baseline",
                    "fallback_to_p9_execution": True,
                    "validation_selection_score": float(p9_row.get("p9_validation_total_return", np.nan)),
                    "validation_total_return": float(p9_row.get("p9_validation_total_return", np.nan)),
                    "validation_sharpe": float(p9_row.get("p9_validation_sharpe", np.nan)),
                    "validation_mean_turnover": float(p9_row.get("p9_validation_mean_turnover", np.nan)),
                }
            )
            continue
        winner = accepted.sort_values(["selection_score", "delta_validation_return_30bps"], ascending=False).iloc[0]
        rows.append(
            {
                "fold": fold_id,
                "selected_overlay": str(winner["overlay_name"]),
                "family": str(winner["family"]),
                "fallback_to_p9_execution": False,
                "validation_selection_score": float(winner["selection_score"]),
                "validation_total_return": float(winner["validation_total_return"]),
                "validation_sharpe": float(winner["validation_sharpe"]),
                "validation_mean_turnover": float(winner["validation_mean_turnover"]),
                "validation_turnover_reduction_vs_p9": float(winner["validation_turnover_reduction_vs_p9"]),
                "validation_return_20bps": float(winner["validation_total_return_20bps"]),
                "validation_return_30bps": float(winner["validation_total_return_30bps"]),
            }
        )
    return pd.DataFrame(rows)


def replay_selected_overlays(
    panel: pd.DataFrame,
    selections: pd.DataFrame,
    selected: pd.DataFrame,
    selected_p9: pd.DataFrame,
    pf_cfg: Any,
    risk_template: Any,
    spec_by_name: dict[str, Any],
    deriv_formulas: dict[str, Any],
    p3d_formulas: dict[str, Any],
    overlay_specs: list[ExecutionSpec],
) -> dict[str, Any]:
    overlay_map = {spec.name: spec for spec in overlay_specs}
    overlay_map[NO_OVERLAY] = ExecutionSpec(NO_OVERLAY, "Original P9 execution.", "baseline")
    selected_exec = selected.set_index("fold")
    selected_p9_by_fold = selected_p9.set_index("fold")
    fold_rows: list[dict[str, Any]] = []
    equity_parts: list[pd.DataFrame] = []
    position_parts: list[pd.DataFrame] = []
    for _, fold in selections.sort_values("fold").iterrows():
        fold_id = int(fold["fold"])
        base = selected_p9_by_fold.loc[fold_id]
        decision = selected_exec.loc[fold_id]
        spec = overlay_map[str(decision["selected_overlay"])]
        committee = spec_by_name[str(base["p4_spec"])]
        formula_text = p10d.prepare_fold_score(panel, base, committee, deriv_formulas, p3d_formulas)
        risk_cfg = make_fold_risk(risk_template, committee, f"p13_{spec.name}_{fold_id}")
        coef = calibrate_score_coef(panel, pd.Timestamp(fold["train_start"]), pd.Timestamp(fold["train_end"]), p3d.SCORE_COL, pf_cfg.primary_horizon)
        candidate_pf = apply_execution_spec_to_cfg(pf_cfg, spec)
        start = pd.Timestamp(fold["test_start"])
        end = p3d.decision_end_for_closed_positions(pd.Timestamp(fold["test_end"]), pf_cfg.primary_horizon)
        equity, positions, metrics = run_turnover_overlay_backtest(
            panel=panel,
            start=start,
            end=end,
            cfg=candidate_pf,
            risk_cfg=risk_cfg,
            committee=committee,
            execution_spec=spec,
            score_col=p3d.SCORE_COL,
            selected_formulas=formula_text,
            score_to_return_coef=coef,
        )
        enrich_metrics(equity, positions, metrics, pf_cfg.primary_horizon)
        fold_rows.append(
            {
                "fold": fold_id,
                "train_start": fold["train_start"],
                "train_end": fold["train_end"],
                "test_start": fold["test_start"],
                "test_end": fold["test_end"],
                "test_decision_end": end,
                "p4_spec": committee.name,
                "p3d_formula_name": str(base["p3d_formula_name"]),
                "deriv_formula_name": str(base["deriv_formula_name"]),
                "formula": formula_text,
                "selected_overlay": spec.name,
                "overlay_family": spec.family,
                "fallback_to_p9_execution": bool(decision["fallback_to_p9_execution"]),
                "validation_selection_score": float(decision["validation_selection_score"]),
                "score_to_return_coef": coef,
                **metrics,
            }
        )
        if not equity.empty:
            equity = equity.copy()
            positions = positions.copy()
            equity["fold"] = fold_id
            equity["p4_spec"] = committee.name
            equity["formula_name"] = str(base["deriv_formula_name"])
            equity["formula"] = formula_text
            equity["selected_overlay"] = spec.name
            positions["fold"] = fold_id
            positions["p4_spec"] = committee.name
            positions["formula_name"] = str(base["deriv_formula_name"])
            positions["formula"] = formula_text
            positions["selected_overlay"] = spec.name
            equity_parts.append(equity)
            position_parts.append(positions)
    equity_curve = pd.concat(equity_parts, ignore_index=True) if equity_parts else pd.DataFrame()
    positions = pd.concat(position_parts, ignore_index=True) if position_parts else pd.DataFrame()
    folds = pd.DataFrame(fold_rows)
    metrics = p3d.summarize_nested_result({"equity": equity_curve, "positions": positions, "folds": folds}, pf_cfg.rebalance_hours)
    return {"folds": folds, "equity": equity_curve, "positions": positions, "metrics": metrics}


def run_fixed_overlay_diagnostics(
    panel: pd.DataFrame,
    selections: pd.DataFrame,
    selected_p9: pd.DataFrame,
    pf_cfg: Any,
    risk_template: Any,
    spec_by_name: dict[str, Any],
    deriv_formulas: dict[str, Any],
    p3d_formulas: dict[str, Any],
    overlay_specs: list[ExecutionSpec],
) -> dict[str, dict[str, Any]]:
    diagnostic_names = {
        "p13_hysteresis_loose_exit50",
        "p13_cost_gate_k1p5",
        "p13_partial_k65",
        "p13_budget_030",
        "p13_costgate1p5_partial75",
        "p13_costgate1p5_budget030",
        "p13_all_k1p5_partial75_budget030",
        "p13_adaptive_partial_budget030",
    }
    results: dict[str, dict[str, Any]] = {}
    for spec in overlay_specs:
        if spec.name not in diagnostic_names:
            continue
        results[spec.name] = replay_single_overlay(
            panel=panel,
            selections=selections,
            selected_p9=selected_p9,
            pf_cfg=pf_cfg,
            risk_template=risk_template,
            spec_by_name=spec_by_name,
            deriv_formulas=deriv_formulas,
            p3d_formulas=p3d_formulas,
            execution_spec=spec,
        )
    return results


def replay_single_overlay(
    panel: pd.DataFrame,
    selections: pd.DataFrame,
    selected_p9: pd.DataFrame,
    pf_cfg: Any,
    risk_template: Any,
    spec_by_name: dict[str, Any],
    deriv_formulas: dict[str, Any],
    p3d_formulas: dict[str, Any],
    execution_spec: ExecutionSpec,
) -> dict[str, Any]:
    selected_p9_by_fold = selected_p9.set_index("fold")
    fold_rows: list[dict[str, Any]] = []
    equity_parts: list[pd.DataFrame] = []
    position_parts: list[pd.DataFrame] = []
    for _, fold in selections.sort_values("fold").iterrows():
        fold_id = int(fold["fold"])
        base = selected_p9_by_fold.loc[fold_id]
        committee = spec_by_name[str(base["p4_spec"])]
        formula_text = p10d.prepare_fold_score(panel, base, committee, deriv_formulas, p3d_formulas)
        risk_cfg = make_fold_risk(risk_template, committee, f"p13_fixed_{execution_spec.name}_{fold_id}")
        coef = calibrate_score_coef(panel, pd.Timestamp(fold["train_start"]), pd.Timestamp(fold["train_end"]), p3d.SCORE_COL, pf_cfg.primary_horizon)
        candidate_pf = apply_execution_spec_to_cfg(pf_cfg, execution_spec)
        start = pd.Timestamp(fold["test_start"])
        end = p3d.decision_end_for_closed_positions(pd.Timestamp(fold["test_end"]), pf_cfg.primary_horizon)
        equity, positions, metrics = run_turnover_overlay_backtest(
            panel=panel,
            start=start,
            end=end,
            cfg=candidate_pf,
            risk_cfg=risk_cfg,
            committee=committee,
            execution_spec=execution_spec,
            score_col=p3d.SCORE_COL,
            selected_formulas=formula_text,
            score_to_return_coef=coef,
        )
        enrich_metrics(equity, positions, metrics, pf_cfg.primary_horizon)
        fold_rows.append(
            {
                "fold": fold_id,
                "train_start": fold["train_start"],
                "train_end": fold["train_end"],
                "test_start": fold["test_start"],
                "test_end": fold["test_end"],
                "test_decision_end": end,
                "p4_spec": committee.name,
                "p3d_formula_name": str(base["p3d_formula_name"]),
                "deriv_formula_name": str(base["deriv_formula_name"]),
                "formula": formula_text,
                "selected_overlay": execution_spec.name,
                "overlay_family": execution_spec.family,
                "score_to_return_coef": coef,
                **metrics,
            }
        )
        if not equity.empty:
            equity = equity.copy()
            positions = positions.copy()
            equity["fold"] = fold_id
            equity["p4_spec"] = committee.name
            equity["formula_name"] = str(base["deriv_formula_name"])
            equity["formula"] = formula_text
            equity["selected_overlay"] = execution_spec.name
            positions["fold"] = fold_id
            positions["p4_spec"] = committee.name
            positions["formula_name"] = str(base["deriv_formula_name"])
            positions["formula"] = formula_text
            positions["selected_overlay"] = execution_spec.name
            equity_parts.append(equity)
            position_parts.append(positions)
    equity_curve = pd.concat(equity_parts, ignore_index=True) if equity_parts else pd.DataFrame()
    positions = pd.concat(position_parts, ignore_index=True) if position_parts else pd.DataFrame()
    folds = pd.DataFrame(fold_rows)
    metrics = p3d.summarize_nested_result({"equity": equity_curve, "positions": positions, "folds": folds}, pf_cfg.rebalance_hours)
    return {"folds": folds, "equity": equity_curve, "positions": positions, "metrics": metrics}


def make_fold_risk(risk_template: Any, committee: Any, name: str) -> Any:
    return risk_template.__class__(
        **{
            **risk_template.__dict__,
            "name": name,
            "short_resid_confirm_rank": committee.short_resid_confirm_rank,
        }
    )


def run_turnover_overlay_backtest(
    panel: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cfg: Any,
    risk_cfg: Any,
    committee: Any,
    execution_spec: ExecutionSpec,
    score_col: str,
    selected_formulas: str,
    score_to_return_coef: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    label_col = f"label_resid_ret_h{cfg.primary_horizon}"
    ret_col = f"fwd_simple_ret_h{cfg.primary_horizon}"
    symbols = sorted(panel["symbol"].unique())
    mask = (panel["timestamp"] >= start) & (panel["timestamp"] < end)
    timestamps = np.array(sorted(panel.loc[mask, "timestamp"].unique()))
    timestamps = timestamps[:: cfg.rebalance_hours]

    prev_weights = pd.Series(0.0, index=symbols)
    signal_state: dict[str, tuple[int, int]] = {}
    equity = 1.0
    rows_out: list[dict[str, Any]] = []
    pos_out: list[dict[str, Any]] = []

    for ts in timestamps:
        rows = panel[panel["timestamp"] == ts].copy()
        rows = rows[
            np.isfinite(rows[score_col])
            & np.isfinite(rows[p3d.P4_SCORE_COL])
            & np.isfinite(rows[ret_col])
            & np.isfinite(rows[label_col])
        ]
        if len(rows) < cfg.long_k + cfg.short_k:
            continue
        rows["score_rank"] = rows[score_col].rank(pct=True, method="average")
        rows["p4_rank"] = rows[p3d.P4_SCORE_COL].rank(pct=True, method="average")
        rows["ema_rank"] = rows["score_ema_025"].rank(pct=True, method="average")
        rows["base_rank"] = rows["score_baseline"].rank(pct=True, method="average")
        rows = rows.sort_values(score_col, ascending=False)
        rows_by_symbol = rows.set_index("symbol")

        target_sides = p3d.select_p4_gated_target_sides(rows, signal_state, cfg, score_col)
        before_short_guard = len([side for side in target_sides.values() if side < 0])
        target_sides = apply_short_guard(target_sides, rows_by_symbol, committee)
        after_short_guard = len([side for side in target_sides.values() if side < 0])
        if not any(side > 0 for side in target_sides.values()) or not any(side < 0 for side in target_sides.values()):
            continue

        selected = rows_by_symbol.reindex(list(target_sides)).reset_index()
        raw_weights = side_fraction_weights(target_sides, cfg.gross_exposure, committee.long_fraction)
        target_weights, optimizer_info = p3d.build_weights(raw_weights, selected, rows_by_symbol, cfg, risk_cfg)
        target_weights = target_weights[target_weights.abs() > 1e-12]
        if target_weights.abs().sum() <= 0:
            continue

        target_full_weights = pd.Series(0.0, index=symbols)
        target_full_weights.loc[target_weights.index] = target_weights
        full_weights, exec_info = apply_execution_overlay(
            prev_weights=prev_weights,
            target_weights=target_full_weights,
            rows_by_symbol=rows_by_symbol,
            score_col=score_col,
            cfg=cfg,
            spec=execution_spec,
            score_to_return_coef=score_to_return_coef,
        )
        unavailable = ~full_weights.index.isin(rows_by_symbol.index)
        if unavailable.any():
            full_weights.loc[unavailable] = 0.0
        active_weights = full_weights[full_weights.abs() > 1e-12]
        if active_weights.empty:
            continue

        turnover = float((full_weights - prev_weights).abs().sum())
        target_turnover = float((target_full_weights - prev_weights).abs().sum())
        cost = turnover * (cfg.one_way_fee_bps + cfg.one_way_slippage_bps) / 10_000.0
        active_rows = rows_by_symbol.reindex(active_weights.index)
        gross_period_ret = float((active_weights * active_rows[ret_col]).sum())
        target_active = target_full_weights[target_full_weights.abs() > 1e-12]
        target_rows = rows_by_symbol.reindex(target_active.index)
        target_gross_ret = float((target_active * target_rows[ret_col]).sum())
        missed_target_return = target_gross_ret - gross_period_ret
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
                "target_gross_period_return": target_gross_ret,
                "missed_target_return": missed_target_return,
                "missed_positive_target_return": max(0.0, missed_target_return),
                "cost_return": cost,
                "equity": equity,
                "turnover": turnover,
                "target_turnover": target_turnover,
                "turnover_saved": max(0.0, target_turnover - turnover),
                "gross_exposure": float(active_weights.abs().sum()),
                "target_gross_exposure": float(target_full_weights.abs().sum()),
                "rank_ic": rank_ic,
                "top_bottom_label_spread": top_bottom,
                "abs_beta_btc": abs(exposures.get("beta_btc", 0.0)),
                "abs_beta_eth": abs(exposures.get("beta_eth", 0.0)),
                "abs_beta_market": abs(exposures.get("beta_market", 0.0)),
                "symbol_concentration": float((active_weights**2).sum()),
                "max_abs_weight": float(active_weights.abs().max()),
                "delay_tau_days": 0.0,
                "delay_retention": 0.0,
                "delay_new_target_weight": 1.0,
                "short_guard_removed": float(before_short_guard - after_short_guard),
                "target_applied": float(turnover > 1e-12),
                "execution_overlay": execution_spec.name,
                **exec_info,
                **optimizer_info,
                "selected_formulas": selected_formulas,
            }
        )

        for symbol, weight in active_weights.items():
            row = rows_by_symbol.loc[symbol]
            target_weight = float(target_full_weights.loc[symbol])
            contribution = float(weight * row[ret_col])
            target_contribution = float(target_weight * row[ret_col])
            pos_out.append(
                {
                    "timestamp": pd.Timestamp(ts),
                    "exit_timestamp": pd.Timestamp(ts) + pd.Timedelta(hours=cfg.execution_delay_bars + cfg.primary_horizon),
                    "symbol": symbol,
                    "weight": float(weight),
                    "target_weight": target_weight,
                    "desired_target_weight": target_weight,
                    "score": float(row[score_col]),
                    "p4_score": float(row[p3d.P4_SCORE_COL]),
                    "candidate_rank": float(row[p3d.CANDIDATE_RANK_COL]) if p3d.CANDIDATE_RANK_COL in row.index else np.nan,
                    "p4_rank": float(row["p4_rank"]),
                    "ema_rank": float(row["ema_rank"]),
                    "base_rank": float(row["base_rank"]),
                    "rank_a07": float(row["rank_a07"]),
                    "rank_b11": float(row["rank_b11"]),
                    "rank_c09": float(row["rank_c09"]),
                    "fwd_simple_ret": float(row[ret_col]),
                    "label_resid_ret": float(row[label_col]),
                    "contribution": contribution,
                    "target_contribution": target_contribution,
                    "missed_contribution": target_contribution - contribution,
                    "selected_score_cols": "p4_score | candidate_rerank_score",
                }
            )

        prev_weights = full_weights
        signal_state = update_state(signal_state, target_sides, True)

    equity_curve = pd.DataFrame(rows_out)
    positions = pd.DataFrame(pos_out)
    metrics = p3d.aggregate_oos_metrics(equity_curve, cfg.rebalance_hours)
    return equity_curve, positions, metrics


def apply_execution_overlay(
    prev_weights: pd.Series,
    target_weights: pd.Series,
    rows_by_symbol: pd.DataFrame,
    score_col: str,
    cfg: Any,
    spec: ExecutionSpec,
    score_to_return_coef: float,
) -> tuple[pd.Series, dict[str, float]]:
    target_weights = target_weights.reindex(prev_weights.index).fillna(0.0)
    if spec.name == NO_OVERLAY:
        return target_weights.copy(), {
            "execution_kappa": 1.0,
            "execution_edge_estimate": np.nan,
            "execution_cost_estimate": np.nan,
            "execution_edge_cost_ratio": np.nan,
            "execution_gate_blocked": 0.0,
            "execution_budget_scale": 1.0,
        }
    if spec.initial_full_target and prev_weights.abs().sum() <= 1e-12:
        return target_weights.copy(), {
            "execution_kappa": 1.0,
            "execution_edge_estimate": np.nan,
            "execution_cost_estimate": np.nan,
            "execution_edge_cost_ratio": np.nan,
            "execution_gate_blocked": 0.0,
            "execution_budget_scale": 1.0,
        }
    delta = target_weights - prev_weights
    target_turnover = float(delta.abs().sum())
    if target_turnover <= 1e-12:
        return target_weights.copy(), {
            "execution_kappa": 1.0,
            "execution_edge_estimate": 0.0,
            "execution_cost_estimate": 0.0,
            "execution_edge_cost_ratio": np.nan,
            "execution_gate_blocked": 0.0,
            "execution_budget_scale": 1.0,
        }
    expected = expected_returns(rows_by_symbol, prev_weights.index, score_col, score_to_return_coef)
    edge = float((delta * expected).sum())
    cost = target_turnover * (cfg.one_way_fee_bps + cfg.one_way_slippage_bps) / 10_000.0
    ratio = edge / max(cost, 1e-12)
    if spec.cost_gate_buffer > 0 and edge <= spec.cost_gate_buffer * cost:
        return prev_weights.copy(), {
            "execution_kappa": 0.0,
            "execution_edge_estimate": edge,
            "execution_cost_estimate": cost,
            "execution_edge_cost_ratio": ratio,
            "execution_gate_blocked": 1.0,
            "execution_budget_scale": 0.0,
        }
    kappa = float(spec.partial_kappa)
    if spec.adaptive_partial:
        kappa = float(np.clip(spec.min_kappa + 0.15 * max(0.0, ratio - 1.0), spec.min_kappa, spec.max_kappa))
    candidate_delta = kappa * delta
    budget_scale = 1.0
    candidate_turnover = float(candidate_delta.abs().sum())
    if spec.turnover_budget is not None and candidate_turnover > float(spec.turnover_budget) > 0:
        budget_scale = float(spec.turnover_budget) / candidate_turnover
        candidate_delta = budget_scale * candidate_delta
    full = prev_weights + candidate_delta
    return full, {
        "execution_kappa": kappa,
        "execution_edge_estimate": edge,
        "execution_cost_estimate": cost,
        "execution_edge_cost_ratio": ratio,
        "execution_gate_blocked": 0.0,
        "execution_budget_scale": budget_scale,
    }


def expected_returns(rows_by_symbol: pd.DataFrame, symbols: pd.Index, score_col: str, coef: float) -> pd.Series:
    scores = rows_by_symbol[score_col].reindex(symbols).astype(float)
    centered = scores - float(scores.mean(skipna=True))
    return centered.fillna(0.0) * float(coef)


def enrich_metrics(equity: pd.DataFrame, positions: pd.DataFrame, metrics: dict[str, float], horizon: int) -> None:
    p3d.add_gross_metrics(equity, metrics)
    p3d.add_risk_event_metrics(positions, metrics)
    p3d.add_delay_metrics(equity, metrics)
    metrics.update(p3d.cost_adjusted_metrics(equity, horizon, [20.0, 30.0]))
    metrics["q25_segment_sharpe"] = p3d.q25_segment_sharpe(equity, horizon)
    metrics["long_short_balance"] = p3d.long_short_balance(positions)
    metrics["short_abs_contribution_share"] = p3d.short_abs_contribution_share(positions)
    if not equity.empty:
        target_turnover = float(equity["target_turnover"].sum()) if "target_turnover" in equity else np.nan
        turnover = float(equity["turnover"].sum()) if "turnover" in equity else np.nan
        metrics["mean_target_turnover"] = float(equity["target_turnover"].mean()) if "target_turnover" in equity else np.nan
        metrics["total_target_turnover"] = target_turnover
        metrics["mean_turnover_saved"] = float(equity["turnover_saved"].mean()) if "turnover_saved" in equity else np.nan
        metrics["total_turnover_saved"] = float(equity["turnover_saved"].sum()) if "turnover_saved" in equity else np.nan
        metrics["turnover_reduction"] = 1.0 - turnover / target_turnover if target_turnover and target_turnover > 0 else 0.0
        target_gross_sum = float(equity["target_gross_period_return"].sum()) if "target_gross_period_return" in equity else np.nan
        gross_sum = float(equity["gross_period_return"].sum()) if "gross_period_return" in equity else np.nan
        metrics["gross_return_retention_sum"] = gross_sum / target_gross_sum if np.isfinite(target_gross_sum) and abs(target_gross_sum) > 1e-12 else np.nan
        metrics["missed_target_return_sum"] = float(equity["missed_target_return"].sum()) if "missed_target_return" in equity else 0.0
        metrics["missed_positive_target_return"] = float(equity["missed_positive_target_return"].sum()) if "missed_positive_target_return" in equity else 0.0
        metrics["mean_execution_kappa"] = float(equity["execution_kappa"].mean()) if "execution_kappa" in equity else np.nan
        metrics["gate_block_rate"] = float(equity["execution_gate_blocked"].mean()) if "execution_gate_blocked" in equity else 0.0
        metrics["approx_avg_holding_rebalances"] = (
            float(equity["gross_exposure"].mean()) / max(float(equity["turnover"].mean()) / 2.0, 1e-12)
            if "gross_exposure" in equity and "turnover" in equity
            else np.nan
        )


def build_execution_attribution(result: dict[str, Any]) -> pd.DataFrame:
    equity = result["equity"].copy()
    if equity.empty:
        return pd.DataFrame()
    return (
        equity.groupby("selected_overlay")
        .agg(
            periods=("timestamp", "count"),
            period_return_sum=("period_return", "sum"),
            gross_return_sum=("gross_period_return", "sum"),
            target_gross_return_sum=("target_gross_period_return", "sum"),
            cost_sum=("cost_return", "sum"),
            turnover_sum=("turnover", "sum"),
            target_turnover_sum=("target_turnover", "sum"),
            turnover_saved_sum=("turnover_saved", "sum"),
            missed_target_return_sum=("missed_target_return", "sum"),
            missed_positive_target_return_sum=("missed_positive_target_return", "sum"),
            mean_execution_kappa=("execution_kappa", "mean"),
            gate_block_rate=("execution_gate_blocked", "mean"),
        )
        .reset_index()
        .sort_values("period_return_sum", ascending=False)
    )


def build_reality_summary(
    summary: pd.DataFrame,
    cost: pd.DataFrame,
    bootstrap: pd.DataFrame,
    start_vs_p9: pd.DataFrame,
    dropout_vs_p9: pd.DataFrame,
) -> pd.DataFrame:
    cand = summary[summary["variant"] == "p13_turnover_overlay"].iloc[0]
    p9_row = summary[summary["variant"] == "p9_reference"].iloc[0]
    c20 = cost[(cost["variant"] == "p13_turnover_overlay") & (cost["one_way_cost_bps"] == 20.0)].iloc[0]
    p20 = cost[(cost["variant"] == "p9_reference") & (cost["one_way_cost_bps"] == 20.0)].iloc[0]
    c30 = cost[(cost["variant"] == "p13_turnover_overlay") & (cost["one_way_cost_bps"] == 30.0)].iloc[0]
    p30 = cost[(cost["variant"] == "p9_reference") & (cost["one_way_cost_bps"] == 30.0)].iloc[0]
    boot = bootstrap.iloc[0]
    start_positive = float((start_vs_p9["delta_total_return"] > 0).mean()) if "delta_total_return" in start_vs_p9 else np.nan
    dropout_positive = float((dropout_vs_p9["delta_total_return"] > 0).mean()) if "delta_total_return" in dropout_vs_p9 else np.nan
    turnover_reduction = 1.0 - cand["mean_turnover"] / p9_row["mean_turnover"]
    gross_retention = cand["gross_total_return_before_cost"] / p9_row["gross_total_return_before_cost"]
    return pd.DataFrame(
        [
            {"check": "net_delta_vs_p9", "value": cand["total_return"] - p9_row["total_return"], "pass": bool(cand["total_return"] > p9_row["total_return"])},
            {"check": "sharpe_delta_vs_p9", "value": cand["sharpe"] - p9_row["sharpe"], "pass": bool(cand["sharpe"] > p9_row["sharpe"])},
            {"check": "maxdd_not_materially_worse", "value": cand["max_drawdown"] - p9_row["max_drawdown"], "pass": bool(cand["max_drawdown"] >= 1.10 * p9_row["max_drawdown"])},
            {"check": "return_20bps_delta_vs_p9", "value": c20["total_return"] - p20["total_return"], "pass": bool(c20["total_return"] > p20["total_return"])},
            {"check": "return_30bps_delta_vs_p9", "value": c30["total_return"] - p30["total_return"], "pass": bool(c30["total_return"] > p30["total_return"] and c30["total_return"] > 0.0)},
            {"check": "turnover_reduction_15pct", "value": turnover_reduction, "pass": bool(turnover_reduction >= 0.15)},
            {"check": "gross_retention_80pct", "value": gross_retention, "pass": bool(gross_retention >= 0.80)},
            {"check": "worst5_delta_vs_p9", "value": cand["worst_5_fold_return_sum"] - p9_row["worst_5_fold_return_sum"], "pass": bool(cand["worst_5_fold_return_sum"] >= p9_row["worst_5_fold_return_sum"])},
            {"check": "bootstrap_delta_vs_p9_q05", "value": boot["bootstrap_q05"], "pass": bool(boot["bootstrap_q05"] > -0.0025)},
            {"check": "start_fold_positive_vs_p9", "value": start_positive, "pass": bool(start_positive >= 0.45)},
            {"check": "symbol_dropout_positive_vs_p9", "value": dropout_positive, "pass": bool(dropout_positive >= 0.45)},
            {"check": "strict_mean_abs_market_beta", "value": cand["mean_abs_beta_market"], "pass": bool(cand["mean_abs_beta_market"] < 1e-8)},
            {"check": "strict_max_abs_market_beta", "value": cand["max_abs_beta_market"], "pass": bool(cand["max_abs_beta_market"] < 1e-7)},
        ]
    )


def write_result(run_dir: Path, result: dict[str, Any]) -> None:
    result["folds"].to_csv(run_dir / "walk_forward_folds.csv", index=False)
    result["equity"].to_csv(run_dir / "equity_curve.csv", index=False)
    result["positions"].to_csv(run_dir / "positions.csv", index=False)
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(to_jsonable(result["metrics"]), fh, indent=2)


def plot_curves(output_dir: Path, results: dict[str, dict[str, Any]]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0]})
    for name, result in results.items():
        eq = result["equity"].sort_values("timestamp")
        if eq.empty:
            continue
        net = (1.0 + eq["period_return"]).cumprod() - 1.0
        gross = (1.0 + eq["gross_period_return"]).cumprod() - 1.0
        axes[0].plot(eq["timestamp"], net, label=f"{name} net", linewidth=2.0)
        axes[0].plot(eq["timestamp"], gross, label=f"{name} gross", linewidth=1.2, alpha=0.45)
        axes[1].plot(eq["timestamp"], eq["turnover"].rolling(8, min_periods=1).mean(), label=name, linewidth=1.6)
    axes[0].axhline(0, color="#222222", linewidth=0.8)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].set_title("P13 turnover-aware overlay vs P9")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.25)
    axes[1].set_title("Rolling mean turnover")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir / "p13_curves.png", dpi=160)
    plt.close(fig)


def plot_cost(output_dir: Path, cost: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, frame in cost.groupby("variant"):
        ax.plot(frame["one_way_cost_bps"], frame["total_return"], marker="o", label=name)
    ax.axhline(0, color="#222222", linewidth=0.8)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("one-way cost bps")
    ax.set_ylabel("total return")
    ax.set_title("Cost sensitivity")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "cost_sensitivity.png", dpi=160)
    plt.close(fig)


def plot_selected(output_dir: Path, selected: pd.DataFrame) -> None:
    counts = selected["selected_overlay"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(counts))))
    ax.barh(counts.index, counts.values)
    ax.set_title("Selected turnover overlay by fold")
    ax.set_xlabel("fold count")
    fig.tight_layout()
    fig.savefig(output_dir / "selected_overlay_counts.png", dpi=160)
    plt.close(fig)


def plot_fold_delta(output_dir: Path, fold_delta: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    colors = np.where(fold_delta["delta_total_return"] >= 0, "#2b8a3e", "#c92a2a")
    ax.bar(fold_delta["fold"], fold_delta["delta_total_return"], color=colors)
    ax.axhline(0, color="#222222", linewidth=0.8)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("P13 minus P9 fold return")
    ax.set_xlabel("fold")
    ax.set_ylabel("delta return")
    fig.tight_layout()
    fig.savefig(output_dir / "fold_delta_vs_p9.png", dpi=160)
    plt.close(fig)


def plot_execution_attr(output_dir: Path, attr: pd.DataFrame) -> None:
    if attr.empty:
        return
    fig, ax = plt.subplots(figsize=(11, max(4, 0.35 * len(attr))))
    ordered = attr.sort_values("turnover_saved_sum")
    ax.barh(ordered["selected_overlay"], ordered["turnover_saved_sum"])
    ax.set_title("Turnover saved by selected overlay")
    ax.set_xlabel("turnover saved")
    fig.tight_layout()
    fig.savefig(output_dir / "turnover_saved_by_overlay.png", dpi=160)
    plt.close(fig)


def write_report(
    output_dir: Path,
    summary: pd.DataFrame,
    cost_summary: pd.DataFrame,
    selected: pd.DataFrame,
    validation: pd.DataFrame,
    p9_validation: pd.DataFrame,
    fold_delta: pd.DataFrame,
    bootstrap: pd.DataFrame,
    start_vs_p9: pd.DataFrame,
    dropout_vs_p9: pd.DataFrame,
    execution_attr: pd.DataFrame,
    fixed_summary: pd.DataFrame,
    fixed_cost: pd.DataFrame,
    reality: pd.DataFrame,
    horizon: int,
    validation_days: int,
) -> None:
    accepted = bool(reality["pass"].all())
    lines = [
        "# P13 Turnover-Aware No-Trade Zone / Hysteresis Rebalance",
        "",
        "## Design",
        "",
        "- Fixed P9 alpha, formula selections, strict-neutral allocator, and test folds.",
        "- Separates target generation from execution target.",
        "- Tests wider exit hysteresis, cost-aware no-trade gate, partial rebalance, turnover budget, and combined overlays.",
        "- Overlay parameters are selected only on each fold's historical train/validation window.",
        f"- Horizon: `{horizon}h`; inner validation: `{validation_days}d`.",
        f"- Accepted as new baseline: `{accepted}`.",
        "",
        "## Summary",
        "```text",
        summary.to_string(index=False),
        "```",
        "",
        "## Reality Summary",
        "```text",
        reality.to_string(index=False),
        "```",
        "",
        "## Cost Sensitivity",
        "```text",
        cost_summary.to_string(index=False),
        "```",
        "",
        "## Fixed Matched Overlay Summary",
        "```text",
        fixed_summary.to_string(index=False),
        "```",
        "",
        "## Fixed Matched Overlay Cost Sensitivity",
        "```text",
        fixed_cost.to_string(index=False),
        "```",
        "",
        "## Selected Overlay By Fold",
        "```text",
        selected.to_string(index=False),
        "```",
        "",
        "## Execution Attribution",
        "```text",
        execution_attr.to_string(index=False),
        "```",
        "",
        "## Fold Delta Vs P9",
        "```text",
        fold_delta.to_string(index=False),
        "```",
        "",
        "## Bootstrap",
        "```text",
        bootstrap.to_string(index=False),
        "```",
        "",
        "## Start Fold Sensitivity",
        "```text",
        start_vs_p9.to_string(index=False),
        "```",
        "",
        "## Symbol Dropout",
        "```text",
        dropout_vs_p9.to_string(index=False),
        "```",
        "",
        "## Validation Metrics",
        "```text",
        validation.to_string(index=False),
        "```",
        "",
        "## P9 Validation Baseline",
        "```text",
        p9_validation.to_string(index=False),
        "```",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def update_config_stage(config_path: Path) -> None:
    cfg = load_config(config_path)
    stages = cfg.setdefault("stages", {})
    stages["p13_turnover_no_trade_zone"] = {
        "script": "../../../scripts/run_p13_turnover_no_trade_zone.py",
        "output_dir": "../outputs/p13_turnover_no_trade_zone",
        "report": "../outputs/p13_turnover_no_trade_zone/REPORT.md",
        "summary": "../outputs/p13_turnover_no_trade_zone/variant_summary.csv",
        "cost_sensitivity": "../outputs/p13_turnover_no_trade_zone/cost_sensitivity.csv",
        "selected_turnover_overlay_by_fold": "../outputs/p13_turnover_no_trade_zone/selected_turnover_overlay_by_fold.csv",
    }
    config_path.write_text(json_to_yaml(cfg), encoding="utf-8")


def json_to_yaml(data: dict[str, Any]) -> str:
    import yaml

    return yaml.safe_dump(data, sort_keys=False, allow_unicode=False)


if __name__ == "__main__":
    main()
