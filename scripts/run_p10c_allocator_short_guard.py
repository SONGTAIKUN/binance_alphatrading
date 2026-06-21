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
import yaml
from matplotlib.ticker import PercentFormatter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from crypto_stat_arb.config import ensure_dir, load_config
from crypto_stat_arb.portfolio import portfolio_exposures, spearman_ic, update_state
import run_p3d_p4_gated_rerank_alphagpt as p3d
from run_p3_committee_guard_scan import apply_short_guard, side_fraction_weights
import run_p7_derivatives_gated_rerank_alphagpt as p7
import run_p7b_derivatives_incremental_on_p3d as p7b
import run_p9_clean_derivatives_baseline as p9
import run_p10_strategy_conditioned_regime_gate as p10
import run_p10b_asymmetric_squeeze_short_guard as p10b
from run_turnover_experiments import to_jsonable


EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "baseline_locked_v1"
DEFAULT_CONFIG = EXPERIMENT_ROOT / "configs" / "baseline_locked_v1.yaml"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "outputs" / "p10c_allocator_short_guard"
P3D_DIR = EXPERIMENT_ROOT / "outputs" / "p3d_p4_gated_rerank_alphagpt" / "nested_incremental_run"
P7B_DIR = EXPERIMENT_ROOT / "outputs" / "p7b_derivatives_incremental_on_p3d" / "nested_incremental_run"
P9_DIR = EXPERIMENT_ROOT / "outputs" / "p9_clean_baseline_package" / "nested_incremental_run"
P9_SELECTED = EXPERIMENT_ROOT / "outputs" / "p9_clean_baseline_package" / "selected_incremental_by_fold.csv"
DERIV_PANEL = EXPERIMENT_ROOT / "inputs" / "binance_derivatives" / "derivatives_hourly_panel.csv.gz"
NEUTRALIZE_COLS = ["intercept", "beta_btc", "beta_eth", "beta_market"]


@dataclass(frozen=True)
class AllocatorGuardSpec:
    name: str
    description: str
    guard_mode: str
    allocator_mode: str
    risk_regimes: tuple[str, ...] = ("crowded_trend", "deleveraging_squeeze")
    moderate_threshold: float = 0.68
    high_threshold: float = 0.80
    moderate_multiplier: float = 0.65
    high_multiplier: float = 0.40
    protect_strong_short: bool = True
    strong_short_score: float = -0.45
    strong_short_min_multiplier: float = 0.85
    soft_exposure_penalty: float = 80.0
    soft_hedge_penalty: float = 100.0
    max_abs_weight: float = 0.125


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--horizon", type=int, default=72)
    parser.add_argument("--cost-bps", type=float, nargs="+", default=[0, 4, 6, 8, 10, 12, 15, 20, 30])
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    output_dir = ensure_dir(args.output_dir)
    horizon = int(args.horizon)
    cost_grid = [float(x) for x in args.cost_bps]

    print("[1/8] Loading P9 selections, references, and source panels", flush=True)
    p9_result = p10.load_existing_result(P9_DIR)
    p7b_result = p10.load_existing_result(P7B_DIR)
    p3d_result = p10.load_existing_result(P3D_DIR)
    selections_path = p3d.resolve_config_path(config_path, cfg["inputs"]["formula_selections"])
    selections = pd.read_csv(selections_path, parse_dates=["train_start", "train_end", "test_start", "test_end"])
    selected = pd.read_csv(P9_SELECTED, parse_dates=["train_start", "train_end", "test_start", "test_end", "test_decision_end"])

    panel_path = p3d.resolve_config_path(config_path, cfg["inputs"]["panel"])
    retained = p7.load_retained_features()
    raw_deriv_formulas = p7.build_p7_formula_universe(retained, 160, 2607, 3)
    deriv_formulas = {item.name: item for item in p9.filter_p9_formulas(raw_deriv_formulas, exclude_k01=True)}
    p3d_formulas = {item.name: item for item in p3d.build_incremental_formula_universe(180, 2417, 3)}
    pd.DataFrame([item.to_row() for item in deriv_formulas.values()]).to_csv(output_dir / "candidate_formulas.csv", index=False)

    print("[2/8] Building ex-ante regime and symbol squeeze risk features", flush=True)
    panel = p7.load_panel_p7(config_path, panel_path, p3d.MANUAL_FEATURE_PANEL, horizon)
    asset_regime = p10.load_asset_panel(panel_path)
    deriv_regime = p10.load_derivatives_panel(DERIV_PANEL)
    regime = p10.build_regime_panel(asset_regime, deriv_regime)
    asset_risk = p10b.load_asset_risk_panel(panel_path)
    deriv_risk = p10b.load_derivative_risk_panel(DERIV_PANEL)
    risk_panel = p10b.build_short_squeeze_risk_panel(asset_risk, deriv_risk)
    panel = panel.merge(
        regime[["timestamp", "regime"]],
        on="timestamp",
        how="left",
    )
    risk_cols = [
        "timestamp",
        "symbol",
        "squeeze_short_risk",
        "squeeze_momentum_up",
        "squeeze_long_pressure",
        "squeeze_abs_crowding",
        "squeeze_oi_pressure",
    ]
    panel = panel.merge(risk_panel[risk_cols], on=["timestamp", "symbol"], how="left")
    panel["regime"] = panel["regime"].fillna("normal")
    panel["squeeze_short_risk"] = panel["squeeze_short_risk"].fillna(0.5)
    regime.to_csv(output_dir / "regime_feature_panel.csv", index=False)
    risk_panel.to_csv(output_dir / "short_squeeze_risk_panel.csv", index=False)

    print("[3/8] Replaying P9 formulas with allocator-native short guard variants", flush=True)
    pf_cfg = p3d.make_portfolio_config(cfg, horizon)
    risk_template = p3d.make_risk_template(cfg)
    spec_by_name = {spec.name: spec for spec in p3d.build_committee_specs()}
    specs = build_allocator_guard_specs()
    pd.DataFrame([spec.__dict__ for spec in specs]).to_csv(output_dir / "allocator_guard_specs.csv", index=False)
    results: dict[str, dict[str, Any]] = {
        "p3d_reference": p3d_result,
        "p7b_reference": p7b_result,
        "p9_reference": p9_result,
    }
    for spec in specs:
        print(f"      {spec.name}", flush=True)
        result = replay_p9_with_allocator_guard(
            panel=panel,
            selections=selections,
            selected=selected,
            pf_cfg=pf_cfg,
            risk_template=risk_template,
            spec_by_name=spec_by_name,
            deriv_formulas=deriv_formulas,
            p3d_formulas=p3d_formulas,
            guard_spec=spec,
        )
        results[spec.name] = result
        write_result(ensure_dir(output_dir / "runs" / spec.name), result)

    print("[4/8] Building summary, cost, and attribution diagnostics", flush=True)
    summary = p10.build_summary(results, horizon)
    summary.to_csv(output_dir / "variant_summary.csv", index=False)
    cost_summary = p10.build_cost_summary(results, cost_grid)
    cost_summary.to_csv(output_dir / "cost_sensitivity.csv", index=False)
    fold_delta = p10.build_fold_delta_table(results, "p9_reference")
    fold_delta.to_csv(output_dir / "fold_delta_vs_p9.csv", index=False)
    regime_attr = p10.build_regime_attribution(results, regime)
    regime_attr.to_csv(output_dir / "regime_attribution.csv", index=False)
    stress_attr = p10.build_stress_period_attribution(results, regime)
    stress_attr.to_csv(output_dir / "stress_period_attribution.csv", index=False)
    guard_events = build_guard_event_attribution(results)
    guard_events.to_csv(output_dir / "guard_event_attribution.csv", index=False)
    exposure_summary = build_exposure_summary(results)
    exposure_summary.to_csv(output_dir / "exposure_summary.csv", index=False)

    print("[5/8] Running robustness checks", flush=True)
    best_name = select_best_variant(summary, cost_summary, exposure_summary)
    best_result = results[best_name]
    p9_fold_delta = p9.build_fold_delta(best_result, p9_result, best_name, "p9")
    p3d_fold_delta = p9.build_fold_delta(best_result, p3d_result, best_name, "p3d")
    bootstrap = pd.concat(
        [
            p9.bootstrap_fold_checks(p9_fold_delta, f"{best_name}_minus_p9", int(args.bootstrap_samples)),
            p9.bootstrap_fold_checks(p3d_fold_delta, f"{best_name}_minus_p3d", int(args.bootstrap_samples)),
        ],
        ignore_index=True,
    )
    bootstrap.to_csv(output_dir / "bootstrap_checks.csv", index=False)
    start_vs_p9 = p9.build_start_fold_sensitivity(best_result, p9_result, best_name, "p9", horizon)
    start_vs_p9.to_csv(output_dir / "start_fold_sensitivity_vs_p9.csv", index=False)
    dropout_vs_p9 = p9.build_symbol_dropout_checks(best_result, p9_result, best_name, "p9", horizon)
    dropout_vs_p9.to_csv(output_dir / "symbol_dropout_vs_p9.csv", index=False)
    reality = build_reality_summary(best_name, summary, cost_summary, bootstrap, start_vs_p9, dropout_vs_p9)
    reality.to_csv(output_dir / "reality_summary.csv", index=False)

    print("[6/8] Writing plots", flush=True)
    plot_net_curves(output_dir, results, best_name)
    plot_cost_curves(output_dir, cost_summary, best_name)
    plot_variant_bars(output_dir, summary)
    plot_fold_delta(output_dir, fold_delta, best_name)
    plot_guard_events(output_dir, guard_events, best_name)

    print("[7/8] Writing report and config metadata", flush=True)
    write_report(
        output_dir=output_dir,
        specs=specs,
        best_name=best_name,
        summary=summary,
        cost_summary=cost_summary,
        exposure_summary=exposure_summary,
        fold_delta=fold_delta,
        regime_attr=regime_attr,
        stress_attr=stress_attr,
        guard_events=guard_events,
        bootstrap=bootstrap,
        start_vs_p9=start_vs_p9,
        dropout_vs_p9=dropout_vs_p9,
        reality=reality,
    )
    update_config_stage(config_path, best_name)
    print("[8/8] Done", flush=True)
    print(f"Done: {output_dir}", flush=True)


def build_allocator_guard_specs() -> list[AllocatorGuardSpec]:
    return [
        AllocatorGuardSpec(
            name="p10c_refill_strict_crowded_delev",
            description="Skip vulnerable shorts in crowded/deleveraging candidate pool, refill from next eligible shorts, then use original strict allocator.",
            guard_mode="refill",
            allocator_mode="strict",
        ),
        AllocatorGuardSpec(
            name="p10c_refill_soft80_crowded_delev",
            description="Skip vulnerable shorts and refill, then use soft exposure-penalty allocator.",
            guard_mode="refill",
            allocator_mode="soft",
            soft_exposure_penalty=80.0,
        ),
        AllocatorGuardSpec(
            name="p10c_refill_soft250_crowded_delev",
            description="Skip vulnerable shorts and refill, then use stronger soft exposure-penalty allocator.",
            guard_mode="refill",
            allocator_mode="soft",
            soft_exposure_penalty=250.0,
        ),
        AllocatorGuardSpec(
            name="p10c_scale_soft80_crowded_delev",
            description="Keep P9 side selection but scale vulnerable short raw weights before soft exposure-penalty allocation.",
            guard_mode="scale",
            allocator_mode="soft",
            soft_exposure_penalty=80.0,
        ),
        AllocatorGuardSpec(
            name="p10c_scale_soft250_crowded_delev",
            description="Keep P9 side selection but scale vulnerable short raw weights before stronger soft allocation.",
            guard_mode="scale",
            allocator_mode="soft",
            soft_exposure_penalty=250.0,
        ),
        AllocatorGuardSpec(
            name="p10c_scale_strict_crowded_delev",
            description="Keep P9 side selection, scale vulnerable shorts, then use original strict allocator.",
            guard_mode="scale",
            allocator_mode="strict",
        ),
        AllocatorGuardSpec(
            name="p10c_refill_strict_crowded_only",
            description="Skip/refill vulnerable shorts only in crowded trend, then use original strict allocator.",
            guard_mode="refill",
            allocator_mode="strict",
            risk_regimes=("crowded_trend",),
        ),
    ]


def replay_p9_with_allocator_guard(
    panel: pd.DataFrame,
    selections: pd.DataFrame,
    selected: pd.DataFrame,
    pf_cfg: Any,
    risk_template: Any,
    spec_by_name: dict[str, Any],
    deriv_formulas: dict[str, Any],
    p3d_formulas: dict[str, Any],
    guard_spec: AllocatorGuardSpec,
) -> dict[str, Any]:
    fold_rows = []
    equity_parts = []
    position_parts = []
    selected_by_fold = selected.set_index("fold")
    for _, fold in selections.sort_values("fold").iterrows():
        fold_id = int(fold["fold"])
        row = selected_by_fold.loc[fold_id]
        committee = spec_by_name[str(row["p4_spec"])]
        p7b.compute_base_score(panel, row, committee, p3d_formulas)
        if str(row["deriv_formula_name"]) == p3d.P4_ONLY_FORMULA:
            panel[p3d.SCORE_COL] = panel[p7b.BASE_SCORE_COL]
            panel[p3d.CANDIDATE_RANK_COL] = np.nan
            formula_text = f"P3d_base({row['p3d_formula_name']}) fallback"
        else:
            formula = deriv_formulas[str(row["deriv_formula_name"])]
            p7b.apply_derivative_increment(
                panel,
                formula,
                float(row["candidate_weight_long"]),
                float(row["candidate_weight_short"]),
                float(row["high_risk_candidate_cap"]),
            )
            formula_text = f"P3d_base({row['p3d_formula_name']}) + deriv({row['deriv_formula_name']},w={float(row['candidate_weight_long']):.2f}): {formula.formula}"
        risk_cfg = risk_template.__class__(
            **{
                **risk_template.__dict__,
                "name": f"p10c_{guard_spec.name}_{committee.name}",
                "short_resid_confirm_rank": committee.short_resid_confirm_rank,
            }
        )
        equity, positions, metrics = run_allocator_guard_backtest(
            panel=panel,
            start=fold["test_start"],
            end=p3d.decision_end_for_closed_positions(fold["test_end"], pf_cfg.primary_horizon),
            cfg=pf_cfg,
            risk_cfg=risk_cfg,
            committee=committee,
            guard_spec=guard_spec,
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
                "p4_spec": committee.name,
                "p3d_formula_name": str(row["p3d_formula_name"]),
                "deriv_formula_name": str(row["deriv_formula_name"]),
                "formula": formula_text,
                "candidate_weight_long": float(row["candidate_weight_long"]),
                "candidate_weight_short": float(row["candidate_weight_short"]),
                "high_risk_candidate_cap": float(row["high_risk_candidate_cap"]),
                "guard_spec": guard_spec.name,
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
            equity["p4_spec"] = committee.name
            equity["formula_name"] = str(row["deriv_formula_name"])
            equity["formula"] = formula_text
            equity["guard_spec"] = guard_spec.name
            positions["fold"] = fold_id
            positions["p4_spec"] = committee.name
            positions["formula_name"] = str(row["deriv_formula_name"])
            positions["formula"] = formula_text
            positions["guard_spec"] = guard_spec.name
            equity_parts.append(equity)
            position_parts.append(positions)
    equity_curve = pd.concat(equity_parts, ignore_index=True) if equity_parts else pd.DataFrame()
    positions = pd.concat(position_parts, ignore_index=True) if position_parts else pd.DataFrame()
    folds = pd.DataFrame(fold_rows)
    metrics = p3d.summarize_nested_result({"equity": equity_curve, "positions": positions, "folds": folds}, pf_cfg.rebalance_hours)
    return {"folds": folds, "equity": equity_curve, "positions": positions, "metrics": metrics}


def run_allocator_guard_backtest(
    panel: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cfg: Any,
    risk_cfg: Any,
    committee: Any,
    guard_spec: AllocatorGuardSpec,
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
        regime_name = str(rows["regime"].iloc[0]) if "regime" in rows else "normal"

        if guard_spec.guard_mode == "refill":
            target_sides, blocked = select_p4_gated_target_sides_refill(rows, current_state, cfg, score_col, guard_spec, regime_name)
        else:
            target_sides = p3d.select_p4_gated_target_sides(rows, current_state, cfg, score_col)
            blocked = []
        before_guard = len([side for side in target_sides.values() if side < 0])
        target_sides = apply_short_guard(target_sides, rows_by_symbol, committee)
        after_guard = len([side for side in target_sides.values() if side < 0])
        if not any(side > 0 for side in target_sides.values()) or not any(side < 0 for side in target_sides.values()):
            continue

        state_sides = target_sides.copy()
        selected = rows_by_symbol.reindex(list(target_sides)).reset_index()
        raw_weights = side_fraction_weights(target_sides, cfg.gross_exposure, committee.long_fraction)
        pre_guard_raw_weights = raw_weights.copy()
        multipliers = build_short_guard_multipliers(raw_weights, rows_by_symbol, guard_spec, regime_name)
        if guard_spec.guard_mode == "scale":
            raw_weights = raw_weights * multipliers

        target_weights, optimizer_info = build_allocator_weights(raw_weights, selected, rows_by_symbol, cfg, risk_cfg, guard_spec)
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
        selected_short_risk = rows_by_symbol.loc[[s for s, side in target_sides.items() if side < 0], "squeeze_short_risk"]
        guarded_short_count = int((multipliers[multipliers < 1.0]).count()) if guard_spec.guard_mode == "scale" else len(blocked)

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
                "allocator_short_guard_count": float(guarded_short_count),
                "allocator_blocked_short_count": float(len(blocked)),
                "mean_selected_short_squeeze_risk": float(selected_short_risk.mean()) if len(selected_short_risk) else np.nan,
                "regime": regime_name,
                **optimizer_info,
                "target_applied": 1.0,
                "selected_formulas": selected_formulas,
            }
        )

        for symbol, weight in target_weights.items():
            row = rows_by_symbol.loc[symbol]
            pre_raw = float(pre_guard_raw_weights.get(symbol, 0.0))
            multiplier = float(multipliers.get(symbol, 1.0))
            pos_out.append(
                {
                    "timestamp": pd.Timestamp(ts),
                    "exit_timestamp": pd.Timestamp(ts) + pd.Timedelta(hours=cfg.execution_delay_bars + cfg.primary_horizon),
                    "symbol": symbol,
                    "weight": float(weight),
                    "target_weight": float(target_full_weights.loc[symbol]),
                    "pre_guard_raw_weight": pre_raw,
                    "raw_weight_after_guard": float(raw_weights.get(symbol, 0.0)),
                    "short_guard_multiplier": multiplier,
                    "squeeze_short_risk": float(row.get("squeeze_short_risk", np.nan)),
                    "regime": regime_name,
                    "score": float(row[score_col]),
                    "p4_score": float(row[p3d.P4_SCORE_COL]),
                    "candidate_rank": float(row[p3d.CANDIDATE_RANK_COL]) if p3d.CANDIDATE_RANK_COL in rows_by_symbol else np.nan,
                    "p4_rank": float(row["p4_rank"]),
                    "ema_rank": float(row["ema_rank"]),
                    "base_rank": float(row["base_rank"]),
                    "rank_a07": float(row["rank_a07"]),
                    "rank_b11": float(row["rank_b11"]),
                    "rank_c09": float(row["rank_c09"]),
                    "fwd_simple_ret": float(row[ret_col]),
                    "label_resid_ret": float(row[label_col]),
                    "contribution": float(weight * row[ret_col]),
                    "raw_guard_benefit": float((raw_weights.get(symbol, 0.0) - pre_raw) * row[ret_col]),
                    "selected_score_cols": "p4_score | candidate_rerank_score",
                }
            )

        for symbol in blocked:
            row = rows_by_symbol.loc[symbol]
            pre_raw = -cfg.gross_exposure * (1.0 - committee.long_fraction) / max(cfg.short_k, 1)
            pos_out.append(
                {
                    "timestamp": pd.Timestamp(ts),
                    "exit_timestamp": pd.Timestamp(ts) + pd.Timedelta(hours=cfg.execution_delay_bars + cfg.primary_horizon),
                    "symbol": symbol,
                    "weight": 0.0,
                    "target_weight": 0.0,
                    "pre_guard_raw_weight": float(pre_raw),
                    "raw_weight_after_guard": 0.0,
                    "short_guard_multiplier": 0.0,
                    "squeeze_short_risk": float(row.get("squeeze_short_risk", np.nan)),
                    "regime": regime_name,
                    "score": float(row[score_col]),
                    "p4_score": float(row[p3d.P4_SCORE_COL]),
                    "candidate_rank": float(row[p3d.CANDIDATE_RANK_COL]) if p3d.CANDIDATE_RANK_COL in rows_by_symbol else np.nan,
                    "p4_rank": float(row["p4_rank"]),
                    "ema_rank": float(row["ema_rank"]),
                    "base_rank": float(row["base_rank"]),
                    "rank_a07": float(row["rank_a07"]),
                    "rank_b11": float(row["rank_b11"]),
                    "rank_c09": float(row["rank_c09"]),
                    "fwd_simple_ret": float(row[ret_col]),
                    "label_resid_ret": float(row[label_col]),
                    "contribution": 0.0,
                    "raw_guard_benefit": float(-pre_raw * row[ret_col]),
                    "selected_score_cols": "blocked_by_allocator_short_guard",
                }
            )

        prev_weights = target_full_weights
        current_state = update_state(current_state, state_sides, True)

    equity_curve = pd.DataFrame(rows_out)
    positions = pd.DataFrame(pos_out)
    metrics = p3d.aggregate_oos_metrics(equity_curve, cfg.rebalance_hours)
    return equity_curve, positions, metrics


def select_p4_gated_target_sides_refill(
    rows: pd.DataFrame,
    current_state: dict[str, tuple[int, int]],
    cfg: Any,
    score_col: str,
    guard_spec: AllocatorGuardSpec,
    regime_name: str,
) -> tuple[dict[str, int], list[str]]:
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
    blocked: list[str] = []
    if regime_name in set(guard_spec.risk_regimes):
        kept_forced = []
        for symbol in forced_shorts:
            if short_is_vulnerable(symbol, rows_by_symbol, guard_spec):
                blocked.append(symbol)
            else:
                kept_forced.append(symbol)
        forced_shorts = kept_forced

    longs = p3d.dedupe_keep_order(forced_longs)
    shorts = p3d.dedupe_keep_order(forced_shorts)

    if cfg.use_hysteresis:
        held_longs = [
            symbol
            for symbol, (side, _) in current_state.items()
            if side > 0 and symbol in valid_symbols and rows_by_symbol.loc[symbol, "p4_rank"] >= cfg.long_exit_rank and symbol not in longs
        ]
        held_shorts = [
            symbol
            for symbol, (side, _) in current_state.items()
            if side < 0 and symbol in valid_symbols and rows_by_symbol.loc[symbol, "p4_rank"] <= cfg.short_exit_rank and symbol not in shorts
        ]
        if regime_name in set(guard_spec.risk_regimes):
            next_held = []
            for symbol in held_shorts:
                if short_is_vulnerable(symbol, rows_by_symbol, guard_spec):
                    blocked.append(symbol)
                else:
                    next_held.append(symbol)
            held_shorts = next_held
        longs.extend(p3d.sort_symbols_by_score(held_longs, rows_by_symbol, score_col, ascending=False))
        shorts.extend(p3d.sort_symbols_by_score(held_shorts, rows_by_symbol, score_col, ascending=True))
        excluded = set(longs) | set(shorts)
        long_pool = rows[(rows["p4_rank"] >= cfg.long_entry_rank) & ~rows["symbol"].isin(excluded)].sort_values(score_col, ascending=False)
        short_pool = rows[(rows["p4_rank"] <= cfg.short_entry_rank) & ~rows["symbol"].isin(excluded)].sort_values(score_col, ascending=True)
    else:
        excluded = set(longs) | set(shorts)
        long_pool = rows[(rows["p4_rank"] >= cfg.long_entry_rank) & ~rows["symbol"].isin(excluded)].sort_values(score_col, ascending=False)
        short_pool = rows[(rows["p4_rank"] <= cfg.short_entry_rank) & ~rows["symbol"].isin(excluded)].sort_values(score_col, ascending=True)

    for symbol in long_pool["symbol"]:
        if len(longs) >= cfg.long_k:
            break
        if symbol not in shorts:
            longs.append(symbol)

    for symbol in short_pool["symbol"]:
        if len(shorts) >= cfg.short_k:
            break
        if symbol in longs:
            continue
        if regime_name in set(guard_spec.risk_regimes) and short_is_vulnerable(symbol, rows_by_symbol, guard_spec):
            blocked.append(symbol)
            continue
        shorts.append(symbol)

    longs = p3d.sort_symbols_by_score(longs, rows_by_symbol, score_col, ascending=False)[: cfg.long_k]
    shorts = p3d.sort_symbols_by_score(shorts, rows_by_symbol, score_col, ascending=True)[: cfg.short_k]
    sides = {symbol: 1 for symbol in longs}
    sides.update({symbol: -1 for symbol in shorts if symbol not in sides})
    return sides, p3d.dedupe_keep_order(blocked)


def short_is_vulnerable(symbol: str, rows_by_symbol: pd.DataFrame, guard_spec: AllocatorGuardSpec) -> bool:
    risk = float(rows_by_symbol.loc[symbol].get("squeeze_short_risk", 0.5))
    if not np.isfinite(risk) or risk < guard_spec.moderate_threshold:
        return False
    if guard_spec.protect_strong_short and float(rows_by_symbol.loc[symbol].get(p3d.SCORE_COL, 0.0)) <= guard_spec.strong_short_score:
        return risk >= guard_spec.high_threshold
    return True


def build_short_guard_multipliers(
    raw_weights: pd.Series,
    rows_by_symbol: pd.DataFrame,
    guard_spec: AllocatorGuardSpec,
    regime_name: str,
) -> pd.Series:
    out = pd.Series(1.0, index=raw_weights.index, dtype=float)
    if regime_name not in set(guard_spec.risk_regimes):
        return out
    for symbol, weight in raw_weights.items():
        if weight >= 0:
            continue
        risk = float(rows_by_symbol.loc[symbol].get("squeeze_short_risk", 0.5))
        if not np.isfinite(risk):
            risk = 0.5
        multiplier = 1.0
        if risk >= guard_spec.high_threshold:
            multiplier = guard_spec.high_multiplier
        elif risk >= guard_spec.moderate_threshold:
            multiplier = guard_spec.moderate_multiplier
        if guard_spec.protect_strong_short and float(rows_by_symbol.loc[symbol].get(p3d.SCORE_COL, 0.0)) <= guard_spec.strong_short_score:
            multiplier = max(multiplier, guard_spec.strong_short_min_multiplier)
        out.loc[symbol] = multiplier
    return out


def build_allocator_weights(
    raw_weights: pd.Series,
    selected: pd.DataFrame,
    rows_by_symbol: pd.DataFrame,
    cfg: Any,
    risk_cfg: Any,
    guard_spec: AllocatorGuardSpec,
) -> tuple[pd.Series, dict[str, float]]:
    if guard_spec.allocator_mode == "strict":
        from run_risk_control_experiments import build_weights

        return build_weights(raw_weights, selected, rows_by_symbol, cfg, risk_cfg)
    return soft_project_weights(
        target_weights=raw_weights,
        universe=rows_by_symbol,
        exposure_cols=NEUTRALIZE_COLS,
        max_abs_weight=guard_spec.max_abs_weight,
        gross_limit=float(raw_weights.abs().sum()),
        selected_symbols=set(raw_weights.index),
        fixed_signs={symbol: int(np.sign(weight)) for symbol, weight in raw_weights.items() if abs(weight) > 1e-12},
        exposure_penalty=guard_spec.soft_exposure_penalty,
        hedge_penalty=guard_spec.soft_hedge_penalty,
    )


def soft_project_weights(
    target_weights: pd.Series,
    universe: pd.DataFrame,
    exposure_cols: list[str],
    max_abs_weight: float,
    gross_limit: float,
    selected_symbols: set[str],
    fixed_signs: dict[str, int],
    exposure_penalty: float,
    hedge_penalty: float,
) -> tuple[pd.Series, dict[str, float]]:
    from scipy.optimize import minimize

    symbols = list(universe.index)
    if not symbols or gross_limit <= 1e-12:
        out = pd.Series(0.0, index=symbols)
        return out, optimizer_info(False, gross_limit, 0.0, np.nan, 0)

    target = target_weights.reindex(symbols).fillna(0.0).to_numpy(dtype=float)
    exposure_arrays = []
    for col in exposure_cols:
        if col == "intercept":
            exposure_arrays.append(np.ones(len(symbols)))
        elif col in universe.columns:
            exposure_arrays.append(universe[col].fillna(0.0).to_numpy(dtype=float))
    exposure_matrix = np.vstack(exposure_arrays) if exposure_arrays else np.empty((0, len(symbols)))

    bounds = []
    penalties = np.full(len(symbols), float(hedge_penalty), dtype=float)
    for idx, symbol in enumerate(symbols):
        side = fixed_signs.get(symbol, 0)
        if side > 0:
            bounds.append((0.0, max_abs_weight))
            penalties[idx] = 1.0
        elif side < 0:
            bounds.append((-max_abs_weight, 0.0))
            penalties[idx] = 1.0
        else:
            bounds.append((-max_abs_weight, max_abs_weight))
            if symbol in selected_symbols:
                penalties[idx] = 1.0
    x0 = np.clip(target, -max_abs_weight, max_abs_weight)
    gross = float(np.abs(x0).sum())
    if gross > gross_limit and gross > 0:
        x0 = x0 * (gross_limit / gross)

    def objective(x: np.ndarray) -> float:
        tracking = float(np.sum(penalties * (x - target) ** 2))
        exposure = float(np.sum((exposure_matrix @ x) ** 2)) if exposure_matrix.size else 0.0
        return tracking + float(exposure_penalty) * exposure

    constraints = [{"type": "ineq", "fun": lambda x, limit=gross_limit: float(limit - np.abs(x).sum())}]
    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-12, "disp": False},
    )
    candidate = result.x if result.x is not None else x0
    gross = float(np.abs(candidate).sum())
    if (not bool(result.success)) or gross > gross_limit + 1e-7:
        candidate = x0
        gross = float(np.abs(candidate).sum())
    exposure_error = float(np.max(np.abs(exposure_matrix @ candidate))) if exposure_matrix.size else 0.0
    out = pd.Series(candidate, index=symbols)
    out = out[out.abs() > 1e-12]
    return out, optimizer_info(bool(result.success), gross_limit, float(out.abs().sum()), exposure_error, 1)


def optimizer_info(success: bool, target_gross: float, actual_gross: float, exposure_error: float, attempts: int) -> dict[str, float]:
    return {
        "optimizer_success": float(success),
        "optimizer_target_gross": float(target_gross),
        "optimizer_actual_gross": float(actual_gross),
        "optimizer_exposure_error": float(exposure_error),
        "optimizer_attempts": float(attempts),
    }


def build_guard_event_attribution(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for name, result in results.items():
        if not name.startswith("p10c_"):
            continue
        positions = result["positions"].copy()
        if "raw_guard_benefit" not in positions:
            continue
        events = positions[(positions["pre_guard_raw_weight"] < 0) & (positions["short_guard_multiplier"] < 1.0)].copy()
        if events.empty:
            continue
        grouped = (
            events.groupby(["regime", "symbol"])
            .agg(
                event_count=("symbol", "count"),
                mean_squeeze_short_risk=("squeeze_short_risk", "mean"),
                raw_guard_benefit_sum=("raw_guard_benefit", "sum"),
                post_allocator_contribution_sum=("contribution", "sum"),
                mean_pre_guard_raw_weight=("pre_guard_raw_weight", "mean"),
                mean_final_weight=("weight", "mean"),
            )
            .reset_index()
        )
        grouped["variant"] = name
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_exposure_summary(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for name, result in results.items():
        equity = result["equity"]
        rows.append(
            {
                "variant": name,
                "mean_abs_beta_btc": float(equity["abs_beta_btc"].mean()) if "abs_beta_btc" in equity else np.nan,
                "mean_abs_beta_eth": float(equity["abs_beta_eth"].mean()) if "abs_beta_eth" in equity else np.nan,
                "mean_abs_beta_market": float(equity["abs_beta_market"].mean()) if "abs_beta_market" in equity else np.nan,
                "max_abs_beta_btc": float(equity["abs_beta_btc"].max()) if "abs_beta_btc" in equity else np.nan,
                "max_abs_beta_eth": float(equity["abs_beta_eth"].max()) if "abs_beta_eth" in equity else np.nan,
                "max_abs_beta_market": float(equity["abs_beta_market"].max()) if "abs_beta_market" in equity else np.nan,
                "mean_optimizer_exposure_error": float(equity["optimizer_exposure_error"].mean()) if "optimizer_exposure_error" in equity else np.nan,
                "max_optimizer_exposure_error": float(equity["optimizer_exposure_error"].max()) if "optimizer_exposure_error" in equity else np.nan,
            }
        )
    return pd.DataFrame(rows)


def select_best_variant(summary: pd.DataFrame, cost: pd.DataFrame, exposure: pd.DataFrame) -> str:
    candidates = summary[summary["variant"].str.startswith("p10c_")].copy()
    if candidates.empty:
        return "p9_reference"
    c20 = cost[cost["one_way_cost_bps"] == 20.0][["variant", "total_return"]].rename(columns={"total_return": "return_20bps"})
    c30 = cost[cost["one_way_cost_bps"] == 30.0][["variant", "total_return"]].rename(columns={"total_return": "return_30bps"})
    p9_20 = float(c20[c20["variant"] == "p9_reference"]["return_20bps"].iloc[0])
    p9_30 = float(c30[c30["variant"] == "p9_reference"]["return_30bps"].iloc[0])
    candidates = candidates.merge(c20, on="variant", how="left").merge(c30, on="variant", how="left")
    exposure_for_score = exposure[["variant", "mean_abs_beta_market", "max_abs_beta_market"]].rename(
        columns={
            "mean_abs_beta_market": "score_mean_abs_beta_market",
            "max_abs_beta_market": "score_max_abs_beta_market",
        }
    )
    candidates = candidates.merge(exposure_for_score, on="variant", how="left")
    candidates["accept_score"] = (
        120.0 * candidates["delta_vs_p9_total_return"]
        + 2.0 * candidates["delta_vs_p9_sharpe"]
        + 60.0 * candidates["delta_vs_p9_max_drawdown"]
        + 25.0 * (candidates["return_20bps"] - p9_20)
        + 15.0 * (candidates["return_30bps"] - p9_30)
        - 2.0 * candidates["score_mean_abs_beta_market"].fillna(0.0)
        - 0.5 * candidates["score_max_abs_beta_market"].fillna(0.0)
    )
    return str(candidates.sort_values(["accept_score", "total_return"], ascending=False).iloc[0]["variant"])


def build_reality_summary(
    best_name: str,
    summary: pd.DataFrame,
    cost: pd.DataFrame,
    bootstrap: pd.DataFrame,
    start_vs_p9: pd.DataFrame,
    dropout_vs_p9: pd.DataFrame,
) -> pd.DataFrame:
    best = summary[summary["variant"] == best_name].iloc[0]
    p9_row = summary[summary["variant"] == "p9_reference"].iloc[0]
    best_cost20 = cost[(cost["variant"] == best_name) & (cost["one_way_cost_bps"] == 20.0)].iloc[0]
    p9_cost20 = cost[(cost["variant"] == "p9_reference") & (cost["one_way_cost_bps"] == 20.0)].iloc[0]
    best_cost30 = cost[(cost["variant"] == best_name) & (cost["one_way_cost_bps"] == 30.0)].iloc[0]
    p9_cost30 = cost[(cost["variant"] == "p9_reference") & (cost["one_way_cost_bps"] == 30.0)].iloc[0]
    boot_vs_p9 = bootstrap[bootstrap["metric"] == f"{best_name}_minus_p9"].iloc[0]
    start_positive = float((start_vs_p9["delta_total_return"] > 0).mean()) if "delta_total_return" in start_vs_p9 else np.nan
    dropout_positive = float((dropout_vs_p9["delta_total_return"] > 0).mean()) if "delta_total_return" in dropout_vs_p9 else np.nan
    return pd.DataFrame(
        [
            {"check": "best_variant", "value": best_name, "pass": True},
            {"check": "net_delta_vs_p9", "value": best["total_return"] - p9_row["total_return"], "pass": bool(best["total_return"] > p9_row["total_return"])},
            {"check": "sharpe_delta_vs_p9", "value": best["sharpe"] - p9_row["sharpe"], "pass": bool(best["sharpe"] > p9_row["sharpe"])},
            {"check": "maxdd_delta_vs_p9", "value": best["max_drawdown"] - p9_row["max_drawdown"], "pass": bool(best["max_drawdown"] >= p9_row["max_drawdown"] * 1.05)},
            {"check": "return_20bps_delta_vs_p9", "value": best_cost20["total_return"] - p9_cost20["total_return"], "pass": bool(best_cost20["total_return"] >= p9_cost20["total_return"])},
            {"check": "return_30bps_delta_vs_p9", "value": best_cost30["total_return"] - p9_cost30["total_return"], "pass": bool(best_cost30["total_return"] >= p9_cost30["total_return"])},
            {"check": "worst5_delta_vs_p9", "value": best["worst_5_fold_return_sum"] - p9_row["worst_5_fold_return_sum"], "pass": bool(best["worst_5_fold_return_sum"] >= p9_row["worst_5_fold_return_sum"])},
            {"check": "bootstrap_delta_vs_p9_q05", "value": boot_vs_p9["bootstrap_q05"], "pass": bool(boot_vs_p9["bootstrap_q05"] > -0.0025)},
            {"check": "start_fold_positive_vs_p9", "value": start_positive, "pass": bool(start_positive >= 0.45)},
            {"check": "symbol_dropout_positive_vs_p9", "value": dropout_positive, "pass": bool(dropout_positive >= 0.45)},
        ]
    )


def write_result(run_dir: Path, result: dict[str, Any]) -> None:
    result["folds"].to_csv(run_dir / "walk_forward_folds.csv", index=False)
    result["equity"].to_csv(run_dir / "equity_curve.csv", index=False)
    result["positions"].to_csv(run_dir / "positions.csv", index=False)
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(to_jsonable(result["metrics"]), fh, indent=2)


def plot_net_curves(output_dir: Path, results: dict[str, dict[str, Any]], best_name: str) -> None:
    keep = ["p3d_reference", "p7b_reference", "p9_reference", best_name]
    plt.figure(figsize=(13, 7))
    for name in keep:
        equity = results[name]["equity"].sort_values("timestamp")
        net = (1.0 + equity["period_return"]).cumprod() - 1.0
        plt.plot(equity["timestamp"], net, label=name, linewidth=2.1 if name == best_name else 1.4)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.title("P10c allocator-native short guard net curves")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "p10c_net_curves.png", dpi=180)
    plt.close()


def plot_cost_curves(output_dir: Path, cost: pd.DataFrame, best_name: str) -> None:
    keep = ["p9_reference", best_name, "p10c_refill_strict_crowded_delev", "p10c_scale_soft80_crowded_delev", "p10c_refill_soft80_crowded_delev"]
    frame = cost[cost["variant"].isin(keep)].copy()
    plt.figure(figsize=(12, 7))
    for name, group in frame.groupby("variant"):
        group = group.sort_values("one_way_cost_bps")
        plt.plot(group["one_way_cost_bps"], group["total_return"], marker="o", label=name, linewidth=2.2 if name == best_name else 1.4)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.title("P10c cost sensitivity")
    plt.xlabel("one-way cost bps")
    plt.ylabel("total return")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "cost_sensitivity.png", dpi=180)
    plt.close()


def plot_variant_bars(output_dir: Path, summary: pd.DataFrame) -> None:
    frame = summary.sort_values("total_return")
    plt.figure(figsize=(12, 7))
    colors = np.where(frame["variant"].str.startswith("p10c_"), "#2a9d8f", "#777777")
    plt.barh(frame["variant"], frame["total_return"], color=colors)
    plt.axvline(0.0, color="black", linewidth=0.8)
    plt.title("P10c net return at 6bps")
    plt.gca().xaxis.set_major_formatter(PercentFormatter(1.0))
    plt.tight_layout()
    plt.savefig(output_dir / "variant_net_return.png", dpi=180)
    plt.close()


def plot_fold_delta(output_dir: Path, fold_delta: pd.DataFrame, best_name: str) -> None:
    frame = fold_delta[fold_delta["variant"] == best_name].sort_values("fold")
    if frame.empty:
        return
    colors = np.where(frame["delta_total_return"] >= 0.0, "#2a9d8f", "#c44536")
    plt.figure(figsize=(12, 6))
    plt.bar(frame["fold"].astype(str), frame["delta_total_return"], color=colors)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.title(f"{best_name} minus P9 fold return")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.tight_layout()
    plt.savefig(output_dir / "fold_delta_vs_p9.png", dpi=180)
    plt.close()


def plot_guard_events(output_dir: Path, events: pd.DataFrame, best_name: str) -> None:
    frame = events[events["variant"] == best_name].groupby("symbol")["raw_guard_benefit_sum"].sum().sort_values()
    if frame.empty:
        return
    colors = np.where(frame >= 0.0, "#2a9d8f", "#c44536")
    plt.figure(figsize=(10, 6))
    plt.barh(frame.index, frame.values, color=colors)
    plt.axvline(0.0, color="black", linewidth=0.8)
    plt.title(f"{best_name} raw guard benefit by symbol")
    plt.gca().xaxis.set_major_formatter(PercentFormatter(1.0))
    plt.tight_layout()
    plt.savefig(output_dir / "guard_event_benefit.png", dpi=180)
    plt.close()


def write_report(
    output_dir: Path,
    specs: list[AllocatorGuardSpec],
    best_name: str,
    summary: pd.DataFrame,
    cost_summary: pd.DataFrame,
    exposure_summary: pd.DataFrame,
    fold_delta: pd.DataFrame,
    regime_attr: pd.DataFrame,
    stress_attr: pd.DataFrame,
    guard_events: pd.DataFrame,
    bootstrap: pd.DataFrame,
    start_vs_p9: pd.DataFrame,
    dropout_vs_p9: pd.DataFrame,
    reality: pd.DataFrame,
) -> None:
    summary_cols = [
        "variant",
        "total_return",
        "gross_total_return_before_cost",
        "sharpe",
        "max_drawdown",
        "mean_turnover",
        "positive_fold_rate",
        "worst_5_fold_return_sum",
        "delta_vs_p9_total_return",
        "delta_vs_p9_sharpe",
        "delta_vs_p9_max_drawdown",
    ]
    summary_cols = [col for col in summary_cols if col in summary]
    cost_display = cost_summary[cost_summary["one_way_cost_bps"].isin([0, 6, 10, 15, 20, 30])]
    keep = ["p9_reference", best_name, "p10c_refill_strict_crowded_delev", "p10c_scale_soft80_crowded_delev", "p10c_refill_soft80_crowded_delev"]
    cost_display = cost_display[cost_display["variant"].isin(keep)]
    best_events = guard_events[guard_events["variant"] == best_name] if not guard_events.empty else guard_events
    lines = [
        "# P10c Allocator-Native Short Guard",
        "",
        "## Design",
        "",
        "- Fixed P9 nested/OOS formula selections; no new alpha search and no test-informed formula selection.",
        "- The experiment changes only the allocator.",
        "- Refill variants remove vulnerable short candidates before weight construction and refill from the next eligible short candidates.",
        "- Scale variants keep P9 side selection but reduce vulnerable short raw weights before allocation.",
        "- Strict variants use the original P9 strict hedge projection. Soft variants use a soft exposure penalty, allowing small beta drift instead of hard equality.",
        "",
        "## Allocator Specs",
        "```text",
        pd.DataFrame([spec.__dict__ for spec in specs]).to_string(index=False),
        "```",
        "",
        f"## Selected Best Variant: `{best_name}`",
        "",
        "## Summary",
        "```text",
        summary[summary_cols].to_string(index=False),
        "```",
        "",
        "## Cost Sensitivity",
        "```text",
        cost_display.to_string(index=False),
        "```",
        "",
        "## Exposure Summary",
        "```text",
        exposure_summary.to_string(index=False),
        "```",
        "",
        "## Regime Attribution",
        "```text",
        regime_attr.to_string(index=False),
        "```",
        "",
        "## Stress Attribution",
        "```text",
        stress_attr.to_string(index=False),
        "```",
        "",
        "## Guard Event Attribution",
        "Positive raw_guard_benefit means the allocator guard removed/reduced losing short exposure before final allocation.",
        "```text",
        best_events.to_string(index=False),
        "```",
        "",
        "## Fold Delta Vs P9",
        "```text",
        fold_delta[fold_delta["variant"] == best_name].to_string(index=False),
        "```",
        "",
        "## Bootstrap Checks",
        "```text",
        bootstrap.to_string(index=False),
        "```",
        "",
        "## Start Fold Sensitivity Vs P9",
        "```text",
        start_vs_p9.to_string(index=False),
        "```",
        "",
        "## Symbol Dropout Vs P9",
        "```text",
        dropout_vs_p9.to_string(index=False),
        "```",
        "",
        "## Reality Summary",
        "```text",
        reality.to_string(index=False),
        "```",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def update_config_stage(config_path: Path, best_name: str) -> None:
    cfg = load_config(config_path)
    stages = cfg.setdefault("stages", {})
    stages["p10c_allocator_short_guard"] = {
        "script": "../../../scripts/run_p10c_allocator_short_guard.py",
        "output_dir": "../outputs/p10c_allocator_short_guard",
        "report": "../outputs/p10c_allocator_short_guard/REPORT.md",
        "variant_summary": "../outputs/p10c_allocator_short_guard/variant_summary.csv",
        "cost_sensitivity": "../outputs/p10c_allocator_short_guard/cost_sensitivity.csv",
        "reality_summary": "../outputs/p10c_allocator_short_guard/reality_summary.csv",
        "best_variant": best_name,
    }
    cfg["active_research_allocator_short_guard_candidate"] = {
        "name": best_name,
        "status": "candidate_pending_review",
        "output_dir": "../outputs/p10c_allocator_short_guard",
        "report": "../outputs/p10c_allocator_short_guard/REPORT.md",
        "parent": "p9_clean_drop_k01_cap010",
    }
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False), encoding="utf-8")


if __name__ == "__main__":
    main()
