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
import run_p7b_derivatives_incremental_on_p3d as p7b
import run_p9_clean_derivatives_baseline as p9
import run_p10_strategy_conditioned_regime_gate as p10
import run_p10b_asymmetric_squeeze_short_guard as p10b
import run_p10c_allocator_short_guard as p10c
from run_turnover_experiments import to_jsonable


EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "baseline_locked_v1"
DEFAULT_CONFIG = EXPERIMENT_ROOT / "configs" / "baseline_locked_v1.yaml"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "outputs" / "p10d_nested_allocator_short_guard"
P3D_DIR = EXPERIMENT_ROOT / "outputs" / "p3d_p4_gated_rerank_alphagpt" / "nested_incremental_run"
P7B_DIR = EXPERIMENT_ROOT / "outputs" / "p7b_derivatives_incremental_on_p3d" / "nested_incremental_run"
P9_DIR = EXPERIMENT_ROOT / "outputs" / "p9_clean_baseline_package" / "nested_incremental_run"
P9_SELECTED = EXPERIMENT_ROOT / "outputs" / "p9_clean_baseline_package" / "selected_incremental_by_fold.csv"
DERIV_PANEL = EXPERIMENT_ROOT / "inputs" / "binance_derivatives" / "derivatives_hourly_panel.csv.gz"
NO_GUARD = "__P9_NO_ALLOCATOR_GUARD__"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--horizon", type=int, default=72)
    parser.add_argument("--inner-validation-days", type=int, default=90)
    parser.add_argument("--cost-bps", type=float, nargs="+", default=[0, 4, 6, 8, 10, 12, 15, 20, 30])
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    output_dir = ensure_dir(args.output_dir)
    run_dir = ensure_dir(output_dir / "nested_allocator_run")
    horizon = int(args.horizon)
    cost_grid = [float(x) for x in args.cost_bps]

    print("[1/9] Loading P9 selections, references, and panels", flush=True)
    p9_result = p10.load_existing_result(P9_DIR)
    p7b_result = p10.load_existing_result(P7B_DIR)
    p3d_result = p10.load_existing_result(P3D_DIR)
    selections_path = p3d.resolve_config_path(config_path, cfg["inputs"]["formula_selections"])
    selections = pd.read_csv(selections_path, parse_dates=["train_start", "train_end", "test_start", "test_end"])
    selected_p9 = pd.read_csv(P9_SELECTED, parse_dates=["train_start", "train_end", "test_start", "test_end", "test_decision_end"])

    panel_path = p3d.resolve_config_path(config_path, cfg["inputs"]["panel"])
    retained = p7.load_retained_features()
    raw_deriv_formulas = p7.build_p7_formula_universe(retained, 160, 2607, 3)
    deriv_formulas = {item.name: item for item in p9.filter_p9_formulas(raw_deriv_formulas, exclude_k01=True)}
    p3d_formulas = {item.name: item for item in p3d.build_incremental_formula_universe(180, 2417, 3)}
    pd.DataFrame([item.to_row() for item in deriv_formulas.values()]).to_csv(output_dir / "candidate_formulas.csv", index=False)

    print("[2/9] Building ex-ante regime and squeeze-risk features", flush=True)
    panel = p7.load_panel_p7(config_path, panel_path, p3d.MANUAL_FEATURE_PANEL, horizon)
    asset_regime = p10.load_asset_panel(panel_path)
    deriv_regime = p10.load_derivatives_panel(DERIV_PANEL)
    regime = p10.build_regime_panel(asset_regime, deriv_regime)
    asset_risk = p10b.load_asset_risk_panel(panel_path)
    deriv_risk = p10b.load_derivative_risk_panel(DERIV_PANEL)
    risk_panel = p10b.build_short_squeeze_risk_panel(asset_risk, deriv_risk)
    panel = panel.merge(regime[["timestamp", "regime"]], on="timestamp", how="left")
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

    print("[3/9] Building strict-neutral allocator guard grid", flush=True)
    guard_specs = build_nested_guard_grid()
    pd.DataFrame([spec_to_row(spec) for spec in guard_specs]).to_csv(output_dir / "allocator_guard_grid.csv", index=False)

    pf_cfg = p3d.make_portfolio_config(cfg, horizon)
    risk_template = p3d.make_risk_template(cfg)
    spec_by_name = {spec.name: spec for spec in p3d.build_committee_specs()}

    print("[4/9] Running fold-local validation search", flush=True)
    validation, p9_validation = run_nested_validation_search(
        panel=panel,
        selections=selections,
        selected_p9=selected_p9,
        pf_cfg=pf_cfg,
        risk_template=risk_template,
        spec_by_name=spec_by_name,
        deriv_formulas=deriv_formulas,
        p3d_formulas=p3d_formulas,
        guard_specs=guard_specs,
        validation_days=int(args.inner_validation_days),
    )
    validation.to_csv(output_dir / "validation_allocator_metrics.csv", index=False)
    p9_validation.to_csv(output_dir / "validation_p9_allocator_baseline.csv", index=False)

    print("[5/9] Selecting fold-local allocator params without test data", flush=True)
    selected_alloc = select_allocator_by_fold(validation, p9_validation, selections, selected_p9, horizon)
    selected_alloc.to_csv(output_dir / "selected_allocator_by_fold.csv", index=False)

    print("[6/9] Replaying selected allocator params on test windows only", flush=True)
    nested_result = replay_selected_allocators(
        panel=panel,
        selections=selections,
        selected_alloc=selected_alloc,
        selected_p9=selected_p9,
        pf_cfg=pf_cfg,
        risk_template=risk_template,
        spec_by_name=spec_by_name,
        deriv_formulas=deriv_formulas,
        p3d_formulas=p3d_formulas,
    )
    write_result(run_dir, nested_result)

    print("[7/9] Building OOS diagnostics and robustness checks", flush=True)
    results = {
        "p3d_reference": p3d_result,
        "p7b_reference": p7b_result,
        "p9_reference": p9_result,
        "p10d_nested_allocator": nested_result,
    }
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
    exposure_summary = p10c.build_exposure_summary(results)
    exposure_summary.to_csv(output_dir / "exposure_summary.csv", index=False)

    p9_fold_delta = p9.build_fold_delta(nested_result, p9_result, "p10d_nested_allocator", "p9")
    p3d_fold_delta = p9.build_fold_delta(nested_result, p3d_result, "p10d_nested_allocator", "p3d")
    bootstrap = pd.concat(
        [
            p9.bootstrap_fold_checks(p9_fold_delta, "p10d_nested_allocator_minus_p9", int(args.bootstrap_samples)),
            p9.bootstrap_fold_checks(p3d_fold_delta, "p10d_nested_allocator_minus_p3d", int(args.bootstrap_samples)),
        ],
        ignore_index=True,
    )
    bootstrap.to_csv(output_dir / "bootstrap_checks.csv", index=False)
    start_vs_p9 = p9.build_start_fold_sensitivity(nested_result, p9_result, "p10d_nested_allocator", "p9", horizon)
    start_vs_p9.to_csv(output_dir / "start_fold_sensitivity_vs_p9.csv", index=False)
    dropout_vs_p9 = p9.build_symbol_dropout_checks(nested_result, p9_result, "p10d_nested_allocator", "p9", horizon)
    dropout_vs_p9.to_csv(output_dir / "symbol_dropout_vs_p9.csv", index=False)
    reality = build_reality_summary(summary, cost_summary, exposure_summary, bootstrap, start_vs_p9, dropout_vs_p9)
    reality.to_csv(output_dir / "reality_summary.csv", index=False)
    accepted = bool(reality["pass"].all())

    print("[8/9] Writing plots and report", flush=True)
    plot_curves(output_dir, results)
    plot_cost_curves(output_dir, cost_summary)
    plot_fold_delta(output_dir, fold_delta)
    plot_selected_allocators(output_dir, selected_alloc)
    write_report(
        output_dir=output_dir,
        accepted=accepted,
        guard_specs=guard_specs,
        selected_alloc=selected_alloc,
        validation=validation,
        p9_validation=p9_validation,
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

    print("[9/9] Updating config metadata", flush=True)
    update_config_stage(config_path, accepted)
    print(f"Done: {output_dir}", flush=True)


def build_nested_guard_grid() -> list[p10c.AllocatorGuardSpec]:
    specs: list[p10c.AllocatorGuardSpec] = []
    regime_options = [
        ("crowded_only", ("crowded_trend",)),
        ("crowded_delev", ("crowded_trend", "deleveraging_squeeze")),
    ]
    threshold_pairs = [
        (0.66, 0.80),
        (0.68, 0.80),
        (0.70, 0.82),
        (0.72, 0.84),
    ]
    for regime_name, regimes in regime_options:
        for moderate, high in threshold_pairs:
            suffix = f"{regime_name}_m{int(moderate*100):02d}_h{int(high*100):02d}"
            specs.append(
                p10c.AllocatorGuardSpec(
                    name=f"p10d_refill_strict_{suffix}",
                    description="Nested validation candidate: strict-neutral refill short guard.",
                    guard_mode="refill",
                    allocator_mode="strict",
                    risk_regimes=regimes,
                    moderate_threshold=moderate,
                    high_threshold=high,
                    protect_strong_short=True,
                )
            )
    return specs


def spec_to_row(spec: p10c.AllocatorGuardSpec) -> dict[str, Any]:
    row = dict(spec.__dict__)
    row["risk_regimes"] = "|".join(spec.risk_regimes)
    return row


def run_nested_validation_search(
    panel: pd.DataFrame,
    selections: pd.DataFrame,
    selected_p9: pd.DataFrame,
    pf_cfg: Any,
    risk_template: Any,
    spec_by_name: dict[str, Any],
    deriv_formulas: dict[str, Any],
    p3d_formulas: dict[str, Any],
    guard_specs: list[p10c.AllocatorGuardSpec],
    validation_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_by_fold = selected_p9.set_index("fold")
    rows: list[dict[str, Any]] = []
    p9_rows: list[dict[str, Any]] = []
    for _, fold in selections.sort_values("fold").iterrows():
        fold_id = int(fold["fold"])
        row = selected_by_fold.loc[fold_id]
        committee = spec_by_name[str(row["p4_spec"])]
        validation_start = max(
            pd.Timestamp(fold["train_start"]) + (pd.Timestamp(fold["train_end"]) - pd.Timestamp(fold["train_start"])) / 2,
            pd.Timestamp(fold["train_end"]) - pd.Timedelta(days=validation_days),
        )
        validation_end = p3d.decision_end_for_closed_positions(pd.Timestamp(fold["train_end"]), pf_cfg.primary_horizon)
        formula_text = prepare_fold_score(panel, row, committee, deriv_formulas, p3d_formulas)
        risk_cfg = risk_template.__class__(
            **{
                **risk_template.__dict__,
                "name": f"p10d_validation_p9_{committee.name}",
                "short_resid_confirm_rank": committee.short_resid_confirm_rank,
            }
        )
        p9_equity, p9_positions, p9_metrics = p3d.run_p4_gated_rerank_backtest(
            panel=panel,
            start=validation_start,
            end=validation_end,
            cfg=pf_cfg,
            risk_cfg=risk_cfg,
            spec=committee,
            score_col=p3d.SCORE_COL,
            selected_formulas=formula_text,
        )
        enrich_metrics(p9_equity, p9_positions, p9_metrics, pf_cfg.primary_horizon)
        p9_rows.append(
            {
                "fold": fold_id,
                "validation_start": validation_start,
                "validation_decision_end": validation_end,
                "p9_validation_formula": formula_text,
                **{f"p9_validation_{key}": value for key, value in p9_metrics.items() if p3d.is_scalar(value)},
            }
        )
        for spec in guard_specs:
            equity, positions, metrics = p10c.run_allocator_guard_backtest(
                panel=panel,
                start=validation_start,
                end=validation_end,
                cfg=pf_cfg,
                risk_cfg=risk_cfg,
                committee=committee,
                guard_spec=spec,
                score_col=p3d.SCORE_COL,
                selected_formulas=formula_text,
            )
            enrich_metrics(equity, positions, metrics, pf_cfg.primary_horizon)
            rows.append(
                {
                    "fold": fold_id,
                    "guard_spec": spec.name,
                    "risk_regimes": "|".join(spec.risk_regimes),
                    "moderate_threshold": spec.moderate_threshold,
                    "high_threshold": spec.high_threshold,
                    "validation_start": validation_start,
                    "validation_decision_end": validation_end,
                    **{f"validation_{key}": value for key, value in metrics.items() if p3d.is_scalar(value)},
                }
            )
    validation = pd.DataFrame(rows)
    p9_validation = pd.DataFrame(p9_rows)
    validation = annotate_validation(validation, p9_validation)
    return validation, p9_validation


def prepare_fold_score(
    panel: pd.DataFrame,
    row: pd.Series,
    committee: Any,
    deriv_formulas: dict[str, Any],
    p3d_formulas: dict[str, Any],
) -> str:
    p7b.compute_base_score(panel, row, committee, p3d_formulas)
    if str(row["deriv_formula_name"]) == p3d.P4_ONLY_FORMULA:
        panel[p3d.SCORE_COL] = panel[p7b.BASE_SCORE_COL]
        panel[p3d.CANDIDATE_RANK_COL] = np.nan
        return f"P3d_base({row['p3d_formula_name']}) fallback"
    formula = deriv_formulas[str(row["deriv_formula_name"])]
    p7b.apply_derivative_increment(
        panel,
        formula,
        float(row["candidate_weight_long"]),
        float(row["candidate_weight_short"]),
        float(row["high_risk_candidate_cap"]),
    )
    return f"P3d_base({row['p3d_formula_name']}) + deriv({row['deriv_formula_name']},w={float(row['candidate_weight_long']):.2f}): {formula.formula}"


def enrich_metrics(equity: pd.DataFrame, positions: pd.DataFrame, metrics: dict[str, float], horizon: int) -> None:
    p3d.add_gross_metrics(equity, metrics)
    p3d.add_risk_event_metrics(positions, metrics)
    p3d.add_delay_metrics(equity, metrics)
    metrics.update(p3d.cost_adjusted_metrics(equity, horizon, [20.0, 30.0]))
    metrics["q25_segment_sharpe"] = p3d.q25_segment_sharpe(equity, horizon)
    metrics["long_short_balance"] = p3d.long_short_balance(positions)
    metrics["short_abs_contribution_share"] = p3d.short_abs_contribution_share(positions)


def annotate_validation(validation: pd.DataFrame, p9_validation: pd.DataFrame) -> pd.DataFrame:
    out = validation.merge(p9_validation.drop(columns=["validation_start", "validation_decision_end"], errors="ignore"), on="fold", how="left")
    turnover_limit = np.maximum(out["p9_validation_mean_turnover"] * 1.10, out["p9_validation_mean_turnover"] + 0.015)
    out["delta_validation_total_return"] = out["validation_total_return"] - out["p9_validation_total_return"]
    out["delta_validation_sharpe"] = out["validation_sharpe"] - out["p9_validation_sharpe"]
    out["delta_validation_max_drawdown"] = out["validation_max_drawdown"] - out["p9_validation_max_drawdown"]
    out["delta_validation_return_20bps"] = out["validation_total_return_20bps"] - out["p9_validation_total_return_20bps"]
    out["delta_validation_return_30bps"] = out["validation_total_return_30bps"] - out["p9_validation_total_return_30bps"]
    out["hard_accept"] = (
        (out["delta_validation_total_return"] > 0.0)
        & (out["delta_validation_sharpe"] >= -0.05)
        & (out["delta_validation_return_20bps"] > 0.0)
        & (out["delta_validation_return_30bps"] >= -0.001)
        & (out["validation_max_drawdown"] >= 1.10 * out["p9_validation_max_drawdown"])
        & (out["validation_mean_turnover"] <= turnover_limit)
        & (out["validation_mean_abs_beta_market"] <= 1e-8)
    )
    out["selection_score"] = (
        100.0 * out["delta_validation_total_return"]
        + 1.5 * out["delta_validation_sharpe"]
        + 35.0 * out["delta_validation_max_drawdown"]
        + 20.0 * out["delta_validation_return_20bps"]
        + 10.0 * out["delta_validation_return_30bps"]
        - 2.0 * (out["validation_mean_turnover"] - out["p9_validation_mean_turnover"]).clip(lower=0.0)
    )
    return out


def select_allocator_by_fold(
    validation: pd.DataFrame,
    p9_validation: pd.DataFrame,
    selections: pd.DataFrame,
    selected_p9: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    selected_by_fold = selected_p9.set_index("fold")
    p9_by_fold = p9_validation.set_index("fold")
    rows = []
    ordered = validation.sort_values(["fold", "selection_score", "validation_total_return"], ascending=[True, False, False])
    for _, fold in selections.sort_values("fold").iterrows():
        fold_id = int(fold["fold"])
        base = selected_by_fold.loc[fold_id]
        candidates = ordered[(ordered["fold"] == fold_id) & (ordered["hard_accept"])]
        if candidates.empty:
            p9_row = p9_by_fold.loc[fold_id]
            rows.append(
                {
                    "fold": fold_id,
                    "train_start": fold["train_start"],
                    "train_end": fold["train_end"],
                    "test_start": fold["test_start"],
                    "test_end": fold["test_end"],
                    "test_decision_end": p3d.decision_end_for_closed_positions(fold["test_end"], horizon),
                    "selected_allocator": NO_GUARD,
                    "risk_regimes": "",
                    "moderate_threshold": np.nan,
                    "high_threshold": np.nan,
                    "fallback_to_p9_allocator": True,
                    "validation_selection_score": float(p9_row.get("p9_validation_total_return", np.nan)),
                    "validation_total_return": float(p9_row.get("p9_validation_total_return", np.nan)),
                    "validation_sharpe": float(p9_row.get("p9_validation_sharpe", np.nan)),
                    "validation_max_drawdown": float(p9_row.get("p9_validation_max_drawdown", np.nan)),
                    "validation_mean_turnover": float(p9_row.get("p9_validation_mean_turnover", np.nan)),
                    **base.to_dict(),
                }
            )
            continue
        winner = candidates.iloc[0]
        rows.append(
            {
                "fold": fold_id,
                "train_start": fold["train_start"],
                "train_end": fold["train_end"],
                "test_start": fold["test_start"],
                "test_end": fold["test_end"],
                "test_decision_end": p3d.decision_end_for_closed_positions(fold["test_end"], horizon),
                "selected_allocator": winner["guard_spec"],
                "risk_regimes": winner["risk_regimes"],
                "moderate_threshold": float(winner["moderate_threshold"]),
                "high_threshold": float(winner["high_threshold"]),
                "fallback_to_p9_allocator": False,
                "validation_selection_score": float(winner["selection_score"]),
                "validation_total_return": float(winner["validation_total_return"]),
                "validation_sharpe": float(winner["validation_sharpe"]),
                "validation_max_drawdown": float(winner["validation_max_drawdown"]),
                "validation_mean_turnover": float(winner["validation_mean_turnover"]),
                **base.to_dict(),
            }
        )
    return pd.DataFrame(rows)


def replay_selected_allocators(
    panel: pd.DataFrame,
    selections: pd.DataFrame,
    selected_alloc: pd.DataFrame,
    selected_p9: pd.DataFrame,
    pf_cfg: Any,
    risk_template: Any,
    spec_by_name: dict[str, Any],
    deriv_formulas: dict[str, Any],
    p3d_formulas: dict[str, Any],
) -> dict[str, Any]:
    grid = {spec.name: spec for spec in build_nested_guard_grid()}
    alloc_by_fold = selected_alloc.set_index("fold")
    p9_by_fold = selected_p9.set_index("fold")
    fold_rows = []
    equity_parts = []
    position_parts = []
    for _, fold in selections.sort_values("fold").iterrows():
        fold_id = int(fold["fold"])
        alloc = alloc_by_fold.loc[fold_id]
        row = p9_by_fold.loc[fold_id]
        committee = spec_by_name[str(row["p4_spec"])]
        formula_text = prepare_fold_score(panel, row, committee, deriv_formulas, p3d_formulas)
        risk_cfg = risk_template.__class__(
            **{
                **risk_template.__dict__,
                "name": f"p10d_{alloc['selected_allocator']}_{committee.name}",
                "short_resid_confirm_rank": committee.short_resid_confirm_rank,
            }
        )
        start = pd.Timestamp(fold["test_start"])
        end = p3d.decision_end_for_closed_positions(pd.Timestamp(fold["test_end"]), pf_cfg.primary_horizon)
        if bool(alloc["fallback_to_p9_allocator"]):
            equity, positions, metrics = p3d.run_p4_gated_rerank_backtest(
                panel=panel,
                start=start,
                end=end,
                cfg=pf_cfg,
                risk_cfg=risk_cfg,
                spec=committee,
                score_col=p3d.SCORE_COL,
                selected_formulas=formula_text,
            )
            guard_name = NO_GUARD
        else:
            guard_spec = grid[str(alloc["selected_allocator"])]
            equity, positions, metrics = p10c.run_allocator_guard_backtest(
                panel=panel,
                start=start,
                end=end,
                cfg=pf_cfg,
                risk_cfg=risk_cfg,
                committee=committee,
                guard_spec=guard_spec,
                score_col=p3d.SCORE_COL,
                selected_formulas=formula_text,
            )
            guard_name = guard_spec.name
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
                "p3d_formula_name": str(row["p3d_formula_name"]),
                "deriv_formula_name": str(row["deriv_formula_name"]),
                "formula": formula_text,
                "selected_allocator": guard_name,
                "fallback_to_p9_allocator": bool(alloc["fallback_to_p9_allocator"]),
                "validation_selection_score": float(alloc["validation_selection_score"]),
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
            equity["selected_allocator"] = guard_name
            positions["fold"] = fold_id
            positions["p4_spec"] = committee.name
            positions["formula_name"] = str(row["deriv_formula_name"])
            positions["formula"] = formula_text
            positions["selected_allocator"] = guard_name
            equity_parts.append(equity)
            position_parts.append(positions)
    equity_curve = pd.concat(equity_parts, ignore_index=True) if equity_parts else pd.DataFrame()
    positions = pd.concat(position_parts, ignore_index=True) if position_parts else pd.DataFrame()
    folds = pd.DataFrame(fold_rows)
    metrics = p3d.summarize_nested_result({"equity": equity_curve, "positions": positions, "folds": folds}, pf_cfg.rebalance_hours)
    return {"folds": folds, "equity": equity_curve, "positions": positions, "metrics": metrics}


def build_reality_summary(
    summary: pd.DataFrame,
    cost: pd.DataFrame,
    exposure: pd.DataFrame,
    bootstrap: pd.DataFrame,
    start_vs_p9: pd.DataFrame,
    dropout_vs_p9: pd.DataFrame,
) -> pd.DataFrame:
    best = summary[summary["variant"] == "p10d_nested_allocator"].iloc[0]
    p9_row = summary[summary["variant"] == "p9_reference"].iloc[0]
    c20 = cost[(cost["variant"] == "p10d_nested_allocator") & (cost["one_way_cost_bps"] == 20.0)].iloc[0]
    p20 = cost[(cost["variant"] == "p9_reference") & (cost["one_way_cost_bps"] == 20.0)].iloc[0]
    c30 = cost[(cost["variant"] == "p10d_nested_allocator") & (cost["one_way_cost_bps"] == 30.0)].iloc[0]
    p30 = cost[(cost["variant"] == "p9_reference") & (cost["one_way_cost_bps"] == 30.0)].iloc[0]
    boot = bootstrap[bootstrap["metric"] == "p10d_nested_allocator_minus_p9"].iloc[0]
    exp = exposure[exposure["variant"] == "p10d_nested_allocator"].iloc[0]
    start_positive = float((start_vs_p9["delta_total_return"] > 0).mean())
    dropout_positive = float((dropout_vs_p9["delta_total_return"] > 0).mean())
    return pd.DataFrame(
        [
            {"check": "net_delta_vs_p9", "value": best["total_return"] - p9_row["total_return"], "pass": bool(best["total_return"] > p9_row["total_return"])},
            {"check": "sharpe_delta_vs_p9", "value": best["sharpe"] - p9_row["sharpe"], "pass": bool(best["sharpe"] > p9_row["sharpe"])},
            {"check": "maxdd_not_materially_worse", "value": best["max_drawdown"] - p9_row["max_drawdown"], "pass": bool(best["max_drawdown"] >= 1.10 * p9_row["max_drawdown"])},
            {"check": "return_20bps_delta_vs_p9", "value": c20["total_return"] - p20["total_return"], "pass": bool(c20["total_return"] > p20["total_return"])},
            {"check": "return_30bps_delta_vs_p9", "value": c30["total_return"] - p30["total_return"], "pass": bool(c30["total_return"] > p30["total_return"])},
            {"check": "worst5_delta_vs_p9", "value": best["worst_5_fold_return_sum"] - p9_row["worst_5_fold_return_sum"], "pass": bool(best["worst_5_fold_return_sum"] >= p9_row["worst_5_fold_return_sum"])},
            {"check": "bootstrap_delta_vs_p9_q05", "value": boot["bootstrap_q05"], "pass": bool(boot["bootstrap_q05"] > -0.0025)},
            {"check": "start_fold_positive_vs_p9", "value": start_positive, "pass": bool(start_positive >= 0.45)},
            {"check": "symbol_dropout_positive_vs_p9", "value": dropout_positive, "pass": bool(dropout_positive >= 0.45)},
            {"check": "strict_mean_abs_market_beta", "value": exp["mean_abs_beta_market"], "pass": bool(exp["mean_abs_beta_market"] < 1e-8)},
            {"check": "strict_max_abs_market_beta", "value": exp["max_abs_beta_market"], "pass": bool(exp["max_abs_beta_market"] < 1e-7)},
        ]
    )


def build_guard_event_attribution(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for name, result in results.items():
        if name != "p10d_nested_allocator":
            continue
        positions = result["positions"].copy()
        if "raw_guard_benefit" not in positions or "short_guard_multiplier" not in positions:
            continue
        events = positions[(positions["pre_guard_raw_weight"] < 0) & (positions["short_guard_multiplier"] < 1.0)].copy()
        if events.empty:
            continue
        grouped = (
            events.groupby(["selected_allocator", "regime", "symbol"])
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


def write_result(run_dir: Path, result: dict[str, Any]) -> None:
    result["folds"].to_csv(run_dir / "walk_forward_folds.csv", index=False)
    result["equity"].to_csv(run_dir / "equity_curve.csv", index=False)
    result["positions"].to_csv(run_dir / "positions.csv", index=False)
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(to_jsonable(result["metrics"]), fh, indent=2)


def plot_curves(output_dir: Path, results: dict[str, dict[str, Any]]) -> None:
    keep = ["p3d_reference", "p7b_reference", "p9_reference", "p10d_nested_allocator"]
    plt.figure(figsize=(13, 7))
    for name in keep:
        equity = results[name]["equity"].sort_values("timestamp")
        curve = (1.0 + equity["period_return"]).cumprod() - 1.0
        plt.plot(equity["timestamp"], curve, label=name, linewidth=2.1 if name == "p10d_nested_allocator" else 1.4)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.title("P10d nested allocator short guard OOS net curve")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "p10d_net_curves.png", dpi=180)
    plt.close()


def plot_cost_curves(output_dir: Path, cost: pd.DataFrame) -> None:
    frame = cost[cost["variant"].isin(["p9_reference", "p10d_nested_allocator", "p7b_reference", "p3d_reference"])]
    plt.figure(figsize=(12, 7))
    for name, group in frame.groupby("variant"):
        group = group.sort_values("one_way_cost_bps")
        plt.plot(group["one_way_cost_bps"], group["total_return"], marker="o", label=name, linewidth=2.1 if name == "p10d_nested_allocator" else 1.4)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.title("P10d cost sensitivity")
    plt.xlabel("one-way cost bps")
    plt.ylabel("total return")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "cost_sensitivity.png", dpi=180)
    plt.close()


def plot_fold_delta(output_dir: Path, fold_delta: pd.DataFrame) -> None:
    frame = fold_delta[fold_delta["variant"] == "p10d_nested_allocator"].sort_values("fold")
    colors = np.where(frame["delta_total_return"] >= 0.0, "#2a9d8f", "#c44536")
    plt.figure(figsize=(12, 6))
    plt.bar(frame["fold"].astype(str), frame["delta_total_return"], color=colors)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.title("P10d minus P9 fold return")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.tight_layout()
    plt.savefig(output_dir / "fold_delta_vs_p9.png", dpi=180)
    plt.close()


def plot_selected_allocators(output_dir: Path, selected_alloc: pd.DataFrame) -> None:
    counts = selected_alloc["selected_allocator"].value_counts().sort_values()
    plt.figure(figsize=(10, 5))
    plt.barh(counts.index, counts.values, color="#457b9d")
    plt.title("P10d selected allocator by fold")
    plt.tight_layout()
    plt.savefig(output_dir / "selected_allocator_counts.png", dpi=180)
    plt.close()


def write_report(
    output_dir: Path,
    accepted: bool,
    guard_specs: list[p10c.AllocatorGuardSpec],
    selected_alloc: pd.DataFrame,
    validation: pd.DataFrame,
    p9_validation: pd.DataFrame,
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
    cost_display = cost_summary[cost_summary["one_way_cost_bps"].isin([0, 6, 10, 15, 20, 30])]
    cost_display = cost_display[cost_display["variant"].isin(["p9_reference", "p10d_nested_allocator", "p7b_reference", "p3d_reference"])]
    lines = [
        "# P10d Nested Allocator Short Guard",
        "",
        "## Design",
        "",
        "- Fixed P9 nested/OOS formula selections.",
        "- For each fold, allocator parameters are selected only on that fold's train-tail validation window.",
        "- The selected allocator is then applied only to the following test window.",
        "- Candidate allocators are strict-neutral refill guards only; soft beta-drift variants are excluded from the formal baseline pool.",
        f"- Accepted as new baseline: `{accepted}`.",
        "",
        "## Guard Grid",
        "```text",
        pd.DataFrame([spec_to_row(spec) for spec in guard_specs]).to_string(index=False),
        "```",
        "",
        "## Selected Allocators",
        "```text",
        selected_alloc.to_string(index=False),
        "```",
        "",
        "## Summary",
        "```text",
        summary[[col for col in summary_cols if col in summary]].to_string(index=False),
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
        "## Reality Summary",
        "```text",
        reality.to_string(index=False),
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
        "",
        "## Fold Delta Vs P9",
        "```text",
        fold_delta[fold_delta["variant"] == "p10d_nested_allocator"].to_string(index=False),
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
        "```text",
        guard_events.to_string(index=False),
        "```",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def update_config_stage(config_path: Path, accepted: bool) -> None:
    cfg = load_config(config_path)
    stages = cfg.setdefault("stages", {})
    stages["p10d_nested_allocator_short_guard"] = {
        "script": "../../../scripts/run_p10d_nested_allocator_short_guard.py",
        "output_dir": "../outputs/p10d_nested_allocator_short_guard",
        "report": "../outputs/p10d_nested_allocator_short_guard/REPORT.md",
        "variant_summary": "../outputs/p10d_nested_allocator_short_guard/variant_summary.csv",
        "cost_sensitivity": "../outputs/p10d_nested_allocator_short_guard/cost_sensitivity.csv",
        "selected_allocator_by_fold": "../outputs/p10d_nested_allocator_short_guard/selected_allocator_by_fold.csv",
        "reality_summary": "../outputs/p10d_nested_allocator_short_guard/reality_summary.csv",
        "accepted_as_new_baseline": bool(accepted),
    }
    cfg["active_research_nested_allocator_candidate"] = {
        "name": "p10d_nested_allocator",
        "status": "accepted_as_new_baseline" if accepted else "candidate_failed_reality_checks",
        "output_dir": "../outputs/p10d_nested_allocator_short_guard",
        "report": "../outputs/p10d_nested_allocator_short_guard/REPORT.md",
        "parent": "p9_clean_drop_k01_cap010",
    }
    if accepted:
        cfg["active_research_baseline"] = {
            "name": "p10d_nested_allocator_short_guard",
            "description": "P9 formula baseline plus fold-local nested allocator short guard; strict-neutral refill guard selected on train/validation only.",
            "stage": "p10d_nested_allocator_short_guard",
            "horizon_hours": 72,
            "default_total_one_way_cost_bps": 6.0,
            "output_dir": "../outputs/p10d_nested_allocator_short_guard",
            "report": "../outputs/p10d_nested_allocator_short_guard/REPORT.md",
            "summary": "../outputs/p10d_nested_allocator_short_guard/variant_summary.csv",
            "equity_curve": "../outputs/p10d_nested_allocator_short_guard/nested_allocator_run/equity_curve.csv",
            "positions": "../outputs/p10d_nested_allocator_short_guard/nested_allocator_run/positions.csv",
            "baseline_role": "active_research",
            "parent": "p9_clean_drop_k01_cap010",
        }
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False), encoding="utf-8")


if __name__ == "__main__":
    main()
