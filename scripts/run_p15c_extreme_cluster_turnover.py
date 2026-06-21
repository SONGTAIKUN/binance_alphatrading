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
from scipy.optimize import minimize

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
import run_p15_latent_cluster_risk as p15
from run_p13_turnover_no_trade_zone import make_fold_risk, write_result
from run_turnover_experiments import to_jsonable


EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "baseline_locked_v1"
DEFAULT_CONFIG = EXPERIMENT_ROOT / "configs" / "baseline_locked_v1.yaml"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "outputs" / "p15c_extreme_cluster_turnover"
P9_DIR = EXPERIMENT_ROOT / "outputs" / "p9_clean_baseline_package" / "nested_incremental_run"
P9_SELECTED = EXPERIMENT_ROOT / "outputs" / "p9_clean_baseline_package" / "selected_incremental_by_fold.csv"
NO_EXTREME_CAP = "__P9_NO_EXTREME_CLUSTER_CAP__"


@dataclass(frozen=True)
class ExtremeClusterSpec:
    name: str
    lookback_hours: int
    cluster_k: int
    trigger_share: float
    cluster_cap: float
    min_gross_retention: float
    max_turnover_add: float


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
    run_dir = ensure_dir(output_dir / "nested_extreme_cluster_run")
    p9_cluster_dir = ensure_dir(output_dir / "p9_cluster_reference_run")
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

    print("[2/9] Loading panel and residual risk inputs", flush=True)
    panel = p7.load_panel_p7(config_path, panel_path, p3d.MANUAL_FEATURE_PANEL, horizon)
    residual_wide = (
        panel[["timestamp", "symbol", "residual_log_ret"]]
        .dropna()
        .pivot(index="timestamp", columns="symbol", values="residual_log_ret")
        .sort_index()
    )
    pf_cfg = p3d.make_portfolio_config(cfg, horizon)
    risk_template = p3d.make_risk_template(cfg)

    print("[3/9] Building P15c extreme cluster + turnover-aware grid", flush=True)
    extreme_specs = build_extreme_specs()
    pd.DataFrame([spec.__dict__ for spec in extreme_specs]).to_csv(output_dir / "extreme_cluster_grid.csv", index=False)

    print("[4/9] Running fold-local validation search", flush=True)
    validation, p9_validation = run_validation_search(
        panel=panel,
        residual_wide=residual_wide,
        selections=selections,
        selected_p9=selected_p9,
        pf_cfg=pf_cfg,
        risk_template=risk_template,
        spec_by_name=spec_by_name,
        deriv_formulas=deriv_formulas,
        p3d_formulas=p3d_formulas,
        extreme_specs=extreme_specs,
        validation_days=int(args.inner_validation_days),
    )
    validation.to_csv(output_dir / "validation_extreme_cluster_metrics.csv", index=False)
    p9_validation.to_csv(output_dir / "validation_p9_cluster_reference.csv", index=False)

    print("[5/9] Selecting fold-local P15c spec without test data", flush=True)
    selected = select_extreme_spec_by_fold(validation, p9_validation, selections)
    selected.to_csv(output_dir / "selected_extreme_cluster_by_fold.csv", index=False)

    print("[6/9] Replaying selected P15c specs on test windows only", flush=True)
    nested = replay_selected_specs(
        panel=panel,
        residual_wide=residual_wide,
        selections=selections,
        selected=selected,
        selected_p9=selected_p9,
        pf_cfg=pf_cfg,
        risk_template=risk_template,
        spec_by_name=spec_by_name,
        deriv_formulas=deriv_formulas,
        p3d_formulas=p3d_formulas,
        extreme_specs=extreme_specs,
        force_baseline=False,
    )
    write_result(run_dir, nested)

    print("[7/9] Replaying matched P9 cluster reference", flush=True)
    p9_cluster_ref = replay_selected_specs(
        panel=panel,
        residual_wide=residual_wide,
        selections=selections,
        selected=selected,
        selected_p9=selected_p9,
        pf_cfg=pf_cfg,
        risk_template=risk_template,
        spec_by_name=spec_by_name,
        deriv_formulas=deriv_formulas,
        p3d_formulas=p3d_formulas,
        extreme_specs=extreme_specs,
        force_baseline=True,
    )
    write_result(p9_cluster_dir, p9_cluster_ref)

    print("[8/9] Building diagnostics and plots", flush=True)
    results = {"p15c_extreme_cluster_turnover": nested, "p9_reference": p9_result, "p9_cluster_reference": p9_cluster_ref}
    summary = p15.build_p15_summary(results, horizon)
    summary.to_csv(output_dir / "variant_summary.csv", index=False)
    cost_summary = p10.build_cost_summary({"p15c_extreme_cluster_turnover": nested, "p9_reference": p9_result}, cost_grid)
    cost_summary.to_csv(output_dir / "cost_sensitivity.csv", index=False)
    fold_delta = p9.build_fold_delta(nested, p9_result, "p15c_extreme_cluster_turnover", "p9")
    fold_delta.to_csv(output_dir / "fold_delta_vs_p9.csv", index=False)
    bootstrap = p9.bootstrap_fold_checks(fold_delta, "p15c_extreme_cluster_turnover_minus_p9", int(args.bootstrap_samples))
    bootstrap.to_csv(output_dir / "bootstrap_checks.csv", index=False)
    start_vs_p9 = p9.build_start_fold_sensitivity(nested, p9_result, "p15c_extreme_cluster_turnover", "p9", horizon)
    start_vs_p9.to_csv(output_dir / "start_fold_sensitivity_vs_p9.csv", index=False)
    dropout_vs_p9 = p9.build_symbol_dropout_checks(nested, p9_result, "p15c_extreme_cluster_turnover", "p9", horizon)
    dropout_vs_p9.to_csv(output_dir / "symbol_dropout_vs_p9.csv", index=False)
    cap_attr = build_extreme_cap_attribution(nested)
    cap_attr.to_csv(output_dir / "extreme_cap_attribution.csv", index=False)
    cluster_delta = p15.build_cluster_risk_delta(nested, p9_cluster_ref)
    cluster_delta.to_csv(output_dir / "cluster_risk_delta_vs_p9.csv", index=False)
    reality = build_reality_summary(summary, cost_summary, bootstrap, start_vs_p9, dropout_vs_p9, cluster_delta)
    reality.to_csv(output_dir / "reality_summary.csv", index=False)
    plot_curves(output_dir, {"p15c_extreme_cluster_turnover": nested, "p9_reference": p9_result})
    p15.plot_cost(output_dir, cost_summary)
    plot_selected(output_dir, selected)
    p15.plot_fold_delta(output_dir, fold_delta)

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
        cap_attr=cap_attr,
        cluster_delta=cluster_delta,
        reality=reality,
    )
    update_config_stage(config_path)
    print("[done] P15c complete", flush=True)
    print(f"Done: {output_dir}", flush=True)


def build_extreme_specs() -> list[ExtremeClusterSpec]:
    specs: list[ExtremeClusterSpec] = []
    trigger_cap_pairs = [(0.75, 0.60), (0.80, 0.65), (0.85, 0.70)]
    for lookback in [336, 720, 1440]:
        for cluster_k in [3, 4]:
            for trigger, cap in trigger_cap_pairs:
                for turnover_budget in [0.00, 0.02]:
                    specs.append(
                        ExtremeClusterSpec(
                            name=(
                                f"p15c_lb{lookback}_k{cluster_k}_trig{int(trigger*100)}_"
                                f"cap{int(cap*100)}_turn{int(turnover_budget*100):02d}"
                            ),
                            lookback_hours=lookback,
                            cluster_k=cluster_k,
                            trigger_share=trigger,
                            cluster_cap=cap,
                            min_gross_retention=0.95,
                            max_turnover_add=turnover_budget,
                        )
                    )
    return specs


def run_validation_search(
    panel: pd.DataFrame,
    residual_wide: pd.DataFrame,
    selections: pd.DataFrame,
    selected_p9: pd.DataFrame,
    pf_cfg: Any,
    risk_template: Any,
    spec_by_name: dict[str, Any],
    deriv_formulas: dict[str, Any],
    p3d_formulas: dict[str, Any],
    extreme_specs: list[ExtremeClusterSpec],
    validation_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    p9_rows: list[dict[str, Any]] = []
    selected_by_fold = selected_p9.set_index("fold")
    p9_cache: dict[tuple[int, int, int], dict[str, float]] = {}
    for _, fold in selections.sort_values("fold").iterrows():
        fold_id = int(fold["fold"])
        base = selected_by_fold.loc[fold_id]
        committee = spec_by_name[str(base["p4_spec"])]
        formula_text = p10d.prepare_fold_score(panel, base, committee, deriv_formulas, p3d_formulas)
        risk_cfg = make_fold_risk(risk_template, committee, f"p15c_validation_{fold_id}")
        validation_start = max(pd.Timestamp(fold["train_start"]), pd.Timestamp(fold["train_end"]) - pd.Timedelta(days=validation_days))
        validation_end = min(
            p3d.decision_end_for_closed_positions(validation_start + pd.Timedelta(days=validation_days), pf_cfg.primary_horizon),
            pd.Timestamp(fold["train_end"]),
        )
        for spec in extreme_specs:
            key = (fold_id, spec.lookback_hours, spec.cluster_k)
            if key not in p9_cache:
                p9_equity, p9_positions, p9_metrics = run_extreme_backtest(
                    panel=panel,
                    residual_wide=residual_wide,
                    start=validation_start,
                    end=validation_end,
                    cfg=pf_cfg,
                    risk_cfg=risk_cfg,
                    committee=committee,
                    spec=None,
                    diagnostic_lookback=spec.lookback_hours,
                    diagnostic_k=spec.cluster_k,
                    score_col=p3d.SCORE_COL,
                    selected_formulas=formula_text,
                )
                enrich_extreme_metrics(p9_equity, p9_positions, p9_metrics, pf_cfg.primary_horizon)
                p9_cache[key] = p9_metrics
                p9_rows.append(
                    {
                        "fold": fold_id,
                        "lookback_hours": spec.lookback_hours,
                        "cluster_k": spec.cluster_k,
                        "validation_start": validation_start,
                        "validation_decision_end": validation_end,
                        **{f"p9_validation_{k}": v for k, v in p9_metrics.items() if p3d.is_scalar(v)},
                    }
                )
            equity, positions, metrics = run_extreme_backtest(
                panel=panel,
                residual_wide=residual_wide,
                start=validation_start,
                end=validation_end,
                cfg=pf_cfg,
                risk_cfg=risk_cfg,
                committee=committee,
                spec=spec,
                diagnostic_lookback=spec.lookback_hours,
                diagnostic_k=spec.cluster_k,
                score_col=p3d.SCORE_COL,
                selected_formulas=formula_text,
            )
            enrich_extreme_metrics(equity, positions, metrics, pf_cfg.primary_horizon)
            rows.append(
                {
                    "fold": fold_id,
                    "extreme_spec": spec.name,
                    **spec.__dict__,
                    "validation_start": validation_start,
                    "validation_decision_end": validation_end,
                    **{f"validation_{k}": v for k, v in metrics.items() if p3d.is_scalar(v)},
                }
            )
    validation = annotate_validation(pd.DataFrame(rows), pd.DataFrame(p9_rows))
    return validation, pd.DataFrame(p9_rows)


def annotate_validation(validation: pd.DataFrame, p9_validation: pd.DataFrame) -> pd.DataFrame:
    out = validation.merge(
        p9_validation.drop(columns=["validation_start", "validation_decision_end"], errors="ignore"),
        on=["fold", "lookback_hours", "cluster_k"],
        how="left",
    )
    out["delta_validation_total_return"] = out["validation_total_return"] - out["p9_validation_total_return"]
    out["delta_validation_sharpe"] = out["validation_sharpe"] - out["p9_validation_sharpe"]
    out["delta_validation_max_drawdown"] = out["validation_max_drawdown"] - out["p9_validation_max_drawdown"]
    out["delta_validation_return_20bps"] = out["validation_total_return_20bps"] - out["p9_validation_total_return_20bps"]
    out["delta_validation_return_30bps"] = out["validation_total_return_30bps"] - out["p9_validation_total_return_30bps"]
    out["validation_turnover_ratio_vs_p9"] = out["validation_mean_turnover"] / out["p9_validation_mean_turnover"].replace(0.0, np.nan)
    out["validation_gross_retention_vs_p9"] = out["validation_gross_total_return_before_cost"] / out[
        "p9_validation_gross_total_return_before_cost"
    ].replace(0.0, np.nan)
    out["validation_cluster_hhi_reduction_vs_p9"] = 1.0 - out["validation_mean_cluster_hhi"] / out[
        "p9_validation_mean_cluster_hhi"
    ].replace(0.0, np.nan)
    out["validation_max_cluster_share_reduction_vs_p9"] = 1.0 - out["validation_mean_max_cluster_gross_share"] / out[
        "p9_validation_mean_max_cluster_gross_share"
    ].replace(0.0, np.nan)
    out["hard_accept"] = (
        (out["delta_validation_total_return"] >= -0.0010)
        & (out["delta_validation_sharpe"] >= -0.03)
        & (out["validation_max_drawdown"] >= 1.05 * out["p9_validation_max_drawdown"])
        & (out["delta_validation_return_20bps"] >= -0.0005)
        & (out["delta_validation_return_30bps"] >= -0.0005)
        & (out["validation_turnover_ratio_vs_p9"] <= 1.01)
        & (out["validation_gross_retention_vs_p9"] >= 0.95)
        & (out["validation_cluster_hhi_reduction_vs_p9"] >= 0.015)
        & (out["validation_mean_cluster_cap_applied"] <= 0.30)
        & (out["validation_mean_extreme_cluster_triggered"] <= 0.45)
        & (out["validation_mean_abs_beta_market"] <= 1e-8)
    )
    out["selection_score"] = (
        110.0 * out["delta_validation_total_return"].fillna(-1.0)
        + 90.0 * out["delta_validation_return_20bps"].fillna(-1.0)
        + 90.0 * out["delta_validation_return_30bps"].fillna(-1.0)
        + 1.0 * out["delta_validation_sharpe"].fillna(-5.0)
        + 7.0 * out["validation_cluster_hhi_reduction_vs_p9"].clip(-1.0, 1.0).fillna(0.0)
        + 4.0 * out["validation_max_cluster_share_reduction_vs_p9"].clip(-1.0, 1.0).fillna(0.0)
        - 5.0 * (out["validation_turnover_ratio_vs_p9"].fillna(1.0) - 1.0).clip(lower=0.0)
        - 1.0 * out["validation_mean_cluster_cap_applied"].fillna(0.0)
        - 0.5 * out["validation_mean_turnover_guard_skipped"].fillna(0.0)
    )
    return out


def select_extreme_spec_by_fold(validation: pd.DataFrame, p9_validation: pd.DataFrame, selections: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    by_fold = {int(k): v.copy() for k, v in validation.groupby("fold")}
    p9_by_fold = p9_validation.groupby("fold").first()
    for _, fold in selections.sort_values("fold").iterrows():
        fold_id = int(fold["fold"])
        accepted = by_fold.get(fold_id, pd.DataFrame())
        accepted = accepted[accepted["hard_accept"].astype(bool)].copy() if not accepted.empty else pd.DataFrame()
        if accepted.empty:
            p9_row = p9_by_fold.loc[fold_id]
            rows.append(
                {
                    "fold": fold_id,
                    "selected_extreme_spec": NO_EXTREME_CAP,
                    "fallback_to_p9_sizing": True,
                    "lookback_hours": int(p9_row["lookback_hours"]),
                    "cluster_k": int(p9_row["cluster_k"]),
                    "trigger_share": np.nan,
                    "cluster_cap": np.nan,
                    "max_turnover_add": np.nan,
                    "validation_selection_score": float(p9_row.get("p9_validation_total_return", np.nan)),
                    "validation_total_return": float(p9_row.get("p9_validation_total_return", np.nan)),
                    "validation_sharpe": float(p9_row.get("p9_validation_sharpe", np.nan)),
                    "validation_cluster_hhi_reduction_vs_p9": 0.0,
                }
            )
            continue
        winner = accepted.sort_values(["selection_score", "delta_validation_return_20bps"], ascending=False).iloc[0]
        rows.append(
            {
                "fold": fold_id,
                "selected_extreme_spec": str(winner["extreme_spec"]),
                "fallback_to_p9_sizing": False,
                "lookback_hours": int(winner["lookback_hours"]),
                "cluster_k": int(winner["cluster_k"]),
                "trigger_share": float(winner["trigger_share"]),
                "cluster_cap": float(winner["cluster_cap"]),
                "max_turnover_add": float(winner["max_turnover_add"]),
                "validation_selection_score": float(winner["selection_score"]),
                "validation_total_return": float(winner["validation_total_return"]),
                "validation_sharpe": float(winner["validation_sharpe"]),
                "validation_cluster_hhi_reduction_vs_p9": float(winner["validation_cluster_hhi_reduction_vs_p9"]),
            }
        )
    return pd.DataFrame(rows)


def replay_selected_specs(
    panel: pd.DataFrame,
    residual_wide: pd.DataFrame,
    selections: pd.DataFrame,
    selected: pd.DataFrame,
    selected_p9: pd.DataFrame,
    pf_cfg: Any,
    risk_template: Any,
    spec_by_name: dict[str, Any],
    deriv_formulas: dict[str, Any],
    p3d_formulas: dict[str, Any],
    extreme_specs: list[ExtremeClusterSpec],
    force_baseline: bool,
) -> dict[str, Any]:
    spec_map = {spec.name: spec for spec in extreme_specs}
    selected_exec = selected.set_index("fold")
    selected_p9_by_fold = selected_p9.set_index("fold")
    fold_rows: list[dict[str, Any]] = []
    equity_parts: list[pd.DataFrame] = []
    position_parts: list[pd.DataFrame] = []
    for _, fold in selections.sort_values("fold").iterrows():
        fold_id = int(fold["fold"])
        base = selected_p9_by_fold.loc[fold_id]
        decision = selected_exec.loc[fold_id]
        selected_name = str(decision["selected_extreme_spec"])
        selected_spec = None if bool(decision["fallback_to_p9_sizing"]) else spec_map[selected_name]
        exec_spec = None if force_baseline else selected_spec
        lookback = int(decision["lookback_hours"])
        cluster_k = int(decision["cluster_k"])
        committee = spec_by_name[str(base["p4_spec"])]
        formula_text = p10d.prepare_fold_score(panel, base, committee, deriv_formulas, p3d_formulas)
        risk_cfg = make_fold_risk(risk_template, committee, f"p15c_{selected_name}_{fold_id}")
        start = pd.Timestamp(fold["test_start"])
        end = p3d.decision_end_for_closed_positions(pd.Timestamp(fold["test_end"]), pf_cfg.primary_horizon)
        equity, positions, metrics = run_extreme_backtest(
            panel=panel,
            residual_wide=residual_wide,
            start=start,
            end=end,
            cfg=pf_cfg,
            risk_cfg=risk_cfg,
            committee=committee,
            spec=exec_spec,
            diagnostic_lookback=lookback,
            diagnostic_k=cluster_k,
            score_col=p3d.SCORE_COL,
            selected_formulas=formula_text,
        )
        enrich_extreme_metrics(equity, positions, metrics, pf_cfg.primary_horizon)
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
                "selected_extreme_spec": NO_EXTREME_CAP if exec_spec is None else exec_spec.name,
                "validation_selected_extreme_spec": selected_name,
                "fallback_to_p9_sizing": bool(exec_spec is None),
                "validation_selection_score": float(decision["validation_selection_score"]),
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
            equity["selected_extreme_spec"] = NO_EXTREME_CAP if exec_spec is None else exec_spec.name
            equity["validation_selected_extreme_spec"] = selected_name
            positions["fold"] = fold_id
            positions["p4_spec"] = committee.name
            positions["formula_name"] = str(base["deriv_formula_name"])
            positions["formula"] = formula_text
            positions["selected_extreme_spec"] = NO_EXTREME_CAP if exec_spec is None else exec_spec.name
            positions["validation_selected_extreme_spec"] = selected_name
            equity_parts.append(equity)
            position_parts.append(positions)
    equity_curve = pd.concat(equity_parts, ignore_index=True) if equity_parts else pd.DataFrame()
    positions = pd.concat(position_parts, ignore_index=True) if position_parts else pd.DataFrame()
    folds = pd.DataFrame(fold_rows)
    metrics = p3d.summarize_nested_result({"equity": equity_curve, "positions": positions, "folds": folds}, pf_cfg.rebalance_hours)
    enrich_extreme_metrics(equity_curve, positions, metrics, pf_cfg.primary_horizon)
    return {"folds": folds, "equity": equity_curve, "positions": positions, "metrics": metrics}


def run_extreme_backtest(
    panel: pd.DataFrame,
    residual_wide: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cfg: Any,
    risk_cfg: Any,
    committee: Any,
    spec: ExtremeClusterSpec | None,
    diagnostic_lookback: int,
    diagnostic_k: int,
    score_col: str,
    selected_formulas: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    label_col = f"label_resid_ret_h{cfg.primary_horizon}"
    ret_col = f"fwd_simple_ret_h{cfg.primary_horizon}"
    symbols = sorted(panel["symbol"].unique())
    mask = (panel["timestamp"] >= start) & (panel["timestamp"] < end)
    timestamps = np.array(sorted(panel.loc[mask, "timestamp"].unique()))[:: cfg.rebalance_hours]
    prev_weights = pd.Series(0.0, index=symbols)
    current_state: dict[str, tuple[int, int]] = {}
    model_cache: dict[tuple[pd.Timestamp, int, int], p15.LatentRiskModel] = {}
    equity = 1.0
    rows_out: list[dict[str, Any]] = []
    pos_out: list[dict[str, Any]] = []
    for ts in timestamps:
        timestamp = pd.Timestamp(ts)
        rows = panel[panel["timestamp"] == timestamp].copy()
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
        target_sides = p3d.select_p4_gated_target_sides(rows, current_state, cfg, score_col)
        before_guard = len([side for side in target_sides.values() if side < 0])
        target_sides = apply_short_guard(target_sides, rows_by_symbol, committee)
        after_guard = len([side for side in target_sides.values() if side < 0])
        if not any(side > 0 for side in target_sides.values()) or not any(side < 0 for side in target_sides.values()):
            continue
        selected = rows_by_symbol.reindex(list(target_sides)).reset_index()
        raw_weights = side_fraction_weights(target_sides, cfg.gross_exposure, committee.long_fraction)
        baseline_weights, baseline_info = p3d.build_weights(raw_weights, selected, rows_by_symbol, cfg, risk_cfg)
        baseline_weights = baseline_weights[baseline_weights.abs() > 1e-12]
        if baseline_weights.abs().sum() <= 0:
            continue
        lookback = diagnostic_lookback if spec is None else spec.lookback_hours
        cluster_k = diagnostic_k if spec is None else spec.cluster_k
        model = p15.get_risk_model(model_cache, residual_wide, timestamp, lookback, cluster_k)
        baseline_latent = p15.latent_metrics(baseline_weights, model)
        baseline_full = pd.Series(0.0, index=symbols)
        baseline_full.loc[baseline_weights.index] = baseline_weights
        baseline_turnover = float((baseline_full - prev_weights).abs().sum())
        if spec is None:
            weights = baseline_weights
            cap_info = extreme_info(False, False, False, False, baseline_latent, baseline_latent, np.nan, np.nan, 0.0, baseline_turnover)
            cap_info.update({k: v for k, v in baseline_info.items() if k.startswith("optimizer_")})
        else:
            weights, cap_info = apply_extreme_cluster_cap(
                baseline_weights=baseline_weights,
                raw_weights=raw_weights,
                rows_by_symbol=rows_by_symbol,
                model=model,
                cfg=cfg,
                risk_cfg=risk_cfg,
                spec=spec,
                baseline_latent=baseline_latent,
                prev_weights=prev_weights,
                all_symbols=symbols,
                baseline_turnover=baseline_turnover,
            )
        weights = weights[weights.abs() > 1e-12]
        if weights.abs().sum() <= 0:
            continue
        latent = p15.latent_metrics(weights, model)
        target_full_weights = pd.Series(0.0, index=symbols)
        target_full_weights.loc[weights.index] = weights
        turnover = float((target_full_weights - prev_weights).abs().sum())
        cost = turnover * (cfg.one_way_fee_bps + cfg.one_way_slippage_bps) / 10_000.0
        active_rows = rows_by_symbol.reindex(weights.index)
        gross_period_ret = float((weights * active_rows[ret_col]).sum())
        baseline_rows = rows_by_symbol.reindex(baseline_weights.index)
        baseline_gross_ret = float((baseline_weights * baseline_rows[ret_col]).sum())
        period_ret = gross_period_ret - cost
        equity *= 1.0 + period_ret
        rank_ic = spearman_ic(rows[score_col], rows[label_col])
        long_labels = active_rows.loc[weights[weights > 0].index, label_col]
        short_labels = active_rows.loc[weights[weights < 0].index, label_col]
        top_bottom = float(long_labels.mean() - short_labels.mean()) if len(long_labels) and len(short_labels) else np.nan
        exposures = portfolio_exposures(weights, active_rows, cfg.neutralize_exposures)
        rows_out.append(
            {
                "timestamp": timestamp,
                "exit_timestamp": timestamp + pd.Timedelta(hours=cfg.execution_delay_bars + cfg.primary_horizon),
                "period_return": period_ret,
                "gross_period_return": gross_period_ret,
                "baseline_target_gross_period_return": baseline_gross_ret,
                "cost_return": cost,
                "equity": equity,
                "turnover": turnover,
                "target_turnover": baseline_turnover,
                "turnover_saved": max(0.0, baseline_turnover - turnover),
                "gross_exposure": float(weights.abs().sum()),
                "target_gross_exposure": float(baseline_weights.abs().sum()),
                "gross_exposure_retention": float(weights.abs().sum() / baseline_weights.abs().sum()),
                "rank_ic": rank_ic,
                "top_bottom_label_spread": top_bottom,
                "abs_beta_btc": abs(exposures.get("beta_btc", 0.0)),
                "abs_beta_eth": abs(exposures.get("beta_eth", 0.0)),
                "abs_beta_market": abs(exposures.get("beta_market", 0.0)),
                "symbol_concentration": float((weights**2).sum()),
                "max_abs_weight": float(weights.abs().max()),
                "delay_tau_days": 0.0,
                "delay_retention": 0.0,
                "delay_new_target_weight": 1.0,
                "short_guard_removed": float(before_guard - after_guard),
                "target_applied": 1.0,
                "cluster_lookback_hours": lookback,
                "cluster_k": cluster_k,
                "selected_extreme_spec": NO_EXTREME_CAP if spec is None else spec.name,
                **latent,
                **cap_info,
                "selected_formulas": selected_formulas,
            }
        )
        labels = pd.Series({symbol: model.cluster_by_symbol.get(symbol, 0) for symbol in weights.index})
        for symbol, weight in weights.items():
            row = rows_by_symbol.loc[symbol]
            baseline_weight = float(baseline_full.loc[symbol])
            pos_out.append(
                {
                    "timestamp": timestamp,
                    "exit_timestamp": timestamp + pd.Timedelta(hours=cfg.execution_delay_bars + cfg.primary_horizon),
                    "symbol": symbol,
                    "weight": float(weight),
                    "target_weight": float(weight),
                    "baseline_target_weight": baseline_weight,
                    "score": float(row[score_col]),
                    "p4_score": float(row[p3d.P4_SCORE_COL]),
                    "candidate_rank": float(row[p3d.CANDIDATE_RANK_COL]) if p3d.CANDIDATE_RANK_COL in row.index else np.nan,
                    "p4_rank": float(row["p4_rank"]),
                    "ema_rank": float(row["ema_rank"]),
                    "base_rank": float(row["base_rank"]),
                    "rank_a07": float(row["rank_a07"]),
                    "rank_b11": float(row["rank_b11"]),
                    "rank_c09": float(row["rank_c09"]),
                    "cluster_id": int(labels.loc[symbol]),
                    "fwd_simple_ret": float(row[ret_col]),
                    "label_resid_ret": float(row[label_col]),
                    "contribution": float(weight * row[ret_col]),
                    "baseline_target_contribution": float(baseline_weight * row[ret_col]),
                    "selected_score_cols": "p4_score | candidate_rerank_score",
                }
            )
        prev_weights = target_full_weights
        current_state = update_state(current_state, target_sides, True)
    equity_curve = pd.DataFrame(rows_out)
    positions = pd.DataFrame(pos_out)
    metrics = p3d.aggregate_oos_metrics(equity_curve, cfg.rebalance_hours)
    return equity_curve, positions, metrics


def apply_extreme_cluster_cap(
    baseline_weights: pd.Series,
    raw_weights: pd.Series,
    rows_by_symbol: pd.DataFrame,
    model: p15.LatentRiskModel,
    cfg: Any,
    risk_cfg: Any,
    spec: ExtremeClusterSpec,
    baseline_latent: dict[str, float],
    prev_weights: pd.Series,
    all_symbols: list[str],
    baseline_turnover: float,
) -> tuple[pd.Series, dict[str, float]]:
    if baseline_latent["max_cluster_gross_share"] < spec.trigger_share:
        info = extreme_info(False, False, False, False, baseline_latent, baseline_latent, spec.trigger_share, spec.cluster_cap, 0.0, baseline_turnover)
        info.update({"optimizer_success": 1.0, "optimizer_exposure_error": 0.0, "optimizer_attempts": 0.0})
        return baseline_weights, info
    symbols = list(baseline_weights.index)
    selected = rows_by_symbol.loc[symbols]
    signs = np.sign(raw_weights.reindex(symbols).fillna(baseline_weights).to_numpy(dtype=float))
    if not np.any(signs > 0) or not np.any(signs < 0):
        info = extreme_info(True, False, False, False, baseline_latent, baseline_latent, spec.trigger_share, spec.cluster_cap, 0.0, baseline_turnover)
        info.update({"optimizer_success": 0.0, "optimizer_exposure_error": np.nan, "optimizer_attempts": 0.0})
        return baseline_weights, info
    baseline = baseline_weights.reindex(symbols).to_numpy(dtype=float)
    gross_limit = float(np.sum(np.abs(baseline)))
    max_abs = float(risk_cfg.max_symbol_weight or cfg.gross_exposure)
    bounds = [(0.0, max_abs) if sign > 0 else (-max_abs, 0.0) for sign in signs]
    exposure_matrix = p15.exposure_matrix_for(selected, cfg.neutralize_exposures)
    cluster_ids = np.array([model.cluster_by_symbol.get(symbol, 0) for symbol in symbols], dtype=int)
    unique_clusters = sorted(set(cluster_ids.tolist()))

    def objective(x: np.ndarray) -> float:
        return float(np.sum((x - baseline) ** 2) + 1e-5 * np.sum(x**2))

    def gradient(x: np.ndarray) -> np.ndarray:
        return 2.0 * (x - baseline) + 2e-5 * x

    constraints: list[dict[str, Any]] = [
        {"type": "eq", "fun": lambda x, a=exposure_matrix: a @ x, "jac": lambda x, a=exposure_matrix: a},
        {"type": "ineq", "fun": lambda x, s=signs, limit=gross_limit: float(limit - s @ x), "jac": lambda x, s=signs: -s},
        {
            "type": "ineq",
            "fun": lambda x, s=signs, floor=spec.min_gross_retention, limit=gross_limit: float(s @ x - floor * limit),
            "jac": lambda x, s=signs: s,
        },
    ]
    for cluster_id in unique_clusters:
        mask = (cluster_ids == cluster_id).astype(float)
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda x, m=mask, s=signs, cap=spec.cluster_cap: float(cap * (s @ x) - np.sum(m * s * x)),
                "jac": lambda x, m=mask, s=signs, cap=spec.cluster_cap: cap * s - m * s,
            }
        )
    result = minimize(
        objective,
        baseline,
        jac=gradient,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 120, "ftol": 1e-9, "disp": False},
    )
    x = result.x if result.x is not None else baseline
    exposure_error = p15.max_constraint_error(exposure_matrix, x)
    gross = float(signs @ x)
    bounds_ok = all(value >= lo - 1e-8 and value <= hi + 1e-8 for value, (lo, hi) in zip(x, bounds))
    candidate = pd.Series(x, index=symbols)
    after = p15.latent_metrics(candidate, model)
    constraints_ok = (
        exposure_error <= 1e-7
        and gross <= gross_limit + 1e-7
        and gross >= spec.min_gross_retention * gross_limit - 1e-7
        and after["max_cluster_gross_share"] <= spec.cluster_cap + 1e-6
        and bounds_ok
        and bool(result.success)
    )
    if not constraints_ok:
        info = extreme_info(True, False, False, False, baseline_latent, baseline_latent, spec.trigger_share, spec.cluster_cap, 0.0, baseline_turnover)
        info.update({"optimizer_success": 0.0, "optimizer_exposure_error": float(exposure_error), "optimizer_attempts": 1.0})
        return baseline_weights, info
    candidate_full = pd.Series(0.0, index=all_symbols)
    candidate_full.loc[candidate.index] = candidate
    turnover_after = float((candidate_full - prev_weights).abs().sum())
    turnover_add = turnover_after - baseline_turnover
    if turnover_add > spec.max_turnover_add + 1e-12:
        info = extreme_info(True, False, True, True, baseline_latent, baseline_latent, spec.trigger_share, spec.cluster_cap, turnover_add, baseline_turnover)
        info.update({"optimizer_success": 1.0, "optimizer_exposure_error": float(exposure_error), "optimizer_attempts": 1.0})
        return baseline_weights, info
    info = extreme_info(True, True, False, True, baseline_latent, after, spec.trigger_share, spec.cluster_cap, turnover_add, baseline_turnover)
    info.update({"optimizer_success": 1.0, "optimizer_exposure_error": float(exposure_error), "optimizer_attempts": 1.0})
    return candidate, info


def extreme_info(
    triggered: bool,
    applied: bool,
    skipped_turnover: bool,
    solved: bool,
    before: dict[str, float],
    after: dict[str, float],
    trigger_share: float,
    cluster_cap: float,
    turnover_add: float,
    baseline_turnover: float,
) -> dict[str, float]:
    return {
        "extreme_cluster_triggered": float(triggered),
        "cluster_cap_applied": float(applied),
        "turnover_guard_skipped": float(skipped_turnover),
        "cluster_cap_solved": float(solved),
        "trigger_share": float(trigger_share),
        "cluster_cap": float(cluster_cap),
        "turnover_add_vs_p9": float(turnover_add),
        "baseline_turnover": float(baseline_turnover),
        "baseline_max_cluster_gross_share": float(before.get("max_cluster_gross_share", np.nan)),
        "baseline_cluster_hhi": float(before.get("cluster_hhi", np.nan)),
        "cluster_share_reduction": float(before.get("max_cluster_gross_share", np.nan) - after.get("max_cluster_gross_share", np.nan)),
        "cluster_hhi_reduction_period": float(before.get("cluster_hhi", np.nan) - after.get("cluster_hhi", np.nan)),
    }


def enrich_extreme_metrics(equity: pd.DataFrame, positions: pd.DataFrame, metrics: dict[str, float], horizon: int) -> None:
    p15.enrich_cluster_metrics(equity, positions, metrics, horizon)
    if equity.empty:
        return
    for col in ["extreme_cluster_triggered", "cluster_cap_applied", "turnover_guard_skipped", "cluster_cap_solved", "turnover_add_vs_p9"]:
        if col in equity:
            metrics[f"mean_{col}"] = float(equity[col].mean())


def build_extreme_cap_attribution(result: dict[str, Any]) -> pd.DataFrame:
    eq = result["equity"].copy()
    if eq.empty:
        return pd.DataFrame()
    return (
        eq.groupby("selected_extreme_spec")
        .agg(
            periods=("timestamp", "count"),
            period_return_sum=("period_return", "sum"),
            gross_return_sum=("gross_period_return", "sum"),
            cost_sum=("cost_return", "sum"),
            turnover_sum=("turnover", "sum"),
            mean_extreme_cluster_triggered=("extreme_cluster_triggered", "mean"),
            mean_cluster_cap_applied=("cluster_cap_applied", "mean"),
            mean_turnover_guard_skipped=("turnover_guard_skipped", "mean"),
            mean_turnover_add_vs_p9=("turnover_add_vs_p9", "mean"),
            mean_cluster_hhi=("cluster_hhi", "mean"),
            mean_baseline_cluster_hhi=("baseline_cluster_hhi", "mean"),
            mean_max_cluster_gross_share=("max_cluster_gross_share", "mean"),
            mean_baseline_max_cluster_gross_share=("baseline_max_cluster_gross_share", "mean"),
            mean_gross_exposure_retention=("gross_exposure_retention", "mean"),
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
    cluster_delta: pd.DataFrame,
) -> pd.DataFrame:
    cand = summary[summary["variant"] == "p15c_extreme_cluster_turnover"].iloc[0]
    p9_row = summary[summary["variant"] == "p9_reference"].iloc[0]
    c20 = cost[(cost["variant"] == "p15c_extreme_cluster_turnover") & (cost["one_way_cost_bps"] == 20.0)].iloc[0]
    p20 = cost[(cost["variant"] == "p9_reference") & (cost["one_way_cost_bps"] == 20.0)].iloc[0]
    c30 = cost[(cost["variant"] == "p15c_extreme_cluster_turnover") & (cost["one_way_cost_bps"] == 30.0)].iloc[0]
    p30 = cost[(cost["variant"] == "p9_reference") & (cost["one_way_cost_bps"] == 30.0)].iloc[0]
    boot = bootstrap.iloc[0]
    hhi_delta = cluster_delta[cluster_delta["metric"] == "cluster_hhi"].iloc[0] if not cluster_delta.empty else pd.Series(dtype=float)
    max_share_delta = cluster_delta[cluster_delta["metric"] == "max_cluster_gross_share"].iloc[0] if not cluster_delta.empty else pd.Series(dtype=float)
    start_positive = float((start_vs_p9["delta_total_return"] > 0).mean()) if "delta_total_return" in start_vs_p9 else np.nan
    dropout_positive = float((dropout_vs_p9["delta_total_return"] > 0).mean()) if "delta_total_return" in dropout_vs_p9 else np.nan
    gross_retention = cand["gross_total_return_before_cost"] / p9_row["gross_total_return_before_cost"]
    return pd.DataFrame(
        [
            {"check": "net_delta_vs_p9_positive", "value": cand["total_return"] - p9_row["total_return"], "pass": bool(cand["total_return"] > p9_row["total_return"])},
            {"check": "sharpe_delta_vs_p9_positive", "value": cand["sharpe"] - p9_row["sharpe"], "pass": bool(cand["sharpe"] > p9_row["sharpe"])},
            {"check": "maxdd_not_worse_than_105x_p9", "value": cand["max_drawdown"] - p9_row["max_drawdown"], "pass": bool(cand["max_drawdown"] >= 1.05 * p9_row["max_drawdown"])},
            {"check": "return_20bps_delta_positive", "value": c20["total_return"] - p20["total_return"], "pass": bool(c20["total_return"] > p20["total_return"])},
            {"check": "return_30bps_delta_positive", "value": c30["total_return"] - p30["total_return"], "pass": bool(c30["total_return"] > p30["total_return"])},
            {"check": "turnover_not_higher_than_102pct_p9", "value": cand["mean_turnover"] / p9_row["mean_turnover"], "pass": bool(cand["mean_turnover"] <= 1.02 * p9_row["mean_turnover"])},
            {"check": "gross_retention_90pct", "value": gross_retention, "pass": bool(gross_retention >= 0.90)},
            {"check": "cluster_hhi_reduction_3pct", "value": hhi_delta.get("reduction_pct", np.nan), "pass": bool(hhi_delta.get("reduction_pct", np.nan) >= 0.03)},
            {"check": "max_cluster_share_reduction_3pct", "value": max_share_delta.get("reduction_pct", np.nan), "pass": bool(max_share_delta.get("reduction_pct", np.nan) >= 0.03)},
            {"check": "worst5_delta_vs_p9", "value": cand["worst_5_fold_return_sum"] - p9_row["worst_5_fold_return_sum"], "pass": bool(cand["worst_5_fold_return_sum"] >= p9_row["worst_5_fold_return_sum"])},
            {"check": "bootstrap_delta_vs_p9_q05", "value": boot["bootstrap_q05"], "pass": bool(boot["bootstrap_q05"] > -0.0025)},
            {"check": "start_fold_positive_vs_p9", "value": start_positive, "pass": bool(start_positive >= 0.35)},
            {"check": "symbol_dropout_positive_vs_p9", "value": dropout_positive, "pass": bool(dropout_positive >= 0.55)},
            {"check": "strict_mean_abs_market_beta", "value": cand["mean_abs_beta_market"], "pass": bool(cand["mean_abs_beta_market"] < 1e-8)},
            {"check": "strict_max_abs_market_beta", "value": cand["max_abs_beta_market"], "pass": bool(cand["max_abs_beta_market"] < 1e-7)},
        ]
    )


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
        if "cluster_hhi" in eq:
            axes[1].plot(eq["timestamp"], eq["cluster_hhi"].rolling(8, min_periods=1).mean(), label=name, linewidth=1.6)
    axes[0].axhline(0, color="#222222", linewidth=0.8)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].set_title("P15c extreme cluster turnover-aware cap vs P9")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.25)
    axes[1].set_title("Rolling mean latent cluster HHI")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir / "p15c_curves.png", dpi=160)
    plt.close(fig)


def plot_selected(output_dir: Path, selected: pd.DataFrame) -> None:
    counts = selected["selected_extreme_spec"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(counts))))
    ax.barh(counts.index, counts.values)
    ax.set_title("Selected P15c spec by fold")
    ax.set_xlabel("fold count")
    fig.tight_layout()
    fig.savefig(output_dir / "selected_extreme_cluster_counts.png", dpi=160)
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
    cap_attr: pd.DataFrame,
    cluster_delta: pd.DataFrame,
    reality: pd.DataFrame,
) -> None:
    accepted = bool(reality["pass"].all())
    lines = [
        "# P15c Extreme Cluster Turnover-Aware Cap",
        "",
        "## Design",
        "",
        "- Fixed P9 alpha, selected formulas, eligible long/short pool, and strict beta-neutral constraints.",
        "- Applies cluster cap only when P9 latent cluster concentration exceeds an extreme trigger.",
        "- Keeps P9 sizing when the cluster cap would add more turnover than the selected fold-local budget.",
        "- PCA remains diagnostic only; no PCA neutralization is traded.",
        f"- Accepted as new baseline: `{accepted}`.",
        "",
        "## Variant Summary",
        "```text",
        summary.to_string(index=False),
        "```",
        "",
        "## Reality Summary",
        "```text",
        reality.to_string(index=False),
        "```",
        "",
        "## Cluster Risk Delta Vs P9",
        "```text",
        cluster_delta.to_string(index=False),
        "```",
        "",
        "## Cost Sensitivity",
        "```text",
        cost_summary.to_string(index=False),
        "```",
        "",
        "## Selected Spec By Fold",
        "```text",
        selected.to_string(index=False),
        "```",
        "",
        "## Extreme Cap Attribution",
        "```text",
        cap_attr.to_string(index=False),
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
        "## P9 Validation Cluster Reference",
        "```text",
        p9_validation.to_string(index=False),
        "```",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def update_config_stage(config_path: Path) -> None:
    cfg = load_config(config_path)
    stages = cfg.setdefault("stages", {})
    stages["p15c_extreme_cluster_turnover"] = {
        "script": "../../../scripts/run_p15c_extreme_cluster_turnover.py",
        "output_dir": "../outputs/p15c_extreme_cluster_turnover",
        "report": "../outputs/p15c_extreme_cluster_turnover/REPORT.md",
        "summary": "../outputs/p15c_extreme_cluster_turnover/variant_summary.csv",
        "selected_extreme_cluster_by_fold": "../outputs/p15c_extreme_cluster_turnover/selected_extreme_cluster_by_fold.csv",
    }
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False), encoding="utf-8")


if __name__ == "__main__":
    main()
