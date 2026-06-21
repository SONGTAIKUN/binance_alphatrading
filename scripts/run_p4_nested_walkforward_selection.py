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
from matplotlib.ticker import PercentFormatter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from crypto_stat_arb.config import ensure_dir, load_config
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
    add_market_regime_features,
    build_committee_specs,
    build_fitness,
    build_locked_baseline_correlations,
    make_risk_template,
    plot_asset_contribution,
    plot_best_curves,
    resolve_config_path,
    run_committee_backtest,
    write_run_result,
)
from run_residual_alphagpt_search import aggregate_oos_metrics, build_candidate_scores, prepare_alpha_panel
from run_risk_control_experiments import add_gross_metrics, add_risk_event_metrics
from run_delay_smoothing_experiments import add_delay_metrics
from run_turnover_experiments import to_jsonable


EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "baseline_locked_v1"
DEFAULT_CONFIG = EXPERIMENT_ROOT / "configs" / "baseline_locked_v1.yaml"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "outputs" / "p4_nested_walkforward_selection"
MANUAL_FEATURE_PANEL = EXPERIMENT_ROOT / "outputs" / "p2_manual_slow_alpha_grid" / "manual_slow_feature_panel.csv.gz"

_WORKER_PANEL: pd.DataFrame | None = None
_WORKER_PF_CFG: Any = None
_WORKER_RISK_TEMPLATE: Any = None
_WORKER_SPECS: dict[str, CommitteeSpec] = {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--horizon", type=int, default=72)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--cost-bps", type=float, nargs="+", default=[0, 4, 6, 8, 10, 12, 15, 20, 30])
    parser.add_argument("--chunksize", type=int, default=12)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    output_dir = ensure_dir(args.output_dir)
    horizon = int(args.horizon)
    workers = max(1, int(args.workers))
    cost_bps = [float(x) for x in args.cost_bps]

    panel_path = resolve_config_path(config_path, cfg["inputs"]["panel"])
    selections_path = resolve_config_path(config_path, cfg["inputs"]["formula_selections"])
    selections = pd.read_csv(selections_path, parse_dates=["train_start", "train_end", "test_start", "test_end"])
    specs = build_committee_specs()
    pd.DataFrame([spec.__dict__ for spec in specs]).to_csv(output_dir / "candidate_specs.csv", index=False)

    print(f"[1/8] Nested validation backtests: {len(selections)} folds x {len(specs)} candidates", flush=True)
    print(f"      workers={workers}, horizon={horizon}h", flush=True)
    tasks = [
        {
            "fold": int(row["fold"]),
            "window": "validation",
            "start": row["train_start"],
            "end": decision_end_for_closed_positions(row["train_end"], horizon),
            "window_end": row["train_end"],
            "spec_name": spec.name,
        }
        for _, row in selections.iterrows()
        for spec in specs
    ]
    worker_args = (
        str(config_path),
        str(panel_path),
        str(MANUAL_FEATURE_PANEL),
        horizon,
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
    validation["fitness"] = build_fitness(validation)
    validation.to_csv(output_dir / "validation_candidate_metrics.csv", index=False)

    print("[2/8] Selecting one candidate per fold using validation-only fitness", flush=True)
    selected = select_specs_by_fold(validation, selections, horizon)
    selected.to_csv(output_dir / "selected_specs_by_fold.csv", index=False)

    print("[3/8] Loading panel in parent and replaying selected candidates on test folds only", flush=True)
    panel = load_panel(config_path, panel_path, MANUAL_FEATURE_PANEL, horizon)
    pf_cfg = make_portfolio_config(cfg, horizon)
    risk_template = make_risk_template(cfg)
    spec_by_name = {spec.name: spec for spec in specs}
    test_result = replay_selected_tests(panel, selections, selected, pf_cfg, risk_template, spec_by_name)
    write_run_result(ensure_dir(output_dir / "nested_test_run"), test_result)

    print("[4/8] Computing nested test metrics and comparisons", flush=True)
    metrics = summarize_nested_result(test_result, horizon)
    write_json(output_dir / "nested_test_metrics.json", metrics)
    folds = test_result["folds"].copy()
    folds.to_csv(output_dir / "nested_test_fold_metrics.csv", index=False)
    baseline = load_baseline_summary(EXPERIMENT_ROOT / "baseline_lock_review" / "horizon_summary.csv")
    summary = pd.DataFrame([{**metrics, "run": "p4_nested_walkforward", "horizon": horizon}])
    corr = build_locked_baseline_correlations({"p4_nested_walkforward": test_result}, horizon)
    summary = summary.merge(corr, on="run", how="left")
    summary["fitness"] = build_fitness(summary)
    summary.to_csv(output_dir / "summary.csv", index=False)
    comparison = build_baseline_comparison(summary, baseline)
    comparison.to_csv(output_dir / "baseline_comparison.csv", index=False)

    print("[5/8] Running nested cost sensitivity and diagnostics", flush=True)
    cost_summary = build_cost_sensitivity({"p4_nested_walkforward": test_result}, cost_bps)
    cost_summary.to_csv(output_dir / "cost_sensitivity.csv", index=False)
    equity = test_result["equity"].sort_values("timestamp")
    positions = test_result["positions"].sort_values(["timestamp", "symbol"])
    regime_summary = build_regime_summary(equity, build_regime_frame(panel), horizon)
    side_summary = build_side_summary(positions)
    asset_summary = build_asset_summary(positions)
    regime_summary.to_csv(output_dir / "regime_summary.csv", index=False)
    side_summary.to_csv(output_dir / "side_contribution.csv", index=False)
    asset_summary.to_csv(output_dir / "asset_contribution.csv", index=False)

    print("[6/8] Writing plots", flush=True)
    plot_nested_curves(output_dir, equity, horizon)
    plot_fold_selection(output_dir, selected)
    plot_validation_vs_test(output_dir, folds)
    plot_cost_sensitivity(output_dir, cost_summary)
    plot_asset_contribution(output_dir, asset_summary)

    print("[7/8] Writing report", flush=True)
    write_report(
        output_dir=output_dir,
        summary=summary,
        comparison=comparison,
        selected=selected,
        validation=validation,
        folds=folds,
        cost_summary=cost_summary,
        regime_summary=regime_summary,
        side_summary=side_summary,
        asset_summary=asset_summary,
        workers=workers,
        horizon=horizon,
    )
    update_config_stage(config_path)

    print("[8/8] Done", flush=True)
    print(f"Done: {output_dir}", flush=True)


def init_worker(config_path_str: str, panel_path_str: str, feature_path_str: str, horizon: int) -> None:
    global _WORKER_PANEL, _WORKER_PF_CFG, _WORKER_RISK_TEMPLATE, _WORKER_SPECS
    config_path = Path(config_path_str)
    cfg = load_config(config_path)
    _WORKER_PANEL = load_panel(config_path, Path(panel_path_str), Path(feature_path_str), horizon)
    _WORKER_PF_CFG = make_portfolio_config(cfg, horizon)
    _WORKER_RISK_TEMPLATE = make_risk_template(cfg)
    _WORKER_SPECS = {spec.name: spec for spec in build_committee_specs()}


def load_panel(config_path: Path, panel_path: Path, feature_path: Path, horizon: int) -> pd.DataFrame:
    panel = pd.read_csv(panel_path, parse_dates=["timestamp"])
    panel = prepare_alpha_panel(panel, [horizon], execution_delay_bars=1)
    build_candidate_scores(panel)
    add_market_regime_features(panel)
    features = pd.read_csv(
        feature_path,
        parse_dates=["timestamp"],
        usecols=["timestamp", "symbol", *COMPONENTS.values()],
    )
    panel = panel.merge(features, on=["timestamp", "symbol"], how="left")
    for key, col in COMPONENTS.items():
        panel[f"rank_{key.lower()}"] = centered_rank_by_timestamp(panel, col)
    return panel


def centered_rank_by_timestamp(frame: pd.DataFrame, value_col: str) -> pd.Series:
    grouped = frame.groupby("timestamp", sort=False)[value_col]
    rank = grouped.rank(method="average")
    count = grouped.transform("count")
    centered = 2.0 * (rank - 1.0) / (count - 1.0) - 1.0
    return centered.where(count > 1)


def run_validation_task(task: dict[str, Any]) -> dict[str, Any]:
    if _WORKER_PANEL is None:
        raise RuntimeError("Worker panel was not initialized.")
    spec = _WORKER_SPECS[str(task["spec_name"])]
    score_col = f"score_{spec.name}"
    panel = _WORKER_PANEL
    panel[score_col] = (
        spec.w_a07 * panel["rank_a07"]
        + spec.w_b11 * panel["rank_b11"]
        + spec.w_c09 * panel["rank_c09"]
    )
    risk_cfg = _WORKER_RISK_TEMPLATE.__class__(
        **{
            **_WORKER_RISK_TEMPLATE.__dict__,
            "name": spec.name,
            "short_resid_confirm_rank": spec.short_resid_confirm_rank,
        }
    )
    formula = (
        f"{spec.w_a07:.1f}*rank(A07)+{spec.w_b11:.1f}*rank(B11)+{spec.w_c09:.1f}*rank(C09); "
        f"long_fraction={spec.long_fraction:.2f}; short_guard={spec.short_guard}"
    )
    equity, positions, metrics = run_committee_backtest(
        panel=panel,
        start=pd.Timestamp(task["start"]),
        end=pd.Timestamp(task["end"]),
        cfg=_WORKER_PF_CFG,
        risk_cfg=risk_cfg,
        spec=spec,
        score_col=score_col,
        selected_formulas=formula,
    )
    panel.drop(columns=[score_col], inplace=True)
    add_gross_metrics(equity, metrics)
    add_risk_event_metrics(positions, metrics)
    add_delay_metrics(equity, metrics)
    metrics["mean_short_guard_removed"] = float(equity["short_guard_removed"].mean()) if "short_guard_removed" in equity else np.nan
    return {
        "fold": int(task["fold"]),
        "window": str(task["window"]),
        "spec_name": spec.name,
        "w_a07": spec.w_a07,
        "w_b11": spec.w_b11,
        "w_c09": spec.w_c09,
        "long_fraction": spec.long_fraction,
        "short_guard": spec.short_guard,
        "short_resid_confirm_rank": spec.short_resid_confirm_rank,
        "validation_start": pd.Timestamp(task["start"]),
        "validation_decision_end": pd.Timestamp(task["end"]),
        "validation_window_end": pd.Timestamp(task["window_end"]),
        **metrics,
    }


def select_specs_by_fold(validation: pd.DataFrame, selections: pd.DataFrame, horizon: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    validation_sorted = validation.sort_values(
        ["fold", "fitness", "total_return", "sharpe"],
        ascending=[True, False, False, False],
    )
    winners = validation_sorted.groupby("fold", sort=True).head(1).set_index("fold")
    for _, fold in selections.sort_values("fold").iterrows():
        fold_id = int(fold["fold"])
        winner = winners.loc[fold_id].to_dict()
        rows.append(
            {
                "fold": fold_id,
                "train_start": fold["train_start"],
                "train_end": fold["train_end"],
                "test_start": fold["test_start"],
                "test_end": fold["test_end"],
                "validation_decision_end": decision_end_for_closed_positions(fold["train_end"], horizon),
                "selected_spec": winner["spec_name"],
                "selected_w_a07": winner["w_a07"],
                "selected_w_b11": winner["w_b11"],
                "selected_w_c09": winner["w_c09"],
                "selected_long_fraction": winner["long_fraction"],
                "selected_short_guard": winner["short_guard"],
                "validation_fitness": winner["fitness"],
                "validation_total_return": winner.get("total_return", np.nan),
                "validation_gross_return": winner.get("gross_total_return_before_cost", np.nan),
                "validation_sharpe": winner.get("sharpe", np.nan),
                "validation_max_drawdown": winner.get("max_drawdown", np.nan),
                "validation_mean_turnover": winner.get("mean_turnover", np.nan),
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
) -> dict[str, Any]:
    fold_rows: list[dict[str, Any]] = []
    equity_parts: list[pd.DataFrame] = []
    position_parts: list[pd.DataFrame] = []
    selected_by_fold = selected.set_index("fold")
    for _, fold in selections.sort_values("fold").iterrows():
        fold_id = int(fold["fold"])
        chosen = selected_by_fold.loc[fold_id]
        spec = spec_by_name[str(chosen["selected_spec"])]
        score_col = f"score_{spec.name}"
        panel[score_col] = (
            spec.w_a07 * panel["rank_a07"]
            + spec.w_b11 * panel["rank_b11"]
            + spec.w_c09 * panel["rank_c09"]
        )
        risk_cfg = risk_template.__class__(
            **{
                **risk_template.__dict__,
                "name": spec.name,
                "short_resid_confirm_rank": spec.short_resid_confirm_rank,
            }
        )
        formula = (
            f"{spec.w_a07:.1f}*rank(A07)+{spec.w_b11:.1f}*rank(B11)+{spec.w_c09:.1f}*rank(C09); "
            f"long_fraction={spec.long_fraction:.2f}; short_guard={spec.short_guard}"
        )
        equity, positions, metrics = run_committee_backtest(
            panel=panel,
            start=fold["test_start"],
            end=decision_end_for_closed_positions(fold["test_end"], pf_cfg.primary_horizon),
            cfg=pf_cfg,
            risk_cfg=risk_cfg,
            spec=spec,
            score_col=score_col,
            selected_formulas=formula,
        )
        panel.drop(columns=[score_col], inplace=True)
        add_gross_metrics(equity, metrics)
        add_risk_event_metrics(positions, metrics)
        add_delay_metrics(equity, metrics)
        metrics["mean_short_guard_removed"] = float(equity["short_guard_removed"].mean()) if "short_guard_removed" in equity else np.nan
        fold_row = {
            "fold": fold_id,
            "train_start": fold["train_start"],
            "train_end": fold["train_end"],
            "test_start": fold["test_start"],
            "test_end": fold["test_end"],
            "test_decision_end": decision_end_for_closed_positions(fold["test_end"], pf_cfg.primary_horizon),
            "selected_spec": spec.name,
            "selected_formula": formula,
            "validation_fitness": chosen["validation_fitness"],
            "validation_total_return": chosen["validation_total_return"],
            **metrics,
        }
        fold_rows.append(fold_row)
        if not equity.empty:
            equity = equity.copy()
            equity["fold"] = fold_id
            equity["selected_spec"] = spec.name
            equity["selected_formula"] = formula
            positions = positions.copy()
            positions["fold"] = fold_id
            positions["selected_spec"] = spec.name
            positions["selected_formula"] = formula
            equity_parts.append(equity)
            position_parts.append(positions)
    equity_curve = pd.concat(equity_parts, ignore_index=True) if equity_parts else pd.DataFrame()
    positions = pd.concat(position_parts, ignore_index=True) if position_parts else pd.DataFrame()
    folds = pd.DataFrame(fold_rows)
    metrics = summarize_nested_result({"equity": equity_curve, "positions": positions, "folds": folds}, pf_cfg.rebalance_hours)
    return {"folds": folds, "equity": equity_curve, "positions": positions, "metrics": metrics}


def decision_end_for_closed_positions(window_end: pd.Timestamp, horizon: int, execution_delay_bars: int = 1) -> pd.Timestamp:
    return pd.Timestamp(window_end) - pd.Timedelta(hours=int(horizon) + int(execution_delay_bars))


def summarize_nested_result(result: dict[str, Any], horizon: int) -> dict[str, float]:
    equity = result["equity"]
    positions = result["positions"]
    folds = result["folds"]
    metrics = aggregate_oos_metrics(equity, horizon)
    metrics["fold_count"] = int(len(folds))
    if not folds.empty and "total_return" in folds:
        metrics["positive_fold_rate"] = float((folds["total_return"] > 0).mean())
    add_gross_metrics(equity, metrics)
    add_risk_event_metrics(positions, metrics)
    add_delay_metrics(equity, metrics)
    metrics["mean_short_guard_removed"] = float(equity["short_guard_removed"].mean()) if "short_guard_removed" in equity else np.nan
    return metrics


def plot_nested_curves(output_dir: Path, equity: pd.DataFrame, horizon: int) -> None:
    if equity.empty:
        return
    frame = equity.sort_values("timestamp")
    net = (1.0 + frame["period_return"]).cumprod() - 1.0
    gross = (1.0 + frame["gross_period_return"]).cumprod() - 1.0
    dd = (1.0 + frame["period_return"]).cumprod()
    dd = dd / dd.cummax() - 1.0
    base_path = EXPERIMENT_ROOT / "baseline_lock_review" / "horizon" / f"h{horizon:03d}_tau0" / "equity_curve.csv"
    base = pd.read_csv(base_path, parse_dates=["timestamp"]).sort_values("timestamp") if base_path.exists() else pd.DataFrame()
    fig, axes = plt.subplots(2, 1, figsize=(13, 8.5), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    axes[0].plot(frame["timestamp"], gross, label="P4 nested gross before cost", linewidth=2.0)
    axes[0].plot(frame["timestamp"], net, label="P4 nested net after 6bps", linewidth=2.0)
    if not base.empty:
        base_curve = (1.0 + base["period_return"]).cumprod() - 1.0
        axes[0].plot(base["timestamp"], base_curve, label="locked h072 baseline net", linewidth=1.5, alpha=0.8)
    axes[0].axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    axes[0].set_title("P4 nested walk-forward parameter selection")
    axes[0].set_ylabel("Cumulative return")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()
    axes[1].fill_between(frame["timestamp"], dd, 0.0, color="#e76f51", alpha=0.28)
    axes[1].plot(frame["timestamp"], dd, color="#e76f51", linewidth=1.3)
    axes[1].set_ylabel("Drawdown")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "nested_net_gross_drawdown.png", dpi=180)
    plt.close(fig)


def plot_fold_selection(output_dir: Path, selected: pd.DataFrame) -> None:
    if selected.empty:
        return
    counts = selected["selected_spec"].value_counts().sort_values()
    plt.figure(figsize=(12, max(4, 0.35 * len(counts))))
    plt.barh(counts.index, counts.values, color="#277da1")
    plt.title("P4 selected spec counts by fold")
    plt.xlabel("Fold count")
    plt.tight_layout()
    plt.savefig(output_dir / "selected_spec_counts.png", dpi=180)
    plt.close()

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(selected["fold"], selected["selected_w_a07"], marker="o", label="A07")
    axes[0].plot(selected["fold"], selected["selected_w_b11"], marker="o", label="B11")
    axes[0].plot(selected["fold"], selected["selected_w_c09"], marker="o", label="C09")
    axes[0].set_title("Selected component weights by fold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(selected["fold"], selected["validation_fitness"], marker="o", color="#43aa8b")
    axes[1].set_title("Validation fitness of selected spec")
    axes[1].grid(True, alpha=0.25)
    axes[2].scatter(selected["fold"], selected["selected_short_guard"], color="#f3722c")
    axes[2].set_title("Selected short guard")
    axes[2].set_xlabel("Fold")
    axes[2].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "selected_params_by_fold.png", dpi=180)
    plt.close(fig)


def plot_validation_vs_test(output_dir: Path, folds: pd.DataFrame) -> None:
    if folds.empty:
        return
    plt.figure(figsize=(7, 6))
    plt.scatter(folds["validation_total_return"], folds["total_return"], color="#277da1", alpha=0.8)
    plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    plt.axvline(0.0, color="black", linewidth=0.8, alpha=0.6)
    plt.title("Selected spec validation return vs next test return")
    plt.xlabel("Validation total return")
    plt.ylabel("Next test total return")
    plt.gca().xaxis.set_major_formatter(PercentFormatter(1.0))
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "validation_vs_test_return.png", dpi=180)
    plt.close()


def plot_cost_sensitivity(output_dir: Path, cost_summary: pd.DataFrame) -> None:
    if cost_summary.empty:
        return
    frame = cost_summary.sort_values("one_way_cost_bps")
    plt.figure(figsize=(10, 6))
    plt.plot(frame["one_way_cost_bps"], frame["total_return"], marker="o", linewidth=2)
    plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    plt.title("P4 nested cost sensitivity")
    plt.xlabel("one-way cost bps")
    plt.ylabel("Net cumulative return")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "cost_sensitivity.png", dpi=180)
    plt.close()


def write_report(
    output_dir: Path,
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    selected: pd.DataFrame,
    validation: pd.DataFrame,
    folds: pd.DataFrame,
    cost_summary: pd.DataFrame,
    regime_summary: pd.DataFrame,
    side_summary: pd.DataFrame,
    asset_summary: pd.DataFrame,
    workers: int,
    horizon: int,
) -> None:
    metric_cols = [
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
        "mean_abs_beta_market",
        "max_abs_weight",
        "mean_short_guard_removed",
        "net_corr_to_locked_baseline",
        "fitness",
    ]
    fold_cols = [
        "fold",
        "selected_spec",
        "validation_fitness",
        "validation_total_return",
        "total_return",
        "gross_total_return_before_cost",
        "sharpe",
        "max_drawdown",
        "mean_turnover",
    ]
    selected_cols = [
        "fold",
        "train_start",
        "train_end",
        "validation_decision_end",
        "test_start",
        "test_end",
        "selected_spec",
        "selected_w_a07",
        "selected_w_b11",
        "selected_w_c09",
        "selected_long_fraction",
        "selected_short_guard",
        "validation_fitness",
        "validation_total_return",
        "validation_sharpe",
        "validation_max_drawdown",
        "validation_mean_turnover",
    ]
    cost_cols = ["one_way_cost_bps", "total_return", "max_drawdown", "cost_drag_sum"]
    comp_cols = [
        "run",
        "horizon",
        "total_return",
        "baseline_total_return",
        "delta_total_return",
        "gross_total_return_before_cost",
        "baseline_gross_total_return_before_cost",
        "delta_gross_total_return_before_cost",
        "mean_turnover",
        "baseline_mean_turnover",
        "delta_mean_turnover",
        "net_corr_to_locked_baseline",
    ]
    spec_counts = selected["selected_spec"].value_counts().reset_index()
    spec_counts.columns = ["selected_spec", "fold_count"]
    lines = [
        "# P4 Nested Walk-Forward Parameter Selection",
        "",
        "## Design",
        "",
        f"- Horizon fixed at `{horizon}h`.",
        "- Candidate set fixed before the nested test: the 252 P3 A07/B11/C09 committee + sizing + short-guard candidates.",
        "- For each fold, parameter selection uses only that fold's `train_start -> train_end` window.",
        "- The selected candidate is then applied to that fold's `test_start -> test_end` window.",
        "- Validation and test decision windows are clipped by `horizon + execution_delay` so every evaluated trade exits inside its own window.",
        "- The final PnL is the stitched sequence of test windows only.",
        f"- Validation backtests were run with `{workers}` worker processes.",
        "",
        "## Nested OOS Summary",
        "",
        "```text",
        summary[[col for col in metric_cols if col in summary.columns]].to_string(index=False),
        "```",
        "",
        "## Baseline Comparison",
        "",
        "```text",
        comparison[[col for col in comp_cols if col in comparison.columns]].to_string(index=False),
        "```",
        "",
        "## Selected Specs",
        "",
        "```text",
        selected[[col for col in selected_cols if col in selected.columns]].to_string(index=False),
        "```",
        "",
        "## Selected Spec Counts",
        "",
        "```text",
        spec_counts.to_string(index=False),
        "```",
        "",
        "## Fold Results",
        "",
        "```text",
        folds[[col for col in fold_cols if col in folds.columns]].to_string(index=False),
        "```",
        "",
        "## Cost Sensitivity",
        "",
        "```text",
        cost_summary[[col for col in cost_cols if col in cost_summary.columns]].to_string(index=False),
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
        "- `validation_candidate_metrics.csv`: validation metrics for every fold/candidate pair.",
        "- `selected_specs_by_fold.csv`: chosen candidate per fold.",
        "- `nested_test_run/`: stitched test-only folds/equity/positions.",
        "- `summary.csv`, `baseline_comparison.csv`, `cost_sensitivity.csv`: headline result tables.",
        "- `nested_net_gross_drawdown.png`, `selected_spec_counts.png`, `selected_params_by_fold.png`, `validation_vs_test_return.png`, `cost_sensitivity.png`, `asset_contribution.png`: plots.",
        "",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(to_jsonable(data), fh, indent=2, ensure_ascii=False)


def update_config_stage(config_path: Path) -> None:
    text = config_path.read_text(encoding="utf-8")
    if "  p4_nested_walkforward_selection:\n" in text:
        return
    marker = "committee_candidates:\n"
    addition = """
  p4_nested_walkforward_selection:
    script: ../../../scripts/run_p4_nested_walkforward_selection.py
    output_dir: ../outputs/p4_nested_walkforward_selection
    report: ../outputs/p4_nested_walkforward_selection/REPORT.md
    selected_specs_by_fold: ../outputs/p4_nested_walkforward_selection/selected_specs_by_fold.csv
    validation_candidate_metrics: ../outputs/p4_nested_walkforward_selection/validation_candidate_metrics.csv
    summary: ../outputs/p4_nested_walkforward_selection/summary.csv
    baseline_comparison: ../outputs/p4_nested_walkforward_selection/baseline_comparison.csv
"""
    if marker not in text:
        config_path.write_text(text.rstrip() + "\n" + addition, encoding="utf-8")
        return
    text = text.replace(marker, addition + "\n" + marker, 1)
    config_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
