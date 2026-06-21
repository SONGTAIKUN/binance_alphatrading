#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
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
from run_p5_alphagpt_slow_search import FormulaSpec
from run_turnover_experiments import to_jsonable


EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "baseline_locked_v1"
DEFAULT_CONFIG = EXPERIMENT_ROOT / "configs" / "baseline_locked_v1.yaml"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "outputs" / "p9_clean_derivatives_baseline"
P3D_DIR = EXPERIMENT_ROOT / "outputs" / "p3d_p4_gated_rerank_alphagpt"
P7B_DIR = EXPERIMENT_ROOT / "outputs" / "p7b_derivatives_incremental_on_p3d"
P3D_SELECTED = P3D_DIR / "selected_incremental_by_fold.csv"
P3D_EQUITY = P3D_DIR / "nested_incremental_run" / "equity_curve.csv"
P7B_EQUITY = P7B_DIR / "nested_incremental_run" / "equity_curve.csv"

P9_GRID = [
    {"candidate_weight_long": 0.05, "candidate_weight_short": 0.05, "high_risk_candidate_cap": 0.03},
    {"candidate_weight_long": 0.075, "candidate_weight_short": 0.075, "high_risk_candidate_cap": 0.05},
    {"candidate_weight_long": 0.10, "candidate_weight_short": 0.10, "high_risk_candidate_cap": 0.10},
]


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
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--cost-bps", type=float, nargs="+", default=[0, 4, 6, 8, 10, 12, 15, 20, 30])
    parser.add_argument("--allow-k01", action="store_true")
    parser.add_argument("--no-require-30bps", action="store_true")
    parser.add_argument("--short-share-limit", type=float, default=0.70)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    output_dir = ensure_dir(args.output_dir)
    run_dir = ensure_dir(output_dir / "nested_incremental_run")
    horizon = int(args.horizon)
    workers = max(1, int(args.workers))

    p7b.P7B_GRID = P9_GRID
    panel_path = p3d.resolve_config_path(config_path, cfg["inputs"]["panel"])
    selections_path = p3d.resolve_config_path(config_path, cfg["inputs"]["formula_selections"])
    selections = pd.read_csv(selections_path, parse_dates=["train_start", "train_end", "test_start", "test_end"])
    p3d_selected = pd.read_csv(P3D_SELECTED, parse_dates=["train_start", "train_end", "test_start", "test_end", "test_decision_end"])

    print("[1/10] Building P9 clean formula universe", flush=True)
    retained = p7.load_retained_features()
    raw_formulas = p7.build_p7_formula_universe(retained, int(args.formulas), int(args.seed), int(args.max_depth))
    formulas = filter_p9_formulas(raw_formulas, exclude_k01=not bool(args.allow_k01))
    formula_table = formula_table_with_tags(raw_formulas, formulas)
    formula_table.to_csv(output_dir / "candidate_formulas.csv", index=False)
    print(f"      raw formulas={len(raw_formulas)}, clean formulas={len(formulas)}", flush=True)

    print("[2/10] Running nested validation search against P3d", flush=True)
    tasks = p7b.build_validation_tasks(selections, p3d_selected, formulas, horizon, int(args.inner_validation_days))
    print(f"      {len(selections)} folds x {len(formulas)} formulas x {len(P9_GRID)} grids = {len(tasks)} tasks; workers={workers}", flush=True)
    p3d_formulas = p3d.build_incremental_formula_universe(180, 2417, 3)
    worker_args = (
        str(config_path),
        str(panel_path),
        str(p3d.MANUAL_FEATURE_PANEL),
        horizon,
        {item.name: item for item in formulas},
        {item.name: item for item in p3d_formulas},
        p3d_selected,
    )
    validation_rows: list[dict[str, Any]] = []
    if workers == 1:
        p7b.init_worker(*worker_args)
        for idx, task in enumerate(tasks, start=1):
            if idx == 1 or idx % 250 == 0 or idx == len(tasks):
                print(f"      validation task {idx}/{len(tasks)}", flush=True)
            validation_rows.append(p7b.run_validation_task(task))
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers, initializer=p7b.init_worker, initargs=worker_args) as pool:
            for idx, row in enumerate(pool.imap_unordered(p7b.run_validation_task, tasks, chunksize=max(1, int(args.chunksize))), start=1):
                validation_rows.append(row)
                if idx == 1 or idx % 250 == 0 or idx == len(tasks):
                    print(f"      validation task {idx}/{len(tasks)}", flush=True)
    validation = pd.DataFrame(validation_rows)

    print("[3/10] Running P3d validation baselines and applying P9 hard accept", flush=True)
    p3d_validation = p7b.run_p3d_validation_baselines(
        config_path=config_path,
        panel_path=panel_path,
        feature_path=p3d.MANUAL_FEATURE_PANEL,
        selections=selections,
        p3d_selected=p3d_selected,
        p3d_formulas=p3d_formulas,
        horizon=horizon,
    )
    p3d_validation.to_csv(output_dir / "validation_p3d_metrics.csv", index=False)
    validation = p7b.annotate_hard_accept(validation, p3d_validation)
    validation = apply_p9_hard_accept(
        validation,
        formulas,
        require_30bps_improvement=not bool(args.no_require_30bps),
        short_share_limit=float(args.short_share_limit),
    )
    validation.to_csv(output_dir / "validation_incremental_metrics.csv", index=False)

    print("[4/10] Selecting best accepted P9 increment per fold", flush=True)
    selected = p7b.select_by_fold(validation, p3d_validation, selections, p3d_selected, horizon)
    selected.to_csv(output_dir / "selected_incremental_by_fold.csv", index=False)

    print("[5/10] Replaying selected P9 increments on test windows only", flush=True)
    panel = p7.load_panel_p7(config_path, panel_path, p3d.MANUAL_FEATURE_PANEL, horizon)
    pf_cfg = p3d.make_portfolio_config(cfg, horizon)
    risk_template = p3d.make_risk_template(cfg)
    spec_by_name = {spec.name: spec for spec in p3d.build_committee_specs()}
    result = p7b.replay_selected_tests(
        panel=panel,
        selections=selections,
        selected=selected,
        pf_cfg=pf_cfg,
        risk_template=risk_template,
        spec_by_name=spec_by_name,
        deriv_formulas={item.name: item for item in formulas},
        p3d_formulas={item.name: item for item in p3d_formulas},
    )
    p3d.write_run_result(run_dir, result)

    print("[6/10] Building summary, cost, and attribution tables", flush=True)
    summary = build_summary(result, horizon)
    summary.to_csv(output_dir / "summary.csv", index=False)
    comparison = build_reference_comparison(summary, result)
    comparison.to_csv(output_dir / "baseline_comparison.csv", index=False)
    cost_summary = build_cost_summary(result, [float(x) for x in args.cost_bps])
    cost_summary.to_csv(output_dir / "cost_sensitivity.csv", index=False)
    regime_summary = p3d.build_regime_summary(result["equity"].sort_values("timestamp"), p3d.build_regime_frame(panel), horizon)
    side_summary = p3d.build_side_summary(result["positions"].sort_values(["timestamp", "symbol"]))
    asset_summary = p3d.build_asset_summary(result["positions"].sort_values(["timestamp", "symbol"]))
    regime_summary.to_csv(output_dir / "regime_summary.csv", index=False)
    side_summary.to_csv(output_dir / "side_contribution.csv", index=False)
    asset_summary.to_csv(output_dir / "asset_contribution.csv", index=False)
    primitive_usage = build_primitive_usage(selected, formulas, retained)
    primitive_usage.to_csv(output_dir / "primitive_usage.csv", index=False)

    p3d_ref = load_existing_result(P3D_DIR / "nested_incremental_run")
    p7b_ref = load_existing_result(P7B_DIR / "nested_incremental_run")
    fold_vs_p3d = build_fold_delta(result, p3d_ref, "p9", "p3d")
    fold_vs_p7b = build_fold_delta(result, p7b_ref, "p9", "p7b")
    fold_vs_p3d.to_csv(output_dir / "fold_attribution_vs_p3d.csv", index=False)
    fold_vs_p7b.to_csv(output_dir / "fold_attribution_vs_p7b.csv", index=False)
    side_vs_p3d = build_side_delta(result, p3d_ref, "p9", "p3d")
    asset_vs_p3d = build_asset_delta(result, p3d_ref, "p9", "p3d")
    side_vs_p3d.to_csv(output_dir / "side_delta_vs_p3d.csv", index=False)
    asset_vs_p3d.to_csv(output_dir / "asset_delta_vs_p3d.csv", index=False)

    print("[7/10] Running P9 reality checks", flush=True)
    bootstrap = pd.concat(
        [
            bootstrap_fold_checks(fold_vs_p3d, "p9_minus_p3d", int(args.bootstrap_samples)),
            bootstrap_fold_checks(fold_vs_p7b, "p9_minus_p7b", int(args.bootstrap_samples)),
        ],
        ignore_index=True,
    )
    bootstrap.to_csv(output_dir / "bootstrap_checks.csv", index=False)
    start_vs_p3d = build_start_fold_sensitivity(result, p3d_ref, "p9", "p3d", horizon)
    start_vs_p7b = build_start_fold_sensitivity(result, p7b_ref, "p9", "p7b", horizon)
    start_vs_p3d.to_csv(output_dir / "start_fold_sensitivity_vs_p3d.csv", index=False)
    start_vs_p7b.to_csv(output_dir / "start_fold_sensitivity_vs_p7b.csv", index=False)
    dropout_vs_p3d = build_symbol_dropout_checks(result, p3d_ref, "p9", "p3d", horizon)
    dropout_vs_p7b = build_symbol_dropout_checks(result, p7b_ref, "p9", "p7b", horizon)
    dropout_vs_p3d.to_csv(output_dir / "symbol_dropout_vs_p3d.csv", index=False)
    dropout_vs_p7b.to_csv(output_dir / "symbol_dropout_vs_p7b.csv", index=False)
    reality_summary = build_reality_summary(summary, comparison, cost_summary, bootstrap, start_vs_p3d, start_vs_p7b, dropout_vs_p3d, dropout_vs_p7b)
    reality_summary.to_csv(output_dir / "reality_summary.csv", index=False)

    print("[8/10] Writing plots", flush=True)
    plot_clean_pnl(output_dir, result, p3d_ref, p7b_ref, horizon)
    p3d.plot_cost_sensitivity(output_dir, cost_summary)
    p3d.plot_selected_counts(output_dir, selected.rename(columns={"deriv_formula_name": "formula_name"}))
    p3d.plot_validation_vs_test(output_dir, result["folds"])
    p3d.plot_asset_contribution(output_dir, asset_summary)
    p7.plot_primitive_usage(output_dir, primitive_usage)
    plot_fold_delta(output_dir, fold_vs_p3d, "fold_delta_vs_p3d.png", "P9 minus P3d test fold return")
    plot_fold_delta(output_dir, fold_vs_p7b, "fold_delta_vs_p7b.png", "P9 minus P7b test fold return")
    plot_asset_delta(output_dir, asset_vs_p3d)
    plot_start_sensitivity(output_dir, start_vs_p3d, "start_fold_sensitivity_vs_p3d.png", "P9 minus P3d by start fold")
    plot_start_sensitivity(output_dir, start_vs_p7b, "start_fold_sensitivity_vs_p7b.png", "P9 minus P7b by start fold")

    print("[9/10] Writing report and config metadata", flush=True)
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
        primitive_usage=primitive_usage,
        fold_vs_p3d=fold_vs_p3d,
        fold_vs_p7b=fold_vs_p7b,
        bootstrap=bootstrap,
        start_vs_p3d=start_vs_p3d,
        start_vs_p7b=start_vs_p7b,
        dropout_vs_p3d=dropout_vs_p3d,
        dropout_vs_p7b=dropout_vs_p7b,
        reality_summary=reality_summary,
        formula_count=len(formulas),
        raw_formula_count=len(raw_formulas),
        workers=workers,
        horizon=horizon,
        validation_days=int(args.inner_validation_days),
        allow_k01=bool(args.allow_k01),
        require_30bps_improvement=not bool(args.no_require_30bps),
        short_share_limit=float(args.short_share_limit),
    )
    update_config_stage(config_path, allow_k01=bool(args.allow_k01), require_30bps_improvement=not bool(args.no_require_30bps), short_share_limit=float(args.short_share_limit))
    print("[10/10] Done", flush=True)
    print(f"Done: {output_dir}", flush=True)


def primitive_names(spec: FormulaSpec) -> set[str]:
    names: set[str] = set()
    for prim in spec.primitives:
        names.update(re.findall(r"P6B\(([^)]+)\)", prim))
    return names


def formula_tags(spec: FormulaSpec) -> dict[str, Any]:
    names = primitive_names(spec)
    has_h = any(name.startswith("H") for name in names)
    has_i = any(name.startswith("I") for name in names)
    has_j = any(name.startswith("J") for name in names)
    has_k = any(name.startswith("K") for name in names)
    has_k01 = "K01_deriv_crowding_contra" in names
    has_derivative = has_h or has_i or has_j or has_k
    has_ohlcv = any(not name[:1] in {"H", "I", "J", "K"} for name in names)
    return {
        "primitive_names": "|".join(sorted(names)),
        "has_h_funding": has_h,
        "has_i_premium_basis": has_i,
        "has_j_trader": has_j,
        "has_k_crowding": has_k,
        "has_k01": has_k01,
        "has_derivative": has_derivative,
        "has_ohlcv": has_ohlcv,
        "mixed_derivative_ohlcv": bool(has_derivative and has_ohlcv),
        "pure_derivative": bool(has_derivative and not has_ohlcv),
        "pure_ohlcv": bool(has_ohlcv and not has_derivative),
    }


def filter_p9_formulas(raw_formulas: list[FormulaSpec], exclude_k01: bool) -> list[FormulaSpec]:
    out = []
    for spec in raw_formulas:
        tags = formula_tags(spec)
        if exclude_k01 and tags["has_k01"]:
            continue
        out.append(spec)
    return out


def formula_table_with_tags(raw_formulas: list[FormulaSpec], clean_formulas: list[FormulaSpec]) -> pd.DataFrame:
    clean_names = {item.name for item in clean_formulas}
    rows = []
    for spec in raw_formulas:
        tags = formula_tags(spec)
        rows.append({**spec.to_row(), **tags, "p9_retained": spec.name in clean_names})
    return pd.DataFrame(rows)


def apply_p9_hard_accept(
    validation: pd.DataFrame,
    formulas: list[FormulaSpec],
    require_30bps_improvement: bool,
    short_share_limit: float,
) -> pd.DataFrame:
    tags = pd.DataFrame([{"deriv_formula_name": spec.name, **formula_tags(spec)} for spec in formulas])
    out = validation.merge(tags, on="deriv_formula_name", how="left")
    out["p9_hard_accept"] = out["hard_accept"].astype(bool) & (out["validation_short_abs_contribution_share"] <= float(short_share_limit)) & (
        out["validation_mean_effective_candidate_weight"] <= 0.100001
    )
    if require_30bps_improvement:
        out["p9_hard_accept"] = out["p9_hard_accept"] & (out["validation_total_return_30bps"] > out["p3d_validation_total_return_30bps"])
    out["hard_accept"] = out["p9_hard_accept"]
    return out


def build_summary(result: dict[str, Any], horizon: int) -> pd.DataFrame:
    metrics = p3d.summarize_nested_result(result, horizon)
    summary = pd.DataFrame([{**metrics, "run": "p9_clean_derivatives_baseline", "horizon": horizon}])
    summary["net_corr_to_p3d"] = p7.corr_to_reference(result["equity"], p7.load_reference_equity(P3D_EQUITY))
    summary["net_corr_to_p7b"] = p7.corr_to_reference(result["equity"], p7.load_reference_equity(P7B_EQUITY))
    summary["net_corr_to_p4_nested"] = p7.corr_to_reference(result["equity"], p3d.load_p4_equity())
    return summary


def build_reference_comparison(summary: pd.DataFrame, result: dict[str, Any]) -> pd.DataFrame:
    out = p7.add_p3d_comparison(summary.copy())
    p7b_summary_path = P7B_DIR / "summary.csv"
    if p7b_summary_path.exists():
        p7b_summary = pd.read_csv(p7b_summary_path)
        if not p7b_summary.empty:
            ref = p7b_summary.iloc[0].to_dict()
            for col in [
                "total_return",
                "gross_total_return_before_cost",
                "sharpe",
                "max_drawdown",
                "gross_max_drawdown",
                "mean_turnover",
                "cost_drag",
                "positive_fold_rate",
            ]:
                if col in out and col in ref:
                    out[f"p7b_{col}"] = float(ref[col])
                    out[f"delta_vs_p7b_{col}"] = out[col] - float(ref[col])
    return out


def build_cost_summary(result: dict[str, Any], cost_bps: list[float]) -> pd.DataFrame:
    cost_summary = p3d.build_cost_sensitivity({"p9_clean_derivatives_baseline": result}, cost_bps)
    cost_summary = p7.add_p3d_cost_comparison(cost_summary)
    p7b_cost_path = P7B_DIR / "cost_sensitivity.csv"
    if p7b_cost_path.exists():
        ref = pd.read_csv(p7b_cost_path)
        cols = [col for col in ["one_way_cost_bps", "total_return", "sharpe", "max_drawdown"] if col in ref]
        ref = ref[cols].rename(columns={col: f"p7b_{col}" for col in cols if col != "one_way_cost_bps"})
        cost_summary = cost_summary.merge(ref, on="one_way_cost_bps", how="left")
        for col in ["total_return", "sharpe", "max_drawdown"]:
            ref_col = f"p7b_{col}"
            if col in cost_summary and ref_col in cost_summary:
                cost_summary[f"delta_vs_p7b_{col}"] = cost_summary[col] - cost_summary[ref_col]
    return cost_summary


def load_existing_result(run_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        "folds": pd.read_csv(run_dir / "walk_forward_folds.csv", parse_dates=["train_start", "train_end", "test_start", "test_end", "test_decision_end"]),
        "equity": pd.read_csv(run_dir / "equity_curve.csv", parse_dates=["timestamp", "exit_timestamp"]),
        "positions": pd.read_csv(run_dir / "positions.csv", parse_dates=["timestamp", "exit_timestamp"]),
    }


def build_primitive_usage(selected: pd.DataFrame, formulas: list[FormulaSpec], retained: pd.DataFrame) -> pd.DataFrame:
    formula_map = {item.name: item for item in formulas}
    retained_names = retained["name"].astype(str).tolist()
    accepted = selected[~selected["fallback_to_p3d"].astype(bool)].copy()
    rows = []
    for name in retained_names:
        fold_count = 0
        weight_sum = 0.0
        for row in accepted.itertuples(index=False):
            spec = formula_map.get(str(row.deriv_formula_name))
            if spec is None:
                continue
            if name in primitive_names(spec):
                fold_count += 1
                weight_sum += float(row.candidate_weight_long)
        retained_row = retained[retained["name"] == name].iloc[0]
        rows.append(
            {
                "primitive": name,
                "family": retained_row["family"],
                "score_col": retained_row["score_col"],
                "selected_fold_count": fold_count,
                "selected_weight_sum": weight_sum,
                "p6b_primary_mean_ic": retained_row["primary_mean_ic"],
                "p6b_turnover_72h": retained_row["signal_turnover_72h"],
                "p6b_mean_abs_p3d_corr": retained_row["mean_abs_p3d_weight_corr"],
            }
        )
    return pd.DataFrame(rows).sort_values(["selected_fold_count", "selected_weight_sum"], ascending=[False, False])


def build_fold_delta(candidate: dict[str, pd.DataFrame], reference: dict[str, pd.DataFrame], cand: str, ref: str) -> pd.DataFrame:
    left = candidate["folds"].copy()
    right = reference["folds"].copy()
    if "formula_name" not in left and "deriv_formula_name" in left:
        left["formula_name"] = left["deriv_formula_name"]
    if "formula_name" not in right and "deriv_formula_name" in right:
        right["formula_name"] = right["deriv_formula_name"]
    cols = ["fold", "total_return", "gross_total_return_before_cost", "sharpe", "max_drawdown", "mean_turnover", "formula_name"]
    left = left[[col for col in cols if col in left]].rename(columns={col: f"{cand}_{col}" for col in cols if col != "fold"})
    right = right[[col for col in cols if col in right]].rename(columns={col: f"{ref}_{col}" for col in cols if col != "fold"})
    out = left.merge(right, on="fold", how="inner")
    for col in ["total_return", "gross_total_return_before_cost", "sharpe", "max_drawdown", "mean_turnover"]:
        out[f"delta_{col}"] = out[f"{cand}_{col}"] - out[f"{ref}_{col}"]
    return out


def build_side_delta(candidate: dict[str, pd.DataFrame], reference: dict[str, pd.DataFrame], cand: str, ref: str) -> pd.DataFrame:
    left = p3d.build_side_summary(candidate["positions"]).add_prefix(f"{cand}_").rename(columns={f"{cand}_side": "side"})
    right = p3d.build_side_summary(reference["positions"]).add_prefix(f"{ref}_").rename(columns={f"{ref}_side": "side"})
    out = left.merge(right, on="side", how="outer").fillna(0.0)
    for col in ["gross_contribution_sum", "mean_abs_weight", "position_count", "abs_contribution_share"]:
        out[f"delta_{col}"] = out[f"{cand}_{col}"] - out[f"{ref}_{col}"]
    return out


def build_asset_delta(candidate: dict[str, pd.DataFrame], reference: dict[str, pd.DataFrame], cand: str, ref: str) -> pd.DataFrame:
    left = p3d.build_asset_summary(candidate["positions"]).add_prefix(f"{cand}_").rename(columns={f"{cand}_symbol": "symbol"})
    right = p3d.build_asset_summary(reference["positions"]).add_prefix(f"{ref}_").rename(columns={f"{ref}_symbol": "symbol"})
    out = left.merge(right, on="symbol", how="outer").fillna(0.0)
    for col in ["gross_contribution_sum", "long_contribution_sum", "short_contribution_sum", "mean_abs_weight", "position_count", "abs_contribution_share"]:
        out[f"delta_{col}"] = out[f"{cand}_{col}"] - out[f"{ref}_{col}"]
    return out.sort_values("delta_gross_contribution_sum", ascending=False)


def bootstrap_fold_checks(fold_delta: pd.DataFrame, metric: str, samples: int) -> pd.DataFrame:
    rng = np.random.default_rng(3909)
    values = fold_delta["delta_total_return"].to_numpy(dtype=float)
    boot = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    sign = rng.choice([-1.0, 1.0], size=(samples, len(values))) * values
    sign_mean = sign.mean(axis=1)
    observed = float(values.mean())
    return pd.DataFrame(
        [
            {
                "metric": metric,
                "fold_count": len(values),
                "observed_mean_delta": observed,
                "bootstrap_q05": float(np.quantile(boot, 0.05)),
                "bootstrap_q50": float(np.quantile(boot, 0.50)),
                "bootstrap_q95": float(np.quantile(boot, 0.95)),
                "positive_delta_fold_rate": float((values > 0).mean()),
                "sign_flip_p_value_mean_le_0": float((sign_mean >= observed).mean()) if observed >= 0 else float((sign_mean <= observed).mean()),
            }
        ]
    )


def build_start_fold_sensitivity(candidate: dict[str, pd.DataFrame], reference: dict[str, pd.DataFrame], cand: str, ref: str, horizon: int) -> pd.DataFrame:
    rows = []
    folds = sorted(set(candidate["equity"]["fold"]).intersection(set(reference["equity"]["fold"])))
    for start in folds:
        for name, result in [(cand, candidate), (ref, reference)]:
            equity = result["equity"][result["equity"]["fold"] >= start].copy()
            positions = result["positions"][result["positions"]["fold"] >= start].copy()
            if equity.empty:
                continue
            metrics = p3d.summarize_nested_result({"equity": equity, "positions": positions, "folds": result["folds"][result["folds"]["fold"] >= start]}, horizon)
            rows.append({"start_fold": start, "variant": name, **metrics})
    out = pd.DataFrame(rows)
    pivot = out.pivot(index="start_fold", columns="variant", values="total_return").reset_index()
    if {cand, ref}.issubset(pivot.columns):
        pivot["delta_total_return"] = pivot[cand] - pivot[ref]
    return pivot


def build_symbol_dropout_checks(candidate: dict[str, pd.DataFrame], reference: dict[str, pd.DataFrame], cand: str, ref: str, horizon: int) -> pd.DataFrame:
    symbols = sorted(set(candidate["positions"]["symbol"]).union(set(reference["positions"]["symbol"])))
    rows = []
    for symbol in symbols:
        for name, result in [(cand, candidate), (ref, reference)]:
            approx = approximate_drop_symbol_result(result, symbol)
            metrics = p3d.summarize_nested_result(approx, horizon)
            rows.append({"symbol_dropped": symbol, "variant": name, "approx_total_return": metrics["total_return"], "approx_sharpe": metrics["sharpe"], "approx_max_drawdown": metrics["max_drawdown"]})
    out = pd.DataFrame(rows)
    pivot = out.pivot(index="symbol_dropped", columns="variant", values="approx_total_return").reset_index()
    if {cand, ref}.issubset(pivot.columns):
        pivot["delta_total_return"] = pivot[cand] - pivot[ref]
    return pivot.sort_values(cand)


def approximate_drop_symbol_result(result: dict[str, pd.DataFrame], symbol: str) -> dict[str, pd.DataFrame]:
    equity = result["equity"].copy()
    positions = result["positions"].copy()
    contribution = positions[positions["symbol"] == symbol].groupby("timestamp")["contribution"].sum()
    equity["period_return"] = equity["period_return"] - equity["timestamp"].map(contribution).fillna(0.0)
    equity["gross_period_return"] = equity["gross_period_return"] - equity["timestamp"].map(contribution).fillna(0.0)
    equity = equity.sort_values("timestamp")
    equity["equity"] = (1.0 + equity["period_return"]).cumprod()
    positions = positions[positions["symbol"] != symbol].copy()
    return {"folds": result["folds"], "equity": equity, "positions": positions}


def build_reality_summary(
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    cost_summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
    start_vs_p3d: pd.DataFrame,
    start_vs_p7b: pd.DataFrame,
    dropout_vs_p3d: pd.DataFrame,
    dropout_vs_p7b: pd.DataFrame,
) -> pd.DataFrame:
    row = summary.iloc[0]
    comp = comparison.iloc[0]
    cost20 = cost_summary[cost_summary["one_way_cost_bps"] == 20.0].iloc[0]
    cost30 = cost_summary[cost_summary["one_way_cost_bps"] == 30.0].iloc[0]
    b_p3d = bootstrap[bootstrap["metric"] == "p9_minus_p3d"].iloc[0]
    b_p7b = bootstrap[bootstrap["metric"] == "p9_minus_p7b"].iloc[0]
    start_p3d = float((start_vs_p3d["delta_total_return"] > 0).mean()) if "delta_total_return" in start_vs_p3d else np.nan
    start_p7b = float((start_vs_p7b["delta_total_return"] > 0).mean()) if "delta_total_return" in start_vs_p7b else np.nan
    drop_p3d = float((dropout_vs_p3d["delta_total_return"] > 0).mean()) if "delta_total_return" in dropout_vs_p3d else np.nan
    drop_p7b = float((dropout_vs_p7b["delta_total_return"] > 0).mean()) if "delta_total_return" in dropout_vs_p7b else np.nan
    return pd.DataFrame(
        [
            {"check": "p9_vs_p3d_net_delta", "value": comp.get("delta_vs_p3d_total_return", np.nan), "pass": bool(comp.get("delta_vs_p3d_total_return", -np.inf) > 0.0)},
            {"check": "p9_vs_p3d_sharpe_delta", "value": comp.get("delta_vs_p3d_sharpe", np.nan), "pass": bool(comp.get("delta_vs_p3d_sharpe", -np.inf) > 0.0)},
            {"check": "p9_vs_p7b_net_delta", "value": comp.get("delta_vs_p7b_total_return", np.nan), "pass": bool(comp.get("delta_vs_p7b_total_return", -np.inf) >= -0.0025)},
            {"check": "p9_vs_p7b_sharpe_delta", "value": comp.get("delta_vs_p7b_sharpe", np.nan), "pass": bool(comp.get("delta_vs_p7b_sharpe", -np.inf) >= -0.05)},
            {"check": "p9_20bps_return", "value": cost20["total_return"], "pass": bool(cost20["total_return"] > 0.0)},
            {"check": "p9_30bps_return", "value": cost30["total_return"], "pass": bool(cost30["total_return"] > -0.02)},
            {"check": "p9_short_abs_share", "value": row.get("short_abs_contribution_share", np.nan), "pass": bool(row.get("short_abs_contribution_share", 0.0) <= 0.70)},
            {"check": "bootstrap_delta_vs_p3d_q05", "value": b_p3d["bootstrap_q05"], "pass": bool(b_p3d["bootstrap_q05"] > -0.0025)},
            {"check": "bootstrap_delta_vs_p7b_q05", "value": b_p7b["bootstrap_q05"], "pass": bool(b_p7b["bootstrap_q05"] > -0.0025)},
            {"check": "start_fold_positive_vs_p3d", "value": start_p3d, "pass": bool(start_p3d >= 0.60)},
            {"check": "start_fold_positive_vs_p7b", "value": start_p7b, "pass": bool(start_p7b >= 0.45)},
            {"check": "symbol_dropout_positive_vs_p3d", "value": drop_p3d, "pass": bool(drop_p3d >= 0.60)},
            {"check": "symbol_dropout_positive_vs_p7b", "value": drop_p7b, "pass": bool(drop_p7b >= 0.45)},
        ]
    )


def plot_clean_pnl(output_dir: Path, result: dict[str, Any], p3d_ref: dict[str, pd.DataFrame], p7b_ref: dict[str, pd.DataFrame], horizon: int) -> None:
    equity = result["equity"].sort_values("timestamp")
    if equity.empty:
        return
    fig, axes = plt.subplots(2, 1, figsize=(13, 8.5), sharex=True, gridspec_kw={"height_ratios": [2.3, 1.0]})
    gross = (1.0 + equity["gross_period_return"]).cumprod() - 1.0
    net = (1.0 + equity["period_return"]).cumprod() - 1.0
    axes[0].plot(equity["timestamp"], gross, label="P9 gross before cost", linewidth=2.1)
    axes[0].plot(equity["timestamp"], net, label="P9 net after 6bps", linewidth=2.1)
    for label, ref, color in [("P3d net", p3d_ref, "#555555"), ("P7b net", p7b_ref, "#8a5a44")]:
        ref_eq = ref["equity"].sort_values("timestamp")
        ref_net = (1.0 + ref_eq["period_return"]).cumprod() - 1.0
        axes[0].plot(ref_eq["timestamp"], ref_net, label=label, linewidth=1.5, alpha=0.75, color=color)
    axes[0].axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    axes[0].set_title(f"P9 clean derivatives baseline nested test ({horizon}h)")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)
    curve = (1.0 + equity["period_return"]).cumprod()
    drawdown = curve / curve.cummax() - 1.0
    axes[1].plot(equity["timestamp"], drawdown, label="P9 net drawdown", linewidth=1.8)
    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "clean_pnl_curve.png", dpi=180)
    plt.close(fig)


def plot_fold_delta(output_dir: Path, fold_delta: pd.DataFrame, filename: str, title: str) -> None:
    plt.figure(figsize=(12, 6))
    colors = np.where(fold_delta["delta_total_return"] >= 0, "#2a9d8f", "#c44536")
    plt.bar(fold_delta["fold"].astype(str), fold_delta["delta_total_return"], color=colors)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.title(title)
    plt.ylabel("delta return")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=180)
    plt.close()


def plot_asset_delta(output_dir: Path, asset_delta: pd.DataFrame) -> None:
    frame = asset_delta.sort_values("delta_gross_contribution_sum")
    plt.figure(figsize=(11, 7))
    colors = np.where(frame["delta_gross_contribution_sum"] >= 0, "#2a9d8f", "#c44536")
    plt.barh(frame["symbol"], frame["delta_gross_contribution_sum"], color=colors)
    plt.axvline(0.0, color="black", linewidth=0.8)
    plt.title("P9 minus P3d asset gross contribution")
    plt.gca().xaxis.set_major_formatter(PercentFormatter(1.0))
    plt.tight_layout()
    plt.savefig(output_dir / "asset_delta_vs_p3d.png", dpi=180)
    plt.close()


def plot_start_sensitivity(output_dir: Path, frame: pd.DataFrame, filename: str, title: str) -> None:
    if "delta_total_return" not in frame:
        return
    plt.figure(figsize=(11, 6))
    plt.plot(frame["start_fold"], frame["delta_total_return"], marker="o")
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.title(title)
    plt.xlabel("start fold")
    plt.ylabel("delta total return")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=180)
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
    primitive_usage: pd.DataFrame,
    fold_vs_p3d: pd.DataFrame,
    fold_vs_p7b: pd.DataFrame,
    bootstrap: pd.DataFrame,
    start_vs_p3d: pd.DataFrame,
    start_vs_p7b: pd.DataFrame,
    dropout_vs_p3d: pd.DataFrame,
    dropout_vs_p7b: pd.DataFrame,
    reality_summary: pd.DataFrame,
    formula_count: int,
    raw_formula_count: int,
    workers: int,
    horizon: int,
    validation_days: int,
    allow_k01: bool,
    require_30bps_improvement: bool,
    short_share_limit: float,
) -> None:
    selected_cols = [
        "fold",
        "p4_spec",
        "p3d_formula_name",
        "deriv_formula_name",
        "candidate_weight_long",
        "candidate_weight_short",
        "high_risk_candidate_cap",
        "hard_accept",
        "fallback_to_p3d",
        "validation_fitness",
        "validation_total_return",
        "validation_sharpe",
        "validation_short_abs_contribution_share",
        "validation_candidate_score_corr_to_p3d",
        "deriv_formula",
    ]
    selected_cols = [col for col in selected_cols if col in selected]
    accepted = validation[validation["hard_accept"].astype(bool)].sort_values("validation_fitness", ascending=False).head(30)
    accepted_cols = [
        "fold",
        "deriv_formula_name",
        "candidate_weight_long",
        "high_risk_candidate_cap",
        "validation_fitness",
        "validation_total_return",
        "validation_total_return_20bps",
        "validation_total_return_30bps",
        "validation_sharpe",
        "validation_short_abs_contribution_share",
        "mixed_derivative_ohlcv",
        "has_h_funding",
        "deriv_formula",
    ]
    accepted_cols = [col for col in accepted_cols if col in accepted]
    cost_display = cost_summary[cost_summary["one_way_cost_bps"].isin([0, 6, 10, 15, 20, 30])]
    lines = [
        "# P9 Clean Derivatives Baseline",
        "",
        "## Design",
        "",
        f"- Horizon fixed at `{horizon}h`.",
        "- Base score is the already-selected P3d score per fold.",
        f"- P9 uses P6b retained OHLCV + derivatives primitives; K01 allowed: `{allow_k01}`.",
        "- Candidate rerank is capped at 10%: weights are `0.05 / 0.075 / 0.10`.",
        f"- Hard accept is against P3d validation; requires 30bps validation improvement: `{require_30bps_improvement}`.",
        f"- Validation short absolute contribution share limit: `{short_share_limit:.2f}`.",
        "- Candidate can only rerank inside the P3d/P4-approved long and short pools; it cannot create a new side from scratch.",
        f"- Each fold selects formula+weight using only the last `{validation_days}` days inside its prior training window.",
        "- If no candidate passes hard accept, the fold falls back to P3d.",
        f"- Raw formulas: `{raw_formula_count}`; P9 clean formulas: `{formula_count}`; workers: `{workers}`.",
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
        "## Selected By Fold",
        "```text",
        selected[selected_cols].to_string(index=False),
        "```",
        "",
        "## Accepted Validation Leaders",
        "```text",
        accepted[accepted_cols].to_string(index=False),
        "```",
        "",
        "## Primitive Usage",
        "```text",
        primitive_usage.to_string(index=False),
        "```",
        "",
        "## Cost Sensitivity",
        "```text",
        cost_display.to_string(index=False),
        "```",
        "",
        "## Regime Summary",
        "```text",
        regime_summary.to_string(index=False),
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
        "",
        "## Fold Delta Vs P3d",
        "```text",
        fold_vs_p3d.to_string(index=False),
        "```",
        "",
        "## Fold Delta Vs P7b",
        "```text",
        fold_vs_p7b.to_string(index=False),
        "```",
        "",
        "## Bootstrap Checks",
        "```text",
        bootstrap.to_string(index=False),
        "```",
        "",
        "## Start Fold Sensitivity Vs P3d",
        "```text",
        start_vs_p3d.to_string(index=False),
        "```",
        "",
        "## Start Fold Sensitivity Vs P7b",
        "```text",
        start_vs_p7b.to_string(index=False),
        "```",
        "",
        "## Symbol Dropout Vs P3d",
        "```text",
        dropout_vs_p3d.to_string(index=False),
        "```",
        "",
        "## Symbol Dropout Vs P7b",
        "```text",
        dropout_vs_p7b.to_string(index=False),
        "```",
        "",
        "## Reality Summary",
        "```text",
        reality_summary.to_string(index=False),
        "```",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def update_config_stage(config_path: Path, allow_k01: bool, require_30bps_improvement: bool, short_share_limit: float) -> None:
    cfg = load_config(config_path)
    stages = cfg.setdefault("stages", {})
    stages["p9_clean_derivatives_baseline"] = {
        "script": "../../../scripts/run_p9_clean_derivatives_baseline.py",
        "output_dir": "../outputs/p9_clean_derivatives_baseline",
        "report": "../outputs/p9_clean_derivatives_baseline/REPORT.md",
        "summary": "../outputs/p9_clean_derivatives_baseline/summary.csv",
        "baseline_comparison": "../outputs/p9_clean_derivatives_baseline/baseline_comparison.csv",
        "cost_sensitivity": "../outputs/p9_clean_derivatives_baseline/cost_sensitivity.csv",
        "selected_incremental_by_fold": "../outputs/p9_clean_derivatives_baseline/selected_incremental_by_fold.csv",
        "reality_summary": "../outputs/p9_clean_derivatives_baseline/reality_summary.csv",
        "clean_pnl_plot": "../outputs/p9_clean_derivatives_baseline/clean_pnl_curve.png",
    }
    cfg["active_research_baseline_candidate"] = {
        "name": "p9_clean_derivatives_baseline",
        "status": "candidate_pending_research_review",
        "output_dir": "../outputs/p9_clean_derivatives_baseline",
        "report": "../outputs/p9_clean_derivatives_baseline/REPORT.md",
        "clean_pnl_plot": "../outputs/p9_clean_derivatives_baseline/clean_pnl_curve.png",
        "parent": "p7b_derivatives_incremental_on_p3d",
        "constraints": {
            "exclude_k01_deriv_crowding_contra": not bool(allow_k01),
            "max_candidate_weight": 0.10,
            "requires_30bps_validation_improvement_vs_p3d": bool(require_30bps_improvement),
            "short_abs_contribution_share_limit": float(short_share_limit),
        },
    }
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False), encoding="utf-8")


if __name__ == "__main__":
    main()
