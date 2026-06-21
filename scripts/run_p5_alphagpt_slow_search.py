#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import random
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

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
from run_p2_family_b_mean_reversion import (
    build_baseline_comparison,
    load_baseline_summary,
    make_locked_risk_config,
    make_portfolio_config,
    run_regime_gated_backtest,
)
from run_p4_nested_walkforward_selection import decision_end_for_closed_positions, summarize_nested_result
from run_residual_alphagpt_search import aggregate_oos_metrics, build_candidate_scores, prepare_alpha_panel
from run_risk_control_experiments import add_gross_metrics, add_risk_event_metrics
from run_delay_smoothing_experiments import add_delay_metrics
from run_turnover_experiments import to_jsonable


EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "baseline_locked_v1"
DEFAULT_CONFIG = EXPERIMENT_ROOT / "configs" / "baseline_locked_v1.yaml"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "outputs" / "p5_alphagpt_slow_search"
MANUAL_FEATURE_PANEL = EXPERIMENT_ROOT / "outputs" / "p2_manual_slow_alpha_grid" / "manual_slow_feature_panel.csv.gz"
SLOW_LABELS = EXPERIMENT_ROOT / "outputs" / "p1_slow_labels" / "slow_residual_labels.csv.gz"
P4_SUMMARY = EXPERIMENT_ROOT / "outputs" / "p4_nested_walkforward_selection" / "summary.csv"

SCORE_COL = "_alphagpt_slow_score"
COMMITTEE_COL = "_alphagpt_slow_committee"

_WORKER_PANEL: pd.DataFrame | None = None
_WORKER_PF_CFG: Any = None
_WORKER_RISK_CFG: Any = None
_WORKER_FORMULAS: dict[str, "FormulaSpec"] = {}
_WORKER_BASELINE: pd.DataFrame | None = None
_WORKER_HORIZON: int = 72


@dataclass(frozen=True)
class FormulaNode:
    op: str
    value: str | float | None = None
    children: tuple["FormulaNode", ...] = ()

    def expr(self) -> str:
        if self.op == "FEATURE":
            return str(self.value)
        if self.op == "CONST":
            return f"{float(self.value):.2f}"
        if self.op == "SCALE":
            return f"{float(self.value):.2f}*{self.children[0].expr()}"
        if self.op == "NEG":
            return f"NEG({self.children[0].expr()})"
        if self.op == "CLIP":
            return f"CLIP({self.children[0].expr()})"
        if self.op in {"ADD", "SUB", "MUL", "DIV_SAFE"}:
            return f"{self.op}({self.children[0].expr()},{self.children[1].expr()})"
        raise ValueError(f"Unknown op: {self.op}")

    def size(self) -> int:
        return 1 + sum(child.size() for child in self.children)

    def primitive_set(self) -> frozenset[str]:
        if self.op == "FEATURE":
            return frozenset([str(self.value)])
        out: set[str] = set()
        for child in self.children:
            out.update(child.primitive_set())
        return frozenset(out)


@dataclass(frozen=True)
class FormulaSpec:
    name: str
    node: FormulaNode
    formula: str
    complexity: int
    primitives: tuple[str, ...]
    source: str

    def to_row(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "formula": self.formula,
            "complexity": self.complexity,
            "primitive_count": len(self.primitives),
            "primitives": "|".join(self.primitives),
            "source": self.source,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--horizon", type=int, default=72)
    parser.add_argument("--formulas", type=int, default=180)
    parser.add_argument("--seed", type=int, default=1707)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--chunksize", type=int, default=8)
    parser.add_argument("--inner-validation-days", type=int, default=90)
    parser.add_argument("--committee-size", type=int, default=3)
    parser.add_argument("--cost-bps", type=float, nargs="+", default=[0, 4, 6, 8, 10, 12, 15, 20, 30])
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
    formulas = generate_formula_specs(count=int(args.formulas), seed=int(args.seed), max_depth=int(args.max_depth))
    pd.DataFrame([item.to_row() for item in formulas]).to_csv(output_dir / "candidate_formulas.csv", index=False)

    print(
        f"[1/9] Nested AlphaGPT slow search: {len(selections)} folds x {len(formulas)} formulas "
        f"(prior-validation fitness), workers={workers}",
        flush=True,
    )
    tasks = build_validation_tasks(selections, formulas, horizon, int(args.inner_validation_days))
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
                print(f"      validation formula task {idx}/{len(tasks)}", flush=True)
            validation_rows.append(run_validation_task(task))
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers, initializer=init_worker, initargs=worker_args) as pool:
            for idx, row in enumerate(pool.imap_unordered(run_validation_task, tasks, chunksize=max(1, int(args.chunksize))), start=1):
                validation_rows.append(row)
                if idx == 1 or idx % 250 == 0 or idx == len(tasks):
                    print(f"      validation formula task {idx}/{len(tasks)}", flush=True)

    validation = pd.DataFrame(validation_rows)
    validation.to_csv(output_dir / "validation_formula_metrics.csv", index=False)

    print("[2/9] Selecting per-fold single formula and low-overlap top-3 committee", flush=True)
    selected = select_formulas_by_fold(validation, selections, formulas, horizon, int(args.committee_size))
    selected.to_csv(output_dir / "selected_formulas_by_fold.csv", index=False)

    print("[3/9] Replaying selected formulas on test windows only", flush=True)
    panel = load_panel(config_path, panel_path, MANUAL_FEATURE_PANEL, horizon)
    pf_cfg = make_portfolio_config(cfg, horizon)
    risk_cfg = make_locked_risk_config(cfg)
    formula_by_name = {item.name: item for item in formulas}
    single_result = replay_selected_tests(
        panel=panel,
        selections=selections,
        selected=selected,
        formulas=formula_by_name,
        pf_cfg=pf_cfg,
        risk_cfg=risk_cfg,
        mode="single",
    )
    committee_result = replay_selected_tests(
        panel=panel,
        selections=selections,
        selected=selected,
        formulas=formula_by_name,
        pf_cfg=pf_cfg,
        risk_cfg=risk_cfg,
        mode="committee_top3",
    )
    write_run_result(ensure_dir(output_dir / "nested_single_run"), single_result)
    write_run_result(ensure_dir(output_dir / "nested_committee_top3_run"), committee_result)

    print("[4/9] Summarizing nested test results", flush=True)
    result_index = {
        "p5_alphagpt_single": single_result,
        "p5_alphagpt_committee_top3": committee_result,
    }
    summary_rows = []
    for name, result in result_index.items():
        row = summarize_nested_result(result, horizon)
        row["run"] = name
        row["horizon"] = horizon
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary = add_correlations_to_reference(summary, result_index, horizon)
    summary.to_csv(output_dir / "summary.csv", index=False)

    baseline = load_baseline_summary(EXPERIMENT_ROOT / "baseline_lock_review" / "horizon_summary.csv")
    comparison = build_baseline_comparison(summary.copy(), baseline)
    comparison = add_p4_comparison(comparison)
    comparison.to_csv(output_dir / "baseline_comparison.csv", index=False)

    print("[5/9] Cost, regime, side, and asset diagnostics", flush=True)
    cost_summary = build_cost_sensitivity(result_index, cost_bps)
    cost_summary.to_csv(output_dir / "cost_sensitivity.csv", index=False)
    best_name = str(summary.sort_values(["total_return", "sharpe"], ascending=[False, False]).iloc[0]["run"])
    best_result = result_index[best_name]
    equity = best_result["equity"].sort_values("timestamp")
    positions = best_result["positions"].sort_values(["timestamp", "symbol"])
    regime_summary = build_regime_summary(equity, build_regime_frame(panel), horizon)
    side_summary = build_side_summary(positions)
    asset_summary = build_asset_summary(positions)
    regime_summary.to_csv(output_dir / "best_regime_summary.csv", index=False)
    side_summary.to_csv(output_dir / "best_side_contribution.csv", index=False)
    asset_summary.to_csv(output_dir / "best_asset_contribution.csv", index=False)

    print("[6/9] IC diagnostics against slow labels where available", flush=True)
    ic_summary = build_selected_ic_summary(panel, selected, formula_by_name, horizon)
    ic_summary.to_csv(output_dir / "selected_formula_ic_summary.csv", index=False)

    print("[7/9] Writing plots", flush=True)
    plot_nested_curves(output_dir, result_index, horizon)
    plot_cost_sensitivity(output_dir, cost_summary)
    plot_selected_counts(output_dir, selected)
    plot_validation_vs_test(output_dir, validation, single_result["folds"], committee_result["folds"])
    plot_asset_contribution(output_dir, asset_summary, best_name)

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
        ic_summary=ic_summary,
        best_name=best_name,
        formula_count=len(formulas),
        workers=workers,
        horizon=horizon,
        inner_validation_days=int(args.inner_validation_days),
    )
    update_config_stage(config_path)

    print("[9/9] Done", flush=True)
    print(f"Done: {output_dir}", flush=True)


def resolve_config_path(config_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def build_validation_tasks(
    selections: pd.DataFrame,
    formulas: list[FormulaSpec],
    horizon: int,
    validation_days: int,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for _, fold in selections.iterrows():
        train_start = pd.Timestamp(fold["train_start"])
        train_end = pd.Timestamp(fold["train_end"])
        validation_start = max(train_start + (train_end - train_start) / 2, train_end - pd.Timedelta(days=validation_days))
        validation_end = decision_end_for_closed_positions(train_end, horizon)
        train_decision_end = decision_end_for_closed_positions(validation_start, horizon)
        if train_decision_end <= train_start:
            train_decision_end = validation_start
        for formula in formulas:
            tasks.append(
                {
                    "fold": int(fold["fold"]),
                    "formula_name": formula.name,
                    "train_start": train_start,
                    "train_decision_end": train_decision_end,
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
    global _WORKER_PANEL, _WORKER_PF_CFG, _WORKER_RISK_CFG, _WORKER_FORMULAS, _WORKER_BASELINE, _WORKER_HORIZON
    config_path = Path(config_path_str)
    cfg = load_config(config_path)
    _WORKER_PANEL = load_panel(config_path, Path(panel_path_str), Path(feature_path_str), horizon)
    _WORKER_PANEL[SCORE_COL] = np.nan
    _WORKER_PF_CFG = make_portfolio_config(cfg, horizon)
    _WORKER_RISK_CFG = make_locked_risk_config(cfg)
    _WORKER_FORMULAS = formulas
    _WORKER_HORIZON = int(horizon)
    _WORKER_BASELINE = load_reference_equity(horizon)


def load_panel(config_path: Path, panel_path: Path, feature_path: Path, horizon: int) -> pd.DataFrame:
    panel = pd.read_csv(panel_path, parse_dates=["timestamp"])
    panel = prepare_alpha_panel(panel, [horizon], execution_delay_bars=1)
    build_candidate_scores(panel)
    feature_cols = sorted(required_feature_columns())
    features = pd.read_csv(feature_path, parse_dates=["timestamp"], usecols=["timestamp", "symbol", *feature_cols])
    panel = panel.merge(features, on=["timestamp", "symbol"], how="left")
    if SLOW_LABELS.exists():
        labels = pd.read_csv(
            SLOW_LABELS,
            parse_dates=["timestamp"],
            usecols=["timestamp", "symbol", "label_path_slow_low_turnover", "label_slow_low_turnover"],
        )
        panel = panel.merge(labels, on=["timestamp", "symbol"], how="left")
    add_transformed_primitives(panel)
    return panel.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def required_feature_columns() -> set[str]:
    cols = {col for _, col in primitive_catalog_base()}
    cols.update(
        {
            "alpha_A_res_trend_quality",
            "alpha_B_seed_168",
            "alpha_B_seed_336",
            "alpha_B_seed_combo",
            "alpha_B_soft_combo",
            "alpha_C_quality_committee",
            "family_b_regime_gate",
            "market_vol_14d",
            "residual_dispersion_7d",
        }
    )
    return cols


def primitive_catalog_base() -> list[tuple[str, str]]:
    return [
        ("CS_RANK(res_mom_72_12)", "cs_res_mom_72_12"),
        ("CS_RANK(res_mom_96_12)", "cs_res_mom_96_12"),
        ("CS_RANK(res_mom_168_24)", "cs_res_mom_168_24"),
        ("CS_RANK(res_mom_336_24)", "cs_res_mom_336_24"),
        ("CS_RANK(res_trend_quality_168_24)", "cs_res_trend_quality_168_24"),
        ("CS_RANK(res_high_mom_168)", "cs_res_high_mom_168"),
        ("CS_RANK(res_high_mom_336)", "cs_res_high_mom_336"),
        ("CS_RANK(res_ma_slope_48_168)", "cs_res_ma_slope_48_168"),
        ("CS_RANK(res_z_168)", "cs_res_z_168"),
        ("CS_RANK(res_z_336)", "cs_res_z_336"),
        ("CS_RANK(ou_speed_168)", "cs_ou_speed_168"),
        ("CS_RANK(ou_speed_336)", "cs_ou_speed_336"),
        ("PCT_RANK(ou_speed_168)", "pct_ou_speed_168"),
        ("PCT_RANK(ou_speed_336)", "pct_ou_speed_336"),
        ("CS_RANK(beta_stability_168)", "cs_beta_stability_168"),
        ("CS_RANK(beta_stability_336)", "cs_beta_stability_336"),
        ("PCT_RANK(beta_stability_168)", "pct_beta_stability_168"),
        ("PCT_RANK(beta_stability_336)", "pct_beta_stability_336"),
        ("CS_RANK(liquidity_quality)", "cs_liquidity_quality"),
        ("PCT_RANK(liquidity_quality)", "pct_liquidity_quality"),
        ("CS_RANK(volume_share_trend_7d_30d)", "cs_volume_share_trend_7d_30d"),
        ("CS_RANK(vol_of_vol_res_168)", "cs_vol_of_vol_res_168"),
        ("stationarity_gate_168", "stationarity_gate_168"),
        ("stationarity_gate_336", "stationarity_gate_336"),
        ("regime_gate", "family_b_regime_gate"),
    ]


def add_transformed_primitives(panel: pd.DataFrame) -> None:
    grouped = panel.groupby("symbol", group_keys=False)
    transform_cols = [
        "cs_res_mom_168_24",
        "cs_res_mom_336_24",
        "cs_res_trend_quality_168_24",
        "cs_res_z_168",
        "cs_res_z_336",
        "cs_liquidity_quality",
        "cs_volume_share_trend_7d_30d",
        "cs_vol_of_vol_res_168",
        "alpha_B_soft_combo",
        "alpha_C_quality_committee",
    ]
    for col in transform_cols:
        if col not in panel:
            continue
        for span in [72, 168, 336]:
            out_col = f"ema{span}_{col}"
            panel[out_col] = grouped[col].transform(
                lambda s, sp=span: s.ewm(span=sp, adjust=False, min_periods=max(24, sp // 3)).mean()
            )

    market = panel.groupby("timestamp")["log_ret"].mean().sort_index()
    market_ret_7d = market.rolling(168, min_periods=48).sum()
    market_ret_30d = market.rolling(720, min_periods=240).sum()
    market_vol = market.rolling(336, min_periods=72).std(ddof=0)
    high_vol_threshold = market_vol.rolling(4320, min_periods=720).quantile(0.80)
    panel["regime_crash_7d"] = panel["timestamp"].map((market_ret_7d < -0.05).astype(float)).fillna(0.0)
    panel["regime_down_30d"] = panel["timestamp"].map((market_ret_30d < 0.0).astype(float)).fillna(0.0)
    panel["regime_high_vol"] = panel["timestamp"].map((market_vol > high_vol_threshold).astype(float)).fillna(0.0)


def primitive_nodes() -> list[FormulaNode]:
    primitives = [FormulaNode("FEATURE", expr) for expr, _ in primitive_catalog_base()]
    high_level = [
        ("A07", "alpha_A_res_trend_quality"),
        ("B11", "alpha_B_seed_combo"),
        ("B12", "alpha_B_soft_combo"),
        ("C09", "alpha_C_quality_committee"),
    ]
    primitives.extend(FormulaNode("FEATURE", expr) for expr, _ in high_level)
    for expr, col in [
        ("EMA_SLOW(CS_RANK(res_mom_168_24),168)", "ema168_cs_res_mom_168_24"),
        ("EMA_SLOW(CS_RANK(res_mom_336_24),168)", "ema168_cs_res_mom_336_24"),
        ("EMA_SLOW(CS_RANK(res_trend_quality_168_24),168)", "ema168_cs_res_trend_quality_168_24"),
        ("EMA_SLOW(CS_RANK(res_z_168),168)", "ema168_cs_res_z_168"),
        ("EMA_SLOW(CS_RANK(res_z_336),168)", "ema168_cs_res_z_336"),
        ("EMA_SLOW(B12,168)", "ema168_alpha_B_soft_combo"),
        ("EMA_SLOW(C09,168)", "ema168_alpha_C_quality_committee"),
        ("EMA_SLOW(CS_RANK(res_mom_336_24),336)", "ema336_cs_res_mom_336_24"),
        ("EMA_SLOW(CS_RANK(res_z_336),336)", "ema336_cs_res_z_336"),
    ]:
        primitives.append(FormulaNode("FEATURE", expr))
    return primitives


def feature_column(expr: str) -> str:
    mapping = dict(primitive_catalog_base())
    mapping.update(
        {
            "A07": "alpha_A_res_trend_quality",
            "B11": "alpha_B_seed_combo",
            "B12": "alpha_B_soft_combo",
            "C09": "alpha_C_quality_committee",
            "EMA_SLOW(CS_RANK(res_mom_168_24),168)": "ema168_cs_res_mom_168_24",
            "EMA_SLOW(CS_RANK(res_mom_336_24),168)": "ema168_cs_res_mom_336_24",
            "EMA_SLOW(CS_RANK(res_trend_quality_168_24),168)": "ema168_cs_res_trend_quality_168_24",
            "EMA_SLOW(CS_RANK(res_z_168),168)": "ema168_cs_res_z_168",
            "EMA_SLOW(CS_RANK(res_z_336),168)": "ema168_cs_res_z_336",
            "EMA_SLOW(B12,168)": "ema168_alpha_B_soft_combo",
            "EMA_SLOW(C09,168)": "ema168_alpha_C_quality_committee",
            "EMA_SLOW(CS_RANK(res_mom_336_24),336)": "ema336_cs_res_mom_336_24",
            "EMA_SLOW(CS_RANK(res_z_336),336)": "ema336_cs_res_z_336",
        }
    )
    if expr not in mapping:
        raise KeyError(f"Unknown primitive expression: {expr}")
    return mapping[expr]


def generate_formula_specs(count: int, seed: int, max_depth: int) -> list[FormulaSpec]:
    rng = random.Random(seed)
    nodes = seed_nodes()
    seen = {node.expr() for node in nodes}
    attempts = 0
    while len(nodes) < count and attempts < count * 30:
        attempts += 1
        node = random_formula_node(rng, max_depth)
        expr = node.expr()
        if expr in seen:
            continue
        if not formula_allowed(node):
            continue
        nodes.append(node)
        seen.add(expr)

    specs: list[FormulaSpec] = []
    for idx, node in enumerate(nodes, start=1):
        source = "seed" if idx <= len(seed_nodes()) else "random_slow_grammar"
        specs.append(
            FormulaSpec(
                name=f"agpt_{idx:04d}",
                node=node,
                formula=node.expr(),
                complexity=node.size(),
                primitives=tuple(sorted(node.primitive_set())),
                source=source,
            )
        )
    return specs


def seed_nodes() -> list[FormulaNode]:
    f = lambda expr: FormulaNode("FEATURE", expr)
    scale = lambda value, node: FormulaNode("SCALE", value, (node,))
    add = lambda left, right: FormulaNode("ADD", children=(left, right))
    mul = lambda left, right: FormulaNode("MUL", children=(left, right))
    neg = lambda node: FormulaNode("NEG", children=(node,))

    return [
        f("A07"),
        f("B11"),
        f("B12"),
        f("C09"),
        add(scale(0.40, f("B11")), scale(0.60, f("C09"))),
        add(scale(0.35, f("A07")), scale(0.65, f("C09"))),
        add(f("CS_RANK(res_mom_168_24)"), scale(0.50, f("CS_RANK(res_high_mom_168)"))),
        add(f("CS_RANK(res_mom_168_24)"), scale(-0.50, f("CS_RANK(vol_of_vol_res_168)"))),
        add(f("CS_RANK(res_mom_336_24)"), scale(-0.50, f("CS_RANK(vol_of_vol_res_168)"))),
        add(f("CS_RANK(res_mom_168_24)"), scale(0.30, f("CS_RANK(volume_share_trend_7d_30d)"))),
        mul(neg(f("CS_RANK(res_z_168)")), f("stationarity_gate_168")),
        mul(mul(neg(f("CS_RANK(res_z_336)")), f("stationarity_gate_336")), f("PCT_RANK(ou_speed_336)")),
        add(
            add(scale(-0.55, f("CS_RANK(res_z_168)")), scale(-0.45, f("CS_RANK(res_z_336)"))),
            add(scale(0.25, f("CS_RANK(ou_speed_168)")), scale(0.15, f("CS_RANK(beta_stability_168)"))),
        ),
        add(
            add(scale(0.45, f("EMA_SLOW(CS_RANK(res_mom_336_24),336)")), scale(0.35, f("EMA_SLOW(B12,168)"))),
            scale(-0.20, f("CS_RANK(vol_of_vol_res_168)")),
        ),
        add(
            add(scale(0.50, f("EMA_SLOW(C09,168)")), scale(0.30, f("CS_RANK(liquidity_quality)"))),
            scale(-0.20, f("CS_RANK(vol_of_vol_res_168)")),
        ),
    ]


def random_formula_node(rng: random.Random, max_depth: int) -> FormulaNode:
    primitives = primitive_nodes()
    if max_depth <= 0 or rng.random() < 0.35:
        node = rng.choice(primitives)
        if rng.random() < 0.25:
            return FormulaNode("SCALE", rng.choice([0.20, 0.35, 0.50, 0.70, 1.20]), (node,))
        return node
    roll = rng.random()
    if roll < 0.25:
        op = rng.choice(["NEG", "CLIP"])
        return FormulaNode(op, children=(random_formula_node(rng, max_depth - 1),))
    op = rng.choice(["ADD", "ADD", "SUB", "MUL", "DIV_SAFE"])
    left = random_formula_node(rng, max_depth - 1)
    right = random_formula_node(rng, max_depth - 1)
    if op in {"ADD", "SUB"} and rng.random() < 0.55:
        right = FormulaNode("SCALE", rng.choice([0.20, 0.35, 0.50, 0.70]), (right,))
    return FormulaNode(op, children=(left, right))


def formula_allowed(node: FormulaNode) -> bool:
    expr = node.expr()
    banned = ["DELTA1", "fwd_", "future", "label_", "1h"]
    if any(item in expr for item in banned):
        return False
    primitives = node.primitive_set()
    if not primitives:
        return False
    if len(primitives) > 6:
        return False
    return True


def evaluate_node(node: FormulaNode, panel: pd.DataFrame) -> pd.Series:
    if node.op == "FEATURE":
        return panel[feature_column(str(node.value))].astype(float)
    if node.op == "CONST":
        return pd.Series(float(node.value), index=panel.index)
    if node.op == "SCALE":
        out = float(node.value) * evaluate_node(node.children[0], panel)
    elif node.op == "NEG":
        out = -evaluate_node(node.children[0], panel)
    elif node.op == "CLIP":
        out = evaluate_node(node.children[0], panel).clip(-3.0, 3.0)
    elif node.op == "ADD":
        out = evaluate_node(node.children[0], panel) + evaluate_node(node.children[1], panel)
    elif node.op == "SUB":
        out = evaluate_node(node.children[0], panel) - evaluate_node(node.children[1], panel)
    elif node.op == "MUL":
        out = evaluate_node(node.children[0], panel) * evaluate_node(node.children[1], panel)
    elif node.op == "DIV_SAFE":
        denom = evaluate_node(node.children[1], panel)
        denom = denom.where(denom.abs() > 1e-4, np.nan)
        out = evaluate_node(node.children[0], panel) / denom
    else:
        raise ValueError(f"Unknown formula op: {node.op}")
    return pd.Series(out, index=panel.index).replace([np.inf, -np.inf], np.nan).clip(-50.0, 50.0)


def run_validation_task(task: dict[str, Any]) -> dict[str, Any]:
    if _WORKER_PANEL is None or _WORKER_PF_CFG is None or _WORKER_RISK_CFG is None:
        raise RuntimeError("Worker state is not initialized.")
    formula = _WORKER_FORMULAS[str(task["formula_name"])]
    panel = _WORKER_PANEL
    panel[SCORE_COL] = evaluate_node(formula.node, panel)
    validation_equity, validation_positions, validation_metrics = run_regime_gated_backtest(
        panel=panel,
        start=pd.Timestamp(task["validation_start"]),
        end=pd.Timestamp(task["validation_decision_end"]),
        cfg=_WORKER_PF_CFG,
        risk_cfg=_WORKER_RISK_CFG,
        delay_cfg=DummyDelay(),
        score_col=SCORE_COL,
        selected_score_cols=[formula.name],
        selected_formulas=formula.formula,
        use_regime_gate=False,
    )
    enrich_metrics(validation_equity, validation_positions, validation_metrics, _WORKER_HORIZON, _WORKER_BASELINE)
    fitness = formula_fitness(validation_metrics, formula.complexity)
    row: dict[str, Any] = {
        "fold": int(task["fold"]),
        "formula_name": formula.name,
        "formula": formula.formula,
        "source": formula.source,
        "complexity": formula.complexity,
        "primitive_count": len(formula.primitives),
        "primitives": "|".join(formula.primitives),
        "train_start": pd.Timestamp(task["train_start"]),
        "train_decision_end": pd.Timestamp(task["train_decision_end"]),
        "validation_start": pd.Timestamp(task["validation_start"]),
        "validation_decision_end": pd.Timestamp(task["validation_decision_end"]),
        "validation_fitness": fitness,
    }
    for key, value in validation_metrics.items():
        if is_scalar(value):
            row[f"validation_{key}"] = value
    return row


@dataclass(frozen=True)
class DummyDelay:
    name: str = "tau_0"
    tau_days: float = 0.0


def enrich_metrics(
    equity: pd.DataFrame,
    positions: pd.DataFrame,
    metrics: dict[str, float],
    horizon: int,
    baseline: pd.DataFrame | None,
) -> None:
    add_gross_metrics(equity, metrics)
    add_risk_event_metrics(positions, metrics)
    add_delay_metrics(equity, metrics)
    metrics.update(cost_adjusted_metrics(equity, horizon, [20.0, 30.0]))
    metrics["breakeven_cost_bps"] = breakeven_cost_bps(equity)
    metrics["q25_segment_sharpe"] = q25_segment_sharpe(equity, horizon)
    metrics["regime_stability"] = regime_stability_score(equity)
    metrics["long_short_balance"] = long_short_balance(positions)
    metrics["corr_to_locked_baseline"] = corr_to_baseline(equity, baseline)


def formula_fitness(validation: dict[str, float], complexity: int) -> float:
    periods = float(validation.get("periods", 0.0) or 0.0)
    if periods < 6:
        return -1000.0 + periods

    val_total = float(validation.get("total_return", 0.0) or 0.0)
    val_dd = abs(float(validation.get("max_drawdown", 0.0) or 0.0))

    return (
        100.0 * val_total
        + 2.00 * float(validation.get("sharpe_20bps", 0.0) or 0.0)
        + 1.50 * float(validation.get("sharpe_30bps", 0.0) or 0.0)
        + 0.08 * float(validation.get("breakeven_cost_bps", 0.0) or 0.0)
        - 6.00 * float(validation.get("mean_turnover", 1.0) or 1.0)
        + 1.50 * float(validation.get("q25_segment_sharpe", 0.0) or 0.0)
        + 25.0 * float(validation.get("regime_stability", 0.0) or 0.0)
        + 3.00 * float(validation.get("long_short_balance", 0.0) or 0.0)
        - 1.50 * abs(float(validation.get("corr_to_locked_baseline", 0.0) or 0.0))
        - 40.0 * val_dd
        - 0.10 * float(complexity)
    )


def cost_adjusted_metrics(equity: pd.DataFrame, horizon: int, cost_bps: list[float]) -> dict[str, float]:
    out: dict[str, float] = {}
    if equity.empty:
        for bps in cost_bps:
            out[f"total_return_{int(bps)}bps"] = 0.0
            out[f"sharpe_{int(bps)}bps"] = 0.0
            out[f"max_drawdown_{int(bps)}bps"] = 0.0
        return out
    gross = equity["gross_period_return"].astype(float)
    turnover = equity["turnover"].astype(float)
    periods_per_year = 365.0 * 24.0 / horizon
    for bps in cost_bps:
        returns = gross - turnover * float(bps) / 10_000.0
        curve = (1.0 + returns).cumprod()
        std = returns.std(ddof=0)
        sharpe = 0.0 if std == 0 or not np.isfinite(std) else float(returns.mean() / std * np.sqrt(periods_per_year))
        dd = curve / curve.cummax() - 1.0
        out[f"total_return_{int(bps)}bps"] = float(curve.iloc[-1] - 1.0)
        out[f"sharpe_{int(bps)}bps"] = sharpe
        out[f"max_drawdown_{int(bps)}bps"] = float(dd.min())
    return out


def breakeven_cost_bps(equity: pd.DataFrame) -> float:
    if equity.empty or float(equity["turnover"].sum()) <= 1e-12:
        return 0.0
    gross = equity["gross_period_return"].astype(float)
    turnover = equity["turnover"].astype(float)

    def total_at(bps: float) -> float:
        return float((1.0 + gross - turnover * bps / 10_000.0).prod() - 1.0)

    if total_at(0.0) <= 0.0:
        return 0.0
    high = 100.0
    if total_at(high) > 0.0:
        return high
    low = 0.0
    for _ in range(28):
        mid = (low + high) / 2.0
        if total_at(mid) > 0.0:
            low = mid
        else:
            high = mid
    return float(low)


def q25_segment_sharpe(equity: pd.DataFrame, horizon: int) -> float:
    if equity.empty:
        return 0.0
    periods_per_year = 365.0 * 24.0 / horizon
    frame = equity[["timestamp", "period_return"]].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    values = []
    for _, group in frame.groupby(pd.Grouper(key="timestamp", freq="30D")):
        returns = group["period_return"].astype(float)
        if len(returns) < 3:
            continue
        std = returns.std(ddof=0)
        if std > 0 and np.isfinite(std):
            values.append(float(returns.mean() / std * np.sqrt(periods_per_year)))
    if not values:
        return 0.0
    return float(np.nanquantile(values, 0.25))


def regime_stability_score(equity: pd.DataFrame) -> float:
    if equity.empty:
        return 0.0
    frame = equity.copy()
    if _WORKER_PANEL is not None:
        regime = (
            _WORKER_PANEL[["timestamp", "regime_crash_7d", "regime_down_30d", "regime_high_vol"]]
            .drop_duplicates("timestamp")
            .set_index("timestamp")
        )
        frame = frame.join(regime, on="timestamp")
    penalties = []
    for col in ["regime_crash_7d", "regime_down_30d", "regime_high_vol"]:
        if col not in frame:
            continue
        subset = frame[frame[col].fillna(0.0) > 0.5]
        if len(subset) < 4:
            continue
        total = float((1.0 + subset["period_return"].astype(float)).prod() - 1.0)
        penalties.append(total)
    if not penalties:
        return 0.0
    return float(min(penalties))


def long_short_balance(positions: pd.DataFrame) -> float:
    if positions.empty or "contribution" not in positions:
        return 0.0
    long_contrib = float(positions.loc[positions["weight"] > 0, "contribution"].sum())
    short_contrib = float(positions.loc[positions["weight"] < 0, "contribution"].sum())
    denom = abs(long_contrib) + abs(short_contrib)
    if denom <= 1e-12:
        return 0.0
    return float(1.0 - abs(abs(long_contrib) - abs(short_contrib)) / denom)


def corr_to_baseline(equity: pd.DataFrame, baseline: pd.DataFrame | None) -> float:
    if equity.empty or baseline is None or baseline.empty:
        return 0.0
    merged = baseline[["timestamp", "period_return"]].rename(columns={"period_return": "baseline_return"}).merge(
        equity[["timestamp", "period_return"]].rename(columns={"period_return": "candidate_return"}),
        on="timestamp",
        how="inner",
    )
    if len(merged) < 5:
        return 0.0
    corr = merged["baseline_return"].corr(merged["candidate_return"])
    return float(corr) if np.isfinite(corr) else 0.0


def select_formulas_by_fold(
    validation: pd.DataFrame,
    selections: pd.DataFrame,
    formulas: list[FormulaSpec],
    horizon: int,
    committee_size: int,
) -> pd.DataFrame:
    formula_by_name = {item.name: item for item in formulas}
    rows = []
    for _, fold in selections.sort_values("fold").iterrows():
        fold_id = int(fold["fold"])
        scored = validation[validation["fold"] == fold_id].sort_values(
            ["validation_fitness", "validation_total_return", "validation_sharpe"],
            ascending=[False, False, False],
        )
        if scored.empty:
            continue
        best = scored.iloc[0].to_dict()
        committee = select_low_overlap_committee(scored, formula_by_name, committee_size)
        rows.append(
            {
                "fold": fold_id,
                "train_start": fold["train_start"],
                "train_end": fold["train_end"],
                "test_start": fold["test_start"],
                "test_end": fold["test_end"],
                "test_decision_end": decision_end_for_closed_positions(fold["test_end"], horizon),
                "selected_single": best["formula_name"],
                "selected_single_formula": best["formula"],
                "selected_single_fitness": best["validation_fitness"],
                "selected_single_validation_return": best["validation_total_return"],
                "selected_single_validation_sharpe": best["validation_sharpe"],
                "selected_committee": "|".join(item["formula_name"] for item in committee),
                "selected_committee_formulas": " | ".join(item["formula"] for item in committee),
                "selected_committee_mean_fitness": float(np.mean([item["validation_fitness"] for item in committee])),
                "selected_committee_mean_validation_return": float(np.mean([item["validation_total_return"] for item in committee])),
                "selected_committee_mean_validation_sharpe": float(np.mean([item["validation_sharpe"] for item in committee])),
            }
        )
    return pd.DataFrame(rows)


def select_low_overlap_committee(
    scored: pd.DataFrame,
    formula_by_name: dict[str, FormulaSpec],
    max_size: int,
    max_jaccard: float = 0.70,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for _, row in scored.iterrows():
        item = row.to_dict()
        candidate = formula_by_name[str(item["formula_name"])]
        if all(primitive_jaccard(candidate, formula_by_name[str(prev["formula_name"])]) <= max_jaccard for prev in selected):
            selected.append(item)
        if len(selected) >= max_size:
            return selected
    selected_names = {str(item["formula_name"]) for item in selected}
    for _, row in scored.iterrows():
        if str(row["formula_name"]) in selected_names:
            continue
        selected.append(row.to_dict())
        if len(selected) >= max_size:
            break
    return selected


def primitive_jaccard(left: FormulaSpec, right: FormulaSpec) -> float:
    a = set(left.primitives)
    b = set(right.primitives)
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def replay_selected_tests(
    panel: pd.DataFrame,
    selections: pd.DataFrame,
    selected: pd.DataFrame,
    formulas: dict[str, FormulaSpec],
    pf_cfg: Any,
    risk_cfg: Any,
    mode: str,
) -> dict[str, Any]:
    fold_rows: list[dict[str, Any]] = []
    equity_parts: list[pd.DataFrame] = []
    position_parts: list[pd.DataFrame] = []
    selected_by_fold = selected.set_index("fold")
    for _, fold in selections.sort_values("fold").iterrows():
        fold_id = int(fold["fold"])
        if fold_id not in selected_by_fold.index:
            continue
        row = selected_by_fold.loc[fold_id]
        if mode == "single":
            formula_names = [str(row["selected_single"])]
            selected_formula_text = str(row["selected_single_formula"])
            score_col = SCORE_COL
            panel[score_col] = evaluate_node(formulas[formula_names[0]].node, panel)
            validation_fitness = float(row["selected_single_fitness"])
        elif mode == "committee_top3":
            formula_names = [part.strip() for part in str(row["selected_committee"]).split("|") if part.strip()]
            selected_formula_text = str(row["selected_committee_formulas"])
            temp_cols = []
            for idx, formula_name in enumerate(formula_names, start=1):
                col = f"{COMMITTEE_COL}_{idx}"
                panel[col] = evaluate_node(formulas[formula_name].node, panel)
                temp_cols.append(col)
            panel[COMMITTEE_COL] = committee_rank_score(panel, temp_cols)
            score_col = COMMITTEE_COL
            validation_fitness = float(row["selected_committee_mean_fitness"])
        else:
            raise ValueError(f"Unknown replay mode: {mode}")

        equity, positions, metrics = run_regime_gated_backtest(
            panel=panel,
            start=fold["test_start"],
            end=decision_end_for_closed_positions(fold["test_end"], pf_cfg.primary_horizon),
            cfg=pf_cfg,
            risk_cfg=risk_cfg,
            delay_cfg=DummyDelay(),
            score_col=score_col,
            selected_score_cols=formula_names,
            selected_formulas=selected_formula_text,
            use_regime_gate=False,
        )
        drop_cols = [score_col]
        if mode == "committee_top3":
            drop_cols.extend([f"{COMMITTEE_COL}_{idx}" for idx in range(1, len(formula_names) + 1)])
        panel.drop(columns=[col for col in drop_cols if col in panel], inplace=True)
        enrich_metrics(equity, positions, metrics, pf_cfg.rebalance_hours, load_reference_equity(pf_cfg.rebalance_hours))
        fold_rows.append(
            {
                "fold": fold_id,
                "train_start": fold["train_start"],
                "train_end": fold["train_end"],
                "test_start": fold["test_start"],
                "test_end": fold["test_end"],
                "test_decision_end": decision_end_for_closed_positions(fold["test_end"], pf_cfg.primary_horizon),
                "mode": mode,
                "selected_formulas": selected_formula_text,
                "selected_formula_names": "|".join(formula_names),
                "validation_fitness": validation_fitness,
                **metrics,
            }
        )
        if not equity.empty:
            equity = equity.copy()
            equity["fold"] = fold_id
            equity["mode"] = mode
            equity["selected_formula_names"] = "|".join(formula_names)
            equity["selected_formulas"] = selected_formula_text
            positions = positions.copy()
            positions["fold"] = fold_id
            positions["mode"] = mode
            positions["selected_formula_names"] = "|".join(formula_names)
            positions["selected_formulas"] = selected_formula_text
            equity_parts.append(equity)
            position_parts.append(positions)

    equity_curve = pd.concat(equity_parts, ignore_index=True) if equity_parts else pd.DataFrame()
    positions = pd.concat(position_parts, ignore_index=True) if position_parts else pd.DataFrame()
    folds = pd.DataFrame(fold_rows)
    metrics = summarize_nested_result({"equity": equity_curve, "positions": positions, "folds": folds}, pf_cfg.rebalance_hours)
    return {"folds": folds, "equity": equity_curve, "positions": positions, "metrics": metrics}


def committee_rank_score(panel: pd.DataFrame, score_cols: list[str]) -> pd.Series:
    grouped = panel.groupby("timestamp", sort=False)
    ranks = [grouped[col].rank(pct=True, method="average") for col in score_cols]
    return pd.concat(ranks, axis=1).mean(axis=1)


def write_run_result(output_dir: Path, result: dict[str, Any]) -> None:
    result["folds"].to_csv(output_dir / "walk_forward_folds.csv", index=False)
    result["equity"].to_csv(output_dir / "equity_curve.csv", index=False)
    result["positions"].to_csv(output_dir / "positions.csv", index=False)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(to_jsonable(result["metrics"]), fh, indent=2, ensure_ascii=False)


def add_correlations_to_reference(summary: pd.DataFrame, result_index: dict[str, dict[str, Any]], horizon: int) -> pd.DataFrame:
    rows = []
    locked = load_reference_equity(horizon)
    p4 = load_p4_equity()
    for run, result in result_index.items():
        equity = result["equity"].sort_values("timestamp")
        rows.append(
            {
                "run": run,
                "net_corr_to_locked_baseline": corr_to_baseline(equity, locked),
                "net_corr_to_p4_nested": corr_to_baseline(equity, p4),
            }
        )
    return summary.merge(pd.DataFrame(rows), on="run", how="left")


def load_reference_equity(horizon: int) -> pd.DataFrame:
    path = EXPERIMENT_ROOT / "baseline_lock_review" / "horizon" / f"h{horizon:03d}_tau0" / "equity_curve.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["timestamp"]).sort_values("timestamp")


def load_p4_equity() -> pd.DataFrame:
    path = EXPERIMENT_ROOT / "outputs" / "p4_nested_walkforward_selection" / "nested_test_run" / "equity_curve.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["timestamp"]).sort_values("timestamp")


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


def build_selected_ic_summary(
    panel: pd.DataFrame,
    selected: pd.DataFrame,
    formulas: dict[str, FormulaSpec],
    horizon: int,
) -> pd.DataFrame:
    label_cols = [col for col in ["label_path_slow_low_turnover", "label_slow_low_turnover"] if col in panel]
    if not label_cols:
        return pd.DataFrame()
    rows = []
    for mode in ["single", "committee_top3"]:
        ts_ics: dict[str, list[float]] = {label: [] for label in label_cols}
        for _, row in selected.iterrows():
            if mode == "single":
                formula_names = [str(row["selected_single"])]
            else:
                formula_names = [part.strip() for part in str(row["selected_committee"]).split("|") if part.strip()]
            score = selected_score_for_ic(panel, [formulas[name] for name in formula_names])
            mask = (panel["timestamp"] >= row["test_start"]) & (
                panel["timestamp"] < decision_end_for_closed_positions(row["test_end"], horizon)
            )
            tmp = panel.loc[mask, ["timestamp", *label_cols]].copy()
            tmp["_score"] = score.loc[tmp.index]
            for label_col in label_cols:
                per_ts = tmp[["timestamp", "_score", label_col]].replace([np.inf, -np.inf], np.nan).dropna()
                grouped = per_ts.groupby("timestamp", sort=False)
                for _, g in grouped:
                    if len(g) >= 8:
                        corr = g["_score"].rank().corr(g[label_col].rank())
                        if np.isfinite(corr):
                            ts_ics[label_col].append(float(corr))
        for label_col, values in ts_ics.items():
            rows.append(
                {
                    "mode": mode,
                    "label": label_col,
                    "mean_ic": float(np.mean(values)) if values else np.nan,
                    "median_ic": float(np.median(values)) if values else np.nan,
                    "icir": float(np.mean(values) / np.std(values, ddof=0)) if values and np.std(values, ddof=0) > 0 else np.nan,
                    "positive_ts_ic_rate": float(np.mean(np.array(values) > 0.0)) if values else np.nan,
                    "timestamp_count": len(values),
                }
            )
    return pd.DataFrame(rows)


def selected_score_for_ic(panel: pd.DataFrame, formula_specs: list[FormulaSpec]) -> pd.Series:
    if len(formula_specs) == 1:
        return evaluate_node(formula_specs[0].node, panel)
    temp = []
    grouped = panel.groupby("timestamp", sort=False)
    for spec in formula_specs:
        raw = evaluate_node(spec.node, panel)
        temp.append(raw.groupby(panel["timestamp"], sort=False).rank(pct=True, method="average"))
    return pd.concat(temp, axis=1).mean(axis=1)


def plot_nested_curves(output_dir: Path, result_index: dict[str, dict[str, Any]], horizon: int) -> None:
    p4 = load_p4_equity()
    fig, axes = plt.subplots(2, 1, figsize=(13, 8.5), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0]})
    for name, result in result_index.items():
        equity = result["equity"].sort_values("timestamp")
        if equity.empty:
            continue
        net = (1.0 + equity["period_return"]).cumprod() - 1.0
        gross = (1.0 + equity["gross_period_return"]).cumprod() - 1.0
        axes[0].plot(equity["timestamp"], net, label=f"{name} net", linewidth=2.0)
        axes[0].plot(equity["timestamp"], gross, label=f"{name} gross", linewidth=1.5, alpha=0.65)
        curve = (1.0 + equity["period_return"]).cumprod()
        dd = curve / curve.cummax() - 1.0
        axes[1].plot(equity["timestamp"], dd, label=name, linewidth=1.8)
    if not p4.empty:
        p4_curve = (1.0 + p4["period_return"]).cumprod() - 1.0
        axes[0].plot(p4["timestamp"], p4_curve, label="P4 nested net", linewidth=1.6, alpha=0.75, color="#444444")
    axes[0].axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    axes[0].set_title(f"P5 AlphaGPT slow search nested test curves ({horizon}h)")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    axes[1].set_title("Net drawdown")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "nested_alphagpt_slow_curves.png", dpi=180)
    plt.close(fig)


def plot_cost_sensitivity(output_dir: Path, cost_summary: pd.DataFrame) -> None:
    if cost_summary.empty:
        return
    plt.figure(figsize=(12, 6.5))
    for name, group in cost_summary.groupby("experiment"):
        group = group.sort_values("one_way_cost_bps")
        plt.plot(group["one_way_cost_bps"], group["total_return"], marker="o", label=name)
    plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    plt.title("P5 AlphaGPT slow search cost sensitivity")
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
    singles = selected["selected_single"].value_counts().head(18).sort_values()
    plt.figure(figsize=(11, 7))
    plt.barh(singles.index, singles.values, color="#2a9d8f")
    plt.title("P5 selected single formula counts")
    plt.xlabel("Fold count")
    plt.tight_layout()
    plt.savefig(output_dir / "selected_single_formula_counts.png", dpi=180)
    plt.close()

    committee_items = []
    for value in selected["selected_committee"].dropna():
        committee_items.extend(part.strip() for part in str(value).split("|") if part.strip())
    if committee_items:
        counts = pd.Series(committee_items).value_counts().head(18).sort_values()
        plt.figure(figsize=(11, 7))
        plt.barh(counts.index, counts.values, color="#457b9d")
        plt.title("P5 selected committee formula counts")
        plt.xlabel("Fold count")
        plt.tight_layout()
        plt.savefig(output_dir / "selected_committee_formula_counts.png", dpi=180)
        plt.close()


def plot_validation_vs_test(
    output_dir: Path,
    validation: pd.DataFrame,
    single_folds: pd.DataFrame,
    committee_folds: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    if not single_folds.empty:
        axes[0].scatter(single_folds["validation_fitness"], single_folds["total_return"], alpha=0.8, label="single")
    if not committee_folds.empty:
        axes[0].scatter(committee_folds["validation_fitness"], committee_folds["total_return"], alpha=0.8, label="committee")
    axes[0].axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    axes[0].set_title("Selected validation fitness vs test fold return")
    axes[0].set_xlabel("validation fitness")
    axes[0].set_ylabel("test fold return")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()
    top = validation.sort_values("validation_fitness", ascending=False).head(40).sort_values("validation_fitness")
    axes[1].barh(top["formula_name"], top["validation_fitness"], color="#2a9d8f")
    axes[1].set_title("Top validation formula observations")
    axes[1].set_xlabel("fitness")
    fig.tight_layout()
    fig.savefig(output_dir / "validation_vs_test.png", dpi=180)
    plt.close(fig)


def plot_asset_contribution(output_dir: Path, asset_summary: pd.DataFrame, best_name: str) -> None:
    if asset_summary.empty:
        return
    frame = asset_summary.sort_values("gross_contribution_sum")
    plt.figure(figsize=(11, 7))
    colors = np.where(frame["gross_contribution_sum"] >= 0.0, "#2a9d8f", "#c44536")
    plt.barh(frame["symbol"], frame["gross_contribution_sum"], color=colors)
    plt.axvline(0.0, color="black", linewidth=0.8, alpha=0.6)
    plt.title(f"{best_name} gross contribution by asset")
    plt.xlabel("Gross contribution sum")
    plt.gca().xaxis.set_major_formatter(PercentFormatter(1.0))
    plt.tight_layout()
    plt.savefig(output_dir / "best_asset_contribution.png", dpi=180)
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
    ic_summary: pd.DataFrame,
    best_name: str,
    formula_count: int,
    workers: int,
    horizon: int,
    inner_validation_days: int,
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
        "net_corr_to_locked_baseline",
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
        "selected_single",
        "selected_single_fitness",
        "selected_single_validation_return",
        "selected_committee",
        "selected_committee_mean_fitness",
        "selected_committee_mean_validation_return",
    ]
    selected_cols = [col for col in selected_cols if col in selected]
    cost_display = cost_summary.copy()
    if not cost_display.empty:
        cost_display = cost_display[cost_display["one_way_cost_bps"].isin([0, 6, 10, 15, 20, 30])]

    lines = [
        "# P5 AlphaGPT Slow Search",
        "",
        "## Design",
        "",
        f"- Horizon fixed at `{horizon}h`.",
        f"- Candidate formulas: `{formula_count}` generated from the constrained slow residual grammar.",
        f"- Each outer fold scores formulas only on the last `{inner_validation_days}` days inside its prior training window.",
        "- Earlier prior data is used only as trailing context for slow features; no test data enters formula selection.",
        "- Test windows are clipped by `horizon + execution_delay` so every evaluated position exits inside its own window.",
        "- Search variants: best single formula and top-3 low-primitive-overlap committee.",
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
        "## Selected Formulas By Fold",
        "",
        "```text",
        selected[selected_cols].to_string(index=False),
        "```",
        "",
        "## Best Result Diagnostics",
        "",
        f"- best nested variant: `{best_name}`",
        "",
        "### Cost Sensitivity",
        "",
        "```text",
        cost_display.to_string(index=False),
        "```",
        "",
        "### Regime Summary",
        "",
        "```text",
        regime_summary.to_string(index=False),
        "```",
        "",
        "### Long/Short Contribution",
        "",
        "```text",
        side_summary.to_string(index=False),
        "```",
        "",
        "### Asset Contribution",
        "",
        "```text",
        asset_summary.to_string(index=False),
        "```",
        "",
        "### Selected Formula IC",
        "",
        "```text",
        ic_summary.to_string(index=False) if not ic_summary.empty else "No label columns available.",
        "```",
        "",
        "## Files",
        "",
        "- `candidate_formulas.csv`: generated formula universe.",
        "- `validation_formula_metrics.csv`: train/validation metrics for every fold/formula pair.",
        "- `selected_formulas_by_fold.csv`: selected single and committee formulas per fold.",
        "- `nested_single_run/`, `nested_committee_top3_run/`: stitched test-only results.",
        "- `summary.csv`, `baseline_comparison.csv`, `cost_sensitivity.csv`: headline tables.",
        "- `nested_alphagpt_slow_curves.png`: net/gross/drawdown curves.",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def update_config_stage(config_path: Path) -> None:
    text = config_path.read_text(encoding="utf-8")
    if "p5_alphagpt_slow_search:" in text:
        return
    marker = "committee_candidates:\n"
    insert = """\n  p5_alphagpt_slow_search:\n    script: ../../../scripts/run_p5_alphagpt_slow_search.py\n    output_dir: ../outputs/p5_alphagpt_slow_search\n    report: ../outputs/p5_alphagpt_slow_search/REPORT.md\n    candidate_formulas: ../outputs/p5_alphagpt_slow_search/candidate_formulas.csv\n    validation_formula_metrics: ../outputs/p5_alphagpt_slow_search/validation_formula_metrics.csv\n    selected_formulas_by_fold: ../outputs/p5_alphagpt_slow_search/selected_formulas_by_fold.csv\n    summary: ../outputs/p5_alphagpt_slow_search/summary.csv\n    baseline_comparison: ../outputs/p5_alphagpt_slow_search/baseline_comparison.csv\n\n"""
    if marker not in text:
        return
    text = text.replace(marker, insert + marker, 1)
    config_path.write_text(text, encoding="utf-8")


def is_scalar(value: Any) -> bool:
    return isinstance(value, (int, float, str, bool, np.integer, np.floating)) or value is None


if __name__ == "__main__":
    main()
