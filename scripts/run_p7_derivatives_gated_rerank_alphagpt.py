#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
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
from run_p2_family_b_mean_reversion import build_baseline_comparison
from run_p5_alphagpt_slow_search import FormulaNode, FormulaSpec
from run_turnover_experiments import to_jsonable


EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "baseline_locked_v1"
DEFAULT_CONFIG = EXPERIMENT_ROOT / "configs" / "baseline_locked_v1.yaml"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "outputs" / "p7_derivatives_gated_rerank_alphagpt"
P6B_PANEL = EXPERIMENT_ROOT / "outputs" / "p6b_derivatives_feature_expansion" / "p6b_feature_panel.csv.gz"
P6B_RETAINED = EXPERIMENT_ROOT / "outputs" / "p6b_derivatives_feature_expansion" / "retained_features.csv"
P3D_SUMMARY = EXPERIMENT_ROOT / "outputs" / "p3d_p4_gated_rerank_alphagpt" / "summary.csv"
P3D_EQUITY = EXPERIMENT_ROOT / "outputs" / "p3d_p4_gated_rerank_alphagpt" / "nested_incremental_run" / "equity_curve.csv"

P7_GRID = [
    {"candidate_weight_long": 0.10, "candidate_weight_short": 0.10, "high_risk_candidate_cap": 0.05},
    {"candidate_weight_long": 0.20, "candidate_weight_short": 0.20, "high_risk_candidate_cap": 0.10},
    {"candidate_weight_long": 0.30, "candidate_weight_short": 0.30, "high_risk_candidate_cap": 0.15},
    {"candidate_weight_long": 0.40, "candidate_weight_short": 0.40, "high_risk_candidate_cap": 0.20},
]

_P7_FEATURE_MAP: dict[str, str] = {}


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

    patch_p3d()
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    output_dir = ensure_dir(args.output_dir)
    horizon = int(args.horizon)
    workers = max(1, int(args.workers))

    panel_path = p3d.resolve_config_path(config_path, cfg["inputs"]["panel"])
    selections_path = p3d.resolve_config_path(config_path, cfg["inputs"]["formula_selections"])
    selections = pd.read_csv(selections_path, parse_dates=["train_start", "train_end", "test_start", "test_end"])
    p4_selected = pd.read_csv(p3d.P4_SELECTED, parse_dates=["train_start", "train_end", "test_start", "test_end"])
    retained = load_retained_features()
    formulas = build_p7_formula_universe(retained, int(args.formulas), int(args.seed), int(args.max_depth))
    pd.DataFrame([item.to_row() for item in formulas]).to_csv(output_dir / "candidate_formulas.csv", index=False)

    tasks = build_validation_tasks(
        selections=selections,
        p4_selected=p4_selected,
        formulas=formulas,
        horizon=horizon,
        validation_days=int(args.inner_validation_days),
    )
    print(
        f"[1/9] P7 derivatives gated rerank search: {len(selections)} folds x {len(formulas)} formulas x "
        f"{len(P7_GRID)} rerank grids = {len(tasks)} validation tasks; workers={workers}",
        flush=True,
    )
    worker_args = (
        str(config_path),
        str(panel_path),
        str(p3d.MANUAL_FEATURE_PANEL),
        horizon,
        {item.name: item for item in formulas},
    )
    validation_rows: list[dict[str, Any]] = []
    if workers == 1:
        init_worker_p7(*worker_args)
        for idx, task in enumerate(tasks, start=1):
            if idx == 1 or idx % 250 == 0 or idx == len(tasks):
                print(f"      validation task {idx}/{len(tasks)}", flush=True)
            validation_rows.append(p3d.run_validation_task(task))
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers, initializer=init_worker_p7, initargs=worker_args) as pool:
            for idx, row in enumerate(pool.imap_unordered(p3d.run_validation_task, tasks, chunksize=max(1, int(args.chunksize))), start=1):
                validation_rows.append(row)
                if idx == 1 or idx % 250 == 0 or idx == len(tasks):
                    print(f"      validation task {idx}/{len(tasks)}", flush=True)
    validation = pd.DataFrame(validation_rows)

    print("[2/9] Running P4 validation baselines for hard-accept comparisons", flush=True)
    p4_validation = p3d.run_p4_validation_baselines(
        config_path=config_path,
        panel_path=panel_path,
        feature_path=p3d.MANUAL_FEATURE_PANEL,
        selections=selections,
        p4_selected=p4_selected,
        horizon=horizon,
    )
    p4_validation.to_csv(output_dir / "validation_p4_metrics.csv", index=False)
    validation = p3d.annotate_hard_accept(validation, p4_validation)
    validation.to_csv(output_dir / "validation_incremental_metrics.csv", index=False)

    print("[3/9] Selecting best accepted candidate per fold, otherwise falling back to P4", flush=True)
    selected = p3d.select_by_fold(validation, p4_validation, selections, p4_selected, horizon)
    selected.to_csv(output_dir / "selected_incremental_by_fold.csv", index=False)

    print("[4/9] Replaying selected P7 formulas on test windows only", flush=True)
    panel = load_panel_p7(config_path, panel_path, p3d.MANUAL_FEATURE_PANEL, horizon)
    pf_cfg = p3d.make_portfolio_config(cfg, horizon)
    risk_template = p3d.make_risk_template(cfg)
    spec_by_name = {spec.name: spec for spec in p3d.build_committee_specs()}
    formula_by_name = {item.name: item for item in formulas}
    result = p3d.replay_selected_tests(panel, selections, selected, pf_cfg, risk_template, spec_by_name, formula_by_name)
    p3d.write_run_result(ensure_dir(output_dir / "nested_incremental_run"), result)

    print("[5/9] Summarizing and comparing against P3d/P4", flush=True)
    metrics = p3d.summarize_nested_result(result, horizon)
    summary = pd.DataFrame([{**metrics, "run": "p7_derivatives_gated_rerank_alphagpt", "horizon": horizon}])
    summary["net_corr_to_p3d"] = corr_to_reference(result["equity"], load_reference_equity(P3D_EQUITY))
    summary["net_corr_to_p4_nested"] = corr_to_reference(result["equity"], p3d.load_p4_equity())
    summary.to_csv(output_dir / "summary.csv", index=False)
    baseline = p3d.load_baseline_summary(EXPERIMENT_ROOT / "baseline_lock_review" / "horizon_summary.csv")
    comparison = build_baseline_comparison(summary.copy(), baseline)
    comparison = p3d.add_p4_comparison(comparison)
    comparison = add_p3d_comparison(comparison)
    comparison.to_csv(output_dir / "baseline_comparison.csv", index=False)

    print("[6/9] Cost, regime, side, and asset diagnostics", flush=True)
    cost_summary = p3d.build_cost_sensitivity({"p7_derivatives_gated_rerank_alphagpt": result}, [float(x) for x in args.cost_bps])
    cost_summary = add_p3d_cost_comparison(cost_summary)
    cost_summary.to_csv(output_dir / "cost_sensitivity.csv", index=False)
    equity = result["equity"].sort_values("timestamp")
    positions = result["positions"].sort_values(["timestamp", "symbol"])
    regime_summary = p3d.build_regime_summary(equity, p3d.build_regime_frame(panel), horizon)
    side_summary = p3d.build_side_summary(positions)
    asset_summary = p3d.build_asset_summary(positions)
    regime_summary.to_csv(output_dir / "regime_summary.csv", index=False)
    side_summary.to_csv(output_dir / "side_contribution.csv", index=False)
    asset_summary.to_csv(output_dir / "asset_contribution.csv", index=False)
    primitive_usage = build_primitive_usage(selected, formulas, retained)
    primitive_usage.to_csv(output_dir / "primitive_usage.csv", index=False)

    print("[7/9] Writing plots", flush=True)
    plot_curves(output_dir, result, horizon)
    p3d.plot_cost_sensitivity(output_dir, cost_summary)
    p3d.plot_selected_counts(output_dir, selected)
    p3d.plot_validation_vs_test(output_dir, result["folds"])
    p3d.plot_asset_contribution(output_dir, asset_summary)
    plot_primitive_usage(output_dir, primitive_usage)

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
        primitive_usage=primitive_usage,
        retained=retained,
        formula_count=len(formulas),
        workers=workers,
        horizon=horizon,
        validation_days=int(args.inner_validation_days),
    )
    update_config_stage(config_path)

    print("[9/9] Done", flush=True)
    print(f"Done: {output_dir}", flush=True)


def patch_p3d() -> None:
    p3d.load_panel = load_panel_p7
    p3d.evaluate_node = evaluate_node_p7


def init_worker_p7(
    config_path_str: str,
    panel_path_str: str,
    feature_path_str: str,
    horizon: int,
    formulas: dict[str, FormulaSpec],
) -> None:
    patch_p3d()
    p3d.init_worker(config_path_str, panel_path_str, feature_path_str, horizon, formulas)


def load_retained_features() -> pd.DataFrame:
    retained = pd.read_csv(P6B_RETAINED)
    retained = retained[retained["retained"].astype(bool)].copy()
    retained = retained.sort_values(["diagnostic_score", "primary_mean_ic"], ascending=[False, False]).reset_index(drop=True)
    return retained


def expr_for(name: str) -> str:
    return f"P6B({name})"


def ema_expr_for(name: str, span: int) -> str:
    return f"EMA_SLOW(P6B({name}),{span})"


def build_p7_feature_map(retained: pd.DataFrame | None = None) -> dict[str, str]:
    retained = retained if retained is not None else load_retained_features()
    mapping: dict[str, str] = {}
    for row in retained.itertuples(index=False):
        score_col = str(row.score_col)
        name = str(row.name)
        mapping[expr_for(name)] = f"p7_rank_{score_col}"
        for span in [168, 336]:
            mapping[ema_expr_for(name, span)] = f"p7_ema{span}_{score_col}"
    return mapping


def load_panel_p7(config_path: Path, panel_path: Path, feature_path: Path, horizon: int) -> pd.DataFrame:
    global _P7_FEATURE_MAP
    retained = load_retained_features()
    panel = p3d.load_panel.__wrapped__(config_path, panel_path, feature_path, horizon) if hasattr(p3d.load_panel, "__wrapped__") else None
    if panel is None:
        # p3d.load_panel is patched in this script, so call the original function through a saved alias.
        panel = ORIGINAL_P3D_LOAD_PANEL(config_path, panel_path, feature_path, horizon)
    usecols = ["timestamp", "symbol", *retained["score_col"].astype(str).tolist()]
    p6b = pd.read_csv(P6B_PANEL, parse_dates=["timestamp"], usecols=usecols)
    panel = panel.merge(p6b, on=["timestamp", "symbol"], how="left")
    grouped = panel.groupby("symbol", group_keys=False)
    for row in retained.itertuples(index=False):
        score_col = str(row.score_col)
        rank_col = f"p7_rank_{score_col}"
        panel[rank_col] = p3d.centered_rank_by_timestamp(panel, score_col)
        for span in [168, 336]:
            panel[f"p7_ema{span}_{score_col}"] = grouped[rank_col].transform(
                lambda s, sp=span: s.ewm(span=sp, adjust=False, min_periods=max(48, sp // 3)).mean()
            )
    _P7_FEATURE_MAP = build_p7_feature_map(retained)
    return panel.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


ORIGINAL_P3D_LOAD_PANEL = p3d.load_panel


def evaluate_node_p7(node: FormulaNode, panel: pd.DataFrame) -> pd.Series:
    if node.op == "FEATURE":
        expr = str(node.value)
        col = feature_column_p7(expr)
        return panel[col].astype(float)
    if node.op == "CONST":
        return pd.Series(float(node.value), index=panel.index)
    if node.op == "SCALE":
        out = float(node.value) * evaluate_node_p7(node.children[0], panel)
    elif node.op == "NEG":
        out = -evaluate_node_p7(node.children[0], panel)
    elif node.op == "CLIP":
        out = evaluate_node_p7(node.children[0], panel).clip(-3.0, 3.0)
    elif node.op == "ADD":
        out = evaluate_node_p7(node.children[0], panel) + evaluate_node_p7(node.children[1], panel)
    elif node.op == "SUB":
        out = evaluate_node_p7(node.children[0], panel) - evaluate_node_p7(node.children[1], panel)
    elif node.op == "MUL":
        out = evaluate_node_p7(node.children[0], panel) * evaluate_node_p7(node.children[1], panel)
    elif node.op == "DIV_SAFE":
        denom = evaluate_node_p7(node.children[1], panel)
        denom = denom.where(denom.abs() > 1e-4, np.nan)
        out = evaluate_node_p7(node.children[0], panel) / denom
    else:
        raise ValueError(f"Unknown formula op: {node.op}")
    return pd.Series(out, index=panel.index).replace([np.inf, -np.inf], np.nan).clip(-50.0, 50.0)


def feature_column_p7(expr: str) -> str:
    global _P7_FEATURE_MAP
    if not _P7_FEATURE_MAP:
        _P7_FEATURE_MAP = build_p7_feature_map()
    if expr not in _P7_FEATURE_MAP:
        raise KeyError(f"Unknown P7 primitive expression: {expr}")
    return _P7_FEATURE_MAP[expr]


def build_p7_formula_universe(retained: pd.DataFrame, count: int, seed: int, max_depth: int) -> list[FormulaSpec]:
    rng = random.Random(seed)
    nodes = seed_nodes_p7(retained)
    seen = {node.expr() for node in nodes}
    primitives = primitive_nodes_p7(retained)
    attempts = 0
    target = max(count, min(len(nodes), count))
    while len(nodes) < target and attempts < count * 80:
        attempts += 1
        node = random_formula_node_p7(rng, primitives, max_depth)
        expr = node.expr()
        if expr in seen or not formula_allowed_p7(node):
            continue
        nodes.append(node)
        seen.add(expr)
    nodes = nodes[:count]
    specs: list[FormulaSpec] = []
    seed_count = len(seed_nodes_p7(retained))
    for idx, node in enumerate(nodes, start=1):
        specs.append(
            FormulaSpec(
                name=f"p7_agpt_{idx:04d}",
                node=node,
                formula=node.expr(),
                complexity=node.size(),
                primitives=tuple(sorted(node.primitive_set())),
                source="p6b_retained_seed" if idx <= seed_count else "p6b_retained_random_grammar",
            )
        )
    return specs


def primitive_nodes_p7(retained: pd.DataFrame) -> list[FormulaNode]:
    nodes: list[FormulaNode] = []
    for row in retained.itertuples(index=False):
        name = str(row.name)
        nodes.append(FormulaNode("FEATURE", expr_for(name)))
        nodes.append(FormulaNode("FEATURE", ema_expr_for(name, 168)))
        nodes.append(FormulaNode("FEATURE", ema_expr_for(name, 336)))
    return nodes


def feature(retained: pd.DataFrame, name: str) -> FormulaNode:
    if name not in set(retained["name"].astype(str)):
        raise KeyError(name)
    return FormulaNode("FEATURE", expr_for(name))


def ema_feature(retained: pd.DataFrame, name: str, span: int = 168) -> FormulaNode:
    if name not in set(retained["name"].astype(str)):
        raise KeyError(name)
    return FormulaNode("FEATURE", ema_expr_for(name, span))


def seed_nodes_p7(retained: pd.DataFrame) -> list[FormulaNode]:
    available = set(retained["name"].astype(str))
    f = lambda name: feature(retained, name)
    e = lambda name, span=168: ema_feature(retained, name, span)
    scale = lambda value, node: FormulaNode("SCALE", value, (node,))
    add = lambda left, right: FormulaNode("ADD", children=(left, right))

    nodes: list[FormulaNode] = []
    for name in retained["name"].astype(str):
        nodes.append(f(name))
        nodes.append(e(name, 168))
        nodes.append(e(name, 336))

    def maybe_combo(required: list[str], node: FormulaNode) -> None:
        if all(name in available for name in required):
            nodes.append(node)

    maybe_combo(
        ["K01_deriv_crowding_contra", "J04_top_trader_contra"],
        add(scale(0.55, f("K01_deriv_crowding_contra")), scale(0.45, f("J04_top_trader_contra"))),
    )
    maybe_combo(
        ["K01_deriv_crowding_contra", "H01_funding_contra_168"],
        add(scale(0.55, f("K01_deriv_crowding_contra")), scale(0.45, f("H01_funding_contra_168"))),
    )
    maybe_combo(
        ["K01_deriv_crowding_contra", "E03_basket_diversifier_mom"],
        add(scale(0.60, f("K01_deriv_crowding_contra")), scale(0.40, f("E03_basket_diversifier_mom"))),
    )
    maybe_combo(
        ["K05_deriv_p3d_diversifier", "J04_top_trader_contra"],
        add(scale(0.60, f("K05_deriv_p3d_diversifier")), scale(0.40, f("J04_top_trader_contra"))),
    )
    maybe_combo(
        ["H03_slow_mom_funding_penalty", "I05_mom_basis_quality"],
        add(scale(0.55, f("H03_slow_mom_funding_penalty")), scale(0.45, f("I05_mom_basis_quality"))),
    )
    maybe_combo(
        ["C05_low_capacity_penalty", "K01_deriv_crowding_contra"],
        add(scale(0.50, f("C05_low_capacity_penalty")), scale(0.50, f("K01_deriv_crowding_contra"))),
    )
    return dedupe_nodes(nodes)


def dedupe_nodes(nodes: list[FormulaNode]) -> list[FormulaNode]:
    out = []
    seen = set()
    for node in nodes:
        expr = node.expr()
        if expr in seen:
            continue
        out.append(node)
        seen.add(expr)
    return out


def random_formula_node_p7(rng: random.Random, primitives: list[FormulaNode], max_depth: int) -> FormulaNode:
    if max_depth <= 0 or rng.random() < 0.42:
        node = rng.choice(primitives)
        if rng.random() < 0.25:
            return FormulaNode("SCALE", rng.choice([0.20, 0.35, 0.50, 0.70, 1.20]), (node,))
        return node
    roll = rng.random()
    if roll < 0.18:
        op = rng.choice(["NEG", "CLIP"])
        return FormulaNode(op, children=(random_formula_node_p7(rng, primitives, max_depth - 1),))
    op = rng.choice(["ADD", "ADD", "SUB", "MUL"])
    left = random_formula_node_p7(rng, primitives, max_depth - 1)
    right = random_formula_node_p7(rng, primitives, max_depth - 1)
    if op in {"ADD", "SUB"} and rng.random() < 0.65:
        right = FormulaNode("SCALE", rng.choice([0.20, 0.35, 0.50, 0.70]), (right,))
    return FormulaNode(op, children=(left, right))


def formula_allowed_p7(node: FormulaNode) -> bool:
    if node.size() > 15:
        return False
    primitives = node.primitive_set()
    if not primitives or len(primitives) > 5:
        return False
    expr = node.expr()
    if "DIV_SAFE" in expr or "future" in expr or "label_" in expr:
        return False
    return True


def build_validation_tasks(
    selections: pd.DataFrame,
    p4_selected: pd.DataFrame,
    formulas: list[FormulaSpec],
    horizon: int,
    validation_days: int,
) -> list[dict[str, Any]]:
    p4_by_fold = p4_selected.set_index("fold")
    tasks: list[dict[str, Any]] = []
    for _, fold in selections.sort_values("fold").iterrows():
        fold_id = int(fold["fold"])
        p4_row = p4_by_fold.loc[fold_id]
        train_start = pd.Timestamp(fold["train_start"])
        train_end = pd.Timestamp(fold["train_end"])
        validation_start = max(train_start + (train_end - train_start) / 2, train_end - pd.Timedelta(days=validation_days))
        validation_end = p3d.decision_end_for_closed_positions(train_end, horizon)
        for formula in formulas:
            for grid in P7_GRID:
                tasks.append(
                    {
                        "fold": fold_id,
                        "formula_name": formula.name,
                        "p4_spec": str(p4_row["selected_spec"]),
                        "candidate_weight_long": float(grid["candidate_weight_long"]),
                        "candidate_weight_short": float(grid["candidate_weight_short"]),
                        "high_risk_candidate_cap": float(grid["high_risk_candidate_cap"]),
                        "validation_start": validation_start,
                        "validation_decision_end": validation_end,
                        "train_end": train_end,
                    }
                )
    return tasks


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


def load_reference_equity(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["timestamp"]).sort_values("timestamp")


def add_p3d_comparison(comparison: pd.DataFrame) -> pd.DataFrame:
    if not P3D_SUMMARY.exists():
        return comparison
    p3d_summary = pd.read_csv(P3D_SUMMARY)
    if p3d_summary.empty:
        return comparison
    ref = p3d_summary.iloc[0].to_dict()
    out = comparison.copy()
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
            out[f"p3d_{col}"] = float(ref[col])
            out[f"delta_vs_p3d_{col}"] = out[col] - float(ref[col])
    return out


def add_p3d_cost_comparison(cost_summary: pd.DataFrame) -> pd.DataFrame:
    p3d_cost_path = EXPERIMENT_ROOT / "outputs" / "p3d_p4_gated_rerank_alphagpt" / "cost_sensitivity.csv"
    if not p3d_cost_path.exists() or cost_summary.empty:
        return cost_summary
    ref = pd.read_csv(p3d_cost_path)
    if "one_way_cost_bps" not in ref or "total_return" not in ref:
        return cost_summary
    cols = [col for col in ["one_way_cost_bps", "total_return", "sharpe", "max_drawdown"] if col in ref]
    ref = ref[cols].rename(columns={col: f"p3d_{col}" for col in cols if col != "one_way_cost_bps"})
    out = cost_summary.merge(ref, on="one_way_cost_bps", how="left")
    for col in ["total_return", "sharpe", "max_drawdown"]:
        p3d_col = f"p3d_{col}"
        if col in out and p3d_col in out:
            out[f"delta_vs_p3d_{col}"] = out[col] - out[p3d_col]
    return out


def build_primitive_usage(selected: pd.DataFrame, formulas: list[FormulaSpec], retained: pd.DataFrame) -> pd.DataFrame:
    formula_map = {item.name: item for item in formulas}
    retained_names = retained["name"].astype(str).tolist()
    rows = []
    accepted = selected[~selected["fallback_to_p4"].astype(bool)]
    for name in retained_names:
        raw = expr_for(name)
        ema168 = ema_expr_for(name, 168)
        ema336 = ema_expr_for(name, 336)
        fold_count = 0
        weighted_count = 0.0
        for row in accepted.itertuples(index=False):
            spec = formula_map.get(str(row.formula_name))
            if spec is None:
                continue
            primitives = set(spec.primitives)
            if raw in primitives or ema168 in primitives or ema336 in primitives:
                fold_count += 1
                weighted_count += float(row.candidate_weight_long)
        retained_row = retained[retained["name"] == name].iloc[0]
        rows.append(
            {
                "primitive": name,
                "family": retained_row["family"],
                "score_col": retained_row["score_col"],
                "selected_fold_count": fold_count,
                "selected_weight_sum": weighted_count,
                "p6b_primary_mean_ic": retained_row["primary_mean_ic"],
                "p6b_turnover_72h": retained_row["signal_turnover_72h"],
                "p6b_mean_abs_p3d_corr": retained_row["mean_abs_p3d_weight_corr"],
            }
        )
    return pd.DataFrame(rows).sort_values(["selected_fold_count", "selected_weight_sum"], ascending=[False, False])


def plot_curves(output_dir: Path, result: dict[str, Any], horizon: int) -> None:
    equity = result["equity"].sort_values("timestamp")
    if equity.empty:
        return
    p3d_equity = load_reference_equity(P3D_EQUITY)
    fig, axes = plt.subplots(2, 1, figsize=(13, 8.5), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0]})
    gross = (1.0 + equity["gross_period_return"]).cumprod() - 1.0
    net = (1.0 + equity["period_return"]).cumprod() - 1.0
    axes[0].plot(equity["timestamp"], gross, label="P7 gross before cost", linewidth=2.0)
    axes[0].plot(equity["timestamp"], net, label="P7 net after 6bps", linewidth=2.0)
    if not p3d_equity.empty:
        p3d_net = (1.0 + p3d_equity["period_return"]).cumprod() - 1.0
        axes[0].plot(p3d_equity["timestamp"], p3d_net, label="P3d active baseline net", linewidth=1.7, alpha=0.8)
    axes[0].axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    axes[0].set_title(f"P7 derivatives P4-gated AlphaGPT nested test ({horizon}h)")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)
    curve = (1.0 + equity["period_return"]).cumprod()
    dd = curve / curve.cummax() - 1.0
    axes[1].plot(equity["timestamp"], dd, label="P7 drawdown", linewidth=1.8)
    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "nested_incremental_curves.png", dpi=180)
    plt.close(fig)


def plot_primitive_usage(output_dir: Path, usage: pd.DataFrame) -> None:
    if usage.empty:
        return
    top = usage.sort_values("selected_fold_count").tail(15)
    plt.figure(figsize=(11, 7))
    plt.barh(top["primitive"], top["selected_fold_count"], color="#386641")
    plt.title("P7 selected primitive usage")
    plt.xlabel("Accepted fold count")
    plt.tight_layout()
    plt.savefig(output_dir / "primitive_usage.png", dpi=180)
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
    retained: pd.DataFrame,
    formula_count: int,
    workers: int,
    horizon: int,
    validation_days: int,
) -> None:
    selected_cols = [
        "fold",
        "p4_spec",
        "formula_name",
        "candidate_weight_long",
        "high_risk_candidate_cap",
        "hard_accept",
        "fallback_to_p4",
        "validation_fitness",
        "validation_total_return",
        "validation_sharpe",
        "validation_short_abs_contribution_share",
        "formula",
    ]
    selected_cols = [col for col in selected_cols if col in selected]
    top_validation = validation.sort_values("validation_fitness", ascending=False).head(25)
    top_validation_cols = [
        "fold",
        "formula_name",
        "candidate_weight_long",
        "high_risk_candidate_cap",
        "hard_accept",
        "validation_fitness",
        "validation_total_return",
        "validation_sharpe",
        "validation_short_abs_contribution_share",
        "validation_candidate_score_corr_to_p4",
    ]
    top_validation_cols = [col for col in top_validation_cols if col in top_validation]
    cost_display = cost_summary[cost_summary["one_way_cost_bps"].isin([0, 6, 10, 15, 20, 30])] if not cost_summary.empty else cost_summary
    lines = [
        "# P7 Derivatives P4-Gated Rerank AlphaGPT",
        "",
        "## Design",
        "",
        f"- Horizon fixed at `{horizon}h`.",
        "- Fixed P3d/P4 structure: P4 defines long/short eligible pools; candidate formulas only rerank inside those pools.",
        "- Candidate primitive universe uses only P6b retained feature scores and their slow EMA variants.",
        f"- Retained primitive count: `{len(retained)}`; generated formulas: `{formula_count}`.",
        f"- Candidate weight grid: `{P7_GRID}`.",
        "- Hard accept uses the same P3d validation rule: beat P4 validation on net, Sharpe, and 20bps return while controlling maxDD, turnover, and short contribution share.",
        f"- Each fold selects formula+weight using only the last `{validation_days}` days inside its prior training window.",
        "- If no candidate passes hard accept, the fold falls back to P4-only.",
        f"- Validation ran with `{workers}` workers.",
        "",
        "## Nested Test Summary",
        "",
        "```text",
        summary.to_string(index=False),
        "```",
        "",
        "## Comparison",
        "",
        "```text",
        comparison.to_string(index=False),
        "```",
        "",
        "## Selected Formulas",
        "",
        "```text",
        selected[selected_cols].to_string(index=False),
        "```",
        "",
        "## Primitive Usage",
        "",
        "```text",
        primitive_usage.to_string(index=False),
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
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def update_config_stage(config_path: Path) -> None:
    cfg = load_config(config_path)
    stages = cfg.setdefault("stages", {})
    stages["p7_derivatives_gated_rerank_alphagpt"] = {
        "script": "../../../scripts/run_p7_derivatives_gated_rerank_alphagpt.py",
        "output_dir": "../outputs/p7_derivatives_gated_rerank_alphagpt",
        "report": "../outputs/p7_derivatives_gated_rerank_alphagpt/REPORT.md",
        "candidate_formulas": "../outputs/p7_derivatives_gated_rerank_alphagpt/candidate_formulas.csv",
        "validation_p4_metrics": "../outputs/p7_derivatives_gated_rerank_alphagpt/validation_p4_metrics.csv",
        "validation_incremental_metrics": "../outputs/p7_derivatives_gated_rerank_alphagpt/validation_incremental_metrics.csv",
        "selected_incremental_by_fold": "../outputs/p7_derivatives_gated_rerank_alphagpt/selected_incremental_by_fold.csv",
        "summary": "../outputs/p7_derivatives_gated_rerank_alphagpt/summary.csv",
        "baseline_comparison": "../outputs/p7_derivatives_gated_rerank_alphagpt/baseline_comparison.csv",
        "cost_sensitivity": "../outputs/p7_derivatives_gated_rerank_alphagpt/cost_sensitivity.csv",
        "primitive_usage": "../outputs/p7_derivatives_gated_rerank_alphagpt/primitive_usage.csv",
    }
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False), encoding="utf-8")


if __name__ == "__main__":
    main()
