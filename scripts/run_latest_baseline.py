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
from matplotlib.ticker import PercentFormatter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from crypto_stat_arb.config import ensure_dir, load_config
from run_p13_turnover_no_trade_zone import write_result
import run_p3d_p4_gated_rerank_alphagpt as p3d
import run_p7_derivatives_gated_rerank_alphagpt as p7
import run_p15c_extreme_cluster_turnover as p15c
from run_p13_turnover_no_trade_zone import make_fold_risk
from run_turnover_experiments import to_jsonable


EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "baseline_locked_v1"
DEFAULT_CONFIG = EXPERIMENT_ROOT / "configs" / "baseline_locked_v1.yaml"
DEFAULT_PANEL = EXPERIMENT_ROOT / "inputs" / "p16_replay_panel_slim.csv.gz"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "outputs" / "latest_baseline_replay"
DEFAULT_P9_SELECTED = EXPERIMENT_ROOT / "outputs" / "p9_clean_baseline_package" / "selected_incremental_by_fold.csv"
DEFAULT_P16_SELECTED = (
    EXPERIMENT_ROOT
    / "outputs"
    / "p16_clean_candidate_package"
    / "fixed_selected_extreme_cluster_by_fold.csv"
)
DEFAULT_RETAINED = EXPERIMENT_ROOT / "outputs" / "p6b_derivatives_feature_expansion" / "retained_features.csv"
DEFAULT_P3D_CANDIDATES = (
    EXPERIMENT_ROOT / "outputs" / "p3d_p4_gated_rerank_alphagpt" / "candidate_formulas.csv"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the latest clean P16/P15c crypto residual stat-arb baseline from "
            "fixed walk-forward selections."
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--panel", default=str(DEFAULT_PANEL))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--p9-selected", default=str(DEFAULT_P9_SELECTED))
    parser.add_argument("--p16-selected", default=str(DEFAULT_P16_SELECTED))
    parser.add_argument("--retained-features", default=str(DEFAULT_RETAINED))
    parser.add_argument("--p3d-candidate-formulas", default=str(DEFAULT_P3D_CANDIDATES))
    parser.add_argument("--horizon", type=int, default=72)
    parser.add_argument("--cost-bps", type=float, nargs="+", default=[0, 6, 10, 15, 20, 30])
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    output_dir = ensure_dir(args.output_dir)
    run_dir = ensure_dir(output_dir / "nested_candidate_run")
    horizon = int(args.horizon)

    panel_path = Path(args.panel).resolve()
    restore_panel_if_needed(panel_path)

    print("[1/6] Loading fixed walk-forward selections", flush=True)
    selections_path = p3d.resolve_config_path(config_path, cfg["inputs"]["formula_selections"])
    selections = pd.read_csv(selections_path, parse_dates=["train_start", "train_end", "test_start", "test_end"])
    selected_p9 = pd.read_csv(
        Path(args.p9_selected).resolve(),
        parse_dates=["train_start", "train_end", "test_start", "test_end", "test_decision_end"],
    )
    selected_p16 = pd.read_csv(Path(args.p16_selected).resolve())

    print("[2/6] Loading processed replay panel", flush=True)
    panel = pd.read_csv(panel_path, parse_dates=["timestamp"])
    required = {"timestamp", "symbol", "residual_log_ret", f"label_resid_ret_h{horizon}", f"fwd_simple_ret_h{horizon}"}
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ValueError(f"Replay panel is missing required columns: {missing}")

    print("[3/6] Loading fixed formula text maps", flush=True)
    retained = pd.read_csv(Path(args.retained_features).resolve())
    p7._P7_FEATURE_MAP = p7.build_p7_feature_map(retained)
    p3d_candidates = pd.read_csv(Path(args.p3d_candidate_formulas).resolve())
    p3d_formula_text = dict(zip(p3d_candidates["name"].astype(str), p3d_candidates["formula"].astype(str)))
    spec_by_name = {spec.name: spec for spec in p3d.build_committee_specs()}
    extreme_specs = p15c.build_extreme_specs()

    print("[4/6] Replaying selected P16 specs on test windows only", flush=True)
    residual_wide = (
        panel[["timestamp", "symbol", "residual_log_ret"]]
        .dropna()
        .pivot(index="timestamp", columns="symbol", values="residual_log_ret")
        .sort_index()
    )
    pf_cfg = p3d.make_portfolio_config(cfg, horizon)
    risk_template = p3d.make_risk_template(cfg)
    result = replay_selected_specs_from_formula_text(
        panel=panel,
        residual_wide=residual_wide,
        selections=selections,
        selected=selected_p16,
        selected_p9=selected_p9,
        pf_cfg=pf_cfg,
        risk_template=risk_template,
        spec_by_name=spec_by_name,
        p3d_formula_text=p3d_formula_text,
        extreme_specs=extreme_specs,
    )
    write_result(run_dir, result)

    print("[5/6] Writing summary and clean PnL chart", flush=True)
    metrics = dict(result["metrics"])
    cost_summary = build_cost_sensitivity(result["equity"], horizon, [float(x) for x in args.cost_bps])
    cost_summary.to_csv(output_dir / "cost_sensitivity.csv", index=False)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(to_jsonable(metrics), fh, indent=2)
    plot_clean_pnl(output_dir / "clean_pnl_curve.png", result["equity"], title="P16/P15c latest baseline")
    write_readme(output_dir, metrics, cost_summary, panel_path)

    print("[6/6] Done", flush=True)
    print(json.dumps(to_jsonable(metrics), indent=2), flush=True)
    print(f"Done: {output_dir}", flush=True)


def restore_panel_if_needed(panel_path: Path) -> None:
    if panel_path.exists():
        return
    parts_dir = panel_path.parent / f"{panel_path.name}.parts"
    parts = sorted(parts_dir.glob("part-*"))
    if not parts:
        raise FileNotFoundError(
            f"Replay panel not found at {panel_path}, and no split parts found in {parts_dir}."
        )
    ensure_dir(panel_path.parent)
    with panel_path.open("wb") as out:
        for part in parts:
            out.write(part.read_bytes())


def replay_selected_specs_from_formula_text(
    panel: pd.DataFrame,
    residual_wide: pd.DataFrame,
    selections: pd.DataFrame,
    selected: pd.DataFrame,
    selected_p9: pd.DataFrame,
    pf_cfg: Any,
    risk_template: Any,
    spec_by_name: dict[str, Any],
    p3d_formula_text: dict[str, str],
    extreme_specs: list[p15c.ExtremeClusterSpec],
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
        exec_spec = None if bool(decision["fallback_to_p9_sizing"]) else spec_map[selected_name]
        committee = spec_by_name[str(base["p4_spec"])]
        formula_text = prepare_fold_score_from_formula_text(panel, base, committee, p3d_formula_text)
        risk_cfg = make_fold_risk(risk_template, committee, f"latest_{selected_name}_{fold_id}")
        start = pd.Timestamp(fold["test_start"])
        end = p3d.decision_end_for_closed_positions(pd.Timestamp(fold["test_end"]), pf_cfg.primary_horizon)
        equity, positions, metrics = p15c.run_extreme_backtest(
            panel=panel,
            residual_wide=residual_wide,
            start=start,
            end=end,
            cfg=pf_cfg,
            risk_cfg=risk_cfg,
            committee=committee,
            spec=exec_spec,
            diagnostic_lookback=int(decision["lookback_hours"]),
            diagnostic_k=int(decision["cluster_k"]),
            score_col=p3d.SCORE_COL,
            selected_formulas=formula_text,
        )
        p15c.enrich_extreme_metrics(equity, positions, metrics, pf_cfg.primary_horizon)
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
                "selected_extreme_spec": p15c.NO_EXTREME_CAP if exec_spec is None else exec_spec.name,
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
            equity["selected_extreme_spec"] = p15c.NO_EXTREME_CAP if exec_spec is None else exec_spec.name
            equity["validation_selected_extreme_spec"] = selected_name
            positions["fold"] = fold_id
            positions["p4_spec"] = committee.name
            positions["formula_name"] = str(base["deriv_formula_name"])
            positions["formula"] = formula_text
            positions["selected_extreme_spec"] = p15c.NO_EXTREME_CAP if exec_spec is None else exec_spec.name
            positions["validation_selected_extreme_spec"] = selected_name
            equity_parts.append(equity)
            position_parts.append(positions)
    equity_curve = pd.concat(equity_parts, ignore_index=True) if equity_parts else pd.DataFrame()
    positions = pd.concat(position_parts, ignore_index=True) if position_parts else pd.DataFrame()
    folds = pd.DataFrame(fold_rows)
    metrics = p3d.summarize_nested_result({"equity": equity_curve, "positions": positions, "folds": folds}, pf_cfg.rebalance_hours)
    p15c.enrich_extreme_metrics(equity_curve, positions, metrics, pf_cfg.primary_horizon)
    return {"folds": folds, "equity": equity_curve, "positions": positions, "metrics": metrics}


def prepare_fold_score_from_formula_text(
    panel: pd.DataFrame,
    base: pd.Series,
    committee: Any,
    p3d_formula_text: dict[str, str],
) -> str:
    panel[p3d.P4_SCORE_COL] = p3d.p4_score(panel, committee)
    p3d_name = str(base["p3d_formula_name"])
    if p3d_name == p3d.P4_ONLY_FORMULA:
        panel["_latest_base_score"] = panel[p3d.P4_SCORE_COL]
    else:
        formula = p3d_formula_text[p3d_name]
        candidate_rank = evaluate_formula_text(formula, panel, namespace="p3d").groupby(
            panel["timestamp"], sort=False
        ).rank(pct=True, method="average")
        panel[p3d.CANDIDATE_RANK_COL] = 2.0 * candidate_rank - 1.0
        p3d.apply_constrained_incremental_score(
            panel,
            candidate_weight_long=float(base["p3d_candidate_weight_long"]),
            candidate_weight_short=float(base["p3d_candidate_weight_short"]),
            high_risk_candidate_cap=float(base["p3d_high_risk_candidate_cap"]),
        )
        panel["_latest_base_score"] = panel[p3d.SCORE_COL]

    deriv_name = str(base["deriv_formula_name"])
    if deriv_name == p3d.P4_ONLY_FORMULA:
        panel[p3d.SCORE_COL] = panel["_latest_base_score"]
        panel[p3d.CANDIDATE_RANK_COL] = np.nan
        return f"P3d_base({p3d_name}) fallback"

    deriv_formula = str(base["deriv_formula"])
    deriv_rank = evaluate_formula_text(deriv_formula, panel, namespace="p7").groupby(
        panel["timestamp"], sort=False
    ).rank(pct=True, method="average")
    deriv_rank = 2.0 * deriv_rank - 1.0
    panel[p3d.CANDIDATE_RANK_COL] = deriv_rank
    side_weight = np.where(
        deriv_rank >= 0.0,
        float(base["candidate_weight_long"]),
        float(base["candidate_weight_short"]),
    )
    high_risk = panel[p3d.HIGH_RISK_REGIME_COL].fillna(False).to_numpy(dtype=bool)
    effective = np.where(high_risk, np.minimum(side_weight, float(base["high_risk_candidate_cap"])), side_weight)
    panel[p3d.EFFECTIVE_CANDIDATE_WEIGHT_COL] = effective
    panel[p3d.SCORE_COL] = (1.0 - effective) * panel["_latest_base_score"].astype(float) + effective * deriv_rank.astype(float)
    return f"P3d_base({p3d_name}) + deriv({deriv_name},w={float(base['candidate_weight_long']):.2f}): {deriv_formula}"


def evaluate_formula_text(expr: str, panel: pd.DataFrame, namespace: str) -> pd.Series:
    expr = strip_outer(str(expr).strip())
    if is_number(expr):
        return pd.Series(float(expr), index=panel.index)
    parts = split_top_level(expr, "*")
    if len(parts) > 1:
        out = pd.Series(1.0, index=panel.index)
        for part in parts:
            out = out * evaluate_formula_text(part, panel, namespace)
        return clean_series(out)
    for op in ["ADD", "SUB", "MUL", "DIV_SAFE"]:
        prefix = f"{op}("
        if expr.startswith(prefix) and expr.endswith(")"):
            left, right = split_args(expr[len(prefix) : -1])
            left_s = evaluate_formula_text(left, panel, namespace)
            right_s = evaluate_formula_text(right, panel, namespace)
            if op == "ADD":
                return clean_series(left_s + right_s)
            if op == "SUB":
                return clean_series(left_s - right_s)
            if op == "MUL":
                return clean_series(left_s * right_s)
            denom = right_s.where(right_s.abs() > 1e-4, np.nan)
            return clean_series(left_s / denom)
    for op in ["NEG", "CLIP"]:
        prefix = f"{op}("
        if expr.startswith(prefix) and expr.endswith(")"):
            inner = evaluate_formula_text(expr[len(prefix) : -1], panel, namespace)
            return clean_series(-inner if op == "NEG" else inner.clip(-3.0, 3.0))
    col = primitive_to_column(expr, namespace)
    return panel[col].astype(float)


def primitive_to_column(expr: str, namespace: str) -> str:
    if namespace == "p7" or expr.startswith("P6B(") or expr.startswith("EMA_SLOW(P6B("):
        col = p7.feature_column_p7(expr)
    else:
        from run_p5_alphagpt_slow_search import feature_column

        col = feature_column(expr)
    return col


def clean_series(series: pd.Series) -> pd.Series:
    return pd.Series(series, index=series.index).replace([np.inf, -np.inf], np.nan).clip(-50.0, 50.0)


def strip_outer(expr: str) -> str:
    expr = expr.strip()
    if expr.startswith("(") and expr.endswith(")"):
        depth = 0
        for idx, char in enumerate(expr):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0 and idx != len(expr) - 1:
                    return expr
        return strip_outer(expr[1:-1])
    return expr


def split_args(text: str) -> list[str]:
    return split_top_level(text, ",")


def split_top_level(text: str, sep: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    for idx, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == sep and depth == 0:
            parts.append(text[start:idx].strip())
            start = idx + 1
    if not parts:
        return [text.strip()]
    parts.append(text[start:].strip())
    return parts


def is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def build_cost_sensitivity(equity: pd.DataFrame, horizon: int, cost_bps: list[float]) -> pd.DataFrame:
    if equity.empty:
        return pd.DataFrame()
    base_cost = equity["cost_return"].astype(float)
    turnover = equity["turnover"].astype(float).replace(0.0, np.nan)
    one_way_cost_bps = (base_cost / turnover * 10000.0).replace([np.inf, -np.inf], np.nan)
    inferred_bps = float(one_way_cost_bps.dropna().median()) if one_way_cost_bps.notna().any() else 6.0
    gross = equity["gross_period_return"].astype(float)
    rows: list[dict[str, Any]] = []
    for bps in cost_bps:
        net_ret = gross - equity["turnover"].astype(float) * float(bps) / 10000.0
        eq = (1.0 + net_ret).cumprod()
        ret = float(eq.iloc[-1] - 1.0)
        ann_factor = 365.0 * 24.0 / float(horizon)
        sharpe = float(net_ret.mean() / net_ret.std(ddof=0) * np.sqrt(ann_factor)) if net_ret.std(ddof=0) > 0 else 0.0
        drawdown = eq / eq.cummax() - 1.0
        rows.append(
            {
                "one_way_cost_bps": float(bps),
                "total_return": ret,
                "sharpe": sharpe,
                "max_drawdown": float(drawdown.min()),
                "inferred_original_cost_bps": inferred_bps,
            }
        )
    return pd.DataFrame(rows)


def plot_clean_pnl(path: Path, equity: pd.DataFrame, title: str) -> None:
    if equity.empty:
        return
    data = equity.sort_values("timestamp").copy()
    pnl = (1.0 + data["period_return"].astype(float)).cumprod() - 1.0
    gross = (1.0 + data["gross_period_return"].astype(float)).cumprod() - 1.0
    start = pd.DataFrame({"timestamp": [data["timestamp"].iloc[0]], "net": [0.0], "gross": [0.0]})
    plot_data = pd.concat(
        [
            start,
            pd.DataFrame({"timestamp": data["timestamp"], "net": pnl, "gross": gross}),
        ],
        ignore_index=True,
    )
    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.plot(plot_data["timestamp"], plot_data["net"], label="net after 6bps cost", linewidth=2.0)
    ax.plot(plot_data["timestamp"], plot_data["gross"], label="gross before cost", linewidth=1.4, alpha=0.75)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title(title)
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Cumulative return")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_readme(output_dir: Path, metrics: dict[str, Any], cost_summary: pd.DataFrame, panel_path: Path) -> None:
    cost_lines = []
    for row in cost_summary.itertuples(index=False):
        cost_lines.append(
            f"- {float(row.one_way_cost_bps):.0f}bps: return {float(row.total_return):.2%}, "
            f"Sharpe {float(row.sharpe):.3f}, maxDD {float(row.max_drawdown):.2%}"
        )
    text = f"""# Latest Baseline Replay

This directory is generated by `scripts/run_latest_baseline.py`.

The run replays fixed walk-forward selections only:

- P9 fixed AlphaGPT incremental selections are loaded from `selected_incremental_by_fold.csv`.
- P16/P15c extreme-cluster turnover-aware allocator choices are loaded from `fixed_selected_extreme_cluster_by_fold.csv`.
- Each fold applies the already selected parameters only to its next test window.

Panel: `{panel_path}`

Core metrics at the default 6bps one-way cost:

- total_return: {float(metrics.get("total_return", 0.0)):.4%}
- gross_total_return: {float(metrics.get("gross_total_return", 0.0)):.4%}
- sharpe: {float(metrics.get("sharpe", 0.0)):.3f}
- max_drawdown: {float(metrics.get("max_drawdown", 0.0)):.4%}
- mean_turnover: {float(metrics.get("mean_turnover", 0.0)):.4f}

Cost sensitivity:

{chr(10).join(cost_lines)}
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
