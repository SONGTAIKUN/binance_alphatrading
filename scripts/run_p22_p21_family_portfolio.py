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
from run_p3_committee_guard_scan import CommitteeSpec
from run_p3d_p4_gated_rerank_alphagpt import (
    CANDIDATE_RANK_COL,
    P4_SCORE_COL,
    SCORE_COL,
    decision_end_for_closed_positions,
    load_panel as load_p3d_panel,
    make_portfolio_config,
    make_risk_template,
)
from run_p4_nested_walkforward_selection import summarize_nested_result
from run_p18_factor_decay_walkforward import DEFAULT_ASSET_PANEL, DEFAULT_MANUAL_PANEL, EXPERIMENT_ROOT
from run_p18c_factor_selector_portfolio import (
    P18C_SCORE,
    add_p18c_entry_fields,
    breakeven_cost,
    build_asset_contribution,
    build_cost_sensitivity,
    build_segment_comparison,
    load_existing_result,
    run_p18c_backtest,
    write_run_result,
)
from run_p18d_guarded_factor_selector import LONG_ROOT
from run_turnover_experiments import to_jsonable


DEFAULT_CONFIG = EXPERIMENT_ROOT / "configs" / "baseline_locked_v1.yaml"
DEFAULT_FOLDS = LONG_ROOT / "inputs" / "long_formula_selections.csv"
DEFAULT_P21_DIR = EXPERIMENT_ROOT / "outputs" / "p21_gross_alpha_family_search"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "outputs" / "p22_p21_family_portfolio"
DEFAULT_P15C_RUN = LONG_ROOT / "p15c_nested" / "nested_extreme_cluster_run"
DEFAULT_COST_BPS = [0, 6, 10, 15, 20, 30]
DEFAULT_PROTOCOLS = ["p20_practical_q25_m5", "p20_loose_decay_corr"]
P22_SCORE = "_p22_fold_local_p21_family_score"


def main() -> None:
    parser = argparse.ArgumentParser(description="P22: plug P21 protocol-selected family scores into strict-neutral allocator.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--asset-panel", default=str(DEFAULT_ASSET_PANEL))
    parser.add_argument("--manual-panel", default=str(DEFAULT_MANUAL_PANEL))
    parser.add_argument("--folds", default=str(DEFAULT_FOLDS))
    parser.add_argument("--p21-dir", default=str(DEFAULT_P21_DIR))
    parser.add_argument("--p15c-run-dir", default=str(DEFAULT_P15C_RUN))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--horizon", type=int, default=72)
    parser.add_argument("--protocols", nargs="+", default=DEFAULT_PROTOCOLS)
    parser.add_argument("--long-fraction", type=float, default=0.50)
    parser.add_argument("--short-confirm-rank", type=float, default=0.35)
    parser.add_argument("--cost-bps", type=float, nargs="+", default=DEFAULT_COST_BPS)
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    horizon = int(args.horizon)
    p21_dir = Path(args.p21_dir)

    print("[1/7] Loading folds, P21 selections, and P15c reference", flush=True)
    folds = pd.read_csv(args.folds, parse_dates=["train_start", "train_end", "test_start", "test_end"]).sort_values("fold")
    selected_all = pd.read_csv(p21_dir / "p21_selected_family_by_fold.csv")
    family_scores = pd.read_csv(p21_dir / "p21_family_score_timeseries.csv.gz", parse_dates=["timestamp"])
    p15c = load_existing_result(Path(args.p15c_run_dir))

    print("[2/7] Loading strict-neutral allocator panel", flush=True)
    base_panel = load_p3d_panel(config_path, Path(args.asset_panel), Path(args.manual_panel), horizon)
    add_p18c_entry_fields(base_panel)
    pf_cfg = make_portfolio_config(cfg, horizon)
    risk_template = make_risk_template(cfg)

    print("[3/7] Replaying P21 family-selected protocols", flush=True)
    results: dict[str, dict[str, Any]] = {}
    selections_out: list[pd.DataFrame] = []
    for protocol in args.protocols:
        print(f"      protocol {protocol}", flush=True)
        protocol_selected = selected_all[selected_all["protocol"].eq(protocol)].copy()
        if protocol_selected.empty:
            raise RuntimeError(f"No P21 selection rows found for protocol={protocol}.")
        protocol_name = f"p22_{protocol.replace('p20_', '')}"
        result, selection = replay_protocol(
            protocol_name=protocol_name,
            protocol=protocol,
            base_panel=base_panel,
            family_scores=family_scores,
            folds=folds,
            selected=protocol_selected,
            p15c=p15c,
            pf_cfg=pf_cfg,
            risk_template=risk_template,
            long_fraction=float(args.long_fraction),
            short_confirm_rank=float(args.short_confirm_rank),
        )
        results[protocol_name] = result
        selections_out.append(selection)
        run_dir = ensure_dir(output_dir / protocol_name / "nested_p22_run")
        write_run_result(run_dir, result)
        selection.to_csv(output_dir / protocol_name / "selected_family_by_fold.csv", index=False)

    print("[4/7] Building comparisons", flush=True)
    refs = {**results, "p15c": p15c}
    comparison = build_comparison(refs)
    comparison.to_csv(output_dir / "comparison_summary.csv", index=False)
    cost = build_cost_sensitivity(refs, [float(x) for x in args.cost_bps])
    cost.to_csv(output_dir / "cost_sensitivity.csv", index=False)
    segments = build_segment_comparison(refs, [float(x) for x in args.cost_bps])
    segments.to_csv(output_dir / "segment_cost_summary.csv", index=False)

    fold_deltas: list[pd.DataFrame] = []
    assets: list[pd.DataFrame] = []
    for name, result in results.items():
        delta = build_fold_delta(result, p15c, name)
        delta.to_csv(output_dir / name / "fold_delta_vs_p15c.csv", index=False)
        fold_deltas.append(delta)
        asset = build_asset_contribution(result["positions"])
        asset["stage"] = name
        asset.to_csv(output_dir / name / "asset_contribution.csv", index=False)
        assets.append(asset)
    fold_delta_all = pd.concat(fold_deltas, ignore_index=True) if fold_deltas else pd.DataFrame()
    fold_delta_all.to_csv(output_dir / "fold_delta_vs_p15c.csv", index=False)
    asset_all = pd.concat(assets, ignore_index=True) if assets else pd.DataFrame()
    asset_all.to_csv(output_dir / "asset_contribution.csv", index=False)
    selected_summary = summarize_selections(pd.concat(selections_out, ignore_index=True))
    selected_summary.to_csv(output_dir / "selection_summary.csv", index=False)

    print("[5/7] Running hard accept checks", flush=True)
    checks = build_accept_checks(comparison, cost, fold_delta_all)
    checks.to_csv(output_dir / "accept_checks.csv", index=False)

    print("[6/7] Writing plots", flush=True)
    plot_curves(output_dir, refs, [0.0, 6.0, 15.0, 20.0, 30.0])
    plot_fold_deltas(output_dir, fold_delta_all)
    plot_selection_summary(output_dir, selected_summary)

    print("[7/7] Writing report", flush=True)
    write_report(output_dir, comparison, cost, segments, selected_summary, fold_delta_all, checks)
    metadata = {
        "horizon": horizon,
        "protocols": list(args.protocols),
        "long_fraction": float(args.long_fraction),
        "short_confirm_rank": float(args.short_confirm_rank),
        "p21_dir": str(p21_dir),
        "p15c_run_dir": str(args.p15c_run_dir),
        "note": "Active folds use P21 protocol-selected family scores in the strict-neutral allocator; failed folds copy P15c exactly.",
    }
    (output_dir / "metadata.json").write_text(json.dumps(to_jsonable(metadata), indent=2), encoding="utf-8")
    print(f"Done: {output_dir}", flush=True)


def replay_protocol(
    protocol_name: str,
    protocol: str,
    base_panel: pd.DataFrame,
    family_scores: pd.DataFrame,
    folds: pd.DataFrame,
    selected: pd.DataFrame,
    p15c: dict[str, Any],
    pf_cfg: Any,
    risk_template: Any,
    long_fraction: float,
    short_confirm_rank: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    score_by_key = {
        (int(fold), str(family)): group.copy()
        for (fold, family), group in family_scores.groupby(["fold", "family"], sort=False)
    }
    selected_by_fold = selected.set_index("fold")
    p15c_folds = p15c["folds"].set_index("fold")
    p15c_equity_by_fold = {int(k): v.copy() for k, v in p15c["equity"].groupby("fold", sort=False)}
    p15c_pos_by_fold = {int(k): v.copy() for k, v in p15c["positions"].groupby("fold", sort=False)}

    spec = CommitteeSpec(
        name=protocol_name,
        w_a07=0.0,
        w_b11=0.0,
        w_c09=0.0,
        long_fraction=float(long_fraction),
        short_guard="p21_family_score_gate",
        short_resid_confirm_rank=float(short_confirm_rank),
    )
    equity_parts: list[pd.DataFrame] = []
    position_parts: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []

    for _, fold in folds.sort_values("fold").iterrows():
        fold_id = int(fold["fold"])
        decision = selected_by_fold.loc[fold_id]
        fallback = bool(decision.get("fallback", True))
        selected_family = str(decision.get("selected_family", "__NO_P21_ALPHA__"))
        score_key = (fold_id, selected_family)
        if fallback or score_key not in score_by_key:
            fold_row = p15c_folds.loc[fold_id].to_dict()
            fold_row.update(
                {
                    "fold": fold_id,
                    "protocol": protocol,
                    "stage": protocol_name,
                    "selected_family": "__P15C_FALLBACK__",
                    "fallback_to_p15c": True,
                    "fallback_reason": "p21_protocol_failed" if fallback else "missing_p21_score",
                    "p21_validation_score": np.nan,
                    "p21_validation_ic_mean": np.nan,
                    "p21_test_ic_mean": np.nan,
                }
            )
            fold_rows.append(fold_row)
            equity = p15c_equity_by_fold.get(fold_id, pd.DataFrame()).copy()
            positions = p15c_pos_by_fold.get(fold_id, pd.DataFrame()).copy()
            if not equity.empty:
                equity["stage"] = protocol_name
                equity["protocol"] = protocol
                equity["selected_family"] = "__P15C_FALLBACK__"
                equity["fallback_to_p15c"] = True
                equity_parts.append(equity)
            if not positions.empty:
                positions["stage"] = protocol_name
                positions["protocol"] = protocol
                positions["selected_family"] = "__P15C_FALLBACK__"
                positions["fallback_to_p15c"] = True
                position_parts.append(positions)
            selection_rows.append(
                selection_row(decision, fold_id, protocol_name, True, "p21_protocol_failed" if fallback else "missing_p21_score")
            )
            continue

        fold_scores = score_by_key[score_key][["timestamp", "symbol", "score"]].rename(columns={"score": P18C_SCORE})
        fold_panel = base_panel.drop(columns=[P18C_SCORE, P4_SCORE_COL, SCORE_COL, CANDIDATE_RANK_COL], errors="ignore").merge(
            fold_scores,
            on=["timestamp", "symbol"],
            how="left",
        )
        fold_panel[P4_SCORE_COL] = fold_panel[P18C_SCORE]
        fold_panel[SCORE_COL] = fold_panel[P18C_SCORE]
        fold_panel[CANDIDATE_RANK_COL] = (
            fold_panel.groupby("timestamp", sort=False)[P18C_SCORE].rank(pct=True, method="average") * 2.0 - 1.0
        )
        risk_cfg = risk_template.__class__(
            **{
                **risk_template.__dict__,
                "name": f"{protocol_name}_{fold_id}",
                "short_resid_confirm_rank": spec.short_resid_confirm_rank,
            }
        )
        start = pd.Timestamp(fold["test_start"])
        end = decision_end_for_closed_positions(pd.Timestamp(fold["test_end"]), pf_cfg.primary_horizon)
        equity, positions, metrics = run_p18c_backtest(
            panel=fold_panel,
            start=start,
            end=end,
            cfg=pf_cfg,
            risk_cfg=risk_cfg,
            spec=spec,
        )
        fold_rows.append(
            {
                "fold": fold_id,
                "train_start": fold["train_start"],
                "train_end": fold["train_end"],
                "test_start": fold["test_start"],
                "test_end": fold["test_end"],
                "test_decision_end": end,
                "protocol": protocol,
                "stage": protocol_name,
                "selected_family": selected_family,
                "fallback_to_p15c": False,
                "fallback_reason": "",
                "p21_validation_score": float(decision.get("validation_score", np.nan)),
                "p21_validation_ic_mean": float(decision.get("validation_ic_mean", np.nan)),
                "p21_test_ic_mean": float(decision.get("test_ic_mean", np.nan)),
                **metrics,
            }
        )
        if not equity.empty:
            equity = equity.copy()
            equity["fold"] = fold_id
            equity["stage"] = protocol_name
            equity["protocol"] = protocol
            equity["selected_family"] = selected_family
            equity["fallback_to_p15c"] = False
            equity["selected_formulas"] = f"P21 family: {selected_family}"
            equity_parts.append(equity)
        if not positions.empty:
            positions = positions.copy()
            positions["fold"] = fold_id
            positions["stage"] = protocol_name
            positions["protocol"] = protocol
            positions["selected_family"] = selected_family
            positions["fallback_to_p15c"] = False
            positions["selected_score_cols"] = f"p21_family::{selected_family}"
            position_parts.append(positions)
        selection_rows.append(selection_row(decision, fold_id, protocol_name, False, ""))

    equity_curve = pd.concat(equity_parts, ignore_index=True) if equity_parts else pd.DataFrame()
    positions = pd.concat(position_parts, ignore_index=True) if position_parts else pd.DataFrame()
    folds_out = pd.DataFrame(fold_rows)
    metrics = summarize_nested_result({"equity": equity_curve, "positions": positions, "folds": folds_out}, pf_cfg.rebalance_hours)
    return {"equity": equity_curve, "positions": positions, "folds": folds_out, "metrics": metrics}, pd.DataFrame(selection_rows)


def selection_row(decision: pd.Series, fold_id: int, stage: str, fallback: bool, reason: str) -> dict[str, Any]:
    return {
        "stage": stage,
        "protocol": decision.get("protocol", ""),
        "fold": int(fold_id),
        "selected_family": "__P15C_FALLBACK__" if fallback else str(decision.get("selected_family")),
        "fallback_to_p15c": bool(fallback),
        "fallback_reason": reason,
        "p21_validation_score": float(decision.get("validation_score", np.nan)),
        "p21_validation_ic_mean": float(decision.get("validation_ic_mean", np.nan)),
        "p21_validation_ic_q25": float(decision.get("validation_ic_q25", np.nan)),
        "p21_validation_second_half_ic": float(decision.get("validation_second_half_ic", np.nan)),
        "p21_validation_decay_delta": float(decision.get("validation_decay_delta", np.nan)),
        "p21_test_ic_mean": float(decision.get("test_ic_mean", np.nan)),
        "p21_test_top_bottom_spread": float(decision.get("test_top_bottom_spread", np.nan)),
        "components": decision.get("components", ""),
    }


def build_comparison(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame([{"stage": name, **result["metrics"]} for name, result in results.items()])


def build_fold_delta(p22: dict[str, Any], p15c: dict[str, Any], stage_name: str) -> pd.DataFrame:
    left_cols = [
        "fold",
        "test_start",
        "selected_family",
        "fallback_to_p15c",
        "fallback_reason",
        "p21_validation_score",
        "p21_validation_ic_mean",
        "p21_test_ic_mean",
        "total_return",
        "gross_total_return_before_cost",
        "sharpe",
        "max_drawdown",
        "mean_turnover",
    ]
    left = p22["folds"][[c for c in left_cols if c in p22["folds"].columns]].copy()
    right = p15c["folds"][["fold", "total_return", "gross_total_return_before_cost", "sharpe", "max_drawdown", "mean_turnover"]].copy()
    merged = left.merge(right, on="fold", suffixes=(f"_{stage_name}", "_p15c"))
    merged["stage"] = stage_name
    for col in ["total_return", "gross_total_return_before_cost", "sharpe", "max_drawdown", "mean_turnover"]:
        merged[f"delta_{col}"] = merged[f"{col}_{stage_name}"] - merged[f"{col}_p15c"]
    return merged


def summarize_selections(selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stage, group in selected.groupby("stage", sort=False):
        active = group[~group["fallback_to_p15c"].astype(bool)].copy()
        rows.append(
            {
                "stage": stage,
                "fold_count": int(len(group)),
                "active_fold_count": int(len(active)),
                "active_fold_rate": float(len(active) / len(group)) if len(group) else np.nan,
                "fallback_fold_count": int(group["fallback_to_p15c"].astype(bool).sum()),
                "selected_test_ic_mean": active["p21_test_ic_mean"].mean() if len(active) else np.nan,
                "selected_test_top_bottom_spread_mean": active["p21_test_top_bottom_spread"].mean() if len(active) else np.nan,
                "most_common_family": active["selected_family"].mode().iloc[0] if len(active) else "__P15C_FALLBACK__",
            }
        )
    counts = (
        selected.groupby(["stage", "selected_family"], dropna=False)
        .size()
        .reset_index(name="selected_fold_count")
        .sort_values(["stage", "selected_fold_count"], ascending=[True, False])
    )
    return pd.DataFrame(rows).merge(
        counts.groupby("stage")
        .apply(lambda x: "; ".join(f"{r.selected_family}:{int(r.selected_fold_count)}" for r in x.itertuples()))
        .rename("family_counts")
        .reset_index(),
        on="stage",
        how="left",
    )


def build_accept_checks(comparison: pd.DataFrame, cost: pd.DataFrame, fold_delta: pd.DataFrame) -> pd.DataFrame:
    p15c = comparison[comparison["stage"].eq("p15c")].iloc[0]
    rows: list[dict[str, Any]] = []
    for stage in comparison["stage"]:
        if stage == "p15c":
            continue
        row = comparison[comparison["stage"].eq(stage)].iloc[0]
        c_stage = cost[cost["stage"].eq(stage)].set_index("one_way_cost_bps")
        c_p15c = cost[cost["stage"].eq("p15c")].set_index("one_way_cost_bps")
        fd = fold_delta[fold_delta["stage"].eq(stage)]
        stage_checks = [
            ("gross_return_gt_p15c", row["gross_total_return_before_cost"] - p15c["gross_total_return_before_cost"], row["gross_total_return_before_cost"] > p15c["gross_total_return_before_cost"]),
            ("net_6bps_gt_p15c", c_stage.loc[6.0, "total_return"] - c_p15c.loc[6.0, "total_return"], c_stage.loc[6.0, "total_return"] > c_p15c.loc[6.0, "total_return"]),
            ("net_20bps_gt_p15c", c_stage.loc[20.0, "total_return"] - c_p15c.loc[20.0, "total_return"], c_stage.loc[20.0, "total_return"] > c_p15c.loc[20.0, "total_return"]),
            ("maxdd_not_20pct_worse", row["max_drawdown"] - p15c["max_drawdown"], abs(row["max_drawdown"]) <= abs(p15c["max_drawdown"]) * 1.2),
            ("positive_fold_rate_not_lower", row["positive_fold_rate"] - p15c["positive_fold_rate"], row["positive_fold_rate"] >= p15c["positive_fold_rate"]),
            ("breakeven_gt_p15c", breakeven_cost(cost, stage) - breakeven_cost(cost, "p15c"), breakeven_cost(cost, stage) > breakeven_cost(cost, "p15c")),
            ("fold14_delta_nonnegative", fold14_delta(fd), fold14_delta(fd) >= 0.0),
            ("worst5_fold_delta_nonnegative", fd.nsmallest(5, "delta_total_return")["delta_total_return"].sum(), fd.nsmallest(5, "delta_total_return")["delta_total_return"].sum() >= 0.0),
        ]
        for check, value, passed in stage_checks:
            rows.append({"stage": stage, "check": check, "value": float(value), "pass": bool(passed)})
    return pd.DataFrame(rows)


def fold14_delta(fold_delta: pd.DataFrame) -> float:
    row = fold_delta[fold_delta["fold"].eq(14)]
    return float(row["delta_total_return"].iloc[0]) if not row.empty else np.nan


def plot_curves(output_dir: Path, results: dict[str, dict[str, Any]], cost_bps: list[float]) -> None:
    for bps in cost_bps:
        fig, ax = plt.subplots(figsize=(14, 7))
        for name, result in results.items():
            equity = result["equity"].sort_values("timestamp")
            gross = equity["gross_period_return"] if "gross_period_return" in equity else equity["period_return"] + equity["cost_return"]
            ret = gross - equity["turnover"] * bps / 10_000.0
            curve = (1.0 + ret).cumprod() - 1.0
            width = 2.3 if name.startswith("p22") else 1.4
            ax.plot(equity["timestamp"], curve, label=name, linewidth=width)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_title(f"P22 P21-family portfolio vs P15c, {bps:g}bps one-way cost")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Cumulative return")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / f"p22_vs_p15c_pnl_{int(bps)}bps.png", dpi=160)
        plt.close(fig)


def plot_fold_deltas(output_dir: Path, fold_delta: pd.DataFrame) -> None:
    if fold_delta.empty:
        return
    stages = list(fold_delta["stage"].drop_duplicates())
    fig, axes = plt.subplots(len(stages), 1, figsize=(14, 4.5 * len(stages)), sharex=True)
    if len(stages) == 1:
        axes = [axes]
    for ax, stage in zip(axes, stages):
        frame = fold_delta[fold_delta["stage"].eq(stage)].sort_values("fold")
        colors = np.where(frame["delta_total_return"] >= 0, "#2a9d8f", "#e76f51")
        ax.bar(frame["fold"], frame["delta_total_return"], color=colors)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_title(f"{stage} fold net return delta vs P15c")
        ax.set_ylabel("Delta return")
    axes[-1].set_xlabel("Fold")
    fig.tight_layout()
    fig.savefig(output_dir / "fold_delta_vs_p15c.png", dpi=160)
    plt.close(fig)


def plot_selection_summary(output_dir: Path, selected_summary: pd.DataFrame) -> None:
    if selected_summary.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(selected_summary["stage"], selected_summary["active_fold_rate"])
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0.0, 1.0)
    ax.set_title("P22 active fold rate by protocol")
    ax.set_ylabel("Active fold rate")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(output_dir / "active_fold_rate.png", dpi=160)
    plt.close(fig)


def write_report(
    output_dir: Path,
    comparison: pd.DataFrame,
    cost: pd.DataFrame,
    segments: pd.DataFrame,
    selected_summary: pd.DataFrame,
    fold_delta: pd.DataFrame,
    checks: pd.DataFrame,
) -> None:
    report = [
        "# P22 P21 Family Portfolio",
        "",
        "P22 plugs P21 protocol-selected family scores into the same strict-neutral allocator used by P18c-style portfolio tests. Folds that fail the P21/P20 protocol are copied from P15c exactly.",
        "",
        "## Selection Summary",
        "",
        "```text",
        selected_summary.to_string(index=False) if not selected_summary.empty else "None",
        "```",
        "",
        "## Comparison",
        "",
        "```text",
        comparison.to_string(index=False),
        "```",
        "",
        "## Cost Sensitivity",
        "",
        "```text",
        cost[cost["one_way_cost_bps"].isin([0.0, 6.0, 15.0, 20.0, 30.0])].to_string(index=False),
        "```",
        "",
        "## Breakeven One-Way Cost",
        "",
        "```text",
        pd.DataFrame(
            [{"stage": stage, "breakeven_one_way_bps": breakeven_cost(cost, stage)} for stage in cost["stage"].drop_duplicates()]
        ).to_string(index=False),
        "```",
        "",
        "## Accept Checks",
        "",
        "```text",
        checks.to_string(index=False) if not checks.empty else "None",
        "```",
        "",
        "## Segment Summary",
        "",
        "```text",
        segments[segments["one_way_cost_bps"].isin([6.0, 15.0, 20.0, 30.0])].to_string(index=False),
        "```",
        "",
        "## Worst Fold Deltas vs P15c",
        "",
        "```text",
        fold_delta.sort_values("delta_total_return").head(16).to_string(index=False) if not fold_delta.empty else "None",
        "```",
        "",
        "## Fold 14",
        "",
        "```text",
        fold_delta[fold_delta["fold"].eq(14)].to_string(index=False) if not fold_delta.empty else "None",
        "```",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
