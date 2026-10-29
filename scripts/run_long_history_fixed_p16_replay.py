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
from run_latest_baseline import (
    DEFAULT_CONFIG,
    DEFAULT_P16_SELECTED,
    DEFAULT_P3D_CANDIDATES,
    DEFAULT_P9_SELECTED,
    DEFAULT_RETAINED,
    build_cost_sensitivity,
    plot_clean_pnl,
    replay_selected_specs_from_formula_text,
)
from run_p13_turnover_no_trade_zone import write_result
from run_turnover_experiments import to_jsonable
import run_p3d_p4_gated_rerank_alphagpt as p3d
import run_p7_derivatives_gated_rerank_alphagpt as p7
import run_p15c_extreme_cluster_turnover as p15c


EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "baseline_locked_v1"
DEFAULT_LONG_PANEL = EXPERIMENT_ROOT / "inputs" / "long_history" / "p16_long_replay_panel.csv.gz"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "outputs" / "long_history_fixed_p16_replay"
# This makes fold 10 start at 2024-07-11 08:00 UTC, matching the locked
# P16/P15c baseline's original first test fold.
DEFAULT_LONG_START = "2023-10-15 08:00:00+00:00"
ORIGINAL_FIRST_TEST = "2024-07-11 08:00:00+00:00"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the locked P16/P15c baseline on the longer 2023+ panel. "
            "This is a fixed-policy stress replay, not a fresh nested parameter search."
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--panel", default=str(DEFAULT_LONG_PANEL))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--p9-selected", default=str(DEFAULT_P9_SELECTED))
    parser.add_argument("--p16-selected", default=str(DEFAULT_P16_SELECTED))
    parser.add_argument("--retained-features", default=str(DEFAULT_RETAINED))
    parser.add_argument("--p3d-candidate-formulas", default=str(DEFAULT_P3D_CANDIDATES))
    parser.add_argument("--horizon", type=int, default=72)
    parser.add_argument("--train-days", type=int, default=180)
    parser.add_argument("--test-days", type=int, default=30)
    parser.add_argument("--start", default=DEFAULT_LONG_START)
    parser.add_argument("--cost-bps", type=float, nargs="+", default=[0, 6, 10, 15, 20, 30])
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    output_dir = ensure_dir(args.output_dir)
    run_dir = ensure_dir(output_dir / "fixed_policy_run")
    horizon = int(args.horizon)

    print("[1/7] Loading long replay panel", flush=True)
    panel_path = Path(args.panel).resolve()
    panel = pd.read_csv(panel_path, parse_dates=["timestamp"])
    required = {"timestamp", "symbol", "residual_log_ret", f"label_resid_ret_h{horizon}", f"fwd_simple_ret_h{horizon}"}
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ValueError(f"Long replay panel is missing required columns: {missing}")
    panel = panel.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    print("[2/7] Building long-history stress folds and mapped selections", flush=True)
    base_selections = pd.read_csv(
        p3d.resolve_config_path(config_path, cfg["inputs"]["formula_selections"]),
        parse_dates=["train_start", "train_end", "test_start", "test_end"],
    )
    base_p9 = pd.read_csv(
        Path(args.p9_selected).resolve(),
        parse_dates=["train_start", "train_end", "test_start", "test_end", "test_decision_end"],
    )
    base_p16 = pd.read_csv(Path(args.p16_selected).resolve())
    selections = build_long_folds(panel, args.start, int(args.train_days), int(args.test_days), horizon)
    selected_p9, selected_p16, mapping = map_locked_selections(selections, base_selections, base_p9, base_p16, horizon)
    selections.to_csv(output_dir / "long_history_stress_folds.csv", index=False)
    mapping.to_csv(output_dir / "fold_selection_mapping.csv", index=False)
    selected_p9.to_csv(output_dir / "mapped_selected_incremental_by_fold.csv", index=False)
    selected_p16.to_csv(output_dir / "mapped_selected_extreme_cluster_by_fold.csv", index=False)

    print("[3/7] Loading formula maps", flush=True)
    retained = pd.read_csv(Path(args.retained_features).resolve())
    p7._P7_FEATURE_MAP = p7.build_p7_feature_map(retained)
    p3d_candidates = pd.read_csv(Path(args.p3d_candidate_formulas).resolve())
    p3d_formula_text = dict(zip(p3d_candidates["name"].astype(str), p3d_candidates["formula"].astype(str)))
    spec_by_name = {spec.name: spec for spec in p3d.build_committee_specs()}
    extreme_specs = p15c.build_extreme_specs()

    print("[4/7] Replaying locked P16/P15c policy on long-history test windows", flush=True)
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

    print("[5/7] Writing cost and segment diagnostics", flush=True)
    metrics = dict(result["metrics"])
    cost_summary = build_cost_sensitivity(result["equity"], horizon, [float(x) for x in args.cost_bps])
    cost_summary.to_csv(output_dir / "cost_sensitivity.csv", index=False)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(to_jsonable(metrics), fh, indent=2)
    segment_summary = build_segment_summary(
        result["equity"],
        horizon=horizon,
        cost_bps=[float(x) for x in args.cost_bps],
        original_first_test=pd.Timestamp(ORIGINAL_FIRST_TEST),
    )
    segment_summary.to_csv(output_dir / "segment_cost_summary.csv", index=False)
    fold_summary = build_fold_summary(result["equity"], selections, horizon, cost_bps=15.0)
    fold_summary.to_csv(output_dir / "fold_summary_15bps.csv", index=False)

    print("[6/7] Plotting clean PnL curves", flush=True)
    plot_clean_pnl(output_dir / "clean_pnl_curve_6bps.png", result["equity"], title="Long-history P16/P15c fixed-policy replay")
    plot_cost_curve(
        output_dir / "pnl_curve_15bps.png",
        result["equity"],
        cost_bps=15.0,
        title="Long-history P16/P15c fixed-policy replay, 15bps one-way cost",
    )

    print("[7/7] Writing report", flush=True)
    write_report(
        output_dir=output_dir,
        panel_path=panel_path,
        panel=panel,
        selections=selections,
        mapping=mapping,
        metrics=metrics,
        cost_summary=cost_summary,
        segment_summary=segment_summary,
        fold_summary=fold_summary,
    )
    print(json.dumps(to_jsonable(metrics), indent=2), flush=True)
    print(f"Done: {output_dir}", flush=True)


def build_long_folds(panel: pd.DataFrame, start: str, train_days: int, test_days: int, horizon: int) -> pd.DataFrame:
    timestamps = pd.DatetimeIndex(pd.to_datetime(panel["timestamp"].drop_duplicates()).sort_values())
    first_ts = pd.Timestamp(timestamps[0])
    last_ts = pd.Timestamp(timestamps[-1])
    start_ts = pd.Timestamp(start)
    min_test_start = first_ts + pd.Timedelta(days=train_days)
    if start_ts < min_test_start:
        start_ts = min_test_start
    start_ts = next_timestamp_at_or_after(timestamps, start_ts)
    rows: list[dict[str, Any]] = []
    fold = 1
    test_start = start_ts
    max_end = last_ts - pd.Timedelta(hours=horizon)
    while test_start + pd.Timedelta(days=test_days) <= max_end:
        train_start = next_timestamp_at_or_after(timestamps, test_start - pd.Timedelta(days=train_days))
        train_end = next_timestamp_at_or_after(timestamps, test_start)
        test_end = next_timestamp_at_or_after(timestamps, test_start + pd.Timedelta(days=test_days))
        rows.append(
            {
                "fold": fold,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": train_end,
                "test_end": test_end,
                "selected_formulas": "locked_p16_fixed_policy_stress",
                "selected_score_cols": "locked_p16_fixed_policy_stress",
                "validation_fitness": np.nan,
                "validation_total_return": np.nan,
                "validation_sharpe": np.nan,
                "validation_mean_turnover": np.nan,
            }
        )
        fold += 1
        test_start = next_timestamp_at_or_after(timestamps, test_start + pd.Timedelta(days=test_days))
    if not rows:
        raise ValueError("No long-history folds could be built.")
    return pd.DataFrame(rows)


def next_timestamp_at_or_after(timestamps: pd.DatetimeIndex, ts: pd.Timestamp) -> pd.Timestamp:
    if timestamps.tz is not None and ts.tzinfo is None:
        ts = ts.tz_localize(timestamps.tz)
    if timestamps.tz is None and ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    idx = int(timestamps.searchsorted(ts, side="left"))
    if idx >= len(timestamps):
        raise ValueError(f"No timestamp at or after {ts}")
    return pd.Timestamp(timestamps[idx])


def map_locked_selections(
    long_folds: pd.DataFrame,
    base_selections: pd.DataFrame,
    base_p9: pd.DataFrame,
    base_p16: pd.DataFrame,
    horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_starts = pd.DatetimeIndex(pd.to_datetime(base_selections["test_start"]).sort_values())
    base_by_fold = base_selections.set_index("fold")
    p9_by_fold = base_p9.set_index("fold")
    p16_by_fold = base_p16.set_index("fold")
    p9_rows: list[dict[str, Any]] = []
    p16_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    for _, row in long_folds.sort_values("fold").iterrows():
        test_start = pd.Timestamp(row["test_start"])
        if base_starts.tz is not None and test_start.tzinfo is None:
            test_start = test_start.tz_localize(base_starts.tz)
        if base_starts.tz is None and test_start.tzinfo is not None:
            test_start = test_start.tz_convert("UTC").tz_localize(None)
        source_pos = int(base_starts.searchsorted(test_start, side="right"))
        source_pos = max(1, min(source_pos, len(base_starts)))
        source_fold = int(base_selections.iloc[source_pos - 1]["fold"])

        p9 = p9_by_fold.loc[source_fold].copy()
        p9["fold"] = int(row["fold"])
        p9["train_start"] = row["train_start"]
        p9["train_end"] = row["train_end"]
        p9["test_start"] = row["test_start"]
        p9["test_end"] = row["test_end"]
        p9["test_decision_end"] = p3d.decision_end_for_closed_positions(pd.Timestamp(row["test_end"]), horizon)
        p9_rows.append(p9.to_dict())

        p16 = p16_by_fold.loc[source_fold].copy()
        p16["fold"] = int(row["fold"])
        p16_rows.append(p16.to_dict())

        base = base_by_fold.loc[source_fold]
        mapping_rows.append(
            {
                "long_fold": int(row["fold"]),
                "long_test_start": row["test_start"],
                "long_test_end": row["test_end"],
                "mapped_source_fold": source_fold,
                "source_test_start": base["test_start"],
                "source_test_end": base["test_end"],
                "mapping_rule": "latest locked source fold not after long test_start; pre-2024-07 uses source fold 1",
            }
        )
    selected_p9 = pd.DataFrame(p9_rows).sort_values("fold").reset_index(drop=True)
    selected_p16 = pd.DataFrame(p16_rows).sort_values("fold").reset_index(drop=True)
    mapping = pd.DataFrame(mapping_rows)
    return selected_p9, selected_p16, mapping


def build_segment_summary(
    equity: pd.DataFrame,
    horizon: int,
    cost_bps: list[float],
    original_first_test: pd.Timestamp,
) -> pd.DataFrame:
    if equity.empty:
        return pd.DataFrame()
    data = equity.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    segments = {
        "full_long_history": data,
        "pre_original_2023_2024H1": data[data["timestamp"] < original_first_test],
        "original_window_2024_07_plus": data[data["timestamp"] >= original_first_test],
    }
    rows: list[dict[str, Any]] = []
    for segment, frame in segments.items():
        if frame.empty:
            continue
        for bps in cost_bps:
            rows.append({"segment": segment, "one_way_cost_bps": bps, **calc_return_metrics(frame, horizon, bps)})
    return pd.DataFrame(rows)


def build_fold_summary(equity: pd.DataFrame, selections: pd.DataFrame, horizon: int, cost_bps: float) -> pd.DataFrame:
    if equity.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    data = equity.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    for _, fold in selections.iterrows():
        frame = data[data["fold"] == int(fold["fold"])]
        if frame.empty:
            continue
        rows.append(
            {
                "fold": int(fold["fold"]),
                "test_start": fold["test_start"],
                "test_end": fold["test_end"],
                **calc_return_metrics(frame, horizon, cost_bps),
            }
        )
    return pd.DataFrame(rows)


def calc_return_metrics(frame: pd.DataFrame, horizon: int, cost_bps: float) -> dict[str, float]:
    gross = frame["gross_period_return"].astype(float)
    turnover = frame["turnover"].astype(float)
    net = gross - turnover * float(cost_bps) / 10000.0
    eq = (1.0 + net).cumprod()
    gross_eq = (1.0 + gross).cumprod()
    ann = 365.0 * 24.0 / float(horizon)
    sharpe = float(net.mean() / net.std(ddof=0) * np.sqrt(ann)) if net.std(ddof=0) > 0 else 0.0
    gross_sharpe = float(gross.mean() / gross.std(ddof=0) * np.sqrt(ann)) if gross.std(ddof=0) > 0 else 0.0
    dd = eq / eq.cummax() - 1.0
    gross_dd = gross_eq / gross_eq.cummax() - 1.0
    return {
        "periods": float(len(frame)),
        "total_return": float(eq.iloc[-1] - 1.0),
        "gross_total_return": float(gross_eq.iloc[-1] - 1.0),
        "sharpe": sharpe,
        "gross_sharpe": gross_sharpe,
        "max_drawdown": float(dd.min()),
        "gross_max_drawdown": float(gross_dd.min()),
        "mean_turnover": float(turnover.mean()),
        "total_turnover": float(turnover.sum()),
        "positive_period_rate": float((net > 0).mean()),
    }


def plot_cost_curve(path: Path, equity: pd.DataFrame, cost_bps: float, title: str) -> None:
    if equity.empty:
        return
    data = equity.sort_values("timestamp").copy()
    gross_ret = data["gross_period_return"].astype(float)
    net_ret = gross_ret - data["turnover"].astype(float) * cost_bps / 10000.0
    gross = (1.0 + gross_ret).cumprod() - 1.0
    net = (1.0 + net_ret).cumprod() - 1.0
    plot_data = pd.concat(
        [
            pd.DataFrame({"timestamp": [data["timestamp"].iloc[0]], "net": [0.0], "gross": [0.0]}),
            pd.DataFrame({"timestamp": data["timestamp"], "net": net, "gross": gross}),
        ],
        ignore_index=True,
    )
    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.plot(plot_data["timestamp"], plot_data["net"], label=f"net after {cost_bps:.0f}bps one-way cost", linewidth=2.0)
    ax.plot(plot_data["timestamp"], plot_data["gross"], label="gross before cost", linewidth=1.4, alpha=0.75)
    ax.axvline(pd.Timestamp(ORIGINAL_FIRST_TEST), color="black", linestyle="--", linewidth=1.0, alpha=0.5)
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


def write_report(
    output_dir: Path,
    panel_path: Path,
    panel: pd.DataFrame,
    selections: pd.DataFrame,
    mapping: pd.DataFrame,
    metrics: dict[str, Any],
    cost_summary: pd.DataFrame,
    segment_summary: pd.DataFrame,
    fold_summary: pd.DataFrame,
) -> None:
    cost_lines = []
    for row in cost_summary.itertuples(index=False):
        cost_lines.append(
            f"- {float(row.one_way_cost_bps):.0f}bps: return {float(row.total_return):.2%}, "
            f"Sharpe {float(row.sharpe):.3f}, maxDD {float(row.max_drawdown):.2%}"
        )
    seg15 = segment_summary[segment_summary["one_way_cost_bps"] == 15.0].copy()
    seg_lines = []
    for row in seg15.itertuples(index=False):
        seg_lines.append(
            f"- {row.segment}: net {float(row.total_return):.2%}, gross {float(row.gross_total_return):.2%}, "
            f"Sharpe {float(row.sharpe):.3f}, maxDD {float(row.max_drawdown):.2%}, "
            f"mean_turnover {float(row.mean_turnover):.3f}"
        )
    positive_folds = float((fold_summary["total_return"] > 0.0).mean()) if not fold_summary.empty else 0.0
    worst_fold = fold_summary.sort_values("total_return").head(1)
    worst_line = ""
    if not worst_fold.empty:
        wf = worst_fold.iloc[0]
        worst_line = (
            f"- worst 15bps fold: fold {int(wf['fold'])}, {wf['test_start']} to {wf['test_end']}, "
            f"return {float(wf['total_return']):.2%}, maxDD {float(wf['max_drawdown']):.2%}"
        )
    gross_total = float(metrics.get("gross_total_return_before_cost", metrics.get("gross_total_return", 0.0)))
    text = f"""# Long History Fixed P16 Replay

This run tests the locked P16/P15c baseline on the extended common 20-symbol history.

Important interpretation note:

- This is a fixed-policy long-history stress replay.
- Folds before the original locked selection window map to source fold 1, because those periods were not part of the original nested AlphaGPT/P15c parameter-selection process.
- Therefore this is useful for stability and regime stress, but it is not a fresh strict nested parameter-selection OOS result for the added 2023-2024H1 history.

Inputs:

- panel: `{panel_path}`
- first panel timestamp: `{panel['timestamp'].min()}`
- last panel timestamp: `{panel['timestamp'].max()}`
- symbols: `{panel['symbol'].nunique()}`
- rows: `{len(panel)}`

Stress folds:

- first test start: `{selections['test_start'].min()}`
- last test end: `{selections['test_end'].max()}`
- folds: `{len(selections)}`
- pre-original folds using mapped source fold 1: `{int((pd.to_datetime(mapping['long_test_start']) < pd.Timestamp(ORIGINAL_FIRST_TEST)).sum())}`

Default 6bps metrics:

- total_return: {float(metrics.get("total_return", 0.0)):.4%}
- gross_total_return_before_cost: {gross_total:.4%}
- sharpe: {float(metrics.get("sharpe", 0.0)):.3f}
- max_drawdown: {float(metrics.get("max_drawdown", 0.0)):.4%}
- mean_turnover: {float(metrics.get("mean_turnover", 0.0)):.4f}

Cost sensitivity:

{chr(10).join(cost_lines)}

15bps segment summary:

{chr(10).join(seg_lines)}

Fold stability at 15bps:

- positive fold rate: {positive_folds:.2%}
{worst_line}
"""
    (output_dir / "REPORT.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
