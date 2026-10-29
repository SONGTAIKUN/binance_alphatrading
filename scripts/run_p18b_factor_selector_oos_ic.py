#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_p18_factor_decay_walkforward import (
    DEFAULT_ASSET_PANEL,
    DEFAULT_DERIVATIVES_PANEL,
    DEFAULT_FOLDS,
    DEFAULT_MANUAL_PANEL,
    DEFAULT_P6B_PANEL,
    EXPERIMENT_ROOT,
    ID_COLS,
    LABEL_HORIZONS,
    centered_rank_matrix,
    load_factor_library,
    rowwise_corr,
)


DEFAULT_P18A = EXPERIMENT_ROOT / "outputs" / "p18_factor_decay_walkforward"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "outputs" / "p18b_factor_selector_oos_ic"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fold-local factor selector OOS IC test.")
    parser.add_argument("--asset-panel", default=str(DEFAULT_ASSET_PANEL))
    parser.add_argument("--manual-panel", default=str(DEFAULT_MANUAL_PANEL))
    parser.add_argument("--p6b-panel", default=str(DEFAULT_P6B_PANEL))
    parser.add_argument("--derivatives-panel", default=str(DEFAULT_DERIVATIVES_PANEL))
    parser.add_argument("--folds", default=str(DEFAULT_FOLDS))
    parser.add_argument("--p18a-dir", default=str(DEFAULT_P18A))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--min-validation-ic", type=float, default=0.005)
    parser.add_argument("--min-autocorr-72h", type=float, default=0.35)
    parser.add_argument("--max-turnover-72h", type=float, default=0.75)
    parser.add_argument("--min-symbols", type=int, default=8)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    p18a_dir = Path(args.p18a_dir)

    print("[1/4] Loading P18a fold metrics", flush=True)
    fold_metrics = pd.read_csv(p18a_dir / "factor_fold_metrics.csv", parse_dates=["test_start"])
    folds = pd.read_csv(args.folds, parse_dates=["train_start", "train_end", "test_start", "test_end"]).sort_values("fold")

    print("[2/4] Selecting factors fold-locally", flush=True)
    selected = select_factors_by_fold(
        fold_metrics,
        top_k=int(args.top_k),
        min_validation_ic=float(args.min_validation_ic),
        min_autocorr_72h=float(args.min_autocorr_72h),
        max_turnover_72h=float(args.max_turnover_72h),
    )
    selected.to_csv(output_dir / "selected_factors_by_fold.csv", index=False)

    print("[3/4] Loading factor panel and computing OOS IC", flush=True)
    panel, factor_meta = load_factor_library(
        asset_panel=Path(args.asset_panel),
        manual_panel=Path(args.manual_panel),
        p6b_panel=Path(args.p6b_panel),
        derivatives_panel=Path(args.derivatives_panel),
        max_factors=0,
    )
    needed_factors = sorted(set(selected["factor"]))
    score_ranks = {factor: centered_rank_matrix(panel, factor, int(args.min_symbols)) for factor in needed_factors}
    label_ranks = {
        horizon: centered_rank_matrix(panel, f"label_resid_ret_h{horizon}", int(args.min_symbols))
        for horizon in sorted(selected["horizon"].unique())
    }
    label_values = {
        horizon: panel.pivot(index="timestamp", columns="symbol", values=f"label_resid_ret_h{horizon}").sort_index()
        for horizon in sorted(selected["horizon"].unique())
    }

    fold_rows, ts_rows = run_oos_selector(
        selected=selected,
        folds=folds,
        score_ranks=score_ranks,
        label_ranks=label_ranks,
        label_values=label_values,
        min_symbols=int(args.min_symbols),
    )
    fold_df = pd.DataFrame(fold_rows)
    ts_df = pd.DataFrame(ts_rows)
    fold_df.to_csv(output_dir / "oos_selector_fold_metrics.csv", index=False)
    ts_df.to_csv(output_dir / "oos_selector_timestamp_metrics.csv", index=False)

    print("[4/4] Writing summary", flush=True)
    summary = summarize_oos(fold_df)
    summary.to_csv(output_dir / "oos_selector_summary.csv", index=False)
    write_report(output_dir, selected, fold_df, summary)
    plot_oos(output_dir, ts_df)
    metadata = {
        "top_k": int(args.top_k),
        "min_validation_ic": float(args.min_validation_ic),
        "min_autocorr_72h": float(args.min_autocorr_72h),
        "max_turnover_72h": float(args.max_turnover_72h),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Done: {output_dir}", flush=True)


def select_factors_by_fold(
    fold_metrics: pd.DataFrame,
    top_k: int,
    min_validation_ic: float,
    min_autocorr_72h: float,
    max_turnover_72h: float,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    data = fold_metrics.copy()
    data["selection_score"] = (
        100.0 * data["validation_oriented_ic_mean"].fillna(-1.0)
        + 50.0 * data["validation_ic_q25"].fillna(-1.0)
        + 6.0 * (data["validation_positive_ts_rate"].fillna(0.0) - 0.5)
        + 1.5 * data["validation_autocorr_72h"].fillna(0.0)
        - 0.8 * data["validation_turnover_72h"].fillna(1.0)
        + 20.0 * data["validation_decay_delta"].fillna(0.0).clip(lower=-0.05, upper=0.05)
    )
    keep = data[
        (data["validation_oriented_ic_mean"] >= min_validation_ic)
        & (data["validation_autocorr_72h"] >= min_autocorr_72h)
        & (data["validation_turnover_72h"] <= max_turnover_72h)
        & (data["validation_ts_count"] >= 200)
    ].copy()
    for (fold, horizon), group in keep.groupby(["fold", "horizon"], sort=True):
        group = group.sort_values("selection_score", ascending=False)
        # Avoid selecting the same economic variable multiple times from duplicated ranks/zscores.
        group = group.drop_duplicates(subset=["source", "raw_name"], keep="first").head(top_k)
        rows.append(group)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def run_oos_selector(
    selected: pd.DataFrame,
    folds: pd.DataFrame,
    score_ranks: dict[str, pd.DataFrame],
    label_ranks: dict[int, pd.DataFrame],
    label_values: dict[int, pd.DataFrame],
    min_symbols: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fold_by_id = folds.set_index("fold")
    fold_rows: list[dict[str, Any]] = []
    ts_rows: list[dict[str, Any]] = []
    for (fold_id, horizon), group in selected.groupby(["fold", "horizon"], sort=True):
        fold = fold_by_id.loc[int(fold_id)]
        start = pd.Timestamp(fold["test_start"])
        end = pd.Timestamp(fold["test_end"])
        combined = None
        weight_sum = 0.0
        for _, row in group.iterrows():
            factor = row["factor"]
            orientation = float(row["orientation"])
            weight = max(float(row["selection_score"]), 0.01)
            matrix = score_ranks[factor].loc[start:end] * orientation
            combined = matrix * weight if combined is None else combined.add(matrix * weight, fill_value=0.0)
            weight_sum += weight
        if combined is None or weight_sum <= 0:
            continue
        combined = combined / weight_sum
        y_rank = label_ranks[int(horizon)].loc[combined.index.min() : combined.index.max()]
        ic_ts = rowwise_corr(combined, y_rank, min_symbols)
        spread_ts = top_bottom_spread(combined, label_values[int(horizon)].loc[combined.index.min() : combined.index.max()])
        turnover_72h = (combined - combined.shift(72)).abs().mean(axis=1)
        autocorr_72h = rowwise_corr(combined, combined.shift(72), min_symbols)
        for ts, val in ic_ts.items():
            ts_rows.append(
                {
                    "timestamp": ts,
                    "fold": int(fold_id),
                    "horizon": int(horizon),
                    "oos_ic": float(val),
                    "top_bottom_spread": float(spread_ts.get(ts, np.nan)),
                    "turnover_72h": float(turnover_72h.get(ts, np.nan)),
                    "autocorr_72h": float(autocorr_72h.get(ts, np.nan)),
                    "selected_count": int(len(group)),
                }
            )
        fold_rows.append(
            {
                "fold": int(fold_id),
                "horizon": int(horizon),
                "test_start": start,
                "test_end": end,
                "selected_count": int(len(group)),
                "oos_ic_mean": float(ic_ts.mean()) if len(ic_ts) else np.nan,
                "oos_ic_q25_ts": float(ic_ts.quantile(0.25)) if len(ic_ts) else np.nan,
                "positive_ts_rate": float((ic_ts > 0).mean()) if len(ic_ts) else np.nan,
                "top_bottom_spread_mean": float(spread_ts.mean()) if len(spread_ts) else np.nan,
                "mean_turnover_72h": float(turnover_72h.mean()),
                "mean_autocorr_72h": float(autocorr_72h.mean()),
                "selected_factors": "|".join(group["raw_name"].astype(str).tolist()),
            }
        )
    return fold_rows, ts_rows


def top_bottom_spread(score: pd.DataFrame, label: pd.DataFrame, q: float = 0.2) -> pd.Series:
    score, label = score.align(label, join="inner", axis=0)
    score, label = score.align(label, join="inner", axis=1)
    values: dict[pd.Timestamp, float] = {}
    n_select = max(1, int(round(score.shape[1] * q)))
    for ts in score.index:
        s = score.loc[ts]
        y = label.loc[ts]
        valid = s.notna() & y.notna()
        if valid.sum() < max(2 * n_select, 4):
            values[ts] = np.nan
            continue
        ranked = s[valid].sort_values()
        bottom = ranked.index[:n_select]
        top = ranked.index[-n_select:]
        values[ts] = float(y[top].mean() - y[bottom].mean())
    return pd.Series(values).dropna()


def summarize_oos(fold_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for horizon, group in fold_df.groupby("horizon", sort=True):
        rows.append(
            {
                "horizon": int(horizon),
                "fold_count": int(len(group)),
                "mean_oos_ic": group["oos_ic_mean"].mean(),
                "median_oos_ic": group["oos_ic_mean"].median(),
                "q25_fold_oos_ic": group["oos_ic_mean"].quantile(0.25),
                "positive_fold_rate": float((group["oos_ic_mean"] > 0).mean()),
                "mean_top_bottom_spread": group["top_bottom_spread_mean"].mean(),
                "mean_turnover_72h": group["mean_turnover_72h"].mean(),
                "mean_autocorr_72h": group["mean_autocorr_72h"].mean(),
                "mean_selected_count": group["selected_count"].mean(),
            }
        )
    return pd.DataFrame(rows)


def write_report(output_dir: Path, selected: pd.DataFrame, fold_df: pd.DataFrame, summary: pd.DataFrame) -> None:
    report = [
        "# P18b Factor Selector OOS IC",
        "",
        "Each fold selects factors using only that fold's historical validation diagnostics, then evaluates the equal-weight oriented selected-factor score on the next test segment.",
        "",
        "## OOS Summary",
        "",
        "```text",
        summary.to_string(index=False),
        "```",
        "",
        "## Common Selected Factors",
        "",
        "```text",
        selected["raw_name"].value_counts().head(40).to_string(),
        "```",
        "",
        "## Weakest OOS Folds",
        "",
        "```text",
        fold_df.sort_values("oos_ic_mean").head(30).to_string(index=False),
        "```",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")


def plot_oos(output_dir: Path, ts_df: pd.DataFrame) -> None:
    if ts_df.empty:
        return
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    for horizon, group in ts_df.groupby("horizon", sort=True):
        by_ts = group.sort_values("timestamp").set_index("timestamp")
        axes[0].plot(by_ts.index, by_ts["oos_ic"].rolling(72, min_periods=12).mean(), label=f"h{horizon}")
        axes[1].plot(by_ts.index, by_ts["top_bottom_spread"].cumsum(), label=f"h{horizon}")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_title("P18b OOS IC, 72-period rolling mean")
    axes[0].set_ylabel("IC")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_title("P18b cumulative top-bottom residual label spread")
    axes[1].set_ylabel("Cumulative spread")
    axes[1].legend(ncol=4)
    fig.tight_layout()
    fig.savefig(output_dir / "p18b_oos_ic_and_spread.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
