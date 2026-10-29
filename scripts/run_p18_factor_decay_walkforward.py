#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "baseline_locked_v1"
LONG_ROOT = EXPERIMENT_ROOT / "outputs" / "long_history_nested_oos"
DEFAULT_ASSET_PANEL = EXPERIMENT_ROOT / "inputs" / "long_history" / "baseline_panel" / "asset_residual_panel.csv"
DEFAULT_MANUAL_PANEL = LONG_ROOT / "inputs" / "manual_slow_feature_panel_long.csv.gz"
DEFAULT_P6B_PANEL = LONG_ROOT / "inputs" / "p6b_feature_panel_long.csv.gz"
DEFAULT_DERIVATIVES_PANEL = (
    EXPERIMENT_ROOT / "inputs" / "long_history" / "binance_derivatives" / "derivatives_hourly_panel.csv.gz"
)
DEFAULT_FOLDS = LONG_ROOT / "inputs" / "long_formula_selections.csv"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "outputs" / "p18_factor_decay_walkforward"


LABEL_HORIZONS = [48, 72, 96, 168]
ID_COLS = ["timestamp", "symbol"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward factor validity and decay diagnostics.")
    parser.add_argument("--asset-panel", default=str(DEFAULT_ASSET_PANEL))
    parser.add_argument("--manual-panel", default=str(DEFAULT_MANUAL_PANEL))
    parser.add_argument("--p6b-panel", default=str(DEFAULT_P6B_PANEL))
    parser.add_argument("--derivatives-panel", default=str(DEFAULT_DERIVATIVES_PANEL))
    parser.add_argument("--folds", default=str(DEFAULT_FOLDS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--validation-days", type=int, default=90)
    parser.add_argument("--min-symbols", type=int, default=8)
    parser.add_argument("--min-folds", type=int, default=16)
    parser.add_argument("--max-factors", type=int, default=0, help="Optional debug cap; 0 means all factors.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Loading factor library", flush=True)
    panel, factor_meta = load_factor_library(
        asset_panel=Path(args.asset_panel),
        manual_panel=Path(args.manual_panel),
        p6b_panel=Path(args.p6b_panel),
        derivatives_panel=Path(args.derivatives_panel),
        max_factors=int(args.max_factors),
    )
    factor_meta.to_csv(output_dir / "factor_library.csv", index=False)
    print(f"      rows={len(panel):,}, factors={len(factor_meta):,}", flush=True)

    print("[2/5] Loading walk-forward folds", flush=True)
    folds = pd.read_csv(
        args.folds,
        parse_dates=["train_start", "train_end", "test_start", "test_end"],
    ).sort_values("fold")
    folds = add_fold_windows(folds, int(args.validation_days))

    print("[3/5] Precomputing label cross-sectional rank matrices", flush=True)
    label_rank_matrices = precompute_label_rank_matrices(panel, LABEL_HORIZONS, int(args.min_symbols))

    print("[4/5] Running factor decay diagnostics", flush=True)
    fold_metrics = run_factor_diagnostics(
        panel=panel,
        factors=factor_meta["factor"].tolist(),
        factor_meta=factor_meta,
        label_rank_matrices=label_rank_matrices,
        folds=folds,
        min_symbols=int(args.min_symbols),
    )
    fold_metrics.to_csv(output_dir / "factor_fold_metrics.csv", index=False)

    print("[5/5] Summarizing factor lifecycle", flush=True)
    summary = summarize_factor_metrics(fold_metrics, min_folds=int(args.min_folds))
    summary.to_csv(output_dir / "factor_decay_summary.csv", index=False)
    period_summary = summarize_by_period(fold_metrics)
    period_summary.to_csv(output_dir / "factor_period_summary.csv", index=False)
    retained = build_retained_candidates(summary)
    retained.to_csv(output_dir / "retained_factor_candidates.csv", index=False)
    write_report(output_dir, factor_meta, summary, period_summary, retained)
    plot_top_factors(output_dir, fold_metrics, summary)
    print(f"Done: {output_dir}", flush=True)


def load_factor_library(
    asset_panel: Path,
    manual_panel: Path,
    p6b_panel: Path,
    derivatives_panel: Path,
    max_factors: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    label_cols = [f"label_resid_ret_h{h}" for h in LABEL_HORIZONS]
    asset_cols = pd.read_csv(asset_panel, nrows=0).columns.tolist()
    base_factor_cols = [col for col in asset_cols if is_base_factor_col(col)]
    asset = pd.read_csv(asset_panel, parse_dates=["timestamp"], usecols=ID_COLS + label_cols + base_factor_cols)
    asset = asset.sort_values(ID_COLS).reset_index(drop=True)
    panel = asset[ID_COLS + label_cols].copy()
    meta_rows: list[dict[str, Any]] = []

    panel, meta_rows = append_factor_block(panel, asset[ID_COLS + base_factor_cols], "base", meta_rows)
    manual = read_factor_block(manual_panel, "slow")
    panel, meta_rows = append_factor_block(panel, manual, "slow", meta_rows)
    p6b = read_factor_block(p6b_panel, "p6b")
    panel, meta_rows = append_factor_block(panel, p6b, "p6b", meta_rows)
    deriv = read_factor_block(derivatives_panel, "deriv")
    panel, meta_rows = append_factor_block(panel, deriv, "deriv", meta_rows)

    factor_meta = pd.DataFrame(meta_rows)
    factor_cols = factor_meta["factor"].tolist()
    if max_factors > 0:
        factor_cols = factor_cols[:max_factors]
        factor_meta = factor_meta[factor_meta["factor"].isin(factor_cols)].reset_index(drop=True)
        panel = panel[ID_COLS + label_cols + factor_cols].copy()

    factor_meta = drop_low_information_factors(panel, factor_meta)
    keep = ID_COLS + label_cols + factor_meta["factor"].tolist()
    return panel[keep].copy(), factor_meta.reset_index(drop=True)


def is_base_factor_col(col: str) -> bool:
    blocked_prefixes = ("fwd_", "label_", "factor_")
    blocked = {
        "timestamp",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "log_close",
        "predicted_log_ret",
        "beta_intercept",
    }
    if col in blocked or col.startswith(blocked_prefixes):
        return False
    explicit = {
        "volume",
        "quote_volume",
        "trades",
        "taker_buy_base",
        "taker_buy_quote",
        "log_quote_volume",
        "log_ret",
        "simple_ret",
        "residual_log_ret",
        "resid_z",
        "resid_mom_short",
        "resid_mom_medium",
        "raw_mom_short",
        "raw_mom_medium",
        "realized_vol_medium",
        "realized_vol_long",
        "idio_vol_medium",
        "idio_vol_long",
        "liquidity_z",
        "turnover_pressure",
        "beta_btc",
        "beta_eth",
        "beta_market",
        "beta_sector",
        "score_baseline",
    }
    return col in explicit or col.startswith("cs_rank_") or col.startswith("cs_zscore_")


def read_factor_block(path: Path, source: str) -> pd.DataFrame:
    cols = pd.read_csv(path, nrows=0).columns.tolist()
    factor_cols = [col for col in cols if col not in ID_COLS]
    frame = pd.read_csv(path, parse_dates=["timestamp"], usecols=ID_COLS + factor_cols)
    return frame


def append_factor_block(
    panel: pd.DataFrame,
    block: pd.DataFrame,
    source: str,
    meta_rows: list[dict[str, Any]],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rename: dict[str, str] = {}
    for col in block.columns:
        if col in ID_COLS:
            continue
        if not pd.api.types.is_numeric_dtype(block[col]):
            continue
        safe = f"{source}__{col}"
        rename[col] = safe
        meta_rows.append({"factor": safe, "source": source, "raw_name": col})
    if not rename:
        return panel, meta_rows
    block = block[ID_COLS + list(rename)].rename(columns=rename)
    panel = panel.merge(block, on=ID_COLS, how="left")
    return panel, meta_rows


def drop_low_information_factors(panel: pd.DataFrame, factor_meta: pd.DataFrame) -> pd.DataFrame:
    keep: list[str] = []
    for factor in factor_meta["factor"]:
        x = panel[factor]
        finite = np.isfinite(x.to_numpy(dtype=float, copy=False))
        if finite.mean() < 0.25:
            continue
        by_ts_std = panel.loc[finite, ["timestamp", factor]].groupby("timestamp")[factor].std(ddof=0)
        if by_ts_std.replace([np.inf, -np.inf], np.nan).dropna().gt(1e-12).mean() < 0.20:
            continue
        keep.append(factor)
    return factor_meta[factor_meta["factor"].isin(keep)].copy()


def add_fold_windows(folds: pd.DataFrame, validation_days: int) -> pd.DataFrame:
    out = folds.copy()
    out["validation_start"] = out["train_end"] - pd.Timedelta(days=validation_days)
    out["validation_start"] = out[["validation_start", "train_start"]].max(axis=1)
    out["fold_period"] = np.where(out["test_start"] < pd.Timestamp("2024-07-01", tz="UTC"), "pre_2024_07", "post_2024_07")
    return out


def precompute_label_rank_matrices(panel: pd.DataFrame, horizons: list[int], min_symbols: int) -> dict[int, pd.DataFrame]:
    label_rank_matrices: dict[int, pd.DataFrame] = {}
    for horizon in horizons:
        label = f"label_resid_ret_h{horizon}"
        label_rank_matrices[horizon] = centered_rank_matrix(panel, label, min_symbols)
    return label_rank_matrices


def centered_rank_by_timestamp(panel: pd.DataFrame, col: str, min_symbols: int) -> pd.Series:
    grouped = panel.groupby("timestamp", sort=False)[col]
    count = grouped.transform("count")
    rank = grouped.rank(method="average")
    centered = 2.0 * (rank - 1.0) / (count - 1.0) - 1.0
    return centered.where(count >= min_symbols)


def run_factor_diagnostics(
    panel: pd.DataFrame,
    factors: list[str],
    factor_meta: pd.DataFrame,
    label_rank_matrices: dict[int, pd.DataFrame],
    folds: pd.DataFrame,
    min_symbols: int,
) -> pd.DataFrame:
    meta_by_factor = factor_meta.set_index("factor").to_dict("index")
    rows: list[dict[str, Any]] = []
    total = len(factors)
    for idx, factor in enumerate(factors, start=1):
        if idx == 1 or idx % 25 == 0 or idx == total:
            print(f"      factor {idx}/{total}: {factor}", flush=True)
        x_rank = centered_rank_matrix(panel, factor, min_symbols)
        turnover_1h, turnover_72h, autocorr_72h = factor_persistence_by_matrix(x_rank)
        for horizon, y_rank in label_rank_matrices.items():
            ic_ts = rowwise_corr(x_rank, y_rank, min_symbols)
            for _, fold in folds.iterrows():
                rows.append(
                    build_fold_factor_row(
                        fold=fold,
                        factor=factor,
                        source=meta_by_factor[factor]["source"],
                        raw_name=meta_by_factor[factor]["raw_name"],
                        horizon=horizon,
                        ic_ts=ic_ts,
                        turnover_1h=turnover_1h,
                        turnover_72h=turnover_72h,
                        autocorr_72h=autocorr_72h,
                    )
                )
    return pd.DataFrame(rows)


def centered_rank_matrix(panel: pd.DataFrame, col: str, min_symbols: int) -> pd.DataFrame:
    wide = panel.pivot(index="timestamp", columns="symbol", values=col).sort_index()
    count = wide.count(axis=1)
    rank = wide.rank(axis=1, method="average", na_option="keep")
    centered = 2.0 * (rank.sub(1.0, axis=0)).div(count.sub(1.0), axis=0) - 1.0
    return centered.where(count >= min_symbols)


def factor_persistence_by_matrix(wide: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    turnover_1h = (wide - wide.shift(1)).abs().mean(axis=1).rename("turnover_1h")
    turnover_72h = (wide - wide.shift(72)).abs().mean(axis=1).rename("turnover_72h")
    autocorr_72h = rowwise_corr(wide, wide.shift(72), min_symbols=8).rename("autocorr_72h")
    return turnover_1h, turnover_72h, autocorr_72h


def rowwise_corr(x: pd.DataFrame, y: pd.DataFrame, min_symbols: int) -> pd.Series:
    x, y = x.align(y, join="inner", axis=0)
    x, y = x.align(y, join="inner", axis=1)
    valid = x.notna() & y.notna()
    n = valid.sum(axis=1).astype(float)
    x0 = x.where(valid)
    y0 = y.where(valid)
    sx = x0.sum(axis=1)
    sy = y0.sum(axis=1)
    sxx = (x0 * x0).sum(axis=1)
    syy = (y0 * y0).sum(axis=1)
    sxy = (x0 * y0).sum(axis=1)
    cov = sxy - sx * sy / n
    vx = sxx - sx * sx / n
    vy = syy - sy * sy / n
    corr = cov / np.sqrt(vx * vy)
    return corr.where((n >= min_symbols) & np.isfinite(corr)).dropna()


def build_fold_factor_row(
    fold: pd.Series,
    factor: str,
    source: str,
    raw_name: str,
    horizon: int,
    ic_ts: pd.Series,
    turnover_1h: pd.Series,
    turnover_72h: pd.Series,
    autocorr_72h: pd.Series,
) -> dict[str, Any]:
    train_start = pd.Timestamp(fold["train_start"])
    validation_start = pd.Timestamp(fold["validation_start"])
    train_end = pd.Timestamp(fold["train_end"])
    train_ic = ic_ts[(ic_ts.index >= train_start) & (ic_ts.index < validation_start)]
    val_ic = ic_ts[(ic_ts.index >= validation_start) & (ic_ts.index <= train_end)]
    sign = orient_sign(train_ic)
    oriented_train = sign * train_ic
    oriented_val = sign * val_ic
    val_first, val_second = split_first_second(oriented_val)
    val_turn1 = turnover_1h[(turnover_1h.index >= validation_start) & (turnover_1h.index <= train_end)]
    val_turn72 = turnover_72h[(turnover_72h.index >= validation_start) & (turnover_72h.index <= train_end)]
    val_ac72 = autocorr_72h[(autocorr_72h.index >= validation_start) & (autocorr_72h.index <= train_end)]
    return {
        "fold": int(fold["fold"]),
        "fold_period": fold["fold_period"],
        "test_start": fold["test_start"],
        "factor": factor,
        "source": source,
        "raw_name": raw_name,
        "horizon": int(horizon),
        "orientation": int(sign),
        "train_ic_mean": safe_mean(train_ic),
        "train_oriented_ic_mean": safe_mean(oriented_train),
        "train_ts_count": int(train_ic.count()),
        "validation_ic_mean": safe_mean(val_ic),
        "validation_oriented_ic_mean": safe_mean(oriented_val),
        "validation_ic_q25": safe_quantile(oriented_val, 0.25),
        "validation_positive_ts_rate": safe_positive_rate(oriented_val),
        "validation_ts_count": int(val_ic.count()),
        "validation_first_half_ic": safe_mean(val_first),
        "validation_second_half_ic": safe_mean(val_second),
        "validation_decay_delta": safe_mean(val_second) - safe_mean(val_first),
        "validation_turnover_1h": safe_mean(val_turn1),
        "validation_turnover_72h": safe_mean(val_turn72),
        "validation_autocorr_72h": safe_mean(val_ac72),
    }


def orient_sign(train_ic: pd.Series) -> int:
    mean = safe_mean(train_ic)
    if not np.isfinite(mean) or abs(mean) < 1e-12:
        return 1
    return 1 if mean >= 0 else -1


def split_first_second(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    if series.empty:
        return series, series
    midpoint = len(series) // 2
    return series.iloc[:midpoint], series.iloc[midpoint:]


def safe_mean(series: pd.Series) -> float:
    if series is None or series.empty:
        return np.nan
    return float(series.replace([np.inf, -np.inf], np.nan).dropna().mean())


def safe_quantile(series: pd.Series, q: float) -> float:
    if series is None or series.empty:
        return np.nan
    return float(series.replace([np.inf, -np.inf], np.nan).dropna().quantile(q))


def safe_positive_rate(series: pd.Series) -> float:
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return np.nan
    return float((clean > 0).mean())


def summarize_factor_metrics(fold_metrics: pd.DataFrame, min_folds: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["factor", "source", "raw_name", "horizon"]
    for keys, group in fold_metrics.groupby(group_cols, sort=False):
        clean = group.dropna(subset=["validation_oriented_ic_mean"]).sort_values("fold")
        if clean.empty:
            continue
        x = np.arange(len(clean), dtype=float)
        y = clean["validation_oriented_ic_mean"].to_numpy(dtype=float)
        slope = float(np.polyfit(x, y, 1)[0]) if len(clean) >= 3 and np.isfinite(y).all() else np.nan
        first_half = clean.iloc[: max(1, len(clean) // 2)]["validation_oriented_ic_mean"]
        second_half = clean.iloc[len(clean) // 2 :]["validation_oriented_ic_mean"]
        score = generalization_score(clean, slope)
        rows.append(
            {
                "factor": keys[0],
                "source": keys[1],
                "raw_name": keys[2],
                "horizon": keys[3],
                "fold_count": int(len(clean)),
                "train_ic_mean": clean["train_ic_mean"].mean(),
                "validation_oriented_ic_mean": clean["validation_oriented_ic_mean"].mean(),
                "validation_oriented_ic_median": clean["validation_oriented_ic_mean"].median(),
                "validation_oriented_ic_q25_fold": clean["validation_oriented_ic_mean"].quantile(0.25),
                "positive_fold_rate": float((clean["validation_oriented_ic_mean"] > 0).mean()),
                "mean_validation_ts_positive_rate": clean["validation_positive_ts_rate"].mean(),
                "mean_validation_ic_q25_ts": clean["validation_ic_q25"].mean(),
                "mean_turnover_1h": clean["validation_turnover_1h"].mean(),
                "mean_turnover_72h": clean["validation_turnover_72h"].mean(),
                "mean_autocorr_72h": clean["validation_autocorr_72h"].mean(),
                "first_half_fold_ic": first_half.mean(),
                "second_half_fold_ic": second_half.mean(),
                "recent_minus_early_ic": second_half.mean() - first_half.mean(),
                "fold_ic_slope": slope,
                "generalization_score": score,
                "retained_p18a": bool(
                    len(clean) >= min_folds
                    and clean["validation_oriented_ic_mean"].mean() > 0.0025
                    and clean["validation_oriented_ic_mean"].quantile(0.25) > -0.015
                    and (clean["validation_oriented_ic_mean"] > 0).mean() >= 0.55
                    and clean["validation_autocorr_72h"].mean() >= 0.35
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("generalization_score", ascending=False)


def generalization_score(group: pd.DataFrame, slope: float) -> float:
    mean_ic = group["validation_oriented_ic_mean"].mean()
    q25 = group["validation_oriented_ic_mean"].quantile(0.25)
    pos = (group["validation_oriented_ic_mean"] > 0).mean()
    ac72 = group["validation_autocorr_72h"].mean()
    turn72 = group["validation_turnover_72h"].mean()
    decay_penalty = max(0.0, -slope * 32.0) if np.isfinite(slope) else 0.0
    return float(
        100.0 * mean_ic
        + 60.0 * q25
        + 8.0 * (pos - 0.5)
        + 1.5 * np.nan_to_num(ac72, nan=0.0)
        - 0.8 * np.nan_to_num(turn72, nan=1.0)
        - 50.0 * decay_penalty
    )


def summarize_by_period(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["factor", "source", "raw_name", "horizon", "fold_period"]
    for keys, group in fold_metrics.groupby(group_cols, sort=False):
        clean = group.dropna(subset=["validation_oriented_ic_mean"])
        if clean.empty:
            continue
        rows.append(
            {
                "factor": keys[0],
                "source": keys[1],
                "raw_name": keys[2],
                "horizon": keys[3],
                "fold_period": keys[4],
                "fold_count": int(len(clean)),
                "validation_oriented_ic_mean": clean["validation_oriented_ic_mean"].mean(),
                "validation_oriented_ic_q25_fold": clean["validation_oriented_ic_mean"].quantile(0.25),
                "positive_fold_rate": float((clean["validation_oriented_ic_mean"] > 0).mean()),
                "mean_turnover_72h": clean["validation_turnover_72h"].mean(),
                "mean_autocorr_72h": clean["validation_autocorr_72h"].mean(),
            }
        )
    return pd.DataFrame(rows)


def build_retained_candidates(summary: pd.DataFrame) -> pd.DataFrame:
    retained = summary[summary["retained_p18a"]].copy()
    if retained.empty:
        return retained
    retained["decay_flag"] = retained["recent_minus_early_ic"] < -0.01
    retained["low_turnover_flag"] = retained["mean_autocorr_72h"] >= retained["mean_autocorr_72h"].median()
    retained["priority"] = np.where(
        (~retained["decay_flag"]) & retained["low_turnover_flag"],
        "core_candidate",
        "watchlist",
    )
    return retained.sort_values(["priority", "generalization_score"], ascending=[True, False])


def write_report(
    output_dir: Path,
    factor_meta: pd.DataFrame,
    summary: pd.DataFrame,
    period_summary: pd.DataFrame,
    retained: pd.DataFrame,
) -> None:
    top_cols = [
        "factor",
        "source",
        "raw_name",
        "horizon",
        "validation_oriented_ic_mean",
        "validation_oriented_ic_q25_fold",
        "positive_fold_rate",
        "mean_turnover_72h",
        "mean_autocorr_72h",
        "recent_minus_early_ic",
        "generalization_score",
        "retained_p18a",
    ]
    retained_display = retained[top_cols + ["priority"]] if not retained.empty else retained
    report = [
        "# P18 Factor Decay Walk-Forward Audit",
        "",
        "Purpose: audit whether previously considered factors retain fold-local predictive value, decay over time, and have slow enough turnover to support better OOS generalization.",
        "",
        "Important: this is a retrospective research audit. A tradable model must still choose factors inside a nested pipeline and keep a final holdout untouched.",
        "",
        f"- factor count after information filter: {len(factor_meta):,}",
        f"- retained P18a factor-horizon candidates: {len(retained):,}",
        "",
        "## Top Generalization Candidates",
        "",
        "```text",
        summary[top_cols].head(40).to_string(index=False),
        "```",
        "",
        "## Retained Candidates",
        "",
        "```text",
        retained_display.head(80).to_string(index=False) if not retained_display.empty else "None",
        "```",
        "",
        "## Most Decayed Candidates",
        "",
        "```text",
        summary.sort_values("recent_minus_early_ic").head(30)[top_cols].to_string(index=False),
        "```",
        "",
        "## Period Contrast",
        "",
        "```text",
        build_period_contrast(period_summary).head(40).to_string(index=False),
        "```",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    metadata = {
        "factor_count": int(len(factor_meta)),
        "retained_count": int(len(retained)),
        "label_horizons": LABEL_HORIZONS,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def build_period_contrast(period_summary: pd.DataFrame) -> pd.DataFrame:
    if period_summary.empty:
        return period_summary
    pivot = period_summary.pivot_table(
        index=["factor", "source", "raw_name", "horizon"],
        columns="fold_period",
        values="validation_oriented_ic_mean",
        aggfunc="mean",
    ).reset_index()
    for col in ["pre_2024_07", "post_2024_07"]:
        if col not in pivot:
            pivot[col] = np.nan
    pivot["post_minus_pre"] = pivot["post_2024_07"] - pivot["pre_2024_07"]
    return pivot.sort_values("post_minus_pre", ascending=False)


def plot_top_factors(output_dir: Path, fold_metrics: pd.DataFrame, summary: pd.DataFrame) -> None:
    top = summary.head(12)[["factor", "horizon"]]
    if top.empty:
        return
    plot_rows = fold_metrics.merge(top, on=["factor", "horizon"], how="inner")
    plot_rows["label"] = plot_rows["raw_name"] + " h" + plot_rows["horizon"].astype(str)
    pivot = plot_rows.pivot_table(index="fold", columns="label", values="validation_oriented_ic_mean", aggfunc="mean")
    ax = pivot.rolling(3, min_periods=1).mean().plot(figsize=(15, 8), linewidth=1.4)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("P18 top factor rolling fold validation IC")
    ax.set_xlabel("Walk-forward fold")
    ax.set_ylabel("Oriented validation IC")
    ax.legend(fontsize=8, ncol=2)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(output_dir / "top_factor_fold_ic.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
