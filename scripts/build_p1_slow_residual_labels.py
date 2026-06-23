#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from crypto_stat_arb.config import ensure_dir, load_config
from crypto_stat_arb.features import BETA_COLS, FACTOR_COLS
from run_turnover_experiments import to_jsonable


EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "baseline_locked_v1"
DEFAULT_CONFIG = EXPERIMENT_ROOT / "configs" / "baseline_locked_v1.yaml"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "outputs" / "p1_slow_labels"

MEDIUM_WEIGHTS = {48: 0.20, 72: 0.30, 96: 0.30, 168: 0.20}
LOW_TURNOVER_WEIGHTS = {72: 0.25, 96: 0.35, 168: 0.40}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--horizons", type=int, nargs="+", default=[48, 72, 96, 168])
    parser.add_argument("--vol-lookback", type=int, default=336)
    parser.add_argument("--path-lambda", type=float, default=0.50)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    output_dir = ensure_dir(args.output_dir)
    horizons = sorted(set(int(h) for h in args.horizons))
    execution_delay_bars = int(cfg["locked_baseline"].get("execution_delay_bars", 1))
    panel_path = resolve_config_path(config_path, cfg["inputs"]["panel"])

    print("[1/5] Loading baseline panel", flush=True)
    panel = pd.read_csv(panel_path, parse_dates=["timestamp"])
    panel = panel.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    validate_panel(panel)

    print("[2/5] Building ex-ante residual terminal and path-aware labels", flush=True)
    labels = build_slow_labels(
        panel=panel,
        horizons=horizons,
        execution_delay_bars=execution_delay_bars,
        vol_lookback=int(args.vol_lookback),
        path_lambda=float(args.path_lambda),
    )

    print("[3/5] Writing label panel", flush=True)
    label_path = output_dir / "slow_residual_labels.csv.gz"
    labels.to_csv(label_path, index=False)

    print("[4/5] Writing diagnostics", flush=True)
    coverage = build_coverage_summary(labels, horizons)
    coverage.to_csv(output_dir / "coverage_summary.csv", index=False)
    correlations = build_correlation_summary(labels)
    correlations.to_csv(output_dir / "label_correlations.csv", index=False)
    distribution = build_distribution_summary(labels)
    distribution.to_csv(output_dir / "label_distribution_summary.csv", index=False)
    write_metadata(
        output_dir=output_dir,
        cfg=cfg,
        panel_path=panel_path,
        label_path=label_path,
        horizons=horizons,
        execution_delay_bars=execution_delay_bars,
        vol_lookback=int(args.vol_lookback),
        path_lambda=float(args.path_lambda),
        rows=len(labels),
    )
    write_report(output_dir, coverage, correlations, distribution)

    print("[5/5] Done", flush=True)
    print(f"Done: {output_dir}", flush=True)


def resolve_config_path(config_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def validate_panel(panel: pd.DataFrame) -> None:
    required = {
        "timestamp",
        "symbol",
        "log_close",
        "log_ret",
        "residual_log_ret",
        *FACTOR_COLS,
        *BETA_COLS,
    }
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ValueError(f"Panel is missing required columns: {missing}")


def build_slow_labels(
    panel: pd.DataFrame,
    horizons: list[int],
    execution_delay_bars: int,
    vol_lookback: int,
    path_lambda: float,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    min_vol_obs = max(72, vol_lookback // 3)
    for symbol, df in panel.groupby("symbol", sort=False):
        print(f"      {symbol}", flush=True)
        df = df.sort_values("timestamp").copy()
        out = df[["timestamp", "symbol"]].copy()
        rv = np.sqrt(df["residual_log_ret"].pow(2).rolling(vol_lookback, min_periods=min_vol_obs).sum())
        out[f"ex_ante_resid_rv_{vol_lookback}h"] = rv

        for horizon in horizons:
            terminal = terminal_ex_ante_residual_return(df, horizon, execution_delay_bars)
            denominator = rv * np.sqrt(horizon / vol_lookback)
            path = residual_path_stats(df, horizon, execution_delay_bars)

            downside_adverse = (-path["min_resid_path"]).clip(lower=0.0)
            upside_adverse = path["max_resid_path"].clip(lower=0.0)
            signed_adverse = pd.Series(
                np.where(terminal >= 0.0, downside_adverse, upside_adverse),
                index=df.index,
            )
            terminal_norm = terminal / denominator.replace(0.0, np.nan)
            signed_adverse_norm = signed_adverse / denominator.replace(0.0, np.nan)
            downside_adverse_norm = downside_adverse / denominator.replace(0.0, np.nan)
            upside_adverse_norm = upside_adverse / denominator.replace(0.0, np.nan)

            path_signed_norm = pd.Series(
                np.where(
                    terminal >= 0.0,
                    terminal_norm - path_lambda * downside_adverse_norm,
                    terminal_norm + path_lambda * upside_adverse_norm,
                ),
                index=df.index,
            )
            path_long_norm = terminal_norm - path_lambda * downside_adverse_norm

            start = df["log_close"].shift(-execution_delay_bars)
            end = df["log_close"].shift(-(execution_delay_bars + horizon))
            fwd_log_ret = end - start

            out[f"fwd_log_ret_h{horizon}"] = fwd_log_ret
            out[f"fwd_simple_ret_h{horizon}"] = np.expm1(fwd_log_ret)
            out[f"label_resid_ret_h{horizon}"] = terminal
            out[f"label_resid_ret_norm_h{horizon}"] = terminal_norm
            out[f"label_path_signed_norm_h{horizon}"] = path_signed_norm
            out[f"label_path_long_norm_h{horizon}"] = path_long_norm
            out[f"label_path_min_resid_h{horizon}"] = path["min_resid_path"]
            out[f"label_path_max_resid_h{horizon}"] = path["max_resid_path"]
            out[f"label_downside_adverse_norm_h{horizon}"] = downside_adverse_norm
            out[f"label_upside_adverse_norm_h{horizon}"] = upside_adverse_norm
            out[f"label_signed_adverse_norm_h{horizon}"] = signed_adverse_norm

        frames.append(out)

    labels = pd.concat(frames, ignore_index=True).sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    add_cross_sectional_rank_labels(labels, horizons)
    add_composite_labels(labels)
    add_horizon_consistency(labels, horizons)
    return labels


def terminal_ex_ante_residual_return(df: pd.DataFrame, horizon: int, execution_delay_bars: int) -> pd.Series:
    start = df["log_close"].shift(-execution_delay_bars)
    end = df["log_close"].shift(-(execution_delay_bars + horizon))
    fwd_log_ret = end - start
    explained = pd.Series(0.0, index=df.index)
    for factor_col, beta_col in zip(FACTOR_COLS, BETA_COLS, strict=True):
        explained = explained + df[beta_col] * forward_sum(df[factor_col], execution_delay_bars + 1, horizon)
    return fwd_log_ret - explained


def residual_path_stats(df: pd.DataFrame, horizon: int, execution_delay_bars: int) -> dict[str, pd.Series]:
    cum_raw = pd.Series(0.0, index=df.index)
    cum_factors = {factor_col: pd.Series(0.0, index=df.index) for factor_col in FACTOR_COLS}
    min_path = pd.Series(np.inf, index=df.index)
    max_path = pd.Series(-np.inf, index=df.index)
    valid = pd.Series(True, index=df.index)

    for offset in range(execution_delay_bars + 1, execution_delay_bars + horizon + 1):
        raw = df["log_ret"].shift(-offset)
        cum_raw = cum_raw + raw.fillna(0.0)
        valid = valid & raw.notna()

        explained = pd.Series(0.0, index=df.index)
        for factor_col, beta_col in zip(FACTOR_COLS, BETA_COLS, strict=True):
            factor = df[factor_col].shift(-offset)
            cum_factors[factor_col] = cum_factors[factor_col] + factor.fillna(0.0)
            valid = valid & factor.notna()
            explained = explained + df[beta_col] * cum_factors[factor_col]

        resid_path = cum_raw - explained
        min_path = pd.Series(np.minimum(min_path.to_numpy(), resid_path.to_numpy()), index=df.index)
        max_path = pd.Series(np.maximum(max_path.to_numpy(), resid_path.to_numpy()), index=df.index)

    return {
        "min_resid_path": min_path.where(valid),
        "max_resid_path": max_path.where(valid),
    }


def forward_sum(series: pd.Series, start_offset: int, horizon: int) -> pd.Series:
    total = pd.Series(0.0, index=series.index)
    valid = pd.Series(True, index=series.index)
    for offset in range(start_offset, start_offset + horizon):
        shifted = series.shift(-offset)
        total = total + shifted.fillna(0.0)
        valid = valid & shifted.notna()
    return total.where(valid)


def add_cross_sectional_rank_labels(labels: pd.DataFrame, horizons: list[int]) -> None:
    for horizon in horizons:
        labels[f"label_resid_rank_centered_h{horizon}"] = centered_rank_by_timestamp(
            labels, f"label_resid_ret_norm_h{horizon}"
        )
        labels[f"label_path_rank_centered_h{horizon}"] = centered_rank_by_timestamp(
            labels, f"label_path_signed_norm_h{horizon}"
        )


def centered_rank_by_timestamp(frame: pd.DataFrame, value_col: str) -> pd.Series:
    grouped = frame.groupby("timestamp", sort=False)[value_col]
    rank = grouped.rank(method="average")
    count = grouped.transform("count")
    centered = 2.0 * (rank - 1.0) / (count - 1.0) - 1.0
    return centered.where(count > 1)


def add_composite_labels(labels: pd.DataFrame) -> None:
    labels["label_slow_medium"] = weighted_sum(labels, MEDIUM_WEIGHTS, "label_resid_rank_centered_h")
    labels["label_slow_low_turnover"] = weighted_sum(labels, LOW_TURNOVER_WEIGHTS, "label_resid_rank_centered_h")
    labels["label_path_slow_medium"] = weighted_sum(labels, MEDIUM_WEIGHTS, "label_path_rank_centered_h")
    labels["label_path_slow_low_turnover"] = weighted_sum(labels, LOW_TURNOVER_WEIGHTS, "label_path_rank_centered_h")

    labels["label_slow_medium_rank_centered"] = centered_rank_by_timestamp(labels, "label_slow_medium")
    labels["label_slow_low_turnover_rank_centered"] = centered_rank_by_timestamp(labels, "label_slow_low_turnover")
    labels["label_path_slow_medium_rank_centered"] = centered_rank_by_timestamp(labels, "label_path_slow_medium")
    labels["label_path_slow_low_turnover_rank_centered"] = centered_rank_by_timestamp(
        labels, "label_path_slow_low_turnover"
    )


def weighted_sum(labels: pd.DataFrame, weights: dict[int, float], prefix: str) -> pd.Series:
    total = pd.Series(0.0, index=labels.index)
    valid = pd.Series(True, index=labels.index)
    for horizon, weight in weights.items():
        col = f"{prefix}{horizon}"
        total = total + weight * labels[col].fillna(0.0)
        valid = valid & labels[col].notna()
    return total.where(valid)


def add_horizon_consistency(labels: pd.DataFrame, horizons: list[int]) -> None:
    resid_cols = [f"label_resid_rank_centered_h{horizon}" for horizon in horizons]
    path_cols = [f"label_path_rank_centered_h{horizon}" for horizon in horizons]
    labels["label_horizon_sign_consistency"] = pairwise_sign_agreement(labels[resid_cols])
    labels["label_path_horizon_sign_consistency"] = pairwise_sign_agreement(labels[path_cols])


def pairwise_sign_agreement(values: pd.DataFrame) -> pd.Series:
    arr = values.to_numpy(dtype=float)
    out = np.full(arr.shape[0], np.nan)
    for row_index in range(arr.shape[0]):
        row = arr[row_index]
        if not np.isfinite(row).all():
            continue
        signs = np.sign(row)
        pairs = 0
        matches = 0
        for left in range(len(signs)):
            for right in range(left + 1, len(signs)):
                pairs += 1
                if signs[left] == signs[right]:
                    matches += 1
        out[row_index] = matches / pairs if pairs else np.nan
    return pd.Series(out, index=values.index)


def build_coverage_summary(labels: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total_rows = len(labels)
    for horizon in horizons:
        for col in [
            f"label_resid_ret_norm_h{horizon}",
            f"label_resid_rank_centered_h{horizon}",
            f"label_path_signed_norm_h{horizon}",
            f"label_path_rank_centered_h{horizon}",
        ]:
            valid = labels[col].notna()
            rows.append(
                {
                    "label": col,
                    "valid_rows": int(valid.sum()),
                    "total_rows": int(total_rows),
                    "coverage": float(valid.mean()),
                    "first_valid_timestamp": labels.loc[valid, "timestamp"].min() if valid.any() else pd.NaT,
                    "last_valid_timestamp": labels.loc[valid, "timestamp"].max() if valid.any() else pd.NaT,
                }
            )
    for col in [
        "label_slow_medium",
        "label_slow_low_turnover",
        "label_path_slow_medium",
        "label_path_slow_low_turnover",
        "label_horizon_sign_consistency",
        "label_path_horizon_sign_consistency",
    ]:
        valid = labels[col].notna()
        rows.append(
            {
                "label": col,
                "valid_rows": int(valid.sum()),
                "total_rows": int(total_rows),
                "coverage": float(valid.mean()),
                "first_valid_timestamp": labels.loc[valid, "timestamp"].min() if valid.any() else pd.NaT,
                "last_valid_timestamp": labels.loc[valid, "timestamp"].max() if valid.any() else pd.NaT,
            }
        )
    return pd.DataFrame(rows)


def build_correlation_summary(labels: pd.DataFrame) -> pd.DataFrame:
    cols = [
        col
        for col in labels.columns
        if col.startswith("label_resid_rank_centered_h")
        or col.startswith("label_path_rank_centered_h")
        or col
        in {
            "label_slow_medium",
            "label_slow_low_turnover",
            "label_path_slow_medium",
            "label_path_slow_low_turnover",
        }
    ]
    corr = labels[cols].corr(method="spearman")
    return corr.reset_index(names="label")


def build_distribution_summary(labels: pd.DataFrame) -> pd.DataFrame:
    cols = [
        col
        for col in labels.columns
        if col.startswith("label_resid_ret_norm_h")
        or col.startswith("label_path_signed_norm_h")
        or col.startswith("label_resid_rank_centered_h")
        or col.startswith("label_path_rank_centered_h")
        or col.startswith("label_slow_")
        or col.startswith("label_path_slow_")
        or col.endswith("sign_consistency")
    ]
    rows = []
    for col in cols:
        series = labels[col].replace([np.inf, -np.inf], np.nan).dropna()
        rows.append(
            {
                "label": col,
                "count": int(series.size),
                "mean": float(series.mean()) if not series.empty else np.nan,
                "std": float(series.std(ddof=0)) if not series.empty else np.nan,
                "p01": float(series.quantile(0.01)) if not series.empty else np.nan,
                "p05": float(series.quantile(0.05)) if not series.empty else np.nan,
                "p50": float(series.quantile(0.50)) if not series.empty else np.nan,
                "p95": float(series.quantile(0.95)) if not series.empty else np.nan,
                "p99": float(series.quantile(0.99)) if not series.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def write_metadata(
    output_dir: Path,
    cfg: dict[str, Any],
    panel_path: Path,
    label_path: Path,
    horizons: list[int],
    execution_delay_bars: int,
    vol_lookback: int,
    path_lambda: float,
    rows: int,
) -> None:
    metadata = {
        "stage": "P1_slow_residual_labels",
        "panel_path": str(panel_path),
        "label_path": str(label_path),
        "rows": rows,
        "horizons": horizons,
        "execution_delay_bars": execution_delay_bars,
        "vol_lookback_hours": vol_lookback,
        "path_lambda": path_lambda,
        "factor_cols": FACTOR_COLS,
        "beta_cols": BETA_COLS,
        "terminal_label_definition": (
            "R_res_i(t,H) = log_close_i[t+execution_delay+H] - log_close_i[t+execution_delay] "
            "- beta_i,t' * sum_{k=execution_delay+1}^{execution_delay+H} factor_returns[t+k]"
        ),
        "rank_label_definition": "2 * (cross_sectional_rank - 1) / (valid_count - 1) - 1",
        "composite_medium": MEDIUM_WEIGHTS,
        "composite_low_turnover": LOW_TURNOVER_WEIGHTS,
        "leakage_guardrail": "Future residual returns are explained using beta columns from the row at decision time t only.",
        "locked_baseline": cfg.get("locked_baseline", {}),
    }
    with (output_dir / "label_metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(to_jsonable(metadata), fh, indent=2, ensure_ascii=False)


def write_report(
    output_dir: Path,
    coverage: pd.DataFrame,
    correlations: pd.DataFrame,
    distribution: pd.DataFrame,
) -> None:
    important_dist = distribution[
        distribution["label"].isin(
            [
                "label_slow_medium",
                "label_slow_low_turnover",
                "label_path_slow_medium",
                "label_path_slow_low_turnover",
                "label_horizon_sign_consistency",
                "label_path_horizon_sign_consistency",
            ]
        )
    ]
    lines = [
        "# P1 Slow Residual Labels",
        "",
        "## Design",
        "",
        "- Builds 48h / 72h / 96h / 168h ex-ante residual return labels.",
        "- Future residual return uses beta known at decision time `t`; beta is not re-estimated in the future window.",
        "- Rank labels are cross-sectional and centered to `[-1, 1]`.",
        "- `label_slow_medium` = 0.20*48h + 0.30*72h + 0.30*96h + 0.20*168h.",
        "- `label_slow_low_turnover` = 0.25*72h + 0.35*96h + 0.40*168h.",
        "- Path-aware signed labels penalize downside path risk for positive terminal residual return and upside squeeze risk for negative terminal residual return.",
        "",
        "## Output",
        "",
        "- `slow_residual_labels.csv.gz`: label panel keyed by `timestamp, symbol`.",
        "- `label_metadata.json`: exact formulas and parameters.",
        "- `coverage_summary.csv`: valid-row coverage.",
        "- `label_correlations.csv`: Spearman correlations.",
        "- `label_distribution_summary.csv`: distribution checks.",
        "",
        "## Coverage",
        "",
        "```text",
        coverage.to_string(index=False),
        "```",
        "",
        "## Important Distributions",
        "",
        "```text",
        important_dist.to_string(index=False),
        "```",
        "",
        "## Correlations",
        "",
        "```text",
        correlations.to_string(index=False),
        "```",
        "",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
