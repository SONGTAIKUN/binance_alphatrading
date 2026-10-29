#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "baseline_locked_v1"
DEFAULT_CONFIG = EXPERIMENT_ROOT / "inputs" / "long_history" / "configs" / "baseline_long_history.yaml"
DEFAULT_ASSET_PANEL = EXPERIMENT_ROOT / "inputs" / "long_history" / "baseline_panel" / "asset_residual_panel.csv"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "outputs" / "residual_factor_neutralization_audit"

FACTOR_COLS = ["factor_btc_ret", "factor_eth_ret", "factor_market_ret", "factor_sector_ret"]
BETA_COLS = ["beta_btc", "beta_eth", "beta_market", "beta_sector"]
RESIDUAL_COLS = [
    "timestamp",
    "symbol",
    "log_close",
    "log_ret",
    *FACTOR_COLS,
    "beta_intercept",
    *BETA_COLS,
    "predicted_log_ret",
    "residual_log_ret",
]

PORTFOLIO_RUNS = [
    (
        "long_history_p15c_nested",
        EXPERIMENT_ROOT / "outputs" / "long_history_nested_oos" / "p15c_nested" / "nested_extreme_cluster_run",
    ),
    (
        "p16_clean_candidate",
        EXPERIMENT_ROOT / "outputs" / "p16_clean_candidate_package" / "nested_candidate_run",
    ),
    (
        "latest_baseline_cost15",
        EXPERIMENT_ROOT / "outputs" / "latest_baseline_cost15" / "nested_candidate_run",
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit residual factor model, ex-ante labels, and portfolio neutralization.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--asset-panel", default=str(DEFAULT_ASSET_PANEL))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--beta-samples-per-symbol", type=int, default=12)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = read_yaml(Path(args.config))
    beta_window = int(cfg["factor_model"]["beta_window"])
    min_beta_obs = int(cfg["factor_model"]["min_beta_obs"])
    execution_delay = int(cfg["labels"]["execution_delay_bars"])
    horizons = [int(value) for value in cfg["labels"]["horizons"]]
    sectors = cfg["sectors"]

    print("[1/8] Loading residual panel columns", flush=True)
    panel = read_asset_panel(Path(args.asset_panel), horizons)

    print("[2/8] Auditing factor construction", flush=True)
    factor_audit = audit_factor_construction(panel, sectors)
    factor_audit.to_csv(output_dir / "factor_construction_audit.csv", index=False)

    print("[3/8] Auditing rolling beta past-only replication", flush=True)
    beta_sample = audit_beta_past_only(
        panel,
        beta_window=beta_window,
        min_beta_obs=min_beta_obs,
        samples_per_symbol=int(args.beta_samples_per_symbol),
    )
    beta_sample.to_csv(output_dir / "beta_past_only_sample.csv", index=False)
    beta_summary = summarize_beta_sample(beta_sample)
    beta_summary.to_csv(output_dir / "beta_past_only_summary.csv", index=False)

    print("[4/8] Auditing residual identity", flush=True)
    residual_identity = audit_residual_identity(panel)
    residual_identity.to_csv(output_dir / "residual_identity_audit.csv", index=False)

    print("[5/8] Auditing factor-model quality diagnostics", flush=True)
    model_quality = audit_factor_model_quality(panel)
    model_quality.to_csv(output_dir / "factor_model_quality.csv", index=False)

    print("[6/8] Auditing forward residual labels", flush=True)
    label_audit = audit_forward_labels(panel, horizons, execution_delay)
    label_audit.to_csv(output_dir / "label_ex_ante_audit.csv", index=False)

    print("[7/8] Auditing realized portfolio neutralization", flush=True)
    neutralization_summary, neutralization_detail = audit_portfolio_runs(panel, PORTFOLIO_RUNS)
    neutralization_summary.to_csv(output_dir / "portfolio_neutralization_summary.csv", index=False)
    neutralization_detail.to_csv(output_dir / "portfolio_neutralization_by_timestamp.csv", index=False)

    print("[8/8] Writing topline report", flush=True)
    topline = build_topline(factor_audit, beta_summary, residual_identity, model_quality, label_audit, neutralization_summary)
    topline.to_csv(output_dir / "topline_findings.csv", index=False)
    metadata = {
        "config": str(Path(args.config).resolve()),
        "asset_panel": str(Path(args.asset_panel).resolve()),
        "beta_window": beta_window,
        "min_beta_obs": min_beta_obs,
        "execution_delay_bars": execution_delay,
        "horizons": horizons,
        "portfolio_runs": [{"name": name, "path": str(path.resolve())} for name, path in PORTFOLIO_RUNS if path.exists()],
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(output_dir, topline, factor_audit, beta_summary, residual_identity, model_quality, label_audit, neutralization_summary)
    print(f"Done: {output_dir}", flush=True)


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def read_asset_panel(path: Path, horizons: list[int]) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    label_cols = []
    for horizon in horizons:
        label_cols.extend([f"fwd_log_ret_h{horizon}", f"label_resid_ret_h{horizon}"])
    usecols = [col for col in [*RESIDUAL_COLS, *label_cols] if col in header]
    panel = pd.read_csv(path, usecols=usecols, parse_dates=["timestamp"])
    panel["timestamp"] = pd.to_datetime(panel["timestamp"], utc=True)
    panel["symbol"] = panel["symbol"].astype(str)
    return panel.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def build_sector_lookup(sectors: dict[str, list[str]]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for sector, symbols in sectors.items():
        for symbol in symbols:
            lookup[symbol] = sector
    return lookup


def audit_factor_construction(panel: pd.DataFrame, sectors: dict[str, list[str]]) -> pd.DataFrame:
    wide_ret = panel.pivot(index="timestamp", columns="symbol", values="log_ret").sort_index()
    sector_lookup = build_sector_lookup(sectors)
    rows: list[dict[str, Any]] = []
    for symbol, group in panel.groupby("symbol", sort=True):
        group = group.sort_values("timestamp")
        idx = pd.DatetimeIndex(group["timestamp"])

        expected_btc = pd.Series(0.0, index=idx) if symbol == "BTCUSDT" else wide_ret["BTCUSDT"].reindex(idx)
        expected_eth = pd.Series(0.0, index=idx) if symbol == "ETHUSDT" else wide_ret["ETHUSDT"].reindex(idx)
        expected_market = wide_ret[[col for col in wide_ret.columns if col != symbol]].mean(axis=1).reindex(idx)

        sector = sector_lookup.get(symbol)
        members = sectors.get(sector, []) if sector else []
        sector_cols = [member for member in members if member in wide_ret.columns and member != symbol]
        expected_sector = wide_ret[sector_cols].mean(axis=1).reindex(idx) if sector_cols else pd.Series(0.0, index=idx)

        rows.append(compare_series(symbol, "factor_btc_ret", group["factor_btc_ret"], expected_btc.to_numpy()))
        rows.append(compare_series(symbol, "factor_eth_ret", group["factor_eth_ret"], expected_eth.to_numpy()))
        rows.append(compare_series(symbol, "factor_market_ret", group["factor_market_ret"], expected_market.to_numpy()))
        rows.append(compare_series(symbol, "factor_sector_ret", group["factor_sector_ret"], expected_sector.to_numpy()))
    return pd.DataFrame(rows)


def compare_series(symbol: str, field: str, observed: pd.Series, expected_values: np.ndarray, tolerance: float = 1e-12) -> dict[str, Any]:
    observed_num = pd.to_numeric(observed, errors="coerce").to_numpy(dtype=float)
    expected = np.asarray(expected_values, dtype=float)
    valid = np.isfinite(observed_num) & np.isfinite(expected)
    both_nan = ~np.isfinite(observed_num) & ~np.isfinite(expected)
    diff = np.abs(observed_num[valid] - expected[valid])
    return {
        "symbol": symbol,
        "field": field,
        "rows": int(len(observed_num)),
        "valid_compared": int(valid.sum()),
        "both_nan": int(both_nan.sum()),
        "observed_notna": int(np.isfinite(observed_num).sum()),
        "expected_notna": int(np.isfinite(expected).sum()),
        "max_abs_diff": float(diff.max()) if len(diff) else np.nan,
        "mean_abs_diff": float(diff.mean()) if len(diff) else np.nan,
        "mismatch_count": int((diff > tolerance).sum()) if len(diff) else 0,
    }


def audit_beta_past_only(
    panel: pd.DataFrame,
    beta_window: int,
    min_beta_obs: int,
    samples_per_symbol: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    needed = ["log_ret", *FACTOR_COLS, "beta_intercept", *BETA_COLS]
    for symbol, group in panel.groupby("symbol", sort=True):
        group = group.sort_values("timestamp").reset_index(drop=True)
        valid_beta = group[["beta_intercept", *BETA_COLS]].notna().all(axis=1)
        valid_positions = np.flatnonzero(valid_beta.to_numpy())
        if len(valid_positions) == 0:
            continue
        sample_count = min(samples_per_symbol, len(valid_positions))
        sample_positions = valid_positions[np.linspace(0, len(valid_positions) - 1, sample_count, dtype=int)]
        for pos in sample_positions:
            row = group.iloc[pos]
            past = group.iloc[max(0, pos - beta_window) : pos]
            inclusive = group.iloc[max(0, pos - beta_window + 1) : pos + 1]
            stored = row[["beta_intercept", *BETA_COLS]].astype(float).to_numpy()
            past_beta, past_obs = fit_ols_beta(past)
            inclusive_beta, inclusive_obs = fit_ols_beta(inclusive)
            past_diff = np.abs(stored - past_beta) if past_beta is not None else np.array([np.nan])
            inclusive_diff = np.abs(stored - inclusive_beta) if inclusive_beta is not None else np.array([np.nan])
            rows.append(
                {
                    "symbol": symbol,
                    "timestamp": row["timestamp"],
                    "past_window_start": past["timestamp"].min() if len(past) else pd.NaT,
                    "past_window_end": past["timestamp"].max() if len(past) else pd.NaT,
                    "stored_beta_finite": bool(np.isfinite(stored).all()),
                    "past_valid_obs": int(past_obs),
                    "inclusive_valid_obs": int(inclusive_obs),
                    "past_max_abs_diff": float(np.nanmax(past_diff)),
                    "past_mean_abs_diff": float(np.nanmean(past_diff)),
                    "current_inclusive_max_abs_diff": float(np.nanmax(inclusive_diff)),
                    "current_inclusive_mean_abs_diff": float(np.nanmean(inclusive_diff)),
                    "stored_beta_intercept": float(stored[0]),
                    "recomputed_past_beta_intercept": float(past_beta[0]) if past_beta is not None else np.nan,
                }
            )
    return pd.DataFrame(rows)


def fit_ols_beta(frame: pd.DataFrame) -> tuple[np.ndarray | None, int]:
    y = frame["log_ret"].astype(float).to_numpy()
    x = frame[FACTOR_COLS].astype(float).to_numpy()
    z = np.column_stack([np.ones(len(frame)), x])
    valid = np.isfinite(y) & np.isfinite(z).all(axis=1)
    obs = int(valid.sum())
    if obs == 0:
        return None, obs
    z_valid = z[valid]
    y_valid = y[valid]
    coef = np.linalg.pinv(z_valid.T @ z_valid, rcond=1e-10) @ (z_valid.T @ y_valid)
    return coef, obs


def summarize_beta_sample(sample: pd.DataFrame) -> pd.DataFrame:
    if sample.empty:
        return pd.DataFrame([{"samples": 0}])
    return pd.DataFrame(
        [
            {
                "samples": int(len(sample)),
                "symbols": int(sample["symbol"].nunique()),
                "min_past_valid_obs": int(sample["past_valid_obs"].min()),
                "max_past_max_abs_diff": float(sample["past_max_abs_diff"].max()),
                "p95_past_max_abs_diff": float(sample["past_max_abs_diff"].quantile(0.95)),
                "mean_past_max_abs_diff": float(sample["past_max_abs_diff"].mean()),
                "median_current_inclusive_max_abs_diff": float(sample["current_inclusive_max_abs_diff"].median()),
                "share_inclusive_diff_larger_than_past": float(
                    (sample["current_inclusive_max_abs_diff"] > sample["past_max_abs_diff"] * 10.0 + 1e-14).mean()
                ),
            }
        ]
    )


def audit_residual_identity(panel: pd.DataFrame) -> pd.DataFrame:
    beta_and_factor = panel[["beta_intercept", *BETA_COLS, *FACTOR_COLS]].astype(float)
    valid = beta_and_factor.notna().all(axis=1) & panel["log_ret"].notna()
    predicted = panel["beta_intercept"].astype(float)
    for beta_col, factor_col in zip(BETA_COLS, FACTOR_COLS, strict=True):
        predicted = predicted + panel[beta_col].astype(float) * panel[factor_col].astype(float)
    residual = panel["log_ret"].astype(float) - predicted
    pred_diff = (panel["predicted_log_ret"].astype(float) - predicted).abs()
    resid_diff = (panel["residual_log_ret"].astype(float) - residual).abs()
    rows = [
        {
            "check": "predicted_log_ret_identity",
            "valid_rows": int(valid.sum()),
            "max_abs_diff": float(pred_diff[valid].max()),
            "p99_abs_diff": float(pred_diff[valid].quantile(0.99)),
            "mean_abs_diff": float(pred_diff[valid].mean()),
        },
        {
            "check": "residual_log_ret_identity",
            "valid_rows": int(valid.sum()),
            "max_abs_diff": float(resid_diff[valid].max()),
            "p99_abs_diff": float(resid_diff[valid].quantile(0.99)),
            "mean_abs_diff": float(resid_diff[valid].mean()),
        },
    ]
    return pd.DataFrame(rows)


def audit_factor_model_quality(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for symbol, group in panel.groupby("symbol", sort=True):
        group = group.sort_values("timestamp")
        valid = group[["log_ret", "predicted_log_ret", "residual_log_ret", *FACTOR_COLS, *BETA_COLS]].notna().all(axis=1)
        valid_group = group.loc[valid].copy()
        raw = valid_group["log_ret"].astype(float)
        pred = valid_group["predicted_log_ret"].astype(float)
        resid = valid_group["residual_log_ret"].astype(float)
        raw_var = float(raw.var(ddof=0)) if len(raw) else np.nan
        resid_var = float(resid.var(ddof=0)) if len(resid) else np.nan
        factor_corr = valid_group[FACTOR_COLS].astype(float).corr().abs()
        off_diag = factor_corr.where(~np.eye(len(factor_corr), dtype=bool))
        x = valid_group[FACTOR_COLS].astype(float)
        x = x.loc[:, x.std(ddof=0) > 1e-14]
        if len(x.columns) >= 1 and len(x) > len(x.columns) + 5:
            x_std = (x - x.mean()) / x.std(ddof=0)
            z = np.column_stack([np.ones(len(x_std)), x_std.to_numpy()])
            condition_number = float(np.linalg.cond(z.T @ z))
        else:
            condition_number = np.nan
        row: dict[str, Any] = {
            "symbol": symbol,
            "rows": int(len(group)),
            "valid_model_rows": int(valid.sum()),
            "valid_model_rate": float(valid.mean()),
            "raw_ret_std": float(raw.std(ddof=0)) if len(raw) else np.nan,
            "predicted_ret_std": float(pred.std(ddof=0)) if len(pred) else np.nan,
            "residual_ret_std": float(resid.std(ddof=0)) if len(resid) else np.nan,
            "residual_to_raw_vol_ratio": float(np.sqrt(resid_var / raw_var)) if raw_var > 0 and resid_var >= 0 else np.nan,
            "approx_out_of_sample_r2": float(1.0 - resid_var / raw_var) if raw_var > 0 and resid_var >= 0 else np.nan,
            "max_abs_factor_corr": float(off_diag.max().max()) if off_diag.notna().any().any() else np.nan,
            "factor_design_condition_number": condition_number,
        }
        for beta_col in BETA_COLS:
            beta = valid_group[beta_col].astype(float)
            row[f"{beta_col}_p01"] = float(beta.quantile(0.01)) if len(beta) else np.nan
            row[f"{beta_col}_p50"] = float(beta.quantile(0.50)) if len(beta) else np.nan
            row[f"{beta_col}_p99"] = float(beta.quantile(0.99)) if len(beta) else np.nan
            row[f"{beta_col}_std"] = float(beta.std(ddof=0)) if len(beta) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def audit_forward_labels(panel: pd.DataFrame, horizons: list[int], execution_delay: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for horizon in horizons:
        fwd_col = f"fwd_log_ret_h{horizon}"
        label_col = f"label_resid_ret_h{horizon}"
        if fwd_col not in panel or label_col not in panel:
            continue
        fwd_diffs = []
        label_diffs = []
        valid_counts = 0
        for _, group in panel.groupby("symbol", sort=False):
            group = group.sort_values("timestamp").reset_index(drop=True)
            start = group["log_close"].shift(-execution_delay)
            end = group["log_close"].shift(-(execution_delay + horizon))
            expected_fwd = end - start
            explained = pd.Series(0.0, index=group.index, dtype=float)
            valid_explained = pd.Series(True, index=group.index)
            for beta_col, factor_col in zip(BETA_COLS, FACTOR_COLS, strict=True):
                factor_sum = forward_sum(group[factor_col], execution_delay + 1, horizon)
                explained = explained + group[beta_col] * factor_sum
                valid_explained = valid_explained & factor_sum.notna() & group[beta_col].notna()
            expected_label = (expected_fwd - explained).where(valid_explained)
            valid = expected_label.notna() & group[label_col].notna() & expected_fwd.notna() & group[fwd_col].notna()
            valid_counts += int(valid.sum())
            fwd_diffs.append((group.loc[valid, fwd_col] - expected_fwd.loc[valid]).abs())
            label_diffs.append((group.loc[valid, label_col] - expected_label.loc[valid]).abs())
        fwd_diff = pd.concat(fwd_diffs, ignore_index=True) if fwd_diffs else pd.Series(dtype=float)
        label_diff = pd.concat(label_diffs, ignore_index=True) if label_diffs else pd.Series(dtype=float)
        rows.append(
            {
                "horizon": horizon,
                "execution_delay_bars": execution_delay,
                "valid_rows": valid_counts,
                "fwd_max_abs_diff": float(fwd_diff.max()) if len(fwd_diff) else np.nan,
                "fwd_p99_abs_diff": float(fwd_diff.quantile(0.99)) if len(fwd_diff) else np.nan,
                "label_max_abs_diff": float(label_diff.max()) if len(label_diff) else np.nan,
                "label_p99_abs_diff": float(label_diff.quantile(0.99)) if len(label_diff) else np.nan,
                "label_formula": f"log_close[t+{execution_delay + horizon}] - log_close[t+{execution_delay}] - beta_t * sum(factors[t+{execution_delay + 1}:t+{execution_delay + horizon}])",
            }
        )
    return pd.DataFrame(rows)


def forward_sum(series: pd.Series, start_offset: int, horizon: int) -> pd.Series:
    total = pd.Series(0.0, index=series.index, dtype=float)
    valid = pd.Series(True, index=series.index)
    for offset in range(start_offset, start_offset + horizon):
        shifted = series.shift(-offset)
        total = total + shifted.fillna(0.0)
        valid = valid & shifted.notna()
    return total.where(valid)


def audit_portfolio_runs(
    panel: pd.DataFrame,
    runs: list[tuple[str, Path]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    beta_panel = panel[["timestamp", "symbol", *BETA_COLS]].copy()
    summary_rows: list[dict[str, Any]] = []
    detail_frames: list[pd.DataFrame] = []
    for name, run_dir in runs:
        positions_path = run_dir / "positions.csv"
        equity_path = run_dir / "equity_curve.csv"
        if not positions_path.exists() or not equity_path.exists():
            summary_rows.append({"run": name, "path": str(run_dir), "exists": False})
            continue
        positions = pd.read_csv(positions_path, parse_dates=["timestamp"])
        positions["timestamp"] = pd.to_datetime(positions["timestamp"], utc=True)
        positions["symbol"] = positions["symbol"].astype(str)
        equity = pd.read_csv(equity_path, parse_dates=["timestamp"])
        equity["timestamp"] = pd.to_datetime(equity["timestamp"], utc=True)
        merged = positions.merge(beta_panel, on=["timestamp", "symbol"], how="left")
        merged["long_weight"] = merged["weight"].clip(lower=0.0)
        merged["short_weight_abs"] = (-merged["weight"]).clip(lower=0.0)
        for beta_col in BETA_COLS:
            merged[f"{beta_col}_exposure"] = merged["weight"] * merged[beta_col]
        grouped = merged.groupby("timestamp", sort=False)
        detail = grouped.agg(
            net_weight=("weight", "sum"),
            gross_exposure=("weight", lambda x: float(np.abs(x).sum())),
            long_gross=("long_weight", "sum"),
            short_gross=("short_weight_abs", "sum"),
            max_abs_weight=("weight", lambda x: float(np.abs(x).max())),
            missing_beta_rows=("beta_btc", lambda x: int(x.isna().sum())),
            beta_btc_exposure=("beta_btc_exposure", "sum"),
            beta_eth_exposure=("beta_eth_exposure", "sum"),
            beta_market_exposure=("beta_market_exposure", "sum"),
            beta_sector_exposure=("beta_sector_exposure", "sum"),
            active_symbols=("symbol", "nunique"),
        ).reset_index()
        detail["run"] = name
        detail = detail.merge(
            equity[
                [
                    col
                    for col in [
                        "timestamp",
                        "abs_beta_btc",
                        "abs_beta_eth",
                        "abs_beta_market",
                        "gross_exposure",
                        "optimizer_success",
                        "optimizer_exposure_error",
                    ]
                    if col in equity.columns
                ]
            ].rename(columns={"gross_exposure": "equity_gross_exposure"}),
            on="timestamp",
            how="left",
        )
        for beta_col in ["beta_btc", "beta_eth", "beta_market"]:
            exposure_col = f"{beta_col}_exposure"
            reported_col = f"abs_{beta_col}"
            if reported_col in detail:
                detail[f"{beta_col}_reported_abs_diff"] = detail[reported_col] - detail[exposure_col].abs()
        detail_frames.append(detail)

        row = {
            "run": name,
            "path": str(run_dir),
            "exists": True,
            "periods": int(len(detail)),
            "position_rows": int(len(positions)),
            "max_missing_beta_rows": int(detail["missing_beta_rows"].max()),
            "mean_net_weight_abs": float(detail["net_weight"].abs().mean()),
            "max_net_weight_abs": float(detail["net_weight"].abs().max()),
            "mean_abs_beta_btc": float(detail["beta_btc_exposure"].abs().mean()),
            "max_abs_beta_btc": float(detail["beta_btc_exposure"].abs().max()),
            "mean_abs_beta_eth": float(detail["beta_eth_exposure"].abs().mean()),
            "max_abs_beta_eth": float(detail["beta_eth_exposure"].abs().max()),
            "mean_abs_beta_market": float(detail["beta_market_exposure"].abs().mean()),
            "max_abs_beta_market": float(detail["beta_market_exposure"].abs().max()),
            "mean_abs_beta_sector": float(detail["beta_sector_exposure"].abs().mean()),
            "max_abs_beta_sector": float(detail["beta_sector_exposure"].abs().max()),
            "mean_gross_exposure": float(detail["gross_exposure"].mean()),
            "max_gross_exposure": float(detail["gross_exposure"].max()),
            "mean_long_short_gross_gap": float((detail["long_gross"] - detail["short_gross"]).abs().mean()),
            "max_long_short_gross_gap": float((detail["long_gross"] - detail["short_gross"]).abs().max()),
            "max_abs_symbol_weight": float(detail["max_abs_weight"].max()),
        }
        if "optimizer_success" in detail:
            row["optimizer_success_rate"] = float(detail["optimizer_success"].dropna().mean())
        if "optimizer_exposure_error" in detail:
            row["max_optimizer_exposure_error"] = float(detail["optimizer_exposure_error"].dropna().max())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    detail_all = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    return summary, detail_all


def build_topline(
    factor_audit: pd.DataFrame,
    beta_summary: pd.DataFrame,
    residual_identity: pd.DataFrame,
    model_quality: pd.DataFrame,
    label_audit: pd.DataFrame,
    neutralization_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    factor_max = float(factor_audit["max_abs_diff"].max())
    factor_mismatches = int(factor_audit["mismatch_count"].sum())
    rows.append({"check": "factor_construction_max_abs_diff", "value": factor_max, "status": "pass" if factor_max < 1e-12 and factor_mismatches == 0 else "fail"})
    rows.append({"check": "factor_construction_mismatch_count", "value": factor_mismatches, "status": "pass" if factor_mismatches == 0 else "fail"})

    beta_max = float(beta_summary["max_past_max_abs_diff"].iloc[0]) if "max_past_max_abs_diff" in beta_summary else np.nan
    beta_obs = int(beta_summary["min_past_valid_obs"].iloc[0]) if "min_past_valid_obs" in beta_summary else 0
    rows.append({"check": "rolling_beta_past_only_max_abs_diff", "value": beta_max, "status": "pass" if beta_max < 1e-8 else "fail"})
    rows.append({"check": "rolling_beta_min_past_valid_obs", "value": beta_obs, "status": "pass" if beta_obs >= 1 else "review"})

    resid_max = float(residual_identity["max_abs_diff"].max())
    rows.append({"check": "residual_identity_max_abs_diff", "value": resid_max, "status": "pass" if resid_max < 1e-12 else "fail"})

    max_factor_corr = float(model_quality["max_abs_factor_corr"].max())
    max_condition = float(model_quality["factor_design_condition_number"].replace([np.inf, -np.inf], np.nan).max())
    median_resid_ratio = float(model_quality["residual_to_raw_vol_ratio"].median())
    min_oos_r2 = float(model_quality["approx_out_of_sample_r2"].min())
    rows.append({"check": "factor_model_max_abs_factor_corr", "value": max_factor_corr, "status": "review" if max_factor_corr > 0.95 else "pass"})
    rows.append({"check": "factor_model_max_condition_number", "value": max_condition, "status": "review" if max_condition > 1e5 else "pass"})
    rows.append({"check": "factor_model_median_residual_to_raw_vol_ratio", "value": median_resid_ratio, "status": "pass" if median_resid_ratio < 1.0 else "review"})
    rows.append({"check": "factor_model_min_approx_oos_r2", "value": min_oos_r2, "status": "review" if min_oos_r2 < -0.10 else "pass"})

    label_max = float(label_audit["label_max_abs_diff"].max())
    fwd_max = float(label_audit["fwd_max_abs_diff"].max())
    rows.append({"check": "forward_return_label_max_abs_diff", "value": fwd_max, "status": "pass" if fwd_max < 1e-12 else "fail"})
    rows.append({"check": "residual_label_current_beta_max_abs_diff", "value": label_max, "status": "pass" if label_max < 1e-12 else "fail"})

    existing_runs = (
        neutralization_summary[neutralization_summary["exists"].fillna(False).astype(bool)]
        if "exists" in neutralization_summary
        else neutralization_summary.iloc[0:0]
    )
    for _, row in existing_runs.iterrows():
        prefix = str(row["run"])
        rows.append({"check": f"{prefix}_max_net_weight_abs", "value": float(row["max_net_weight_abs"]), "status": "pass" if float(row["max_net_weight_abs"]) < 1e-8 else "review"})
        rows.append({"check": f"{prefix}_max_abs_beta_market", "value": float(row["max_abs_beta_market"]), "status": "pass" if float(row["max_abs_beta_market"]) < 1e-7 else "review"})
        rows.append({"check": f"{prefix}_max_missing_beta_rows", "value": int(row["max_missing_beta_rows"]), "status": "pass" if int(row["max_missing_beta_rows"]) == 0 else "fail"})
    return pd.DataFrame(rows)


def write_report(
    output_dir: Path,
    topline: pd.DataFrame,
    factor_audit: pd.DataFrame,
    beta_summary: pd.DataFrame,
    residual_identity: pd.DataFrame,
    model_quality: pd.DataFrame,
    label_audit: pd.DataFrame,
    neutralization_summary: pd.DataFrame,
) -> None:
    lines = [
        "# Residual / Factor / Neutralization Audit",
        "",
        "This audit numerically recomputes factor inputs, rolling betas, residual identities, forward residual labels, and realized portfolio exposures.",
        "",
        "## Topline",
        "",
        "```text",
        topline.to_string(index=False),
        "```",
        "",
        "## Factor Construction",
        "",
        "```text",
        factor_audit.groupby("field").agg(
            valid_compared=("valid_compared", "sum"),
            max_abs_diff=("max_abs_diff", "max"),
            mismatch_count=("mismatch_count", "sum"),
        ).reset_index().to_string(index=False),
        "```",
        "",
        "## Rolling Beta Past-Only",
        "",
        "```text",
        beta_summary.to_string(index=False),
        "```",
        "",
        "## Residual Identity",
        "",
        "```text",
        residual_identity.to_string(index=False),
        "```",
        "",
        "## Factor-Model Quality",
        "",
        "```text",
        model_quality[
            [
                "symbol",
                "valid_model_rate",
                "residual_to_raw_vol_ratio",
                "approx_out_of_sample_r2",
                "max_abs_factor_corr",
                "factor_design_condition_number",
            ]
        ].to_string(index=False),
        "```",
        "",
        "## Forward Residual Labels",
        "",
        "```text",
        label_audit[["horizon", "execution_delay_bars", "valid_rows", "fwd_max_abs_diff", "label_max_abs_diff"]].to_string(index=False),
        "```",
        "",
        "## Portfolio Neutralization",
        "",
        "```text",
        neutralization_summary.to_string(index=False),
        "```",
        "",
        "## Interpretation",
        "",
        "- Factor returns are same-bar completed returns; signals are assumed to be formed after the bar close.",
        "- Rolling betas must match OLS fitted on rows strictly before t.",
        "- Residual labels may use future returns because they are labels, but the factor exposure used in the label must be beta_t, not future beta.",
        "- Portfolio neutralization is checked on realized position weights joined to timestamp-t betas.",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
