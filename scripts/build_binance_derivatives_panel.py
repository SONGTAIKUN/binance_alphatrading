#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crypto_stat_arb.config import load_config


EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "baseline_locked_v1"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "baseline.yaml"
DEFAULT_PANEL = EXPERIMENT_ROOT / "inputs" / "asset_residual_panel.csv"
DEFAULT_INPUT = EXPERIMENT_ROOT / "inputs" / "binance_derivatives" / "raw"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "inputs" / "binance_derivatives" / "derivatives_hourly_panel.csv.gz"

KLINE_COLS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
]
METRIC_COLS = [
    "sum_open_interest",
    "sum_open_interest_value",
    "count_toptrader_long_short_ratio",
    "sum_toptrader_long_short_ratio",
    "count_long_short_ratio",
    "sum_taker_long_short_vol_ratio",
]

SOURCE_FILL_LIMITS = {
    "funding_rate": 8,
    "funding_interval_hours": 8,
    "sum_open_interest": 2,
    "sum_open_interest_value": 2,
    "count_toptrader_long_short_ratio": 2,
    "sum_toptrader_long_short_ratio": 2,
    "count_long_short_ratio": 2,
    "sum_taker_long_short_vol_ratio": 2,
    "index_close": 1,
    "mark_close": 1,
    "premium_close": 1,
}

SOURCE_MAX_CURRENT_AGE_HOURS = {
    "funding_rate": 9.0,
    "funding_interval_hours": 9.0,
    "sum_open_interest": 3.0,
    "sum_open_interest_value": 3.0,
    "count_toptrader_long_short_ratio": 3.0,
    "sum_toptrader_long_short_ratio": 3.0,
    "count_long_short_ratio": 3.0,
    "sum_taker_long_short_vol_ratio": 3.0,
    "index_close": 2.0,
    "mark_close": 2.0,
    "premium_close": 2.0,
}

DERIVATIVE_SOURCE_MAP = {
    "deriv_funding_rate": ("funding_rate",),
    "deriv_funding_annualized": ("funding_rate", "funding_interval_hours"),
    "deriv_oi_value": ("sum_open_interest_value",),
    "deriv_log_oi_value": ("sum_open_interest_value",),
    "deriv_open_interest": ("sum_open_interest",),
    "deriv_account_long_short_ratio": ("count_long_short_ratio",),
    "deriv_top_count_long_short_ratio": ("count_toptrader_long_short_ratio",),
    "deriv_top_sum_long_short_ratio": ("sum_toptrader_long_short_ratio",),
    "deriv_taker_long_short_ratio": ("sum_taker_long_short_vol_ratio",),
    "deriv_mark_index_basis": ("mark_close", "index_close"),
    "deriv_premium_index": ("premium_close",),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--panel", default=str(DEFAULT_PANEL))
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--symbols", nargs="*", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    symbols = list(args.symbols or cfg["data"]["symbols"])
    raw_dir = Path(args.input_dir).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    panel_ts = load_panel_timestamps(args.panel)
    frames = []
    for idx, symbol in enumerate(symbols, start=1):
        frame = build_symbol_panel(symbol, raw_dir, panel_ts)
        frames.append(frame)
        coverage = frame.drop(columns=["timestamp", "symbol"]).notna().mean().mean()
        print(f"[{idx:02d}/{len(symbols):02d}] {symbol} parsed rows={len(frame):,} avg_coverage={coverage:.2%}", flush=True)

    out = pd.concat(frames, ignore_index=True).sort_values(["timestamp", "symbol"])
    out.to_csv(output_path, index=False, compression="gzip")
    write_metadata(out, output_path.with_suffix("").with_suffix(".metadata.json"))
    print(f"Derivatives panel: {output_path}", flush=True)


def load_panel_timestamps(panel_path: str) -> pd.DatetimeIndex:
    timestamps = pd.read_csv(panel_path, usecols=["timestamp"], parse_dates=["timestamp"])["timestamp"]
    timestamps = pd.to_datetime(timestamps, utc=True)
    return pd.DatetimeIndex(sorted(timestamps.drop_duplicates()))


def build_symbol_panel(symbol: str, raw_dir: Path, panel_ts: pd.DatetimeIndex) -> pd.DataFrame:
    base = pd.DataFrame({"timestamp": panel_ts})
    base["symbol"] = symbol

    funding = read_funding(symbol, raw_dir)
    metrics = read_metrics(symbol, raw_dir)
    index_px = read_price_klines(symbol, raw_dir, "indexPriceKlines", "index")
    mark_px = read_price_klines(symbol, raw_dir, "markPriceKlines", "mark")
    premium = read_price_klines(symbol, raw_dir, "premiumIndexKlines", "premium")

    for frame in [funding, metrics, index_px, mark_px, premium]:
        if frame.empty:
            continue
        base = base.merge(frame, on=["timestamp", "symbol"], how="left")

    base.sort_values("timestamp", inplace=True)
    base = apply_asof_fill(symbol, base)
    base = build_features(base)
    return base


def read_funding(symbol: str, raw_dir: Path) -> pd.DataFrame:
    files = sorted((raw_dir / "monthly" / "fundingRate" / symbol).glob("*.zip"))
    frames = []
    for path in files:
        df = read_zip_csv(path)
        if df.empty or "calc_time" not in df:
            continue
        df["timestamp"] = pd.to_datetime(df["calc_time"], unit="ms", utc=True).dt.floor("h")
        df["symbol"] = symbol
        df["funding_rate"] = pd.to_numeric(df.get("last_funding_rate"), errors="coerce")
        df["funding_interval_hours"] = pd.to_numeric(df.get("funding_interval_hours"), errors="coerce")
        frames.append(df[["timestamp", "symbol", "funding_rate", "funding_interval_hours"]])
    return concat_drop_duplicate(frames)


def read_metrics(symbol: str, raw_dir: Path) -> pd.DataFrame:
    files = sorted((raw_dir / "daily" / "metrics" / symbol).glob("*.zip"))
    frames = []
    for path in files:
        df = read_zip_csv(path)
        if df.empty or "create_time" not in df:
            continue
        df["timestamp"] = pd.to_datetime(df["create_time"], utc=True)
        df["symbol"] = symbol
        for col in METRIC_COLS:
            if col in df:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                df[col] = np.nan
        frames.append(df[["timestamp", "symbol", *METRIC_COLS]])
    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, ignore_index=True).sort_values("timestamp").drop_duplicates("timestamp")
    data = data.set_index("timestamp")
    agg = {
        "symbol": "last",
        "sum_open_interest": "last",
        "sum_open_interest_value": "last",
        "count_toptrader_long_short_ratio": "mean",
        "sum_toptrader_long_short_ratio": "mean",
        "count_long_short_ratio": "mean",
        "sum_taker_long_short_vol_ratio": "mean",
    }
    hourly = data.resample("1h", label="left", closed="left").agg(agg).dropna(how="all")
    hourly["symbol"] = symbol
    return hourly.reset_index()


def read_price_klines(symbol: str, raw_dir: Path, series: str, prefix: str) -> pd.DataFrame:
    files = sorted((raw_dir / "monthly" / series / symbol / "1h").glob("*.zip"))
    frames = []
    for path in files:
        df = read_zip_csv(path)
        if df.empty:
            continue
        if "open_time" not in df:
            df = read_zip_csv(path, header=None)
            if len(df.columns) == len(KLINE_COLS):
                df.columns = KLINE_COLS
        if "open_time" not in df:
            continue
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["symbol"] = symbol
        for col in ["open", "high", "low", "close"]:
            df[f"{prefix}_{col}"] = pd.to_numeric(df[col], errors="coerce")
        frames.append(df[["timestamp", "symbol", f"{prefix}_open", f"{prefix}_high", f"{prefix}_low", f"{prefix}_close"]])
    return concat_drop_duplicate(frames)


def read_zip_csv(path: Path, header: int | None = 0) -> pd.DataFrame:
    try:
        with zipfile.ZipFile(path) as zf:
            names = [name for name in zf.namelist() if name.endswith(".csv")]
            if not names:
                return pd.DataFrame()
            with zf.open(names[0]) as fh:
                return pd.read_csv(fh, header=header)
    except (zipfile.BadZipFile, OSError, pd.errors.ParserError):
        return pd.DataFrame()


def concat_drop_duplicate(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("timestamp").drop_duplicates(["timestamp", "symbol"])


def apply_asof_fill(symbol: str, base: pd.DataFrame) -> pd.DataFrame:
    base = base.copy()
    base["timestamp"] = pd.to_datetime(base["timestamp"], utc=True)
    # Keep fills short. Unlimited ffill would turn a missing archive segment into
    # stale "known" data for months, which is a hidden data quality bug.
    for col, limit in SOURCE_FILL_LIMITS.items():
        if col not in base:
            continue
        observed = base[col].notna()
        last_observed_ts = pd.Series(pd.NaT, index=base.index, dtype="datetime64[ns, UTC]")
        last_observed_ts.loc[observed] = base.loc[observed, "timestamp"]
        last_observed_ts = last_observed_ts.ffill()
        age_hours = (base["timestamp"] - last_observed_ts).dt.total_seconds() / 3600.0
        filled = base[col].ffill(limit=limit)
        base[f"{col}_source_observed"] = observed
        base[f"{col}_source_was_filled"] = (filled.notna() & ~observed).astype(float)
        base[f"{col}_source_age_hours"] = age_hours.where(filled.notna())
        base[col] = filled
    base["symbol"] = symbol
    return base


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame[["timestamp", "symbol"]].copy()
    frame = frame.copy()

    if {"mark_close", "index_close"}.issubset(frame.columns):
        frame["mark_index_basis"] = np.log(frame["mark_close"].clip(lower=1e-12) / frame["index_close"].clip(lower=1e-12))
    else:
        frame["mark_index_basis"] = np.nan
    if "premium_close" not in frame:
        frame["premium_close"] = np.nan

    raw_cols = [
        "funding_rate",
        "funding_interval_hours",
        "sum_open_interest",
        "sum_open_interest_value",
        "count_toptrader_long_short_ratio",
        "sum_toptrader_long_short_ratio",
        "count_long_short_ratio",
        "sum_taker_long_short_vol_ratio",
        "mark_index_basis",
        "premium_close",
    ]
    for col in raw_cols:
        if col not in frame:
            frame[col] = np.nan
        frame[f"{col}_lag1h"] = frame[col].shift(1)

    x = frame
    out["deriv_funding_rate"] = x["funding_rate_lag1h"]
    interval = x["funding_interval_hours_lag1h"].replace(0.0, np.nan)
    out["deriv_funding_annualized"] = x["funding_rate_lag1h"] * (24.0 / interval) * 365.0
    out["deriv_oi_value"] = x["sum_open_interest_value_lag1h"]
    out["deriv_log_oi_value"] = np.log(x["sum_open_interest_value_lag1h"].clip(lower=1.0))
    out["deriv_open_interest"] = x["sum_open_interest_lag1h"]
    out["deriv_account_long_short_ratio"] = np.log(x["count_long_short_ratio_lag1h"].clip(lower=1e-6))
    out["deriv_top_count_long_short_ratio"] = np.log(x["count_toptrader_long_short_ratio_lag1h"].clip(lower=1e-6))
    out["deriv_top_sum_long_short_ratio"] = np.log(x["sum_toptrader_long_short_ratio_lag1h"].clip(lower=1e-6))
    out["deriv_taker_long_short_ratio"] = np.log(x["sum_taker_long_short_vol_ratio_lag1h"].clip(lower=1e-6))
    out["deriv_mark_index_basis"] = x["mark_index_basis_lag1h"]
    out["deriv_premium_index"] = x["premium_close_lag1h"]
    add_derivative_freshness_columns(out, x)

    for col in [
        "deriv_funding_rate",
        "deriv_funding_annualized",
        "deriv_log_oi_value",
        "deriv_account_long_short_ratio",
        "deriv_top_count_long_short_ratio",
        "deriv_top_sum_long_short_ratio",
        "deriv_taker_long_short_ratio",
        "deriv_mark_index_basis",
        "deriv_premium_index",
    ]:
        for span in [24, 72, 168, 336]:
            out[f"{col}_ewm_{span}"] = (
                out[col].ewm(span=span, min_periods=max(12, span // 4), adjust=False).mean().where(out[col].notna())
            )
        out[f"{col}_delta_24"] = out[col] - out[col].shift(24)
        out[f"{col}_delta_168"] = out[col] - out[col].shift(168)
        rolling_mean = out[col].rolling(168, min_periods=48).mean()
        rolling_std = out[col].rolling(168, min_periods=48).std(ddof=0)
        out[f"{col}_z_168"] = (out[col] - rolling_mean) / rolling_std.replace(0.0, np.nan)

    out["deriv_crowding_pressure_72"] = (
        out["deriv_funding_rate_ewm_72"]
        + 0.35 * out["deriv_premium_index_ewm_72"]
        + 0.25 * out["deriv_account_long_short_ratio_ewm_72"]
        + 0.25 * out["deriv_top_sum_long_short_ratio_ewm_72"]
    )
    out["deriv_taker_pressure_24"] = out["deriv_taker_long_short_ratio_ewm_24"]
    out["deriv_oi_growth_72_336"] = out["deriv_log_oi_value_ewm_72"] - out["deriv_log_oi_value_ewm_336"]
    out["deriv_funding_accel_24_168"] = out["deriv_funding_rate_ewm_24"] - out["deriv_funding_rate_ewm_168"]
    out["deriv_premium_accel_24_168"] = out["deriv_premium_index_ewm_24"] - out["deriv_premium_index_ewm_168"]
    out["deriv_abs_funding_stress_168"] = out["deriv_funding_rate_z_168"].abs()
    out["deriv_abs_basis_stress_168"] = out["deriv_mark_index_basis_z_168"].abs()
    return out


def add_derivative_freshness_columns(out: pd.DataFrame, frame: pd.DataFrame) -> None:
    freshness_cols: list[str] = []
    age_cols: list[str] = []
    stale_cols: list[str] = []
    for output_col, source_cols in DERIVATIVE_SOURCE_MAP.items():
        ages = []
        observed_flags = []
        max_expected_age = 0.0
        for source_col in source_cols:
            age_col = f"{source_col}_source_age_hours"
            observed_col = f"{source_col}_source_observed"
            if age_col in frame:
                ages.append(frame[age_col].shift(1) + 1.0)
                max_expected_age = max(max_expected_age, SOURCE_MAX_CURRENT_AGE_HOURS.get(source_col, 2.0))
            if observed_col in frame:
                observed_flags.append(frame[observed_col].shift(1))
        if ages:
            source_age = pd.concat(ages, axis=1).max(axis=1)
        else:
            source_age = pd.Series(np.nan, index=out.index)
        if observed_flags:
            source_observed = pd.concat(observed_flags, axis=1).all(axis=1)
        else:
            source_observed = pd.Series(False, index=out.index)

        stale = source_age > max_expected_age
        decay_denominator = max(max_expected_age, 1.0)
        decay = 1.0 - (source_age - max_expected_age).clip(lower=0.0) / decay_denominator
        freshness = np.where(source_age.notna(), np.where(stale, decay.clip(lower=0.0, upper=1.0), 1.0), np.nan)

        age_name = f"{output_col}_source_age_hours"
        observed_name = f"{output_col}_source_observed_lag1h"
        stale_name = f"{output_col}_is_stale"
        freshness_name = f"{output_col}_freshness_weight"
        out[age_name] = source_age
        out[observed_name] = source_observed.astype(float).where(source_age.notna())
        out[stale_name] = stale.astype(float).where(source_age.notna())
        out[freshness_name] = freshness
        age_cols.append(age_name)
        stale_cols.append(stale_name)
        freshness_cols.append(freshness_name)

    if age_cols:
        out["deriv_data_quality_max_source_age_hours"] = out[age_cols].max(axis=1)
    if freshness_cols:
        out["deriv_data_quality_min_freshness_weight"] = out[freshness_cols].min(axis=1)
    if stale_cols:
        out["deriv_data_quality_any_stale"] = out[stale_cols].fillna(0.0).astype(float).gt(0.0).any(axis=1).astype(float)


def write_metadata(out: pd.DataFrame, path: Path) -> None:
    import json

    meta = {
        "rows": int(len(out)),
        "symbols": sorted(out["symbol"].unique().tolist()),
        "timestamp_min": str(out["timestamp"].min()),
        "timestamp_max": str(out["timestamp"].max()),
        "columns": out.columns.tolist(),
        "coverage": {col: float(out[col].notna().mean()) for col in out.columns if col not in {"timestamp", "symbol"}},
        "ex_ante_rule": "All raw Binance derivatives fields are shifted by 1 hour before feature construction.",
        "freshness_rule": (
            "For each raw source, the panel records source age at the decision timestamp after the 1h lag, "
            "a stale flag versus source-specific maximum expected age, and a freshness weight."
        ),
    }
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
