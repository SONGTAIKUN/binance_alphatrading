from __future__ import annotations

import numpy as np
import pandas as pd


FACTOR_COLS = [
    "factor_btc_ret",
    "factor_eth_ret",
    "factor_market_ret",
    "factor_sector_ret",
]

BETA_COLS = [
    "beta_btc",
    "beta_eth",
    "beta_market",
    "beta_sector",
]


def rolling_zscore(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    min_periods = min_periods or max(8, window // 3)
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std(ddof=0)
    return (series - mean) / std.replace(0.0, np.nan)


def add_residual_features(data: pd.DataFrame, residual_z_window: int, windows: dict[str, int]) -> pd.DataFrame:
    short = int(windows.get("short", 4))
    medium = int(windows.get("medium", 24))
    long = int(windows.get("long", 168))

    frames: list[pd.DataFrame] = []
    for _, df in data.groupby("symbol", sort=False):
        df = df.sort_values("timestamp").copy()
        resid = df["residual_log_ret"]
        raw = df["log_ret"]

        df["resid_z"] = rolling_zscore(resid, residual_z_window)
        df["resid_mom_short"] = resid.rolling(short, min_periods=short).sum()
        df["resid_mom_medium"] = resid.rolling(medium, min_periods=medium).sum()
        df["raw_mom_short"] = raw.rolling(short, min_periods=short).sum()
        df["raw_mom_medium"] = raw.rolling(medium, min_periods=medium).sum()
        df["realized_vol_medium"] = raw.rolling(medium, min_periods=medium).std(ddof=0)
        df["realized_vol_long"] = raw.rolling(long, min_periods=max(24, long // 3)).std(ddof=0)
        df["idio_vol_medium"] = resid.rolling(medium, min_periods=medium).std(ddof=0)
        df["idio_vol_long"] = resid.rolling(long, min_periods=max(24, long // 3)).std(ddof=0)
        df["liquidity_z"] = rolling_zscore(df["log_quote_volume"], long)
        df["turnover_pressure"] = df["quote_volume"] / df["quote_volume"].rolling(long, min_periods=max(24, long // 3)).mean()
        frames.append(df)

    return pd.concat(frames, ignore_index=True).sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def add_cross_sectional_features(data: pd.DataFrame, feature_cols: list[str] | None = None) -> pd.DataFrame:
    feature_cols = feature_cols or [
        "resid_z",
        "resid_mom_short",
        "resid_mom_medium",
        "raw_mom_short",
        "raw_mom_medium",
        "realized_vol_medium",
        "idio_vol_medium",
        "liquidity_z",
        "turnover_pressure",
        "beta_btc",
        "beta_eth",
        "beta_market",
        "beta_sector",
    ]

    out = data.copy()
    grouped = out.groupby("timestamp", sort=False)
    for col in feature_cols:
        if col not in out.columns:
            continue
        out[f"cs_rank_{col}"] = grouped[col].rank(pct=True, method="average")
        mean = grouped[col].transform("mean")
        std = grouped[col].transform("std").replace(0.0, np.nan)
        out[f"cs_zscore_{col}"] = (out[col] - mean) / std
    return out


def add_baseline_score(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    out["score_baseline"] = -out["cs_zscore_resid_z"]
    return out


def add_forward_labels(
    data: pd.DataFrame,
    horizons: list[int],
    execution_delay_bars: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for _, df in data.groupby("symbol", sort=False):
        df = df.sort_values("timestamp").copy()
        for horizon in horizons:
            start = df["log_close"].shift(-execution_delay_bars)
            end = df["log_close"].shift(-(execution_delay_bars + horizon))
            fwd_log_ret = end - start
            df[f"fwd_log_ret_h{horizon}"] = fwd_log_ret
            df[f"fwd_simple_ret_h{horizon}"] = np.expm1(fwd_log_ret)

            factor_fwd = []
            for factor_col, beta_col in zip(FACTOR_COLS, BETA_COLS, strict=True):
                future_factor_sum = _forward_sum(df[factor_col], execution_delay_bars + 1, horizon)
                factor_fwd.append(df[beta_col] * future_factor_sum)

            explained = sum(factor_fwd)
            df[f"label_resid_ret_h{horizon}"] = fwd_log_ret - explained
        frames.append(df)

    out = pd.concat(frames, ignore_index=True).sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    for horizon in horizons:
        col = f"label_resid_ret_h{horizon}"
        out[f"label_rank_h{horizon}"] = out.groupby("timestamp")[col].rank(pct=True, method="average")
    return out


def _forward_sum(series: pd.Series, start_offset: int, horizon: int) -> pd.Series:
    total = pd.Series(0.0, index=series.index)
    valid = pd.Series(True, index=series.index)
    for offset in range(start_offset, start_offset + horizon):
        shifted = series.shift(-offset)
        total = total + shifted.fillna(0.0)
        valid = valid & shifted.notna()
    return total.where(valid)
