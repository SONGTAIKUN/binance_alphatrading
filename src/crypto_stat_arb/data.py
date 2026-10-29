from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


OHLCV_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
    "quote_volume": "sum",
    "trades": "sum",
    "taker_buy_base": "sum",
    "taker_buy_quote": "sum",
}


def load_symbol_1h(klines_dir: str | Path, symbol: str) -> pd.DataFrame:
    path = Path(klines_dir) / symbol / "15m.csv"
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    df = df.set_index("timestamp")

    hourly = df.resample("1h", label="left", closed="left").agg(OHLCV_AGG)
    counts = df["close"].resample("1h", label="left", closed="left").count()
    hourly = hourly.loc[counts == 4].copy()
    hourly["symbol"] = symbol
    hourly = hourly.reset_index()
    return hourly


def load_universe_1h(klines_dir: str | Path, symbols: list[str]) -> pd.DataFrame:
    frames = [load_symbol_1h(klines_dir, symbol) for symbol in symbols]
    data = pd.concat(frames, ignore_index=True)

    counts = data.groupby("timestamp")["symbol"].nunique()
    complete_ts = counts[counts == len(symbols)].index
    data = data[data["timestamp"].isin(complete_ts)].copy()
    data = data.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    data["log_close"] = np.log(data["close"].astype(float))
    data["log_ret"] = data.groupby("symbol")["log_close"].diff()
    data["simple_ret"] = data.groupby("symbol")["close"].pct_change()
    data["log_volume"] = np.log(data["volume"].astype(float).clip(lower=1.0))
    data["log_quote_volume"] = np.log(data["quote_volume"].astype(float).clip(lower=1.0))
    data = add_robust_ohlcv_preprocessing(data)
    return data


def add_robust_ohlcv_preprocessing(data: pd.DataFrame) -> pd.DataFrame:
    """Add ex-ante-safe OHLCV hygiene features without changing raw fields."""

    out = data.copy()
    by_symbol = out.groupby("symbol", group_keys=False)

    out["log_volume_diff_24"] = by_symbol["log_volume"].diff(24)
    out["log_volume_diff_168"] = by_symbol["log_volume"].diff(168)
    out["log_quote_volume_diff_24"] = by_symbol["log_quote_volume"].diff(24)
    out["log_quote_volume_diff_168"] = by_symbol["log_quote_volume"].diff(168)

    timestamp_totals = out.groupby("timestamp", sort=False)[["volume", "quote_volume"]].transform("sum")
    out["volume_share"] = safe_divide(out["volume"], timestamp_totals["volume"])
    out["quote_volume_share"] = safe_divide(out["quote_volume"], timestamp_totals["quote_volume"])
    out["log_volume_share"] = np.log(out["volume_share"].clip(lower=1e-12))
    out["log_quote_volume_share"] = np.log(out["quote_volume_share"].clip(lower=1e-12))
    out["volume_share_change_24"] = by_symbol["log_volume_share"].diff(24)
    out["quote_volume_share_change_24"] = by_symbol["log_quote_volume_share"].diff(24)

    base_ratio = safe_divide(out["taker_buy_base"], out["volume"]).clip(0.0, 1.0)
    quote_ratio = safe_divide(out["taker_buy_quote"], out["quote_volume"]).clip(0.0, 1.0)
    out["taker_buy_base_ratio"] = base_ratio
    out["taker_buy_quote_ratio"] = quote_ratio
    out["taker_imbalance_base"] = 2.0 * base_ratio - 1.0
    out["taker_imbalance_quote"] = 2.0 * quote_ratio - 1.0
    out["taker_imbalance_quote_ewm_24"] = by_symbol["taker_imbalance_quote"].transform(
        lambda s: s.ewm(span=24, min_periods=8, adjust=False).mean()
    )
    out["taker_imbalance_quote_ewm_168"] = by_symbol["taker_imbalance_quote"].transform(
        lambda s: s.ewm(span=168, min_periods=48, adjust=False).mean()
    )
    out["taker_imbalance_quote_delta_24"] = out["taker_imbalance_quote_ewm_24"] - out["taker_imbalance_quote_ewm_168"]

    out["log_high_low_range"] = np.log(out["high"].astype(float).clip(lower=1e-12) / out["low"].astype(float).clip(lower=1e-12))
    out["close_location_value"] = (
        2.0 * safe_divide(out["close"] - out["low"], out["high"] - out["low"]).clip(0.0, 1.0) - 1.0
    )
    out["amihud_1h"] = safe_divide(out["log_ret"].abs(), out["quote_volume"].astype(float).clip(lower=1.0) / 1_000_000.0)

    out["log_ret_winsor_720"] = by_symbol["log_ret"].transform(rolling_past_winsor)
    past_vol = by_symbol["log_ret"].transform(lambda s: s.rolling(720, min_periods=240).std(ddof=0).shift(1))
    out["log_ret_robust_z_720"] = safe_divide(out["log_ret"], past_vol).clip(-8.0, 8.0)
    out["abs_log_ret_outlier_720"] = (out["log_ret"].abs() > 8.0 * past_vol).astype(float).where(past_vol.notna())
    return out


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = denominator.astype(float).replace(0.0, np.nan)
    return numerator.astype(float) / denom


def rolling_past_winsor(series: pd.Series, window: int = 720, min_periods: int = 240) -> pd.Series:
    lower = series.rolling(window, min_periods=min_periods).quantile(0.01).shift(1)
    upper = series.rolling(window, min_periods=min_periods).quantile(0.99).shift(1)
    return series.clip(lower=lower, upper=upper)


def make_wide(data: pd.DataFrame, value_col: str) -> pd.DataFrame:
    return data.pivot(index="timestamp", columns="symbol", values=value_col).sort_index()
