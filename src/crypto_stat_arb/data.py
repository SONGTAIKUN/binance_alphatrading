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
    data["log_quote_volume"] = np.log(data["quote_volume"].astype(float).clip(lower=1.0))
    return data


def make_wide(data: pd.DataFrame, value_col: str) -> pd.DataFrame:
    return data.pivot(index="timestamp", columns="symbol", values=value_col).sort_index()
