from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import make_wide


@dataclass(frozen=True)
class ResidualConfig:
    beta_window: int
    min_beta_obs: int


def build_sector_lookup(sectors: dict[str, list[str]]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for sector, symbols in sectors.items():
        for symbol in symbols:
            lookup[symbol] = sector
    return lookup


def _exclusive_mean(ret: pd.DataFrame, symbol: str, members: list[str] | None = None) -> pd.Series:
    cols = members if members is not None else list(ret.columns)
    cols = [col for col in cols if col in ret.columns and col != symbol]
    if not cols:
        return pd.Series(0.0, index=ret.index)
    return ret[cols].mean(axis=1)


def add_factor_inputs(data: pd.DataFrame, sectors: dict[str, list[str]]) -> pd.DataFrame:
    ret = make_wide(data, "log_ret")
    sector_lookup = build_sector_lookup(sectors)
    out = data.copy()

    pieces: list[pd.DataFrame] = []
    for symbol, symbol_df in out.groupby("symbol", sort=False):
        symbol_df = symbol_df.sort_values("timestamp").copy()
        sector = sector_lookup.get(symbol)
        sector_members = sectors.get(sector, []) if sector is not None else []

        symbol_df["factor_btc_ret"] = 0.0 if symbol == "BTCUSDT" else ret["BTCUSDT"].reindex(symbol_df["timestamp"]).to_numpy()
        symbol_df["factor_eth_ret"] = 0.0 if symbol == "ETHUSDT" else ret["ETHUSDT"].reindex(symbol_df["timestamp"]).to_numpy()
        symbol_df["factor_market_ret"] = _exclusive_mean(ret, symbol).reindex(symbol_df["timestamp"]).to_numpy()
        symbol_df["factor_sector_ret"] = _exclusive_mean(ret, symbol, sector_members).reindex(symbol_df["timestamp"]).to_numpy()
        pieces.append(symbol_df)

    return pd.concat(pieces, ignore_index=True).sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def rolling_residualize(
    data: pd.DataFrame,
    cfg: ResidualConfig,
    factor_cols: list[str] | None = None,
) -> pd.DataFrame:
    factor_cols = factor_cols or [
        "factor_btc_ret",
        "factor_eth_ret",
        "factor_market_ret",
        "factor_sector_ret",
    ]
    frames: list[pd.DataFrame] = []
    for symbol, symbol_df in data.groupby("symbol", sort=False):
        frames.append(_rolling_residualize_one(symbol_df.sort_values("timestamp").copy(), cfg, factor_cols))
    return pd.concat(frames, ignore_index=True).sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def _rolling_residualize_one(df: pd.DataFrame, cfg: ResidualConfig, factor_cols: list[str]) -> pd.DataFrame:
    y = df["log_ret"].astype(float).to_numpy()
    x = df[factor_cols].astype(float).to_numpy()
    n = len(df)
    p = len(factor_cols) + 1

    z = np.column_stack([np.ones(n), x])
    valid = np.isfinite(y) & np.isfinite(z).all(axis=1)
    z_valid = np.where(valid[:, None], z, np.nan)
    y_valid = np.where(valid, y, np.nan)

    xtx_sums = {}
    for a in range(p):
        for b in range(a, p):
            s = pd.Series(z_valid[:, a] * z_valid[:, b])
            xtx_sums[(a, b)] = s.rolling(cfg.beta_window, min_periods=cfg.min_beta_obs).sum().shift(1).to_numpy()

    xty_sums = {}
    for a in range(p):
        s = pd.Series(z_valid[:, a] * y_valid)
        xty_sums[a] = s.rolling(cfg.beta_window, min_periods=cfg.min_beta_obs).sum().shift(1).to_numpy()

    count = (
        pd.Series(valid.astype(float))
        .rolling(cfg.beta_window, min_periods=cfg.min_beta_obs)
        .sum()
        .shift(1)
        .to_numpy()
    )

    betas = np.full((n, p), np.nan)
    predicted = np.full(n, np.nan)
    residual = np.full(n, np.nan)

    for i in range(n):
        if not np.isfinite(count[i]) or count[i] < cfg.min_beta_obs or not np.isfinite(z[i]).all():
            continue

        xtx = np.empty((p, p), dtype=float)
        for a in range(p):
            for b in range(p):
                key = (a, b) if a <= b else (b, a)
                xtx[a, b] = xtx_sums[key][i]
        xty = np.array([xty_sums[a][i] for a in range(p)], dtype=float)

        if not np.isfinite(xtx).all() or not np.isfinite(xty).all():
            continue

        coef = np.linalg.pinv(xtx, rcond=1e-10) @ xty
        betas[i, :] = coef
        predicted[i] = float(z[i] @ coef)
        residual[i] = y[i] - predicted[i] if np.isfinite(y[i]) else np.nan

    out = df.copy()
    out["beta_intercept"] = betas[:, 0]
    out["beta_btc"] = betas[:, 1]
    out["beta_eth"] = betas[:, 2]
    out["beta_market"] = betas[:, 3]
    out["beta_sector"] = betas[:, 4]
    out["predicted_log_ret"] = predicted
    out["residual_log_ret"] = residual
    return out
