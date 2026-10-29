from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FactorModelSpec:
    """Versioned rolling factor-model contract for one system layer."""

    name: str
    namespace: str
    factor_cols: tuple[str, ...]
    beta_window: int
    min_beta_obs: int

    def __post_init__(self) -> None:
        if self.namespace not in {"alpha", "risk"}:
            raise ValueError("namespace must be 'alpha' or 'risk'")
        if not self.factor_cols:
            raise ValueError("factor_cols cannot be empty")
        if self.min_beta_obs <= len(self.factor_cols) + 1:
            raise ValueError("min_beta_obs is too small for the factor count")
        if self.beta_window < self.min_beta_obs:
            raise ValueError("beta_window must be >= min_beta_obs")


NO_SECTOR_FACTORS = (
    "factor_btc_ret",
    "factor_eth_ret",
    "factor_market_ret",
)


def factor_key(factor_col: str) -> str:
    return factor_col.removeprefix("factor_").removesuffix("_ret")


def namespaced_factor_col(namespace: str, factor_col: str) -> str:
    return f"{namespace}_factor_{factor_key(factor_col)}_ret"


def namespaced_beta_col(namespace: str, factor_col: str) -> str:
    return f"{namespace}_beta_{factor_key(factor_col)}"


def rolling_factor_model(data: pd.DataFrame, spec: FactorModelSpec) -> pd.DataFrame:
    """Fit past-only rolling OLS and write only namespaced model fields.

    The coefficient used at timestamp t is estimated from rows ending at t-1.
    Alpha and portfolio-risk models therefore cannot silently overwrite each
    other's beta fields.
    """

    required = {"timestamp", "symbol", "log_ret", *spec.factor_cols}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Missing factor-model columns: {missing}")

    pieces: list[pd.DataFrame] = []
    for _, frame in data.groupby("symbol", sort=False):
        frame = frame.sort_values("timestamp").copy()
        pieces.append(_rolling_factor_model_one(frame, spec))
    return pd.concat(pieces, ignore_index=True).sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def _rolling_factor_model_one(frame: pd.DataFrame, spec: FactorModelSpec) -> pd.DataFrame:
    y = frame["log_ret"].astype(float).to_numpy()
    x = frame[list(spec.factor_cols)].astype(float).to_numpy()
    n = len(frame)
    p = x.shape[1] + 1
    design = np.column_stack([np.ones(n), x])
    valid = np.isfinite(y) & np.isfinite(design).all(axis=1)
    valid_design = np.where(valid[:, None], design, np.nan)
    valid_y = np.where(valid, y, np.nan)

    xtx_sums: dict[tuple[int, int], np.ndarray] = {}
    for left in range(p):
        for right in range(left, p):
            values = valid_design[:, left] * valid_design[:, right]
            xtx_sums[(left, right)] = (
                pd.Series(values)
                .rolling(spec.beta_window, min_periods=spec.min_beta_obs)
                .sum()
                .shift(1)
                .to_numpy()
            )
    xty_sums = {
        col: (
            pd.Series(valid_design[:, col] * valid_y)
            .rolling(spec.beta_window, min_periods=spec.min_beta_obs)
            .sum()
            .shift(1)
            .to_numpy()
        )
        for col in range(p)
    }
    counts = (
        pd.Series(valid.astype(float))
        .rolling(spec.beta_window, min_periods=spec.min_beta_obs)
        .sum()
        .shift(1)
        .to_numpy()
    )

    coefficients = np.full((n, p), np.nan)
    predicted = np.full(n, np.nan)
    residual = np.full(n, np.nan)
    for row in range(n):
        if not np.isfinite(counts[row]) or counts[row] < spec.min_beta_obs:
            continue
        if not np.isfinite(design[row]).all():
            continue
        xtx = np.empty((p, p), dtype=float)
        for left in range(p):
            for right in range(p):
                key = (left, right) if left <= right else (right, left)
                xtx[left, right] = xtx_sums[key][row]
        xty = np.array([xty_sums[col][row] for col in range(p)], dtype=float)
        if not np.isfinite(xtx).all() or not np.isfinite(xty).all():
            continue
        coef = np.linalg.pinv(xtx, rcond=1e-10) @ xty
        coefficients[row] = coef
        predicted[row] = float(design[row] @ coef)
        residual[row] = y[row] - predicted[row] if np.isfinite(y[row]) else np.nan

    namespace = spec.namespace
    out = frame.copy()
    for factor_col in spec.factor_cols:
        out[namespaced_factor_col(namespace, factor_col)] = out[factor_col]
    out[f"{namespace}_beta_intercept"] = coefficients[:, 0]
    for index, factor_col in enumerate(spec.factor_cols, start=1):
        out[namespaced_beta_col(namespace, factor_col)] = coefficients[:, index]
    out[f"{namespace}_predicted_log_ret"] = predicted
    out[f"{namespace}_residual_log_ret"] = residual
    out[f"{namespace}_model_version"] = spec.name
    return out


def add_portfolio_risk_aliases(data: pd.DataFrame) -> pd.DataFrame:
    """Create the legacy allocator interface from explicit risk-model fields."""

    mapping = {
        "risk_beta_btc": "beta_btc",
        "risk_beta_eth": "beta_eth",
        "risk_beta_market": "beta_market",
    }
    missing = sorted(set(mapping).difference(data.columns))
    if missing:
        raise ValueError(f"Missing risk beta columns for allocator adapter: {missing}")
    out = data.copy()
    for source, target in mapping.items():
        out[target] = out[source]
    return out
