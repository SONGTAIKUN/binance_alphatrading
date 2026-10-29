from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


MEDIUM_WEIGHTS = {48: 0.20, 72: 0.30, 96: 0.30, 168: 0.20}
LOW_TURNOVER_WEIGHTS = {72: 0.25, 96: 0.35, 168: 0.40}


@dataclass(frozen=True)
class SlowLabelSpec:
    horizons: tuple[int, ...] = (48, 72, 96, 168)
    execution_delay_bars: int = 1
    volatility_lookback: int = 336
    path_lambda: float = 0.50
    residual_col: str = "alpha_residual_log_ret"
    log_close_col: str = "log_close"
    raw_return_col: str = "log_ret"
    factor_cols: tuple[str, ...] = (
        "alpha_factor_btc_ret",
        "alpha_factor_eth_ret",
        "alpha_factor_market_ret",
    )
    beta_cols: tuple[str, ...] = (
        "alpha_beta_btc",
        "alpha_beta_eth",
        "alpha_beta_market",
    )

    def __post_init__(self) -> None:
        if len(self.factor_cols) != len(self.beta_cols):
            raise ValueError("factor_cols and beta_cols must have equal length")
        if tuple(sorted(set(self.horizons))) != self.horizons:
            raise ValueError("horizons must be unique and sorted")
        if any(horizon <= 0 for horizon in self.horizons):
            raise ValueError("horizons must be positive")


def build_slow_residual_labels(panel: pd.DataFrame, spec: SlowLabelSpec) -> pd.DataFrame:
    required = {
        "timestamp",
        "symbol",
        spec.log_close_col,
        spec.raw_return_col,
        spec.residual_col,
        *spec.factor_cols,
        *spec.beta_cols,
    }
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ValueError(f"Missing slow-label columns: {missing}")

    pieces: list[pd.DataFrame] = []
    min_vol_obs = max(72, spec.volatility_lookback // 3)
    for _, frame in panel.groupby("symbol", sort=False):
        frame = frame.sort_values("timestamp").copy()
        out = frame[["timestamp", "symbol"]].copy()
        residual = frame[spec.residual_col].astype(float)
        rv = np.sqrt(
            residual.pow(2)
            .rolling(spec.volatility_lookback, min_periods=min_vol_obs)
            .sum()
        )
        out[f"ex_ante_resid_rv_{spec.volatility_lookback}h"] = rv
        paths = _residual_paths(frame, spec)

        for horizon in spec.horizons:
            path = paths[horizon]
            terminal = path["terminal"]
            denominator = rv * np.sqrt(horizon / spec.volatility_lookback)
            denominator = denominator.replace(0.0, np.nan)
            downside_adverse = (-path["min"]).clip(lower=0.0)
            upside_adverse = path["max"].clip(lower=0.0)
            terminal_norm = terminal / denominator
            downside_adverse_norm = downside_adverse / denominator
            upside_adverse_norm = upside_adverse / denominator
            signed_adverse = pd.Series(
                np.where(terminal >= 0.0, downside_adverse, upside_adverse),
                index=frame.index,
            )
            signed_adverse_norm = signed_adverse / denominator
            path_signed_norm = pd.Series(
                np.where(
                    terminal >= 0.0,
                    terminal_norm - spec.path_lambda * downside_adverse_norm,
                    terminal_norm + spec.path_lambda * upside_adverse_norm,
                ),
                index=frame.index,
            )
            path_long_norm = terminal_norm - spec.path_lambda * downside_adverse_norm

            start = frame[spec.log_close_col].shift(-spec.execution_delay_bars)
            end = frame[spec.log_close_col].shift(-(spec.execution_delay_bars + horizon))
            forward_log_return = end - start
            out[f"fwd_log_ret_h{horizon}"] = forward_log_return
            out[f"fwd_simple_ret_h{horizon}"] = np.expm1(forward_log_return)
            out[f"label_resid_ret_h{horizon}"] = terminal
            out[f"label_resid_ret_norm_h{horizon}"] = terminal_norm
            out[f"label_path_signed_norm_h{horizon}"] = path_signed_norm
            out[f"label_path_long_norm_h{horizon}"] = path_long_norm
            out[f"label_path_min_resid_h{horizon}"] = path["min"]
            out[f"label_path_max_resid_h{horizon}"] = path["max"]
            out[f"label_downside_adverse_norm_h{horizon}"] = downside_adverse_norm
            out[f"label_upside_adverse_norm_h{horizon}"] = upside_adverse_norm
            out[f"label_signed_adverse_norm_h{horizon}"] = signed_adverse_norm
        pieces.append(out)

    labels = pd.concat(pieces, ignore_index=True).sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    _add_cross_sectional_rank_labels(labels, spec.horizons)
    _add_composites(labels)
    _add_horizon_consistency(labels, spec.horizons)
    return labels


def _residual_paths(frame: pd.DataFrame, spec: SlowLabelSpec) -> dict[int, dict[str, pd.Series]]:
    max_horizon = max(spec.horizons)
    cumulative_raw = pd.Series(0.0, index=frame.index)
    cumulative_factors = {factor: pd.Series(0.0, index=frame.index) for factor in spec.factor_cols}
    min_path = pd.Series(np.inf, index=frame.index)
    max_path = pd.Series(-np.inf, index=frame.index)
    valid = pd.Series(True, index=frame.index)
    paths: dict[int, dict[str, pd.Series]] = {}

    for step in range(1, max_horizon + 1):
        offset = spec.execution_delay_bars + step
        raw = frame[spec.raw_return_col].shift(-offset)
        cumulative_raw = cumulative_raw + raw.fillna(0.0)
        valid = valid & raw.notna()
        explained = pd.Series(0.0, index=frame.index)
        for factor_col, beta_col in zip(spec.factor_cols, spec.beta_cols, strict=True):
            factor = frame[factor_col].shift(-offset)
            cumulative_factors[factor_col] = cumulative_factors[factor_col] + factor.fillna(0.0)
            valid = valid & factor.notna() & frame[beta_col].notna()
            explained = explained + frame[beta_col] * cumulative_factors[factor_col]
        residual_path = cumulative_raw - explained
        min_path = pd.Series(np.minimum(min_path.to_numpy(), residual_path.to_numpy()), index=frame.index)
        max_path = pd.Series(np.maximum(max_path.to_numpy(), residual_path.to_numpy()), index=frame.index)
        if step in spec.horizons:
            paths[step] = {
                "terminal": residual_path.where(valid).copy(),
                "min": min_path.where(valid).copy(),
                "max": max_path.where(valid).copy(),
            }
    return paths


def centered_rank_by_timestamp(frame: pd.DataFrame, value_col: str) -> pd.Series:
    grouped = frame.groupby("timestamp", sort=False)[value_col]
    rank = grouped.rank(method="average")
    count = grouped.transform("count")
    centered = 2.0 * (rank - 1.0) / (count - 1.0) - 1.0
    return centered.where(count > 1)


def _add_cross_sectional_rank_labels(labels: pd.DataFrame, horizons: tuple[int, ...]) -> None:
    for horizon in horizons:
        labels[f"label_resid_rank_centered_h{horizon}"] = centered_rank_by_timestamp(
            labels, f"label_resid_ret_norm_h{horizon}"
        )
        labels[f"label_path_rank_centered_h{horizon}"] = centered_rank_by_timestamp(
            labels, f"label_path_signed_norm_h{horizon}"
        )


def _weighted_sum(labels: pd.DataFrame, weights: dict[int, float], prefix: str) -> pd.Series:
    total = pd.Series(0.0, index=labels.index)
    valid = pd.Series(True, index=labels.index)
    for horizon, weight in weights.items():
        col = f"{prefix}{horizon}"
        total = total + weight * labels[col].fillna(0.0)
        valid = valid & labels[col].notna()
    return total.where(valid)


def _add_composites(labels: pd.DataFrame) -> None:
    labels["label_slow_medium"] = _weighted_sum(labels, MEDIUM_WEIGHTS, "label_resid_rank_centered_h")
    labels["label_slow_low_turnover"] = _weighted_sum(
        labels, LOW_TURNOVER_WEIGHTS, "label_resid_rank_centered_h"
    )
    labels["label_path_slow_medium"] = _weighted_sum(labels, MEDIUM_WEIGHTS, "label_path_rank_centered_h")
    labels["label_path_slow_low_turnover"] = _weighted_sum(
        labels, LOW_TURNOVER_WEIGHTS, "label_path_rank_centered_h"
    )
    for col in (
        "label_slow_medium",
        "label_slow_low_turnover",
        "label_path_slow_medium",
        "label_path_slow_low_turnover",
    ):
        labels[f"{col}_rank_centered"] = centered_rank_by_timestamp(labels, col)


def _add_horizon_consistency(labels: pd.DataFrame, horizons: tuple[int, ...]) -> None:
    labels["label_horizon_sign_consistency"] = _pairwise_sign_agreement(
        labels[[f"label_resid_rank_centered_h{horizon}" for horizon in horizons]]
    )
    labels["label_path_horizon_sign_consistency"] = _pairwise_sign_agreement(
        labels[[f"label_path_rank_centered_h{horizon}" for horizon in horizons]]
    )


def _pairwise_sign_agreement(values: pd.DataFrame) -> pd.Series:
    arr = values.to_numpy(dtype=float)
    finite = np.isfinite(arr).all(axis=1)
    signs = np.sign(arr)
    pairs = []
    for left in range(arr.shape[1]):
        for right in range(left + 1, arr.shape[1]):
            pairs.append((signs[:, left] == signs[:, right]).astype(float))
    result = np.mean(np.column_stack(pairs), axis=1) if pairs else np.full(len(arr), np.nan)
    result[~finite] = np.nan
    return pd.Series(result, index=values.index)
