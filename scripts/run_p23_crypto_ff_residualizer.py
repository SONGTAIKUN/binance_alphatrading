#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from crypto_stat_arb.config import ensure_dir, load_config
from crypto_stat_arb.features import rolling_zscore
from crypto_stat_arb.portfolio import PortfolioConfig, run_long_short_backtest, summarize_equity_curve


EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "baseline_locked_v1"
DEFAULT_CONFIG = EXPERIMENT_ROOT / "configs" / "baseline_locked_v1.yaml"
DEFAULT_ASSET_PANEL = EXPERIMENT_ROOT / "inputs" / "long_history" / "baseline_panel" / "asset_residual_panel.csv"
DEFAULT_DERIVATIVES_PANEL = (
    EXPERIMENT_ROOT / "inputs" / "long_history" / "binance_derivatives" / "derivatives_hourly_panel.csv.gz"
)
DEFAULT_FOLDS = EXPERIMENT_ROOT / "outputs" / "long_history_nested_oos" / "inputs" / "long_formula_selections.csv"
DEFAULT_P15C_RUN = (
    EXPERIMENT_ROOT / "outputs" / "long_history_nested_oos" / "p15c_nested" / "nested_extreme_cluster_run"
)
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "outputs" / "p23_crypto_ff_residualizer"

FACTOR_NAMES = ["MKT", "LIQ", "MOM", "REV", "CARRY", "CROWDING", "VOL", "BETA"]
FF_BETA_COLS = [f"ff_beta_{name}" for name in FACTOR_NAMES]
LABEL_HORIZONS = [48, 72, 96, 168]

ASSET_COLS = [
    "timestamp",
    "symbol",
    "log_close",
    "log_ret",
    "simple_ret",
    "residual_log_ret",
    "resid_z",
    "score_baseline",
    "realized_vol_long",
    "log_quote_volume",
    "beta_btc",
    "beta_eth",
    "beta_market",
    "beta_sector",
    "fwd_log_ret_h48",
    "fwd_simple_ret_h48",
    "label_resid_ret_h48",
    "fwd_log_ret_h72",
    "fwd_simple_ret_h72",
    "label_resid_ret_h72",
    "fwd_log_ret_h96",
    "fwd_simple_ret_h96",
    "label_resid_ret_h96",
    "fwd_log_ret_h168",
    "fwd_simple_ret_h168",
    "label_resid_ret_h168",
]
DERIVATIVE_COLS = [
    "timestamp",
    "symbol",
    "deriv_funding_rate_ewm_168",
    "deriv_mark_index_basis_ewm_168",
    "deriv_premium_index_ewm_168",
    "deriv_oi_growth_72_336",
    "deriv_taker_pressure_24",
    "deriv_crowding_pressure_72",
    "deriv_data_quality_min_freshness_weight",
    "deriv_data_quality_any_stale",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="P23: rolling past-only crypto FF-like residualizer diagnostics and allocator smoke test."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--asset-panel", default=str(DEFAULT_ASSET_PANEL))
    parser.add_argument("--derivatives-panel", default=str(DEFAULT_DERIVATIVES_PANEL))
    parser.add_argument("--folds", default=str(DEFAULT_FOLDS))
    parser.add_argument("--p15c-run-dir", default=str(DEFAULT_P15C_RUN))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--horizon", type=int, default=72)
    parser.add_argument("--beta-window", type=int, default=720)
    parser.add_argument("--min-beta-obs", type=int, default=240)
    parser.add_argument("--top-bottom-frac", type=float, default=0.30)
    parser.add_argument("--min-leg-count", type=int, default=3)
    parser.add_argument("--cost-bps", type=float, nargs="+", default=[0, 6, 15, 20, 30])
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    horizon = int(args.horizon)
    beta_window = int(args.beta_window)
    min_beta_obs = int(args.min_beta_obs)

    print("[1/8] Loading asset and derivatives panels", flush=True)
    panel = load_panel(Path(args.asset_panel), Path(args.derivatives_panel))
    timestamps = pd.Index(sorted(panel["timestamp"].unique()), name="timestamp")
    symbols = sorted(panel["symbol"].unique())

    print("[2/8] Building stage-0 MKT residuals for MOM/REV characteristics", flush=True)
    returns = pivot(panel, "log_ret", timestamps, symbols)
    mkt = returns.mean(axis=1, skipna=True).rename("MKT")
    stage0 = build_stage0_residuals(returns, mkt, beta_window, min_beta_obs)

    print("[3/8] Building t-1 crypto FF-like factor-mimicking returns", flush=True)
    chars = build_characteristics(panel, timestamps, symbols, stage0)
    factor_returns, factor_diagnostics = build_factor_returns(
        returns=returns,
        chars=chars,
        top_bottom_frac=float(args.top_bottom_frac),
        min_leg_count=int(args.min_leg_count),
    )
    factor_returns.to_csv(output_dir / "crypto_ff_rolling_factor_returns.csv")
    factor_diagnostics.to_csv(output_dir / "crypto_ff_factor_portfolio_diagnostics.csv", index=False)

    print("[4/8] Rolling past-only multivariate residualization", flush=True)
    residual_panel = build_ff_residual_panel(
        base_panel=panel,
        factor_returns=factor_returns,
        returns=returns,
        beta_window=beta_window,
        min_beta_obs=min_beta_obs,
    )
    residual_panel = add_ff_features_and_labels(residual_panel, factor_returns, horizon_list=LABEL_HORIZONS, execution_delay=1)
    residual_panel.to_csv(output_dir / "crypto_ff_residual_panel.csv.gz", index=False)

    print("[5/8] Residual, beta, collinearity, and label-IC diagnostics", flush=True)
    factor_corr, factor_pairwise, factor_vif = factor_collinearity(factor_returns)
    factor_corr.to_csv(output_dir / "factor_correlation_matrix.csv")
    factor_pairwise.to_csv(output_dir / "factor_pairwise_correlations.csv", index=False)
    factor_vif.to_csv(output_dir / "factor_vif_condition.csv", index=False)

    residual_quality = build_residual_quality(residual_panel)
    residual_quality.to_csv(output_dir / "residual_quality_by_symbol.csv", index=False)
    residual_stability = build_residual_stability(residual_panel)
    residual_stability.to_csv(output_dir / "residual_rolling_stability.csv", index=False)
    beta_stability = build_beta_stability(residual_panel)
    beta_stability.to_csv(output_dir / "beta_stability.csv", index=False)
    ic_ts, ic_summary = build_label_ic(residual_panel, horizon)
    ic_ts.to_csv(output_dir / "label_ic_by_timestamp.csv", index=False)
    ic_summary.to_csv(output_dir / "label_ic_summary.csv", index=False)

    print("[6/8] Running strict-neutral allocator smoke test", flush=True)
    allocator = run_allocator_comparison(
        residual_panel=residual_panel,
        folds_path=Path(args.folds),
        config_path=Path(args.config),
        horizon=horizon,
    )
    write_allocator_outputs(output_dir, allocator, [float(x) for x in args.cost_bps], Path(args.p15c_run_dir))

    print("[7/8] Writing plots", flush=True)
    plot_factor_corr(output_dir, factor_corr)
    plot_allocator_curves(output_dir, allocator, [6.0, 15.0])

    print("[8/8] Writing report", flush=True)
    topline = build_topline(
        factor_vif=factor_vif,
        factor_pairwise=factor_pairwise,
        residual_quality=residual_quality,
        residual_stability=residual_stability,
        beta_stability=beta_stability,
        ic_summary=ic_summary,
        allocator_summary=allocator["summary"],
    )
    topline.to_csv(output_dir / "topline_findings.csv", index=False)
    metadata = {
        "asset_panel": str(Path(args.asset_panel).resolve()),
        "derivatives_panel": str(Path(args.derivatives_panel).resolve()),
        "folds": str(Path(args.folds).resolve()),
        "horizon": horizon,
        "beta_window": beta_window,
        "min_beta_obs": min_beta_obs,
        "top_bottom_frac": float(args.top_bottom_frac),
        "min_leg_count": int(args.min_leg_count),
        "past_only_rules": [
            "Factor portfolio weights for return t use t-1 characteristics.",
            "Stage-0 MKT beta and final FF betas are rolling estimates shifted by one bar.",
            "FF residual labels use beta at decision time t times future factor returns, matching the existing label convention.",
        ],
        "allocator_note": "Allocator smoke test uses the existing strict-neutral long/short allocator with the same BTC/ETH/market neutralization columns; it is not a full AlphaGPT/P15c rerun.",
        "small_universe_warning": "The universe has only 20 symbols, so factor-mimicking portfolios are noisier and self-inclusion is more material than in equity FF.",
    }
    (output_dir / "metadata.json").write_text(json.dumps(to_jsonable(metadata), indent=2), encoding="utf-8")
    write_report(output_dir, topline, factor_pairwise, factor_vif, residual_quality, residual_stability, beta_stability, ic_summary, allocator)
    print(f"Done: {output_dir}", flush=True)


def load_panel(asset_path: Path, derivatives_path: Path) -> pd.DataFrame:
    asset_header = set(pd.read_csv(asset_path, nrows=0).columns)
    asset_usecols = [col for col in ASSET_COLS if col in asset_header]
    asset = pd.read_csv(asset_path, usecols=asset_usecols, parse_dates=["timestamp"])
    asset["timestamp"] = pd.to_datetime(asset["timestamp"], utc=True)
    asset["symbol"] = asset["symbol"].astype(str)

    deriv_header = set(pd.read_csv(derivatives_path, nrows=0).columns)
    deriv_usecols = [col for col in DERIVATIVE_COLS if col in deriv_header]
    derivatives = pd.read_csv(derivatives_path, usecols=deriv_usecols, parse_dates=["timestamp"])
    derivatives["timestamp"] = pd.to_datetime(derivatives["timestamp"], utc=True)
    derivatives["symbol"] = derivatives["symbol"].astype(str)

    out = asset.merge(derivatives, on=["timestamp", "symbol"], how="left")
    return out.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def pivot(panel: pd.DataFrame, col: str, timestamps: pd.Index, symbols: list[str]) -> pd.DataFrame:
    return (
        panel.pivot(index="timestamp", columns="symbol", values=col)
        .reindex(index=timestamps, columns=symbols)
        .sort_index()
    )


def build_stage0_residuals(
    returns: pd.DataFrame,
    mkt: pd.Series,
    beta_window: int,
    min_beta_obs: int,
) -> dict[str, pd.DataFrame]:
    betas = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
    intercepts = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
    predicted = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
    residual = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
    x = pd.DataFrame({"MKT": mkt}, index=returns.index)
    for symbol in returns.columns:
        beta, pred, resid = rolling_ols_predict(returns[symbol], x, beta_window, min_beta_obs)
        intercepts[symbol] = beta["intercept"]
        betas[symbol] = beta["MKT"]
        predicted[symbol] = pred
        residual[symbol] = resid
    return {
        "intercept": intercepts,
        "beta_mkt": betas,
        "predicted": predicted,
        "residual": residual,
    }


def build_characteristics(
    panel: pd.DataFrame,
    timestamps: pd.Index,
    symbols: list[str],
    stage0: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    out = panel.copy()
    by_time = out.groupby("timestamp", sort=False)
    for col in [
        "deriv_funding_rate_ewm_168",
        "deriv_mark_index_basis_ewm_168",
        "deriv_premium_index_ewm_168",
        "deriv_oi_growth_72_336",
        "deriv_taker_pressure_24",
        "deriv_crowding_pressure_72",
    ]:
        if col not in out.columns:
            out[col] = np.nan
        mean = by_time[col].transform("mean")
        std = by_time[col].transform("std").replace(0.0, np.nan)
        out[f"cs_{col}"] = (out[col] - mean) / std

    out["char_carry"] = (
        out["cs_deriv_funding_rate_ewm_168"]
        + out["cs_deriv_mark_index_basis_ewm_168"]
        + out["cs_deriv_premium_index_ewm_168"]
    ) / 3.0
    out["char_crowding"] = (
        out["cs_deriv_oi_growth_72_336"]
        + out["cs_deriv_taker_pressure_24"]
        + out["cs_deriv_crowding_pressure_72"]
    ) / 3.0

    liq = -pivot(out, "log_quote_volume", timestamps, symbols).shift(1)
    vol = pivot(out, "realized_vol_long", timestamps, symbols).shift(1)
    carry = pivot(out, "char_carry", timestamps, symbols).shift(1)
    crowding = pivot(out, "char_crowding", timestamps, symbols).shift(1)

    stage0_resid = stage0["residual"]
    mom_raw = stage0_resid.shift(24).rolling(144, min_periods=72).sum()
    mom = mom_raw.shift(1)
    rev = stage0_resid.apply(lambda s: rolling_zscore(s, 168, 56), axis=0).shift(1)
    beta = stage0["beta_mkt"].shift(1)
    return {
        "LIQ": liq,
        "MOM": mom,
        "REV": rev,
        "CARRY": carry,
        "CROWDING": crowding,
        "VOL": vol,
        "BETA": beta,
    }


def build_factor_returns(
    returns: pd.DataFrame,
    chars: dict[str, pd.DataFrame],
    top_bottom_frac: float,
    min_leg_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    factor_returns = pd.DataFrame(index=returns.index)
    factor_returns["MKT"] = returns.mean(axis=1, skipna=True)
    diag_rows: list[dict[str, Any]] = []

    previous_weights: dict[str, pd.Series] = {
        "MKT": pd.Series(0.0, index=returns.columns, dtype=float),
        **{name: pd.Series(0.0, index=returns.columns, dtype=float) for name in chars},
    }
    for ts, row in returns.iterrows():
        valid = row.notna()
        weights = pd.Series(0.0, index=returns.columns, dtype=float)
        if valid.any():
            weights.loc[valid] = 1.0 / float(valid.sum())
        diag_rows.append(weight_diag(ts, "MKT", weights, previous_weights["MKT"], int(valid.sum()), int(valid.sum()), 0))
        previous_weights["MKT"] = weights

    for factor, signal_frame in chars.items():
        values: list[float] = []
        for ts in returns.index:
            signal = signal_frame.loc[ts]
            ret = returns.loc[ts]
            weights, top_count, bottom_count, valid_count = long_short_weights(signal, ret, top_bottom_frac, min_leg_count)
            values.append(float((weights * ret.fillna(0.0)).sum()) if weights.abs().sum() > 0 else np.nan)
            diag_rows.append(weight_diag(ts, factor, weights, previous_weights[factor], valid_count, top_count, bottom_count))
            previous_weights[factor] = weights
        factor_returns[factor] = values

    return factor_returns[FACTOR_NAMES], pd.DataFrame(diag_rows)


def long_short_weights(
    signal: pd.Series,
    returns: pd.Series,
    top_bottom_frac: float,
    min_leg_count: int,
) -> tuple[pd.Series, int, int, int]:
    valid = signal.notna() & returns.notna()
    valid_signal = signal.loc[valid].sort_values()
    weights = pd.Series(0.0, index=signal.index, dtype=float)
    valid_count = int(len(valid_signal))
    leg_count = max(min_leg_count, int(np.floor(valid_count * top_bottom_frac)))
    if valid_count < 2 * leg_count or leg_count <= 0:
        return weights, 0, 0, valid_count
    bottom = valid_signal.index[:leg_count]
    top = valid_signal.index[-leg_count:]
    weights.loc[top] = 1.0 / leg_count
    weights.loc[bottom] = -1.0 / leg_count
    return weights, int(len(top)), int(len(bottom)), valid_count


def weight_diag(
    timestamp: pd.Timestamp,
    factor: str,
    weights: pd.Series,
    previous: pd.Series,
    valid_count: int,
    top_count: int,
    bottom_count: int,
) -> dict[str, Any]:
    return {
        "timestamp": timestamp,
        "factor": factor,
        "valid_signal_count": valid_count,
        "top_count": top_count,
        "bottom_count": bottom_count,
        "gross_weight": float(weights.abs().sum()),
        "net_weight": float(weights.sum()),
        "one_way_turnover": float((weights - previous).abs().sum() / 2.0),
        "max_abs_weight": float(weights.abs().max()) if len(weights) else 0.0,
    }


def rolling_ols_predict(
    y: pd.Series,
    factors: pd.DataFrame,
    beta_window: int,
    min_beta_obs: int,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    y = y.astype(float).reindex(factors.index)
    x = factors.astype(float)
    names = ["intercept", *list(x.columns)]
    y_arr = y.to_numpy(dtype=float)
    x_arr = x.to_numpy(dtype=float)
    n = len(y)
    p = len(names)
    z = np.column_stack([np.ones(n), x_arr])
    valid = np.isfinite(y_arr) & np.isfinite(z).all(axis=1)
    z_valid = np.where(valid[:, None], z, np.nan)
    y_valid = np.where(valid, y_arr, np.nan)

    xtx_sums: dict[tuple[int, int], np.ndarray] = {}
    for a in range(p):
        for b in range(a, p):
            xtx_sums[(a, b)] = (
                pd.Series(z_valid[:, a] * z_valid[:, b], index=y.index)
                .rolling(beta_window, min_periods=min_beta_obs)
                .sum()
                .shift(1)
                .to_numpy()
            )
    xty_sums: dict[int, np.ndarray] = {}
    for a in range(p):
        xty_sums[a] = (
            pd.Series(z_valid[:, a] * y_valid, index=y.index)
            .rolling(beta_window, min_periods=min_beta_obs)
            .sum()
            .shift(1)
            .to_numpy()
        )
    counts = (
        pd.Series(valid.astype(float), index=y.index)
        .rolling(beta_window, min_periods=min_beta_obs)
        .sum()
        .shift(1)
        .to_numpy()
    )

    betas = np.full((n, p), np.nan, dtype=float)
    pred = np.full(n, np.nan, dtype=float)
    resid = np.full(n, np.nan, dtype=float)
    for i in range(n):
        if not np.isfinite(counts[i]) or counts[i] < min_beta_obs or not np.isfinite(z[i]).all():
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
        pred[i] = float(z[i] @ coef)
        resid[i] = y_arr[i] - pred[i] if np.isfinite(y_arr[i]) else np.nan

    beta_frame = pd.DataFrame(betas, index=y.index, columns=names)
    return beta_frame, pd.Series(pred, index=y.index), pd.Series(resid, index=y.index)


def build_ff_residual_panel(
    base_panel: pd.DataFrame,
    factor_returns: pd.DataFrame,
    returns: pd.DataFrame,
    beta_window: int,
    min_beta_obs: int,
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    x = factor_returns[FACTOR_NAMES]
    for symbol in returns.columns:
        beta, pred, resid = rolling_ols_predict(returns[symbol], x, beta_window, min_beta_obs)
        part = pd.DataFrame(
            {
                "timestamp": returns.index,
                "symbol": symbol,
                "ff_predicted_log_ret": pred.to_numpy(),
                "ff_residual_log_ret": resid.to_numpy(),
                "ff_beta_intercept": beta["intercept"].to_numpy(),
            }
        )
        for factor in FACTOR_NAMES:
            part[f"ff_beta_{factor}"] = beta[factor].to_numpy()
        pieces.append(part)
    ff = pd.concat(pieces, ignore_index=True)
    keep_cols = [
        "timestamp",
        "symbol",
        "log_close",
        "log_ret",
        "simple_ret",
        "residual_log_ret",
        "resid_z",
        "score_baseline",
        "beta_btc",
        "beta_eth",
        "beta_market",
        "beta_sector",
        *[f"fwd_log_ret_h{h}" for h in LABEL_HORIZONS],
        *[f"fwd_simple_ret_h{h}" for h in LABEL_HORIZONS],
        *[f"label_resid_ret_h{h}" for h in LABEL_HORIZONS],
    ]
    keep_cols = [col for col in keep_cols if col in base_panel.columns]
    out = base_panel[keep_cols].merge(ff, on=["timestamp", "symbol"], how="left")
    return out.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def add_ff_features_and_labels(
    panel: pd.DataFrame,
    factor_returns: pd.DataFrame,
    horizon_list: list[int],
    execution_delay: int,
) -> pd.DataFrame:
    out_parts: list[pd.DataFrame] = []
    factor_forward_sums = {
        horizon: {
            factor: forward_sum(factor_returns[factor], execution_delay + 1, horizon)
            for factor in FACTOR_NAMES
        }
        for horizon in horizon_list
    }
    for symbol, group in panel.groupby("symbol", sort=False):
        group = group.sort_values("timestamp").copy()
        group["ff_resid_z"] = rolling_zscore(group["ff_residual_log_ret"], 168, 56)
        group["ff_resid_mom_168_24"] = group["ff_residual_log_ret"].shift(24).rolling(144, min_periods=72).sum()
        group["ff_idio_vol_long"] = group["ff_residual_log_ret"].rolling(168, min_periods=56).std(ddof=0)
        for horizon in horizon_list:
            fwd_col = f"fwd_log_ret_h{horizon}"
            if fwd_col not in group.columns:
                continue
            explained = pd.Series(0.0, index=group.index, dtype=float)
            valid = group[fwd_col].notna()
            for factor in FACTOR_NAMES:
                fwd_factor = group["timestamp"].map(factor_forward_sums[horizon][factor])
                beta_col = f"ff_beta_{factor}"
                explained = explained + group[beta_col] * fwd_factor
                valid = valid & fwd_factor.notna() & group[beta_col].notna()
            group[f"ff_label_resid_ret_h{horizon}"] = (group[fwd_col] - explained).where(valid)
        out_parts.append(group)
    out = pd.concat(out_parts, ignore_index=True).sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    mean = out.groupby("timestamp", sort=False)["ff_resid_z"].transform("mean")
    std = out.groupby("timestamp", sort=False)["ff_resid_z"].transform("std").replace(0.0, np.nan)
    out["ff_score_neg_resid_z"] = -((out["ff_resid_z"] - mean) / std)
    for horizon in horizon_list:
        col = f"ff_label_resid_ret_h{horizon}"
        if col in out.columns:
            out[f"ff_label_rank_h{horizon}"] = out.groupby("timestamp", sort=False)[col].rank(pct=True, method="average")
    return out


def forward_sum(series: pd.Series, start_offset: int, horizon: int) -> pd.Series:
    total = pd.Series(0.0, index=series.index, dtype=float)
    valid = pd.Series(True, index=series.index)
    for offset in range(start_offset, start_offset + horizon):
        shifted = series.shift(-offset)
        total = total + shifted.fillna(0.0)
        valid = valid & shifted.notna()
    return total.where(valid)


def factor_collinearity(factor_returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = factor_returns[FACTOR_NAMES].replace([np.inf, -np.inf], np.nan).dropna()
    corr = data.corr()
    rows: list[dict[str, Any]] = []
    for i, left in enumerate(FACTOR_NAMES):
        for right in FACTOR_NAMES[i + 1 :]:
            value = float(corr.loc[left, right])
            rows.append({"factor_1": left, "factor_2": right, "corr": value, "abs_corr": abs(value)})
    pairwise = pd.DataFrame(rows).sort_values("abs_corr", ascending=False)
    c = corr.to_numpy(dtype=float)
    inv = np.linalg.pinv(c, rcond=1e-12)
    eig = np.linalg.eigvalsh(c)
    vif_rows = []
    for idx, factor in enumerate(FACTOR_NAMES):
        vif_rows.append(
            {
                "factor": factor,
                "vif": float(inv[idx, idx]),
                "max_abs_pairwise_corr": float(
                    pairwise[(pairwise["factor_1"] == factor) | (pairwise["factor_2"] == factor)]["abs_corr"].max()
                ),
            }
        )
    vif_rows.append(
        {
            "factor": "__SYSTEM__",
            "vif": np.nan,
            "max_abs_pairwise_corr": float(pairwise["abs_corr"].max()),
            "condition_number_corr": float(np.linalg.cond(c)),
            "min_eigenvalue_corr": float(eig.min()),
            "max_eigenvalue_corr": float(eig.max()),
            "complete_rows": int(len(data)),
        }
    )
    return corr, pairwise, pd.DataFrame(vif_rows)


def build_residual_quality(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for symbol, group in panel.groupby("symbol", sort=True):
        raw = group["log_ret"].astype(float)
        current = group["residual_log_ret"].astype(float)
        ff = group["ff_residual_log_ret"].astype(float)
        raw_std = float(raw.std(ddof=0))
        current_std = float(current.std(ddof=0))
        ff_std = float(ff.std(ddof=0))
        rows.append(
            {
                "symbol": symbol,
                "raw_ret_std": raw_std,
                "current_resid_std": current_std,
                "ff_resid_std": ff_std,
                "current_residual_to_raw_vol_ratio": current_std / raw_std if raw_std > 0 else np.nan,
                "ff_residual_to_raw_vol_ratio": ff_std / raw_std if raw_std > 0 else np.nan,
                "current_r2_like": 1.0 - (current_std**2 / raw_std**2) if raw_std > 0 else np.nan,
                "ff_r2_like": 1.0 - (ff_std**2 / raw_std**2) if raw_std > 0 else np.nan,
                "current_resid_autocorr_1h": float(current.autocorr(1)),
                "ff_resid_autocorr_1h": float(ff.autocorr(1)),
                "ff_valid_rate": float(ff.notna().mean()),
            }
        )
    return pd.DataFrame(rows)


def build_residual_stability(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for symbol, group in panel.groupby("symbol", sort=True):
        group = group.sort_values("timestamp")
        for model, col in [("current", "residual_log_ret"), ("crypto_ff", "ff_residual_log_ret")]:
            vol = group[col].rolling(720, min_periods=240).std(ddof=0)
            values = vol.replace([np.inf, -np.inf], np.nan).dropna()
            rows.append(
                {
                    "model": model,
                    "symbol": symbol,
                    "median_rolling_30d_resid_vol": float(values.median()) if len(values) else np.nan,
                    "mean_rolling_30d_resid_vol": float(values.mean()) if len(values) else np.nan,
                    "cv_rolling_30d_resid_vol": float(values.std(ddof=0) / values.mean()) if len(values) and values.mean() > 0 else np.nan,
                    "p95_rolling_30d_resid_vol": float(values.quantile(0.95)) if len(values) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def build_beta_stability(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    current_cols = ["beta_btc", "beta_eth", "beta_market", "beta_sector"]
    ff_cols = FF_BETA_COLS
    for symbol, group in panel.groupby("symbol", sort=True):
        group = group.sort_values("timestamp")
        for model, cols in [("current", current_cols), ("crypto_ff", ff_cols)]:
            for col in cols:
                if col not in group.columns:
                    continue
                s = group[col].replace([np.inf, -np.inf], np.nan).astype(float)
                rows.append(
                    {
                        "model": model,
                        "symbol": symbol,
                        "beta": col,
                        "valid_rate": float(s.notna().mean()),
                        "mean_abs_1h_change": float(s.diff().abs().mean()),
                        "median_abs_1h_change": float(s.diff().abs().median()),
                        "mean_rolling_7d_std": float(s.rolling(168, min_periods=56).std(ddof=0).mean()),
                        "mean_rolling_30d_std": float(s.rolling(720, min_periods=240).std(ddof=0).mean()),
                    }
                )
    return pd.DataFrame(rows)


def build_label_ic(panel: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    tests = [
        ("current_score_vs_current_label", "score_baseline", f"label_resid_ret_h{horizon}"),
        ("ff_score_vs_ff_label", "ff_score_neg_resid_z", f"ff_label_resid_ret_h{horizon}"),
        ("current_score_vs_ff_label", "score_baseline", f"ff_label_resid_ret_h{horizon}"),
        ("ff_score_vs_current_label", "ff_score_neg_resid_z", f"label_resid_ret_h{horizon}"),
    ]
    rows: list[dict[str, Any]] = []
    for name, score_col, label_col in tests:
        if score_col not in panel.columns or label_col not in panel.columns:
            continue
        for ts, group in panel.groupby("timestamp", sort=True):
            score = group[score_col].astype(float)
            label = group[label_col].astype(float)
            valid = score.notna() & label.notna() & np.isfinite(score) & np.isfinite(label)
            if int(valid.sum()) < 8:
                continue
            rows.append(
                {
                    "timestamp": ts,
                    "test": name,
                    "ic": float(score[valid].rank().corr(label[valid].rank())),
                    "symbols": int(valid.sum()),
                }
            )
    ts_df = pd.DataFrame(rows)
    summary_rows = []
    for name, group in ts_df.groupby("test", sort=True):
        ic = group["ic"].replace([np.inf, -np.inf], np.nan).dropna()
        summary_rows.append(
            {
                "test": name,
                "observations": int(len(ic)),
                "mean_ic": float(ic.mean()) if len(ic) else np.nan,
                "median_ic": float(ic.median()) if len(ic) else np.nan,
                "q25_ic": float(ic.quantile(0.25)) if len(ic) else np.nan,
                "positive_ic_rate": float((ic > 0).mean()) if len(ic) else np.nan,
                "ic_ir": float(ic.mean() / ic.std(ddof=0)) if len(ic) and ic.std(ddof=0) > 0 else np.nan,
            }
        )
    return ts_df, pd.DataFrame(summary_rows)


def run_allocator_comparison(
    residual_panel: pd.DataFrame,
    folds_path: Path,
    config_path: Path,
    horizon: int,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    pf_cfg = make_portfolio_config(cfg, horizon)
    folds = pd.read_csv(folds_path, parse_dates=["train_start", "train_end", "test_start", "test_end"]).sort_values("fold")
    current_panel = residual_panel.copy()
    ff_panel = residual_panel.copy()
    ff_panel[f"label_resid_ret_h{horizon}"] = ff_panel[f"ff_label_resid_ret_h{horizon}"]
    runs = {
        "current_neg_resid_z_allocator": (current_panel, "score_baseline"),
        "crypto_ff_neg_resid_z_allocator": (ff_panel, "ff_score_neg_resid_z"),
    }
    out: dict[str, Any] = {}
    rows = []
    for name, (panel, score_col) in runs.items():
        result = run_folded_backtest(panel, folds, pf_cfg, score_col)
        out[name] = result
        metrics = summarize_equity_curve(result["equity"], pf_cfg.rebalance_hours)
        rows.append({"run": name, **metrics})
    out["summary"] = pd.DataFrame(rows)
    return out


def make_portfolio_config(cfg: dict[str, Any], horizon: int) -> PortfolioConfig:
    locked = cfg["locked_baseline"]
    return PortfolioConfig(
        rebalance_hours=horizon,
        long_k=int(locked["long_k"]),
        short_k=int(locked["short_k"]),
        gross_exposure=float(locked["gross_exposure"]),
        one_way_fee_bps=float(locked["one_way_fee_bps"]),
        one_way_slippage_bps=float(locked["one_way_slippage_bps"]),
        neutralize_exposures=list(locked["neutralize_exposures"]),
        primary_horizon=horizon,
        execution_delay_bars=1,
        use_hysteresis=True,
        min_hold_rebalances=max(1, math.ceil(float(locked["min_hold_days"]) * 24.0 / float(horizon))),
    )


def run_folded_backtest(
    panel: pd.DataFrame,
    folds: pd.DataFrame,
    cfg: PortfolioConfig,
    score_col: str,
) -> dict[str, pd.DataFrame]:
    equity_parts: list[pd.DataFrame] = []
    position_parts: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    for _, fold in folds.iterrows():
        fold_id = int(fold["fold"])
        start = pd.Timestamp(fold["test_start"])
        end = pd.Timestamp(fold["test_end"]) - pd.Timedelta(hours=cfg.primary_horizon + cfg.execution_delay_bars)
        if end <= start:
            continue
        equity, positions, metrics = run_long_short_backtest(panel, start, end, cfg, score_col=score_col)
        if equity.empty:
            continue
        equity["fold"] = fold_id
        positions["fold"] = fold_id
        fold_rows.append(
            {
                "fold": fold_id,
                "train_start": fold["train_start"],
                "train_end": fold["train_end"],
                "test_start": fold["test_start"],
                "test_end": fold["test_end"],
                **metrics,
            }
        )
        equity_parts.append(equity)
        position_parts.append(positions)
    equity_all = pd.concat(equity_parts, ignore_index=True).sort_values(["timestamp", "fold"]).reset_index(drop=True)
    equity_all["equity"] = (1.0 + equity_all["period_return"]).cumprod()
    positions_all = pd.concat(position_parts, ignore_index=True).sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    return {"equity": equity_all, "positions": positions_all, "folds": pd.DataFrame(fold_rows)}


def write_allocator_outputs(
    output_dir: Path,
    allocator: dict[str, Any],
    cost_bps: list[float],
    p15c_run_dir: Path,
) -> None:
    allocator["summary"].to_csv(output_dir / "allocator_smoke_summary.csv", index=False)
    for name, result in allocator.items():
        if name == "summary":
            continue
        run_dir = ensure_dir(output_dir / name)
        result["equity"].to_csv(run_dir / "equity_curve.csv", index=False)
        result["positions"].to_csv(run_dir / "positions.csv", index=False)
        result["folds"].to_csv(run_dir / "fold_metrics.csv", index=False)
    cost = build_cost_sensitivity({k: v for k, v in allocator.items() if k != "summary"}, cost_bps)
    if p15c_run_dir.exists() and (p15c_run_dir / "equity_curve.csv").exists():
        p15c_equity = pd.read_csv(p15c_run_dir / "equity_curve.csv", parse_dates=["timestamp", "exit_timestamp"])
        cost = pd.concat([cost, build_cost_sensitivity({"p15c_reference": {"equity": p15c_equity}}, cost_bps)], ignore_index=True)
    cost.to_csv(output_dir / "allocator_cost_sensitivity.csv", index=False)


def build_cost_sensitivity(results: dict[str, dict[str, pd.DataFrame]], cost_bps: list[float]) -> pd.DataFrame:
    rows = []
    for name, result in results.items():
        equity = result["equity"].copy()
        if equity.empty:
            continue
        for bps in cost_bps:
            period_ret = equity["gross_period_return"].astype(float) - equity["turnover"].astype(float) * float(bps) / 10_000.0
            eq = (1.0 + period_ret).cumprod()
            ret_std = period_ret.std(ddof=0)
            sharpe = 0.0 if ret_std == 0 or not np.isfinite(ret_std) else float(period_ret.mean() / ret_std * np.sqrt(365 * 24 / 72))
            dd = eq / eq.cummax() - 1.0
            rows.append(
                {
                    "run": name,
                    "one_way_cost_bps": float(bps),
                    "total_return": float(eq.iloc[-1] - 1.0),
                    "sharpe": sharpe,
                    "max_drawdown": float(dd.min()),
                    "total_turnover": float(equity["turnover"].sum()),
                    "cost_drag": float((equity["turnover"] * float(bps) / 10_000.0).sum()),
                }
            )
    return pd.DataFrame(rows)


def plot_factor_corr(output_dir: Path, corr: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)), corr.index)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Crypto FF-Like Factor Return Correlation")
    fig.tight_layout()
    fig.savefig(output_dir / "factor_correlation_heatmap.png", dpi=160)
    plt.close(fig)


def plot_allocator_curves(output_dir: Path, allocator: dict[str, Any], cost_levels: list[float]) -> None:
    for bps in cost_levels:
        fig, ax = plt.subplots(figsize=(11, 5))
        for name, result in allocator.items():
            if name == "summary":
                continue
            equity = result["equity"].sort_values("timestamp").copy()
            period_ret = equity["gross_period_return"].astype(float) - equity["turnover"].astype(float) * bps / 10_000.0
            curve = (1.0 + period_ret).cumprod() - 1.0
            ax.plot(equity["timestamp"], curve, label=name)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_title(f"Allocator Smoke Test PnL, one-way cost={bps:g}bps")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"allocator_pnl_{int(bps)}bps.png", dpi=160)
        plt.close(fig)


def build_topline(
    factor_vif: pd.DataFrame,
    factor_pairwise: pd.DataFrame,
    residual_quality: pd.DataFrame,
    residual_stability: pd.DataFrame,
    beta_stability: pd.DataFrame,
    ic_summary: pd.DataFrame,
    allocator_summary: pd.DataFrame,
) -> pd.DataFrame:
    system = factor_vif[factor_vif["factor"].eq("__SYSTEM__")].iloc[0]
    top_pair = factor_pairwise.iloc[0]
    current_q = residual_quality["current_residual_to_raw_vol_ratio"].median()
    ff_q = residual_quality["ff_residual_to_raw_vol_ratio"].median()
    current_stab = residual_stability[residual_stability["model"].eq("current")]["cv_rolling_30d_resid_vol"].median()
    ff_stab = residual_stability[residual_stability["model"].eq("crypto_ff")]["cv_rolling_30d_resid_vol"].median()
    current_beta = beta_stability[beta_stability["model"].eq("current")]["mean_abs_1h_change"].median()
    ff_beta = beta_stability[beta_stability["model"].eq("crypto_ff")]["mean_abs_1h_change"].median()
    ic_map = ic_summary.set_index("test")["mean_ic"].to_dict()
    alloc_map = allocator_summary.set_index("run")["gross_total_return_before_cost"].to_dict()
    return pd.DataFrame(
        [
            {
                "check": "factor_max_abs_pairwise_corr",
                "value": float(system["max_abs_pairwise_corr"]),
                "status": "pass" if float(system["max_abs_pairwise_corr"]) < 0.90 else "review",
            },
            {
                "check": "factor_top_correlated_pair",
                "value": f"{top_pair['factor_1']}~{top_pair['factor_2']} corr={top_pair['corr']:.4f}",
                "status": "info",
            },
            {
                "check": "factor_condition_number",
                "value": float(system["condition_number_corr"]),
                "status": "pass" if float(system["condition_number_corr"]) < 100.0 else "review",
            },
            {
                "check": "median_residual_to_raw_vol_ratio_current",
                "value": float(current_q),
                "status": "info",
            },
            {
                "check": "median_residual_to_raw_vol_ratio_crypto_ff",
                "value": float(ff_q),
                "status": "pass" if ff_q <= current_q else "review",
            },
            {
                "check": "median_rolling_resid_vol_cv_current",
                "value": float(current_stab),
                "status": "info",
            },
            {
                "check": "median_rolling_resid_vol_cv_crypto_ff",
                "value": float(ff_stab),
                "status": "pass" if ff_stab <= current_stab * 1.10 else "review",
            },
            {
                "check": "median_beta_1h_change_current",
                "value": float(current_beta),
                "status": "info",
            },
            {
                "check": "median_beta_1h_change_crypto_ff",
                "value": float(ff_beta),
                "status": "pass" if ff_beta <= current_beta * 1.50 else "review",
            },
            {
                "check": "mean_ic_current_score_current_label",
                "value": float(ic_map.get("current_score_vs_current_label", np.nan)),
                "status": "info",
            },
            {
                "check": "mean_ic_ff_score_ff_label",
                "value": float(ic_map.get("ff_score_vs_ff_label", np.nan)),
                "status": "pass"
                if ic_map.get("ff_score_vs_ff_label", -999) >= ic_map.get("current_score_vs_current_label", 999)
                else "review",
            },
            {
                "check": "allocator_gross_return_current_score",
                "value": float(alloc_map.get("current_neg_resid_z_allocator", np.nan)),
                "status": "info",
            },
            {
                "check": "allocator_gross_return_ff_score",
                "value": float(alloc_map.get("crypto_ff_neg_resid_z_allocator", np.nan)),
                "status": "pass"
                if alloc_map.get("crypto_ff_neg_resid_z_allocator", -999)
                >= alloc_map.get("current_neg_resid_z_allocator", 999)
                else "review",
            },
        ]
    )


def write_report(
    output_dir: Path,
    topline: pd.DataFrame,
    factor_pairwise: pd.DataFrame,
    factor_vif: pd.DataFrame,
    residual_quality: pd.DataFrame,
    residual_stability: pd.DataFrame,
    beta_stability: pd.DataFrame,
    ic_summary: pd.DataFrame,
    allocator: dict[str, Any],
) -> None:
    residual_compare = pd.DataFrame(
        [
            {
                "model": "current",
                "median_resid_to_raw": residual_quality["current_residual_to_raw_vol_ratio"].median(),
                "median_r2_like": residual_quality["current_r2_like"].median(),
                "median_30d_vol_cv": residual_stability[residual_stability["model"].eq("current")][
                    "cv_rolling_30d_resid_vol"
                ].median(),
            },
            {
                "model": "crypto_ff",
                "median_resid_to_raw": residual_quality["ff_residual_to_raw_vol_ratio"].median(),
                "median_r2_like": residual_quality["ff_r2_like"].median(),
                "median_30d_vol_cv": residual_stability[residual_stability["model"].eq("crypto_ff")][
                    "cv_rolling_30d_resid_vol"
                ].median(),
            },
        ]
    )
    beta_summary = beta_stability.groupby("model").agg(
        median_abs_1h_change=("mean_abs_1h_change", "median"),
        median_rolling_7d_std=("mean_rolling_7d_std", "median"),
        median_rolling_30d_std=("mean_rolling_30d_std", "median"),
    ).reset_index()
    lines = [
        "# P23 Crypto FF-Like Rolling Residualizer",
        "",
        "This package upgrades the crypto FF-like idea into a rolling past-only residualizer. It is still a research diagnostic, not a full AlphaGPT/P15c rerun.",
        "",
        "## Topline",
        "",
        "```text",
        topline.to_string(index=False),
        "```",
        "",
        "## Factor Collinearity",
        "",
        "```text",
        factor_vif.to_string(index=False),
        "```",
        "",
        "```text",
        factor_pairwise.head(12).to_string(index=False),
        "```",
        "",
        "## Residual Quality",
        "",
        "```text",
        residual_compare.to_string(index=False),
        "```",
        "",
        "## Beta Stability",
        "",
        "```text",
        beta_summary.to_string(index=False),
        "```",
        "",
        "## Label IC",
        "",
        "```text",
        ic_summary.to_string(index=False),
        "```",
        "",
        "## Allocator Smoke Test",
        "",
        "```text",
        allocator["summary"].to_string(index=False),
        "```",
        "",
        "## Caveats",
        "",
        "- MOM/REV characteristics are built from a rolling MKT-only stage-0 residual, not from the old BTC/ETH/sector residual.",
        "- The 20-symbol universe is small, so factor-mimicking portfolios still have material self-inclusion noise.",
        "- The allocator section is a constant-score smoke test using the existing strict-neutral allocator; it does not re-run AlphaGPT, P20 selection, or P15c cluster logic.",
        "- A production replacement would need nested OOS factor/allocator selection and likely leave-one-symbol-out factor construction.",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


if __name__ == "__main__":
    main()
