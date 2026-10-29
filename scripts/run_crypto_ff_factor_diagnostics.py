#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "baseline_locked_v1"
DEFAULT_ASSET_PANEL = EXPERIMENT_ROOT / "inputs" / "long_history" / "baseline_panel" / "asset_residual_panel.csv"
DEFAULT_DERIVATIVES_PANEL = (
    EXPERIMENT_ROOT / "inputs" / "long_history" / "binance_derivatives" / "derivatives_hourly_panel.csv.gz"
)
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "outputs" / "crypto_ff_factor_diagnostics"
CURRENT_MODEL_QUALITY = EXPERIMENT_ROOT / "outputs" / "residual_factor_neutralization_audit" / "factor_model_quality.csv"

ASSET_COLS = [
    "timestamp",
    "symbol",
    "log_ret",
    "residual_log_ret",
    "resid_z",
    "realized_vol_long",
    "log_quote_volume",
    "quote_volume_share",
    "beta_btc",
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
    parser = argparse.ArgumentParser(description="Build crypto Fama-French-like factor-mimicking portfolios and diagnose collinearity.")
    parser.add_argument("--asset-panel", default=str(DEFAULT_ASSET_PANEL))
    parser.add_argument("--derivatives-panel", default=str(DEFAULT_DERIVATIVES_PANEL))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--top-bottom-frac", type=float, default=0.30)
    parser.add_argument("--min-leg-count", type=int, default=3)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/6] Loading asset and derivative feature columns", flush=True)
    panel = load_panel(Path(args.asset_panel), Path(args.derivatives_panel))

    print("[2/6] Building ex-ante characteristics", flush=True)
    panel = add_crypto_ff_characteristics(panel)
    specs = factor_specs()
    pd.DataFrame(specs).to_csv(output_dir / "crypto_ff_factor_specs.csv", index=False)

    print("[3/6] Building factor-mimicking portfolio returns", flush=True)
    factor_returns, portfolio_diagnostics = build_factor_returns(
        panel,
        specs,
        top_bottom_frac=float(args.top_bottom_frac),
        min_leg_count=int(args.min_leg_count),
    )
    factor_returns.to_csv(output_dir / "crypto_ff_factor_returns.csv", index=False)
    portfolio_diagnostics.to_csv(output_dir / "factor_portfolio_diagnostics.csv", index=False)

    print("[4/6] Computing collinearity diagnostics", flush=True)
    corr, pairwise, vif_condition = factor_collinearity(factor_returns)
    corr.to_csv(output_dir / "factor_correlation_matrix.csv")
    pairwise.to_csv(output_dir / "factor_pairwise_correlations.csv", index=False)
    vif_condition.to_csv(output_dir / "factor_vif_condition.csv", index=False)
    return_summary = summarize_factor_returns(factor_returns, portfolio_diagnostics)
    return_summary.to_csv(output_dir / "factor_return_summary.csv", index=False)

    print("[5/6] Running simple asset-level explanatory regressions", flush=True)
    regression_quality = asset_regression_quality(panel, factor_returns)
    regression_quality.to_csv(output_dir / "asset_regression_quality.csv", index=False)
    comparison = compare_to_current_model(regression_quality)
    comparison.to_csv(output_dir / "current_vs_crypto_ff_quality.csv", index=False)

    print("[6/6] Writing report", flush=True)
    topline = build_topline(vif_condition, pairwise, return_summary, regression_quality, comparison)
    topline.to_csv(output_dir / "topline_findings.csv", index=False)
    metadata = {
        "asset_panel": str(Path(args.asset_panel).resolve()),
        "derivatives_panel": str(Path(args.derivatives_panel).resolve()),
        "top_bottom_frac": float(args.top_bottom_frac),
        "min_leg_count": int(args.min_leg_count),
        "ex_ante_rule": "All non-MKT factor portfolio weights use timestamp t-1 characteristics and timestamp t returns.",
        "small_universe_warning": "This is a 20-symbol diagnostic prototype; factor portfolios are noisier than equity Fama-French portfolios.",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(output_dir, topline, corr, pairwise, vif_condition, return_summary, portfolio_diagnostics, regression_quality, comparison)
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

    panel = asset.merge(derivatives, on=["timestamp", "symbol"], how="left")
    return panel.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def add_crypto_ff_characteristics(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    by_symbol = out.groupby("symbol", group_keys=False)

    out["char_liq"] = -out["log_quote_volume"].astype(float)
    out["res_mom_168_24"] = by_symbol["residual_log_ret"].transform(
        lambda s: s.shift(24).rolling(144, min_periods=72).sum()
    )
    out["char_mom"] = out["res_mom_168_24"]
    out["char_rev"] = out["resid_z"].astype(float)
    out["char_vol"] = out["realized_vol_long"].astype(float)
    out["char_beta"] = out["beta_btc"].astype(float)

    for col in [
        "deriv_funding_rate_ewm_168",
        "deriv_mark_index_basis_ewm_168",
        "deriv_premium_index_ewm_168",
        "deriv_oi_growth_72_336",
        "deriv_taker_pressure_24",
        "deriv_crowding_pressure_72",
    ]:
        if col not in out:
            out[col] = np.nan
        out[f"cs_{col}"] = cross_sectional_z(out, col)

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

    for col in ["char_liq", "char_mom", "char_rev", "char_carry", "char_crowding", "char_vol", "char_beta"]:
        out[f"{col}_lag1"] = by_symbol[col].shift(1)
    return out


def cross_sectional_z(frame: pd.DataFrame, col: str) -> pd.Series:
    grouped = frame.groupby("timestamp", sort=False)[col]
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0.0, np.nan)
    return (frame[col] - mean) / std


def factor_specs() -> list[dict[str, str]]:
    return [
        {
            "factor": "MKT",
            "signal_col": "",
            "definition": "Equal-weight all-symbol crypto basket return.",
            "orientation": "long equal-weight universe",
        },
        {
            "factor": "LIQ",
            "signal_col": "char_liq_lag1",
            "definition": "Low liquidity / small quote-volume minus high liquidity / large quote-volume.",
            "orientation": "long low-liquidity, short high-liquidity",
        },
        {
            "factor": "MOM",
            "signal_col": "char_mom_lag1",
            "definition": "High slow residual momentum minus low slow residual momentum, using 168h lookback and 24h skip.",
            "orientation": "long high residual momentum, short low residual momentum",
        },
        {
            "factor": "REV",
            "signal_col": "char_rev_lag1",
            "definition": "High residual z-score minus low residual z-score.",
            "orientation": "long high residual z, short low residual z",
        },
        {
            "factor": "CARRY",
            "signal_col": "char_carry_lag1",
            "definition": "High funding/basis/premium composite minus low funding/basis/premium composite.",
            "orientation": "long high carry/richness, short low carry/richness",
        },
        {
            "factor": "CROWDING",
            "signal_col": "char_crowding_lag1",
            "definition": "High OI-growth/taker/crowding pressure minus low crowding pressure.",
            "orientation": "long high crowding, short low crowding",
        },
        {
            "factor": "VOL",
            "signal_col": "char_vol_lag1",
            "definition": "High realized-volatility assets minus low realized-volatility assets.",
            "orientation": "long high vol, short low vol",
        },
        {
            "factor": "BETA",
            "signal_col": "char_beta_lag1",
            "definition": "High BTC beta assets minus low BTC beta assets.",
            "orientation": "long high BTC beta, short low BTC beta",
        },
    ]


def build_factor_returns(
    panel: pd.DataFrame,
    specs: list[dict[str, str]],
    top_bottom_frac: float,
    min_leg_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    symbols = sorted(panel["symbol"].unique())
    timestamps = sorted(panel["timestamp"].dropna().unique())
    previous_weights = {spec["factor"]: pd.Series(0.0, index=symbols, dtype=float) for spec in specs}
    return_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []

    for timestamp, group in panel.groupby("timestamp", sort=True):
        group = group.set_index("symbol").reindex(symbols)
        returns = group["log_ret"].astype(float)
        row: dict[str, Any] = {"timestamp": timestamp}
        for spec in specs:
            factor = spec["factor"]
            if factor == "MKT":
                weights = pd.Series(0.0, index=symbols, dtype=float)
                valid_ret = returns.notna()
                if valid_ret.any():
                    weights.loc[valid_ret] = 1.0 / float(valid_ret.sum())
                factor_ret = float((weights * returns.fillna(0.0)).sum()) if valid_ret.any() else np.nan
                top_count = int(valid_ret.sum())
                bottom_count = 0
                valid_signal_count = int(valid_ret.sum())
            else:
                signal = group[spec["signal_col"]].astype(float)
                weights, top_count, bottom_count, valid_signal_count = long_short_weights(
                    signal,
                    returns,
                    top_bottom_frac=top_bottom_frac,
                    min_leg_count=min_leg_count,
                )
                factor_ret = float((weights * returns.fillna(0.0)).sum()) if weights.abs().sum() > 0 else np.nan

            turnover = float((weights - previous_weights[factor]).abs().sum())
            previous_weights[factor] = weights
            row[factor] = factor_ret
            diagnostic_rows.append(
                {
                    "timestamp": timestamp,
                    "factor": factor,
                    "valid_signal_count": valid_signal_count,
                    "top_count": top_count,
                    "bottom_count": bottom_count,
                    "gross_weight": float(weights.abs().sum()),
                    "net_weight": float(weights.sum()),
                    "one_way_turnover": turnover / 2.0,
                    "max_abs_weight": float(weights.abs().max()),
                }
            )
        return_rows.append(row)

    return pd.DataFrame(return_rows), pd.DataFrame(diagnostic_rows)


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


def factor_collinearity(factor_returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    factors = [col for col in factor_returns.columns if col != "timestamp"]
    data = factor_returns[factors].replace([np.inf, -np.inf], np.nan).dropna()
    corr = data.corr()
    rows: list[dict[str, Any]] = []
    for i, left in enumerate(factors):
        for right in factors[i + 1 :]:
            value = float(corr.loc[left, right])
            rows.append({"factor_1": left, "factor_2": right, "corr": value, "abs_corr": abs(value)})
    pairwise = pd.DataFrame(rows).sort_values("abs_corr", ascending=False)

    corr_values = corr.to_numpy(dtype=float)
    eig = np.linalg.eigvalsh(corr_values)
    condition_number = float(np.linalg.cond(corr_values))
    inv_corr = np.linalg.pinv(corr_values, rcond=1e-12)
    vif_rows = [
        {
            "factor": factor,
            "vif": float(inv_corr[index, index]),
            "max_abs_pairwise_corr": float(pairwise[(pairwise["factor_1"] == factor) | (pairwise["factor_2"] == factor)]["abs_corr"].max()),
        }
        for index, factor in enumerate(factors)
    ]
    vif_rows.append(
        {
            "factor": "__SYSTEM__",
            "vif": np.nan,
            "max_abs_pairwise_corr": float(pairwise["abs_corr"].max()),
            "condition_number_corr": condition_number,
            "min_eigenvalue_corr": float(eig.min()),
            "max_eigenvalue_corr": float(eig.max()),
            "complete_rows": int(len(data)),
        }
    )
    return corr, pairwise, pd.DataFrame(vif_rows)


def summarize_factor_returns(factor_returns: pd.DataFrame, diagnostics: pd.DataFrame) -> pd.DataFrame:
    factors = [col for col in factor_returns.columns if col != "timestamp"]
    rows: list[dict[str, Any]] = []
    periods_per_year = 365.0 * 24.0
    for factor in factors:
        series = factor_returns[factor].replace([np.inf, -np.inf], np.nan).dropna().astype(float)
        diag = diagnostics[diagnostics["factor"] == factor]
        std = float(series.std(ddof=0)) if len(series) else np.nan
        rows.append(
            {
                "factor": factor,
                "observations": int(len(series)),
                "mean_hourly_return": float(series.mean()) if len(series) else np.nan,
                "hourly_vol": std,
                "annualized_vol": float(std * np.sqrt(periods_per_year)) if np.isfinite(std) else np.nan,
                "hourly_sharpe_like": float(series.mean() / std) if std > 0 else np.nan,
                "autocorr_1h": float(series.autocorr(1)) if len(series) > 2 else np.nan,
                "mean_one_way_turnover": float(diag["one_way_turnover"].mean()) if len(diag) else np.nan,
                "mean_valid_signal_count": float(diag["valid_signal_count"].mean()) if len(diag) else np.nan,
                "mean_gross_weight": float(diag["gross_weight"].mean()) if len(diag) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def asset_regression_quality(panel: pd.DataFrame, factor_returns: pd.DataFrame) -> pd.DataFrame:
    factors = [col for col in factor_returns.columns if col != "timestamp"]
    factors_frame = factor_returns[["timestamp", *factors]].copy()
    rows: list[dict[str, Any]] = []
    for symbol, group in panel[["timestamp", "symbol", "log_ret"]].groupby("symbol", sort=True):
        data = group.merge(factors_frame, on="timestamp", how="left").replace([np.inf, -np.inf], np.nan)
        data = data.dropna(subset=["log_ret", *factors])
        if len(data) <= len(factors) + 10:
            continue
        y = data["log_ret"].astype(float).to_numpy()
        x = data[factors].astype(float)
        x_std = (x - x.mean()) / x.std(ddof=0).replace(0.0, np.nan)
        valid_cols = x_std.columns[x_std.notna().all()].tolist()
        x_std = x_std[valid_cols]
        z = np.column_stack([np.ones(len(x_std)), x_std.to_numpy()])
        coef = np.linalg.pinv(z.T @ z, rcond=1e-10) @ (z.T @ y)
        pred = z @ coef
        resid = y - pred
        raw_std = float(np.std(y, ddof=0))
        resid_std = float(np.std(resid, ddof=0))
        rows.append(
            {
                "symbol": symbol,
                "observations": int(len(data)),
                "factor_count": int(len(valid_cols)),
                "design_condition_number": float(np.linalg.cond(z.T @ z)),
                "raw_ret_std": raw_std,
                "residual_ret_std": resid_std,
                "residual_to_raw_vol_ratio": float(resid_std / raw_std) if raw_std > 0 else np.nan,
                "full_sample_r2": float(1.0 - (resid_std**2) / (raw_std**2)) if raw_std > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def compare_to_current_model(regression_quality: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rows.append(summarize_quality("crypto_ff_like_full_sample", regression_quality))
    if CURRENT_MODEL_QUALITY.exists():
        current = pd.read_csv(CURRENT_MODEL_QUALITY)
        rows.append(
            {
                "model": "current_btc_eth_market_sector_rolling",
                "symbols": int(current["symbol"].nunique()),
                "median_residual_to_raw_vol_ratio": float(current["residual_to_raw_vol_ratio"].median()),
                "mean_residual_to_raw_vol_ratio": float(current["residual_to_raw_vol_ratio"].mean()),
                "min_r2": float(current["approx_out_of_sample_r2"].min()),
                "median_r2": float(current["approx_out_of_sample_r2"].median()),
                "max_design_condition_number": float(current["factor_design_condition_number"].replace([np.inf, -np.inf], np.nan).max()),
                "median_design_condition_number": float(current["factor_design_condition_number"].replace([np.inf, -np.inf], np.nan).median()),
                "note": "Rolling past-only current model quality from residual_factor_neutralization_audit.",
            }
        )
    return pd.DataFrame(rows)


def summarize_quality(model: str, quality: pd.DataFrame) -> dict[str, Any]:
    return {
        "model": model,
        "symbols": int(quality["symbol"].nunique()) if "symbol" in quality else 0,
        "median_residual_to_raw_vol_ratio": float(quality["residual_to_raw_vol_ratio"].median()),
        "mean_residual_to_raw_vol_ratio": float(quality["residual_to_raw_vol_ratio"].mean()),
        "min_r2": float(quality["full_sample_r2"].min()),
        "median_r2": float(quality["full_sample_r2"].median()),
        "max_design_condition_number": float(quality["design_condition_number"].replace([np.inf, -np.inf], np.nan).max()),
        "median_design_condition_number": float(quality["design_condition_number"].replace([np.inf, -np.inf], np.nan).median()),
        "note": "Full-sample explanatory regression for diagnostics only; not a walk-forward backtest.",
    }


def build_topline(
    vif_condition: pd.DataFrame,
    pairwise: pd.DataFrame,
    return_summary: pd.DataFrame,
    regression_quality: pd.DataFrame,
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    system = vif_condition[vif_condition["factor"] == "__SYSTEM__"].iloc[0]
    max_vif = float(vif_condition[vif_condition["factor"] != "__SYSTEM__"]["vif"].max())
    top_pair = pairwise.iloc[0]
    crypto_row = comparison[comparison["model"] == "crypto_ff_like_full_sample"].iloc[0]
    rows = [
        {
            "check": "crypto_ff_max_abs_pairwise_corr",
            "value": float(system["max_abs_pairwise_corr"]),
            "status": "pass" if float(system["max_abs_pairwise_corr"]) < 0.90 else "review",
        },
        {
            "check": "crypto_ff_top_correlated_pair",
            "value": f"{top_pair['factor_1']}~{top_pair['factor_2']} corr={top_pair['corr']:.4f}",
            "status": "info",
        },
        {
            "check": "crypto_ff_corr_condition_number",
            "value": float(system["condition_number_corr"]),
            "status": "pass" if float(system["condition_number_corr"]) < 100.0 else "review",
        },
        {
            "check": "crypto_ff_max_vif",
            "value": max_vif,
            "status": "pass" if max_vif < 10.0 else "review",
        },
        {
            "check": "crypto_ff_median_residual_to_raw_vol_ratio",
            "value": float(crypto_row["median_residual_to_raw_vol_ratio"]),
            "status": "pass" if float(crypto_row["median_residual_to_raw_vol_ratio"]) < 0.75 else "review",
        },
        {
            "check": "crypto_ff_median_r2",
            "value": float(crypto_row["median_r2"]),
            "status": "pass" if float(crypto_row["median_r2"]) > 0.40 else "review",
        },
        {
            "check": "highest_turnover_factor",
            "value": return_summary.sort_values("mean_one_way_turnover", ascending=False).iloc[0]["factor"],
            "status": "info",
        },
    ]
    return pd.DataFrame(rows)


def write_report(
    output_dir: Path,
    topline: pd.DataFrame,
    corr: pd.DataFrame,
    pairwise: pd.DataFrame,
    vif_condition: pd.DataFrame,
    return_summary: pd.DataFrame,
    portfolio_diagnostics: pd.DataFrame,
    regression_quality: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    diag_summary = portfolio_diagnostics.groupby("factor").agg(
        mean_valid_signal_count=("valid_signal_count", "mean"),
        mean_one_way_turnover=("one_way_turnover", "mean"),
        mean_gross_weight=("gross_weight", "mean"),
        mean_max_abs_weight=("max_abs_weight", "mean"),
    ).reset_index()
    lines = [
        "# Crypto Fama-French-Like Factor Diagnostics",
        "",
        "This is not a strategy backtest. It builds factor-mimicking portfolio returns with t-1 characteristics and diagnoses collinearity, turnover, and simple explanatory power.",
        "",
        "## Topline",
        "",
        "```text",
        topline.to_string(index=False),
        "```",
        "",
        "## Factor Correlation Matrix",
        "",
        "```text",
        corr.to_string(),
        "```",
        "",
        "## Top Pairwise Correlations",
        "",
        "```text",
        pairwise.head(12).to_string(index=False),
        "```",
        "",
        "## VIF / Condition",
        "",
        "```text",
        vif_condition.to_string(index=False),
        "```",
        "",
        "## Factor Return / Turnover Summary",
        "",
        "```text",
        return_summary.to_string(index=False),
        "```",
        "",
        "## Portfolio Construction Diagnostics",
        "",
        "```text",
        diag_summary.to_string(index=False),
        "```",
        "",
        "## Asset Regression Quality",
        "",
        "```text",
        regression_quality.to_string(index=False),
        "```",
        "",
        "## Comparison To Current Residual Model",
        "",
        "```text",
        comparison.to_string(index=False),
        "```",
        "",
        "## Caveats",
        "",
        "- Universe has only 20 symbols, so long-short factor portfolios are much noisier than equity FF portfolios.",
        "- MOM and REV are built from the existing residual model, so this is a prototype of the construction idea rather than a clean replacement residualizer.",
        "- The asset regression is full-sample explanatory diagnostics only; it is not walk-forward and not an OOS performance claim.",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
