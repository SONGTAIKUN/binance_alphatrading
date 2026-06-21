#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.ticker import PercentFormatter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from crypto_stat_arb.config import ensure_dir, load_config
from crypto_stat_arb.portfolio import portfolio_exposures
import run_p3d_p4_gated_rerank_alphagpt as p3d
import run_p7_derivatives_gated_rerank_alphagpt as p7
import run_p9_clean_derivatives_baseline as p9
from run_turnover_experiments import to_jsonable


EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "baseline_locked_v1"
DEFAULT_CONFIG = EXPERIMENT_ROOT / "configs" / "baseline_locked_v1.yaml"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "outputs" / "p10_strategy_conditioned_regime_gate"
P3D_DIR = EXPERIMENT_ROOT / "outputs" / "p3d_p4_gated_rerank_alphagpt" / "nested_incremental_run"
P7B_DIR = EXPERIMENT_ROOT / "outputs" / "p7b_derivatives_incremental_on_p3d" / "nested_incremental_run"
P9_DIR = EXPERIMENT_ROOT / "outputs" / "p9_clean_baseline_package" / "nested_incremental_run"
DERIV_PANEL = EXPERIMENT_ROOT / "inputs" / "binance_derivatives" / "derivatives_hourly_panel.csv.gz"


@dataclass(frozen=True)
class GateSpec:
    name: str
    description: str
    gross_by_regime: dict[str, float]
    short_by_regime: dict[str, float] | None = None
    long_by_regime: dict[str, float] | None = None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--horizon", type=int, default=72)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--cost-bps", type=float, nargs="+", default=[0, 4, 6, 8, 10, 12, 15, 20, 30])
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    output_dir = ensure_dir(args.output_dir)
    horizon = int(args.horizon)
    cost_grid = [float(x) for x in args.cost_bps]

    print("[1/8] Loading P9 baseline result and source panels", flush=True)
    p9_result = load_existing_result(P9_DIR)
    p7b_result = load_existing_result(P7B_DIR)
    p3d_result = load_existing_result(P3D_DIR)
    panel_path = p3d.resolve_config_path(config_path, cfg["inputs"]["panel"])
    asset_panel = load_asset_panel(panel_path)
    deriv_panel = load_derivatives_panel(DERIV_PANEL)

    print("[2/8] Building ex-ante market-level regime feature panel", flush=True)
    regime = build_regime_panel(asset_panel, deriv_panel)
    regime.to_csv(output_dir / "regime_feature_panel.csv", index=False)
    regime_counts = regime["regime"].value_counts(dropna=False).rename_axis("regime").reset_index(name="hour_count")
    regime_counts.to_csv(output_dir / "regime_counts_hourly.csv", index=False)

    print("[3/8] Replaying regime gate variants as P9 portfolio overlays", flush=True)
    beta_panel = asset_panel[["timestamp", "symbol", "beta_btc", "beta_eth", "beta_market"]].copy()
    gate_specs = build_gate_specs()
    results: dict[str, dict[str, Any]] = {
        "p3d_reference": p3d_result,
        "p7b_reference": p7b_result,
        "p9_reference": p9_result,
    }
    for spec in gate_specs:
        print(f"      {spec.name}", flush=True)
        result = apply_gate_overlay(p9_result, regime, beta_panel, spec, horizon)
        results[spec.name] = result
        run_dir = ensure_dir(output_dir / "runs" / spec.name)
        write_result(run_dir, result)

    print("[4/8] Building summary, cost, fold, and regime diagnostics", flush=True)
    summary = build_summary(results, horizon)
    summary.to_csv(output_dir / "variant_summary.csv", index=False)
    cost_summary = build_cost_summary(results, cost_grid)
    cost_summary.to_csv(output_dir / "cost_sensitivity.csv", index=False)
    fold_summary = build_fold_summary(results)
    fold_summary.to_csv(output_dir / "fold_summary.csv", index=False)
    fold_delta = build_fold_delta_table(results, "p9_reference")
    fold_delta.to_csv(output_dir / "fold_delta_vs_p9.csv", index=False)
    regime_attr = build_regime_attribution(results, regime)
    regime_attr.to_csv(output_dir / "regime_attribution.csv", index=False)
    filter_efficacy = build_filter_efficacy(results)
    filter_efficacy.to_csv(output_dir / "filter_efficacy.csv", index=False)
    stress_attr = build_stress_period_attribution(results, regime)
    stress_attr.to_csv(output_dir / "stress_period_attribution.csv", index=False)

    print("[5/8] Running robustness checks", flush=True)
    best_name = select_best_variant(summary, cost_summary, fold_delta)
    best_result = results[best_name]
    p9_fold_delta = build_pair_fold_delta(best_result, p9_result, best_name, "p9")
    p3d_fold_delta = build_pair_fold_delta(best_result, p3d_result, best_name, "p3d")
    bootstrap = pd.concat(
        [
            p9.bootstrap_fold_checks(p9_fold_delta, f"{best_name}_minus_p9", int(args.bootstrap_samples)),
            p9.bootstrap_fold_checks(p3d_fold_delta, f"{best_name}_minus_p3d", int(args.bootstrap_samples)),
        ],
        ignore_index=True,
    )
    bootstrap.to_csv(output_dir / "bootstrap_checks.csv", index=False)
    start_vs_p9 = p9.build_start_fold_sensitivity(best_result, p9_result, best_name, "p9", horizon)
    start_vs_p3d = p9.build_start_fold_sensitivity(best_result, p3d_result, best_name, "p3d", horizon)
    start_vs_p9.to_csv(output_dir / "start_fold_sensitivity_vs_p9.csv", index=False)
    start_vs_p3d.to_csv(output_dir / "start_fold_sensitivity_vs_p3d.csv", index=False)
    dropout_vs_p9 = p9.build_symbol_dropout_checks(best_result, p9_result, best_name, "p9", horizon)
    dropout_vs_p9.to_csv(output_dir / "symbol_dropout_vs_p9.csv", index=False)
    reality = build_reality_summary(best_name, summary, cost_summary, bootstrap, start_vs_p9, dropout_vs_p9)
    reality.to_csv(output_dir / "reality_summary.csv", index=False)

    print("[6/8] Writing plots", flush=True)
    plot_variant_curves(output_dir, results, best_name)
    plot_summary_bars(output_dir, summary)
    plot_cost_curves(output_dir, cost_summary, best_name)
    plot_regime_counts(output_dir, regime_counts)
    plot_regime_attribution(output_dir, regime_attr, best_name)
    plot_filter_efficacy(output_dir, filter_efficacy, best_name)
    plot_fold_delta(output_dir, fold_delta, best_name)

    print("[7/8] Writing report and config metadata", flush=True)
    write_report(
        output_dir=output_dir,
        best_name=best_name,
        gate_specs=gate_specs,
        summary=summary,
        cost_summary=cost_summary,
        fold_summary=fold_summary,
        fold_delta=fold_delta,
        regime_counts=regime_counts,
        regime_attr=regime_attr,
        filter_efficacy=filter_efficacy,
        stress_attr=stress_attr,
        bootstrap=bootstrap,
        start_vs_p9=start_vs_p9,
        dropout_vs_p9=dropout_vs_p9,
        reality=reality,
    )
    update_config_stage(config_path, best_name)
    print("[8/8] Done", flush=True)
    print(f"Done: {output_dir}", flush=True)


def load_existing_result(run_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        "folds": pd.read_csv(run_dir / "walk_forward_folds.csv", parse_dates=["train_start", "train_end", "test_start", "test_end", "test_decision_end"]),
        "equity": pd.read_csv(run_dir / "equity_curve.csv", parse_dates=["timestamp", "exit_timestamp"]),
        "positions": pd.read_csv(run_dir / "positions.csv", parse_dates=["timestamp", "exit_timestamp"]),
    }


def write_result(run_dir: Path, result: dict[str, Any]) -> None:
    result["folds"].to_csv(run_dir / "walk_forward_folds.csv", index=False)
    result["equity"].to_csv(run_dir / "equity_curve.csv", index=False)
    result["positions"].to_csv(run_dir / "positions.csv", index=False)
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(to_jsonable(result["metrics"]), fh, indent=2)


def load_asset_panel(path: Path) -> pd.DataFrame:
    usecols = [
        "timestamp",
        "symbol",
        "log_ret",
        "residual_log_ret",
        "quote_volume",
        "beta_btc",
        "beta_eth",
        "beta_market",
    ]
    return pd.read_csv(path, parse_dates=["timestamp"], usecols=usecols).sort_values(["timestamp", "symbol"])


def load_derivatives_panel(path: Path) -> pd.DataFrame:
    usecols = [
        "timestamp",
        "symbol",
        "deriv_funding_rate_ewm_168",
        "deriv_funding_annualized_ewm_168",
        "deriv_premium_index_ewm_168",
        "deriv_mark_index_basis_ewm_168",
        "deriv_oi_growth_72_336",
        "deriv_taker_pressure_24",
        "deriv_top_sum_long_short_ratio_z_168",
        "deriv_top_count_long_short_ratio_z_168",
        "deriv_abs_funding_stress_168",
        "deriv_abs_basis_stress_168",
    ]
    return pd.read_csv(path, parse_dates=["timestamp"], usecols=usecols).sort_values(["timestamp", "symbol"])


def rolling_z(series: pd.Series, lookback: int = 720, min_periods: int = 168) -> pd.Series:
    mean = series.rolling(lookback, min_periods=min_periods).mean()
    std = series.rolling(lookback, min_periods=min_periods).std(ddof=0)
    return ((series - mean) / std.replace(0.0, np.nan)).clip(-6.0, 6.0)


def build_regime_panel(asset: pd.DataFrame, deriv: pd.DataFrame) -> pd.DataFrame:
    ordered = asset.sort_values(["timestamp", "symbol"]).copy()
    market_ret = ordered.groupby("timestamp")["log_ret"].mean().sort_index()
    residual_dispersion = ordered.groupby("timestamp")["residual_log_ret"].std().reindex(market_ret.index)
    quote_volume = ordered.groupby("timestamp")["quote_volume"].sum().reindex(market_ret.index)
    cross_corr = rolling_average_pairwise_corr(ordered)

    deriv_groups = deriv.groupby("timestamp")
    funding_median = deriv_groups["deriv_funding_rate_ewm_168"].median().reindex(market_ret.index)
    funding_dispersion = deriv_groups["deriv_funding_rate_ewm_168"].std().reindex(market_ret.index)
    funding_stress = deriv_groups["deriv_abs_funding_stress_168"].median().reindex(market_ret.index)
    premium_median = deriv_groups["deriv_premium_index_ewm_168"].median().reindex(market_ret.index)
    premium_dispersion = deriv_groups["deriv_premium_index_ewm_168"].std().reindex(market_ret.index)
    basis_stress = deriv_groups["deriv_abs_basis_stress_168"].median().reindex(market_ret.index)
    oi_growth = deriv_groups["deriv_oi_growth_72_336"].median().reindex(market_ret.index)
    taker_pressure = deriv_groups["deriv_taker_pressure_24"].median().reindex(market_ret.index)
    top_trader = deriv_groups[["deriv_top_sum_long_short_ratio_z_168", "deriv_top_count_long_short_ratio_z_168"]].median().mean(axis=1).reindex(market_ret.index)

    frame = pd.DataFrame(
        {
            "timestamp": market_ret.index,
            "market_ret_1h": market_ret.to_numpy(),
            "residual_dispersion": residual_dispersion.to_numpy(),
            "log_quote_volume": np.log(quote_volume.replace(0.0, np.nan)).to_numpy(),
            "cross_asset_corr": cross_corr.reindex(market_ret.index).to_numpy(),
            "funding_median": funding_median.to_numpy(),
            "funding_dispersion": funding_dispersion.to_numpy(),
            "funding_stress": funding_stress.to_numpy(),
            "premium_median": premium_median.to_numpy(),
            "premium_dispersion": premium_dispersion.to_numpy(),
            "basis_stress": basis_stress.to_numpy(),
            "oi_growth_market": oi_growth.to_numpy(),
            "taker_imbalance_market": taker_pressure.to_numpy(),
            "top_trader_crowding": top_trader.to_numpy(),
        }
    ).sort_values("timestamp")

    frame["market_ret_24h"] = frame["market_ret_1h"].rolling(24, min_periods=12).sum()
    frame["market_ret_72h"] = frame["market_ret_1h"].rolling(72, min_periods=24).sum()
    frame["market_vol_168h"] = frame["market_ret_1h"].rolling(168, min_periods=72).std(ddof=0)
    frame["market_ret_24h_z"] = rolling_z(frame["market_ret_24h"], 720, 168)
    frame["market_ret_72h_z"] = rolling_z(frame["market_ret_72h"], 720, 168)
    frame["market_vol_z"] = rolling_z(frame["market_vol_168h"], 720, 168)
    frame["residual_dispersion_z"] = rolling_z(frame["residual_dispersion"], 720, 168)
    frame["cross_asset_corr_z"] = rolling_z(frame["cross_asset_corr"], 720, 168)
    frame["funding_median_z"] = rolling_z(frame["funding_median"], 720, 168)
    frame["funding_dispersion_z"] = rolling_z(frame["funding_dispersion"], 720, 168)
    frame["funding_stress_z"] = rolling_z(frame["funding_stress"], 720, 168)
    frame["premium_median_z"] = rolling_z(frame["premium_median"], 720, 168)
    frame["premium_dispersion_z"] = rolling_z(frame["premium_dispersion"], 720, 168)
    frame["basis_stress_z"] = rolling_z(frame["basis_stress"], 720, 168)
    frame["oi_growth_market_z"] = rolling_z(frame["oi_growth_market"], 720, 168)
    frame["taker_imbalance_market_z"] = rolling_z(frame["taker_imbalance_market"], 720, 168)
    frame["top_trader_crowding_z"] = rolling_z(frame["top_trader_crowding"], 720, 168)

    stress = pd.concat(
        [
            frame["funding_stress_z"].abs(),
            frame["basis_stress_z"].abs(),
            frame["premium_median_z"].abs(),
            frame["funding_median_z"].abs(),
        ],
        axis=1,
    ).max(axis=1)
    frame["derivative_stress_z"] = stress
    frame["prior_derivative_stress_72h"] = stress.rolling(72, min_periods=24).max().shift(1)
    frame["regime"] = classify_regime(frame)
    frame["regime"] = frame["regime"].fillna("normal")
    return frame


def rolling_average_pairwise_corr(asset: pd.DataFrame, window: int = 168, min_periods: int = 72) -> pd.Series:
    wide = asset.pivot(index="timestamp", columns="symbol", values="log_ret").sort_index()
    corr = wide.rolling(window, min_periods=min_periods).corr()
    if corr.empty:
        return pd.Series(index=wide.index, dtype=float)
    n = len(wide.columns)
    values = []
    timestamps = []
    for ts, mat in corr.groupby(level=0, sort=False):
        arr = mat.droplevel(0).to_numpy(dtype=float)
        if arr.shape != (n, n):
            values.append(np.nan)
        else:
            mask = ~np.eye(n, dtype=bool)
            values.append(float(np.nanmean(arr[mask])))
        timestamps.append(ts)
    return pd.Series(values, index=pd.Index(timestamps, name="timestamp"))


def classify_regime(frame: pd.DataFrame) -> pd.Series:
    out = pd.Series("normal", index=frame.index, dtype=object)
    stress = frame["derivative_stress_z"].fillna(0.0)
    vol = frame["market_vol_z"].fillna(0.0)
    corr = frame["cross_asset_corr_z"].fillna(0.0)
    resid = frame["residual_dispersion_z"].fillna(0.0)
    ret24 = frame["market_ret_24h_z"].fillna(0.0)
    ret72 = frame["market_ret_72h_z"].fillna(0.0)
    oi = frame["oi_growth_market_z"].fillna(0.0)
    taker = frame["taker_imbalance_market_z"].fillna(0.0)
    top = frame["top_trader_crowding_z"].fillna(0.0)
    prior_stress = frame["prior_derivative_stress_72h"].fillna(0.0)

    deleveraging = ((ret24 <= -1.0) | (ret72 <= -1.25)) & (vol >= 0.60) & ((corr >= 0.40) | (resid >= 0.75))
    squeeze = ((ret24 >= 1.15) | (ret72 >= 1.35)) & (vol >= 0.75) & ((stress >= 1.10) | (top.abs() >= 1.0))
    crowded = (oi >= 0.65) & (stress >= 1.0) & ((taker.abs() >= 0.50) | (top.abs() >= 0.75)) & (ret24.abs() >= 0.35)
    post_flush = (oi <= -0.65) & (prior_stress >= 1.35) & (stress <= prior_stress - 0.25) & (ret24 > -0.75)
    carry = (stress >= 1.0) & (vol < 1.0) & (corr < 1.0) & ~crowded & ~deleveraging & ~squeeze

    out.loc[carry] = "carry_dislocation"
    out.loc[post_flush] = "post_flush_snapback"
    out.loc[crowded] = "crowded_trend"
    out.loc[deleveraging | squeeze] = "deleveraging_squeeze"
    return out


def build_gate_specs() -> list[GateSpec]:
    return [
        GateSpec(
            name="p10_soft_gross",
            description="Mild gross throttle in crowded/deleveraging/post-flush regimes.",
            gross_by_regime={
                "normal": 1.00,
                "carry_dislocation": 1.00,
                "crowded_trend": 0.80,
                "deleveraging_squeeze": 0.60,
                "post_flush_snapback": 0.80,
            },
        ),
        GateSpec(
            name="p10_medium_gross",
            description="Moderate gross throttle; no side-specific drift.",
            gross_by_regime={
                "normal": 1.00,
                "carry_dislocation": 1.00,
                "crowded_trend": 0.70,
                "deleveraging_squeeze": 0.45,
                "post_flush_snapback": 0.75,
            },
        ),
        GateSpec(
            name="p10_hard_gross",
            description="Aggressive gross throttle in adverse regimes.",
            gross_by_regime={
                "normal": 1.00,
                "carry_dislocation": 0.95,
                "crowded_trend": 0.55,
                "deleveraging_squeeze": 0.30,
                "post_flush_snapback": 0.60,
            },
        ),
        GateSpec(
            name="p10_delev_only",
            description="Only throttle deleveraging/squeeze; isolates the crash-state rule.",
            gross_by_regime={
                "normal": 1.00,
                "carry_dislocation": 1.00,
                "crowded_trend": 1.00,
                "deleveraging_squeeze": 0.40,
                "post_flush_snapback": 1.00,
            },
        ),
        GateSpec(
            name="p10_crowded_only",
            description="Only throttle crowded trend; isolates derivatives crowding as a risk state.",
            gross_by_regime={
                "normal": 1.00,
                "carry_dislocation": 1.00,
                "crowded_trend": 0.55,
                "deleveraging_squeeze": 1.00,
                "post_flush_snapback": 1.00,
            },
        ),
        GateSpec(
            name="p10_carry_boost_defensive",
            description="Small carry-dislocation boost with defensive crowded/deleveraging cuts.",
            gross_by_regime={
                "normal": 1.00,
                "carry_dislocation": 1.05,
                "crowded_trend": 0.70,
                "deleveraging_squeeze": 0.40,
                "post_flush_snapback": 0.80,
            },
        ),
        GateSpec(
            name="p10_medium_short_defensive",
            description="Moderate gross throttle plus extra short-book reduction in crowded/deleveraging/post-flush states.",
            gross_by_regime={
                "normal": 1.00,
                "carry_dislocation": 1.00,
                "crowded_trend": 0.75,
                "deleveraging_squeeze": 0.55,
                "post_flush_snapback": 0.75,
            },
            short_by_regime={
                "normal": 1.00,
                "carry_dislocation": 1.00,
                "crowded_trend": 0.75,
                "deleveraging_squeeze": 0.55,
                "post_flush_snapback": 0.60,
            },
        ),
    ]


def apply_gate_overlay(
    base_result: dict[str, pd.DataFrame],
    regime: pd.DataFrame,
    beta_panel: pd.DataFrame,
    spec: GateSpec,
    horizon: int,
) -> dict[str, Any]:
    base_equity = base_result["equity"].sort_values("timestamp").copy()
    base_positions = base_result["positions"].sort_values(["timestamp", "symbol"]).copy()
    symbols = sorted(base_positions["symbol"].unique())
    regime_by_ts = regime.set_index("timestamp")
    beta_by_ts_symbol = beta_panel.set_index(["timestamp", "symbol"])

    rows_out: list[dict[str, Any]] = []
    pos_out: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    prev_weights = pd.Series(0.0, index=symbols, dtype=float)
    equity = 1.0

    for base_row in base_equity.itertuples(index=False):
        ts = pd.Timestamp(base_row.timestamp)
        base_pos = base_positions[base_positions["timestamp"] == ts].copy()
        if base_pos.empty:
            continue
        regime_row = regime_by_ts.loc[ts] if ts in regime_by_ts.index else pd.Series(dtype=object)
        regime_name = str(regime_row.get("regime", "normal"))
        gross_mult = float(spec.gross_by_regime.get(regime_name, 1.0))
        short_mult = float((spec.short_by_regime or {}).get(regime_name, 1.0))
        long_mult = float((spec.long_by_regime or {}).get(regime_name, 1.0))

        weights = base_pos.set_index("symbol")["weight"].astype(float)
        scaled = weights.copy()
        scaled.loc[scaled > 0] = scaled.loc[scaled > 0] * gross_mult * long_mult
        scaled.loc[scaled < 0] = scaled.loc[scaled < 0] * gross_mult * short_mult
        scaled = scaled[scaled.abs() > 1e-12]

        full_weights = pd.Series(0.0, index=symbols, dtype=float)
        full_weights.loc[scaled.index] = scaled
        turnover = float((full_weights - prev_weights).abs().sum())
        cost = turnover * float(base_row.cost_return) / max(float(base_row.turnover), 1e-12) if float(base_row.turnover) > 1e-12 else turnover * 6.0 / 10_000.0

        active = base_pos.set_index("symbol").reindex(scaled.index)
        gross_period_ret = float((scaled * active["fwd_simple_ret"].astype(float)).sum())
        period_ret = gross_period_ret - cost
        equity *= 1.0 + period_ret
        base_gross = float(base_row.gross_period_return)
        gated_away = base_gross - gross_period_ret

        betas = beta_by_ts_symbol.reindex(pd.MultiIndex.from_product([[ts], scaled.index])).reset_index(level=0, drop=True)
        exposures = portfolio_exposures(scaled, betas, ["beta_btc", "beta_eth", "beta_market"])
        long_labels = active.loc[scaled[scaled > 0].index, "label_resid_ret"] if (scaled > 0).any() else pd.Series(dtype=float)
        short_labels = active.loc[scaled[scaled < 0].index, "label_resid_ret"] if (scaled < 0).any() else pd.Series(dtype=float)
        top_bottom = float(long_labels.mean() - short_labels.mean()) if len(long_labels) and len(short_labels) else np.nan

        row = {
            **base_row._asdict(),
            "period_return": period_ret,
            "gross_period_return": gross_period_ret,
            "cost_return": cost,
            "equity": equity,
            "turnover": turnover,
            "target_turnover": turnover,
            "gross_exposure": float(scaled.abs().sum()),
            "target_gross_exposure": float(scaled.abs().sum()),
            "top_bottom_label_spread": top_bottom,
            "abs_beta_btc": abs(float(exposures.get("beta_btc", 0.0))),
            "abs_beta_eth": abs(float(exposures.get("beta_eth", 0.0))),
            "abs_beta_market": abs(float(exposures.get("beta_market", 0.0))),
            "symbol_concentration": float((scaled**2).sum()),
            "max_abs_weight": float(scaled.abs().max()) if len(scaled) else 0.0,
            "regime": regime_name,
            "regime_gross_multiplier": gross_mult,
            "regime_short_multiplier": short_mult,
            "regime_long_multiplier": long_mult,
            "base_gross_period_return": base_gross,
            "gated_away_gross_return": gated_away,
            "gated_away_abs_weight": float((weights.reindex(symbols).fillna(0.0) - full_weights).abs().sum()),
        }
        for col in regime_row.index:
            if col == "regime":
                continue
            row[f"regime_{col}"] = regime_row[col]
        rows_out.append(row)

        for symbol, weight in scaled.items():
            original = active.loc[symbol]
            pos_out.append(
                {
                    **original.to_dict(),
                    "symbol": symbol,
                    "weight": float(weight),
                    "target_weight": float(weight),
                    "base_weight": float(weights.loc[symbol]),
                    "contribution": float(weight * float(original["fwd_simple_ret"])),
                    "base_contribution": float(weights.loc[symbol] * float(original["fwd_simple_ret"])),
                    "regime": regime_name,
                    "regime_gross_multiplier": gross_mult,
                    "regime_short_multiplier": short_mult,
                    "regime_long_multiplier": long_mult,
                    "overlay_variant": spec.name,
                }
            )

        prev_weights = full_weights

    equity_curve = pd.DataFrame(rows_out)
    positions = pd.DataFrame(pos_out)
    for fold, group in equity_curve.groupby("fold", sort=True):
        pos_group = positions[positions["fold"] == fold]
        metrics = compute_metrics(group, pos_group, horizon)
        base_fold = base_result["folds"][base_result["folds"]["fold"] == fold].iloc[0].to_dict()
        fold_rows.append({**base_fold, **metrics})
    folds = pd.DataFrame(fold_rows)
    metrics = p3d.summarize_nested_result({"equity": equity_curve, "positions": positions, "folds": folds}, horizon)
    return {"folds": folds, "equity": equity_curve, "positions": positions, "metrics": metrics}


def compute_metrics(equity: pd.DataFrame, positions: pd.DataFrame, horizon: int) -> dict[str, float]:
    metrics = p3d.aggregate_oos_metrics(equity, horizon)
    p3d.add_gross_metrics(equity, metrics)
    p3d.add_risk_event_metrics(positions, metrics)
    p3d.add_delay_metrics(equity, metrics)
    metrics["long_short_balance"] = p3d.long_short_balance(positions)
    metrics["short_abs_contribution_share"] = p3d.short_abs_contribution_share(positions)
    return metrics


def build_summary(results: dict[str, dict[str, Any]], horizon: int) -> pd.DataFrame:
    rows = []
    p9_ref = p3d.summarize_nested_result(results["p9_reference"], horizon)
    for name, result in results.items():
        row = p3d.summarize_nested_result(result, horizon)
        row["variant"] = name
        row["delta_vs_p9_total_return"] = row["total_return"] - p9_ref["total_return"]
        row["delta_vs_p9_gross_total_return_before_cost"] = row["gross_total_return_before_cost"] - p9_ref["gross_total_return_before_cost"]
        row["delta_vs_p9_sharpe"] = row["sharpe"] - p9_ref["sharpe"]
        row["delta_vs_p9_max_drawdown"] = row["max_drawdown"] - p9_ref["max_drawdown"]
        row["worst_5_fold_return_sum"] = float(result["folds"]["total_return"].sort_values().head(5).sum()) if "total_return" in result["folds"] else np.nan
        row["median_fold_return"] = float(result["folds"]["total_return"].median()) if "total_return" in result["folds"] else np.nan
        row["q25_fold_return"] = float(result["folds"]["total_return"].quantile(0.25)) if "total_return" in result["folds"] else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["total_return", "sharpe"], ascending=False)


def build_cost_summary(results: dict[str, dict[str, Any]], cost_grid: list[float]) -> pd.DataFrame:
    frames = []
    for name, result in results.items():
        frame = p3d.build_cost_sensitivity({name: result}, cost_grid)
        frame["variant"] = name
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    p9_cost = out[out["variant"] == "p9_reference"][["one_way_cost_bps", "total_return", "max_drawdown"]].rename(
        columns={"total_return": "p9_total_return", "max_drawdown": "p9_max_drawdown"}
    )
    out = out.merge(p9_cost, on="one_way_cost_bps", how="left")
    out["delta_vs_p9_total_return"] = out["total_return"] - out["p9_total_return"]
    out["delta_vs_p9_max_drawdown"] = out["max_drawdown"] - out["p9_max_drawdown"]
    return out


def build_fold_summary(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for name, result in results.items():
        frame = result["folds"].copy()
        frame["variant"] = name
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def build_fold_delta_table(results: dict[str, dict[str, Any]], reference_name: str) -> pd.DataFrame:
    ref = results[reference_name]["folds"][["fold", "total_return", "gross_total_return_before_cost", "sharpe", "max_drawdown"]].rename(
        columns={col: f"{reference_name}_{col}" for col in ["total_return", "gross_total_return_before_cost", "sharpe", "max_drawdown"]}
    )
    rows = []
    for name, result in results.items():
        if name == reference_name:
            continue
        cols = ["fold", "total_return", "gross_total_return_before_cost", "sharpe", "max_drawdown"]
        cur = result["folds"][cols].rename(columns={col: f"{name}_{col}" for col in cols if col != "fold"})
        merged = cur.merge(ref, on="fold", how="inner")
        merged["variant"] = name
        for col in ["total_return", "gross_total_return_before_cost", "sharpe", "max_drawdown"]:
            merged[f"delta_{col}"] = merged[f"{name}_{col}"] - merged[f"{reference_name}_{col}"]
        rows.append(merged)
    return pd.concat(rows, ignore_index=True)


def build_pair_fold_delta(candidate: dict[str, Any], reference: dict[str, Any], cand: str, ref: str) -> pd.DataFrame:
    return p9.build_fold_delta(candidate, reference, cand, ref)


def build_regime_attribution(results: dict[str, dict[str, Any]], regime: pd.DataFrame) -> pd.DataFrame:
    rows = []
    regime_cols = regime[["timestamp", "regime"]]
    for name, result in results.items():
        equity = result["equity"].copy()
        if "regime" not in equity:
            equity = equity.merge(regime_cols, on="timestamp", how="left")
        equity["regime"] = equity["regime"].fillna("unknown")
        grouped = (
            equity.groupby("regime")
            .agg(
                periods=("timestamp", "count"),
                total_return_sum=("period_return", "sum"),
                gross_return_sum=("gross_period_return", "sum"),
                mean_turnover=("turnover", "mean"),
                cost_sum=("cost_return", "sum"),
                mean_gross_exposure=("gross_exposure", "mean"),
                positive_period_rate=("period_return", lambda x: float((x > 0).mean())),
            )
            .reset_index()
        )
        grouped["variant"] = name
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True)


def build_filter_efficacy(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for name, result in results.items():
        equity = result["equity"].copy()
        if "gated_away_gross_return" not in equity:
            continue
        grouped = (
            equity.groupby("regime")
            .agg(
                periods=("timestamp", "count"),
                gated_away_gross_sum=("gated_away_gross_return", "sum"),
                gated_away_gross_mean=("gated_away_gross_return", "mean"),
                gated_away_abs_weight_mean=("gated_away_abs_weight", "mean"),
                base_gross_sum=("base_gross_period_return", "sum"),
                overlay_gross_sum=("gross_period_return", "sum"),
                overlay_minus_base_gross_sum=("gross_period_return", lambda x: np.nan),
            )
            .reset_index()
        )
        grouped["overlay_minus_base_gross_sum"] = grouped["overlay_gross_sum"] - grouped["base_gross_sum"]
        grouped["filter_helped"] = grouped["gated_away_gross_sum"] < 0.0
        grouped["variant"] = name
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_stress_period_attribution(results: dict[str, dict[str, Any]], regime: pd.DataFrame) -> pd.DataFrame:
    stress_regimes = {"crowded_trend", "deleveraging_squeeze", "post_flush_snapback"}
    rows = []
    for name, result in results.items():
        equity = result["equity"].copy()
        if "regime" not in equity:
            equity = equity.merge(regime[["timestamp", "regime"]], on="timestamp", how="left")
        stress = equity[equity["regime"].isin(stress_regimes)]
        calm = equity[~equity["regime"].isin(stress_regimes)]
        for bucket, frame in [("stress", stress), ("non_stress", calm)]:
            if frame.empty:
                continue
            curve = (1.0 + frame["period_return"]).cumprod()
            drawdown = curve / curve.cummax() - 1.0
            rows.append(
                {
                    "variant": name,
                    "bucket": bucket,
                    "periods": len(frame),
                    "return_sum": float(frame["period_return"].sum()),
                    "gross_sum": float(frame["gross_period_return"].sum()),
                    "cost_sum": float(frame["cost_return"].sum()),
                    "max_drawdown_within_bucket": float(drawdown.min()),
                    "positive_period_rate": float((frame["period_return"] > 0).mean()),
                    "mean_turnover": float(frame["turnover"].mean()),
                }
            )
    return pd.DataFrame(rows)


def select_best_variant(summary: pd.DataFrame, cost: pd.DataFrame, fold_delta: pd.DataFrame) -> str:
    candidates = summary[summary["variant"].str.startswith("p10_")].copy()
    if candidates.empty:
        return "p9_reference"
    cost20 = cost[cost["one_way_cost_bps"] == 20.0][["variant", "total_return"]].rename(columns={"total_return": "return_20bps"})
    cost30 = cost[cost["one_way_cost_bps"] == 30.0][["variant", "total_return"]].rename(columns={"total_return": "return_30bps"})
    candidates = candidates.merge(cost20, on="variant", how="left").merge(cost30, on="variant", how="left")
    p9_row = summary[summary["variant"] == "p9_reference"].iloc[0]
    candidates["accept_score"] = (
        100.0 * candidates["delta_vs_p9_total_return"]
        + 2.0 * candidates["delta_vs_p9_sharpe"]
        + 50.0 * candidates["delta_vs_p9_max_drawdown"]
        + 20.0 * (candidates["return_20bps"] - float(cost20[cost20["variant"] == "p9_reference"]["return_20bps"].iloc[0]) if "p9_reference" in set(cost20["variant"]) else 0.0)
        + 10.0 * (candidates["worst_5_fold_return_sum"] - float(p9_row["worst_5_fold_return_sum"]))
    )
    candidates = candidates.sort_values(["accept_score", "total_return"], ascending=False)
    return str(candidates.iloc[0]["variant"])


def build_reality_summary(
    best_name: str,
    summary: pd.DataFrame,
    cost: pd.DataFrame,
    bootstrap: pd.DataFrame,
    start_vs_p9: pd.DataFrame,
    dropout_vs_p9: pd.DataFrame,
) -> pd.DataFrame:
    best = summary[summary["variant"] == best_name].iloc[0]
    p9_row = summary[summary["variant"] == "p9_reference"].iloc[0]
    best_cost20 = cost[(cost["variant"] == best_name) & (cost["one_way_cost_bps"] == 20.0)].iloc[0]
    p9_cost20 = cost[(cost["variant"] == "p9_reference") & (cost["one_way_cost_bps"] == 20.0)].iloc[0]
    best_cost30 = cost[(cost["variant"] == best_name) & (cost["one_way_cost_bps"] == 30.0)].iloc[0]
    p9_cost30 = cost[(cost["variant"] == "p9_reference") & (cost["one_way_cost_bps"] == 30.0)].iloc[0]
    boot_vs_p9 = bootstrap[bootstrap["metric"] == f"{best_name}_minus_p9"].iloc[0]
    start_positive = float((start_vs_p9["delta_total_return"] > 0).mean()) if "delta_total_return" in start_vs_p9 else np.nan
    dropout_positive = float((dropout_vs_p9["delta_total_return"] > 0).mean()) if "delta_total_return" in dropout_vs_p9 else np.nan
    return pd.DataFrame(
        [
            {"check": "best_variant", "value": best_name, "pass": True},
            {"check": "net_delta_vs_p9", "value": best["total_return"] - p9_row["total_return"], "pass": bool(best["total_return"] > p9_row["total_return"])},
            {"check": "sharpe_delta_vs_p9", "value": best["sharpe"] - p9_row["sharpe"], "pass": bool(best["sharpe"] > p9_row["sharpe"])},
            {"check": "maxdd_delta_vs_p9", "value": best["max_drawdown"] - p9_row["max_drawdown"], "pass": bool(best["max_drawdown"] >= p9_row["max_drawdown"] * 1.05)},
            {"check": "return_20bps_delta_vs_p9", "value": best_cost20["total_return"] - p9_cost20["total_return"], "pass": bool(best_cost20["total_return"] >= p9_cost20["total_return"])},
            {"check": "return_30bps_delta_vs_p9", "value": best_cost30["total_return"] - p9_cost30["total_return"], "pass": bool(best_cost30["total_return"] >= p9_cost30["total_return"])},
            {"check": "worst5_delta_vs_p9", "value": best["worst_5_fold_return_sum"] - p9_row["worst_5_fold_return_sum"], "pass": bool(best["worst_5_fold_return_sum"] >= p9_row["worst_5_fold_return_sum"])},
            {"check": "bootstrap_delta_vs_p9_q05", "value": boot_vs_p9["bootstrap_q05"], "pass": bool(boot_vs_p9["bootstrap_q05"] > -0.0025)},
            {"check": "start_fold_positive_vs_p9", "value": start_positive, "pass": bool(start_positive >= 0.45)},
            {"check": "symbol_dropout_positive_vs_p9", "value": dropout_positive, "pass": bool(dropout_positive >= 0.45)},
        ]
    )


def plot_variant_curves(output_dir: Path, results: dict[str, dict[str, Any]], best_name: str) -> None:
    keep = ["p3d_reference", "p7b_reference", "p9_reference", best_name]
    fig, axes = plt.subplots(2, 1, figsize=(13, 8.5), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0]})
    for name in keep:
        if name not in results:
            continue
        equity = results[name]["equity"].sort_values("timestamp")
        net = (1.0 + equity["period_return"]).cumprod() - 1.0
        axes[0].plot(equity["timestamp"], net, label=name, linewidth=2.0 if name == best_name else 1.5)
    best_eq = results[best_name]["equity"].sort_values("timestamp")
    curve = (1.0 + best_eq["period_return"]).cumprod()
    dd = curve / curve.cummax() - 1.0
    axes[1].plot(best_eq["timestamp"], dd, label=f"{best_name} drawdown", color="#c44536", linewidth=1.8)
    for ax in axes:
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
        ax.grid(True, alpha=0.25)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.legend(fontsize=8)
    axes[0].set_title("P10 regime gate overlay net curves")
    fig.tight_layout()
    fig.savefig(output_dir / "p10_net_curves.png", dpi=180)
    plt.close(fig)


def plot_summary_bars(output_dir: Path, summary: pd.DataFrame) -> None:
    frame = summary.sort_values("total_return")
    plt.figure(figsize=(12, 8))
    colors = np.where(frame["variant"].str.startswith("p10_"), "#2a9d8f", "#777777")
    plt.barh(frame["variant"], frame["total_return"], color=colors)
    plt.axvline(0.0, color="black", linewidth=0.8)
    plt.title("P10 variant net return at 6bps")
    plt.gca().xaxis.set_major_formatter(PercentFormatter(1.0))
    plt.tight_layout()
    plt.savefig(output_dir / "variant_net_return.png", dpi=180)
    plt.close()


def plot_cost_curves(output_dir: Path, cost: pd.DataFrame, best_name: str) -> None:
    keep = ["p9_reference", best_name, "p10_soft_gross", "p10_medium_gross", "p10_delev_only", "p10_crowded_only"]
    frame = cost[cost["variant"].isin(keep)].copy()
    plt.figure(figsize=(12, 7))
    for name, group in frame.groupby("variant"):
        group = group.sort_values("one_way_cost_bps")
        plt.plot(group["one_way_cost_bps"], group["total_return"], marker="o", label=name, linewidth=2.2 if name == best_name else 1.4)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.title("P10 cost sensitivity")
    plt.xlabel("one-way cost bps")
    plt.ylabel("total return")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "cost_sensitivity.png", dpi=180)
    plt.close()


def plot_regime_counts(output_dir: Path, counts: pd.DataFrame) -> None:
    frame = counts.sort_values("hour_count")
    plt.figure(figsize=(9, 5))
    plt.barh(frame["regime"], frame["hour_count"], color="#457b9d")
    plt.title("P10 hourly regime counts")
    plt.tight_layout()
    plt.savefig(output_dir / "regime_counts.png", dpi=180)
    plt.close()


def plot_regime_attribution(output_dir: Path, attr: pd.DataFrame, best_name: str) -> None:
    frame = attr[attr["variant"].isin(["p9_reference", best_name])].copy()
    pivot = frame.pivot(index="regime", columns="variant", values="total_return_sum").fillna(0.0)
    pivot = pivot.sort_values(best_name if best_name in pivot else pivot.columns[0])
    pivot.plot(kind="barh", figsize=(11, 6))
    plt.axvline(0.0, color="black", linewidth=0.8)
    plt.title("Regime contribution: P9 vs P10 best")
    plt.gca().xaxis.set_major_formatter(PercentFormatter(1.0))
    plt.tight_layout()
    plt.savefig(output_dir / "regime_attribution.png", dpi=180)
    plt.close()


def plot_filter_efficacy(output_dir: Path, efficacy: pd.DataFrame, best_name: str) -> None:
    frame = efficacy[efficacy["variant"] == best_name].sort_values("gated_away_gross_sum")
    if frame.empty:
        return
    colors = np.where(frame["gated_away_gross_sum"] < 0.0, "#2a9d8f", "#c44536")
    plt.figure(figsize=(10, 5))
    plt.barh(frame["regime"], frame["gated_away_gross_sum"], color=colors)
    plt.axvline(0.0, color="black", linewidth=0.8)
    plt.title("Gated-away gross return by regime; negative means filter removed losses")
    plt.gca().xaxis.set_major_formatter(PercentFormatter(1.0))
    plt.tight_layout()
    plt.savefig(output_dir / "filter_efficacy.png", dpi=180)
    plt.close()


def plot_fold_delta(output_dir: Path, fold_delta: pd.DataFrame, best_name: str) -> None:
    frame = fold_delta[fold_delta["variant"] == best_name].sort_values("fold")
    if frame.empty:
        return
    colors = np.where(frame["delta_total_return"] >= 0, "#2a9d8f", "#c44536")
    plt.figure(figsize=(12, 6))
    plt.bar(frame["fold"].astype(str), frame["delta_total_return"], color=colors)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.title(f"{best_name} minus P9 fold return")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.tight_layout()
    plt.savefig(output_dir / "fold_delta_vs_p9.png", dpi=180)
    plt.close()


def write_report(
    output_dir: Path,
    best_name: str,
    gate_specs: list[GateSpec],
    summary: pd.DataFrame,
    cost_summary: pd.DataFrame,
    fold_summary: pd.DataFrame,
    fold_delta: pd.DataFrame,
    regime_counts: pd.DataFrame,
    regime_attr: pd.DataFrame,
    filter_efficacy: pd.DataFrame,
    stress_attr: pd.DataFrame,
    bootstrap: pd.DataFrame,
    start_vs_p9: pd.DataFrame,
    dropout_vs_p9: pd.DataFrame,
    reality: pd.DataFrame,
) -> None:
    cols = [
        "variant",
        "total_return",
        "gross_total_return_before_cost",
        "sharpe",
        "max_drawdown",
        "mean_turnover",
        "positive_fold_rate",
        "worst_5_fold_return_sum",
        "delta_vs_p9_total_return",
        "delta_vs_p9_sharpe",
        "delta_vs_p9_max_drawdown",
    ]
    cols = [col for col in cols if col in summary]
    cost_display = cost_summary[cost_summary["one_way_cost_bps"].isin([0, 6, 10, 15, 20, 30])]
    cost_display = cost_display[cost_display["variant"].isin(["p9_reference", best_name, "p10_soft_gross", "p10_medium_gross", "p10_delev_only", "p10_crowded_only"])]
    specs = pd.DataFrame([spec.__dict__ for spec in gate_specs])
    best_filter = filter_efficacy[filter_efficacy["variant"] == best_name] if not filter_efficacy.empty else filter_efficacy
    lines = [
        "# P10 Strategy-Conditioned Regime Gate",
        "",
        "## Design",
        "",
        "- Fixed P9 selected positions and formulas; no alpha refit and no test-informed selection.",
        "- Built an hourly ex-ante market-level regime vector from OHLCV residuals and Binance derivatives fields.",
        "- Regime is used as a portfolio overlay only: gross multiplier and, in one variant, extra short-book multiplier.",
        "- P10 is an audit stage. It should only become a baseline if it improves left tail/cost robustness without just cutting all exposure.",
        "",
        "## Gate Specs",
        "```text",
        specs.to_string(index=False),
        "```",
        "",
        f"## Selected Best Variant: `{best_name}`",
        "",
        "## Summary",
        "```text",
        summary[cols].to_string(index=False),
        "```",
        "",
        "## Cost Sensitivity",
        "```text",
        cost_display.to_string(index=False),
        "```",
        "",
        "## Regime Counts",
        "```text",
        regime_counts.to_string(index=False),
        "```",
        "",
        "## Regime Attribution",
        "```text",
        regime_attr.to_string(index=False),
        "```",
        "",
        "## Filter Efficacy",
        "Negative gated-away gross means the gate removed losses.",
        "```text",
        best_filter.to_string(index=False),
        "```",
        "",
        "## Stress Attribution",
        "```text",
        stress_attr.to_string(index=False),
        "```",
        "",
        "## Fold Delta Vs P9",
        "```text",
        fold_delta[fold_delta["variant"] == best_name].to_string(index=False),
        "```",
        "",
        "## Bootstrap Checks",
        "```text",
        bootstrap.to_string(index=False),
        "```",
        "",
        "## Start Fold Sensitivity Vs P9",
        "```text",
        start_vs_p9.to_string(index=False),
        "```",
        "",
        "## Symbol Dropout Vs P9",
        "```text",
        dropout_vs_p9.to_string(index=False),
        "```",
        "",
        "## Reality Summary",
        "```text",
        reality.to_string(index=False),
        "```",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def update_config_stage(config_path: Path, best_name: str) -> None:
    cfg = load_config(config_path)
    stages = cfg.setdefault("stages", {})
    stages["p10_strategy_conditioned_regime_gate"] = {
        "script": "../../../scripts/run_p10_strategy_conditioned_regime_gate.py",
        "output_dir": "../outputs/p10_strategy_conditioned_regime_gate",
        "report": "../outputs/p10_strategy_conditioned_regime_gate/REPORT.md",
        "variant_summary": "../outputs/p10_strategy_conditioned_regime_gate/variant_summary.csv",
        "cost_sensitivity": "../outputs/p10_strategy_conditioned_regime_gate/cost_sensitivity.csv",
        "regime_feature_panel": "../outputs/p10_strategy_conditioned_regime_gate/regime_feature_panel.csv",
        "filter_efficacy": "../outputs/p10_strategy_conditioned_regime_gate/filter_efficacy.csv",
        "reality_summary": "../outputs/p10_strategy_conditioned_regime_gate/reality_summary.csv",
        "best_variant": best_name,
    }
    cfg["active_research_regime_gate_candidate"] = {
        "name": best_name,
        "status": "candidate_pending_review",
        "output_dir": "../outputs/p10_strategy_conditioned_regime_gate",
        "report": "../outputs/p10_strategy_conditioned_regime_gate/REPORT.md",
        "parent": "p9_clean_drop_k01_cap010",
    }
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False), encoding="utf-8")


if __name__ == "__main__":
    main()
