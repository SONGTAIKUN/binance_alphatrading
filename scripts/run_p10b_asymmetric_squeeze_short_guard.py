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
from crypto_stat_arb.portfolio import constrained_project_weights, portfolio_exposures
import run_p3d_p4_gated_rerank_alphagpt as p3d
import run_p9_clean_derivatives_baseline as p9
import run_p10_strategy_conditioned_regime_gate as p10
from run_turnover_experiments import to_jsonable


EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "baseline_locked_v1"
DEFAULT_CONFIG = EXPERIMENT_ROOT / "configs" / "baseline_locked_v1.yaml"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "outputs" / "p10b_asymmetric_squeeze_short_guard"
P3D_DIR = EXPERIMENT_ROOT / "outputs" / "p3d_p4_gated_rerank_alphagpt" / "nested_incremental_run"
P7B_DIR = EXPERIMENT_ROOT / "outputs" / "p7b_derivatives_incremental_on_p3d" / "nested_incremental_run"
P9_DIR = EXPERIMENT_ROOT / "outputs" / "p9_clean_baseline_package" / "nested_incremental_run"
DERIV_PANEL = EXPERIMENT_ROOT / "inputs" / "binance_derivatives" / "derivatives_hourly_panel.csv.gz"

NEUTRALIZE_COLS = ["intercept", "beta_btc", "beta_eth", "beta_market"]


@dataclass(frozen=True)
class ShortGuardSpec:
    name: str
    description: str
    moderate_threshold: float
    high_threshold: float
    moderate_multiplier: float
    high_multiplier: float
    risk_regimes: tuple[str, ...] = ("crowded_trend", "deleveraging_squeeze", "post_flush_snapback")
    require_risk_regime: bool = True
    protect_strong_short: bool = False
    strong_short_score: float = -0.45
    strong_short_min_multiplier: float = 0.85
    hedged_projection: bool = False
    gross_mode: str = "guarded_gross"


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

    print("[1/8] Loading locked P9/P7b/P3d results and source panels", flush=True)
    p9_result = p10.load_existing_result(P9_DIR)
    p7b_result = p10.load_existing_result(P7B_DIR)
    p3d_result = p10.load_existing_result(P3D_DIR)
    panel_path = p3d.resolve_config_path(config_path, cfg["inputs"]["panel"])
    asset_regime = p10.load_asset_panel(panel_path)
    deriv_regime = p10.load_derivatives_panel(DERIV_PANEL)
    asset_risk = load_asset_risk_panel(panel_path)
    deriv_risk = load_derivative_risk_panel(DERIV_PANEL)

    print("[2/8] Building ex-ante regime and symbol squeeze-risk panels", flush=True)
    regime = p10.build_regime_panel(asset_regime, deriv_regime)
    risk_panel = build_short_squeeze_risk_panel(asset_risk, deriv_risk)
    regime.to_csv(output_dir / "regime_feature_panel.csv", index=False)
    risk_panel.to_csv(output_dir / "short_squeeze_risk_panel.csv", index=False)

    print("[3/8] Replaying P10b asymmetric short-guard variants", flush=True)
    beta_panel = asset_risk[["timestamp", "symbol", "beta_btc", "beta_eth", "beta_market"]].copy()
    specs = build_short_guard_specs()
    results: dict[str, dict[str, Any]] = {
        "p3d_reference": p3d_result,
        "p7b_reference": p7b_result,
        "p9_reference": p9_result,
    }
    for spec in specs:
        print(f"      {spec.name}", flush=True)
        result = apply_short_guard_overlay(p9_result, regime, risk_panel, beta_panel, spec, horizon)
        results[spec.name] = result
        run_dir = ensure_dir(output_dir / "runs" / spec.name)
        p10.write_result(run_dir, result)

    print("[4/8] Building summary, cost, fold, guard, and regime diagnostics", flush=True)
    summary = p10.build_summary(results, horizon)
    summary.to_csv(output_dir / "variant_summary.csv", index=False)
    cost_summary = p10.build_cost_summary(results, cost_grid)
    cost_summary.to_csv(output_dir / "cost_sensitivity.csv", index=False)
    fold_summary = p10.build_fold_summary(results)
    fold_summary.to_csv(output_dir / "fold_summary.csv", index=False)
    fold_delta = p10.build_fold_delta_table(results, "p9_reference")
    fold_delta.to_csv(output_dir / "fold_delta_vs_p9.csv", index=False)
    regime_attr = p10.build_regime_attribution(results, regime)
    regime_attr.to_csv(output_dir / "regime_attribution.csv", index=False)
    filter_efficacy = p10.build_filter_efficacy(results)
    filter_efficacy.to_csv(output_dir / "filter_efficacy.csv", index=False)
    stress_attr = p10.build_stress_period_attribution(results, regime)
    stress_attr.to_csv(output_dir / "stress_period_attribution.csv", index=False)
    guard_symbol = build_guard_symbol_attribution(results)
    guard_symbol.to_csv(output_dir / "guard_symbol_attribution.csv", index=False)
    guard_regime = build_guard_regime_attribution(results)
    guard_regime.to_csv(output_dir / "guard_regime_attribution.csv", index=False)

    print("[5/8] Running robustness checks", flush=True)
    best_name = select_best_p10b(summary, cost_summary)
    best_result = results[best_name]
    p9_fold_delta = p9.build_fold_delta(best_result, p9_result, best_name, "p9")
    p3d_fold_delta = p9.build_fold_delta(best_result, p3d_result, best_name, "p3d")
    bootstrap = pd.concat(
        [
            p9.bootstrap_fold_checks(p9_fold_delta, f"{best_name}_minus_p9", int(args.bootstrap_samples)),
            p9.bootstrap_fold_checks(p3d_fold_delta, f"{best_name}_minus_p3d", int(args.bootstrap_samples)),
        ],
        ignore_index=True,
    )
    bootstrap.to_csv(output_dir / "bootstrap_checks.csv", index=False)
    start_vs_p9 = p9.build_start_fold_sensitivity(best_result, p9_result, best_name, "p9", horizon)
    start_vs_p9.to_csv(output_dir / "start_fold_sensitivity_vs_p9.csv", index=False)
    dropout_vs_p9 = p9.build_symbol_dropout_checks(best_result, p9_result, best_name, "p9", horizon)
    dropout_vs_p9.to_csv(output_dir / "symbol_dropout_vs_p9.csv", index=False)
    reality = build_reality_summary(best_name, summary, cost_summary, bootstrap, start_vs_p9, dropout_vs_p9)
    reality.to_csv(output_dir / "reality_summary.csv", index=False)

    print("[6/8] Writing plots", flush=True)
    plot_net_curves(output_dir, results, best_name)
    plot_cost_curves(output_dir, cost_summary, best_name)
    plot_variant_bars(output_dir, summary)
    plot_guard_benefit(output_dir, guard_symbol, best_name)
    plot_fold_delta(output_dir, fold_delta, best_name)

    print("[7/8] Writing report and config metadata", flush=True)
    write_report(
        output_dir=output_dir,
        specs=specs,
        best_name=best_name,
        summary=summary,
        cost_summary=cost_summary,
        fold_delta=fold_delta,
        regime_attr=regime_attr,
        filter_efficacy=filter_efficacy,
        stress_attr=stress_attr,
        guard_symbol=guard_symbol,
        guard_regime=guard_regime,
        bootstrap=bootstrap,
        start_vs_p9=start_vs_p9,
        dropout_vs_p9=dropout_vs_p9,
        reality=reality,
    )
    update_config_stage(config_path, best_name)
    print("[8/8] Done", flush=True)
    print(f"Done: {output_dir}", flush=True)


def load_asset_risk_panel(path: Path) -> pd.DataFrame:
    usecols = [
        "timestamp",
        "symbol",
        "log_ret",
        "residual_log_ret",
        "resid_z",
        "raw_mom_medium",
        "resid_mom_medium",
        "quote_volume",
        "beta_btc",
        "beta_eth",
        "beta_market",
    ]
    return pd.read_csv(path, parse_dates=["timestamp"], usecols=usecols).sort_values(["symbol", "timestamp"])


def load_derivative_risk_panel(path: Path) -> pd.DataFrame:
    usecols = [
        "timestamp",
        "symbol",
        "deriv_funding_rate_z_168",
        "deriv_premium_index_z_168",
        "deriv_mark_index_basis_z_168",
        "deriv_oi_growth_72_336",
        "deriv_taker_pressure_24",
        "deriv_top_sum_long_short_ratio_z_168",
        "deriv_top_count_long_short_ratio_z_168",
        "deriv_abs_funding_stress_168",
        "deriv_abs_basis_stress_168",
    ]
    return pd.read_csv(path, parse_dates=["timestamp"], usecols=usecols).sort_values(["symbol", "timestamp"])


def build_short_squeeze_risk_panel(asset: pd.DataFrame, deriv: pd.DataFrame) -> pd.DataFrame:
    frame = asset.copy().sort_values(["symbol", "timestamp"])
    grouped = frame.groupby("symbol", group_keys=False)
    frame["raw_ret_24h"] = grouped["log_ret"].rolling(24, min_periods=12).sum().reset_index(level=0, drop=True)
    frame["raw_ret_72h"] = grouped["log_ret"].rolling(72, min_periods=24).sum().reset_index(level=0, drop=True)
    frame["resid_ret_24h"] = grouped["residual_log_ret"].rolling(24, min_periods=12).sum().reset_index(level=0, drop=True)
    frame["resid_ret_72h"] = grouped["residual_log_ret"].rolling(72, min_periods=24).sum().reset_index(level=0, drop=True)
    frame["resid_ret_168h"] = grouped["residual_log_ret"].rolling(168, min_periods=72).sum().reset_index(level=0, drop=True)
    merged = frame.merge(deriv, on=["timestamp", "symbol"], how="left")

    rank_cols = [
        "raw_ret_24h",
        "raw_ret_72h",
        "resid_ret_24h",
        "resid_ret_72h",
        "resid_z",
        "deriv_funding_rate_z_168",
        "deriv_premium_index_z_168",
        "deriv_mark_index_basis_z_168",
        "deriv_oi_growth_72_336",
        "deriv_taker_pressure_24",
        "deriv_top_sum_long_short_ratio_z_168",
        "deriv_top_count_long_short_ratio_z_168",
    ]
    for col in rank_cols:
        merged[f"{col}_rank01"] = cs_rank01(merged, col)
    for col in [
        "deriv_funding_rate_z_168",
        "deriv_premium_index_z_168",
        "deriv_mark_index_basis_z_168",
        "deriv_taker_pressure_24",
        "deriv_top_sum_long_short_ratio_z_168",
        "deriv_top_count_long_short_ratio_z_168",
    ]:
        merged[f"abs_{col}_rank01"] = cs_rank01(merged.assign(_abs=merged[col].abs()), "_abs")

    momentum_up = merged[
        [
            "raw_ret_24h_rank01",
            "raw_ret_72h_rank01",
            "resid_ret_24h_rank01",
            "resid_ret_72h_rank01",
        ]
    ].mean(axis=1)
    residual_overheat = merged["resid_z_rank01"]
    long_pressure = merged[
        [
            "deriv_funding_rate_z_168_rank01",
            "deriv_premium_index_z_168_rank01",
            "deriv_mark_index_basis_z_168_rank01",
            "deriv_taker_pressure_24_rank01",
            "deriv_top_sum_long_short_ratio_z_168_rank01",
            "deriv_top_count_long_short_ratio_z_168_rank01",
        ]
    ].mean(axis=1)
    abs_crowding = merged[
        [
            "abs_deriv_funding_rate_z_168_rank01",
            "abs_deriv_premium_index_z_168_rank01",
            "abs_deriv_mark_index_basis_z_168_rank01",
            "abs_deriv_taker_pressure_24_rank01",
            "abs_deriv_top_sum_long_short_ratio_z_168_rank01",
            "abs_deriv_top_count_long_short_ratio_z_168_rank01",
        ]
    ].mean(axis=1)
    oi_pressure = merged["deriv_oi_growth_72_336_rank01"]
    merged["squeeze_short_risk"] = (
        0.35 * momentum_up
        + 0.20 * residual_overheat
        + 0.20 * long_pressure
        + 0.15 * abs_crowding
        + 0.10 * oi_pressure
    ).clip(0.0, 1.0)
    merged["squeeze_momentum_up"] = momentum_up
    merged["squeeze_long_pressure"] = long_pressure
    merged["squeeze_abs_crowding"] = abs_crowding
    merged["squeeze_oi_pressure"] = oi_pressure
    keep = [
        "timestamp",
        "symbol",
        "squeeze_short_risk",
        "squeeze_momentum_up",
        "squeeze_long_pressure",
        "squeeze_abs_crowding",
        "squeeze_oi_pressure",
        "raw_ret_24h",
        "raw_ret_72h",
        "resid_ret_24h",
        "resid_ret_72h",
        "resid_z",
        "beta_btc",
        "beta_eth",
        "beta_market",
    ]
    return merged[keep].sort_values(["timestamp", "symbol"])


def cs_rank01(frame: pd.DataFrame, col: str) -> pd.Series:
    ranks = frame.groupby("timestamp")[col].rank(pct=True, method="average")
    return ranks.fillna(0.5).clip(0.0, 1.0)


def build_short_guard_specs() -> list[ShortGuardSpec]:
    return [
        ShortGuardSpec(
            name="p10b_soft_raw",
            description="Only scale vulnerable shorts in risk regimes; no hedge projection.",
            moderate_threshold=0.72,
            high_threshold=0.84,
            moderate_multiplier=0.75,
            high_multiplier=0.55,
        ),
        ShortGuardSpec(
            name="p10b_medium_raw",
            description="Medium vulnerable-short cut in risk regimes; no hedge projection.",
            moderate_threshold=0.68,
            high_threshold=0.80,
            moderate_multiplier=0.65,
            high_multiplier=0.40,
        ),
        ShortGuardSpec(
            name="p10b_medium_confirm_raw",
            description="Medium short cut, but protect high-conviction P9 shorts.",
            moderate_threshold=0.68,
            high_threshold=0.80,
            moderate_multiplier=0.65,
            high_multiplier=0.40,
            protect_strong_short=True,
        ),
        ShortGuardSpec(
            name="p10b_crowded_delev_confirm_raw",
            description="Medium confirmed short cut only in crowded trend and deleveraging/squeeze regimes.",
            moderate_threshold=0.68,
            high_threshold=0.80,
            moderate_multiplier=0.65,
            high_multiplier=0.40,
            risk_regimes=("crowded_trend", "deleveraging_squeeze"),
            protect_strong_short=True,
        ),
        ShortGuardSpec(
            name="p10b_crowded_only_confirm_raw",
            description="Medium confirmed short cut only in crowded trend regimes.",
            moderate_threshold=0.68,
            high_threshold=0.80,
            moderate_multiplier=0.65,
            high_multiplier=0.40,
            risk_regimes=("crowded_trend",),
            protect_strong_short=True,
        ),
        ShortGuardSpec(
            name="p10b_delev_only_confirm_raw",
            description="Medium confirmed short cut only in deleveraging/squeeze regimes.",
            moderate_threshold=0.68,
            high_threshold=0.80,
            moderate_multiplier=0.65,
            high_multiplier=0.40,
            risk_regimes=("deleveraging_squeeze",),
            protect_strong_short=True,
        ),
        ShortGuardSpec(
            name="p10b_medium_anyregime_raw",
            description="Medium short cut when symbol risk is very high even outside risk regimes.",
            moderate_threshold=0.70,
            high_threshold=0.84,
            moderate_multiplier=0.70,
            high_multiplier=0.45,
            require_risk_regime=False,
            protect_strong_short=True,
        ),
        ShortGuardSpec(
            name="p10b_soft_projected",
            description="Soft vulnerable-short cut, then project back to strict market/beta neutral portfolio.",
            moderate_threshold=0.72,
            high_threshold=0.84,
            moderate_multiplier=0.75,
            high_multiplier=0.55,
            hedged_projection=True,
        ),
        ShortGuardSpec(
            name="p10b_medium_confirm_projected",
            description="Medium confirmed short cut, then strict market/beta neutral projection.",
            moderate_threshold=0.68,
            high_threshold=0.80,
            moderate_multiplier=0.65,
            high_multiplier=0.40,
            protect_strong_short=True,
            hedged_projection=True,
        ),
        ShortGuardSpec(
            name="p10b_crowded_delev_confirm_projected",
            description="Crowded/deleveraging confirmed short cut, then strict market/beta neutral projection.",
            moderate_threshold=0.68,
            high_threshold=0.80,
            moderate_multiplier=0.65,
            high_multiplier=0.40,
            risk_regimes=("crowded_trend", "deleveraging_squeeze"),
            protect_strong_short=True,
            hedged_projection=True,
        ),
    ]


def apply_short_guard_overlay(
    base_result: dict[str, pd.DataFrame],
    regime: pd.DataFrame,
    risk_panel: pd.DataFrame,
    beta_panel: pd.DataFrame,
    spec: ShortGuardSpec,
    horizon: int,
) -> dict[str, Any]:
    base_equity = base_result["equity"].sort_values("timestamp").copy()
    base_positions = base_result["positions"].sort_values(["timestamp", "symbol"]).copy()
    symbols = sorted(beta_panel["symbol"].unique())
    regime_by_ts = regime.set_index("timestamp")
    risk_by_ts_symbol = risk_panel.set_index(["timestamp", "symbol"])
    beta_by_ts = {ts: group.set_index("symbol") for ts, group in beta_panel.groupby("timestamp", sort=False)}
    base_positions_by_ts = {ts: group.copy() for ts, group in base_positions.groupby("timestamp", sort=False)}

    rows_out: list[dict[str, Any]] = []
    pos_out: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    prev_weights = pd.Series(0.0, index=symbols, dtype=float)
    equity = 1.0

    for base_row in base_equity.itertuples(index=False):
        ts = pd.Timestamp(base_row.timestamp)
        base_pos = base_positions_by_ts.get(ts)
        if base_pos is None or base_pos.empty:
            continue
        regime_row = regime_by_ts.loc[ts] if ts in regime_by_ts.index else pd.Series(dtype=object)
        regime_name = str(regime_row.get("regime", "normal"))
        base_weights = base_pos.set_index("symbol")["weight"].astype(float)
        active = base_pos.set_index("symbol")
        risk_rows = risk_by_ts_symbol.reindex(pd.MultiIndex.from_product([[ts], active.index])).reset_index(level=0, drop=True)
        multipliers = pd.Series(1.0, index=active.index, dtype=float)
        guard_active = (regime_name in set(spec.risk_regimes)) if spec.require_risk_regime else True

        for symbol, weight in base_weights.items():
            if weight >= 0:
                continue
            risk = float(risk_rows.loc[symbol, "squeeze_short_risk"]) if symbol in risk_rows.index else 0.5
            if not np.isfinite(risk):
                risk = 0.5
            local_active = guard_active
            if not local_active:
                continue
            multiplier = 1.0
            if risk >= spec.high_threshold:
                multiplier = spec.high_multiplier
            elif risk >= spec.moderate_threshold:
                multiplier = spec.moderate_multiplier
            if spec.protect_strong_short and float(active.loc[symbol].get("score", 0.0)) <= spec.strong_short_score:
                multiplier = max(multiplier, spec.strong_short_min_multiplier)
            multipliers.loc[symbol] = multiplier

        guarded_target = base_weights * multipliers
        optimizer_info = {
            "optimizer_success": np.nan,
            "optimizer_target_gross": np.nan,
            "optimizer_actual_gross": np.nan,
            "optimizer_exposure_error": np.nan,
            "optimizer_attempts": np.nan,
        }
        if spec.hedged_projection:
            universe = beta_by_ts.get(ts, pd.DataFrame()).copy()
            target = pd.Series(0.0, index=universe.index, dtype=float)
            target.loc[guarded_target.index] = guarded_target
            fixed_signs = {symbol: int(np.sign(weight)) for symbol, weight in guarded_target.items() if abs(weight) > 1e-12}
            gross_limit = float(guarded_target.abs().sum())
            projected, optimizer_info = constrained_project_weights(
                target_weights=target,
                universe=universe.reset_index(),
                exposure_cols=NEUTRALIZE_COLS,
                max_abs_weight=0.125,
                gross_limit=gross_limit,
                fixed_signs=fixed_signs,
                hedge_penalty=50.0,
                return_info=True,
            )
            if float(projected.abs().sum()) > 1e-12:
                guarded_target = projected[projected.abs() > 1e-12]

        full_weights = pd.Series(0.0, index=symbols, dtype=float)
        full_weights.loc[guarded_target.index] = guarded_target
        turnover = float((full_weights - prev_weights).abs().sum())
        cost = turnover * 6.0 / 10_000.0

        active_all = active.reindex(guarded_target.index)
        gross_period_ret = float((guarded_target * active_all["fwd_simple_ret"].astype(float)).sum())
        period_ret = gross_period_ret - cost
        equity *= 1.0 + period_ret
        base_gross = float(base_row.gross_period_return)
        gated_away = base_gross - gross_period_ret

        betas = beta_by_ts.get(ts, pd.DataFrame()).reindex(guarded_target.index)
        exposures = portfolio_exposures(guarded_target, betas, NEUTRALIZE_COLS)
        long_labels = active_all.loc[guarded_target[guarded_target > 0].index, "label_resid_ret"] if (guarded_target > 0).any() else pd.Series(dtype=float)
        short_labels = active_all.loc[guarded_target[guarded_target < 0].index, "label_resid_ret"] if (guarded_target < 0).any() else pd.Series(dtype=float)
        top_bottom = float(long_labels.mean() - short_labels.mean()) if len(long_labels) and len(short_labels) else np.nan
        removed_short_abs = float(((base_weights - guarded_target.reindex(base_weights.index).fillna(0.0)).where(base_weights < 0, 0.0)).abs().sum())

        row = {
            **base_row._asdict(),
            "period_return": period_ret,
            "gross_period_return": gross_period_ret,
            "cost_return": cost,
            "equity": equity,
            "turnover": turnover,
            "target_turnover": turnover,
            "gross_exposure": float(guarded_target.abs().sum()),
            "target_gross_exposure": float(guarded_target.abs().sum()),
            "top_bottom_label_spread": top_bottom,
            "abs_beta_btc": abs(float(exposures.get("beta_btc", 0.0))),
            "abs_beta_eth": abs(float(exposures.get("beta_eth", 0.0))),
            "abs_beta_market": abs(float(exposures.get("beta_market", 0.0))),
            "symbol_concentration": float((guarded_target**2).sum()),
            "max_abs_weight": float(guarded_target.abs().max()) if len(guarded_target) else 0.0,
            "regime": regime_name,
            "base_gross_period_return": base_gross,
            "gated_away_gross_return": gated_away,
            "gated_away_abs_weight": float((base_weights.reindex(symbols).fillna(0.0) - full_weights).abs().sum()),
            "short_guard_removed_abs_weight": removed_short_abs,
            "short_guarded_count": float((multipliers[base_weights.index] < 1.0).sum()),
            **optimizer_info,
        }
        rows_out.append(row)

        for symbol, weight in guarded_target.items():
            original = active.reindex([symbol]).iloc[0] if symbol in active.index else pd.Series(dtype=object)
            base_weight = float(base_weights.get(symbol, 0.0))
            fwd_ret = float(original.get("fwd_simple_ret", 0.0))
            multiplier = float(multipliers.get(symbol, np.nan))
            risk_value = float(risk_rows["squeeze_short_risk"].get(symbol, np.nan)) if symbol in risk_rows.index else np.nan
            base_contribution = base_weight * fwd_ret
            contribution = float(weight * fwd_ret)
            pos_out.append(
                {
                    **original.to_dict(),
                    "timestamp": ts,
                    "symbol": symbol,
                    "weight": float(weight),
                    "target_weight": float(weight),
                    "base_weight": base_weight,
                    "contribution": contribution,
                    "base_contribution": base_contribution,
                    "guard_benefit": contribution - base_contribution,
                    "short_guard_multiplier": multiplier,
                    "squeeze_short_risk": risk_value,
                    "regime": regime_name,
                    "overlay_variant": spec.name,
                }
            )
        prev_weights = full_weights

    equity_curve = pd.DataFrame(rows_out)
    positions = pd.DataFrame(pos_out)
    for fold, group in equity_curve.groupby("fold", sort=True):
        pos_group = positions[positions["fold"] == fold]
        metrics = p10.compute_metrics(group, pos_group, horizon)
        base_fold = base_result["folds"][base_result["folds"]["fold"] == fold].iloc[0].to_dict()
        fold_rows.append({**base_fold, **metrics})
    folds = pd.DataFrame(fold_rows)
    metrics = p3d.summarize_nested_result({"equity": equity_curve, "positions": positions, "folds": folds}, horizon)
    return {"folds": folds, "equity": equity_curve, "positions": positions, "metrics": metrics}


def build_guard_symbol_attribution(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for name, result in results.items():
        if not name.startswith("p10b_"):
            continue
        positions = result["positions"].copy()
        if "short_guard_multiplier" not in positions:
            continue
        guarded = positions[(positions["base_weight"] < 0) & (positions["short_guard_multiplier"] < 1.0)].copy()
        if guarded.empty:
            continue
        grouped = (
            guarded.groupby("symbol")
            .agg(
                guarded_count=("symbol", "count"),
                mean_squeeze_short_risk=("squeeze_short_risk", "mean"),
                base_contribution_sum=("base_contribution", "sum"),
                contribution_sum=("contribution", "sum"),
                guard_benefit_sum=("guard_benefit", "sum"),
                mean_removed_abs_weight=("base_weight", lambda x: float(x.abs().mean())),
            )
            .reset_index()
        )
        grouped["variant"] = name
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_guard_regime_attribution(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for name, result in results.items():
        if not name.startswith("p10b_"):
            continue
        positions = result["positions"].copy()
        if "short_guard_multiplier" not in positions:
            continue
        guarded = positions[(positions["base_weight"] < 0) & (positions["short_guard_multiplier"] < 1.0)].copy()
        if guarded.empty:
            continue
        grouped = (
            guarded.groupby("regime")
            .agg(
                guarded_count=("symbol", "count"),
                base_contribution_sum=("base_contribution", "sum"),
                contribution_sum=("contribution", "sum"),
                guard_benefit_sum=("guard_benefit", "sum"),
                mean_squeeze_short_risk=("squeeze_short_risk", "mean"),
            )
            .reset_index()
        )
        grouped["variant"] = name
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def select_best_p10b(summary: pd.DataFrame, cost: pd.DataFrame) -> str:
    candidates = summary[summary["variant"].str.startswith("p10b_")].copy()
    if candidates.empty:
        return "p9_reference"
    p9_row = summary[summary["variant"] == "p9_reference"].iloc[0]
    c20 = cost[cost["one_way_cost_bps"] == 20.0][["variant", "total_return"]].rename(columns={"total_return": "return_20bps"})
    c30 = cost[cost["one_way_cost_bps"] == 30.0][["variant", "total_return"]].rename(columns={"total_return": "return_30bps"})
    p9_20 = float(c20[c20["variant"] == "p9_reference"]["return_20bps"].iloc[0])
    p9_30 = float(c30[c30["variant"] == "p9_reference"]["return_30bps"].iloc[0])
    candidates = candidates.merge(c20, on="variant", how="left").merge(c30, on="variant", how="left")
    candidates["accept_score"] = (
        120.0 * candidates["delta_vs_p9_total_return"]
        + 2.0 * candidates["delta_vs_p9_sharpe"]
        + 60.0 * candidates["delta_vs_p9_max_drawdown"]
        + 25.0 * (candidates["return_20bps"] - p9_20)
        + 15.0 * (candidates["return_30bps"] - p9_30)
        + 8.0 * (candidates["worst_5_fold_return_sum"] - float(p9_row["worst_5_fold_return_sum"]))
    )
    return str(candidates.sort_values(["accept_score", "total_return"], ascending=False).iloc[0]["variant"])


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


def plot_net_curves(output_dir: Path, results: dict[str, dict[str, Any]], best_name: str) -> None:
    keep = ["p3d_reference", "p7b_reference", "p9_reference", best_name]
    plt.figure(figsize=(13, 7))
    for name in keep:
        equity = results[name]["equity"].sort_values("timestamp")
        net = (1.0 + equity["period_return"]).cumprod() - 1.0
        plt.plot(equity["timestamp"], net, label=name, linewidth=2.1 if name == best_name else 1.4)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.title("P10b asymmetric squeeze-aware short guard net curves")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "p10b_net_curves.png", dpi=180)
    plt.close()


def plot_cost_curves(output_dir: Path, cost: pd.DataFrame, best_name: str) -> None:
    keep = ["p9_reference", best_name, "p10b_soft_raw", "p10b_medium_confirm_raw", "p10b_medium_confirm_projected"]
    frame = cost[cost["variant"].isin(keep)].copy()
    plt.figure(figsize=(12, 7))
    for name, group in frame.groupby("variant"):
        group = group.sort_values("one_way_cost_bps")
        plt.plot(group["one_way_cost_bps"], group["total_return"], marker="o", label=name, linewidth=2.2 if name == best_name else 1.4)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.title("P10b cost sensitivity")
    plt.xlabel("one-way cost bps")
    plt.ylabel("total return")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "cost_sensitivity.png", dpi=180)
    plt.close()


def plot_variant_bars(output_dir: Path, summary: pd.DataFrame) -> None:
    frame = summary.sort_values("total_return")
    plt.figure(figsize=(12, 7))
    colors = np.where(frame["variant"].str.startswith("p10b_"), "#2a9d8f", "#777777")
    plt.barh(frame["variant"], frame["total_return"], color=colors)
    plt.axvline(0.0, color="black", linewidth=0.8)
    plt.title("P10b net return at 6bps")
    plt.gca().xaxis.set_major_formatter(PercentFormatter(1.0))
    plt.tight_layout()
    plt.savefig(output_dir / "variant_net_return.png", dpi=180)
    plt.close()


def plot_guard_benefit(output_dir: Path, guard_symbol: pd.DataFrame, best_name: str) -> None:
    frame = guard_symbol[guard_symbol["variant"] == best_name].sort_values("guard_benefit_sum")
    if frame.empty:
        return
    colors = np.where(frame["guard_benefit_sum"] >= 0.0, "#2a9d8f", "#c44536")
    plt.figure(figsize=(10, 6))
    plt.barh(frame["symbol"], frame["guard_benefit_sum"], color=colors)
    plt.axvline(0.0, color="black", linewidth=0.8)
    plt.title(f"{best_name} guarded short benefit by symbol")
    plt.gca().xaxis.set_major_formatter(PercentFormatter(1.0))
    plt.tight_layout()
    plt.savefig(output_dir / "guard_symbol_benefit.png", dpi=180)
    plt.close()


def plot_fold_delta(output_dir: Path, fold_delta: pd.DataFrame, best_name: str) -> None:
    frame = fold_delta[fold_delta["variant"] == best_name].sort_values("fold")
    if frame.empty:
        return
    colors = np.where(frame["delta_total_return"] >= 0.0, "#2a9d8f", "#c44536")
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
    specs: list[ShortGuardSpec],
    best_name: str,
    summary: pd.DataFrame,
    cost_summary: pd.DataFrame,
    fold_delta: pd.DataFrame,
    regime_attr: pd.DataFrame,
    filter_efficacy: pd.DataFrame,
    stress_attr: pd.DataFrame,
    guard_symbol: pd.DataFrame,
    guard_regime: pd.DataFrame,
    bootstrap: pd.DataFrame,
    start_vs_p9: pd.DataFrame,
    dropout_vs_p9: pd.DataFrame,
    reality: pd.DataFrame,
) -> None:
    summary_cols = [
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
    summary_cols = [col for col in summary_cols if col in summary]
    cost_display = cost_summary[cost_summary["one_way_cost_bps"].isin([0, 6, 10, 15, 20, 30])]
    cost_display = cost_display[cost_display["variant"].isin(["p9_reference", best_name, "p10b_soft_raw", "p10b_medium_confirm_raw", "p10b_medium_confirm_projected"])]
    best_filter = filter_efficacy[filter_efficacy["variant"] == best_name] if not filter_efficacy.empty else filter_efficacy
    best_guard_symbol = guard_symbol[guard_symbol["variant"] == best_name] if not guard_symbol.empty else guard_symbol
    best_guard_regime = guard_regime[guard_regime["variant"] == best_name] if not guard_regime.empty else guard_regime
    lines = [
        "# P10b Asymmetric Squeeze-Aware Short Guard",
        "",
        "## Design",
        "",
        "- Fixed P9 selected formulas and positions; this is a portfolio overlay audit, not an alpha refit.",
        "- The guard only scales short positions with high ex-ante squeeze risk.",
        "- Squeeze risk uses only timestamp-t known OHLCV residual momentum plus Binance derivatives crowding, OI, premium, funding, and taker/top-trader pressure.",
        "- Raw variants expose the guard effect directly; projected variants additionally project back to strict intercept/BTC/ETH/market neutrality.",
        "",
        "## Guard Specs",
        "```text",
        pd.DataFrame([spec.__dict__ for spec in specs]).to_string(index=False),
        "```",
        "",
        f"## Selected Best Variant: `{best_name}`",
        "",
        "## Summary",
        "```text",
        summary[summary_cols].to_string(index=False),
        "```",
        "",
        "## Cost Sensitivity",
        "```text",
        cost_display.to_string(index=False),
        "```",
        "",
        "## Filter Efficacy",
        "Negative gated-away gross means the guard removed losses.",
        "```text",
        best_filter.to_string(index=False),
        "```",
        "",
        "## Guarded Short Attribution By Regime",
        "Positive guard_benefit means scaling the short improved gross PnL.",
        "```text",
        best_guard_regime.to_string(index=False),
        "```",
        "",
        "## Guarded Short Attribution By Symbol",
        "```text",
        best_guard_symbol.to_string(index=False),
        "```",
        "",
        "## Regime Attribution",
        "```text",
        regime_attr.to_string(index=False),
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
    stages["p10b_asymmetric_squeeze_short_guard"] = {
        "script": "../../../scripts/run_p10b_asymmetric_squeeze_short_guard.py",
        "output_dir": "../outputs/p10b_asymmetric_squeeze_short_guard",
        "report": "../outputs/p10b_asymmetric_squeeze_short_guard/REPORT.md",
        "variant_summary": "../outputs/p10b_asymmetric_squeeze_short_guard/variant_summary.csv",
        "cost_sensitivity": "../outputs/p10b_asymmetric_squeeze_short_guard/cost_sensitivity.csv",
        "short_squeeze_risk_panel": "../outputs/p10b_asymmetric_squeeze_short_guard/short_squeeze_risk_panel.csv",
        "reality_summary": "../outputs/p10b_asymmetric_squeeze_short_guard/reality_summary.csv",
        "best_variant": best_name,
    }
    cfg["active_research_short_guard_candidate"] = {
        "name": best_name,
        "status": "candidate_pending_review",
        "output_dir": "../outputs/p10b_asymmetric_squeeze_short_guard",
        "report": "../outputs/p10b_asymmetric_squeeze_short_guard/REPORT.md",
        "parent": "p9_clean_drop_k01_cap010",
    }
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False), encoding="utf-8")


if __name__ == "__main__":
    main()
