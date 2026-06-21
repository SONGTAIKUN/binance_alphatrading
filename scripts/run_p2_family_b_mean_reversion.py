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
from crypto_stat_arb.portfolio import (
    PortfolioConfig,
    portfolio_exposures,
    select_target_sides,
    sides_to_weights,
    spearman_ic,
    update_state,
)
from crypto_stat_arb.walk_forward import WalkForwardConfig
from run_delay_smoothing_experiments import DelayConfig, add_delay_metrics
from run_p0_baseline_lock_review import (
    build_asset_summary,
    build_cost_sensitivity,
    build_regime_frame,
    build_regime_summary,
    build_side_summary,
)
from run_residual_alphagpt_search import (
    aggregate_oos_metrics,
    build_candidate_scores,
    prepare_alpha_panel,
)
from run_risk_control_experiments import (
    RiskControlConfig,
    add_gross_metrics,
    add_risk_event_metrics,
    apply_short_residual_confirmation,
    build_weights,
)
from run_turnover_experiments import to_jsonable


EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "baseline_locked_v1"
DEFAULT_CONFIG = EXPERIMENT_ROOT / "configs" / "baseline_locked_v1.yaml"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "outputs" / "p2_family_b_mean_reversion"
SLOW_LABELS = EXPERIMENT_ROOT / "outputs" / "p1_slow_labels" / "slow_residual_labels.csv.gz"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--horizons", type=int, nargs="+", default=[48, 96])
    parser.add_argument("--cost-bps", type=float, nargs="+", default=[0, 4, 6, 8, 10, 12, 15, 20, 30])
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    output_dir = ensure_dir(args.output_dir)
    horizons = sorted(set(int(h) for h in args.horizons))
    cost_bps = [float(x) for x in args.cost_bps]

    panel_path = resolve_config_path(config_path, cfg["inputs"]["panel"])
    selections_path = resolve_config_path(config_path, cfg["inputs"]["formula_selections"])
    if not SLOW_LABELS.exists():
        raise FileNotFoundError(f"P1 slow label file not found: {SLOW_LABELS}")

    print("[1/7] Loading panel, P1 labels, and folds", flush=True)
    panel = pd.read_csv(panel_path, parse_dates=["timestamp"])
    panel = prepare_alpha_panel(panel, sorted(set(horizons + [48, 72, 96, 168])), execution_delay_bars=1)
    build_candidate_scores(panel)
    labels = pd.read_csv(
        SLOW_LABELS,
        parse_dates=["timestamp"],
        usecols=[
            "timestamp",
            "symbol",
            "label_slow_medium",
            "label_slow_low_turnover",
            "label_path_slow_medium",
            "label_path_slow_low_turnover",
            "label_horizon_sign_consistency",
            "label_path_horizon_sign_consistency",
        ],
    )
    panel = panel.merge(labels, on=["timestamp", "symbol"], how="left")
    selections = pd.read_csv(selections_path, parse_dates=["train_start", "train_end", "test_start", "test_end"])

    print("[2/7] Building Family B residual anchor / stationarity features", flush=True)
    candidates = build_family_b_features(panel)
    pd.DataFrame(candidates).to_csv(output_dir / "candidate_formulas.csv", index=False)
    feature_cols = [
        "timestamp",
        "symbol",
        "family_b_regime_gate",
        *sorted({feature for item in candidates for feature in item["features"].split("|") if feature}),
        *[item["score_col"] for item in candidates],
    ]
    feature_cols = [col for col in feature_cols if col in panel.columns]
    panel[feature_cols].to_csv(output_dir / "family_b_feature_panel.csv.gz", index=False)

    print("[3/7] Running strict-neutral walk-forward backtests with regime gross gate", flush=True)
    wf_cfg = WalkForwardConfig(train_days=180, test_days=30, step_days=30, mode="rolling")
    risk_cfg = make_locked_risk_config(cfg)
    summary_rows: list[dict[str, Any]] = []
    result_index: dict[str, dict[str, Any]] = {}
    runs_dir = ensure_dir(output_dir / "runs")
    for horizon in horizons:
        pf_cfg = make_portfolio_config(cfg, horizon)
        for candidate in candidates:
            key = f"h{horizon:03d}_{candidate['name']}"
            print(f"      {key}", flush=True)
            result = run_candidate_replay(
                panel=panel,
                selections=selections,
                pf_cfg=pf_cfg,
                wf_cfg=wf_cfg,
                risk_cfg=risk_cfg,
                score_col=candidate["score_col"],
                formula=candidate["formula"],
                use_regime_gate=True,
            )
            result_index[key] = result
            run_dir = ensure_dir(runs_dir / key)
            write_run_result(run_dir, result)
            summary_rows.append({"run": key, "horizon": horizon, **candidate, **result["metrics"]})

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "candidate_summary.csv", index=False)
    summary.sort_values(["total_return", "sharpe"], ascending=[False, False]).to_csv(
        output_dir / "candidate_summary_by_net.csv", index=False
    )

    print("[4/7] Computing OOS IC against P1 slow labels", flush=True)
    ic_summary = build_ic_summary(panel, selections, candidates)
    ic_summary.to_csv(output_dir / "ic_summary.csv", index=False)

    print("[5/7] Building best-run diagnostics", flush=True)
    baseline = load_baseline_summary(EXPERIMENT_ROOT / "baseline_lock_review" / "horizon_summary.csv")
    comparison = build_baseline_comparison(summary, baseline)
    comparison.to_csv(output_dir / "baseline_comparison.csv", index=False)
    best_key = str(summary.sort_values(["total_return", "sharpe"], ascending=[False, False]).iloc[0]["run"])
    best_result = result_index[best_key]
    best_run_dir = ensure_dir(output_dir / "best_run")
    write_run_result(best_run_dir, best_result)
    best_horizon = int(summary.loc[summary["run"] == best_key, "horizon"].iloc[0])
    best_equity = best_result["equity"].sort_values("timestamp")
    best_positions = best_result["positions"].sort_values(["timestamp", "symbol"])
    regime_summary = build_regime_summary(best_equity, build_regime_frame(panel), best_horizon)
    side_summary = build_side_summary(best_positions)
    asset_summary = build_asset_summary(best_positions)
    regime_summary.to_csv(output_dir / "best_regime_summary.csv", index=False)
    side_summary.to_csv(output_dir / "best_side_contribution.csv", index=False)
    asset_summary.to_csv(output_dir / "best_asset_contribution.csv", index=False)

    top_keys = list(summary.sort_values(["total_return", "sharpe"], ascending=[False, False])["run"].head(5))
    cost_summary = build_cost_sensitivity({key: result_index[key] for key in top_keys}, cost_bps)
    cost_summary.to_csv(output_dir / "top5_cost_sensitivity.csv", index=False)
    corr_combo = build_best_vs_reference_combo(best_result, best_key)
    with (output_dir / "best_vs_reference_corr_combo.json").open("w", encoding="utf-8") as fh:
        json.dump(to_jsonable(corr_combo), fh, indent=2, ensure_ascii=False)

    print("[6/7] Writing plots", flush=True)
    plot_summary(output_dir, summary)
    plot_best_curves(output_dir, best_equity, best_key)
    plot_top_curves(output_dir, result_index, top_keys)
    plot_ic_summary(output_dir, ic_summary)
    plot_cost_sensitivity(output_dir, cost_summary)
    plot_asset_contribution(output_dir, asset_summary)

    print("[7/7] Writing report", flush=True)
    write_report(
        output_dir=output_dir,
        summary=summary,
        ic_summary=ic_summary,
        comparison=comparison,
        best_key=best_key,
        regime_summary=regime_summary,
        side_summary=side_summary,
        asset_summary=asset_summary,
        cost_summary=cost_summary,
        corr_combo=corr_combo,
    )
    update_config_stage(config_path)
    print(f"Done: {output_dir}", flush=True)


def resolve_config_path(config_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


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
        min_hold_rebalances=max(1, math.ceil(float(locked["min_hold_days"]) * 24 / horizon)),
    )


def make_locked_risk_config(cfg: dict[str, Any]) -> RiskControlConfig:
    risk = cfg["locked_baseline"]["risk_control"]
    return RiskControlConfig(
        name=str(cfg["locked_baseline"]["name"]),
        description="Locked P0 strict-neutral allocator.",
        max_symbol_weight=float(risk["max_symbol_weight"]),
        short_resid_confirm_rank=float(risk["short_resid_confirm_rank"]),
        allocation_method=str(risk["allocation_method"]),
        optimizer_target_gross_mode=str(risk["optimizer_target_gross_mode"]),
        hedge_penalty=float(risk["hedge_penalty"]),
    )


def build_family_b_features(panel: pd.DataFrame) -> list[dict[str, str]]:
    panel.sort_values(["symbol", "timestamp"], inplace=True)
    grouped = panel.groupby("symbol", group_keys=False)
    panel["resid_cum"] = grouped["residual_log_ret"].cumsum()

    for span in [168, 336]:
        panel[f"res_ewm_{span}"] = grouped["resid_cum"].transform(
            lambda s, sp=span: s.ewm(span=sp, adjust=False, min_periods=max(48, sp // 3)).mean()
        )
        dev = panel["resid_cum"] - panel[f"res_ewm_{span}"]
        panel[f"res_dev_{span}"] = dev
        panel[f"res_z_{span}"] = grouped[f"res_dev_{span}"].transform(
            lambda s, w=span: rolling_zscore(s, w, min_periods=max(48, w // 3))
        )
        panel[f"ou_speed_{span}"] = grouped["resid_cum"].transform(lambda s, w=span: rolling_ou_speed(s, w))
        panel[f"ou_halflife_{span}"] = np.log(2.0) / panel[f"ou_speed_{span}"].replace(0.0, np.nan)
        panel[f"stationarity_gate_{span}"] = half_life_gate(panel[f"ou_halflife_{span}"])
        panel[f"beta_instability_{span}"] = beta_instability(panel, span)
        panel[f"beta_stability_{span}"] = -panel[f"beta_instability_{span}"]

    res_rv_24 = grouped["residual_log_ret"].transform(lambda s: np.sqrt(s.pow(2).rolling(24, min_periods=12).sum()))
    panel["vol_of_vol_res_168"] = (
        res_rv_24.groupby(panel["symbol"]).transform(lambda s: s.rolling(168, min_periods=72).std(ddof=0))
    )
    panel["quote_volume_mean_7d"] = grouped["quote_volume"].transform(lambda s: s.rolling(168, min_periods=72).mean())
    panel["amihud_7d"] = grouped.apply(
        lambda df: (df["log_ret"].abs() / df["quote_volume"].replace(0.0, np.nan)).rolling(168, min_periods=72).mean()
    ).reset_index(level=0, drop=True)

    panel["cs_liq_volume"] = centered_rank_by_timestamp(panel, "quote_volume_mean_7d")
    panel["cs_amihud"] = centered_rank_by_timestamp(panel, "amihud_7d")
    panel["cs_vol_of_vol_res_168"] = centered_rank_by_timestamp(panel, "vol_of_vol_res_168")
    panel["liquidity_quality"] = panel["cs_liq_volume"] - panel["cs_amihud"] - panel["cs_vol_of_vol_res_168"]

    for feature in [
        "res_z_168",
        "res_z_336",
        "ou_speed_168",
        "ou_speed_336",
        "beta_stability_168",
        "beta_stability_336",
        "liquidity_quality",
    ]:
        panel[f"cs_{feature}"] = centered_rank_by_timestamp(panel, feature)
        panel[f"pct_{feature}"] = pct_rank_by_timestamp(panel, feature)

    add_regime_gate(panel)

    panel["alpha_B_revert_168"] = -panel["cs_res_z_168"]
    panel["alpha_B_revert_336"] = -panel["cs_res_z_336"]
    panel["alpha_B_stationary_168"] = -panel["cs_res_z_168"] * panel["stationarity_gate_168"]
    panel["alpha_B_stationary_336"] = -panel["cs_res_z_336"] * panel["stationarity_gate_336"]
    panel["alpha_B_ou_168"] = (
        -panel["cs_res_z_168"] * panel["stationarity_gate_168"] * panel["pct_ou_speed_168"].fillna(0.0)
    )
    panel["alpha_B_ou_336"] = (
        -panel["cs_res_z_336"] * panel["stationarity_gate_336"] * panel["pct_ou_speed_336"].fillna(0.0)
    )
    panel["alpha_B_beta_168"] = (
        panel["alpha_B_ou_168"] * panel["pct_beta_stability_168"].fillna(0.0)
    )
    panel["alpha_B_beta_336"] = (
        panel["alpha_B_ou_336"] * panel["pct_beta_stability_336"].fillna(0.0)
    )
    panel["alpha_B_seed_168"] = (
        panel["alpha_B_beta_168"] * panel["pct_liquidity_quality"].fillna(0.0)
    )
    panel["alpha_B_seed_336"] = (
        panel["alpha_B_beta_336"] * panel["pct_liquidity_quality"].fillna(0.0)
    )
    panel["alpha_B_seed_combo"] = 0.60 * panel["alpha_B_seed_168"] + 0.40 * panel["alpha_B_seed_336"]
    panel["alpha_B_soft_combo"] = (
        -0.55 * panel["cs_res_z_168"]
        -0.45 * panel["cs_res_z_336"]
        + 0.25 * panel["cs_ou_speed_168"].fillna(0.0)
        + 0.15 * panel["cs_beta_stability_168"].fillna(0.0)
        + 0.10 * panel["cs_liquidity_quality"].fillna(0.0)
    )

    specs: list[tuple[str, str, str, list[str]]] = [
        ("B01_revert_168", "alpha_B_revert_168", "-rank_cs(res_z_168)", ["res_z_168"]),
        ("B02_revert_336", "alpha_B_revert_336", "-rank_cs(res_z_336)", ["res_z_336"]),
        (
            "B03_stationary_168",
            "alpha_B_stationary_168",
            "-rank_cs(res_z_168)*stationarity_gate_168",
            ["res_z_168", "stationarity_gate_168"],
        ),
        (
            "B04_stationary_336",
            "alpha_B_stationary_336",
            "-rank_cs(res_z_336)*stationarity_gate_336",
            ["res_z_336", "stationarity_gate_336"],
        ),
        (
            "B05_ou_168",
            "alpha_B_ou_168",
            "-rank_cs(res_z_168)*stationarity_gate_168*pct_rank(ou_speed_168)",
            ["res_z_168", "stationarity_gate_168", "ou_speed_168"],
        ),
        (
            "B06_ou_336",
            "alpha_B_ou_336",
            "-rank_cs(res_z_336)*stationarity_gate_336*pct_rank(ou_speed_336)",
            ["res_z_336", "stationarity_gate_336", "ou_speed_336"],
        ),
        (
            "B07_beta_168",
            "alpha_B_beta_168",
            "-rank_cs(res_z_168)*stationarity_gate_168*pct_rank(ou_speed_168)*pct_rank(beta_stability_168)",
            ["res_z_168", "stationarity_gate_168", "ou_speed_168", "beta_stability_168"],
        ),
        (
            "B08_beta_336",
            "alpha_B_beta_336",
            "-rank_cs(res_z_336)*stationarity_gate_336*pct_rank(ou_speed_336)*pct_rank(beta_stability_336)",
            ["res_z_336", "stationarity_gate_336", "ou_speed_336", "beta_stability_336"],
        ),
        (
            "B09_seed_168",
            "alpha_B_seed_168",
            "-rank_cs(res_z_168)*stationarity_gate_168*pct_rank(ou_speed_168)*pct_rank(beta_stability_168)*pct_rank(liquidity_quality)",
            ["res_z_168", "stationarity_gate_168", "ou_speed_168", "beta_stability_168", "liquidity_quality"],
        ),
        (
            "B10_seed_336",
            "alpha_B_seed_336",
            "-rank_cs(res_z_336)*stationarity_gate_336*pct_rank(ou_speed_336)*pct_rank(beta_stability_336)*pct_rank(liquidity_quality)",
            ["res_z_336", "stationarity_gate_336", "ou_speed_336", "beta_stability_336", "liquidity_quality"],
        ),
        (
            "B11_seed_combo",
            "alpha_B_seed_combo",
            "0.60*seed_168+0.40*seed_336",
            ["alpha_B_seed_168", "alpha_B_seed_336"],
        ),
        (
            "B12_soft_combo",
            "alpha_B_soft_combo",
            "-0.55*rank_cs(res_z_168)-0.45*rank_cs(res_z_336)+0.25*rank_cs(ou_speed_168)+0.15*rank_cs(beta_stability_168)+0.10*rank_cs(liquidity_quality)",
            ["res_z_168", "res_z_336", "ou_speed_168", "beta_stability_168", "liquidity_quality"],
        ),
    ]

    panel.sort_values(["timestamp", "symbol"], inplace=True)
    panel.reset_index(drop=True, inplace=True)
    return [
        {"name": name, "score_col": score_col, "formula": formula, "features": "|".join(features)}
        for name, score_col, formula, features in specs
    ]


def rolling_zscore(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std(ddof=0)
    return (series - mean) / std.replace(0.0, np.nan)


def rolling_ou_speed(resid_cum: pd.Series, window: int) -> pd.Series:
    x = resid_cum.shift(1)
    y = resid_cum.diff()
    min_periods = max(48, window // 3)
    cov = x.rolling(window, min_periods=min_periods).cov(y)
    var = x.rolling(window, min_periods=min_periods).var(ddof=0)
    slope = cov / var.replace(0.0, np.nan)
    speed = -slope
    return speed.where(slope < 0.0)


def half_life_gate(halflife: pd.Series) -> pd.Series:
    fast_penalty = halflife / 48.0
    slow_penalty = 240.0 / halflife
    gate = np.minimum(fast_penalty, slow_penalty)
    gate = pd.Series(gate, index=halflife.index).clip(lower=0.0, upper=1.0)
    return gate.where(np.isfinite(halflife), 0.0)


def beta_instability(panel: pd.DataFrame, window: int) -> pd.Series:
    grouped = panel.groupby("symbol", group_keys=False)
    parts = []
    for col in ["beta_btc", "beta_eth", "beta_market", "beta_sector"]:
        parts.append(
            grouped[col].transform(lambda s, w=window: s.rolling(w, min_periods=max(48, w // 3)).std(ddof=0)).fillna(0.0)
            ** 2
        )
    return np.sqrt(sum(parts))


def centered_rank_by_timestamp(frame: pd.DataFrame, value_col: str) -> pd.Series:
    grouped = frame.groupby("timestamp", sort=False)[value_col]
    rank = grouped.rank(method="average")
    count = grouped.transform("count")
    centered = 2.0 * (rank - 1.0) / (count - 1.0) - 1.0
    return centered.where(count > 1)


def pct_rank_by_timestamp(frame: pd.DataFrame, value_col: str) -> pd.Series:
    grouped = frame.groupby("timestamp", sort=False)[value_col]
    rank = grouped.rank(method="average", pct=True)
    return rank.where(grouped.transform("count") > 1)


def add_regime_gate(panel: pd.DataFrame) -> None:
    market = panel.groupby("timestamp")["log_ret"].mean().sort_index()
    residual_dispersion = panel.groupby("timestamp")["residual_log_ret"].std().sort_index()
    market_vol = market.rolling(336, min_periods=72).std(ddof=0)
    residual_dispersion_7d = residual_dispersion.rolling(168, min_periods=48).mean()

    vol_q80 = market_vol.rolling(4320, min_periods=720).quantile(0.80)
    disp_q80 = residual_dispersion_7d.rolling(4320, min_periods=720).quantile(0.80)
    high_vol = market_vol > vol_q80
    high_disp = residual_dispersion_7d > disp_q80
    gate = pd.Series(1.0, index=market.index)
    gate.loc[high_vol | high_disp] = 0.50
    gate.loc[high_vol & high_disp] = 0.35

    panel["market_vol_14d"] = panel["timestamp"].map(market_vol)
    panel["residual_dispersion_7d"] = panel["timestamp"].map(residual_dispersion_7d)
    panel["family_b_regime_gate"] = panel["timestamp"].map(gate).fillna(1.0)


def run_candidate_replay(
    panel: pd.DataFrame,
    selections: pd.DataFrame,
    pf_cfg: PortfolioConfig,
    wf_cfg: WalkForwardConfig,
    risk_cfg: RiskControlConfig,
    score_col: str,
    formula: str,
    use_regime_gate: bool,
) -> dict[str, Any]:
    fold_rows: list[dict[str, Any]] = []
    equity_parts: list[pd.DataFrame] = []
    position_parts: list[pd.DataFrame] = []

    for _, selection in selections.iterrows():
        equity, positions, metrics = run_regime_gated_backtest(
            panel=panel,
            start=selection["test_start"],
            end=selection["test_end"],
            cfg=pf_cfg,
            risk_cfg=risk_cfg,
            delay_cfg=DelayConfig("tau_0", 0.0),
            score_col=score_col,
            selected_score_cols=[score_col],
            selected_formulas=formula,
            use_regime_gate=use_regime_gate,
        )
        fold_id = int(selection["fold"])
        if not equity.empty:
            equity = equity.copy()
            equity["fold"] = fold_id
            equity["formula"] = formula
            positions = positions.copy()
            positions["fold"] = fold_id
            positions["formula"] = formula
            equity_parts.append(equity)
            position_parts.append(positions)
        fold_rows.append({"fold": fold_id, **selection.to_dict(), **metrics})

    fold_summary = pd.DataFrame(fold_rows)
    equity_curve = pd.concat(equity_parts, ignore_index=True) if equity_parts else pd.DataFrame()
    positions = pd.concat(position_parts, ignore_index=True) if position_parts else pd.DataFrame()
    metrics = aggregate_oos_metrics(equity_curve, pf_cfg.rebalance_hours)
    metrics["fold_count"] = int(len(selections))
    if not fold_summary.empty and "total_return" in fold_summary:
        metrics["positive_fold_rate"] = float((fold_summary["total_return"] > 0).mean())
    add_gross_metrics(equity_curve, metrics)
    add_risk_event_metrics(positions, metrics)
    add_delay_metrics(equity_curve, metrics)
    if not equity_curve.empty and "regime_gross_scale" in equity_curve:
        metrics["mean_regime_gross_scale"] = float(equity_curve["regime_gross_scale"].mean())
        metrics["low_regime_gross_scale_rate"] = float((equity_curve["regime_gross_scale"] < 0.999).mean())
    return {"folds": fold_summary, "equity": equity_curve, "positions": positions, "metrics": metrics}


def run_regime_gated_backtest(
    panel: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cfg: PortfolioConfig,
    risk_cfg: RiskControlConfig,
    delay_cfg: DelayConfig,
    score_col: str,
    selected_score_cols: list[str],
    selected_formulas: str,
    use_regime_gate: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    label_col = f"label_resid_ret_h{cfg.primary_horizon}"
    ret_col = f"fwd_simple_ret_h{cfg.primary_horizon}"
    symbols = sorted(panel["symbol"].unique())
    mask = (panel["timestamp"] >= start) & (panel["timestamp"] < end)
    timestamps = np.array(sorted(panel.loc[mask, "timestamp"].unique()))
    timestamps = timestamps[:: cfg.rebalance_hours]

    prev_weights = pd.Series(0.0, index=symbols)
    current_state: dict[str, tuple[int, int]] = {}
    equity = 1.0
    rows_out: list[dict[str, Any]] = []
    pos_out: list[dict[str, Any]] = []

    for ts in timestamps:
        rows = panel[panel["timestamp"] == ts].copy()
        rows = rows[np.isfinite(rows[score_col]) & np.isfinite(rows[ret_col]) & np.isfinite(rows[label_col])]
        if len(rows) < cfg.long_k + cfg.short_k:
            continue

        rows["score_rank"] = rows[score_col].rank(pct=True, method="average")
        rows["ema_rank"] = rows["score_ema_025"].rank(pct=True, method="average")
        rows["base_rank"] = rows["score_baseline"].rank(pct=True, method="average")
        rows = rows.sort_values(score_col, ascending=False)
        rows_by_symbol = rows.set_index("symbol")

        target_sides = select_target_sides(rows, current_state, cfg, score_col)
        target_sides = apply_short_residual_confirmation(target_sides, rows_by_symbol, risk_cfg)
        if not any(side > 0 for side in target_sides.values()) or not any(side < 0 for side in target_sides.values()):
            continue

        state_sides = target_sides.copy()
        selected = rows_by_symbol.reindex(list(target_sides)).reset_index()
        raw_weights = sides_to_weights(target_sides, cfg.gross_exposure)
        target_weights, optimizer_info = build_weights(raw_weights, selected, rows_by_symbol, cfg, risk_cfg)
        target_weights = target_weights[target_weights.abs() > 1e-12]
        if target_weights.abs().sum() <= 0:
            continue

        regime_scale = float(rows["family_b_regime_gate"].dropna().iloc[0]) if use_regime_gate else 1.0
        target_weights = target_weights * regime_scale

        target_full_weights = pd.Series(0.0, index=symbols)
        target_full_weights.loc[target_weights.index] = target_weights
        turnover = float((target_full_weights - prev_weights).abs().sum())
        cost = turnover * (cfg.one_way_fee_bps + cfg.one_way_slippage_bps) / 10_000.0

        active_weights = target_full_weights[target_full_weights.abs() > 1e-12]
        active_rows = rows_by_symbol.reindex(active_weights.index)
        gross_period_ret = float((active_weights * active_rows[ret_col]).sum())
        period_ret = gross_period_ret - cost
        equity *= 1.0 + period_ret

        rank_ic = spearman_ic(rows[score_col], rows[label_col])
        long_labels = active_rows.loc[active_weights[active_weights > 0].index, label_col]
        short_labels = active_rows.loc[active_weights[active_weights < 0].index, label_col]
        top_bottom = float(long_labels.mean() - short_labels.mean()) if len(long_labels) and len(short_labels) else np.nan
        exposures = portfolio_exposures(active_weights, active_rows, cfg.neutralize_exposures)

        rows_out.append(
            {
                "timestamp": pd.Timestamp(ts),
                "exit_timestamp": pd.Timestamp(ts) + pd.Timedelta(hours=cfg.execution_delay_bars + cfg.primary_horizon),
                "period_return": period_ret,
                "gross_period_return": gross_period_ret,
                "cost_return": cost,
                "equity": equity,
                "turnover": turnover,
                "target_turnover": turnover,
                "turnover_saved": 0.0,
                "gross_exposure": float(active_weights.abs().sum()),
                "target_gross_exposure": float(target_full_weights.abs().sum()),
                "rank_ic": rank_ic,
                "top_bottom_label_spread": top_bottom,
                "abs_beta_btc": abs(exposures.get("beta_btc", 0.0)),
                "abs_beta_eth": abs(exposures.get("beta_eth", 0.0)),
                "abs_beta_market": abs(exposures.get("beta_market", 0.0)),
                "symbol_concentration": float((active_weights**2).sum()),
                "max_abs_weight": float(active_weights.abs().max()),
                "delay_tau_days": delay_cfg.tau_days,
                "delay_retention": 0.0,
                "delay_new_target_weight": 1.0,
                "regime_gross_scale": regime_scale,
                **optimizer_info,
                "target_applied": 1.0,
                "selected_formulas": selected_formulas,
            }
        )

        for symbol, weight in active_weights.items():
            pos_out.append(
                {
                    "timestamp": pd.Timestamp(ts),
                    "exit_timestamp": pd.Timestamp(ts) + pd.Timedelta(hours=cfg.execution_delay_bars + cfg.primary_horizon),
                    "symbol": symbol,
                    "weight": float(weight),
                    "target_weight": float(target_full_weights.loc[symbol]),
                    "score": float(rows_by_symbol.loc[symbol, score_col]),
                    "ema_rank": float(rows_by_symbol.loc[symbol, "ema_rank"]),
                    "base_rank": float(rows_by_symbol.loc[symbol, "base_rank"]),
                    "fwd_simple_ret": float(rows_by_symbol.loc[symbol, ret_col]),
                    "label_resid_ret": float(rows_by_symbol.loc[symbol, label_col]),
                    "contribution": float(weight * rows_by_symbol.loc[symbol, ret_col]),
                    "selected_score_cols": " | ".join(selected_score_cols),
                    "regime_gross_scale": regime_scale,
                }
            )

        prev_weights = target_full_weights
        current_state = update_state(current_state, state_sides, True)

    equity_curve = pd.DataFrame(rows_out)
    positions = pd.DataFrame(pos_out)
    metrics = aggregate_oos_metrics(equity_curve, cfg.rebalance_hours)
    return equity_curve, positions, metrics


def write_run_result(output_dir: Path, result: dict[str, Any]) -> None:
    result["folds"].to_csv(output_dir / "walk_forward_folds.csv", index=False)
    result["equity"].to_csv(output_dir / "equity_curve.csv", index=False)
    result["positions"].to_csv(output_dir / "positions.csv", index=False)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(to_jsonable(result["metrics"]), fh, indent=2, ensure_ascii=False)


def build_ic_summary(panel: pd.DataFrame, selections: pd.DataFrame, candidates: list[dict[str, str]]) -> pd.DataFrame:
    label_cols = [
        "label_slow_medium",
        "label_slow_low_turnover",
        "label_path_slow_medium",
        "label_path_slow_low_turnover",
    ]
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        score_col = candidate["score_col"]
        for label_col in label_cols:
            fold_ics: list[float] = []
            ts_ics: list[float] = []
            for _, fold in selections.iterrows():
                mask = (panel["timestamp"] >= fold["test_start"]) & (panel["timestamp"] < fold["test_end"])
                test = panel.loc[mask, ["timestamp", score_col, label_col]].replace([np.inf, -np.inf], np.nan).dropna()
                if test.empty:
                    continue
                per_ts = test.groupby("timestamp", sort=False).apply(
                    lambda g: g[score_col].rank().corr(g[label_col].rank()) if len(g) >= 8 else np.nan,
                    include_groups=False,
                ).dropna()
                if not per_ts.empty:
                    fold_ics.append(float(per_ts.mean()))
                    ts_ics.extend(float(x) for x in per_ts)
            rows.append(
                {
                    "name": candidate["name"],
                    "score_col": score_col,
                    "formula": candidate["formula"],
                    "label": label_col,
                    "mean_ic": float(np.mean(ts_ics)) if ts_ics else np.nan,
                    "median_ic": float(np.median(ts_ics)) if ts_ics else np.nan,
                    "ic_std": float(np.std(ts_ics, ddof=0)) if ts_ics else np.nan,
                    "icir": float(np.mean(ts_ics) / np.std(ts_ics, ddof=0)) if ts_ics and np.std(ts_ics, ddof=0) > 0 else np.nan,
                    "positive_ts_ic_rate": float(np.mean(np.array(ts_ics) > 0.0)) if ts_ics else np.nan,
                    "positive_fold_ic_rate": float(np.mean(np.array(fold_ics) > 0.0)) if fold_ics else np.nan,
                    "fold_count": int(len(fold_ics)),
                    "timestamp_count": int(len(ts_ics)),
                }
            )
    return pd.DataFrame(rows).sort_values(["label", "mean_ic"], ascending=[True, False])


def load_baseline_summary(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def build_baseline_comparison(summary: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    if baseline.empty:
        return summary.copy()
    cols = [
        "horizon",
        "total_return",
        "gross_total_return_before_cost",
        "sharpe",
        "max_drawdown",
        "gross_max_drawdown",
        "mean_turnover",
        "cost_drag",
    ]
    base = baseline[[col for col in cols if col in baseline.columns]].copy()
    base = base.rename(columns={col: f"baseline_{col}" for col in base.columns if col != "horizon"})
    out = summary.merge(base, on="horizon", how="left")
    for col in cols:
        if col == "horizon":
            continue
        base_col = f"baseline_{col}"
        if base_col in out:
            out[f"delta_{col}"] = out[col] - out[base_col]
    return out.sort_values(["delta_total_return", "total_return"], ascending=[False, False])


def build_best_vs_reference_combo(best_result: dict[str, Any], best_key: str) -> dict[str, Any]:
    root = EXPERIMENT_ROOT
    base_path = root / "baseline_lock_review" / "horizon" / "h048_tau0" / "equity_curve.csv"
    fam_a_path = root / "outputs" / "p2_family_a_momentum" / "best_run" / "equity_curve.csv"
    out: dict[str, Any] = {"best_key": best_key}
    best = best_result["equity"].sort_values("timestamp")
    if base_path.exists():
        base = pd.read_csv(base_path, parse_dates=["timestamp"]).sort_values("timestamp")
        merged = base[["timestamp", "period_return"]].rename(columns={"period_return": "baseline"}).merge(
            best[["timestamp", "period_return"]].rename(columns={"period_return": "family_b"}),
            on="timestamp",
            how="inner",
        )
        out["corr_to_locked_baseline"] = float(merged["baseline"].corr(merged["family_b"])) if len(merged) > 3 else np.nan
        out["aligned_periods_with_baseline"] = int(len(merged))
    if fam_a_path.exists():
        fam_a = pd.read_csv(fam_a_path, parse_dates=["timestamp"]).sort_values("timestamp")
        merged = fam_a[["timestamp", "period_return"]].rename(columns={"period_return": "family_a"}).merge(
            best[["timestamp", "period_return"]].rename(columns={"period_return": "family_b"}),
            on="timestamp",
            how="inner",
        )
        out["corr_to_family_a_res_trend_quality"] = float(merged["family_a"].corr(merged["family_b"])) if len(merged) > 3 else np.nan
        out["aligned_periods_with_family_a"] = int(len(merged))
    return out


def plot_summary(output_dir: Path, summary: pd.DataFrame) -> None:
    top = summary.sort_values(["total_return", "sharpe"], ascending=[False, False]).head(14).sort_values("total_return")
    plt.figure(figsize=(12, 7))
    colors = np.where(top["total_return"] >= 0.0, "#2a9d8f", "#c44536")
    plt.barh(top["run"], top["total_return"], color=colors)
    plt.axvline(0.0, color="black", linewidth=0.8, alpha=0.6)
    plt.title("Family B top runs by net return")
    plt.xlabel("Net cumulative return")
    plt.gca().xaxis.set_major_formatter(PercentFormatter(1.0))
    plt.tight_layout()
    plt.savefig(output_dir / "family_b_top_net_returns.png", dpi=180)
    plt.close()


def plot_best_curves(output_dir: Path, equity: pd.DataFrame, best_key: str) -> None:
    if equity.empty:
        return
    frame = equity.sort_values("timestamp")
    net = (1.0 + frame["period_return"]).cumprod() - 1.0
    gross = (1.0 + frame["gross_period_return"]).cumprod() - 1.0
    plt.figure(figsize=(12, 6.5))
    plt.plot(frame["timestamp"], gross, label="gross before cost", linewidth=2)
    plt.plot(frame["timestamp"], net, label="net after cost", linewidth=2)
    plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    plt.title(f"Best Family B run: {best_key}")
    plt.ylabel("Cumulative return")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "best_net_gross_curve.png", dpi=180)
    plt.close()


def plot_top_curves(output_dir: Path, result_index: dict[str, dict[str, Any]], top_keys: list[str]) -> None:
    plt.figure(figsize=(12, 6.5))
    for key in top_keys:
        equity = result_index[key]["equity"].sort_values("timestamp")
        if equity.empty:
            continue
        net = (1.0 + equity["period_return"]).cumprod() - 1.0
        plt.plot(equity["timestamp"], net, label=key, linewidth=1.8)
    plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    plt.title("Family B top-5 net curves")
    plt.ylabel("Cumulative return")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "top5_net_curves.png", dpi=180)
    plt.close()


def plot_ic_summary(output_dir: Path, ic_summary: pd.DataFrame) -> None:
    target = ic_summary[ic_summary["label"] == "label_path_slow_low_turnover"].copy()
    if target.empty:
        return
    target = target.sort_values("mean_ic").tail(14)
    plt.figure(figsize=(12, 7))
    colors = np.where(target["mean_ic"] >= 0.0, "#2a9d8f", "#c44536")
    plt.barh(target["name"], target["mean_ic"], color=colors)
    plt.axvline(0.0, color="black", linewidth=0.8, alpha=0.6)
    plt.title("Family B OOS IC vs path slow low-turnover label")
    plt.xlabel("Mean cross-sectional Spearman IC")
    plt.grid(True, axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "ic_path_slow_low_turnover.png", dpi=180)
    plt.close()


def plot_cost_sensitivity(output_dir: Path, cost_summary: pd.DataFrame) -> None:
    if cost_summary.empty:
        return
    plt.figure(figsize=(12, 6.5))
    for name, group in cost_summary.groupby("experiment"):
        group = group.sort_values("one_way_cost_bps")
        plt.plot(group["one_way_cost_bps"], group["total_return"], marker="o", label=name)
    plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    plt.title("Family B top-5 cost sensitivity")
    plt.xlabel("one-way cost bps")
    plt.ylabel("Net cumulative return")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "top5_cost_sensitivity.png", dpi=180)
    plt.close()


def plot_asset_contribution(output_dir: Path, asset_summary: pd.DataFrame) -> None:
    if asset_summary.empty:
        return
    frame = asset_summary.sort_values("gross_contribution_sum")
    plt.figure(figsize=(11, 7))
    colors = np.where(frame["gross_contribution_sum"] >= 0.0, "#2a9d8f", "#c44536")
    plt.barh(frame["symbol"], frame["gross_contribution_sum"], color=colors)
    plt.axvline(0.0, color="black", linewidth=0.8, alpha=0.6)
    plt.title("Best Family B gross contribution by asset")
    plt.xlabel("Gross contribution sum")
    plt.gca().xaxis.set_major_formatter(PercentFormatter(1.0))
    plt.tight_layout()
    plt.savefig(output_dir / "best_asset_contribution.png", dpi=180)
    plt.close()


def write_report(
    output_dir: Path,
    summary: pd.DataFrame,
    ic_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    best_key: str,
    regime_summary: pd.DataFrame,
    side_summary: pd.DataFrame,
    asset_summary: pd.DataFrame,
    cost_summary: pd.DataFrame,
    corr_combo: dict[str, Any],
) -> None:
    display_cols = [
        "run",
        "horizon",
        "name",
        "total_return",
        "gross_total_return_before_cost",
        "sharpe",
        "max_drawdown",
        "gross_max_drawdown",
        "mean_turnover",
        "cost_drag",
        "positive_fold_rate",
        "mean_regime_gross_scale",
        "low_regime_gross_scale_rate",
        "mean_abs_beta_market",
        "max_abs_weight",
    ]
    comp_cols = [
        "run",
        "horizon",
        "name",
        "total_return",
        "baseline_total_return",
        "delta_total_return",
        "gross_total_return_before_cost",
        "baseline_gross_total_return_before_cost",
        "delta_gross_total_return_before_cost",
        "mean_turnover",
        "baseline_mean_turnover",
        "delta_mean_turnover",
    ]
    ic_focus = ic_summary[
        ic_summary["label"].isin(["label_slow_low_turnover", "label_path_slow_low_turnover"])
    ].sort_values(["label", "mean_ic"], ascending=[True, False])
    cost_pivot = (
        cost_summary.pivot_table(index="one_way_cost_bps", columns="experiment", values="total_return")
        .reset_index()
        .sort_values("one_way_cost_bps")
    )
    lines = [
        "# P2 Family B Residual Mean Reversion With Stationarity Filter",
        "",
        "## Design",
        "",
        "- OHLCV-only residual slow-anchor mean reversion.",
        "- Uses residual cumulative index deviation from EWMA anchors at 168h and 336h.",
        "- OU speed and half-life are estimated with trailing rolling regressions only.",
        "- Stationarity gate is strongest when half-life is between 48h and 240h.",
        "- Beta stability and liquidity quality are used as non-directional gates.",
        "- Regime gross gate reduces exposure in high market-vol or high residual-dispersion regimes.",
        "- Backtests use the locked strict-neutral allocator.",
        "",
        "## Best Run",
        "",
        f"- best_run: `{best_key}`",
        "",
        "## Candidate Summary",
        "",
        "```text",
        summary.sort_values(["total_return", "sharpe"], ascending=[False, False])[
            [col for col in display_cols if col in summary.columns]
        ].to_string(index=False),
        "```",
        "",
        "## Baseline Comparison",
        "",
        "```text",
        comparison[[col for col in comp_cols if col in comparison.columns]].head(20).to_string(index=False),
        "```",
        "",
        "## OOS IC Focus",
        "",
        "```text",
        ic_focus.to_string(index=False),
        "```",
        "",
        "## Reference Correlation",
        "",
        "```json",
        json.dumps(to_jsonable(corr_combo), indent=2, ensure_ascii=False),
        "```",
        "",
        "## Top-5 Cost Sensitivity",
        "",
        "```text",
        cost_pivot.to_string(index=False),
        "```",
        "",
        "## Best Run Regime Summary",
        "",
        "```text",
        regime_summary.to_string(index=False),
        "```",
        "",
        "## Best Run Long/Short Contribution",
        "",
        "```text",
        side_summary.to_string(index=False),
        "```",
        "",
        "## Best Run Asset Contribution",
        "",
        "```text",
        asset_summary.to_string(index=False),
        "```",
        "",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def update_config_stage(config_path: Path) -> None:
    marker = "  p2_family_b_mean_reversion:\n"
    text = config_path.read_text(encoding="utf-8")
    if marker in text:
        return
    addition = """
  p2_family_b_mean_reversion:
    script: ../../../scripts/run_p2_family_b_mean_reversion.py
    output_dir: ../outputs/p2_family_b_mean_reversion
    report: ../outputs/p2_family_b_mean_reversion/REPORT.md
    candidate_summary: ../outputs/p2_family_b_mean_reversion/candidate_summary.csv
    ic_summary: ../outputs/p2_family_b_mean_reversion/ic_summary.csv
    baseline_comparison: ../outputs/p2_family_b_mean_reversion/baseline_comparison.csv
"""
    config_path.write_text(text.rstrip() + "\n" + addition, encoding="utf-8")


if __name__ == "__main__":
    main()
