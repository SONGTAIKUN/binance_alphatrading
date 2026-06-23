#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from crypto_stat_arb.config import ensure_dir, load_config
from run_p2_family_b_mean_reversion import centered_rank_by_timestamp
from run_residual_alphagpt_search import build_candidate_scores, prepare_alpha_panel


EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "baseline_locked_v1"
DEFAULT_CONFIG = EXPERIMENT_ROOT / "configs" / "baseline_locked_v1.yaml"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "outputs" / "p6_feature_expansion"
SLOW_LABELS = EXPERIMENT_ROOT / "outputs" / "p1_slow_labels" / "slow_residual_labels.csv.gz"
P2_FEATURES = EXPERIMENT_ROOT / "outputs" / "p2_manual_slow_alpha_grid" / "manual_slow_feature_panel.csv.gz"
P3D_POSITIONS = EXPERIMENT_ROOT / "outputs" / "p3d_p4_gated_rerank_alphagpt" / "nested_incremental_run" / "positions.csv"
P3D_EQUITY = EXPERIMENT_ROOT / "outputs" / "p3d_p4_gated_rerank_alphagpt" / "nested_incremental_run" / "equity_curve.csv"

PRIMARY_LABEL = "label_path_slow_low_turnover"
LABEL_COLS = [
    "label_slow_medium",
    "label_slow_low_turnover",
    "label_path_slow_medium",
    "label_path_slow_low_turnover",
    "label_horizon_sign_consistency",
    "label_path_horizon_sign_consistency",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--diagnostic-horizon", type=int, default=72)
    parser.add_argument("--max-abs-p3d-corr", type=float, default=0.35)
    parser.add_argument("--min-primary-ic", type=float, default=0.005)
    parser.add_argument("--min-positive-fold-rate", type=float, default=0.52)
    parser.add_argument("--min-q25-fold-ic", type=float, default=-0.03)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    output_dir = ensure_dir(args.output_dir)
    horizon = int(args.diagnostic_horizon)

    panel_path = resolve_config_path(config_path, cfg["inputs"]["panel"])
    selections_path = resolve_config_path(config_path, cfg["inputs"]["formula_selections"])
    selections = pd.read_csv(selections_path, parse_dates=["train_start", "train_end", "test_start", "test_end"])

    print("[1/7] Loading panel, P1 labels, P2 slow features, and P3d references", flush=True)
    panel = pd.read_csv(panel_path, parse_dates=["timestamp"])
    panel = prepare_alpha_panel(panel, [48, 72, 96, 168], execution_delay_bars=1)
    build_candidate_scores(panel)
    labels = pd.read_csv(SLOW_LABELS, parse_dates=["timestamp"], usecols=["timestamp", "symbol", *LABEL_COLS])
    p2_features = pd.read_csv(P2_FEATURES, parse_dates=["timestamp"])
    panel = panel.merge(labels, on=["timestamp", "symbol"], how="left")
    panel = panel.merge(p2_features, on=["timestamp", "symbol"], how="left", suffixes=("", "_p2"))

    print("[2/7] Building OHLCV-only C/D/E/F/G feature candidates", flush=True)
    candidates = build_p6_features(panel)
    candidate_meta = pd.DataFrame(candidates)
    candidate_meta.to_csv(output_dir / "candidate_formulas.csv", index=False)
    write_feature_panel(panel, candidates, output_dir / "p6_feature_panel.csv.gz")

    print("[3/7] Computing walk-forward OOS IC and path-aware IC", flush=True)
    ic_summary, fold_ic = build_ic_diagnostics(panel, selections, candidates, horizon)
    ic_summary.to_csv(output_dir / "ic_summary.csv", index=False)
    fold_ic.to_csv(output_dir / "fold_ic.csv", index=False)

    print("[4/7] Computing signal turnover and P3d correlation diagnostics", flush=True)
    turnover = build_signal_turnover(panel, selections, candidates, horizon)
    p3d_corr = build_p3d_weight_correlations(panel, candidates)
    diagnostics = build_candidate_diagnostics(candidate_meta, ic_summary, turnover, p3d_corr)
    diagnostics.to_csv(output_dir / "candidate_diagnostics.csv", index=False)

    print("[5/7] Applying retention rules", flush=True)
    retained, watchlist = select_retained_features(
        diagnostics=diagnostics,
        max_abs_p3d_corr=float(args.max_abs_p3d_corr),
        min_primary_ic=float(args.min_primary_ic),
        min_positive_fold_rate=float(args.min_positive_fold_rate),
        min_q25_fold_ic=float(args.min_q25_fold_ic),
    )
    retained.to_csv(output_dir / "retained_features.csv", index=False)
    watchlist.to_csv(output_dir / "watchlist_features.csv", index=False)

    print("[6/7] Writing plots", flush=True)
    plot_ic_bar(output_dir, diagnostics)
    plot_turnover_corr(output_dir, diagnostics, retained)
    plot_family_counts(output_dir, diagnostics, retained)

    print("[7/7] Writing report and config stage", flush=True)
    write_report(output_dir, diagnostics, retained, watchlist, args)
    update_config_stage(config_path)
    print(f"Done: {output_dir}", flush=True)


def resolve_config_path(config_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def build_p6_features(panel: pd.DataFrame) -> list[dict[str, Any]]:
    panel.sort_values(["symbol", "timestamp"], inplace=True)
    grouped = panel.groupby("symbol", group_keys=False)
    timestamp_grouped = panel.groupby("timestamp", group_keys=False)

    # Residual equity curve known at t from past residual returns.
    panel["resid_equity"] = grouped["residual_log_ret"].cumsum()
    panel["market_ret"] = panel.groupby("timestamp")["log_ret"].transform("mean")
    panel["market_resid_ret"] = panel.groupby("timestamp")["residual_log_ret"].transform("mean")

    dollar_vol = panel["quote_volume"].replace(0.0, np.nan).astype(float)
    total_dollar_vol = panel.groupby("timestamp")["quote_volume"].transform("sum").replace(0.0, np.nan)
    panel["volume_share"] = panel["quote_volume"] / total_dollar_vol
    panel["log_dollar_volume"] = np.log(dollar_vol)
    panel["amihud_1h"] = panel["simple_ret"].abs() / dollar_vol
    panel["resid_amihud_1h"] = panel["residual_log_ret"].abs() / dollar_vol
    panel["hl_range"] = np.log(panel["high"].replace(0.0, np.nan) / panel["low"].replace(0.0, np.nan)).replace(
        [np.inf, -np.inf], np.nan
    )
    panel["oc_range"] = (np.log(panel["close"].replace(0.0, np.nan) / panel["open"].replace(0.0, np.nan))).abs()

    for window in [72, 168, 336, 504]:
        minp = max(48, window // 3)
        panel[f"dollar_vol_ewm_{window}"] = grouped["log_dollar_volume"].transform(
            lambda s, span=window: s.ewm(span=span, min_periods=max(24, span // 4), adjust=False).mean()
        )
        panel[f"volume_share_ewm_{window}"] = grouped["volume_share"].transform(
            lambda s, span=window: s.ewm(span=span, min_periods=max(24, span // 4), adjust=False).mean()
        )
        panel[f"amihud_{window}"] = grouped["amihud_1h"].transform(lambda s, w=window, mp=minp: s.rolling(w, min_periods=mp).mean())
        panel[f"resid_amihud_{window}"] = grouped["resid_amihud_1h"].transform(
            lambda s, w=window, mp=minp: s.rolling(w, min_periods=mp).mean()
        )
        panel[f"resid_vol_{window}"] = grouped["residual_log_ret"].transform(lambda s, w=window, mp=minp: s.rolling(w, min_periods=mp).std(ddof=0))
        panel[f"range_vol_{window}"] = grouped["hl_range"].transform(lambda s, w=window, mp=minp: s.rolling(w, min_periods=mp).mean())
        panel[f"market_vol_{window}"] = panel["timestamp"].map(
            panel.groupby("timestamp")["market_ret"].first().rolling(window, min_periods=minp).std(ddof=0)
        )
        panel[f"resid_corr_market_{window}"] = grouped.apply(
            lambda g, w=window, mp=minp: g["residual_log_ret"].rolling(w, min_periods=mp).corr(g["market_resid_ret"])
        ).reset_index(level=0, drop=True)
        panel[f"resid_corr_raw_market_{window}"] = grouped.apply(
            lambda g, w=window, mp=minp: g["residual_log_ret"].rolling(w, min_periods=mp).corr(g["market_ret"])
        ).reset_index(level=0, drop=True)

    panel["dollar_volume_trend_7d_30d"] = panel["dollar_vol_ewm_168"] - panel["dollar_vol_ewm_504"]
    panel["volume_share_trend_7d_30d_p6"] = panel["volume_share_ewm_168"] - panel["volume_share_ewm_504"]
    panel["amihud_improvement_7d_30d"] = -(np.log(panel["amihud_168"].replace(0.0, np.nan)) - np.log(panel["amihud_504"].replace(0.0, np.nan)))
    panel["resid_amihud_improvement_7d_30d"] = -(
        np.log(panel["resid_amihud_168"].replace(0.0, np.nan)) - np.log(panel["resid_amihud_504"].replace(0.0, np.nan))
    )
    panel["volume_persistence_7d"] = grouped["volume_share"].transform(
        lambda s: s.rolling(168, min_periods=48).corr(s.shift(24))
    )

    panel["vol_compression_72_336"] = -(np.log(panel["resid_vol_72"].replace(0.0, np.nan)) - np.log(panel["resid_vol_336"].replace(0.0, np.nan)))
    panel["vol_compression_168_504"] = -(np.log(panel["resid_vol_168"].replace(0.0, np.nan)) - np.log(panel["resid_vol_504"].replace(0.0, np.nan)))
    panel["range_compression_72_336"] = -(np.log(panel["range_vol_72"].replace(0.0, np.nan)) - np.log(panel["range_vol_336"].replace(0.0, np.nan)))
    panel["breakout_risk_72"] = np.log(panel["range_vol_72"].replace(0.0, np.nan)) + np.log(panel["resid_vol_72"].replace(0.0, np.nan))
    panel["vol_of_vol_336"] = grouped["resid_vol_72"].transform(lambda s: s.rolling(336, min_periods=96).std(ddof=0))

    market_resid = panel.groupby("timestamp")["residual_log_ret"].std(ddof=0)
    breadth = panel.groupby("timestamp")["residual_log_ret"].apply(lambda s: float((s > 0.0).mean()))
    panel["market_resid_dispersion_7d"] = panel["timestamp"].map(market_resid.rolling(168, min_periods=48).mean())
    panel["market_resid_dispersion_trend"] = panel["timestamp"].map(
        market_resid.rolling(168, min_periods=48).mean() - market_resid.rolling(504, min_periods=168).mean()
    )
    panel["resid_positive_breadth_7d"] = panel["timestamp"].map(breadth.rolling(168, min_periods=48).mean())
    panel["crowding_same_side"] = np.sign(panel["res_mom_168_24"].fillna(0.0)) * (panel["resid_positive_breadth_7d"] - 0.5)
    panel["resid_uncrowded_mom"] = panel["res_mom_168_24"] * (1.0 - panel["crowding_same_side"].abs()).clip(lower=0.0)
    panel["idiosyncratic_purity_336"] = -panel["resid_corr_raw_market_336"].abs()
    panel["resid_basket_diversifier_336"] = -panel["resid_corr_market_336"].abs()

    market_ret_30d = panel.groupby("timestamp")["market_ret"].first().rolling(720, min_periods=168).sum()
    market_ret_7d = panel.groupby("timestamp")["market_ret"].first().rolling(168, min_periods=48).sum()
    market_vol_14d = panel.groupby("timestamp")["market_ret"].first().rolling(336, min_periods=72).std(ddof=0)
    high_vol_threshold = market_vol_14d.expanding(min_periods=720).quantile(2 / 3).shift(1)
    low_vol_threshold = market_vol_14d.expanding(min_periods=720).quantile(1 / 3).shift(1)
    panel["regime_market_up_30d"] = panel["timestamp"].map((market_ret_30d > 0.0).astype(float)).fillna(0.0)
    panel["regime_market_down_30d"] = panel["timestamp"].map((market_ret_30d < 0.0).astype(float)).fillna(0.0)
    panel["regime_crash_7d_p6"] = panel["timestamp"].map((market_ret_7d < -0.05).astype(float)).fillna(0.0)
    panel["regime_high_vol_p6"] = panel["timestamp"].map((market_vol_14d > high_vol_threshold).astype(float)).fillna(0.0)
    panel["regime_low_vol_p6"] = panel["timestamp"].map((market_vol_14d < low_vol_threshold).astype(float)).fillna(0.0)
    panel["regime_normal_p6"] = (1.0 - panel["regime_crash_7d_p6"]) * (1.0 - panel["regime_high_vol_p6"]).clip(lower=0.0)

    raw_feature_cols = [
        "dollar_volume_trend_7d_30d",
        "volume_share_trend_7d_30d_p6",
        "amihud_improvement_7d_30d",
        "resid_amihud_improvement_7d_30d",
        "volume_persistence_7d",
        "vol_compression_72_336",
        "vol_compression_168_504",
        "range_compression_72_336",
        "breakout_risk_72",
        "vol_of_vol_336",
        "market_resid_dispersion_7d",
        "market_resid_dispersion_trend",
        "resid_positive_breadth_7d",
        "crowding_same_side",
        "resid_uncrowded_mom",
        "idiosyncratic_purity_336",
        "resid_basket_diversifier_336",
        "regime_market_up_30d",
        "regime_market_down_30d",
        "regime_crash_7d_p6",
        "regime_high_vol_p6",
        "regime_low_vol_p6",
        "regime_normal_p6",
    ]
    for col in raw_feature_cols:
        if col in panel:
            panel[f"cs_{col}"] = centered_rank_by_timestamp(panel, col)

    # Family C: liquidity, volume persistence, and carry-like residual momentum conditioning.
    panel["p6_C01_liq_persistent_mom"] = (
        panel["cs_res_mom_168_24"] + 0.35 * panel["cs_volume_share_trend_7d_30d_p6"] + 0.25 * panel["cs_amihud_improvement_7d_30d"]
    )
    panel["p6_C02_dollar_volume_mom"] = panel["cs_res_mom_336_24"] + 0.40 * panel["cs_dollar_volume_trend_7d_30d"]
    panel["p6_C03_amihud_quality"] = (
        0.50 * panel["cs_res_mom_168_24"] + 0.35 * panel["cs_resid_amihud_improvement_7d_30d"] + 0.15 * panel["cs_volume_persistence_7d"]
    )
    panel["p6_C04_liquidity_reversion_guard"] = panel["alpha_B_soft_combo"] + 0.35 * panel["cs_amihud_improvement_7d_30d"]
    panel["p6_C05_low_capacity_penalty"] = panel["cs_res_mom_336_24"] + 0.30 * panel["cs_dollar_volume_trend_7d_30d"] - 0.30 * panel["cs_breakout_risk_72"]

    # Family D: volatility compression, breakout risk, and residual vol quality.
    panel["p6_D01_compression_mom"] = panel["cs_res_mom_168_24"] + 0.45 * panel["cs_vol_compression_72_336"]
    panel["p6_D02_deep_compression_mom"] = panel["cs_res_mom_336_24"] + 0.50 * panel["cs_vol_compression_168_504"]
    panel["p6_D03_range_compression_quality"] = (
        0.45 * panel["cs_res_mom_168_24"] + 0.35 * panel["cs_range_compression_72_336"] - 0.20 * panel["cs_breakout_risk_72"]
    )
    panel["p6_D04_low_vov_momentum"] = panel["cs_res_mom_336_24"] - 0.45 * panel["cs_vol_of_vol_336"]
    panel["p6_D05_breakout_risk_contra"] = panel["alpha_B_soft_combo"] - 0.35 * panel["cs_breakout_risk_72"]

    # Family E: residual crowding, dispersion, and co-movement.
    panel["p6_E01_uncrowded_mom"] = centered_rank_by_timestamp(panel, "resid_uncrowded_mom")
    panel["p6_E02_purity_mom"] = panel["cs_res_mom_168_24"] + 0.40 * panel["cs_idiosyncratic_purity_336"]
    panel["p6_E03_basket_diversifier_mom"] = panel["cs_res_mom_336_24"] + 0.45 * panel["cs_resid_basket_diversifier_336"]
    panel["p6_E04_dispersion_reversion"] = panel["alpha_B_soft_combo"] - 0.25 * panel["cs_market_resid_dispersion_trend"]
    panel["p6_E05_anti_crowded_reversion"] = -centered_rank_by_timestamp(panel, "crowding_same_side") + 0.25 * panel["alpha_B_seed_combo"]

    # Family F: market-regime conditional slow alphas.
    panel["p6_F01_normal_regime_mom"] = panel["alpha_A_res_trend_quality"] * panel["regime_normal_p6"]
    panel["p6_F02_uptrend_mom"] = panel["cs_res_mom_336_24"] * panel["regime_market_up_30d"]
    panel["p6_F03_downtrend_quality_revert"] = panel["alpha_B_soft_combo"] * panel["regime_market_down_30d"]
    panel["p6_F04_low_vol_reversion"] = panel["alpha_B_seed_combo"] * panel["regime_low_vol_p6"]
    panel["p6_F05_high_vol_defensive_quality"] = (
        0.50 * panel["cs_liquidity_quality"] + 0.35 * panel["cs_idiosyncratic_purity_336"] - 0.30 * panel["cs_breakout_risk_72"]
    ) * panel["regime_high_vol_p6"]

    # Family G: ensemble/stability/meta-alpha, still ex-ante and label-free.
    panel["p6_G01_slow_consensus"] = (
        0.25 * panel["p6_C01_liq_persistent_mom"]
        + 0.25 * panel["p6_D02_deep_compression_mom"]
        + 0.25 * panel["p6_E02_purity_mom"]
        + 0.25 * panel["p6_F01_normal_regime_mom"]
    )
    component_cols = ["p6_C01_liq_persistent_mom", "p6_D02_deep_compression_mom", "p6_E02_purity_mom", "p6_F01_normal_regime_mom"]
    panel["p6_component_disagreement"] = panel[component_cols].std(axis=1, ddof=0)
    panel["p6_G02_consensus_low_disagreement"] = panel["p6_G01_slow_consensus"] - 0.35 * centered_rank_by_timestamp(panel, "p6_component_disagreement")
    panel["p6_G03_liq_purity_compression"] = (
        0.30 * panel["cs_liquidity_quality"] + 0.35 * panel["cs_idiosyncratic_purity_336"] + 0.35 * panel["cs_vol_compression_168_504"]
    )
    panel["p6_G04_p3d_diversifier_candidate"] = (
        0.40 * panel["p6_E01_uncrowded_mom"] + 0.30 * panel["p6_D04_low_vov_momentum"] + 0.30 * panel["p6_C03_amihud_quality"]
    )
    panel["p6_G05_stable_low_turnover_meta"] = (
        0.35 * panel["cs_res_mom_336_24"] + 0.25 * panel["cs_dollar_volume_trend_7d_30d"] + 0.25 * panel["cs_vol_compression_168_504"] + 0.15 * panel["cs_idiosyncratic_purity_336"]
    )

    specs = [
        ("C01_liq_persistent_mom", "C_liquidity_volume", "p6_C01_liq_persistent_mom", "res_mom_168_24 + volume_share_trend + amihud_improvement"),
        ("C02_dollar_volume_mom", "C_liquidity_volume", "p6_C02_dollar_volume_mom", "res_mom_336_24 + dollar_volume_trend"),
        ("C03_amihud_quality", "C_liquidity_volume", "p6_C03_amihud_quality", "res_mom_168_24 + residual_amihud_improvement + volume_persistence"),
        ("C04_liquidity_reversion_guard", "C_liquidity_volume", "p6_C04_liquidity_reversion_guard", "B_soft_combo + amihud_improvement"),
        ("C05_low_capacity_penalty", "C_liquidity_volume", "p6_C05_low_capacity_penalty", "res_mom_336_24 + dollar_volume_trend - breakout_risk"),
        ("D01_compression_mom", "D_volatility_quality", "p6_D01_compression_mom", "res_mom_168_24 + vol_compression_72_336"),
        ("D02_deep_compression_mom", "D_volatility_quality", "p6_D02_deep_compression_mom", "res_mom_336_24 + vol_compression_168_504"),
        ("D03_range_compression_quality", "D_volatility_quality", "p6_D03_range_compression_quality", "res_mom_168_24 + range_compression - breakout_risk"),
        ("D04_low_vov_momentum", "D_volatility_quality", "p6_D04_low_vov_momentum", "res_mom_336_24 - vol_of_vol"),
        ("D05_breakout_risk_contra", "D_volatility_quality", "p6_D05_breakout_risk_contra", "B_soft_combo - breakout_risk"),
        ("E01_uncrowded_mom", "E_crowding_dispersion", "p6_E01_uncrowded_mom", "rank_cs(res_mom adjusted by residual breadth crowding)"),
        ("E02_purity_mom", "E_crowding_dispersion", "p6_E02_purity_mom", "res_mom_168_24 + idiosyncratic_purity"),
        ("E03_basket_diversifier_mom", "E_crowding_dispersion", "p6_E03_basket_diversifier_mom", "res_mom_336_24 + residual basket diversification"),
        ("E04_dispersion_reversion", "E_crowding_dispersion", "p6_E04_dispersion_reversion", "B_soft_combo - residual_dispersion_trend"),
        ("E05_anti_crowded_reversion", "E_crowding_dispersion", "p6_E05_anti_crowded_reversion", "-crowding_same_side + B_seed_combo"),
        ("F01_normal_regime_mom", "F_regime_conditional", "p6_F01_normal_regime_mom", "A_trend_quality * normal_regime"),
        ("F02_uptrend_mom", "F_regime_conditional", "p6_F02_uptrend_mom", "res_mom_336_24 * market_up_30d"),
        ("F03_downtrend_quality_revert", "F_regime_conditional", "p6_F03_downtrend_quality_revert", "B_soft_combo * market_down_30d"),
        ("F04_low_vol_reversion", "F_regime_conditional", "p6_F04_low_vol_reversion", "B_seed_combo * low_market_vol"),
        ("F05_high_vol_defensive_quality", "F_regime_conditional", "p6_F05_high_vol_defensive_quality", "liquidity + purity - breakout_risk in high vol"),
        ("G01_slow_consensus", "G_meta_ensemble", "p6_G01_slow_consensus", "average C/D/E/F slow components"),
        ("G02_consensus_low_disagreement", "G_meta_ensemble", "p6_G02_consensus_low_disagreement", "G01 - component disagreement penalty"),
        ("G03_liq_purity_compression", "G_meta_ensemble", "p6_G03_liq_purity_compression", "liquidity + purity + compression quality"),
        ("G04_p3d_diversifier_candidate", "G_meta_ensemble", "p6_G04_p3d_diversifier_candidate", "uncrowded + low_vov + amihud quality"),
        ("G05_stable_low_turnover_meta", "G_meta_ensemble", "p6_G05_stable_low_turnover_meta", "slow momentum + liquidity + compression + purity"),
    ]
    return [
        {"name": name, "family": family, "score_col": score_col, "formula": formula, "features": score_col}
        for name, family, score_col, formula in specs
    ]


def write_feature_panel(panel: pd.DataFrame, candidates: list[dict[str, Any]], output_path: Path) -> None:
    cols = ["timestamp", "symbol"]
    for candidate in candidates:
        cols.append(str(candidate["score_col"]))
    raw_cols = [
        col
        for col in panel.columns
        if col.startswith(("cs_dollar_", "cs_volume_", "cs_amihud", "cs_resid_amihud", "cs_vol_compression", "cs_range_", "cs_breakout", "cs_vol_of_vol", "cs_idiosyncratic", "cs_resid_basket"))
        or col.startswith(("regime_", "p6_component"))
    ]
    cols = list(dict.fromkeys(cols + raw_cols))
    panel[[col for col in cols if col in panel.columns]].to_csv(output_path, index=False)


def build_ic_diagnostics(
    panel: pd.DataFrame, selections: pd.DataFrame, candidates: list[dict[str, Any]], horizon: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    fold_rows = []
    fold_timestamps = build_oos_rebalance_timestamps(panel, selections, horizon)
    for candidate in candidates:
        score_col = str(candidate["score_col"])
        for label_col in LABEL_COLS:
            ts_ics: list[float] = []
            fold_ics: list[float] = []
            for _, fold in selections.iterrows():
                fold_id = int(fold["fold"])
                timestamps = fold_timestamps.get(fold_id, [])
                if not timestamps:
                    continue
                mask = panel["timestamp"].isin(timestamps)
                test = panel.loc[mask, ["timestamp", score_col, label_col]].replace([np.inf, -np.inf], np.nan).dropna()
                if test.empty:
                    continue
                per_ts = test.groupby("timestamp", sort=False).apply(lambda g: rank_corr(g[score_col], g[label_col]), include_groups=False)
                per_ts = per_ts.dropna()
                if per_ts.empty:
                    continue
                fold_ic = float(per_ts.mean())
                fold_ics.append(fold_ic)
                ts_ics.extend(float(x) for x in per_ts)
                fold_rows.append(
                    {
                        "fold": fold_id,
                        "name": candidate["name"],
                        "family": candidate["family"],
                        "score_col": score_col,
                        "label": label_col,
                        "fold_ic": fold_ic,
                        "timestamp_count": int(len(per_ts)),
                    }
                )
            ts_array = np.array(ts_ics, dtype=float)
            fold_array = np.array(fold_ics, dtype=float)
            rows.append(
                {
                    "name": candidate["name"],
                    "family": candidate["family"],
                    "score_col": score_col,
                    "label": label_col,
                    "mean_ic": float(np.nanmean(ts_array)) if len(ts_array) else np.nan,
                    "median_ic": float(np.nanmedian(ts_array)) if len(ts_array) else np.nan,
                    "ic_std": float(np.nanstd(ts_array, ddof=0)) if len(ts_array) else np.nan,
                    "icir": float(np.nanmean(ts_array) / np.nanstd(ts_array, ddof=0)) if len(ts_array) and np.nanstd(ts_array, ddof=0) > 0 else np.nan,
                    "q25_fold_ic": float(np.nanquantile(fold_array, 0.25)) if len(fold_array) else np.nan,
                    "positive_ts_ic_rate": float(np.nanmean(ts_array > 0.0)) if len(ts_array) else np.nan,
                    "positive_fold_ic_rate": float(np.nanmean(fold_array > 0.0)) if len(fold_array) else np.nan,
                    "fold_count": int(len(fold_array)),
                    "timestamp_count": int(len(ts_array)),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(fold_rows)


def build_oos_rebalance_timestamps(panel: pd.DataFrame, selections: pd.DataFrame, horizon: int) -> dict[int, list[pd.Timestamp]]:
    out: dict[int, list[pd.Timestamp]] = {}
    for _, fold in selections.iterrows():
        mask = (panel["timestamp"] >= fold["test_start"]) & (panel["timestamp"] < fold["test_end"])
        unique_ts = np.array(sorted(panel.loc[mask, "timestamp"].unique()))
        out[int(fold["fold"])] = [pd.Timestamp(x) for x in unique_ts[::horizon]]
    return out


def rank_corr(a: pd.Series, b: pd.Series) -> float:
    x_raw = pd.to_numeric(a, errors="coerce").to_numpy(dtype=float)
    y_raw = pd.to_numeric(b, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x_raw) & np.isfinite(y_raw)
    if int(mask.sum()) < 8:
        return np.nan
    x = pd.Series(x_raw[mask]).rank(method="average").to_numpy(dtype=float)
    y = pd.Series(y_raw[mask]).rank(method="average").to_numpy(dtype=float)
    return safe_corr_arrays(x, y)


def safe_corr_arrays(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 8 or len(y) < 8:
        return np.nan
    x = x.astype(float, copy=False)
    y = y.astype(float, copy=False)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 8:
        return np.nan
    x = x[mask]
    y = y[mask]
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt(np.dot(x, x) * np.dot(y, y)))
    if denom <= 1e-12:
        return np.nan
    return float(np.dot(x, y) / denom)


def build_signal_turnover(
    panel: pd.DataFrame, selections: pd.DataFrame, candidates: list[dict[str, Any]], horizon: int
) -> pd.DataFrame:
    timestamps = []
    for _, fold in selections.iterrows():
        unique_ts = np.array(sorted(panel.loc[(panel["timestamp"] >= fold["test_start"]) & (panel["timestamp"] < fold["test_end"]), "timestamp"].unique()))
        timestamps.extend(list(unique_ts[::horizon]))
    oos_ts = sorted(pd.unique(pd.Series(timestamps)))
    base = panel.loc[panel["timestamp"].isin(oos_ts), ["timestamp", "symbol"]].copy()
    rows = []
    for candidate in candidates:
        score_col = str(candidate["score_col"])
        frame = panel.loc[panel["timestamp"].isin(oos_ts), ["timestamp", "symbol", score_col]].copy()
        frame["rank_centered"] = centered_rank_by_timestamp(frame, score_col)
        frame.sort_values(["symbol", "timestamp"], inplace=True)
        frame["rank_delta"] = frame.groupby("symbol")["rank_centered"].diff().abs()
        frame["rank_autocorr_pair"] = frame.groupby("symbol")["rank_centered"].shift(1)
        pair = frame[["rank_centered", "rank_autocorr_pair"]].replace([np.inf, -np.inf], np.nan).dropna()
        rows.append(
            {
                "name": candidate["name"],
                "score_col": score_col,
                "signal_turnover_72h": float(frame["rank_delta"].mean()),
                "signal_turnover_q75_72h": float(frame["rank_delta"].quantile(0.75)),
                "signal_rank_autocorr_72h": float(pair["rank_centered"].corr(pair["rank_autocorr_pair"])) if len(pair) > 20 else np.nan,
                "oos_rebalance_count": int(len(oos_ts)),
            }
        )
    return pd.DataFrame(rows)


def build_p3d_weight_correlations(panel: pd.DataFrame, candidates: list[dict[str, Any]]) -> pd.DataFrame:
    if not P3D_POSITIONS.exists():
        return pd.DataFrame({"name": [c["name"] for c in candidates], "mean_p3d_weight_corr": np.nan})
    positions = pd.read_csv(P3D_POSITIONS, parse_dates=["timestamp"])
    p3d_ts = sorted(positions["timestamp"].dropna().unique())
    weights = positions[["timestamp", "symbol", "weight"]].copy()
    base = panel.loc[panel["timestamp"].isin(p3d_ts), ["timestamp", "symbol"]].merge(
        weights, on=["timestamp", "symbol"], how="left"
    )
    base["weight"] = base["weight"].fillna(0.0)
    rows = []
    for candidate in candidates:
        score_col = str(candidate["score_col"])
        frame = base.merge(panel.loc[panel["timestamp"].isin(p3d_ts), ["timestamp", "symbol", score_col]], on=["timestamp", "symbol"], how="left")
        frame["rank_centered"] = centered_rank_by_timestamp(frame, score_col)
        per_ts = frame.groupby("timestamp", sort=False).apply(
            lambda g: safe_corr_arrays(g["rank_centered"].to_numpy(dtype=float), g["weight"].to_numpy(dtype=float))
            if g["rank_centered"].notna().sum() >= 8 and g["weight"].abs().sum() > 0
            else np.nan,
            include_groups=False,
        ).dropna()
        rows.append(
            {
                "name": candidate["name"],
                "score_col": score_col,
                "mean_p3d_weight_corr": float(per_ts.mean()) if len(per_ts) else np.nan,
                "median_p3d_weight_corr": float(per_ts.median()) if len(per_ts) else np.nan,
                "mean_abs_p3d_weight_corr": float(per_ts.abs().mean()) if len(per_ts) else np.nan,
                "p3d_corr_timestamp_count": int(len(per_ts)),
            }
        )
    return pd.DataFrame(rows)


def build_candidate_diagnostics(
    candidate_meta: pd.DataFrame,
    ic_summary: pd.DataFrame,
    turnover: pd.DataFrame,
    p3d_corr: pd.DataFrame,
) -> pd.DataFrame:
    primary = ic_summary[ic_summary["label"] == PRIMARY_LABEL].copy()
    primary = primary.rename(
        columns={
            "mean_ic": "primary_mean_ic",
            "median_ic": "primary_median_ic",
            "icir": "primary_icir",
            "q25_fold_ic": "primary_q25_fold_ic",
            "positive_fold_ic_rate": "primary_positive_fold_ic_rate",
            "positive_ts_ic_rate": "primary_positive_ts_ic_rate",
        }
    )
    path_medium = ic_summary[ic_summary["label"] == "label_path_slow_medium"][["name", "mean_ic"]].rename(
        columns={"mean_ic": "path_medium_mean_ic"}
    )
    slow_low = ic_summary[ic_summary["label"] == "label_slow_low_turnover"][["name", "mean_ic"]].rename(
        columns={"mean_ic": "slow_low_turnover_mean_ic"}
    )
    out = (
        candidate_meta.merge(
            primary[
                [
                    "name",
                    "primary_mean_ic",
                    "primary_median_ic",
                    "primary_icir",
                    "primary_q25_fold_ic",
                    "primary_positive_fold_ic_rate",
                    "primary_positive_ts_ic_rate",
                    "fold_count",
                    "timestamp_count",
                ]
            ],
            on="name",
            how="left",
        )
        .merge(path_medium, on="name", how="left")
        .merge(slow_low, on="name", how="left")
        .merge(turnover, on=["name", "score_col"], how="left")
        .merge(p3d_corr, on=["name", "score_col"], how="left")
    )
    out["diagnostic_score"] = (
        100.0 * out["primary_mean_ic"].fillna(0.0)
        + 50.0 * out["primary_q25_fold_ic"].fillna(-0.05)
        + 1.5 * out["primary_positive_fold_ic_rate"].fillna(0.0)
        - 0.60 * out["signal_turnover_72h"].fillna(2.0)
        - 1.25 * out["mean_abs_p3d_weight_corr"].fillna(1.0)
    )
    return out.sort_values(["diagnostic_score", "primary_mean_ic"], ascending=[False, False])


def select_retained_features(
    diagnostics: pd.DataFrame,
    max_abs_p3d_corr: float,
    min_primary_ic: float,
    min_positive_fold_rate: float,
    min_q25_fold_ic: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    turnover_threshold = float(diagnostics["signal_turnover_72h"].quantile(0.60))
    frame = diagnostics.copy()
    frame["retention_turnover_threshold"] = turnover_threshold
    frame["pass_ic"] = frame["primary_mean_ic"] >= min_primary_ic
    frame["pass_stability"] = (frame["primary_positive_fold_ic_rate"] >= min_positive_fold_rate) & (
        frame["primary_q25_fold_ic"] >= min_q25_fold_ic
    )
    frame["pass_turnover"] = frame["signal_turnover_72h"] <= turnover_threshold
    frame["pass_low_corr"] = frame["mean_abs_p3d_weight_corr"].abs() <= max_abs_p3d_corr
    frame["retained"] = frame["pass_ic"] & frame["pass_stability"] & frame["pass_turnover"] & frame["pass_low_corr"]
    retained = frame[frame["retained"]].sort_values(["family", "diagnostic_score"], ascending=[True, False])
    watchlist = frame[
        (~frame["retained"])
        & (frame["primary_mean_ic"] > 0.0)
        & (frame["mean_abs_p3d_weight_corr"].abs() <= max_abs_p3d_corr)
    ].sort_values(["diagnostic_score", "primary_mean_ic"], ascending=[False, False])
    return retained, watchlist


def plot_ic_bar(output_dir: Path, diagnostics: pd.DataFrame) -> None:
    top = diagnostics.sort_values("primary_mean_ic").tail(20)
    plt.figure(figsize=(12, 8))
    colors = np.where(top["primary_mean_ic"] >= 0.0, "#2a9d8f", "#c44536")
    plt.barh(top["name"], top["primary_mean_ic"], color=colors)
    plt.axvline(0.0, color="black", linewidth=0.8, alpha=0.6)
    plt.title("P6 candidate IC vs path slow low-turnover label")
    plt.xlabel("Mean OOS cross-sectional Spearman IC")
    plt.grid(True, axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "top_primary_ic.png", dpi=180)
    plt.close()


def plot_turnover_corr(output_dir: Path, diagnostics: pd.DataFrame, retained: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 7))
    for family, group in diagnostics.groupby("family"):
        plt.scatter(group["signal_turnover_72h"], group["mean_abs_p3d_weight_corr"], label=family, alpha=0.72)
    if not retained.empty:
        plt.scatter(retained["signal_turnover_72h"], retained["mean_abs_p3d_weight_corr"], s=90, facecolors="none", edgecolors="black", linewidths=1.5)
    plt.title("P6 signal turnover vs correlation to P3d weights")
    plt.xlabel("Mean 72h rank turnover")
    plt.ylabel("Mean abs correlation to P3d weights")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "turnover_vs_p3d_corr.png", dpi=180)
    plt.close()


def plot_family_counts(output_dir: Path, diagnostics: pd.DataFrame, retained: pd.DataFrame) -> None:
    counts = diagnostics.groupby("family").size().rename("candidate_count").to_frame()
    retained_counts = retained.groupby("family").size().rename("retained_count").to_frame()
    frame = counts.merge(retained_counts, left_index=True, right_index=True, how="left").fillna(0).reset_index()
    frame.sort_values("family", inplace=True)
    x = np.arange(len(frame))
    width = 0.38
    plt.figure(figsize=(11, 6))
    plt.bar(x - width / 2, frame["candidate_count"], width, label="candidates", color="#457b9d")
    plt.bar(x + width / 2, frame["retained_count"], width, label="retained", color="#2a9d8f")
    plt.xticks(x, frame["family"], rotation=25, ha="right")
    plt.title("P6 candidate and retained count by family")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "family_retention_counts.png", dpi=180)
    plt.close()


def write_report(output_dir: Path, diagnostics: pd.DataFrame, retained: pd.DataFrame, watchlist: pd.DataFrame, args: argparse.Namespace) -> None:
    top_cols = [
        "name",
        "family",
        "primary_mean_ic",
        "primary_q25_fold_ic",
        "primary_positive_fold_ic_rate",
        "signal_turnover_72h",
        "mean_abs_p3d_weight_corr",
        "diagnostic_score",
        "formula",
    ]
    lines = [
        "# P6 Feature Expansion",
        "",
        "## Design",
        "",
        "- Built OHLCV-only, ex-ante C/D/E/F/G feature candidates.",
        "- Did not run AlphaGPT or portfolio allocation in this stage.",
        "- Diagnostics use frozen walk-forward test windows sampled at 72h rebalance decision points.",
        f"- Primary retention label: `{PRIMARY_LABEL}`.",
        "- Retention requires positive path-aware IC, fold stability, low 72h signal turnover, and low correlation to P3d weights.",
        "",
        "## Retention Rules",
        "",
        "```text",
        f"primary_mean_ic >= {args.min_primary_ic}",
        f"primary_positive_fold_ic_rate >= {args.min_positive_fold_rate}",
        f"primary_q25_fold_ic >= {args.min_q25_fold_ic}",
        f"signal_turnover_72h <= universe 60th percentile",
        f"mean_abs_p3d_weight_corr <= {args.max_abs_p3d_corr}",
        "```",
        "",
        "## Retained Features",
        "",
        "```text",
        retained[top_cols].to_string(index=False) if not retained.empty else "None",
        "```",
        "",
        "## Top Diagnostics",
        "",
        "```text",
        diagnostics.sort_values(["diagnostic_score", "primary_mean_ic"], ascending=[False, False]).head(20)[top_cols].to_string(index=False),
        "```",
        "",
        "## Watchlist",
        "",
        "```text",
        watchlist.head(20)[top_cols].to_string(index=False) if not watchlist.empty else "None",
        "```",
        "",
        "## Files",
        "",
        "- `p6_feature_panel.csv.gz`: C/D/E/F/G feature panel.",
        "- `candidate_formulas.csv`: feature definitions.",
        "- `ic_summary.csv`, `fold_ic.csv`: walk-forward IC diagnostics.",
        "- `candidate_diagnostics.csv`: merged IC, turnover, P3d-correlation diagnostics.",
        "- `retained_features.csv`: features retained for the next P3d-gated search.",
        "- `watchlist_features.csv`: positive but not fully retained features.",
        "- plots: `top_primary_ic.png`, `turnover_vs_p3d_corr.png`, `family_retention_counts.png`.",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def update_config_stage(config_path: Path) -> None:
    text = config_path.read_text(encoding="utf-8")
    if "p6_feature_expansion:" in text:
        return
    marker = "  p3d_p4_gated_rerank_alphagpt:\n"
    insert = """  p6_feature_expansion:\n    script: ../../../scripts/run_p6_feature_expansion.py\n    output_dir: ../outputs/p6_feature_expansion\n    report: ../outputs/p6_feature_expansion/REPORT.md\n    feature_panel: ../outputs/p6_feature_expansion/p6_feature_panel.csv.gz\n    candidate_formulas: ../outputs/p6_feature_expansion/candidate_formulas.csv\n    ic_summary: ../outputs/p6_feature_expansion/ic_summary.csv\n    candidate_diagnostics: ../outputs/p6_feature_expansion/candidate_diagnostics.csv\n    retained_features: ../outputs/p6_feature_expansion/retained_features.csv\n\n"""
    stages_marker = "stages:\n"
    if stages_marker in text and marker in text:
        text = text.replace(marker, insert + "\n" + marker, 1)
        config_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
