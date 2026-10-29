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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from crypto_stat_arb.config import ensure_dir
from run_p3d_p4_gated_rerank_alphagpt import decision_end_for_closed_positions, load_panel as load_p3d_panel
from run_p18_factor_decay_walkforward import (
    DEFAULT_ASSET_PANEL,
    DEFAULT_DERIVATIVES_PANEL,
    DEFAULT_MANUAL_PANEL,
    DEFAULT_P6B_PANEL,
    EXPERIMENT_ROOT,
    centered_rank_matrix,
    factor_persistence_by_matrix,
    load_factor_library,
    rowwise_corr,
    safe_mean,
    safe_positive_rate,
    safe_quantile,
    split_first_second,
)
from run_p18d_guarded_factor_selector import LONG_ROOT, add_regime_features
from run_p20_selection_protocol import ProtocolSpec, build_protocols
from run_turnover_experiments import to_jsonable


DEFAULT_CONFIG = EXPERIMENT_ROOT / "configs" / "baseline_locked_v1.yaml"
DEFAULT_FOLDS = LONG_ROOT / "inputs" / "long_formula_selections.csv"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "outputs" / "p21_gross_alpha_family_search"


@dataclass(frozen=True)
class Component:
    factor: str
    weight: float = 1.0


@dataclass(frozen=True)
class Family:
    name: str
    description: str
    components: tuple[Component, ...]


def main() -> None:
    parser = argparse.ArgumentParser(description="P21: search new gross alpha families before allocator integration.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--asset-panel", default=str(DEFAULT_ASSET_PANEL))
    parser.add_argument("--manual-panel", default=str(DEFAULT_MANUAL_PANEL))
    parser.add_argument("--p6b-panel", default=str(DEFAULT_P6B_PANEL))
    parser.add_argument("--derivatives-panel", default=str(DEFAULT_DERIVATIVES_PANEL))
    parser.add_argument("--folds", default=str(DEFAULT_FOLDS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--horizon", type=int, default=72)
    parser.add_argument("--validation-days", type=int, default=90)
    parser.add_argument("--min-symbols", type=int, default=8)
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    horizon = int(args.horizon)
    validation_days = int(args.validation_days)
    min_symbols = int(args.min_symbols)

    print("[1/7] Loading factor library and folds", flush=True)
    factor_panel, factor_meta = load_factor_library(
        asset_panel=Path(args.asset_panel),
        manual_panel=Path(args.manual_panel),
        p6b_panel=Path(args.p6b_panel),
        derivatives_panel=Path(args.derivatives_panel),
        max_factors=0,
    )
    folds = pd.read_csv(args.folds, parse_dates=["train_start", "train_end", "test_start", "test_end"]).sort_values("fold")
    label_rank = centered_rank_matrix(factor_panel, f"label_resid_ret_h{horizon}", min_symbols=min_symbols)

    print("[2/7] Building P21 family definitions", flush=True)
    families = resolve_families(build_families(), set(factor_meta["factor"]))
    family_meta = pd.DataFrame(
        [
            {
                "family": family.name,
                "description": family.description,
                "component_count": len(family.components),
                "components": "|".join(component.factor for component in family.components),
            }
            for family in families
        ]
    )
    family_meta.to_csv(output_dir / "p21_family_library.csv", index=False)
    if not families:
        raise RuntimeError("No P21 family definitions resolved to available factors.")
    print(f"      families={len(families)}", flush=True)

    print("[3/7] Precomputing component rank matrices", flush=True)
    needed = sorted({component.factor for family in families for component in family.components})
    rank_matrices = {factor: centered_rank_matrix(factor_panel, factor, min_symbols=min_symbols) for factor in needed}

    print("[4/7] Loading ex-ante regime state", flush=True)
    regime_panel = load_p3d_panel(Path(args.config), Path(args.asset_panel), Path(args.manual_panel), horizon)
    add_regime_features(regime_panel)
    regime = (
        regime_panel[["timestamp", "_p18d_high_risk_regime"]]
        .drop_duplicates("timestamp")
        .sort_values("timestamp")
        .set_index("timestamp")["_p18d_high_risk_regime"]
        .astype(bool)
    )

    print("[5/7] Running fold-local family diagnostics", flush=True)
    fold_metrics, family_score_ts = run_family_diagnostics(
        families=families,
        folds=folds,
        rank_matrices=rank_matrices,
        label_rank=label_rank,
        regime=regime,
        horizon=horizon,
        validation_days=validation_days,
        min_symbols=min_symbols,
    )
    fold_metrics.to_csv(output_dir / "p21_family_fold_metrics.csv", index=False)
    family_score_ts.to_csv(output_dir / "p21_family_score_timeseries.csv.gz", index=False)

    print("[6/7] Applying P20-style protocols and selecting fold-local candidates", flush=True)
    protocols = build_protocols()
    summary = summarize_families(fold_metrics, protocols)
    summary.to_csv(output_dir / "p21_family_summary.csv", index=False)
    selected = select_family_by_fold(fold_metrics, protocols)
    selected.to_csv(output_dir / "p21_selected_family_by_fold.csv", index=False)
    selected_summary = summarize_selected(selected)
    selected_summary.to_csv(output_dir / "p21_selected_protocol_summary.csv", index=False)

    print("[7/7] Writing plots and report", flush=True)
    plot_outputs(output_dir, fold_metrics, summary, selected)
    write_report(output_dir, family_meta, summary, selected_summary, selected)
    metadata = {
        "horizon": horizon,
        "validation_days": validation_days,
        "min_symbols": min_symbols,
        "note": "P21 evaluates gross alpha families only. Allocator integration is intentionally out of scope.",
    }
    (output_dir / "metadata.json").write_text(json.dumps(to_jsonable(metadata), indent=2), encoding="utf-8")
    print(f"Done: {output_dir}", flush=True)


def build_families() -> list[Family]:
    return [
        Family(
            name="p21_low_vol_residual_quality",
            description="Slow residual low-volatility / low idiosyncratic risk quality family.",
            components=components(
                "base__realized_vol_long",
                "base__realized_vol_medium",
                "base__idio_vol_long",
                "base__idio_vol_medium",
                "p6b__cs_vol_of_vol_336",
                "slow__cs_vol_of_vol_res_168",
            ),
        ),
        Family(
            name="p21_deriv_crowding_contra_slow",
            description="Crowding and top-trader positioning contra family.",
            components=components(
                "deriv__deriv_crowding_pressure_72",
                "deriv__deriv_top_sum_long_short_ratio",
                "deriv__deriv_top_sum_long_short_ratio_ewm_72",
                "deriv__deriv_top_sum_long_short_ratio_ewm_168",
                "p6b__p6b_J04_top_trader_contra",
                "p6b__p6b_K01_deriv_crowding_contra",
                "p6b__p6b_J03_account_longs_contra",
            ),
        ),
        Family(
            name="p21_funding_basis_reversion",
            description="Funding, premium, and basis mean-reversion / carry-dislocation family.",
            components=components(
                "deriv__deriv_funding_rate_ewm_24",
                "deriv__deriv_funding_rate_ewm_72",
                "deriv__deriv_funding_rate_ewm_168",
                "deriv__deriv_funding_annualized_ewm_72",
                "deriv__deriv_mark_index_basis_ewm_168",
                "deriv__deriv_premium_index_ewm_168",
                "p6b__p6b_H01_funding_contra_168",
                "p6b__p6b_I01_premium_contra_168",
                "p6b__p6b_I02_basis_contra_168",
            ),
        ),
        Family(
            name="p21_oi_stock_unwind",
            description="Open-interest stock and OI-growth unwind family.",
            components=components(
                "deriv__deriv_open_interest",
                "deriv__deriv_log_oi_value",
                "deriv__deriv_log_oi_value_ewm_168",
                "deriv__deriv_log_oi_value_ewm_336",
                "deriv__deriv_oi_growth_72_336",
                "p6b__p6b_J01_oi_growth_mom_quality",
                "p6b__p6b_J02_oi_growth_crowding_fade",
            ),
        ),
        Family(
            name="p21_volume_flow_reversal",
            description="Slow volume/taker flow exhaustion and liquidity-conditioned reversal family.",
            components=components(
                "base__volume",
                "base__quote_volume",
                "base__log_quote_volume",
                "base__taker_buy_base",
                "base__taker_buy_quote",
                "base__trades",
                "p6b__p6_C01_liq_persistent_mom",
                "p6b__p6_C02_dollar_volume_mom",
                "p6b__p6_C03_amihud_quality",
                "slow__cs_liquidity_quality",
            ),
        ),
        Family(
            name="p21_uncrowded_residual_trend",
            description="Residual trend continuation only when liquidity/crowding quality is supportive.",
            components=components(
                "slow__cs_res_mom_168_24",
                "slow__cs_res_mom_336_24",
                "slow__cs_res_trend_quality_168_24",
                "slow__cs_res_ma_slope_48_168",
                "p6b__p6_E01_uncrowded_mom",
                "p6b__p6_E02_purity_mom",
                "p6b__p6b_K02_uncrowded_slow_mom",
                "p6b__p6_D04_low_vov_momentum",
            ),
        ),
        Family(
            name="p21_compression_breakout_quality",
            description="Volatility/range compression with breakout-risk quality control.",
            components=components(
                "p6b__p6_D01_compression_mom",
                "p6b__p6_D02_deep_compression_mom",
                "p6b__p6_D03_range_compression_quality",
                "p6b__cs_vol_compression_72_336",
                "p6b__cs_vol_compression_168_504",
                "p6b__cs_range_compression_72_336",
                "p6b__cs_breakout_risk_72",
            ),
        ),
        Family(
            name="p21_defensive_regime_quality",
            description="High-volatility defensive quality and low-volatility reversion regime family.",
            components=components(
                "p6b__p6_F01_normal_regime_mom",
                "p6b__p6_F04_low_vol_reversion",
                "p6b__p6_F05_high_vol_defensive_quality",
                "p6b__p6_G05_stable_low_turnover_meta",
                "slow__family_b_regime_gate",
                "slow__stationarity_gate_168",
                "slow__stationarity_gate_336",
            ),
        ),
        Family(
            name="p21_beta_stability_purity",
            description="Stable beta, idiosyncratic purity, and basket-diversifier family.",
            components=components(
                "slow__pct_beta_stability_168",
                "slow__pct_beta_stability_336",
                "slow__cs_beta_stability_168",
                "slow__cs_beta_stability_336",
                "p6b__cs_idiosyncratic_purity_336",
                "p6b__p6_E03_basket_diversifier_mom",
                "p6b__cs_resid_basket_diversifier_336",
            ),
        ),
        Family(
            name="p21_lowrisk_carry_crowding_combo",
            description="Composite of low residual risk, derivatives crowding contra, and funding/basis dislocation.",
            components=components(
                "base__realized_vol_long",
                "base__idio_vol_long",
                "deriv__deriv_crowding_pressure_72",
                "p6b__p6b_J04_top_trader_contra",
                "deriv__deriv_funding_rate_ewm_72",
                "deriv__deriv_top_sum_long_short_ratio_ewm_168",
                "p6b__p6b_K01_deriv_crowding_contra",
            ),
        ),
    ]


def components(*names: str) -> tuple[Component, ...]:
    return tuple(Component(name, 1.0) for name in names)


def resolve_families(families: list[Family], available: set[str]) -> list[Family]:
    resolved: list[Family] = []
    for family in families:
        comps = tuple(component for component in family.components if component.factor in available)
        if len(comps) >= 3:
            resolved.append(Family(family.name, family.description, comps))
    return resolved


def run_family_diagnostics(
    families: list[Family],
    folds: pd.DataFrame,
    rank_matrices: dict[str, pd.DataFrame],
    label_rank: pd.DataFrame,
    regime: pd.Series,
    horizon: int,
    validation_days: int,
    min_symbols: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    score_rows: list[pd.DataFrame] = []
    total = len(families) * len(folds)
    done = 0
    for family in families:
        for _, fold in folds.iterrows():
            done += 1
            if done == 1 or done % 40 == 0 or done == total:
                print(f"      family-fold {done}/{total}: {family.name} fold {int(fold['fold'])}", flush=True)
            row, scores = build_family_fold_row(
                family=family,
                fold=fold,
                rank_matrices=rank_matrices,
                label_rank=label_rank,
                regime=regime,
                horizon=horizon,
                validation_days=validation_days,
                min_symbols=min_symbols,
            )
            metric_rows.append(row)
            if not scores.empty:
                score_rows.append(scores)
    metrics = pd.DataFrame(metric_rows)
    scores = pd.concat(score_rows, ignore_index=True) if score_rows else pd.DataFrame()
    return metrics, scores


def build_family_fold_row(
    family: Family,
    fold: pd.Series,
    rank_matrices: dict[str, pd.DataFrame],
    label_rank: pd.DataFrame,
    regime: pd.Series,
    horizon: int,
    validation_days: int,
    min_symbols: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    fold_id = int(fold["fold"])
    train_start = pd.Timestamp(fold["train_start"])
    validation_start, validation_end = validation_window(fold, validation_days, horizon)
    test_start = pd.Timestamp(fold["test_start"])
    test_end = decision_end_for_closed_positions(pd.Timestamp(fold["test_end"]), horizon)
    component_orientations: dict[str, int] = {}
    oriented_components: list[pd.DataFrame] = []
    usable_components: list[str] = []
    for component in family.components:
        matrix = rank_matrices[component.factor]
        train_ic = rowwise_corr(
            matrix.loc[train_start:validation_start],
            label_rank.loc[train_start:validation_start],
            min_symbols=min_symbols,
        )
        sign = orient_sign(train_ic)
        component_orientations[component.factor] = sign
        oriented_components.append(matrix * sign * float(component.weight))
        usable_components.append(component.factor)
    score = sum(oriented_components) / max(sum(abs(component.weight) for component in family.components), 1e-12)
    val_ic = family_ic_timeseries(score, label_rank, validation_start, validation_end, min_symbols)
    test_ic = family_ic_timeseries(score, label_rank, test_start, test_end, min_symbols)
    val_first, val_second = split_first_second(val_ic)
    val_spread = family_spread_timeseries(score, label_rank, validation_start, validation_end)
    test_spread = family_spread_timeseries(score, label_rank, test_start, test_end)
    turnover_1h, turnover_72h, autocorr_72h = factor_persistence_by_matrix(score)
    val_turn1 = turnover_1h[(turnover_1h.index >= validation_start) & (turnover_1h.index <= validation_end)]
    val_turn72 = turnover_72h[(turnover_72h.index >= validation_start) & (turnover_72h.index <= validation_end)]
    val_ac72 = autocorr_72h[(autocorr_72h.index >= validation_start) & (autocorr_72h.index <= validation_end)]
    validation_high, recent_high, current_high = regime_coverage(regime, validation_start, validation_end, pd.Timestamp(fold["train_end"]))
    component_corr = mean_component_corr(oriented_components, validation_start, validation_end)
    score_ts = score.loc[test_start:test_end].stack().dropna().rename("score").reset_index()
    score_ts["family"] = family.name
    score_ts["fold"] = fold_id
    row = {
        "family": family.name,
        "description": family.description,
        "fold": fold_id,
        "test_start": test_start,
        "component_count": len(usable_components),
        "components": "|".join(usable_components),
        "component_orientations": json.dumps(component_orientations, sort_keys=True),
        "validation_start": validation_start,
        "validation_decision_end": validation_end,
        "validation_ic_mean": safe_mean(val_ic),
        "validation_ic_q25": safe_quantile(val_ic, 0.25),
        "validation_ic_median": safe_quantile(val_ic, 0.50),
        "validation_positive_ts_rate": safe_positive_rate(val_ic),
        "validation_first_half_ic": safe_mean(val_first),
        "validation_second_half_ic": safe_mean(val_second),
        "validation_decay_delta": safe_mean(val_second) - safe_mean(val_first),
        "validation_top_bottom_spread": safe_mean(val_spread),
        "validation_turnover_1h": safe_mean(val_turn1),
        "validation_turnover_72h": safe_mean(val_turn72),
        "validation_autocorr_72h": safe_mean(val_ac72),
        "validation_high_risk_rate": validation_high,
        "recent_high_risk_rate": recent_high,
        "current_high_risk": current_high,
        "mean_abs_component_corr": component_corr,
        "test_ic_mean": safe_mean(test_ic),
        "test_ic_q25": safe_quantile(test_ic, 0.25),
        "test_positive_ts_rate": safe_positive_rate(test_ic),
        "test_top_bottom_spread": safe_mean(test_spread),
    }
    for protocol in build_protocols():
        row[f"pass_{protocol.name}"] = protocol_pass(row, protocol)
        row[f"score_{protocol.name}"] = protocol_score(row, protocol)
    return row, score_ts


def validation_window(fold: pd.Series, validation_days: int, horizon: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    validation_start = max(pd.Timestamp(fold["train_start"]), pd.Timestamp(fold["train_end"]) - pd.Timedelta(days=validation_days))
    validation_end = min(
        decision_end_for_closed_positions(validation_start + pd.Timedelta(days=validation_days), horizon),
        pd.Timestamp(fold["train_end"]),
    )
    return validation_start, validation_end


def orient_sign(train_ic: pd.Series) -> int:
    mean_ic = safe_mean(train_ic)
    if not np.isfinite(mean_ic) or abs(mean_ic) < 1e-12:
        return 1
    return 1 if mean_ic >= 0 else -1


def family_ic_timeseries(score: pd.DataFrame, label_rank: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, min_symbols: int) -> pd.Series:
    idx = score.loc[start:end].index
    if len(idx) == 0:
        return pd.Series(dtype=float)
    return rowwise_corr(score.loc[idx], label_rank.loc[idx], min_symbols=min_symbols)


def family_spread_timeseries(score: pd.DataFrame, label_rank: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    score_window, label_window = score.loc[start:end].align(label_rank.loc[start:end], join="inner", axis=0)
    score_window, label_window = score_window.align(label_window, join="inner", axis=1)
    rows: list[tuple[pd.Timestamp, float]] = []
    for ts in score_window.index:
        s = score_window.loc[ts]
        y = label_window.loc[ts]
        valid = s.notna() & y.notna()
        if valid.sum() < 8:
            continue
        sv = s[valid]
        yv = y[valid]
        high = sv >= sv.quantile(0.80)
        low = sv <= sv.quantile(0.20)
        if high.sum() == 0 or low.sum() == 0:
            continue
        rows.append((pd.Timestamp(ts), float(yv[high].mean() - yv[low].mean())))
    return pd.Series({ts: value for ts, value in rows}, dtype=float)


def regime_coverage(
    regime: pd.Series,
    validation_start: pd.Timestamp,
    validation_end: pd.Timestamp,
    train_end: pd.Timestamp,
    recent_days: int = 14,
) -> tuple[float, float, bool]:
    validation = regime[(regime.index >= validation_start) & (regime.index <= validation_end)]
    recent_start = train_end - pd.Timedelta(days=recent_days)
    recent = regime[(regime.index >= recent_start) & (regime.index <= train_end)]
    current = regime[regime.index <= train_end].tail(1)
    return (
        float(validation.mean()) if len(validation) else np.nan,
        float(recent.mean()) if len(recent) else np.nan,
        bool(current.iloc[0]) if len(current) else False,
    )


def mean_component_corr(components: list[pd.DataFrame], start: pd.Timestamp, end: pd.Timestamp) -> float:
    if len(components) < 2:
        return 0.0
    values: list[float] = []
    for i, left in enumerate(components):
        for right in components[i + 1 :]:
            x, y = left.loc[start:end].align(right.loc[start:end], join="inner", axis=0)
            x, y = x.align(y, join="inner", axis=1)
            xv = x.to_numpy(dtype=float).ravel()
            yv = y.to_numpy(dtype=float).ravel()
            valid = np.isfinite(xv) & np.isfinite(yv)
            if valid.sum() < 30:
                continue
            corr = np.corrcoef(xv[valid], yv[valid])[0, 1]
            if np.isfinite(corr):
                values.append(abs(float(corr)))
    return float(np.mean(values)) if values else np.nan


def protocol_pass(row: dict[str, Any], protocol: ProtocolSpec) -> bool:
    mean_ic = float(row["validation_ic_mean"])
    q25 = float(row["validation_ic_q25"])
    second = float(row["validation_second_half_ic"])
    decay = float(row["validation_decay_delta"])
    pos = float(row["validation_positive_ts_rate"])
    corr = float(row["mean_abs_component_corr"]) if np.isfinite(row["mean_abs_component_corr"]) else 0.0
    validation_high = float(row["validation_high_risk_rate"])
    recent_high = float(row["recent_high_risk_rate"])
    current_high = bool(row["current_high_risk"])
    quality = (
        np.isfinite(mean_ic)
        and mean_ic > max(0.01, protocol.min_second_half_ic)
        and np.isfinite(q25)
        and q25 >= protocol.q25_floor
        and np.isfinite(second)
        and second >= protocol.min_second_half_ic
        and np.isfinite(decay)
        and decay >= protocol.min_decay_delta
        and np.isfinite(pos)
        and pos >= 0.55
        and corr <= protocol.max_abs_pair_corr
    )
    regime_supported = True
    if current_high and recent_high >= protocol.recent_high_risk_trigger:
        regime_supported = validation_high >= protocol.min_validation_high_risk_rate
    return bool(quality and regime_supported)


def protocol_score(row: dict[str, Any], protocol: ProtocolSpec) -> float:
    if not protocol_pass(row, protocol):
        return -np.inf
    return float(
        100.0 * row["validation_ic_mean"]
        + 40.0 * row["validation_ic_q25"]
        + 25.0 * row["validation_second_half_ic"]
        + 10.0 * row["validation_positive_ts_rate"]
        + 2.0 * row["validation_autocorr_72h"]
        - 1.5 * row["validation_turnover_72h"]
        - 2.0 * row["mean_abs_component_corr"]
    )


def summarize_families(fold_metrics: pd.DataFrame, protocols: list[ProtocolSpec]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family, group in fold_metrics.groupby("family", sort=False):
        clean = group.copy()
        row: dict[str, Any] = {
            "family": family,
            "description": clean["description"].iloc[0],
            "fold_count": int(len(clean)),
            "validation_ic_mean": clean["validation_ic_mean"].mean(),
            "validation_ic_q25_fold": clean["validation_ic_mean"].quantile(0.25),
            "validation_ic_q25_ts_mean": clean["validation_ic_q25"].mean(),
            "validation_positive_fold_rate": float((clean["validation_ic_mean"] > 0).mean()),
            "validation_positive_ts_rate_mean": clean["validation_positive_ts_rate"].mean(),
            "validation_second_half_ic_mean": clean["validation_second_half_ic"].mean(),
            "validation_decay_delta_mean": clean["validation_decay_delta"].mean(),
            "validation_turnover_72h_mean": clean["validation_turnover_72h"].mean(),
            "validation_autocorr_72h_mean": clean["validation_autocorr_72h"].mean(),
            "mean_abs_component_corr": clean["mean_abs_component_corr"].mean(),
            "test_ic_mean": clean["test_ic_mean"].mean(),
            "test_ic_q25_fold": clean["test_ic_mean"].quantile(0.25),
            "test_positive_fold_rate": float((clean["test_ic_mean"] > 0).mean()),
            "test_top_bottom_spread_mean": clean["test_top_bottom_spread"].mean(),
        }
        for protocol in protocols:
            passed = clean[f"pass_{protocol.name}"].astype(bool)
            row[f"{protocol.name}_pass_fold_rate"] = float(passed.mean())
            row[f"{protocol.name}_pass_count"] = int(passed.sum())
            if passed.any():
                row[f"{protocol.name}_oos_test_ic_mean_when_passed"] = clean.loc[passed, "test_ic_mean"].mean()
                row[f"{protocol.name}_oos_test_positive_rate_when_passed"] = float((clean.loc[passed, "test_ic_mean"] > 0).mean())
            else:
                row[f"{protocol.name}_oos_test_ic_mean_when_passed"] = np.nan
                row[f"{protocol.name}_oos_test_positive_rate_when_passed"] = np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["research_score"] = (
        100.0 * out["validation_ic_mean"]
        + 50.0 * out["validation_ic_q25_fold"]
        + 30.0 * out["test_ic_mean"]
        + 5.0 * out["test_positive_fold_rate"]
        - 2.0 * out["mean_abs_component_corr"]
        - 0.5 * out["validation_turnover_72h_mean"]
    )
    return out.sort_values("research_score", ascending=False)


def select_family_by_fold(fold_metrics: pd.DataFrame, protocols: list[ProtocolSpec]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for protocol in protocols:
        score_col = f"score_{protocol.name}"
        pass_col = f"pass_{protocol.name}"
        for fold, group in fold_metrics.groupby("fold", sort=True):
            passed = group[group[pass_col].astype(bool)].copy()
            if passed.empty:
                rows.append(
                    {
                        "protocol": protocol.name,
                        "fold": int(fold),
                        "selected_family": "__NO_P21_ALPHA__",
                        "fallback": True,
                        "validation_score": np.nan,
                        "validation_ic_mean": np.nan,
                        "test_ic_mean": np.nan,
                        "test_top_bottom_spread": np.nan,
                    }
                )
                continue
            winner = passed.sort_values(score_col, ascending=False).iloc[0]
            rows.append(
                {
                    "protocol": protocol.name,
                    "fold": int(fold),
                    "selected_family": winner["family"],
                    "fallback": False,
                    "validation_score": float(winner[score_col]),
                    "validation_ic_mean": float(winner["validation_ic_mean"]),
                    "validation_ic_q25": float(winner["validation_ic_q25"]),
                    "validation_second_half_ic": float(winner["validation_second_half_ic"]),
                    "validation_decay_delta": float(winner["validation_decay_delta"]),
                    "test_ic_mean": float(winner["test_ic_mean"]),
                    "test_ic_q25": float(winner["test_ic_q25"]),
                    "test_positive_ts_rate": float(winner["test_positive_ts_rate"]),
                    "test_top_bottom_spread": float(winner["test_top_bottom_spread"]),
                    "components": winner["components"],
                }
            )
    return pd.DataFrame(rows)


def summarize_selected(selected: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for protocol, group in selected.groupby("protocol", sort=False):
        active = group[~group["fallback"].astype(bool)].copy()
        rows.append(
            {
                "protocol": protocol,
                "active_fold_count": int(len(active)),
                "active_fold_rate": float(len(active) / len(group)) if len(group) else np.nan,
                "selected_test_ic_mean": active["test_ic_mean"].mean() if len(active) else np.nan,
                "selected_test_ic_q25_fold": active["test_ic_mean"].quantile(0.25) if len(active) else np.nan,
                "selected_test_positive_fold_rate": float((active["test_ic_mean"] > 0).mean()) if len(active) else np.nan,
                "selected_test_top_bottom_spread_mean": active["test_top_bottom_spread"].mean() if len(active) else np.nan,
                "most_common_family": active["selected_family"].mode().iloc[0] if len(active) else "__NO_P21_ALPHA__",
            }
        )
    return pd.DataFrame(rows)


def plot_outputs(output_dir: Path, fold_metrics: pd.DataFrame, summary: pd.DataFrame, selected: pd.DataFrame) -> None:
    top = summary.head(12).copy()
    if not top.empty:
        fig, ax = plt.subplots(figsize=(13, 6))
        ax.barh(top["family"], top["test_ic_mean"])
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.invert_yaxis()
        ax.set_title("P21 family OOS test IC mean")
        ax.set_xlabel("Test IC mean")
        fig.tight_layout()
        fig.savefig(output_dir / "p21_family_test_ic.png", dpi=160)
        plt.close(fig)

    pivot = fold_metrics.pivot_table(index="fold", columns="family", values="test_ic_mean", aggfunc="mean")
    if not pivot.empty:
        ax = pivot.rolling(3, min_periods=1).mean().plot(figsize=(15, 8), linewidth=1.4)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title("P21 family rolling OOS test IC by fold")
        ax.set_xlabel("Fold")
        ax.set_ylabel("Test IC")
        ax.legend(fontsize=7, ncol=2)
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig(output_dir / "p21_family_rolling_test_ic.png", dpi=160)
        plt.close(fig)

    selected_active = selected[~selected["fallback"].astype(bool)].copy()
    if not selected_active.empty:
        pivot_sel = selected_active.pivot_table(index="fold", columns="protocol", values="test_ic_mean", aggfunc="mean")
        ax = pivot_sel.plot(kind="bar", figsize=(14, 6))
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title("P21 protocol-selected family OOS test IC")
        ax.set_xlabel("Fold")
        ax.set_ylabel("Test IC")
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig(output_dir / "p21_selected_family_test_ic.png", dpi=160)
        plt.close(fig)


def write_report(
    output_dir: Path,
    family_meta: pd.DataFrame,
    summary: pd.DataFrame,
    selected_summary: pd.DataFrame,
    selected: pd.DataFrame,
) -> None:
    display_cols = [
        "family",
        "validation_ic_mean",
        "validation_ic_q25_fold",
        "validation_ic_q25_ts_mean",
        "validation_positive_fold_rate",
        "validation_turnover_72h_mean",
        "validation_autocorr_72h_mean",
        "mean_abs_component_corr",
        "test_ic_mean",
        "test_ic_q25_fold",
        "test_positive_fold_rate",
        "p20_practical_q25_m5_pass_count",
        "p20_loose_decay_corr_pass_count",
        "research_score",
    ]
    report = [
        "# P21 Gross Alpha Family Search",
        "",
        "P21 searches new gross alpha families before allocator integration. It uses only ex-ante data and fold-local train orientation; validation applies P20-style quality, decay, homogeneity, and regime coverage gates. Test IC is reported only as OOS diagnostics.",
        "",
        f"- family count: {len(family_meta)}",
        "",
        "## Family Summary",
        "",
        "```text",
        summary[display_cols].to_string(index=False) if not summary.empty else "None",
        "```",
        "",
        "## Protocol-Selected Families",
        "",
        "```text",
        selected_summary.to_string(index=False) if not selected_summary.empty else "None",
        "```",
        "",
        "## Selection Detail",
        "",
        "```text",
        selected.to_string(index=False) if not selected.empty else "None",
        "```",
        "",
        "## Family Library",
        "",
        "```text",
        family_meta.to_string(index=False),
        "```",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
