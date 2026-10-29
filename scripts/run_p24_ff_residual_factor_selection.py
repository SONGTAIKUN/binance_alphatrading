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
from matplotlib.ticker import PercentFormatter

warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from crypto_stat_arb.config import ensure_dir, load_config
from crypto_stat_arb.features import rolling_zscore
from run_p18_factor_decay_walkforward import centered_rank_matrix
from run_p18c_factor_selector_portfolio import (
    add_p18c_entry_fields,
    build_asset_contribution,
    build_cost_sensitivity,
    build_segment_comparison,
    load_existing_result,
    write_run_result,
)
from run_p18d_guarded_factor_selector import LONG_ROOT, add_regime_features
from run_p20_selection_protocol import build_protocols
from run_p21_gross_alpha_family_search import (
    Component,
    Family,
    resolve_families,
    run_family_diagnostics,
    select_family_by_fold,
    summarize_families,
    summarize_selected,
)
from run_p22_p21_family_portfolio import build_fold_delta, replay_protocol
from run_p3d_p4_gated_rerank_alphagpt import load_panel as load_p3d_panel, make_portfolio_config, make_risk_template
from run_turnover_experiments import to_jsonable


EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "baseline_locked_v1"
DEFAULT_CONFIG = EXPERIMENT_ROOT / "configs" / "baseline_locked_v1.yaml"
DEFAULT_ASSET_PANEL = EXPERIMENT_ROOT / "inputs" / "long_history" / "baseline_panel" / "asset_residual_panel.csv"
DEFAULT_MANUAL_PANEL = LONG_ROOT / "inputs" / "manual_slow_feature_panel_long.csv.gz"
DEFAULT_FOLDS = LONG_ROOT / "inputs" / "long_formula_selections.csv"
DEFAULT_P23_PANEL = EXPERIMENT_ROOT / "outputs" / "p23_crypto_ff_residualizer" / "crypto_ff_residual_panel.csv.gz"
DEFAULT_P15C_RUN = LONG_ROOT / "p15c_nested" / "nested_extreme_cluster_run"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "outputs" / "p24_ff_residual_factor_selection"
P24_SCORE = "_p24_ff_family_score"
LABEL_HORIZONS = [48, 72, 96, 168]
DEFAULT_PROTOCOLS = ["p20_strict_q25_nonneg", "p20_balanced_q25_m3", "p20_practical_q25_m5", "p20_loose_decay_corr"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="P24: feed P23 FF residual features into P20/P21 selection and strict-neutral allocator replay."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--asset-panel", default=str(DEFAULT_ASSET_PANEL))
    parser.add_argument("--manual-panel", default=str(DEFAULT_MANUAL_PANEL))
    parser.add_argument("--ff-panel", default=str(DEFAULT_P23_PANEL))
    parser.add_argument("--folds", default=str(DEFAULT_FOLDS))
    parser.add_argument("--p15c-run-dir", default=str(DEFAULT_P15C_RUN))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--horizon", type=int, default=72)
    parser.add_argument("--validation-days", type=int, default=90)
    parser.add_argument("--min-symbols", type=int, default=8)
    parser.add_argument("--protocols", nargs="+", default=DEFAULT_PROTOCOLS)
    parser.add_argument("--long-fraction", type=float, default=0.50)
    parser.add_argument("--short-confirm-rank", type=float, default=0.35)
    parser.add_argument("--cost-bps", type=float, nargs="+", default=[0, 6, 10, 15, 20, 30])
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    horizon = int(args.horizon)
    validation_days = int(args.validation_days)
    min_symbols = int(args.min_symbols)
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)

    print("[1/8] Loading P23 FF residual panel and folds", flush=True)
    ff_panel = load_ff_panel(Path(args.ff_panel))
    folds = pd.read_csv(args.folds, parse_dates=["train_start", "train_end", "test_start", "test_end"]).sort_values("fold")

    print("[2/8] Building FF residual slow factor library", flush=True)
    factor_panel, factor_meta = build_ff_factor_library(ff_panel)
    factor_panel.to_csv(output_dir / "p24_ff_factor_panel.csv.gz", index=False)
    factor_meta.to_csv(output_dir / "p24_ff_factor_library.csv", index=False)
    label_rank = centered_rank_matrix(factor_panel, f"label_resid_ret_h{horizon}", min_symbols=min_symbols)
    families = resolve_families(build_ff_families(), set(factor_meta["factor"]))
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
    family_meta.to_csv(output_dir / "p24_family_library.csv", index=False)
    if not families:
        raise RuntimeError("No P24 FF residual families resolved to available factors.")
    print(f"      factors={len(factor_meta)}, families={len(families)}", flush=True)

    print("[3/8] Precomputing component rank matrices and regime state", flush=True)
    needed = sorted({component.factor for family in families for component in family.components})
    rank_matrices = {factor: centered_rank_matrix(factor_panel, factor, min_symbols=min_symbols) for factor in needed}
    regime_panel = load_p3d_panel(config_path, Path(args.asset_panel), Path(args.manual_panel), horizon)
    add_regime_features(regime_panel)
    regime = (
        regime_panel[["timestamp", "_p18d_high_risk_regime"]]
        .drop_duplicates("timestamp")
        .sort_values("timestamp")
        .set_index("timestamp")["_p18d_high_risk_regime"]
        .astype(bool)
    )

    print("[4/8] Running P21-style family diagnostics with FF residual labels", flush=True)
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
    fold_metrics.to_csv(output_dir / "p24_family_fold_metrics.csv", index=False)
    family_score_ts.to_csv(output_dir / "p24_family_score_timeseries.csv.gz", index=False)

    print("[5/8] Applying P20 protocols", flush=True)
    all_protocols = build_protocols()
    protocols = [spec for spec in all_protocols if spec.name in set(args.protocols)]
    summary = summarize_families(fold_metrics, protocols)
    summary.to_csv(output_dir / "p24_family_summary.csv", index=False)
    selected = select_family_by_fold(fold_metrics, protocols)
    selected.to_csv(output_dir / "p24_selected_family_by_fold.csv", index=False)
    selected_summary = summarize_selected(selected)
    selected_summary.to_csv(output_dir / "p24_selected_protocol_summary.csv", index=False)

    print("[6/8] Replaying selected FF family scores through strict-neutral allocator", flush=True)
    base_panel = load_p3d_panel(config_path, Path(args.asset_panel), Path(args.manual_panel), horizon)
    base_panel = replace_labels_with_ff(base_panel, ff_panel, horizon)
    add_p18c_entry_fields(base_panel)
    pf_cfg = make_portfolio_config(cfg, horizon)
    risk_template = make_risk_template(cfg)
    p15c = load_existing_result(Path(args.p15c_run_dir))
    results: dict[str, dict[str, Any]] = {}
    selection_parts: list[pd.DataFrame] = []
    for protocol in args.protocols:
        protocol_selected = selected[selected["protocol"].eq(protocol)].copy()
        if protocol_selected.empty:
            continue
        stage = f"p24_{protocol.replace('p20_', '')}"
        print(f"      replay {stage}", flush=True)
        result, selection = replay_protocol(
            protocol_name=stage,
            protocol=protocol,
            base_panel=base_panel,
            family_scores=family_score_ts.rename(columns={"score": P24_SCORE}).rename(columns={P24_SCORE: "score"}),
            folds=folds,
            selected=protocol_selected,
            p15c=p15c,
            pf_cfg=pf_cfg,
            risk_template=risk_template,
            long_fraction=float(args.long_fraction),
            short_confirm_rank=float(args.short_confirm_rank),
        )
        results[stage] = result
        selection_parts.append(selection)
        run_dir = ensure_dir(output_dir / stage / "nested_p24_run")
        write_run_result(run_dir, result)
        selection.to_csv(output_dir / stage / "selected_family_by_fold.csv", index=False)

    print("[7/8] Building comparisons and plots", flush=True)
    refs = {**results, "p15c": p15c}
    comparison = build_comparison(refs)
    comparison.to_csv(output_dir / "comparison_summary.csv", index=False)
    cost = build_cost_sensitivity(refs, [float(x) for x in args.cost_bps])
    cost.to_csv(output_dir / "cost_sensitivity.csv", index=False)
    segments = build_segment_comparison(refs, [float(x) for x in args.cost_bps])
    segments.to_csv(output_dir / "segment_cost_summary.csv", index=False)
    fold_deltas: list[pd.DataFrame] = []
    assets: list[pd.DataFrame] = []
    for stage, result in results.items():
        delta = build_fold_delta(result, p15c, stage)
        delta["stage"] = stage
        delta.to_csv(output_dir / stage / "fold_delta_vs_p15c.csv", index=False)
        fold_deltas.append(delta)
        asset = build_asset_contribution(result["positions"])
        asset["stage"] = stage
        asset.to_csv(output_dir / stage / "asset_contribution.csv", index=False)
        assets.append(asset)
    fold_delta_all = pd.concat(fold_deltas, ignore_index=True) if fold_deltas else pd.DataFrame()
    fold_delta_all.to_csv(output_dir / "fold_delta_vs_p15c.csv", index=False)
    asset_all = pd.concat(assets, ignore_index=True) if assets else pd.DataFrame()
    asset_all.to_csv(output_dir / "asset_contribution.csv", index=False)
    if selection_parts:
        pd.concat(selection_parts, ignore_index=True).to_csv(output_dir / "allocator_selected_family_by_fold.csv", index=False)
    plot_curves(output_dir, refs, [0.0, 6.0, 15.0, 20.0, 30.0])
    plot_family_summary(output_dir, summary)

    print("[8/8] Writing report", flush=True)
    topline = build_topline(summary, selected_summary, comparison, cost, fold_delta_all)
    topline.to_csv(output_dir / "topline_findings.csv", index=False)
    metadata = {
        "horizon": horizon,
        "validation_days": validation_days,
        "min_symbols": min_symbols,
        "protocols": list(args.protocols),
        "ff_panel": str(Path(args.ff_panel).resolve()),
        "note": "P24 uses P23 FF residual labels and slow FF residual primitives, selected with P20/P21 protocol, then replayed through strict-neutral allocator with P15c fallback.",
    }
    (output_dir / "metadata.json").write_text(json.dumps(to_jsonable(metadata), indent=2), encoding="utf-8")
    write_report(output_dir, topline, family_meta, summary, selected_summary, comparison, cost, fold_delta_all)
    print(f"Done: {output_dir}", flush=True)


def load_ff_panel(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    needed = [
        "timestamp",
        "symbol",
        "ff_residual_log_ret",
        "ff_resid_z",
        "ff_resid_mom_168_24",
        "ff_idio_vol_long",
        "ff_score_neg_resid_z",
        *[f"ff_beta_{name}" for name in ["MKT", "LIQ", "MOM", "REV", "CARRY", "CROWDING", "VOL", "BETA"]],
        *[f"ff_label_resid_ret_h{h}" for h in LABEL_HORIZONS],
    ]
    usecols = [col for col in needed if col in header]
    frame = pd.read_csv(path, usecols=usecols, parse_dates=["timestamp"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["symbol"] = frame["symbol"].astype(str)
    return frame.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def build_ff_factor_library(ff: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pieces: list[pd.DataFrame] = []
    for _, group in ff.groupby("symbol", sort=False):
        g = group.sort_values("timestamp").copy()
        resid = g["ff_residual_log_ret"]
        resid_cum = resid.fillna(0.0).cumsum()
        g["ff_resid_mom_336_24"] = resid.shift(24).rolling(312, min_periods=120).sum()
        g["ff_resid_mom_504_48"] = resid.shift(48).rolling(456, min_periods=168).sum()
        g["ff_resid_reversal_z"] = -g["ff_resid_z"]
        g["ff_resid_abs_z"] = -g["ff_resid_z"].abs()
        g["ff_resid_high_mom_168"] = resid_cum - resid_cum.rolling(168, min_periods=56).max()
        g["ff_resid_high_mom_336"] = resid_cum - resid_cum.rolling(336, min_periods=120).max()
        g["ff_resid_ma_slope_48_168"] = resid_cum.ewm(span=48, adjust=False, min_periods=24).mean() - resid_cum.ewm(
            span=168, adjust=False, min_periods=56
        ).mean()
        g["ff_resid_trend_quality_168_24"] = g["ff_resid_mom_168_24"] / (
            resid.rolling(168, min_periods=56).std(ddof=0).replace(0.0, np.nan)
        )
        g["ff_resid_trend_quality_336_24"] = g["ff_resid_mom_336_24"] / (
            resid.rolling(336, min_periods=120).std(ddof=0).replace(0.0, np.nan)
        )
        beta_cols = [col for col in g.columns if col.startswith("ff_beta_")]
        g["ff_beta_abs_sum"] = -g[beta_cols].abs().sum(axis=1)
        g["ff_beta_mkt_abs"] = -g["ff_beta_MKT"].abs()
        g["ff_beta_vol_abs"] = -g["ff_beta_VOL"].abs()
        g["ff_beta_beta_abs"] = -g["ff_beta_BETA"].abs()
        g["ff_beta_stability_168"] = -g[beta_cols].diff().abs().rolling(168, min_periods=56).mean().mean(axis=1)
        g["ff_beta_stability_720"] = -g[beta_cols].diff().abs().rolling(720, min_periods=240).mean().mean(axis=1)
        g["ff_low_idio_vol"] = -g["ff_idio_vol_long"]
        g["ff_quality_lowrisk_combo"] = (g["ff_low_idio_vol"] + g["ff_beta_stability_168"] + g["ff_beta_abs_sum"]) / 3.0
        pieces.append(g)
    out = pd.concat(pieces, ignore_index=True).sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    for horizon in LABEL_HORIZONS:
        src = f"ff_label_resid_ret_h{horizon}"
        if src in out.columns:
            out[f"label_resid_ret_h{horizon}"] = out[src]
    factor_cols = [
        "ff_score_neg_resid_z",
        "ff_resid_z",
        "ff_resid_reversal_z",
        "ff_resid_abs_z",
        "ff_resid_mom_168_24",
        "ff_resid_mom_336_24",
        "ff_resid_mom_504_48",
        "ff_resid_high_mom_168",
        "ff_resid_high_mom_336",
        "ff_resid_ma_slope_48_168",
        "ff_resid_trend_quality_168_24",
        "ff_resid_trend_quality_336_24",
        "ff_idio_vol_long",
        "ff_low_idio_vol",
        "ff_beta_abs_sum",
        "ff_beta_mkt_abs",
        "ff_beta_vol_abs",
        "ff_beta_beta_abs",
        "ff_beta_stability_168",
        "ff_beta_stability_720",
        "ff_quality_lowrisk_combo",
        "ff_beta_MKT",
        "ff_beta_LIQ",
        "ff_beta_MOM",
        "ff_beta_REV",
        "ff_beta_CARRY",
        "ff_beta_CROWDING",
        "ff_beta_VOL",
        "ff_beta_BETA",
    ]
    keep_labels = [f"label_resid_ret_h{h}" for h in LABEL_HORIZONS if f"label_resid_ret_h{h}" in out.columns]
    factor_cols = [col for col in factor_cols if col in out.columns]
    renamed = {col: f"ff__{col}" for col in factor_cols}
    panel = out[["timestamp", "symbol", *keep_labels, *factor_cols]].rename(columns=renamed)
    meta = pd.DataFrame([{"factor": f"ff__{col}", "source": "ff", "raw_name": col} for col in factor_cols])
    meta = drop_low_information(panel, meta)
    keep = ["timestamp", "symbol", *keep_labels, *meta["factor"].tolist()]
    return panel[keep].copy(), meta.reset_index(drop=True)


def drop_low_information(panel: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    keep: list[str] = []
    for factor in meta["factor"]:
        x = panel[factor]
        finite = np.isfinite(x.to_numpy(dtype=float, copy=False))
        if finite.mean() < 0.25:
            continue
        by_ts_std = panel.loc[finite, ["timestamp", factor]].groupby("timestamp")[factor].std(ddof=0)
        if by_ts_std.replace([np.inf, -np.inf], np.nan).dropna().gt(1e-12).mean() < 0.20:
            continue
        keep.append(factor)
    return meta[meta["factor"].isin(keep)].copy()


def c(name: str, weight: float = 1.0) -> Component:
    return Component(name, weight)


def build_ff_families() -> list[Family]:
    return [
        Family(
            name="p24_ff_mean_reversion_quality",
            description="FF residual z-score mean reversion gated by low residual risk and beta stability.",
            components=(
                c("ff__ff_score_neg_resid_z", 1.0),
                c("ff__ff_resid_reversal_z", 1.0),
                c("ff__ff_resid_abs_z", 0.5),
                c("ff__ff_low_idio_vol", 0.5),
                c("ff__ff_beta_stability_168", 0.5),
            ),
        ),
        Family(
            name="p24_ff_slow_residual_momentum",
            description="Slow FF residual continuation with 168-504h lookbacks and 24-48h skips.",
            components=(
                c("ff__ff_resid_mom_168_24", 1.0),
                c("ff__ff_resid_mom_336_24", 1.0),
                c("ff__ff_resid_mom_504_48", 0.8),
                c("ff__ff_resid_trend_quality_168_24", 0.7),
                c("ff__ff_resid_trend_quality_336_24", 0.7),
            ),
        ),
        Family(
            name="p24_ff_high_residual_momentum",
            description="Distance to FF residual highs and slow residual MA slope.",
            components=(
                c("ff__ff_resid_high_mom_168", 1.0),
                c("ff__ff_resid_high_mom_336", 1.0),
                c("ff__ff_resid_ma_slope_48_168", 1.0),
                c("ff__ff_resid_mom_336_24", 0.5),
            ),
        ),
        Family(
            name="p24_ff_beta_stability_quality",
            description="Stable and low-common-factor-exposure FF residual quality.",
            components=(
                c("ff__ff_beta_stability_168", 1.0),
                c("ff__ff_beta_stability_720", 1.0),
                c("ff__ff_beta_abs_sum", 1.0),
                c("ff__ff_beta_mkt_abs", 0.7),
                c("ff__ff_beta_vol_abs", 0.7),
                c("ff__ff_low_idio_vol", 0.7),
            ),
        ),
        Family(
            name="p24_ff_vol_beta_exposure_contra",
            description="Contra exposure to FF VOL/BETA risk loadings, designed as a risk-aware alpha candidate.",
            components=(
                c("ff__ff_beta_vol_abs", 1.0),
                c("ff__ff_beta_beta_abs", 1.0),
                c("ff__ff_beta_VOL", 0.5),
                c("ff__ff_beta_BETA", 0.5),
                c("ff__ff_quality_lowrisk_combo", 0.8),
            ),
        ),
        Family(
            name="p24_ff_residual_quality_combo",
            description="Composite FF residual signal combining mean reversion, slow trend, and beta quality.",
            components=(
                c("ff__ff_score_neg_resid_z", 0.8),
                c("ff__ff_resid_mom_336_24", 0.6),
                c("ff__ff_resid_trend_quality_336_24", 0.6),
                c("ff__ff_quality_lowrisk_combo", 1.0),
                c("ff__ff_beta_stability_720", 0.6),
            ),
        ),
    ]


def replace_labels_with_ff(base_panel: pd.DataFrame, ff_panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    label_col = f"ff_label_resid_ret_h{horizon}"
    repl = ff_panel[["timestamp", "symbol", label_col]].rename(columns={label_col: f"label_resid_ret_h{horizon}_ff"})
    out = base_panel.merge(repl, on=["timestamp", "symbol"], how="left")
    out[f"label_resid_ret_h{horizon}"] = out[f"label_resid_ret_h{horizon}_ff"]
    out.drop(columns=[f"label_resid_ret_h{horizon}_ff"], inplace=True)
    return out


def build_comparison(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for name, result in results.items():
        metrics = result.get("metrics")
        if metrics is None:
            metrics = summarize_result(result["equity"])
        rows.append({"stage": name, **metrics})
    return pd.DataFrame(rows)


def summarize_result(equity: pd.DataFrame) -> dict[str, float]:
    returns = equity["period_return"].astype(float)
    gross = equity["gross_period_return"].astype(float)
    std = returns.std(ddof=0)
    sharpe = 0.0 if std == 0 or not np.isfinite(std) else float(returns.mean() / std * np.sqrt(365 * 24 / 72))
    eq = (1.0 + returns).cumprod()
    dd = eq / eq.cummax() - 1.0
    return {
        "periods": int(len(equity)),
        "total_return": float(eq.iloc[-1] - 1.0) if len(eq) else 0.0,
        "gross_total_return_before_cost": float((1.0 + gross).prod() - 1.0) if len(gross) else 0.0,
        "sharpe": sharpe,
        "max_drawdown": float(dd.min()) if len(dd) else 0.0,
        "mean_turnover": float(equity["turnover"].mean()) if "turnover" in equity else np.nan,
        "total_turnover": float(equity["turnover"].sum()) if "turnover" in equity else np.nan,
    }


def plot_curves(output_dir: Path, refs: dict[str, dict[str, Any]], cost_bps: list[float]) -> None:
    for bps in cost_bps:
        fig, ax = plt.subplots(figsize=(13, 6))
        for name, result in refs.items():
            equity = result["equity"].sort_values("timestamp")
            gross = equity["gross_period_return"] if "gross_period_return" in equity else equity["period_return"] + equity["cost_return"]
            ret = gross - equity["turnover"] * bps / 10_000.0
            curve = (1.0 + ret).cumprod() - 1.0
            width = 2.3 if name.startswith("p24") else 1.5
            ax.plot(equity["timestamp"], curve, label=name, linewidth=width)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_title(f"P24 FF residual factor selection, {bps:g}bps one-way cost")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / f"p24_pnl_{int(bps)}bps.png", dpi=160)
        plt.close(fig)


def plot_family_summary(output_dir: Path, summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    ordered = summary.sort_values("research_score", ascending=True)
    ax.barh(ordered["family"], ordered["research_score"])
    ax.set_title("P24 FF family research score")
    ax.set_xlabel("Research score")
    fig.tight_layout()
    fig.savefig(output_dir / "p24_family_research_score.png", dpi=160)
    plt.close(fig)


def build_topline(
    summary: pd.DataFrame,
    selected_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    cost: pd.DataFrame,
    fold_delta: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    best_family = summary.iloc[0] if not summary.empty else {}
    rows.append(
        {
            "check": "best_ff_family_by_research_score",
            "value": best_family.get("family", ""),
            "status": "info",
        }
    )
    rows.append(
        {
            "check": "best_ff_family_test_ic_mean",
            "value": float(best_family.get("test_ic_mean", np.nan)) if len(best_family) else np.nan,
            "status": "info",
        }
    )
    for _, row in comparison.iterrows():
        stage = str(row["stage"])
        if stage == "p15c":
            continue
        p15c_row = comparison[comparison["stage"].eq("p15c")]
        p15c_ret = float(p15c_row["total_return"].iloc[0]) if not p15c_row.empty else np.nan
        p15c_gross = float(p15c_row["gross_total_return_before_cost"].iloc[0]) if not p15c_row.empty else np.nan
        rows.append(
            {
                "check": f"{stage}_net_return_delta_vs_p15c",
                "value": float(row["total_return"]) - p15c_ret,
                "status": "pass" if float(row["total_return"]) > p15c_ret else "review",
            }
        )
        rows.append(
            {
                "check": f"{stage}_gross_return_delta_vs_p15c",
                "value": float(row["gross_total_return_before_cost"]) - p15c_gross,
                "status": "pass" if float(row["gross_total_return_before_cost"]) > p15c_gross else "review",
            }
        )
    for stage, group in cost.groupby("stage" if "stage" in cost.columns else "run"):
        name_col = "stage" if "stage" in cost.columns else "run"
        if stage == "p15c":
            continue
        b20 = group[group["one_way_cost_bps"].eq(20.0)]
        p15 = cost[(cost[name_col].eq("p15c")) & (cost["one_way_cost_bps"].eq(20.0))]
        if not b20.empty and not p15.empty:
            rows.append(
                {
                    "check": f"{stage}_20bps_return_delta_vs_p15c",
                    "value": float(b20["total_return"].iloc[0]) - float(p15["total_return"].iloc[0]),
                    "status": "pass"
                    if float(b20["total_return"].iloc[0]) > float(p15["total_return"].iloc[0])
                    else "review",
                }
            )
    if not fold_delta.empty:
        for stage, group in fold_delta.groupby("stage", sort=False) if "stage" in fold_delta else []:
            rows.append(
                {
                    "check": f"{stage}_folds_beating_p15c",
                    "value": int((group["delta_total_return"] > 0).sum()),
                    "status": "info",
                }
            )
    if not selected_summary.empty:
        for _, row in selected_summary.iterrows():
            active_rate = (
                float(row["active_fold_rate"])
                if "active_fold_rate" in selected_summary.columns
                else 1.0 - float(row.get("fallback_rate", np.nan))
            )
            rows.append(
                {
                    "check": f"{row['protocol']}_active_fold_rate",
                    "value": active_rate,
                    "status": "info",
                }
            )
    return pd.DataFrame(rows)


def write_report(
    output_dir: Path,
    topline: pd.DataFrame,
    family_meta: pd.DataFrame,
    summary: pd.DataFrame,
    selected_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    cost: pd.DataFrame,
    fold_delta: pd.DataFrame,
) -> None:
    lines = [
        "# P24 FF Residual Factor Selection",
        "",
        "This experiment feeds P23 rolling crypto-FF residual primitives and FF residual labels into the P20/P21 selection protocol, then replays selected scores through the strict-neutral allocator with P15c fallback.",
        "",
        "## Topline",
        "",
        "```text",
        topline.to_string(index=False),
        "```",
        "",
        "## FF Family Library",
        "",
        "```text",
        family_meta.to_string(index=False),
        "```",
        "",
        "## Family Summary",
        "",
        "```text",
        summary.head(12).to_string(index=False),
        "```",
        "",
        "## Selected Protocol Summary",
        "",
        "```text",
        selected_summary.to_string(index=False),
        "```",
        "",
        "## Portfolio Comparison",
        "",
        "```text",
        comparison.to_string(index=False),
        "```",
        "",
        "## Cost Sensitivity",
        "",
        "```text",
        cost.to_string(index=False),
        "```",
        "",
        "## Fold Delta Vs P15c",
        "",
        "```text",
        fold_delta.head(40).to_string(index=False) if not fold_delta.empty else "No active P24 fold deltas.",
        "```",
        "",
        "## Caveats",
        "",
        "- This is still not a full AlphaGPT formula search; it is a constrained family-selection test over FF residual primitives.",
        "- Folds rejected by P20 protocols fall back to P15c, so active fold rate matters as much as total return.",
        "- P24 uses FF residual labels for family selection and backtest IC metrics, while realized PnL still comes from actual forward simple returns.",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
