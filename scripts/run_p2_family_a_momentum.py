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
from crypto_stat_arb.features import add_forward_labels
from crypto_stat_arb.portfolio import PortfolioConfig
from crypto_stat_arb.walk_forward import WalkForwardConfig
from run_delay_smoothing_experiments import (
    DelayConfig,
    add_delay_metrics,
    run_delay_backtest,
)
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
)
from run_turnover_experiments import to_jsonable


EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "baseline_locked_v1"
DEFAULT_CONFIG = EXPERIMENT_ROOT / "configs" / "baseline_locked_v1.yaml"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "outputs" / "p2_family_a_momentum"
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
    labels_path = SLOW_LABELS
    if not labels_path.exists():
        raise FileNotFoundError(f"P1 slow label file not found: {labels_path}")

    print("[1/7] Loading panel, P1 labels, and folds", flush=True)
    panel = pd.read_csv(panel_path, parse_dates=["timestamp"])
    panel = prepare_alpha_panel(panel, sorted(set(horizons + [48, 72, 96, 168])), execution_delay_bars=1)
    build_candidate_scores(panel)
    slow_labels = pd.read_csv(
        labels_path,
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
    panel = panel.merge(slow_labels, on=["timestamp", "symbol"], how="left")
    selections = pd.read_csv(selections_path, parse_dates=["train_start", "train_end", "test_start", "test_end"])

    print("[2/7] Building Family A residual momentum/high-momentum features", flush=True)
    candidates = build_family_a_features(panel)
    candidate_meta = pd.DataFrame(candidates)
    candidate_meta.to_csv(output_dir / "candidate_formulas.csv", index=False)
    feature_cols = [
        "timestamp",
        "symbol",
        *sorted({feature for item in candidates for feature in item["features"].split("|") if feature}),
        *[item["score_col"] for item in candidates],
    ]
    existing_feature_cols = [col for col in feature_cols if col in panel.columns]
    panel[existing_feature_cols].to_csv(output_dir / "family_a_feature_panel.csv.gz", index=False)

    print("[3/7] Running strict-neutral walk-forward backtests", flush=True)
    wf_cfg = make_walk_forward_config(cfg)
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
            )
            result_index[key] = result
            run_dir = ensure_dir(runs_dir / key)
            write_run_result(run_dir, result)
            summary_rows.append(
                {
                    "run": key,
                    "horizon": horizon,
                    **candidate,
                    **result["metrics"],
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "candidate_summary.csv", index=False)
    summary.sort_values(["total_return", "sharpe"], ascending=[False, False]).to_csv(
        output_dir / "candidate_summary_by_net.csv", index=False
    )
    summary.sort_values(["gross_total_return_before_cost", "gross_max_drawdown"], ascending=[False, False]).to_csv(
        output_dir / "candidate_summary_by_gross.csv", index=False
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
    best_equity = best_result["equity"].sort_values("timestamp")
    best_positions = best_result["positions"].sort_values(["timestamp", "symbol"])
    best_horizon = int(summary.loc[summary["run"] == best_key, "horizon"].iloc[0])
    regime_summary = build_regime_summary(best_equity, build_regime_frame(panel), best_horizon)
    side_summary = build_side_summary(best_positions)
    asset_summary = build_asset_summary(best_positions)
    regime_summary.to_csv(output_dir / "best_regime_summary.csv", index=False)
    side_summary.to_csv(output_dir / "best_side_contribution.csv", index=False)
    asset_summary.to_csv(output_dir / "best_asset_contribution.csv", index=False)

    top_keys = list(summary.sort_values(["total_return", "sharpe"], ascending=[False, False])["run"].head(5))
    cost_inputs = {key: result_index[key] for key in top_keys}
    cost_summary = build_cost_sensitivity(cost_inputs, cost_bps)
    cost_summary.to_csv(output_dir / "top5_cost_sensitivity.csv", index=False)

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
    )
    update_config_stage(config_path)
    print(f"Done: {output_dir}", flush=True)


def resolve_config_path(config_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def make_walk_forward_config(cfg: dict[str, Any]) -> WalkForwardConfig:
    # Use the same folds as the frozen committee selections; this config is kept for metadata parity.
    return WalkForwardConfig(train_days=180, test_days=30, step_days=30, mode="rolling")


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


def build_family_a_features(panel: pd.DataFrame) -> list[dict[str, str]]:
    panel.sort_values(["symbol", "timestamp"], inplace=True)
    grouped = panel.groupby("symbol", group_keys=False)

    panel["resid_cum"] = grouped["residual_log_ret"].cumsum()
    for window in [168, 336, 504]:
        panel[f"res_rv_{window}"] = grouped["residual_log_ret"].transform(
            lambda s, w=window: np.sqrt(s.pow(2).rolling(w, min_periods=max(48, w // 3)).sum())
        )

    momentum_specs = [(72, 12, 168), (96, 12, 168), (168, 24, 336), (336, 24, 504)]
    for lookback, skip, rv_window in momentum_specs:
        length = lookback - skip
        raw_sum = grouped["residual_log_ret"].transform(
            lambda s, sk=skip, ln=length: s.shift(sk).rolling(ln, min_periods=max(24, ln // 3)).sum()
        )
        panel[f"res_mom_{lookback}_{skip}"] = raw_sum / panel[f"res_rv_{rv_window}"].replace(0.0, np.nan)

    for high_window in [168, 336]:
        rolling_high = grouped["resid_cum"].transform(
            lambda s, w=high_window: s.rolling(w, min_periods=max(48, w // 3)).max()
        )
        panel[f"res_high_mom_{high_window}"] = panel["resid_cum"] - rolling_high

    panel["res_ewm_48"] = grouped["resid_cum"].transform(lambda s: s.ewm(span=48, adjust=False, min_periods=24).mean())
    panel["res_ewm_168"] = grouped["resid_cum"].transform(lambda s: s.ewm(span=168, adjust=False, min_periods=72).mean())
    panel["res_ma_slope_48_168"] = panel["res_ewm_48"] - panel["res_ewm_168"]

    res_rv_24 = grouped["residual_log_ret"].transform(
        lambda s: np.sqrt(s.pow(2).rolling(24, min_periods=12).sum())
    )
    panel["vol_of_vol_res_168"] = (
        res_rv_24.groupby(panel["symbol"]).transform(lambda s: s.rolling(168, min_periods=72).std(ddof=0))
    )
    panel["res_trend_quality_168_24"] = panel["res_mom_168_24"] / panel["vol_of_vol_res_168"].replace(0.0, np.nan)

    total_quote = panel.groupby("timestamp")["quote_volume"].transform("sum")
    panel["volume_share"] = panel["quote_volume"] / total_quote.replace(0.0, np.nan)
    panel["log_volume_share"] = np.log(panel["volume_share"].replace(0.0, np.nan))
    panel["volume_share_trend_7d_30d"] = grouped["log_volume_share"].transform(
        lambda s: s.rolling(168, min_periods=72).mean() - s.rolling(720, min_periods=240).mean()
    )

    base_features = [
        "res_mom_72_12",
        "res_mom_96_12",
        "res_mom_168_24",
        "res_mom_336_24",
        "res_high_mom_168",
        "res_high_mom_336",
        "res_ma_slope_48_168",
        "res_trend_quality_168_24",
        "vol_of_vol_res_168",
        "volume_share_trend_7d_30d",
    ]
    for feature in base_features:
        panel[f"cs_{feature}"] = centered_rank_by_timestamp(panel, feature)

    score_specs: list[tuple[str, str, str, list[str]]] = [
        ("A01_res_mom_72_12", "alpha_A_res_mom_72_12", "rank_cs(res_mom_72_12)", ["res_mom_72_12"]),
        ("A02_res_mom_96_12", "alpha_A_res_mom_96_12", "rank_cs(res_mom_96_12)", ["res_mom_96_12"]),
        ("A03_res_mom_168_24", "alpha_A_res_mom_168_24", "rank_cs(res_mom_168_24)", ["res_mom_168_24"]),
        ("A04_res_mom_336_24", "alpha_A_res_mom_336_24", "rank_cs(res_mom_336_24)", ["res_mom_336_24"]),
        ("A05_res_high_mom_168", "alpha_A_res_high_mom_168", "rank_cs(res_high_mom_168)", ["res_high_mom_168"]),
        ("A06_res_ma_slope", "alpha_A_res_ma_slope", "rank_cs(res_ma_slope)", ["res_ma_slope_48_168"]),
        (
            "A07_res_trend_quality",
            "alpha_A_res_trend_quality",
            "rank_cs(res_trend_quality)",
            ["res_trend_quality_168_24"],
        ),
    ]

    for _, col, _, features in score_specs:
        panel[col] = panel[f"cs_{features[0]}"]

    panel["alpha_A_mom168_high168"] = panel["cs_res_mom_168_24"] + 0.5 * panel["cs_res_high_mom_168"]
    panel["alpha_A_mom168_low_vov"] = panel["cs_res_mom_168_24"] - 0.5 * panel["cs_vol_of_vol_res_168"]
    panel["alpha_A_mom168_liqtrend"] = panel["cs_res_mom_168_24"] + 0.3 * panel["cs_volume_share_trend_7d_30d"]
    panel["alpha_A_template"] = (
        panel["cs_res_mom_168_24"]
        + 0.5 * panel["cs_res_high_mom_168"]
        - 0.5 * panel["cs_vol_of_vol_res_168"]
        + 0.3 * panel["cs_volume_share_trend_7d_30d"]
    )
    panel["alpha_A_template_fast"] = (
        panel["cs_res_mom_72_12"]
        + 0.5 * panel["cs_res_high_mom_168"]
        - 0.5 * panel["cs_vol_of_vol_res_168"]
        + 0.3 * panel["cs_volume_share_trend_7d_30d"]
    )
    panel["alpha_A_template_long"] = (
        panel["cs_res_mom_336_24"]
        + 0.5 * panel["cs_res_high_mom_336"]
        - 0.5 * panel["cs_vol_of_vol_res_168"]
        + 0.3 * panel["cs_volume_share_trend_7d_30d"]
    )

    score_specs.extend(
        [
            (
                "A08_mom168_high168",
                "alpha_A_mom168_high168",
                "rank_cs(res_mom_168_24)+0.5*rank_cs(res_high_mom_168)",
                ["res_mom_168_24", "res_high_mom_168"],
            ),
            (
                "A09_mom168_low_vov",
                "alpha_A_mom168_low_vov",
                "rank_cs(res_mom_168_24)-0.5*rank_cs(vol_of_vol_res_168)",
                ["res_mom_168_24", "vol_of_vol_res_168"],
            ),
            (
                "A10_mom168_liqtrend",
                "alpha_A_mom168_liqtrend",
                "rank_cs(res_mom_168_24)+0.3*rank_cs(volume_share_trend_7d_30d)",
                ["res_mom_168_24", "volume_share_trend_7d_30d"],
            ),
            (
                "A11_alpha_A_template",
                "alpha_A_template",
                "rank_cs(res_mom_168_24)+0.5*rank_cs(res_high_mom_168)-0.5*rank_cs(vol_of_vol_res_168)+0.3*rank_cs(volume_share_trend_7d_30d)",
                ["res_mom_168_24", "res_high_mom_168", "vol_of_vol_res_168", "volume_share_trend_7d_30d"],
            ),
            (
                "A12_alpha_A_template_fast",
                "alpha_A_template_fast",
                "rank_cs(res_mom_72_12)+0.5*rank_cs(res_high_mom_168)-0.5*rank_cs(vol_of_vol_res_168)+0.3*rank_cs(volume_share_trend_7d_30d)",
                ["res_mom_72_12", "res_high_mom_168", "vol_of_vol_res_168", "volume_share_trend_7d_30d"],
            ),
            (
                "A13_alpha_A_template_long",
                "alpha_A_template_long",
                "rank_cs(res_mom_336_24)+0.5*rank_cs(res_high_mom_336)-0.5*rank_cs(vol_of_vol_res_168)+0.3*rank_cs(volume_share_trend_7d_30d)",
                ["res_mom_336_24", "res_high_mom_336", "vol_of_vol_res_168", "volume_share_trend_7d_30d"],
            ),
        ]
    )

    candidates = [
        {
            "name": name,
            "score_col": score_col,
            "formula": formula,
            "features": "|".join(features),
        }
        for name, score_col, formula, features in score_specs
    ]
    panel.sort_values(["timestamp", "symbol"], inplace=True)
    panel.reset_index(drop=True, inplace=True)
    return candidates


def centered_rank_by_timestamp(frame: pd.DataFrame, value_col: str) -> pd.Series:
    grouped = frame.groupby("timestamp", sort=False)[value_col]
    rank = grouped.rank(method="average")
    count = grouped.transform("count")
    centered = 2.0 * (rank - 1.0) / (count - 1.0) - 1.0
    return centered.where(count > 1)


def run_candidate_replay(
    panel: pd.DataFrame,
    selections: pd.DataFrame,
    pf_cfg: PortfolioConfig,
    wf_cfg: WalkForwardConfig,
    risk_cfg: RiskControlConfig,
    score_col: str,
    formula: str,
) -> dict[str, Any]:
    fold_rows: list[dict[str, Any]] = []
    equity_parts: list[pd.DataFrame] = []
    position_parts: list[pd.DataFrame] = []

    for _, selection in selections.iterrows():
        equity, positions, metrics = run_delay_backtest(
            panel=panel,
            start=selection["test_start"],
            end=selection["test_end"],
            cfg=pf_cfg,
            risk_cfg=risk_cfg,
            delay_cfg=DelayConfig("tau_0", 0.0),
            score_col=score_col,
            selected_score_cols=[score_col],
            selected_formulas=formula,
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
    return {"folds": fold_summary, "equity": equity_curve, "positions": positions, "metrics": metrics}


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
        "label_horizon_sign_consistency",
        "label_path_horizon_sign_consistency",
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
                )
                per_ts = per_ts.dropna()
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
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def build_baseline_comparison(summary: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    if baseline.empty:
        return summary.copy()
    base_cols = [
        "horizon",
        "total_return",
        "gross_total_return_before_cost",
        "sharpe",
        "max_drawdown",
        "gross_max_drawdown",
        "mean_turnover",
        "cost_drag",
    ]
    base = baseline[[col for col in base_cols if col in baseline.columns]].copy()
    base = base.rename(columns={col: f"baseline_{col}" for col in base.columns if col != "horizon"})
    out = summary.merge(base, on="horizon", how="left")
    for col in [
        "total_return",
        "gross_total_return_before_cost",
        "sharpe",
        "max_drawdown",
        "gross_max_drawdown",
        "mean_turnover",
        "cost_drag",
    ]:
        base_col = f"baseline_{col}"
        if base_col in out:
            out[f"delta_{col}"] = out[col] - out[base_col]
    return out.sort_values(["delta_total_return", "total_return"], ascending=[False, False])


def plot_summary(output_dir: Path, summary: pd.DataFrame) -> None:
    top = summary.sort_values(["total_return", "sharpe"], ascending=[False, False]).head(14).copy()
    top = top.sort_values("total_return")
    plt.figure(figsize=(12, 7))
    colors = np.where(top["total_return"] >= 0.0, "#2a9d8f", "#c44536")
    plt.barh(top["run"], top["total_return"], color=colors)
    plt.axvline(0.0, color="black", linewidth=0.8, alpha=0.6)
    plt.title("Family A top runs by net return")
    plt.xlabel("Net cumulative return")
    plt.gca().xaxis.set_major_formatter(PercentFormatter(1.0))
    plt.tight_layout()
    plt.savefig(output_dir / "family_a_top_net_returns.png", dpi=180)
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
    plt.title(f"Best Family A run: {best_key}")
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
    plt.title("Family A top-5 net curves")
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
    plt.title("Family A OOS IC vs path slow low-turnover label")
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
    plt.title("Family A top-5 cost sensitivity")
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
    plt.title("Best Family A gross contribution by asset")
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
        "# P2 Family A Residual Momentum / High-Momentum",
        "",
        "## Design",
        "",
        "- OHLCV-only Family A features: residual medium-term momentum, distance to residual high, residual MA slope, trend quality, volume-share trend.",
        "- Momentum features skip the most recent 12h or 24h to avoid short-term close noise.",
        "- Backtests use the locked strict-neutral allocator: 12.5% max symbol cap, short residual confirmation, all-symbol constrained hedge projection.",
        "- No external funding/OI/liquidation data is used.",
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
        "## Files",
        "",
        "- `candidate_formulas.csv`: Family A candidate definitions.",
        "- `family_a_feature_panel.csv.gz`: feature and score panel.",
        "- `candidate_summary.csv`: portfolio metrics for every candidate/horizon run.",
        "- `ic_summary.csv`: OOS IC against P1 labels.",
        "- `baseline_comparison.csv`: deltas versus locked baseline by horizon.",
        "- `runs/`: raw folds/equity/positions per run.",
        "- `best_run/`: raw details for the best run.",
        "- plots: `family_a_top_net_returns.png`, `best_net_gross_curve.png`, `top5_net_curves.png`, `ic_path_slow_low_turnover.png`, `top5_cost_sensitivity.png`, `best_asset_contribution.png`.",
        "",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def update_config_stage(config_path: Path) -> None:
    marker = "  p2_family_a_momentum:\n"
    text = config_path.read_text(encoding="utf-8")
    if marker in text:
        return
    addition = """
  p2_family_a_momentum:
    script: ../../../scripts/run_p2_family_a_momentum.py
    output_dir: ../outputs/p2_family_a_momentum
    report: ../outputs/p2_family_a_momentum/REPORT.md
    candidate_summary: ../outputs/p2_family_a_momentum/candidate_summary.csv
    ic_summary: ../outputs/p2_family_a_momentum/ic_summary.csv
    baseline_comparison: ../outputs/p2_family_a_momentum/baseline_comparison.csv
"""
    config_path.write_text(text.rstrip() + "\n" + addition, encoding="utf-8")


if __name__ == "__main__":
    main()
