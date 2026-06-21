#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
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
    spearman_ic,
    update_state,
)
from run_delay_smoothing_experiments import DelayConfig, add_delay_metrics
from run_p0_baseline_lock_review import (
    build_asset_summary,
    build_cost_sensitivity,
    build_regime_frame,
    build_regime_summary,
    build_side_summary,
)
from run_p2_family_b_mean_reversion import (
    build_baseline_comparison,
    centered_rank_by_timestamp,
    load_baseline_summary,
    make_portfolio_config,
)
from run_residual_alphagpt_search import aggregate_oos_metrics, build_candidate_scores, prepare_alpha_panel
from run_risk_control_experiments import (
    RiskControlConfig,
    add_gross_metrics,
    add_risk_event_metrics,
    build_weights,
)
from run_turnover_experiments import to_jsonable


EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "baseline_locked_v1"
DEFAULT_CONFIG = EXPERIMENT_ROOT / "configs" / "baseline_locked_v1.yaml"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "outputs" / "p3_committee_guard_scan"
MANUAL_FEATURE_PANEL = EXPERIMENT_ROOT / "outputs" / "p2_manual_slow_alpha_grid" / "manual_slow_feature_panel.csv.gz"

COMPONENTS = {
    "A07": "alpha_A_res_trend_quality",
    "B11": "alpha_B_seed_combo",
    "C09": "alpha_C_quality_committee",
}


@dataclass(frozen=True)
class CommitteeSpec:
    name: str
    w_a07: float
    w_b11: float
    w_c09: float
    long_fraction: float
    short_guard: str
    short_resid_confirm_rank: float


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--horizon", type=int, default=72)
    parser.add_argument("--cost-bps", type=float, nargs="+", default=[0, 4, 6, 8, 10, 12, 15, 20, 30])
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    output_dir = ensure_dir(args.output_dir)
    horizon = int(args.horizon)
    cost_bps = [float(x) for x in args.cost_bps]

    panel_path = resolve_config_path(config_path, cfg["inputs"]["panel"])
    selections_path = resolve_config_path(config_path, cfg["inputs"]["formula_selections"])

    print("[1/7] Loading base panel, manual feature scores, and folds", flush=True)
    panel = pd.read_csv(panel_path, parse_dates=["timestamp"])
    panel = prepare_alpha_panel(panel, [horizon], execution_delay_bars=1)
    build_candidate_scores(panel)
    add_market_regime_features(panel)
    features = pd.read_csv(
        MANUAL_FEATURE_PANEL,
        parse_dates=["timestamp"],
        usecols=["timestamp", "symbol", *COMPONENTS.values(), "family_b_regime_gate"],
    )
    panel = panel.merge(features, on=["timestamp", "symbol"], how="left")
    selections = pd.read_csv(selections_path, parse_dates=["train_start", "train_end", "test_start", "test_end"])

    print("[2/7] Building normalized A07/B11/C09 component ranks and committee grid", flush=True)
    for key, col in COMPONENTS.items():
        rank_col = f"rank_{key.lower()}"
        panel[rank_col] = centered_rank_by_timestamp(panel, col)

    specs = build_committee_specs()
    pd.DataFrame([spec.__dict__ for spec in specs]).to_csv(output_dir / "committee_specs.csv", index=False)

    print("[3/7] Running strict-neutral committee guard scan", flush=True)
    pf_cfg = make_portfolio_config(cfg, horizon)
    risk_template = make_risk_template(cfg)
    result_index: dict[str, dict[str, Any]] = {}
    summary_rows: list[dict[str, Any]] = []
    runs_dir = ensure_dir(output_dir / "runs")
    for idx, spec in enumerate(specs, start=1):
        print(f"      [{idx}/{len(specs)}] {spec.name}", flush=True)
        score_col = f"score_{spec.name}"
        panel[score_col] = (
            spec.w_a07 * panel["rank_a07"]
            + spec.w_b11 * panel["rank_b11"]
            + spec.w_c09 * panel["rank_c09"]
        )
        result = run_committee_replay(
            panel=panel,
            selections=selections,
            pf_cfg=pf_cfg,
            risk_template=risk_template,
            spec=spec,
            score_col=score_col,
        )
        result_index[spec.name] = result
        write_run_result(ensure_dir(runs_dir / spec.name), result)
        summary_rows.append({"horizon": horizon, **spec.__dict__, **result["metrics"]})
        panel.drop(columns=[score_col], inplace=True)

    summary = pd.DataFrame(summary_rows)
    corr_summary = build_locked_baseline_correlations(result_index, horizon)
    summary = summary.merge(corr_summary, left_on="name", right_on="run", how="left").drop(columns=["run"])
    summary["fitness"] = build_fitness(summary)
    summary.to_csv(output_dir / "summary.csv", index=False)
    summary.sort_values(["fitness", "total_return"], ascending=[False, False]).to_csv(output_dir / "summary_by_fitness.csv", index=False)
    summary.sort_values(["total_return", "sharpe"], ascending=[False, False]).to_csv(output_dir / "summary_by_net.csv", index=False)
    summary.sort_values(["max_drawdown", "total_return"], ascending=[False, False]).to_csv(output_dir / "summary_by_drawdown.csv", index=False)

    print("[4/7] Building top-run diagnostics", flush=True)
    baseline = load_baseline_summary(EXPERIMENT_ROOT / "baseline_lock_review" / "horizon_summary.csv")
    comparison = build_baseline_comparison(summary.rename(columns={"name": "run"}), baseline)
    comparison.to_csv(output_dir / "baseline_comparison.csv", index=False)

    top_names = select_top_names(summary)
    cost_summary = build_cost_sensitivity({name: result_index[name] for name in top_names}, cost_bps)
    cost_summary.to_csv(output_dir / "top_cost_sensitivity.csv", index=False)

    best_name = str(summary.sort_values(["fitness", "total_return"], ascending=[False, False]).iloc[0]["name"])
    best_result = result_index[best_name]
    write_run_result(ensure_dir(output_dir / "best_run"), best_result)
    best_equity = best_result["equity"].sort_values("timestamp")
    best_positions = best_result["positions"].sort_values(["timestamp", "symbol"])
    regime_summary = build_regime_summary(best_equity, build_regime_frame(panel), horizon)
    side_summary = build_side_summary(best_positions)
    asset_summary = build_asset_summary(best_positions)
    regime_summary.to_csv(output_dir / "best_regime_summary.csv", index=False)
    side_summary.to_csv(output_dir / "best_side_contribution.csv", index=False)
    asset_summary.to_csv(output_dir / "best_asset_contribution.csv", index=False)

    print("[5/7] Writing plots", flush=True)
    plot_summary(output_dir, summary)
    plot_top_curves(output_dir, result_index, top_names)
    plot_best_curves(output_dir, best_equity, best_name)
    plot_cost_sensitivity(output_dir, cost_summary)
    plot_asset_contribution(output_dir, asset_summary)

    print("[6/7] Writing report", flush=True)
    write_report(
        output_dir=output_dir,
        summary=summary,
        comparison=comparison,
        cost_summary=cost_summary,
        best_name=best_name,
        top_names=top_names,
        regime_summary=regime_summary,
        side_summary=side_summary,
        asset_summary=asset_summary,
    )

    print("[7/7] Done", flush=True)
    print(f"Done: {output_dir}", flush=True)


def resolve_config_path(config_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def add_market_regime_features(panel: pd.DataFrame) -> None:
    market = panel.groupby("timestamp")["log_ret"].mean().sort_index()
    panel["market_trailing_ret_7d"] = panel["timestamp"].map(market.rolling(168, min_periods=48).sum())


def make_risk_template(cfg: dict[str, Any]) -> RiskControlConfig:
    risk = cfg["locked_baseline"]["risk_control"]
    return RiskControlConfig(
        name="committee_guard_scan",
        description="Locked strict-neutral allocator with scanned committee score and short guards.",
        max_symbol_weight=float(risk["max_symbol_weight"]),
        short_resid_confirm_rank=float(risk["short_resid_confirm_rank"]),
        allocation_method=str(risk["allocation_method"]),
        optimizer_target_gross_mode=str(risk["optimizer_target_gross_mode"]),
        hedge_penalty=float(risk["hedge_penalty"]),
    )


def build_committee_specs() -> list[CommitteeSpec]:
    weight_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    weight_triples: list[tuple[float, float, float]] = []
    for w_a in weight_values:
        for w_b in weight_values:
            w_c = round(1.0 - w_a - w_b, 10)
            if w_c < -1e-12 or w_c > 1.0 + 1e-12:
                continue
            if abs(round(w_c / 0.2) * 0.2 - w_c) > 1e-9:
                continue
            weight_triples.append((round(w_a, 1), round(w_b, 1), round(w_c, 1)))

    side_fracs = [0.45, 0.50, 0.55]
    guards = [
        ("ema35", 0.35),
        ("ema30", 0.30),
        ("comp2_ema35", 0.35),
        ("c09b11_ema35", 0.35),
    ]
    specs: list[CommitteeSpec] = []
    for w_a, w_b, w_c in weight_triples:
        for long_fraction in side_fracs:
            for guard_name, confirm_rank in guards:
                name = (
                    f"a{int(w_a*10):02d}_b{int(w_b*10):02d}_c{int(w_c*10):02d}"
                    f"_lf{int(long_fraction*100):02d}_{guard_name}"
                )
                specs.append(
                    CommitteeSpec(
                        name=name,
                        w_a07=w_a,
                        w_b11=w_b,
                        w_c09=w_c,
                        long_fraction=long_fraction,
                        short_guard=guard_name,
                        short_resid_confirm_rank=confirm_rank,
                    )
                )
    return specs


def run_committee_replay(
    panel: pd.DataFrame,
    selections: pd.DataFrame,
    pf_cfg: PortfolioConfig,
    risk_template: RiskControlConfig,
    spec: CommitteeSpec,
    score_col: str,
) -> dict[str, Any]:
    fold_rows: list[dict[str, Any]] = []
    equity_parts: list[pd.DataFrame] = []
    position_parts: list[pd.DataFrame] = []
    risk_cfg = risk_template.__class__(
        **{
            **risk_template.__dict__,
            "name": spec.name,
            "short_resid_confirm_rank": spec.short_resid_confirm_rank,
        }
    )

    formula = (
        f"{spec.w_a07:.1f}*rank(A07)+{spec.w_b11:.1f}*rank(B11)+{spec.w_c09:.1f}*rank(C09); "
        f"long_fraction={spec.long_fraction:.2f}; short_guard={spec.short_guard}"
    )
    for _, selection in selections.iterrows():
        equity, positions, metrics = run_committee_backtest(
            panel=panel,
            start=selection["test_start"],
            end=selection["test_end"],
            cfg=pf_cfg,
            risk_cfg=risk_cfg,
            spec=spec,
            score_col=score_col,
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
    metrics["mean_short_guard_removed"] = float(equity_curve["short_guard_removed"].mean()) if "short_guard_removed" in equity_curve else np.nan
    return {"folds": fold_summary, "equity": equity_curve, "positions": positions, "metrics": metrics}


def run_committee_backtest(
    panel: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cfg: PortfolioConfig,
    risk_cfg: RiskControlConfig,
    spec: CommitteeSpec,
    score_col: str,
    selected_formulas: str,
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
        before_guard = len([side for side in target_sides.values() if side < 0])
        target_sides = apply_short_guard(target_sides, rows_by_symbol, spec)
        after_guard = len([side for side in target_sides.values() if side < 0])
        if not any(side > 0 for side in target_sides.values()) or not any(side < 0 for side in target_sides.values()):
            continue

        state_sides = target_sides.copy()
        selected = rows_by_symbol.reindex(list(target_sides)).reset_index()
        raw_weights = side_fraction_weights(target_sides, cfg.gross_exposure, spec.long_fraction)
        target_weights, optimizer_info = build_weights(raw_weights, selected, rows_by_symbol, cfg, risk_cfg)
        target_weights = target_weights[target_weights.abs() > 1e-12]
        if target_weights.abs().sum() <= 0:
            continue

        target_full_weights = pd.Series(0.0, index=symbols)
        target_full_weights.loc[target_weights.index] = target_weights
        turnover = float((target_full_weights - prev_weights).abs().sum())
        cost = turnover * (cfg.one_way_fee_bps + cfg.one_way_slippage_bps) / 10_000.0

        active_rows = rows_by_symbol.reindex(target_weights.index)
        gross_period_ret = float((target_weights * active_rows[ret_col]).sum())
        period_ret = gross_period_ret - cost
        equity *= 1.0 + period_ret

        rank_ic = spearman_ic(rows[score_col], rows[label_col])
        long_labels = active_rows.loc[target_weights[target_weights > 0].index, label_col]
        short_labels = active_rows.loc[target_weights[target_weights < 0].index, label_col]
        top_bottom = float(long_labels.mean() - short_labels.mean()) if len(long_labels) and len(short_labels) else np.nan
        exposures = portfolio_exposures(target_weights, active_rows, cfg.neutralize_exposures)

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
                "gross_exposure": float(target_weights.abs().sum()),
                "target_gross_exposure": float(target_full_weights.abs().sum()),
                "rank_ic": rank_ic,
                "top_bottom_label_spread": top_bottom,
                "abs_beta_btc": abs(exposures.get("beta_btc", 0.0)),
                "abs_beta_eth": abs(exposures.get("beta_eth", 0.0)),
                "abs_beta_market": abs(exposures.get("beta_market", 0.0)),
                "symbol_concentration": float((target_weights**2).sum()),
                "max_abs_weight": float(target_weights.abs().max()),
                "delay_tau_days": 0.0,
                "delay_retention": 0.0,
                "delay_new_target_weight": 1.0,
                "short_guard_removed": float(before_guard - after_guard),
                **optimizer_info,
                "target_applied": 1.0,
                "selected_formulas": selected_formulas,
            }
        )

        for symbol, weight in target_weights.items():
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
                    "rank_a07": float(rows_by_symbol.loc[symbol, "rank_a07"]),
                    "rank_b11": float(rows_by_symbol.loc[symbol, "rank_b11"]),
                    "rank_c09": float(rows_by_symbol.loc[symbol, "rank_c09"]),
                    "fwd_simple_ret": float(active_rows.loc[symbol, ret_col]),
                    "label_resid_ret": float(active_rows.loc[symbol, label_col]),
                    "contribution": float(weight * active_rows.loc[symbol, ret_col]),
                    "selected_score_cols": "rank_a07 | rank_b11 | rank_c09",
                }
            )

        prev_weights = target_full_weights
        current_state = update_state(current_state, state_sides, True)

    equity_curve = pd.DataFrame(rows_out)
    positions = pd.DataFrame(pos_out)
    metrics = aggregate_oos_metrics(equity_curve, cfg.rebalance_hours)
    return equity_curve, positions, metrics


def apply_short_guard(target_sides: dict[str, int], rows_by_symbol: pd.DataFrame, spec: CommitteeSpec) -> dict[str, int]:
    out: dict[str, int] = {}
    for symbol, side in target_sides.items():
        if side >= 0:
            out[symbol] = side
            continue
        row = rows_by_symbol.loc[symbol]
        if row["ema_rank"] > spec.short_resid_confirm_rank:
            continue
        if spec.short_guard == "ema35" or spec.short_guard == "ema30":
            out[symbol] = side
            continue
        if spec.short_guard == "comp2_ema35":
            short_votes = int(row["rank_a07"] < 0.0) + int(row["rank_b11"] < 0.0) + int(row["rank_c09"] < 0.0)
            if short_votes >= 2:
                out[symbol] = side
            continue
        if spec.short_guard == "c09b11_ema35":
            if row["rank_c09"] < 0.0 and row["rank_b11"] < 0.0:
                out[symbol] = side
            continue
        raise ValueError(f"Unknown short guard: {spec.short_guard}")
    return out


def side_fraction_weights(target_sides: dict[str, int], gross_exposure: float, long_fraction: float) -> pd.Series:
    longs = [symbol for symbol, side in target_sides.items() if side > 0]
    shorts = [symbol for symbol, side in target_sides.items() if side < 0]
    weights = pd.Series(0.0, index=list(target_sides), dtype=float)
    if longs:
        weights.loc[longs] = gross_exposure * long_fraction / len(longs)
    if shorts:
        weights.loc[shorts] = -gross_exposure * (1.0 - long_fraction) / len(shorts)
    return weights


def build_locked_baseline_correlations(result_index: dict[str, dict[str, Any]], horizon: int) -> pd.DataFrame:
    rows = []
    base_path = EXPERIMENT_ROOT / "baseline_lock_review" / "horizon" / f"h{horizon:03d}_tau0" / "equity_curve.csv"
    base = pd.read_csv(base_path, parse_dates=["timestamp"]).sort_values("timestamp") if base_path.exists() else pd.DataFrame()
    for run, result in result_index.items():
        row: dict[str, Any] = {"run": run}
        equity = result["equity"].sort_values("timestamp")
        if base.empty or equity.empty:
            row["net_corr_to_locked_baseline"] = np.nan
            row["gross_corr_to_locked_baseline"] = np.nan
            row["aligned_periods_with_locked_baseline"] = 0
        else:
            merged = base[["timestamp", "period_return", "gross_period_return"]].rename(
                columns={"period_return": "baseline_net", "gross_period_return": "baseline_gross"}
            ).merge(
                equity[["timestamp", "period_return", "gross_period_return"]].rename(
                    columns={"period_return": "candidate_net", "gross_period_return": "candidate_gross"}
                ),
                on="timestamp",
                how="inner",
            )
            row["aligned_periods_with_locked_baseline"] = int(len(merged))
            row["net_corr_to_locked_baseline"] = float(merged["baseline_net"].corr(merged["candidate_net"])) if len(merged) > 3 else np.nan
            row["gross_corr_to_locked_baseline"] = float(merged["baseline_gross"].corr(merged["candidate_gross"])) if len(merged) > 3 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def build_fitness(summary: pd.DataFrame) -> pd.Series:
    return (
        100.0 * summary["total_return"].fillna(0.0)
        + 3.0 * summary["sharpe"].fillna(0.0)
        - 45.0 * summary["max_drawdown"].abs().fillna(1.0)
        - 0.35 * summary["mean_turnover"].fillna(2.0)
        - 30.0 * summary["cost_drag"].fillna(1.0)
    )


def select_top_names(summary: pd.DataFrame) -> list[str]:
    names = []
    views = [
        summary.sort_values(["fitness", "total_return"], ascending=[False, False]),
        summary.sort_values(["total_return", "sharpe"], ascending=[False, False]),
        summary.sort_values(["max_drawdown", "total_return"], ascending=[False, False]),
        summary[summary["total_return"] > 0].sort_values(["net_corr_to_locked_baseline", "total_return"], ascending=[True, False]),
    ]
    for view in views:
        for name in view["name"].head(3):
            if name not in names:
                names.append(str(name))
            if len(names) >= 8:
                return names
    return names


def write_run_result(output_dir: Path, result: dict[str, Any]) -> None:
    result["folds"].to_csv(output_dir / "walk_forward_folds.csv", index=False)
    result["equity"].to_csv(output_dir / "equity_curve.csv", index=False)
    result["positions"].to_csv(output_dir / "positions.csv", index=False)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(to_jsonable(result["metrics"]), fh, indent=2, ensure_ascii=False)


def plot_summary(output_dir: Path, summary: pd.DataFrame) -> None:
    top = summary.sort_values(["fitness", "total_return"], ascending=[False, False]).head(18).sort_values("fitness")
    labels = top["name"]
    plt.figure(figsize=(12, 8))
    colors = np.where(top["total_return"] >= 0.0, "#2a9d8f", "#c44536")
    plt.barh(labels, top["fitness"], color=colors)
    plt.axvline(0.0, color="black", linewidth=0.8, alpha=0.6)
    plt.title("P3 committee guard scan: top runs by fitness")
    plt.xlabel("Fitness")
    plt.grid(True, axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "top_fitness.png", dpi=180)
    plt.close()


def plot_best_curves(output_dir: Path, equity: pd.DataFrame, best_name: str) -> None:
    if equity.empty:
        return
    frame = equity.sort_values("timestamp")
    net = (1.0 + frame["period_return"]).cumprod() - 1.0
    gross = (1.0 + frame["gross_period_return"]).cumprod() - 1.0
    dd = (1.0 + frame["period_return"]).cumprod()
    dd = dd / dd.cummax() - 1.0
    fig, axes = plt.subplots(2, 1, figsize=(13, 8.5), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    axes[0].plot(frame["timestamp"], gross, label="gross before cost", linewidth=2)
    axes[0].plot(frame["timestamp"], net, label="net after 6bps", linewidth=2)
    axes[0].axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    axes[0].set_title(f"Best P3 committee run: {best_name}")
    axes[0].set_ylabel("Cumulative return")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()
    axes[1].fill_between(frame["timestamp"], dd, 0.0, color="#e76f51", alpha=0.28)
    axes[1].plot(frame["timestamp"], dd, color="#e76f51", linewidth=1.3)
    axes[1].set_ylabel("Drawdown")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "best_net_gross_drawdown.png", dpi=180)
    plt.close(fig)


def plot_top_curves(output_dir: Path, result_index: dict[str, dict[str, Any]], top_names: list[str]) -> None:
    plt.figure(figsize=(12, 6.5))
    for name in top_names:
        equity = result_index[name]["equity"].sort_values("timestamp")
        if equity.empty:
            continue
        curve = (1.0 + equity["period_return"]).cumprod() - 1.0
        plt.plot(equity["timestamp"], curve, label=name, linewidth=1.5)
    plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    plt.title("P3 selected top committee curves")
    plt.ylabel("Net cumulative return")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(output_dir / "top_net_curves.png", dpi=180)
    plt.close()


def plot_cost_sensitivity(output_dir: Path, cost_summary: pd.DataFrame) -> None:
    if cost_summary.empty:
        return
    plt.figure(figsize=(12, 6.5))
    for name, group in cost_summary.groupby("experiment"):
        group = group.sort_values("one_way_cost_bps")
        plt.plot(group["one_way_cost_bps"], group["total_return"], marker="o", label=name)
    plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    plt.title("P3 top committee cost sensitivity")
    plt.xlabel("one-way cost bps")
    plt.ylabel("Net cumulative return")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(output_dir / "top_cost_sensitivity.png", dpi=180)
    plt.close()


def plot_asset_contribution(output_dir: Path, asset_summary: pd.DataFrame) -> None:
    if asset_summary.empty:
        return
    frame = asset_summary.sort_values("gross_contribution_sum")
    plt.figure(figsize=(11, 7))
    colors = np.where(frame["gross_contribution_sum"] >= 0.0, "#2a9d8f", "#c44536")
    plt.barh(frame["symbol"], frame["gross_contribution_sum"], color=colors)
    plt.axvline(0.0, color="black", linewidth=0.8, alpha=0.6)
    plt.title("Best P3 gross contribution by asset")
    plt.xlabel("Gross contribution sum")
    plt.gca().xaxis.set_major_formatter(PercentFormatter(1.0))
    plt.tight_layout()
    plt.savefig(output_dir / "best_asset_contribution.png", dpi=180)
    plt.close()


def write_report(
    output_dir: Path,
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    cost_summary: pd.DataFrame,
    best_name: str,
    top_names: list[str],
    regime_summary: pd.DataFrame,
    side_summary: pd.DataFrame,
    asset_summary: pd.DataFrame,
) -> None:
    display_cols = [
        "name",
        "w_a07",
        "w_b11",
        "w_c09",
        "long_fraction",
        "short_guard",
        "total_return",
        "gross_total_return_before_cost",
        "sharpe",
        "max_drawdown",
        "gross_max_drawdown",
        "mean_turnover",
        "cost_drag",
        "positive_fold_rate",
        "net_corr_to_locked_baseline",
        "mean_short_guard_removed",
        "fitness",
    ]
    comp_cols = [
        "run",
        "total_return",
        "baseline_total_return",
        "delta_total_return",
        "gross_total_return_before_cost",
        "baseline_gross_total_return_before_cost",
        "delta_gross_total_return_before_cost",
        "mean_turnover",
        "baseline_mean_turnover",
        "delta_mean_turnover",
        "net_corr_to_locked_baseline",
    ]
    cost_pivot = (
        cost_summary.pivot_table(index="one_way_cost_bps", columns="experiment", values="total_return")
        .reset_index()
        .sort_values("one_way_cost_bps")
    )
    lines = [
        "# P3 Committee Guard Scan",
        "",
        "## Design",
        "",
        "- Components: `A07=rank(alpha_A_res_trend_quality)`, `B11=rank(alpha_B_seed_combo)`, `C09=rank(alpha_C_quality_committee)`.",
        "- Horizon fixed at `72h`.",
        "- Allocator fixed to locked strict-neutral risk controls: 12.5% symbol cap, constrained hedge projection, clipped gross target.",
        "- Scans component weights, long/short pre-optimizer sizing, and short exposure guards.",
        "",
        "## Best Fitness Run",
        "",
        f"- best_run: `{best_name}`",
        "",
        "## Top Summary",
        "",
        "```text",
        summary.sort_values(["fitness", "total_return"], ascending=[False, False])[
            [col for col in display_cols if col in summary.columns]
        ].head(40).to_string(index=False),
        "```",
        "",
        "## Baseline Comparison",
        "",
        "```text",
        comparison[[col for col in comp_cols if col in comparison.columns]].head(30).to_string(index=False),
        "```",
        "",
        "## Cost Sensitivity",
        "",
        f"Top-run set: `{', '.join(top_names)}`",
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
        "- `committee_specs.csv`: scanned parameter grid.",
        "- `summary.csv`: all metrics.",
        "- `summary_by_fitness.csv`, `summary_by_net.csv`, `summary_by_drawdown.csv`: ranking views.",
        "- `top_cost_sensitivity.csv`: cost robustness for selected top runs.",
        "- `runs/`: raw folds/equity/positions per run.",
        "- `best_run/`: raw details for best fitness run.",
        "- plots: `top_fitness.png`, `top_net_curves.png`, `best_net_gross_drawdown.png`, `top_cost_sensitivity.png`, `best_asset_contribution.png`.",
        "",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
