#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crypto_stat_arb.config import ensure_dir, load_config
from crypto_stat_arb.portfolio import PortfolioConfig, run_long_short_backtest, summarize_equity_curve
from crypto_stat_arb.walk_forward import (
    WalkForwardConfig,
    calibrate_score_to_return_coef,
    make_walk_forward_folds,
    run_walk_forward,
)


BASE_PANEL = PROJECT_ROOT / "outputs" / "baseline_residual_24h" / "asset_residual_panel.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "baseline.yaml"))
    parser.add_argument("--panel", default=str(BASE_PANEL))
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_name = "turnover_experiments_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = ensure_dir(args.output_dir or PROJECT_ROOT / "outputs" / run_name)

    print(f"[1/4] Loading cached asset panel: {args.panel}", flush=True)
    panel = pd.read_csv(args.panel, parse_dates=["timestamp"])
    panel = prepare_scores(panel)

    pf_base = PortfolioConfig(
        rebalance_hours=int(cfg["portfolio"]["rebalance_hours"]),
        long_k=int(cfg["portfolio"]["long_k"]),
        short_k=int(cfg["portfolio"]["short_k"]),
        gross_exposure=float(cfg["portfolio"]["gross_exposure"]),
        one_way_fee_bps=float(cfg["portfolio"]["one_way_fee_bps"]),
        one_way_slippage_bps=float(cfg["portfolio"]["one_way_slippage_bps"]),
        neutralize_exposures=list(cfg["portfolio"]["neutralize_exposures"]),
        primary_horizon=int(cfg["labels"]["primary_horizon"]),
        execution_delay_bars=int(cfg["labels"]["execution_delay_bars"]),
    )
    wf_cfg = WalkForwardConfig(
        train_days=int(cfg["walk_forward"]["train_days"]),
        test_days=int(cfg["walk_forward"]["test_days"]),
        step_days=int(cfg["walk_forward"]["step_days"]),
        mode=str(cfg["walk_forward"].get("mode", "rolling")),
    )

    experiments = [
        (
            "00_baseline",
            "score_baseline",
            pf_base,
            "Reference: daily long top 4 / short bottom 4 with no turnover control.",
        ),
        (
            "01_ema_smoothing",
            "score_ema_025",
            pf_base,
            "Use EMA-smoothed score only.",
        ),
        (
            "02_ema_hysteresis",
            "score_ema_025",
            replace(pf_base, use_hysteresis=True),
            "EMA score plus separate entry/exit rank thresholds.",
        ),
        (
            "03_ema_hysteresis_min_hold",
            "score_ema_025",
            replace(pf_base, use_hysteresis=True, min_hold_rebalances=3),
            "EMA + hysteresis + minimum 3 rebalances holding period.",
        ),
        (
            "04_ema_hysteresis_min_hold_cost_gate",
            "score_ema_025",
            replace(pf_base, use_hysteresis=True, min_hold_rebalances=3, cost_gate_buffer=1.25),
            "EMA + hysteresis + min hold + train-calibrated cost-aware rebalance gate.",
        ),
    ]

    print("[2/4] Running four turnover-control experiments plus baseline", flush=True)
    summary_rows: list[dict[str, Any]] = []
    for name, score_col, pf_cfg, description in experiments:
        print(f"      running {name}", flush=True)
        exp_dir = ensure_dir(output_dir / name)
        folds, equity, positions, metrics = run_walk_forward(panel, wf_cfg, pf_cfg, score_col=score_col)
        write_experiment(exp_dir, folds, equity, positions, metrics, description, score_col)
        summary_rows.append({"experiment": name, "score_col": score_col, "description": description, **metrics})

    print("[3/4] Running AlphaGPT-style turnover-aware formula search", flush=True)
    formula_dir = ensure_dir(output_dir / "05_alphagpt_turnover_search")
    formula_result = run_formula_search(panel, wf_cfg, replace(pf_base, use_hysteresis=True, min_hold_rebalances=3, cost_gate_buffer=1.25))
    write_experiment(
        formula_dir,
        formula_result["folds"],
        formula_result["equity"],
        formula_result["positions"],
        formula_result["metrics"],
        "Per-fold formula search using past validation only, then test with low-turnover portfolio controls.",
        "per_fold_selected_formula",
    )
    formula_result["selections"].to_csv(formula_dir / "formula_selections.csv", index=False)
    summary_rows.append(
        {
            "experiment": "05_alphagpt_turnover_search",
            "score_col": "per_fold_selected_formula",
            "description": "Turnover-aware formula search on top of experiment 04 controls.",
            **formula_result["metrics"],
        }
    )

    print("[4/4] Writing comparison report", flush=True)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "summary.csv", index=False)
    write_report(output_dir, summary, formula_result["selections"])
    print(f"Done: {output_dir}", flush=True)


def prepare_scores(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.sort_values(["symbol", "timestamp"]).copy()
    out["score_baseline"] = -out["cs_zscore_resid_z"]
    out["score_ema_025"] = (
        out.groupby("symbol", group_keys=False)["score_baseline"]
        .apply(lambda s: s.ewm(alpha=0.25, adjust=False, min_periods=1).mean())
    )

    out["score_rank"] = out.groupby("timestamp")["score_baseline"].rank(pct=True, method="average")
    rank_change = out.groupby("symbol")["score_rank"].diff().abs()
    out["rank_stability_7d"] = -rank_change.groupby(out["symbol"]).rolling(7, min_periods=2).mean().reset_index(level=0, drop=True)
    out["score_rank_stability_7d"] = out.groupby("timestamp")["rank_stability_7d"].rank(pct=True, method="average")
    out["score_low_idio_vol"] = -out["cs_zscore_idio_vol_medium"]
    out["score_high_liquidity"] = out["cs_zscore_liquidity_z"]
    out["score_neg_mom_medium"] = -out["cs_zscore_resid_mom_medium"]
    out["score_neg_mom_short"] = -out["cs_zscore_resid_mom_short"]
    return out.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def run_formula_search(
    panel: pd.DataFrame,
    wf_cfg: WalkForwardConfig,
    pf_cfg: PortfolioConfig,
) -> dict[str, Any]:
    candidates = build_formula_candidates(panel)
    folds = make_walk_forward_folds(panel, wf_cfg, "score_baseline", pf_cfg.primary_horizon)
    fold_rows = []
    equity_parts = []
    position_parts = []
    selections = []

    for fold_id, fold in enumerate(folds, start=1):
        inner_train_end = fold["train_end"] - pd.Timedelta(days=60)
        if inner_train_end <= fold["train_start"]:
            inner_train_end = fold["train_start"] + (fold["train_end"] - fold["train_start"]) / 2
        validation_start = inner_train_end
        validation_end = fold["train_end"]

        scored = []
        for formula_name, score_col in candidates:
            coef = calibrate_score_to_return_coef(panel, fold["train_start"], inner_train_end, score_col, pf_cfg.primary_horizon)
            train_pf = replace(pf_cfg, score_to_return_coef=coef)
            _, _, validation_metrics = run_long_short_backtest(
                panel=panel,
                start=validation_start,
                end=validation_end,
                cfg=train_pf,
                score_col=score_col,
            )
            scored.append(
                {
                    "formula": formula_name,
                    "score_col": score_col,
                    "validation_fitness": formula_fitness(validation_metrics),
                    **{f"validation_{k}": v for k, v in validation_metrics.items()},
                }
            )

        scored_df = pd.DataFrame(scored).sort_values("validation_fitness", ascending=False)
        best = scored_df.iloc[0].to_dict()
        test_coef = calibrate_score_to_return_coef(panel, fold["train_start"], fold["train_end"], best["score_col"], pf_cfg.primary_horizon)
        test_pf = replace(pf_cfg, score_to_return_coef=test_coef)
        equity, positions, test_metrics = run_long_short_backtest(
            panel=panel,
            start=fold["test_start"],
            end=fold["test_end"],
            cfg=test_pf,
            score_col=str(best["score_col"]),
        )

        if not equity.empty:
            equity = equity.copy()
            equity["fold"] = fold_id
            equity["formula"] = best["formula"]
            positions = positions.copy()
            positions["fold"] = fold_id
            positions["formula"] = best["formula"]
            equity_parts.append(equity)
            position_parts.append(positions)

        fold_rows.append({"fold": fold_id, **fold, **test_metrics, "selected_formula": best["formula"]})
        selections.append({"fold": fold_id, **fold, **best, "test_score_to_return_coef": test_coef})

    fold_summary = pd.DataFrame(fold_rows)
    equity_curve = pd.concat(equity_parts, ignore_index=True) if equity_parts else pd.DataFrame()
    positions = pd.concat(position_parts, ignore_index=True) if position_parts else pd.DataFrame()
    if not equity_curve.empty:
        equity_curve = equity_curve.sort_values("timestamp").reset_index(drop=True)
        equity_curve["aggregate_equity"] = (1.0 + equity_curve["period_return"]).cumprod()
        metric_curve = equity_curve.copy()
        metric_curve["equity"] = metric_curve["aggregate_equity"]
        metrics = summarize_equity_curve(metric_curve, pf_cfg.rebalance_hours)
    else:
        metrics = summarize_equity_curve(equity_curve, pf_cfg.rebalance_hours)
    metrics["fold_count"] = int(len(folds))

    return {
        "folds": fold_summary,
        "equity": equity_curve,
        "positions": positions,
        "metrics": metrics,
        "selections": pd.DataFrame(selections),
    }


def build_formula_candidates(panel: pd.DataFrame) -> list[tuple[str, str]]:
    specs = [
        ("NEG(cs_zscore_resid_z)", "score_f_base", lambda x: x["score_baseline"]),
        ("EMA(NEG(cs_zscore_resid_z),0.25)", "score_f_ema", lambda x: x["score_ema_025"]),
        ("NEG(cs_zscore_resid_mom_medium)", "score_f_neg_mom_medium", lambda x: x["score_neg_mom_medium"]),
        ("ADD(base,0.35*stability)", "score_f_base_stability", lambda x: x["score_baseline"] + 0.35 * x["score_rank_stability_7d"]),
        ("ADD(ema,0.35*stability)", "score_f_ema_stability", lambda x: x["score_ema_025"] + 0.35 * x["score_rank_stability_7d"]),
        ("ADD(ema,0.20*liquidity)", "score_f_ema_liquidity", lambda x: x["score_ema_025"] + 0.20 * x["score_high_liquidity"]),
        ("SUB(ema,0.20*idio_vol)", "score_f_ema_low_vol", lambda x: x["score_ema_025"] + 0.20 * x["score_low_idio_vol"]),
        (
            "ADD(ema,0.25*stability,0.15*liquidity)",
            "score_f_ema_stability_liq",
            lambda x: x["score_ema_025"] + 0.25 * x["score_rank_stability_7d"] + 0.15 * x["score_high_liquidity"],
        ),
        (
            "ADD(ema,0.25*stability,0.15*low_vol)",
            "score_f_ema_stability_low_vol",
            lambda x: x["score_ema_025"] + 0.25 * x["score_rank_stability_7d"] + 0.15 * x["score_low_idio_vol"],
        ),
        (
            "ADD(0.70*ema,0.30*neg_mom_medium)",
            "score_f_ema_mom_medium",
            lambda x: 0.70 * x["score_ema_025"] + 0.30 * x["score_neg_mom_medium"],
        ),
        (
            "ADD(0.60*ema,0.25*stability,0.15*neg_mom_medium)",
            "score_f_ema_stability_mom",
            lambda x: 0.60 * x["score_ema_025"] + 0.25 * x["score_rank_stability_7d"] + 0.15 * x["score_neg_mom_medium"],
        ),
    ]

    for _, score_col, fn in specs:
        panel[score_col] = fn(panel)
    return [(formula, score_col) for formula, score_col, _ in specs]


def formula_fitness(metrics: dict[str, float]) -> float:
    return (
        100.0 * metrics.get("total_return", 0.0)
        + 3.0 * metrics.get("sharpe", 0.0)
        + 20.0 * metrics.get("mean_rank_ic", 0.0)
        - 50.0 * abs(metrics.get("max_drawdown", 0.0))
        - 10.0 * metrics.get("mean_turnover", 0.0)
        - 1.5 * metrics.get("cost_drag", 0.0)
    )


def write_experiment(
    output_dir: Path,
    folds: pd.DataFrame,
    equity: pd.DataFrame,
    positions: pd.DataFrame,
    metrics: dict[str, float],
    description: str,
    score_col: str,
) -> None:
    folds.to_csv(output_dir / "walk_forward_folds.csv", index=False)
    equity.to_csv(output_dir / "equity_curve.csv", index=False)
    positions.to_csv(output_dir / "positions.csv", index=False)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(to_jsonable(metrics), fh, indent=2, ensure_ascii=False)
    (output_dir / "README.md").write_text(
        f"# {output_dir.name}\n\n"
        f"{description}\n\n"
        f"- score_col: `{score_col}`\n"
        f"- metrics: `metrics.json`\n",
        encoding="utf-8",
    )


def write_report(output_dir: Path, summary: pd.DataFrame, selections: pd.DataFrame) -> None:
    cols = [
        "experiment",
        "total_return",
        "gross_total_return_before_cost",
        "sharpe",
        "max_drawdown",
        "mean_rank_ic",
        "mean_turnover",
        "cost_drag",
        "target_apply_rate",
    ]
    lines = ["# Turnover Experiments", "", "## Summary", "", "```text", summary[cols].to_string(index=False), "```", ""]
    if not selections.empty:
        counts = selections["formula"].value_counts().reset_index()
        counts.columns = ["formula", "selected_folds"]
        lines.extend(["## Formula Search Selection Counts", "", "```text", counts.to_string(index=False), "```", ""])
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(key): to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(value) for value in obj]
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


if __name__ == "__main__":
    main()
