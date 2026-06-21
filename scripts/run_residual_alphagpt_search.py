#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from crypto_stat_arb.config import ensure_dir, load_config
from crypto_stat_arb.features import add_forward_labels
from crypto_stat_arb.portfolio import PortfolioConfig, run_long_short_backtest, summarize_equity_curve
from crypto_stat_arb.walk_forward import WalkForwardConfig, make_walk_forward_folds, run_walk_forward
from run_turnover_experiments import prepare_scores, to_jsonable, write_experiment


BASE_PANEL = PROJECT_ROOT / "outputs" / "baseline_residual_24h" / "asset_residual_panel.csv"
COMMITTEE_COL = "_score_committee"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "baseline.yaml"))
    parser.add_argument("--panel", default=str(BASE_PANEL))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--validation-days", type=int, default=60)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_name = "residual_alphagpt_search_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = ensure_dir(args.output_dir or PROJECT_ROOT / "outputs" / run_name)

    print(f"[1/5] Loading cached asset panel: {args.panel}", flush=True)
    panel = pd.read_csv(args.panel, parse_dates=["timestamp"])

    settings = [
        {"name": "h048_minhold5d", "horizon": 48, "min_hold_days": 5},
        {"name": "h024_minhold7d", "horizon": 24, "min_hold_days": 7},
    ]
    horizons = sorted({int(s["horizon"]) for s in settings})

    print("[2/5] Preparing residual alpha candidate scores", flush=True)
    panel = prepare_alpha_panel(panel, horizons, int(cfg["labels"]["execution_delay_bars"]))
    candidates = build_candidate_scores(panel)
    candidate_meta = pd.DataFrame([candidate | {"complexity": formula_complexity(candidate["formula"])} for candidate in candidates])
    candidate_meta.to_csv(output_dir / "candidate_formulas.csv", index=False)
    print(f"      candidates={len(candidates)}", flush=True)

    wf_cfg = WalkForwardConfig(
        train_days=int(cfg["walk_forward"]["train_days"]),
        test_days=int(cfg["walk_forward"]["test_days"]),
        step_days=int(cfg["walk_forward"]["step_days"]),
        mode=str(cfg["walk_forward"].get("mode", "rolling")),
    )
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
        use_hysteresis=True,
    )

    print("[3/5] Running searches", flush=True)
    summary_rows: list[dict[str, Any]] = []
    selection_frames: list[pd.DataFrame] = []
    for setting in settings:
        name = str(setting["name"])
        horizon = int(setting["horizon"])
        min_hold_days = int(setting["min_hold_days"])
        min_hold_rebalances = max(1, math.ceil(min_hold_days * 24 / horizon))
        pf_cfg = replace(
            pf_base,
            rebalance_hours=horizon,
            primary_horizon=horizon,
            min_hold_rebalances=min_hold_rebalances,
        )
        setting_dir = ensure_dir(output_dir / name)
        print(
            f"      {name}: horizon={horizon}h, min_hold={min_hold_days}d ({min_hold_rebalances} rebalances)",
            flush=True,
        )

        baseline = run_baseline(panel, wf_cfg, pf_cfg, setting_dir)
        summary_rows.append({"setting": name, "variant": "baseline_ema", **setting, **baseline["metrics"]})

        single = run_formula_search(
            panel=panel,
            wf_cfg=wf_cfg,
            pf_cfg=pf_cfg,
            candidates=candidates,
            validation_days=int(args.validation_days),
            variant="single",
        )
        write_search_result(setting_dir / "single_formula", single, "Best single formula selected per fold.")
        summary_rows.append({"setting": name, "variant": "single_formula", **setting, **single["metrics"]})
        selection_frames.append(single["selections"].assign(setting=name, variant="single_formula"))

        committee = run_formula_search(
            panel=panel,
            wf_cfg=wf_cfg,
            pf_cfg=pf_cfg,
            candidates=candidates,
            validation_days=int(args.validation_days),
            variant="committee",
        )
        write_search_result(setting_dir / "committee_top3", committee, "Top-3 low-correlation formula committee selected per fold.")
        summary_rows.append({"setting": name, "variant": "committee_top3", **setting, **committee["metrics"]})
        selection_frames.append(committee["selections"].assign(setting=name, variant="committee_top3"))

    print("[4/5] Writing aggregate reports", flush=True)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "summary.csv", index=False)
    selections = pd.concat(selection_frames, ignore_index=True) if selection_frames else pd.DataFrame()
    selections.to_csv(output_dir / "formula_selections.csv", index=False)
    write_review(output_dir, summary, selections)

    print("[5/5] Done", flush=True)
    print(f"Done: {output_dir}", flush=True)


def prepare_alpha_panel(panel: pd.DataFrame, horizons: list[int], execution_delay_bars: int) -> pd.DataFrame:
    out = prepare_scores(panel)
    missing = [h for h in horizons if f"label_resid_ret_h{h}" not in out.columns or f"fwd_simple_ret_h{h}" not in out.columns]
    if missing:
        out = add_forward_labels(out, horizons=sorted(set(horizons)), execution_delay_bars=execution_delay_bars)

    out = out.sort_values(["symbol", "timestamp"]).copy()
    out["score_rank_base"] = out.groupby("timestamp")["score_baseline"].rank(pct=True, method="average")
    rank_change = out.groupby("symbol")["score_rank_base"].diff().abs()
    for name, hours in [("1d", 24), ("3d", 72), ("7d", 168)]:
        raw = rank_change.groupby(out["symbol"]).rolling(hours, min_periods=max(8, hours // 4)).mean().reset_index(level=0, drop=True)
        out[f"rank_instability_{name}"] = raw
        out[f"score_rank_stability_{name}"] = -cross_sectional_zscore(out, raw)

    for beta_col in ["beta_btc", "beta_eth", "beta_market", "beta_sector"]:
        beta_change = out.groupby("symbol")[beta_col].diff().abs()
        out[f"{beta_col}_instability_7d"] = (
            beta_change.groupby(out["symbol"]).rolling(168, min_periods=48).mean().reset_index(level=0, drop=True)
        )

    beta_instability = np.sqrt(
        sum(out[f"{col}_instability_7d"].fillna(0.0) ** 2 for col in ["beta_btc", "beta_eth", "beta_market", "beta_sector"])
    )
    out["score_beta_stability_7d"] = -cross_sectional_zscore(out, beta_instability)
    out["score_tanh_ema"] = np.tanh(out["score_ema_025"])
    out["score_soft_ema"] = out["score_ema_025"] / (1.0 + out["cs_zscore_resid_z"].abs().clip(upper=5.0))
    out["score_turn_short"] = out["cs_zscore_resid_mom_short"]
    out["score_turn_medium"] = out["cs_zscore_resid_mom_medium"]
    out["score_low_realized_vol"] = -out["cs_zscore_realized_vol_medium"]
    return out.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def cross_sectional_zscore(panel: pd.DataFrame, values: pd.Series) -> pd.Series:
    tmp = pd.DataFrame({"timestamp": panel["timestamp"], "value": values})
    mean = tmp.groupby("timestamp")["value"].transform("mean")
    std = tmp.groupby("timestamp")["value"].transform("std").replace(0.0, np.nan)
    return (tmp["value"] - mean) / std


def build_candidate_scores(panel: pd.DataFrame) -> list[dict[str, str]]:
    specs: list[tuple[str, str, Any]] = [
        ("BASE:NEG(cs_zscore_resid_z)", "alpha_base", lambda x: x["score_baseline"]),
        ("EMA(BASE,0.25)", "alpha_ema", lambda x: x["score_ema_025"]),
        ("TANH(EMA)", "alpha_tanh_ema", lambda x: x["score_tanh_ema"]),
        ("SOFT_EXTREME(EMA)", "alpha_soft_ema", lambda x: x["score_soft_ema"]),
        ("TURN_SHORT", "alpha_turn_short", lambda x: x["score_turn_short"]),
        ("TURN_MEDIUM", "alpha_turn_medium", lambda x: x["score_turn_medium"]),
        ("ADD(EMA,0.25*TURN_SHORT)", "alpha_ema_turn_short", lambda x: x["score_ema_025"] + 0.25 * x["score_turn_short"]),
        ("ADD(EMA,0.25*TURN_MEDIUM)", "alpha_ema_turn_medium", lambda x: x["score_ema_025"] + 0.25 * x["score_turn_medium"]),
        ("ADD(0.70*EMA,0.30*TURN_SHORT)", "alpha_mix_turn_short", lambda x: 0.70 * x["score_ema_025"] + 0.30 * x["score_turn_short"]),
        ("ADD(0.70*EMA,0.30*TURN_MEDIUM)", "alpha_mix_turn_medium", lambda x: 0.70 * x["score_ema_025"] + 0.30 * x["score_turn_medium"]),
    ]

    quality_terms = [
        ("STABILITY_1D", "score_rank_stability_1d", 0.20),
        ("STABILITY_3D", "score_rank_stability_3d", 0.25),
        ("STABILITY_7D", "score_rank_stability_7d", 0.30),
        ("LIQUIDITY", "score_high_liquidity", 0.20),
        ("LOW_IDIO_VOL", "score_low_idio_vol", 0.20),
        ("LOW_REALIZED_VOL", "score_low_realized_vol", 0.15),
        ("BETA_STABILITY", "score_beta_stability_7d", 0.15),
    ]

    for label, col, weight in quality_terms:
        specs.append((f"ADD(EMA,{weight:.2f}*{label})", f"alpha_ema_{col}", lambda x, c=col, w=weight: x["score_ema_025"] + w * x[c]))
        specs.append((f"ADD(SOFT_EMA,{weight:.2f}*{label})", f"alpha_soft_{col}", lambda x, c=col, w=weight: x["score_soft_ema"] + w * x[c]))

    combo_specs = [
        (
            "ADD(EMA,0.20*STABILITY_7D,0.15*LIQUIDITY)",
            "alpha_ema_stability_liq",
            lambda x: x["score_ema_025"] + 0.20 * x["score_rank_stability_7d"] + 0.15 * x["score_high_liquidity"],
        ),
        (
            "ADD(EMA,0.20*STABILITY_7D,0.15*LOW_IDIO_VOL)",
            "alpha_ema_stability_lowvol",
            lambda x: x["score_ema_025"] + 0.20 * x["score_rank_stability_7d"] + 0.15 * x["score_low_idio_vol"],
        ),
        (
            "ADD(EMA,0.20*STABILITY_7D,0.15*TURN_SHORT)",
            "alpha_ema_stability_turn",
            lambda x: x["score_ema_025"] + 0.20 * x["score_rank_stability_7d"] + 0.15 * x["score_turn_short"],
        ),
        (
            "ADD(EMA,0.15*LIQUIDITY,0.15*LOW_IDIO_VOL)",
            "alpha_ema_liq_lowvol",
            lambda x: x["score_ema_025"] + 0.15 * x["score_high_liquidity"] + 0.15 * x["score_low_idio_vol"],
        ),
        (
            "ADD(EMA,0.15*STABILITY_7D,0.10*LIQUIDITY,0.10*LOW_IDIO_VOL)",
            "alpha_ema_quality",
            lambda x: x["score_ema_025"]
            + 0.15 * x["score_rank_stability_7d"]
            + 0.10 * x["score_high_liquidity"]
            + 0.10 * x["score_low_idio_vol"],
        ),
        (
            "ADD(SOFT_EMA,0.15*STABILITY_7D,0.10*LIQUIDITY,0.10*LOW_IDIO_VOL)",
            "alpha_soft_quality",
            lambda x: x["score_soft_ema"]
            + 0.15 * x["score_rank_stability_7d"]
            + 0.10 * x["score_high_liquidity"]
            + 0.10 * x["score_low_idio_vol"],
        ),
        (
            "ADD(0.65*EMA,0.20*TURN_SHORT,0.15*STABILITY_7D)",
            "alpha_ema_turn_stability",
            lambda x: 0.65 * x["score_ema_025"] + 0.20 * x["score_turn_short"] + 0.15 * x["score_rank_stability_7d"],
        ),
        (
            "ADD(0.60*EMA,0.20*TURN_SHORT,0.10*LIQUIDITY,0.10*LOW_IDIO_VOL)",
            "alpha_ema_turn_quality",
            lambda x: 0.60 * x["score_ema_025"]
            + 0.20 * x["score_turn_short"]
            + 0.10 * x["score_high_liquidity"]
            + 0.10 * x["score_low_idio_vol"],
        ),
    ]
    specs.extend(combo_specs)

    candidates = []
    seen_cols: set[str] = set()
    for formula, score_col, fn in specs:
        if score_col in seen_cols:
            raise ValueError(f"Duplicate score column: {score_col}")
        panel[score_col] = fn(panel)
        candidates.append({"formula": formula, "score_col": score_col})
        seen_cols.add(score_col)
    return candidates


def run_baseline(panel: pd.DataFrame, wf_cfg: WalkForwardConfig, pf_cfg: PortfolioConfig, output_dir: Path) -> dict[str, Any]:
    baseline_dir = ensure_dir(output_dir / "baseline_ema")
    folds, equity, positions, metrics = run_walk_forward(panel, wf_cfg, pf_cfg, score_col="score_ema_025")
    write_experiment(
        baseline_dir,
        folds,
        equity,
        positions,
        metrics,
        "Fixed baseline: EMA score + hysteresis + configured min hold.",
        "score_ema_025",
    )
    return {"folds": folds, "equity": equity, "positions": positions, "metrics": metrics}


def run_formula_search(
    panel: pd.DataFrame,
    wf_cfg: WalkForwardConfig,
    pf_cfg: PortfolioConfig,
    candidates: list[dict[str, str]],
    validation_days: int,
    variant: str,
) -> dict[str, Any]:
    folds = make_walk_forward_folds(panel, wf_cfg, "score_ema_025", pf_cfg.primary_horizon)
    fold_rows: list[dict[str, Any]] = []
    equity_parts: list[pd.DataFrame] = []
    position_parts: list[pd.DataFrame] = []
    selection_rows: list[dict[str, Any]] = []

    for fold_id, fold in enumerate(folds, start=1):
        validation_start = max(fold["train_start"], fold["train_end"] - pd.Timedelta(days=validation_days))
        scored = score_candidates_on_validation(panel, candidates, pf_cfg, validation_start, fold["train_end"])

        if variant == "single":
            selected = [scored.iloc[0].to_dict()]
            score_col = str(selected[0]["score_col"])
            selected_formulas = str(selected[0]["formula"])
        elif variant == "committee":
            selected = select_low_corr_committee(panel, scored, validation_start, fold["train_end"], max_size=3, max_abs_corr=0.85)
            score_col = COMMITTEE_COL
            selected_formulas = " | ".join(str(item["formula"]) for item in selected)
            panel[COMMITTEE_COL] = committee_score(panel, [str(item["score_col"]) for item in selected])
        else:
            raise ValueError(f"Unknown variant: {variant}")

        equity, positions, test_metrics = run_long_short_backtest(
            panel=panel,
            start=fold["test_start"],
            end=fold["test_end"],
            cfg=pf_cfg,
            score_col=score_col,
        )

        if not equity.empty:
            equity = equity.copy()
            equity["fold"] = fold_id
            equity["selected_formulas"] = selected_formulas
            positions = positions.copy()
            positions["fold"] = fold_id
            positions["selected_formulas"] = selected_formulas
            equity_parts.append(equity)
            position_parts.append(positions)

        fold_rows.append({"fold": fold_id, **fold, **test_metrics, "selected_formulas": selected_formulas})
        selection_rows.append(
            {
                "fold": fold_id,
                **fold,
                "selected_formulas": selected_formulas,
                "selected_score_cols": " | ".join(str(item["score_col"]) for item in selected),
                "validation_fitness": float(np.mean([item["validation_fitness"] for item in selected])),
                "validation_total_return": float(np.mean([item["validation_total_return"] for item in selected])),
                "validation_sharpe": float(np.mean([item["validation_sharpe"] for item in selected])),
                "validation_mean_turnover": float(np.mean([item["validation_mean_turnover"] for item in selected])),
            }
        )

    fold_summary = pd.DataFrame(fold_rows)
    equity_curve = pd.concat(equity_parts, ignore_index=True) if equity_parts else pd.DataFrame()
    positions = pd.concat(position_parts, ignore_index=True) if position_parts else pd.DataFrame()
    metrics = aggregate_oos_metrics(equity_curve, pf_cfg.rebalance_hours)
    metrics["fold_count"] = int(len(folds))
    if not fold_summary.empty:
        metrics["positive_fold_rate"] = float((fold_summary["total_return"] > 0).mean())

    return {
        "folds": fold_summary,
        "equity": equity_curve,
        "positions": positions,
        "metrics": metrics,
        "selections": pd.DataFrame(selection_rows),
    }


def score_candidates_on_validation(
    panel: pd.DataFrame,
    candidates: list[dict[str, str]],
    pf_cfg: PortfolioConfig,
    validation_start: pd.Timestamp,
    validation_end: pd.Timestamp,
) -> pd.DataFrame:
    rows = []
    for candidate in candidates:
        _, _, metrics = run_long_short_backtest(
            panel=panel,
            start=validation_start,
            end=validation_end,
            cfg=pf_cfg,
            score_col=candidate["score_col"],
        )
        complexity = formula_complexity(candidate["formula"])
        rows.append(
            {
                **candidate,
                "complexity": complexity,
                "validation_fitness": formula_fitness(metrics, complexity),
                **{f"validation_{key}": value for key, value in metrics.items()},
            }
        )
    return pd.DataFrame(rows).sort_values("validation_fitness", ascending=False).reset_index(drop=True)


def formula_fitness(metrics: dict[str, float], complexity: float) -> float:
    return (
        100.0 * metrics.get("total_return", 0.0)
        + 3.0 * metrics.get("sharpe", 0.0)
        + 20.0 * metrics.get("gross_total_return_before_cost", 0.0)
        - 40.0 * abs(metrics.get("max_drawdown", 0.0))
        - 20.0 * metrics.get("cost_drag", 0.0)
        - 8.0 * metrics.get("mean_turnover", 0.0)
        + 10.0 * metrics.get("positive_period_rate", 0.0)
        - 0.15 * complexity
    )


def formula_complexity(formula: str) -> int:
    tokens = formula.replace("(", ",").replace(")", ",").replace("*", ",").replace("+", ",").split(",")
    return sum(1 for token in tokens if token.strip())


def select_low_corr_committee(
    panel: pd.DataFrame,
    scored: pd.DataFrame,
    validation_start: pd.Timestamp,
    validation_end: pd.Timestamp,
    max_size: int,
    max_abs_corr: float,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    validation = panel[(panel["timestamp"] >= validation_start) & (panel["timestamp"] < validation_end)]
    for _, row in scored.iterrows():
        item = row.to_dict()
        if len(selected) == 0:
            selected.append(item)
        elif all(abs(score_corr(validation, str(item["score_col"]), str(prev["score_col"]))) <= max_abs_corr for prev in selected):
            selected.append(item)
        if len(selected) >= max_size:
            break

    if len(selected) < max_size:
        selected_cols = {str(item["score_col"]) for item in selected}
        for _, row in scored.iterrows():
            if str(row["score_col"]) not in selected_cols:
                selected.append(row.to_dict())
                selected_cols.add(str(row["score_col"]))
            if len(selected) >= max_size:
                break
    return selected


def score_corr(frame: pd.DataFrame, left: str, right: str) -> float:
    valid = frame[[left, right]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < 20:
        return 1.0
    return float(valid[left].rank().corr(valid[right].rank()))


def committee_score(panel: pd.DataFrame, score_cols: list[str]) -> pd.Series:
    ranks = []
    grouped = panel.groupby("timestamp", sort=False)
    for col in score_cols:
        ranks.append(grouped[col].rank(pct=True, method="average"))
    return pd.concat(ranks, axis=1).mean(axis=1)


def aggregate_oos_metrics(equity_curve: pd.DataFrame, rebalance_hours: int) -> dict[str, float]:
    if equity_curve.empty:
        metrics = summarize_equity_curve(equity_curve, rebalance_hours)
        return metrics
    equity_curve = equity_curve.sort_values("timestamp").reset_index(drop=True).copy()
    equity_curve["aggregate_equity"] = (1.0 + equity_curve["period_return"]).cumprod()
    metric_curve = equity_curve.copy()
    metric_curve["equity"] = metric_curve["aggregate_equity"]
    return summarize_equity_curve(metric_curve, rebalance_hours)


def write_search_result(output_dir: Path, result: dict[str, Any], description: str) -> None:
    ensure_dir(output_dir)
    result["folds"].to_csv(output_dir / "walk_forward_folds.csv", index=False)
    result["equity"].to_csv(output_dir / "equity_curve.csv", index=False)
    result["positions"].to_csv(output_dir / "positions.csv", index=False)
    result["selections"].to_csv(output_dir / "formula_selections.csv", index=False)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(to_jsonable(result["metrics"]), fh, indent=2, ensure_ascii=False)
    (output_dir / "README.md").write_text(f"# {output_dir.name}\n\n{description}\n", encoding="utf-8")


def write_review(output_dir: Path, summary: pd.DataFrame, selections: pd.DataFrame) -> None:
    display_cols = [
        "setting",
        "variant",
        "horizon",
        "min_hold_days",
        "total_return",
        "gross_total_return_before_cost",
        "sharpe",
        "max_drawdown",
        "mean_rank_ic",
        "mean_turnover",
        "cost_drag",
        "positive_fold_rate",
    ]
    summary_sorted = summary.sort_values(["total_return", "sharpe"], ascending=[False, False]).reset_index(drop=True)
    best = summary_sorted.iloc[0].to_dict()
    lines = [
        "# Residual AlphaGPT Search Review",
        "",
        "## Design",
        "",
        "- Search settings: `48h/5d` and `24h/7d`.",
        "- Formula selection uses only the last 60 days of each training window as validation.",
        "- Test windows are untouched until after formula or committee selection.",
        "- Candidate family: residual cheapness, turning confirmation, rank stability, liquidity, low idiosyncratic volatility, beta stability.",
        "- Committee variant picks top validation formulas subject to a low-correlation preference, then averages cross-sectional formula ranks.",
        "",
        "## Summary",
        "",
        "```text",
        summary_sorted[display_cols].to_string(index=False),
        "```",
        "",
        "## Best By Net Return",
        "",
        f"- setting: {best['setting']}",
        f"- variant: {best['variant']}",
        f"- total_return: {best['total_return']}",
        f"- gross_total_return_before_cost: {best['gross_total_return_before_cost']}",
        f"- sharpe: {best['sharpe']}",
        f"- max_drawdown: {best['max_drawdown']}",
        f"- mean_turnover: {best['mean_turnover']}",
        f"- cost_drag: {best['cost_drag']}",
        "",
    ]

    if not selections.empty:
        counts = (
            selections.groupby(["setting", "variant", "selected_formulas"])
            .size()
            .reset_index(name="selected_folds")
            .sort_values(["setting", "variant", "selected_folds"], ascending=[True, True, False])
        )
        lines.extend(
            [
                "## Selection Counts",
                "",
                "```text",
                counts.to_string(index=False),
                "```",
                "",
            ]
        )

    (output_dir / "REVIEW.md").write_text("\n".join(lines), encoding="utf-8")
    summary_sorted.to_csv(output_dir / "summary_sorted.csv", index=False)
    with (output_dir / "best_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(to_jsonable(best), fh, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
