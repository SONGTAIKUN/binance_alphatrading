#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
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
import run_p3d_p4_gated_rerank_alphagpt as p3d
import run_p4_nested_walkforward_selection as p4
import run_p15c_extreme_cluster_turnover as p15c
import run_p27_no_sector_native_factor_remine as p27
import run_p28_stable_family_ensemble as p28
from run_p18c_factor_selector_portfolio import load_existing_result
from run_p26_full_nested_pipeline import combine_test_results, parallel_fold_map, validation_split
from run_turnover_experiments import to_jsonable


EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "baseline_locked_v1"
LONG_ROOT = EXPERIMENT_ROOT / "outputs" / "long_history_nested_oos"
P28_ROOT = EXPERIMENT_ROOT / "outputs" / "p28_stable_family_ensemble"
DEFAULT_CONFIG = EXPERIMENT_ROOT / "configs" / "baseline_locked_v1.yaml"
DEFAULT_PANEL = P28_ROOT / "inputs" / "p28_prepared_panel.pkl"
DEFAULT_FOLDS = LONG_ROOT / "inputs" / "long_formula_selections.csv"
DEFAULT_P9_SELECTED = LONG_ROOT / "p9_nested" / "selected_incremental_by_fold.csv"
DEFAULT_P15C_SELECTED = LONG_ROOT / "p15c_nested" / "selected_extreme_cluster_by_fold.csv"
DEFAULT_P15C_RUN = LONG_ROOT / "p15c_nested" / "nested_extreme_cluster_run"
DEFAULT_RISK_RESIDUAL = (
    EXPERIMENT_ROOT / "outputs" / "p26_full_nested_pipeline" / "inputs" / "p15c_legacy_residual_wide.pkl"
)
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "outputs" / "p29_tail_aware_s1_selection"

HORIZON = 72
MAX_LABEL_HORIZON = 168
TRAIN_BLOCK_DAYS = 28
EVAL_BLOCK_DAYS = 14
S1_SMOOTHING_ALPHA = 0.25


@dataclass(frozen=True)
class TailVariant:
    name: str
    long_weight: float
    short_weight: float
    extreme_threshold: float | None = None


_PANEL: pd.DataFrame | None = None
_CFG: dict[str, Any] | None = None
_S1_MEMBERS: dict[str, p27.NativeFormula] = {}
_TAIL_SELECTION: pd.DataFrame | None = None
_TAIL_VALIDATION: pd.DataFrame | None = None
_P9_SELECTED: pd.DataFrame | None = None
_P15C_SELECTED: pd.DataFrame | None = None
_RISK_RESIDUAL: pd.DataFrame | None = None


def main() -> None:
    parser = argparse.ArgumentParser(description="P29 tail-aware S1 factor selection and nested OOS replay.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--panel", default=str(DEFAULT_PANEL))
    parser.add_argument("--folds", default=str(DEFAULT_FOLDS))
    parser.add_argument("--p9-selected", default=str(DEFAULT_P9_SELECTED))
    parser.add_argument("--p15c-selected", default=str(DEFAULT_P15C_SELECTED))
    parser.add_argument("--p15c-run", default=str(DEFAULT_P15C_RUN))
    parser.add_argument("--risk-residual", default=str(DEFAULT_RISK_RESIDUAL))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument("--cost-bps", type=float, nargs="+", default=[0, 6, 8, 15, 20, 30])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_dir = ensure_dir(Path(args.output_dir).resolve())
    config_path = Path(args.config).resolve()
    panel_path = Path(args.panel).resolve()
    folds = pd.read_csv(
        Path(args.folds).resolve(),
        parse_dates=["train_start", "train_end", "test_start", "test_end"],
    ).sort_values("fold")
    selected_p9 = pd.read_csv(Path(args.p9_selected).resolve())
    selected_p15c = pd.read_csv(Path(args.p15c_selected).resolve())
    s1 = p28.build_family_definitions()["S1_downside_snapback"]
    members = {item.name: item for item in s1.members}
    variants = build_variants()
    pd.DataFrame([{"family": s1.name, **item.to_row()} for item in s1.members]).to_csv(
        output_dir / "s1_fixed_members.csv", index=False
    )
    pd.DataFrame([item.__dict__ for item in variants]).to_csv(output_dir / "tail_execution_grid.csv", index=False)

    workers = max(1, int(args.workers))
    member_path = output_dir / "tail_member_train_metrics.csv"
    selection_path = output_dir / "tail_committee_selection_by_fold.csv"
    validation_path = output_dir / "tail_committee_validation_metrics.csv"
    test_path = output_dir / "tail_committee_test_oos_metrics.csv"
    print(f"[1/5] Train-only S1 tail member selection ({len(folds)} folds, {workers} workers)", flush=True)
    if not args.force and all(path.exists() for path in [member_path, selection_path, validation_path, test_path]):
        print("      using cached tail diagnostics", flush=True)
        member_metrics = pd.read_csv(member_path)
        tail_selection = pd.read_csv(selection_path)
        tail_validation = pd.read_csv(validation_path)
        tail_test = pd.read_csv(test_path)
    else:
        rows = parallel_fold_map(
            worker=tail_diagnostic_worker,
            folds=folds,
            workers=workers,
            initializer=init_tail_worker,
            initargs=(str(panel_path), str(config_path), members),
        )
        member_metrics = pd.concat([item["members"] for item in rows], ignore_index=True)
        tail_selection = pd.concat([item["selection"] for item in rows], ignore_index=True)
        tail_validation = pd.concat([item["validation"] for item in rows], ignore_index=True)
        tail_test = pd.concat([item["test"] for item in rows], ignore_index=True)
        member_metrics.to_csv(member_path, index=False)
        tail_selection.to_csv(selection_path, index=False)
        tail_validation.to_csv(validation_path, index=False)
        tail_test.to_csv(test_path, index=False)
    tail_summary = summarize_tail_oos(tail_test, tail_selection)
    tail_summary.to_csv(output_dir / "tail_committee_oos_summary.csv", index=False)

    print("[2/5] Validation-only side/weight selection and test replay", flush=True)
    flag = output_dir / "trade_complete.flag"
    selected_variant_path = output_dir / "selected_tail_variant_by_fold.csv"
    variant_validation_path = output_dir / "tail_variant_validation_metrics.csv"
    run_dirs = {item.name: output_dir / item.name / "nested_test_run" for item in variants}
    run_dirs["P29_nested"] = output_dir / "P29_nested" / "nested_test_run"
    replay_dir = output_dir / "P15c_replay_check" / "nested_test_run"
    cached = (
        not args.force
        and flag.exists()
        and selected_variant_path.exists()
        and variant_validation_path.exists()
        and all((path / "metrics.json").exists() for path in run_dirs.values())
        and (replay_dir / "metrics.json").exists()
    )
    if cached:
        print("      using cached trading results", flush=True)
        variant_validation = pd.read_csv(variant_validation_path)
        selected_variant = pd.read_csv(selected_variant_path)
        results = {name: load_existing_result(path) for name, path in run_dirs.items()}
        baseline_replay = load_existing_result(replay_dir)
    else:
        rows = parallel_fold_map(
            worker=tail_trade_worker,
            folds=folds,
            workers=workers,
            initializer=init_tail_trade_worker,
            initargs=(
                str(panel_path), str(config_path), members, tail_selection, tail_validation,
                selected_p9, selected_p15c, str(Path(args.risk_residual).resolve()),
            ),
        )
        variant_validation = pd.concat([item["validation"] for item in rows], ignore_index=True)
        selected_variant = pd.concat([item["selected"] for item in rows], ignore_index=True).sort_values("fold")
        variant_validation.to_csv(variant_validation_path, index=False)
        selected_variant.to_csv(selected_variant_path, index=False)
        results = {}
        for variant in variants:
            result = combine_test_results([item["tests"][variant.name] for item in rows])
            results[variant.name] = result
            p3d.write_run_result(ensure_dir(run_dirs[variant.name]), result)
        nested = combine_test_results([item["nested"] for item in rows])
        results["P29_nested"] = nested
        p3d.write_run_result(ensure_dir(run_dirs["P29_nested"]), nested)
        baseline_replay = combine_test_results([item["baseline_replay"] for item in rows])
        p3d.write_run_result(ensure_dir(replay_dir), baseline_replay)
        flag.write_text("complete\n", encoding="utf-8")

    print("[3/5] Matched comparison and strict audits", flush=True)
    p15c_result = load_existing_result(Path(args.p15c_run).resolve())
    replay_check = p28.compare_baseline_replay(p15c_result, baseline_replay)
    replay_check.to_csv(output_dir / "p15c_replay_equivalence_check.csv", index=False)
    if not replay_check["pass"].all():
        raise RuntimeError("P29 P15c replay equivalence check failed")
    all_results = {"P15c": p15c_result, **results}
    summary = p27.build_trading_summary(all_results)
    cost = p3d.build_cost_sensitivity(all_results, [float(value) for value in args.cost_bps])
    regime = p27.build_regime_comparison(all_results, [0.0, 8.0])
    folds_out = p28.build_fold_comparison({"P15c": p15c_result, "P28e_nested": results["P29_nested"]})
    folds_out = folds_out.rename(columns={column: column.replace("P28e", "P29") for column in folds_out.columns})
    summary.to_csv(output_dir / "matched_trading_summary.csv", index=False)
    cost.to_csv(output_dir / "cost_sensitivity.csv", index=False)
    regime.to_csv(output_dir / "quarterly_regime.csv", index=False)
    folds_out.to_csv(output_dir / "fold_comparison.csv", index=False)
    fixed_audit, fixed_quarterly = build_fixed_candidate_audit(
        all_results, tail_selection, variant_validation, summary, cost, regime
    )
    fixed_audit.to_csv(output_dir / "fixed_candidate_oos_audit.csv", index=False)
    fixed_quarterly.to_csv(output_dir / "fixed_candidate_quarterly_delta.csv", index=False)
    audit = build_audit(folds, tail_selection, selected_variant, results["P29_nested"])
    audit.to_csv(output_dir / "nested_protocol_audit.csv", index=False)
    if not audit["all_pass"].all():
        raise RuntimeError("P29 nested chronology audit failed")
    acceptance = build_acceptance(summary, cost, selected_variant)
    acceptance.to_csv(output_dir / "acceptance_checks.csv", index=False)

    print("[4/5] Writing plots", flush=True)
    plot_tail_metrics(output_dir, member_metrics, tail_summary)
    plot_pnl(output_dir, all_results)
    plot_selection(output_dir, selected_variant)

    print("[5/5] Writing report and manifest", flush=True)
    write_report(
        output_dir, member_metrics, tail_summary, variant_validation, selected_variant,
        summary, cost, regime, folds_out, fixed_audit, fixed_quarterly, audit, acceptance,
    )
    manifest = {
        "experiment": "P29 tail-aware S1 selection",
        "family": "S1_downside_snapback only",
        "selection_target": "P4 eligible-pool tail spread, side uplift, and marginal net contribution",
        "outer_folds": int(len(folds)),
        "workers": workers,
        "train_label_closure_hours": MAX_LABEL_HORIZON + 1,
        "trade_horizon_hours": HORIZON,
        "variants": [item.__dict__ for item in variants],
        "test_data_used_for_selection": False,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(to_jsonable(manifest), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Done: {output_dir}", flush=True)


def build_variants() -> list[TailVariant]:
    return [
        TailVariant("P29_sym025", 0.025, 0.025),
        TailVariant("P29_sym050", 0.050, 0.050),
        TailVariant("P29_long050", 0.050, 0.000),
        TailVariant("P29_short050", 0.000, 0.050),
        TailVariant("P29_extreme075", 0.075, 0.075, 0.50),
    ]


def init_tail_worker(panel_path: str, config_path: str, members: dict[str, p27.NativeFormula]) -> None:
    global _PANEL, _CFG, _S1_MEMBERS
    _PANEL = pd.read_pickle(panel_path)
    _CFG = load_config(Path(config_path))
    _S1_MEMBERS = members


def tail_diagnostic_worker(fold: dict[str, Any]) -> dict[str, pd.DataFrame]:
    if _PANEL is None or _CFG is None:
        raise RuntimeError("P29 tail worker is not initialized")
    fold_id = int(fold["fold"])
    validation_start = validation_split(fold)
    train_label_end = validation_start - pd.Timedelta(hours=MAX_LABEL_HORIZON + 1)
    validation_label_end = p4.decision_end_for_closed_positions(pd.Timestamp(fold["train_end"]), MAX_LABEL_HORIZON)
    test_label_end = p4.decision_end_for_closed_positions(pd.Timestamp(fold["test_end"]), MAX_LABEL_HORIZON)
    history_start = pd.Timestamp(fold["train_start"]) - pd.Timedelta(hours=504)
    local = _PANEL.loc[
        (_PANEL["timestamp"] >= history_start) & (_PANEL["timestamp"] < pd.Timestamp(fold["test_end"]))
    ].copy()
    local[p3d.P4_SCORE_COL] = local[p28.fold_score_column("p4", fold_id)].astype(float)
    local["_anchor"] = local[p28.fold_score_column("p15c", fold_id)].astype(float)
    pf_cfg = p3d.make_portfolio_config(_CFG, HORIZON)
    masks = {
        "train": (local["timestamp"] >= pd.Timestamp(fold["train_start"])) & (local["timestamp"] < train_label_end),
        "validation": (local["timestamp"] >= validation_start) & (local["timestamp"] < validation_label_end),
        "test": (local["timestamp"] >= pd.Timestamp(fold["test_start"])) & (local["timestamp"] < test_label_end),
    }
    member_rows = []
    raw_scores: dict[str, pd.Series] = {}
    survivors = []
    for member in _S1_MEMBERS.values():
        score = p27.formula_score(member, local, S1_SMOOTHING_ALPHA)
        raw_scores[member.name] = score
        events = build_tail_events(local.loc[masks["train"]], score.loc[masks["train"]], pf_cfg)
        metrics = summarize_tail_events(events, pd.Timestamp(fold["train_start"]), TRAIN_BLOCK_DAYS, "train")
        passed = bool(
            metrics["train_blocks"] >= 3
            and metrics["train_mean_delta_tail_spread"] > 0.0
            and metrics["train_q25_delta_tail_spread"] >= -0.05
            and metrics["train_positive_tail_block_rate"] >= 0.55
            and metrics["train_mean_delta_net8"] > 0.0
            and metrics["train_positive_net8_block_rate"] >= 0.50
            and metrics["train_mean_long_uplift"] >= -0.02
            and metrics["train_mean_short_uplift"] >= -0.02
        )
        if passed:
            survivors.append(member.name)
        member_rows.append({
            "fold": fold_id,
            **member.to_row(),
            **metrics,
            "tail_stability_pass": passed,
            "train_label_end": train_label_end,
        })
    selection = pd.DataFrame([{
        "fold": fold_id,
        "selected_members": "|".join(survivors),
        "selected_member_count": len(survivors),
        "committee_active": bool(survivors),
        "train_label_end": train_label_end,
        "validation_start": validation_start,
        "validation_label_end": validation_label_end,
        "test_start": fold["test_start"],
        "test_label_end": test_label_end,
    }])
    if survivors:
        score = p28.family_committee_score(local, [p27.evaluate_formula(_S1_MEMBERS[name], local) for name in survivors])
        validation_events = build_tail_events(local.loc[masks["validation"]], score.loc[masks["validation"]], pf_cfg)
        test_events = build_tail_events(local.loc[masks["test"]], score.loc[masks["test"]], pf_cfg)
        validation = summarize_tail_events(validation_events, validation_start, EVAL_BLOCK_DAYS, "validation")
        test = summarize_tail_events(test_events, pd.Timestamp(fold["test_start"]), EVAL_BLOCK_DAYS, "test")
    else:
        validation = empty_tail_summary("validation")
        test = empty_tail_summary("test")
    return {
        "members": pd.DataFrame(member_rows),
        "selection": selection,
        "validation": pd.DataFrame([{"fold": fold_id, **validation}]),
        "test": pd.DataFrame([{"fold": fold_id, **test}]),
    }


def build_tail_events(frame: pd.DataFrame, candidate_score: pd.Series, cfg: Any) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    work = frame.copy()
    work["_candidate"] = candidate_score.reindex(work.index)
    timestamps = np.array(sorted(work["timestamp"].unique()))[::HORIZON]
    symbols = sorted(work["symbol"].unique())
    previous_base = pd.Series(0.0, index=symbols)
    previous_candidate = pd.Series(0.0, index=symbols)
    rows_out = []
    label_col = "label_slow_low_turnover_rank_centered"
    return_col = f"fwd_simple_ret_h{HORIZON}"
    for timestamp in timestamps:
        rows = work.loc[work["timestamp"].eq(timestamp)].copy()
        rows = rows[["symbol", p3d.P4_SCORE_COL, "_anchor", "_candidate", label_col, return_col]].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        if len(rows) < cfg.long_k + cfg.short_k:
            continue
        rows["p4_rank"] = rows[p3d.P4_SCORE_COL].rank(pct=True, method="average")
        long_pool = rows.loc[rows["p4_rank"] >= cfg.long_entry_rank]
        short_pool = rows.loc[rows["p4_rank"] <= cfg.short_entry_rank]
        if len(long_pool) < cfg.long_k or len(short_pool) < cfg.short_k:
            continue
        base_long = long_pool.nlargest(cfg.long_k, "_anchor")
        base_short = short_pool.nsmallest(cfg.short_k, "_anchor")
        cand_long = long_pool.nlargest(cfg.long_k, "_candidate")
        cand_short = short_pool.nsmallest(cfg.short_k, "_candidate")
        base_target = equal_tail_weights(symbols, base_long["symbol"], base_short["symbol"])
        cand_target = equal_tail_weights(symbols, cand_long["symbol"], cand_short["symbol"])
        base_turnover = float((base_target - previous_base).abs().sum())
        cand_turnover = float((cand_target - previous_candidate).abs().sum())
        base_spread = float(base_long[label_col].mean() - base_short[label_col].mean())
        cand_spread = float(cand_long[label_col].mean() - cand_short[label_col].mean())
        long_uplift = float(cand_long[label_col].mean() - base_long[label_col].mean())
        short_uplift = float(base_short[label_col].mean() - cand_short[label_col].mean())
        base_gross = float(0.5 * base_long[return_col].mean() - 0.5 * base_short[return_col].mean())
        cand_gross = float(0.5 * cand_long[return_col].mean() - 0.5 * cand_short[return_col].mean())
        delta_turnover = cand_turnover - base_turnover
        delta_gross = cand_gross - base_gross
        rows_out.append({
            "timestamp": timestamp,
            "baseline_tail_spread": base_spread,
            "candidate_tail_spread": cand_spread,
            "delta_tail_spread": cand_spread - base_spread,
            "long_uplift": long_uplift,
            "short_uplift": short_uplift,
            "baseline_gross_return_proxy": base_gross,
            "candidate_gross_return_proxy": cand_gross,
            "delta_gross_return_proxy": delta_gross,
            "baseline_target_turnover": base_turnover,
            "candidate_target_turnover": cand_turnover,
            "delta_target_turnover": delta_turnover,
            "delta_net8_proxy": delta_gross - delta_turnover * 8.0 / 10_000.0,
            "delta_net20_proxy": delta_gross - delta_turnover * 20.0 / 10_000.0,
            "long_overlap": len(set(base_long["symbol"]) & set(cand_long["symbol"])) / cfg.long_k,
            "short_overlap": len(set(base_short["symbol"]) & set(cand_short["symbol"])) / cfg.short_k,
        })
        previous_base = base_target
        previous_candidate = cand_target
    return pd.DataFrame(rows_out)


def equal_tail_weights(symbols: list[str], longs: pd.Series, shorts: pd.Series) -> pd.Series:
    weights = pd.Series(0.0, index=symbols)
    if len(longs):
        weights.loc[list(longs)] = 0.5 / len(longs)
    if len(shorts):
        weights.loc[list(shorts)] = -0.5 / len(shorts)
    return weights


def summarize_tail_events(events: pd.DataFrame, start: pd.Timestamp, block_days: int, prefix: str) -> dict[str, Any]:
    if events.empty:
        return empty_tail_summary(prefix)
    elapsed = (pd.to_datetime(events["timestamp"], utc=True) - start).dt.total_seconds() / 86400.0
    events = events.copy()
    events["block"] = np.floor(elapsed / block_days).astype(int)
    block = events.groupby("block").agg(
        delta_tail_spread=("delta_tail_spread", "mean"),
        delta_net8=("delta_net8_proxy", "mean"),
        delta_net20=("delta_net20_proxy", "mean"),
    )
    return {
        f"{prefix}_periods": len(events),
        f"{prefix}_blocks": len(block),
        f"{prefix}_mean_baseline_tail_spread": float(events["baseline_tail_spread"].mean()),
        f"{prefix}_mean_candidate_tail_spread": float(events["candidate_tail_spread"].mean()),
        f"{prefix}_mean_delta_tail_spread": float(events["delta_tail_spread"].mean()),
        f"{prefix}_q25_delta_tail_spread": float(block["delta_tail_spread"].quantile(0.25)),
        f"{prefix}_positive_tail_block_rate": float((block["delta_tail_spread"] > 0.0).mean()),
        f"{prefix}_mean_long_uplift": float(events["long_uplift"].mean()),
        f"{prefix}_mean_short_uplift": float(events["short_uplift"].mean()),
        f"{prefix}_mean_delta_gross": float(events["delta_gross_return_proxy"].mean()),
        f"{prefix}_mean_delta_turnover": float(events["delta_target_turnover"].mean()),
        f"{prefix}_mean_delta_net8": float(events["delta_net8_proxy"].mean()),
        f"{prefix}_mean_delta_net20": float(events["delta_net20_proxy"].mean()),
        f"{prefix}_positive_net8_block_rate": float((block["delta_net8"] > 0.0).mean()),
        f"{prefix}_positive_net20_block_rate": float((block["delta_net20"] > 0.0).mean()),
        f"{prefix}_mean_long_overlap": float(events["long_overlap"].mean()),
        f"{prefix}_mean_short_overlap": float(events["short_overlap"].mean()),
    }


def empty_tail_summary(prefix: str) -> dict[str, Any]:
    names = [
        "periods", "blocks", "mean_baseline_tail_spread", "mean_candidate_tail_spread",
        "mean_delta_tail_spread", "q25_delta_tail_spread", "positive_tail_block_rate",
        "mean_long_uplift", "mean_short_uplift", "mean_delta_gross", "mean_delta_turnover",
        "mean_delta_net8", "mean_delta_net20", "positive_net8_block_rate",
        "positive_net20_block_rate", "mean_long_overlap", "mean_short_overlap",
    ]
    return {f"{prefix}_{name}": (0 if name in {"periods", "blocks"} else np.nan) for name in names}


def summarize_tail_oos(test: pd.DataFrame, selection: pd.DataFrame) -> pd.DataFrame:
    merged = test.merge(selection[["fold", "committee_active", "selected_member_count"]], on="fold", how="left")
    active = merged.loc[merged["committee_active"].astype(bool)]
    return pd.DataFrame([{
        "folds": int(merged["fold"].nunique()),
        "active_folds": int(len(active)),
        "active_fold_rate": float(len(active) / merged["fold"].nunique()),
        "mean_selected_members": float(active["selected_member_count"].mean()) if not active.empty else 0.0,
        "mean_test_delta_tail_spread": float(active["test_mean_delta_tail_spread"].mean()),
        "q25_fold_delta_tail_spread": float(active["test_mean_delta_tail_spread"].quantile(0.25)),
        "positive_tail_fold_rate": float((active["test_mean_delta_tail_spread"] > 0.0).mean()),
        "mean_test_long_uplift": float(active["test_mean_long_uplift"].mean()),
        "mean_test_short_uplift": float(active["test_mean_short_uplift"].mean()),
        "mean_test_delta_gross_proxy": float(active["test_mean_delta_gross"].mean()),
        "mean_test_delta_net8_proxy": float(active["test_mean_delta_net8"].mean()),
        "mean_test_delta_net20_proxy": float(active["test_mean_delta_net20"].mean()),
        "positive_net8_fold_rate": float((active["test_mean_delta_net8"] > 0.0).mean()),
        "mean_test_delta_turnover": float(active["test_mean_delta_turnover"].mean()),
        "mean_test_long_overlap": float(active["test_mean_long_overlap"].mean()),
        "mean_test_short_overlap": float(active["test_mean_short_overlap"].mean()),
    }])


def init_tail_trade_worker(
    panel_path: str,
    config_path: str,
    members: dict[str, p27.NativeFormula],
    tail_selection: pd.DataFrame,
    tail_validation: pd.DataFrame,
    selected_p9: pd.DataFrame,
    selected_p15c: pd.DataFrame,
    residual_path: str,
) -> None:
    global _PANEL, _CFG, _S1_MEMBERS, _TAIL_SELECTION, _TAIL_VALIDATION
    global _P9_SELECTED, _P15C_SELECTED, _RISK_RESIDUAL
    _PANEL = pd.read_pickle(panel_path)
    _CFG = load_config(Path(config_path))
    _S1_MEMBERS = members
    _TAIL_SELECTION = tail_selection.set_index("fold")
    _TAIL_VALIDATION = tail_validation.set_index("fold")
    _P9_SELECTED = selected_p9.set_index("fold")
    _P15C_SELECTED = selected_p15c.set_index("fold")
    _RISK_RESIDUAL = pd.read_pickle(residual_path)


def tail_trade_worker(fold: dict[str, Any]) -> dict[str, Any]:
    if any(value is None for value in [_PANEL, _CFG, _TAIL_SELECTION, _TAIL_VALIDATION, _P9_SELECTED, _P15C_SELECTED, _RISK_RESIDUAL]):
        raise RuntimeError("P29 trade worker is not initialized")
    fold_id = int(fold["fold"])
    history_start = pd.Timestamp(fold["train_start"]) - pd.Timedelta(hours=504)
    local = _PANEL.loc[
        (_PANEL["timestamp"] >= history_start) & (_PANEL["timestamp"] < pd.Timestamp(fold["test_end"]))
    ].copy()
    local[p3d.P4_SCORE_COL] = local[p28.fold_score_column("p4", fold_id)].astype(float)
    local["_anchor"] = local[p28.fold_score_column("p15c", fold_id)].astype(float)
    selection = _TAIL_SELECTION.loc[fold_id]
    names = [name for name in str(selection["selected_members"]).split("|") if name and name != "nan"]
    if names:
        candidate = p28.family_committee_score(
            local, [p27.evaluate_formula(_S1_MEMBERS[name], local) for name in names]
        )
    else:
        candidate = local["_anchor"].copy()
    score_by_variant = {
        variant.name: side_aware_score(local, candidate, variant) if names else local["_anchor"].copy()
        for variant in build_variants()
    }

    base_row = _P9_SELECTED.loc[fold_id]
    committee = {item.name: item for item in p3d.build_committee_specs()}[str(base_row["p4_spec"])]
    pf_cfg = p3d.make_portfolio_config(_CFG, HORIZON)
    risk_cfg = p15c.make_fold_risk(p3d.make_risk_template(_CFG), committee, f"p29_fold_{fold_id}")
    cluster_row = _P15C_SELECTED.loc[fold_id]
    cluster_map = {item.name: item for item in p15c.build_extreme_specs()}
    cluster_spec = None if bool(cluster_row["fallback_to_p9_sizing"]) else cluster_map[str(cluster_row["selected_extreme_spec"])]
    validation_start = validation_split(fold)
    validation_trade_end = p4.decision_end_for_closed_positions(pd.Timestamp(fold["train_end"]), HORIZON)
    test_trade_end = p4.decision_end_for_closed_positions(pd.Timestamp(fold["test_end"]), HORIZON)
    validation_tail = _TAIL_VALIDATION.loc[fold_id]

    all_scores = {"P15c": local["_anchor"], **score_by_variant}
    validation_runs: dict[str, dict[str, Any]] = {}
    rows = []
    for name, score in all_scores.items():
        run = p28.run_window(
            local, validation_start, validation_trade_end, fold, name, pf_cfg, risk_cfg, committee,
            cluster_spec, cluster_row, _RISK_RESIDUAL, score,
        )
        validation_runs[name] = run
        metrics = run["folds"].iloc[0].to_dict()
        rows.append({
            "fold": fold_id,
            "variant": name,
            "committee_active": bool(names),
            **{key: value for key, value in validation_tail.to_dict().items() if str(key).startswith("validation_")},
            **{f"validation_{key}": value for key, value in metrics.items() if p28.scalar(value)},
        })
    validation_frame = annotate_acceptance(pd.DataFrame(rows))
    selected_row = select_variant(validation_frame, fold)
    selected_name = str(selected_row["selected_variant"])

    test_runs = {}
    for name, score in all_scores.items():
        test_runs[name] = p28.run_window(
            local, pd.Timestamp(fold["test_start"]), test_trade_end, fold, name, pf_cfg, risk_cfg, committee,
            cluster_spec, cluster_row, _RISK_RESIDUAL, score,
        )
    nested = p28.clone_run_with_name(test_runs[selected_name], "P29_nested", selected_name)
    return {
        "validation": validation_frame,
        "selected": pd.DataFrame([selected_row]),
        "tests": {variant.name: test_runs[variant.name] for variant in build_variants()},
        "nested": nested,
        "baseline_replay": test_runs["P15c"],
    }


def side_aware_score(local: pd.DataFrame, candidate: pd.Series, variant: TailVariant) -> pd.Series:
    p4_rank = local[p3d.P4_SCORE_COL].groupby(local["timestamp"], sort=False).rank(pct=True, method="average")
    weights = np.zeros(len(local), dtype=float)
    pf_cfg = p3d.make_portfolio_config(_CFG, HORIZON)
    long_mask = p4_rank.ge(pf_cfg.long_entry_rank).to_numpy()
    short_mask = p4_rank.le(pf_cfg.short_entry_rank).to_numpy()
    weights[long_mask] = variant.long_weight
    weights[short_mask] = variant.short_weight
    if variant.extreme_threshold is not None:
        weights[np.abs(candidate.to_numpy(dtype=float)) < variant.extreme_threshold] = 0.0
    combined = (1.0 - weights) * local["_anchor"].to_numpy(dtype=float) + weights * candidate.to_numpy(dtype=float)
    return p27.centered_rank(local, pd.Series(combined, index=local.index))


def annotate_acceptance(frame: pd.DataFrame) -> pd.DataFrame:
    base = frame.loc[frame["variant"].eq("P15c")].iloc[0]
    out = frame.copy()
    out["delta_net8"] = out["validation_total_return_8bps"] - float(base["validation_total_return_8bps"])
    out["delta_net20"] = out["validation_total_return_20bps"] - float(base["validation_total_return_20bps"])
    out["delta_gross"] = out["validation_gross_total_return_before_cost"] - float(base["validation_gross_total_return_before_cost"])
    out["delta_sharpe8"] = out["validation_sharpe_8bps"] - float(base["validation_sharpe_8bps"])
    out["delta_maxdd8"] = out["validation_max_drawdown_8bps"] - float(base["validation_max_drawdown_8bps"])
    tail_gate = (
        out["committee_active"].astype(bool)
        & out["validation_mean_delta_tail_spread"].fillna(-1.0).gt(0.0)
        & out["validation_q25_delta_tail_spread"].fillna(-1.0).ge(-0.05)
        & out["validation_mean_delta_net8"].fillna(-1.0).gt(0.0)
        & out["validation_positive_tail_block_rate"].fillna(0.0).ge(0.50)
    )
    turnover_limit = max(float(base["validation_mean_turnover"]) * 1.03, float(base["validation_mean_turnover"]) + 0.005)
    out["tail_gate_pass"] = tail_gate
    out["hard_accept"] = (
        ~out["variant"].eq("P15c")
        & tail_gate
        & out["delta_net8"].gt(0.0)
        & out["delta_net20"].gt(0.0)
        & out["delta_gross"].ge(-0.0005)
        & out["delta_sharpe8"].ge(-0.03)
        & out["validation_max_drawdown_8bps"].ge(1.08 * float(base["validation_max_drawdown_8bps"]))
        & out["validation_mean_turnover"].le(turnover_limit)
    )
    out["selection_score"] = (
        140.0 * out["delta_net8"]
        + 100.0 * out["delta_net20"]
        + 40.0 * out["delta_gross"]
        + 0.8 * out["delta_sharpe8"]
        + 25.0 * out["delta_maxdd8"]
        + 8.0 * out["validation_mean_delta_tail_spread"].fillna(0.0)
        + 20.0 * out["validation_mean_delta_net8"].fillna(0.0)
        - 4.0 * (out["validation_mean_turnover"] - float(base["validation_mean_turnover"])).clip(lower=0.0)
    )
    return out


def select_variant(validation: pd.DataFrame, fold: dict[str, Any]) -> dict[str, Any]:
    accepted = validation.loc[validation["hard_accept"].astype(bool)]
    if accepted.empty:
        winner = validation.loc[validation["variant"].eq("P15c")].iloc[0]
        fallback = True
    else:
        winner = accepted.sort_values(["selection_score", "delta_net20", "delta_net8"], ascending=False).iloc[0]
        fallback = False
    return {
        "fold": int(fold["fold"]),
        "train_end": fold["train_end"],
        "test_start": fold["test_start"],
        "test_end": fold["test_end"],
        "selected_variant": str(winner["variant"]),
        "fallback_to_p15c": fallback,
        "accepted_variant_count": int(len(accepted)),
        "validation_selection_score": float(winner["selection_score"]),
        "validation_delta_tail_spread": float(winner.get("validation_mean_delta_tail_spread", np.nan)),
        "validation_delta_net8_proxy": float(winner.get("validation_mean_delta_net8", np.nan)),
        "validation_net8": float(winner["validation_total_return_8bps"]),
        "validation_net20": float(winner["validation_total_return_20bps"]),
        "validation_gross": float(winner["validation_gross_total_return_before_cost"]),
    }


def build_audit(
    folds: pd.DataFrame,
    selection: pd.DataFrame,
    selected_variant: pd.DataFrame,
    result: dict[str, Any],
) -> pd.DataFrame:
    timing = selection[["fold", "train_label_end", "validation_start", "validation_label_end"]].copy()
    for column in ["train_label_end", "validation_start", "validation_label_end"]:
        timing[column] = pd.to_datetime(timing[column], utc=True)
    merged = folds.merge(timing, on="fold").merge(selected_variant[["fold", "selected_variant"]], on="fold")
    test_max = result["equity"].groupby("fold")["timestamp"].max().rename("max_test_decision_time")
    merged = merged.merge(test_max, on="fold", how="left")
    merged["train_labels_closed_before_validation"] = (
        merged["train_label_end"] + pd.Timedelta(hours=MAX_LABEL_HORIZON + 1)
    ) <= merged["validation_start"]
    merged["validation_labels_closed_before_test"] = (
        merged["validation_label_end"] + pd.Timedelta(hours=MAX_LABEL_HORIZON + 1)
    ) <= merged["train_end"]
    merged["selection_before_test"] = merged["train_end"] <= merged["test_start"]
    merged["test_trades_close_inside_fold"] = (
        merged["max_test_decision_time"] + pd.Timedelta(hours=HORIZON + 1)
    ) <= merged["test_end"]
    merged["all_pass"] = merged[[
        "train_labels_closed_before_validation", "validation_labels_closed_before_test",
        "selection_before_test", "test_trades_close_inside_fold",
    ]].all(axis=1)
    return merged


def build_acceptance(summary: pd.DataFrame, cost: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    base = summary.loc[summary["run"].eq("P15c")].iloc[0]
    cand = summary.loc[summary["run"].eq("P29_nested")].iloc[0]
    base8 = cost.loc[cost["experiment"].eq("P15c") & cost["one_way_cost_bps"].eq(8.0)].iloc[0]
    cand8 = cost.loc[cost["experiment"].eq("P29_nested") & cost["one_way_cost_bps"].eq(8.0)].iloc[0]
    checks = [
        ("gross_above_p15c", cand["gross_total_return_before_cost"] - base["gross_total_return_before_cost"], cand["gross_total_return_before_cost"] > base["gross_total_return_before_cost"] + 1e-5),
        ("net8_above_p15c", cand8["total_return"] - base8["total_return"], cand8["total_return"] > base8["total_return"] + 1e-5),
        ("sharpe_above_p15c", cand["sharpe"] - base["sharpe"], cand["sharpe"] > base["sharpe"] + 1e-4),
        ("maxdd_within_110pct", cand["max_drawdown"] - base["max_drawdown"], cand["max_drawdown"] >= 1.10 * base["max_drawdown"]),
        ("turnover_not_above_103pct", cand["mean_turnover"] / base["mean_turnover"], cand["mean_turnover"] <= 1.03 * base["mean_turnover"]),
        ("positive_fold_rate_not_lower", cand["positive_fold_rate"] - base["positive_fold_rate"], cand["positive_fold_rate"] >= base["positive_fold_rate"]),
        ("not_all_fallback", float(selected["fallback_to_p15c"].mean()), float(selected["fallback_to_p15c"].mean()) < 1.0),
    ]
    return pd.DataFrame([{"check": name, "value": value, "pass": bool(passed)} for name, value, passed in checks])


def build_fixed_candidate_audit(
    results: dict[str, dict[str, Any]],
    tail_selection: pd.DataFrame,
    validation: pd.DataFrame,
    summary: pd.DataFrame,
    cost: pd.DataFrame,
    regime: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit fixed variants without treating their outer-test comparison as model selection."""
    def fold_returns(result: dict[str, Any]) -> pd.DataFrame:
        equity = result["equity"].sort_values(["fold", "timestamp"])
        rows = []
        for fold, frame in equity.groupby("fold", sort=True):
            gross_returns = frame["gross_period_return"].astype(float)
            net8_returns = gross_returns - frame["turnover"].astype(float) * 8.0 / 10_000.0
            rows.append({
                "fold": int(fold),
                "gross": float(np.prod(1.0 + gross_returns) - 1.0),
                "net8": float(np.prod(1.0 + net8_returns) - 1.0),
            })
        return pd.DataFrame(rows).set_index("fold")

    base_folds = fold_returns(results["P15c"])
    active_folds = set(
        tail_selection.loc[tail_selection["selected_member_count"].gt(0), "fold"].astype(int)
    )
    base_summary = summary.loc[summary["run"].eq("P15c")].iloc[0]
    base8 = cost.loc[cost["experiment"].eq("P15c") & cost["one_way_cost_bps"].eq(8.0)].iloc[0]
    rows: list[dict[str, Any]] = []
    material_delta = 1e-4  # One basis point of fold-level return.
    for name, result in results.items():
        if name in {"P15c", "P29_nested"}:
            continue
        candidate_folds = fold_returns(result)
        delta8 = candidate_folds["net8"] - base_folds["net8"]
        delta_gross = candidate_folds["gross"] - base_folds["gross"]
        active_index = sorted(active_folds.intersection(delta8.index.astype(int)))
        active_delta = delta8.reindex(active_index).dropna()
        active_gross = delta_gross.reindex(active_index).dropna()
        positive = active_delta.clip(lower=0.0)
        top3_share = positive.nlargest(3).sum() / positive.sum() if positive.sum() > 0 else np.nan
        candidate_summary = summary.loc[summary["run"].eq(name)].iloc[0]
        candidate8 = cost.loc[
            cost["experiment"].eq(name) & cost["one_way_cost_bps"].eq(8.0)
        ].iloc[0]
        validation_delta = validation.loc[validation["variant"].eq(name), ["fold", "delta_net8"]]
        association = pd.DataFrame({"fold": active_delta.index, "test_delta_net8": active_delta.values}).merge(
            validation_delta, on="fold", how="left"
        ).dropna()
        rows.append({
            "variant": name,
            "active_folds": len(active_delta),
            "material_win_folds": int(active_delta.gt(material_delta).sum()),
            "material_loss_folds": int(active_delta.lt(-material_delta).sum()),
            "material_flat_folds": int(active_delta.abs().le(material_delta).sum()),
            "mean_active_fold_delta_net8": float(active_delta.mean()),
            "median_active_fold_delta_net8": float(active_delta.median()),
            "q25_active_fold_delta_net8": float(active_delta.quantile(0.25)),
            "worst_active_fold_delta_net8": float(active_delta.min()),
            "best_active_fold_delta_net8": float(active_delta.max()),
            "top3_positive_delta_share": float(top3_share),
            "mean_active_fold_delta_gross": float(active_gross.mean()),
            "full_sample_delta_gross": float(
                candidate_summary["gross_total_return_before_cost"]
                - base_summary["gross_total_return_before_cost"]
            ),
            "full_sample_delta_net8": float(candidate8["total_return"] - base8["total_return"]),
            "validation_test_delta_corr": float(
                association[["delta_net8", "test_delta_net8"]].corr().iloc[0, 1]
            ) if len(association) >= 3 else np.nan,
        })
    audit = pd.DataFrame(rows).sort_values("full_sample_delta_net8", ascending=False)

    quarterly = regime.loc[regime["one_way_cost_bps"].eq(8.0)].pivot(
        index="regime", columns="run", values="total_return"
    )
    quarterly_rows: list[dict[str, Any]] = []
    for variant in audit["variant"]:
        for period in quarterly.index:
            quarterly_rows.append({
                "variant": variant,
                "regime": period,
                "p15c_net8": float(quarterly.loc[period, "P15c"]),
                "candidate_net8": float(quarterly.loc[period, variant]),
                "delta_net8": float(quarterly.loc[period, variant] - quarterly.loc[period, "P15c"]),
            })
    return audit, pd.DataFrame(quarterly_rows)


def plot_tail_metrics(output_dir: Path, members: pd.DataFrame, summary: pd.DataFrame) -> None:
    member_view = members.groupby("formula_name").agg(
        train_tail_delta=("train_mean_delta_tail_spread", "mean"),
        pass_rate=("tail_stability_pass", "mean"),
    ).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].bar(member_view["formula_name"], member_view["train_tail_delta"], color="#8d99ae")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_title("S1 train tail-spread delta")
    axes[0].set_ylabel("Delta slow-label tail spread")
    axes[0].grid(axis="y", alpha=0.25)
    metrics = ["mean_test_delta_tail_spread", "mean_test_long_uplift", "mean_test_short_uplift"]
    values = [float(summary.iloc[0][column]) for column in metrics]
    axes[1].bar(["tail spread", "long uplift", "short uplift"], values, color=["#457b9d", "#2a9d8f", "#d1495b"])
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_title("S1 committee nested OOS tail diagnostics")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "tail_factor_diagnostics.png", dpi=180)
    plt.close(fig)


def plot_pnl(output_dir: Path, results: dict[str, dict[str, Any]]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0]})
    colors = {
        "P15c": "#777777", "P29_sym025": "#5b8e7d", "P29_sym050": "#6c8ebf",
        "P29_long050": "#c49a5a", "P29_short050": "#9a6fb0",
        "P29_extreme075": "#3d7a80", "P29_nested": "#d1495b",
    }
    for name, result in results.items():
        equity = result["equity"].sort_values("timestamp")
        returns = equity["gross_period_return"] - equity["turnover"] * 8.0 / 10_000.0
        curve = (1.0 + returns).cumprod() - 1.0
        axes[0].plot(equity["timestamp"], curve, label=f"{name} net 8bps", color=colors[name], linewidth=2.2 if name == "P29_nested" else 1.1)
        if name == "P29_nested":
            wealth = 1.0 + curve
            drawdown = wealth / wealth.cummax() - 1.0
            axes[1].fill_between(equity["timestamp"], drawdown, 0.0, color=colors[name], alpha=0.2)
            axes[1].plot(equity["timestamp"], drawdown, color=colors[name], linewidth=1.2)
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_title("P29 tail-aware S1 matched nested OOS")
    axes[0].set_ylabel("Cumulative return")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].legend(ncol=2)
    axes[0].grid(alpha=0.25)
    axes[1].set_title("P29 nested net 8bps drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "p29_matched_nested_oos_pnl.png", dpi=180)
    plt.close(fig)


def plot_selection(output_dir: Path, selected: pd.DataFrame) -> None:
    counts = selected["selected_variant"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(counts.index, counts.values, color="#457b9d")
    ax.set_title("P29 validation-selected tail variant")
    ax.set_xlabel("Outer folds")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "selected_tail_variant.png", dpi=180)
    plt.close(fig)


def write_report(
    output_dir: Path,
    member_metrics: pd.DataFrame,
    tail_summary: pd.DataFrame,
    validation: pd.DataFrame,
    selected: pd.DataFrame,
    summary: pd.DataFrame,
    cost: pd.DataFrame,
    regime: pd.DataFrame,
    folds_out: pd.DataFrame,
    fixed_audit: pd.DataFrame,
    fixed_quarterly: pd.DataFrame,
    audit: pd.DataFrame,
    acceptance: pd.DataFrame,
) -> None:
    base = summary.loc[summary["run"].eq("P15c")].iloc[0]
    nested = summary.loc[summary["run"].eq("P29_nested")].iloc[0]
    base8 = cost.loc[cost["experiment"].eq("P15c") & cost["one_way_cost_bps"].eq(8.0)].iloc[0]
    nested8 = cost.loc[cost["experiment"].eq("P29_nested") & cost["one_way_cost_bps"].eq(8.0)].iloc[0]
    tail = tail_summary.iloc[0]
    verdict = "可升级为研究 baseline candidate" if acceptance["pass"].all() else "不覆盖 P15c"
    member_summary = member_metrics.groupby("formula_name").agg(
        tail_pass_rate=("tail_stability_pass", "mean"),
        mean_train_tail_delta=("train_mean_delta_tail_spread", "mean"),
        mean_train_delta_net8=("train_mean_delta_net8", "mean"),
    ).reset_index()
    lines = [
        "# P29 tail-aware S1 selection 报告",
        "",
        "## 结论",
        "",
        f"- **{verdict}**。",
        f"- P29 nested gross/net6/net8/Sharpe/MaxDD：{nested['gross_total_return_before_cost']:.2%} / {nested['total_return']:.2%} / {nested8['total_return']:.2%} / {nested['sharpe']:.3f} / {nested['max_drawdown']:.2%}。",
        f"- P15c 对照：{base['gross_total_return_before_cost']:.2%} / {base['total_return']:.2%} / {base8['total_return']:.2%} / {base['sharpe']:.3f} / {base['max_drawdown']:.2%}。",
        f"- P29 fallback 比例 {selected['fallback_to_p15c'].mean():.1%}；32 folds 中启用新 tail rerank {int((~selected['fallback_to_p15c']).sum())} 次。",
        "",
        "## 1. Tail-aware 因子层",
        "",
        "不再使用全市场平均 IC。每个时点先按 P4 score 定义 eligible long/short pool，再比较 S1 与 P15c 在 pool 内选出的实际尾部。",
        "",
        f"- S1 committee active folds：{int(tail['active_folds'])}/{int(tail['folds'])}。",
        f"- OOS delta tail spread：{tail['mean_test_delta_tail_spread']:.5f}；q25 fold：{tail['q25_fold_delta_tail_spread']:.5f}；正 fold 比例：{tail['positive_tail_fold_rate']:.1%}。",
        f"- long uplift：{tail['mean_test_long_uplift']:.5f}；short uplift：{tail['mean_test_short_uplift']:.5f}。",
        f"- 简化等权 tail proxy 的 OOS 8bps 边际贡献：{tail['mean_test_delta_net8_proxy']:.5%}；正 fold 比例：{tail['positive_net8_fold_rate']:.1%}。",
        "",
        "```text",
        member_summary.to_string(index=False),
        "```",
        "",
        "## 2. 最终交易层",
        "",
        "```text",
        summary[["run", "gross_total_return_before_cost", "total_return", "sharpe", "max_drawdown", "mean_turnover", "positive_fold_rate", "worst_5_fold_return_sum"]].to_string(index=False),
        "```",
        "",
        "### 成本敏感性",
        "```text",
        cost.to_string(index=False),
        "```",
        "",
        "### Validation 选择频次",
        "```text",
        selected["selected_variant"].value_counts().to_string(),
        "```",
        "",
        "### 固定候选的 outer-test 审计",
        "",
        "这些固定权重路径的每个交易点仍然是 OOS，但在看到全部 outer-test 后从网格里挑最好者属于事后模型选择，不能据此升级 baseline。material win/loss 以单 fold 1bp 为阈值。",
        "",
        "```text",
        fixed_audit.to_string(index=False),
        "```",
        "",
        "`P29_sym050` 的完整 OOS 汇总较好，但正增量集中于少数 fold，且 validation-test 增量相关性很弱。它只能作为预注册的 forward-only 候选。",
        "",
        "### 升级检查",
        "```text",
        acceptance.to_string(index=False),
        "```",
        "",
        "## 3. 严格性",
        "",
        f"- {int(audit['all_pass'].sum())}/{len(audit)} folds 通过 train/validation label closure、selection-before-test 和 test trade closure 检查。",
        "- S1 member 只用 train tail metrics 选择；side/weight 只用 validation tail metrics 和 validation portfolio PnL 选择。",
        "- test tail diagnostics 仅用于报告，不反馈给同一 fold。",
        "- 所有候选仍受 P4 eligible pool、P15c strict-neutral allocator、symbol cap、short guard 和 cluster cap 约束。",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
