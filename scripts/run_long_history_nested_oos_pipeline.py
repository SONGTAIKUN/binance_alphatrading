#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

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
from run_long_history_fixed_p16_replay import build_long_folds
from run_residual_alphagpt_search import build_candidate_scores, prepare_alpha_panel
from run_turnover_experiments import to_jsonable
import run_p4_nested_walkforward_selection as p4
import run_p3d_p4_gated_rerank_alphagpt as p3d
import run_p5_alphagpt_slow_search as p5
import run_p7_derivatives_gated_rerank_alphagpt as p7
import run_p7b_derivatives_incremental_on_p3d as p7b
import run_p9_clean_derivatives_baseline as p9
import run_p15c_extreme_cluster_turnover as p15c
from run_p13_turnover_no_trade_zone import write_result
from run_p6_feature_expansion import build_p6_features, write_feature_panel
from run_p6b_derivatives_feature_expansion import build_derivatives_candidates, write_feature_panel_extended


EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "baseline_locked_v1"
DEFAULT_CONFIG = EXPERIMENT_ROOT / "configs" / "baseline_locked_v1.yaml"
DEFAULT_ASSET_PANEL = EXPERIMENT_ROOT / "inputs" / "long_history" / "baseline_panel" / "asset_residual_panel.csv"
DEFAULT_REPLAY_PANEL = EXPERIMENT_ROOT / "inputs" / "long_history" / "p16_long_replay_panel.csv.gz"
DEFAULT_DERIVATIVES_PANEL = (
    EXPERIMENT_ROOT / "inputs" / "long_history" / "binance_derivatives" / "derivatives_hourly_panel.csv.gz"
)
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "outputs" / "long_history_nested_oos"
DEFAULT_LONG_START = "2023-10-15 08:00:00+00:00"
DEFAULT_COST_BPS = [0, 6, 10, 15, 20, 30]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a long-history nested OOS pipeline: P4 -> P3d -> P9 -> P15c."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--asset-panel", default=str(DEFAULT_ASSET_PANEL))
    parser.add_argument("--replay-panel", default=str(DEFAULT_REPLAY_PANEL))
    parser.add_argument("--derivatives-panel", default=str(DEFAULT_DERIVATIVES_PANEL))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--horizon", type=int, default=72)
    parser.add_argument("--train-days", type=int, default=180)
    parser.add_argument("--test-days", type=int, default=30)
    parser.add_argument("--start", default=DEFAULT_LONG_START)
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument("--chunksize", type=int, default=8)
    parser.add_argument("--p3d-formulas", type=int, default=180)
    parser.add_argument("--p9-formulas", type=int, default=160)
    parser.add_argument("--p3d-seed", type=int, default=2417)
    parser.add_argument("--p9-seed", type=int, default=2607)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--inner-validation-days", type=int, default=90)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    output_dir = ensure_dir(args.output_dir)
    inputs_dir = ensure_dir(output_dir / "inputs")
    stage_dirs = {
        "p4": ensure_dir(output_dir / "p4_nested"),
        "p3d": ensure_dir(output_dir / "p3d_nested"),
        "p9": ensure_dir(output_dir / "p9_nested"),
        "p15c": ensure_dir(output_dir / "p15c_nested"),
    }

    print("[0/6] Preparing long-history nested inputs", flush=True)
    prepared = prepare_inputs(
        config_path=config_path,
        asset_panel=Path(args.asset_panel).resolve(),
        replay_panel=Path(args.replay_panel).resolve(),
        derivatives_panel=Path(args.derivatives_panel).resolve(),
        inputs_dir=inputs_dir,
        start=args.start,
        train_days=int(args.train_days),
        test_days=int(args.test_days),
        horizon=int(args.horizon),
        force=bool(args.force),
    )
    if args.prepare_only:
        print(f"Prepared inputs under {inputs_dir}", flush=True)
        return

    workers = max(1, int(args.workers))
    horizon = int(args.horizon)
    cfg = load_config(config_path)
    print(f"Using workers={workers}, horizon={horizon}h", flush=True)

    p4_result, p4_selected = run_p4_stage(
        config_path=config_path,
        cfg=cfg,
        panel_path=prepared["asset_panel"],
        feature_path=prepared["manual_features"],
        selections=prepared["selections"],
        output_dir=stage_dirs["p4"],
        horizon=horizon,
        workers=workers,
        chunksize=int(args.chunksize),
        force=bool(args.force),
    )

    p3d_result, p3d_selected, p3d_formulas = run_p3d_stage(
        config_path=config_path,
        cfg=cfg,
        panel_path=prepared["asset_panel"],
        feature_path=prepared["manual_features"],
        selections=prepared["selections"],
        p4_selected=p4_selected,
        output_dir=stage_dirs["p3d"],
        horizon=horizon,
        workers=workers,
        chunksize=int(args.chunksize),
        formula_count=int(args.p3d_formulas),
        seed=int(args.p3d_seed),
        max_depth=int(args.max_depth),
        validation_days=int(args.inner_validation_days),
        force=bool(args.force),
    )

    p9_result, p9_selected, deriv_formulas = run_p9_stage(
        config_path=config_path,
        cfg=cfg,
        panel_path=prepared["asset_panel"],
        feature_path=prepared["manual_features"],
        p6b_panel=prepared["p6b_features"],
        retained_features=prepared["retained_all"],
        selections=prepared["selections"],
        p3d_selected=p3d_selected,
        output_dir=stage_dirs["p9"],
        horizon=horizon,
        workers=workers,
        chunksize=int(args.chunksize),
        formula_count=int(args.p9_formulas),
        seed=int(args.p9_seed),
        max_depth=int(args.max_depth),
        validation_days=int(args.inner_validation_days),
        force=bool(args.force),
    )

    p15c_result, p15c_selected = run_p15c_stage(
        config_path=config_path,
        cfg=cfg,
        panel_path=prepared["asset_panel"],
        feature_path=prepared["manual_features"],
        p6b_panel=prepared["p6b_features"],
        retained_features=prepared["retained_all"],
        derivatives_panel=prepared["derivatives_panel"],
        selections=prepared["selections"],
        p9_selected=p9_selected,
        p3d_formulas=p3d_formulas,
        deriv_formulas=deriv_formulas,
        output_dir=stage_dirs["p15c"],
        horizon=horizon,
        validation_days=int(args.inner_validation_days),
        force=bool(args.force),
    )

    print("[6/6] Writing long-history nested OOS package", flush=True)
    write_package(
        output_dir=output_dir,
        inputs=prepared,
        results={"p4": p4_result, "p3d": p3d_result, "p9": p9_result, "p15c": p15c_result},
        selected={"p4": p4_selected, "p3d": p3d_selected, "p9": p9_selected, "p15c": p15c_selected},
        horizon=horizon,
        cost_bps=DEFAULT_COST_BPS,
    )
    print(f"Done: {output_dir}", flush=True)


def prepare_inputs(
    config_path: Path,
    asset_panel: Path,
    replay_panel: Path,
    derivatives_panel: Path,
    inputs_dir: Path,
    start: str,
    train_days: int,
    test_days: int,
    horizon: int,
    force: bool,
) -> dict[str, Any]:
    selections_path = inputs_dir / "long_formula_selections.csv"
    manual_path = inputs_dir / "manual_slow_feature_panel_long.csv.gz"
    p6b_path = inputs_dir / "p6b_feature_panel_long.csv.gz"
    retained_path = inputs_dir / "retained_features_all_predefined.csv"
    config_long_path = inputs_dir / "long_nested_config.yaml"

    if force or not selections_path.exists():
        replay_head = pd.read_csv(replay_panel, parse_dates=["timestamp"], usecols=["timestamp", "symbol"])
        selections = build_long_folds(replay_head, start, train_days, test_days, horizon)
        selections.to_csv(selections_path, index=False)
    selections = pd.read_csv(selections_path, parse_dates=["train_start", "train_end", "test_start", "test_end"])

    manual_missing = missing_required_slow_features(manual_path) if manual_path.exists() else {"__missing__"}
    if force or manual_missing:
        build_long_manual_feature_panel(asset_panel, replay_panel, manual_path)

    if force or not p6b_path.exists() or not retained_path.exists():
        build_long_p6b_panel(asset_panel, replay_panel, derivatives_panel, p6b_path, retained_path)

    if force or not config_long_path.exists():
        cfg = load_config(config_path)
        cfg.setdefault("inputs", {})
        cfg["inputs"]["panel"] = str(asset_panel)
        cfg["inputs"]["formula_selections"] = str(selections_path)
        config_long_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False), encoding="utf-8")

    return {
        "asset_panel": asset_panel,
        "replay_panel": replay_panel,
        "derivatives_panel": derivatives_panel,
        "selections": selections,
        "selections_path": selections_path,
        "manual_features": manual_path,
        "p6b_features": p6b_path,
        "retained_all": retained_path,
        "config": config_long_path,
    }


def missing_required_slow_features(feature_path: Path) -> set[str]:
    available = set(pd.read_csv(feature_path, nrows=0).columns.tolist())
    return p5.required_feature_columns() - available


def build_long_manual_feature_panel(asset_panel: Path, replay_panel: Path, manual_path: Path) -> None:
    print("      building full P3d slow-feature long panel", flush=True)
    required = sorted(p5.required_feature_columns())
    replay_available = set(pd.read_csv(replay_panel, nrows=0).columns.tolist())
    proxy_cols = {
        "rank_a07",
        "rank_b11",
        "rank_c09",
        "score_baseline",
        "pct_beta_stability_168",
        "pct_beta_stability_336",
        "pct_ou_speed_336",
        "cs_ou_speed_168",
        "cs_res_z_336",
        "cs_liquidity_quality",
    }
    usecols = ["timestamp", "symbol", *sorted((set(required) | proxy_cols) & replay_available)]
    manual = pd.read_csv(replay_panel, parse_dates=["timestamp"], usecols=usecols)

    if "alpha_C_quality_committee" not in manual and "rank_c09" in manual:
        manual["alpha_C_quality_committee"] = manual["rank_c09"]
    if "alpha_B_seed_168" not in manual:
        manual["alpha_B_seed_168"] = first_existing(manual, ["alpha_B_seed_combo", "rank_b11"])
    if "alpha_B_seed_336" not in manual:
        manual["alpha_B_seed_336"] = first_existing(manual, ["alpha_B_seed_combo", "rank_b11"])
    if "cs_res_z_168" not in manual:
        manual["cs_res_z_168"] = first_existing(manual, ["cs_res_z_336", "score_baseline"])
    if "cs_beta_stability_168" not in manual:
        manual["cs_beta_stability_168"] = first_existing(manual, ["pct_beta_stability_168"])
    if "cs_beta_stability_336" not in manual:
        manual["cs_beta_stability_336"] = first_existing(manual, ["pct_beta_stability_336"])
    if "pct_ou_speed_168" not in manual:
        manual["pct_ou_speed_168"] = first_existing(manual, ["cs_ou_speed_168", "pct_ou_speed_336"])
    if "pct_liquidity_quality" not in manual:
        manual["pct_liquidity_quality"] = first_existing(manual, ["cs_liquidity_quality"])

    market_state = build_long_market_state_features(asset_panel)
    manual = manual.merge(market_state, on="timestamp", how="left", suffixes=("", "_computed"))
    for col in ["market_vol_14d", "residual_dispersion_7d"]:
        computed = f"{col}_computed"
        if computed in manual:
            if col in manual:
                manual[col] = manual[col].fillna(manual[computed])
                manual = manual.drop(columns=[computed])
            else:
                manual[col] = manual[computed]
                manual = manual.drop(columns=[computed])

    missing = [col for col in required if col not in manual.columns]
    if missing:
        print(f"      warning: filling unavailable slow-feature proxies with 0: {missing}", flush=True)
        for col in missing:
            manual[col] = 0.0

    manual = manual[["timestamp", "symbol", *required]].sort_values(["timestamp", "symbol"])
    manual.to_csv(manual_path, index=False)


def first_existing(frame: pd.DataFrame, candidates: list[str]) -> pd.Series:
    for col in candidates:
        if col in frame:
            return frame[col]
    return pd.Series(0.0, index=frame.index)


def build_long_market_state_features(asset_panel: Path) -> pd.DataFrame:
    usecols = ["timestamp", "symbol", "log_ret", "residual_log_ret"]
    panel = pd.read_csv(asset_panel, parse_dates=["timestamp"], usecols=usecols)
    hourly_market_ret = panel.groupby("timestamp")["log_ret"].mean().sort_index()
    market_vol_14d = hourly_market_ret.rolling(336, min_periods=72).std(ddof=0).rename("market_vol_14d")
    hourly_residual_dispersion = panel.groupby("timestamp")["residual_log_ret"].std(ddof=0).sort_index()
    residual_dispersion_7d = (
        hourly_residual_dispersion.rolling(168, min_periods=48).mean().rename("residual_dispersion_7d")
    )
    state = pd.concat([market_vol_14d, residual_dispersion_7d], axis=1).reset_index()
    return state


def build_long_p6b_panel(asset_panel: Path, replay_panel: Path, derivatives_panel: Path, p6b_path: Path, retained_path: Path) -> None:
    print("      building full predefined P6/P6b long feature panel", flush=True)
    panel = pd.read_csv(asset_panel, parse_dates=["timestamp"])
    panel = prepare_alpha_panel(panel, [48, 72, 96, 168], execution_delay_bars=1)
    build_candidate_scores(panel)

    replay_cols = pd.read_csv(replay_panel, nrows=0).columns.tolist()
    helper_cols = [
        col
        for col in replay_cols
        if col
        not in {
            "fwd_simple_ret_h72",
            "label_resid_ret_h72",
            "residual_log_ret",
            "score_baseline",
        }
    ]
    helper_cols = [col for col in helper_cols if col not in panel.columns or col in {"timestamp", "symbol"}]
    helper = pd.read_csv(replay_panel, parse_dates=["timestamp"], usecols=helper_cols)
    panel = panel.merge(helper, on=["timestamp", "symbol"], how="left")
    for raw_col, ranked_col in [
        ("res_mom_168_24", "cs_res_mom_168_24"),
        ("res_mom_336_24", "cs_res_mom_336_24"),
    ]:
        if raw_col not in panel.columns and ranked_col in panel.columns:
            panel[raw_col] = panel[ranked_col]
    derivatives = pd.read_csv(derivatives_panel, parse_dates=["timestamp"])
    panel = panel.merge(derivatives, on=["timestamp", "symbol"], how="left")

    candidates = build_p6_features(panel)
    candidates.extend(build_derivatives_candidates(panel))
    write_feature_panel_extended(panel, candidates, p6b_path)
    retained = predefined_retained_table(candidates)
    retained.to_csv(retained_path, index=False)


def predefined_retained_table(candidates: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    n = len(candidates)
    for idx, candidate in enumerate(candidates):
        rows.append(
            {
                "name": candidate["name"],
                "family": candidate["family"],
                "score_col": candidate["score_col"],
                "formula": candidate["formula"],
                "features": candidate["features"],
                "primary_mean_ic": 0.0,
                "primary_median_ic": 0.0,
                "primary_icir": 0.0,
                "primary_q25_fold_ic": 0.0,
                "primary_positive_fold_ic_rate": 0.0,
                "primary_positive_ts_ic_rate": 0.0,
                "fold_count": 0,
                "timestamp_count": 0,
                "path_medium_mean_ic": 0.0,
                "slow_low_turnover_mean_ic": 0.0,
                "signal_turnover_72h": 0.0,
                "signal_turnover_q75_72h": 0.0,
                "signal_rank_autocorr_72h": 0.0,
                "oos_rebalance_count": 0,
                "mean_p3d_weight_corr": 0.0,
                "median_p3d_weight_corr": 0.0,
                "mean_abs_p3d_weight_corr": 0.0,
                "p3d_corr_timestamp_count": 0,
                "diagnostic_score": float(n - idx),
                "retention_turnover_threshold": np.nan,
                "pass_ic": True,
                "pass_stability": True,
                "pass_turnover": True,
                "pass_low_corr": True,
                "retained": True,
            }
        )
    return pd.DataFrame(rows)


def run_p4_stage(
    config_path: Path,
    cfg: dict[str, Any],
    panel_path: Path,
    feature_path: Path,
    selections: pd.DataFrame,
    output_dir: Path,
    horizon: int,
    workers: int,
    chunksize: int,
    force: bool,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    run_dir = ensure_dir(output_dir / "nested_test_run")
    selected_path = output_dir / "selected_specs_by_fold.csv"
    if not force and selected_path.exists() and (run_dir / "equity_curve.csv").exists():
        return load_result(run_dir), pd.read_csv(selected_path, parse_dates=["train_start", "train_end", "test_start", "test_end"])

    specs = p4.build_committee_specs()
    pd.DataFrame([spec.__dict__ for spec in specs]).to_csv(output_dir / "candidate_specs.csv", index=False)
    tasks = [
        {
            "fold": int(row["fold"]),
            "window": "validation",
            "start": row["train_start"],
            "end": p4.decision_end_for_closed_positions(row["train_end"], horizon),
            "window_end": row["train_end"],
            "spec_name": spec.name,
        }
        for _, row in selections.iterrows()
        for spec in specs
    ]
    print(f"[1/6] P4 validation: {len(tasks)} tasks", flush=True)
    rows = run_pool(
        tasks,
        initializer=p4.init_worker,
        initargs=(str(config_path), str(panel_path), str(feature_path), horizon),
        worker=p4.run_validation_task,
        workers=workers,
        chunksize=chunksize,
        label="P4",
    )
    validation = pd.DataFrame(rows)
    validation["fitness"] = p4.build_fitness(validation)
    validation.to_csv(output_dir / "validation_candidate_metrics.csv", index=False)
    selected = p4.select_specs_by_fold(validation, selections, horizon)
    selected.to_csv(selected_path, index=False)

    panel = p4.load_panel(config_path, panel_path, feature_path, horizon)
    pf_cfg = p4.make_portfolio_config(cfg, horizon)
    risk_template = p4.make_risk_template(cfg)
    result = p4.replay_selected_tests(panel, selections, selected, pf_cfg, risk_template, {spec.name: spec for spec in specs})
    p4.write_run_result(run_dir, result)
    write_metrics(output_dir / "summary.json", result["metrics"])
    return result, selected


def run_p3d_stage(
    config_path: Path,
    cfg: dict[str, Any],
    panel_path: Path,
    feature_path: Path,
    selections: pd.DataFrame,
    p4_selected: pd.DataFrame,
    output_dir: Path,
    horizon: int,
    workers: int,
    chunksize: int,
    formula_count: int,
    seed: int,
    max_depth: int,
    validation_days: int,
    force: bool,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, list[Any]]:
    run_dir = ensure_dir(output_dir / "nested_incremental_run")
    selected_path = output_dir / "selected_incremental_by_fold.csv"
    formulas = p3d.build_incremental_formula_universe(formula_count, seed, max_depth)
    pd.DataFrame([item.to_row() for item in formulas]).to_csv(output_dir / "candidate_formulas.csv", index=False)
    if not force and selected_path.exists() and (run_dir / "equity_curve.csv").exists():
        return (
            load_result(run_dir),
            pd.read_csv(selected_path, parse_dates=["train_start", "train_end", "test_start", "test_end", "test_decision_end"]),
            formulas,
        )

    tasks = p3d.build_validation_tasks(selections, p4_selected, formulas, horizon, validation_days)
    print(f"[2/6] P3d AlphaGPT validation: {len(tasks)} tasks", flush=True)
    rows = run_pool(
        tasks,
        initializer=p3d.init_worker,
        initargs=(str(config_path), str(panel_path), str(feature_path), horizon, {item.name: item for item in formulas}),
        worker=p3d.run_validation_task,
        workers=workers,
        chunksize=chunksize,
        label="P3d",
    )
    validation = pd.DataFrame(rows)
    p4_validation = p3d.run_p4_validation_baselines(config_path, panel_path, feature_path, selections, p4_selected, horizon)
    p4_validation.to_csv(output_dir / "validation_p4_metrics.csv", index=False)
    validation = p3d.annotate_hard_accept(validation, p4_validation)
    validation.to_csv(output_dir / "validation_incremental_metrics.csv", index=False)
    selected = p3d.select_by_fold(validation, p4_validation, selections, p4_selected, horizon)
    selected.to_csv(selected_path, index=False)

    panel = p3d.load_panel(config_path, panel_path, feature_path, horizon)
    pf_cfg = p3d.make_portfolio_config(cfg, horizon)
    risk_template = p3d.make_risk_template(cfg)
    result = p3d.replay_selected_tests(
        panel, selections, selected, pf_cfg, risk_template, {spec.name: spec for spec in p3d.build_committee_specs()}, {item.name: item for item in formulas}
    )
    p3d.write_run_result(run_dir, result)
    write_metrics(output_dir / "summary.json", result["metrics"])
    return result, selected, formulas


def run_p9_stage(
    config_path: Path,
    cfg: dict[str, Any],
    panel_path: Path,
    feature_path: Path,
    p6b_panel: Path,
    retained_features: Path,
    selections: pd.DataFrame,
    p3d_selected: pd.DataFrame,
    output_dir: Path,
    horizon: int,
    workers: int,
    chunksize: int,
    formula_count: int,
    seed: int,
    max_depth: int,
    validation_days: int,
    force: bool,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, list[Any]]:
    run_dir = ensure_dir(output_dir / "nested_incremental_run")
    selected_path = output_dir / "selected_incremental_by_fold.csv"
    set_p7_paths(p6b_panel, retained_features)
    retained = p7.load_retained_features()
    raw_formulas = p7.build_p7_formula_universe(retained, formula_count, seed, max_depth)
    formulas = p9.filter_p9_formulas(raw_formulas, exclude_k01=True)
    p9.formula_table_with_tags(raw_formulas, formulas).to_csv(output_dir / "candidate_formulas.csv", index=False)
    p3d_formulas = p3d.build_incremental_formula_universe(180, 2417, 3)
    if not force and selected_path.exists() and (run_dir / "equity_curve.csv").exists():
        return (
            load_result(run_dir),
            pd.read_csv(selected_path, parse_dates=["train_start", "train_end", "test_start", "test_end", "test_decision_end"]),
            formulas,
        )

    p7b.P7B_GRID = p9.P9_GRID
    tasks = p7b.build_validation_tasks(selections, p3d_selected, formulas, horizon, validation_days)
    print(f"[3/6] P9 derivatives AlphaGPT validation: {len(tasks)} tasks", flush=True)
    rows = run_pool(
        tasks,
        initializer=init_long_p7b_worker,
        initargs=(
            str(config_path),
            str(panel_path),
            str(feature_path),
            str(p6b_panel),
            str(retained_features),
            horizon,
            {item.name: item for item in formulas},
            {item.name: item for item in p3d_formulas},
            p3d_selected,
        ),
        worker=p7b.run_validation_task,
        workers=workers,
        chunksize=chunksize,
        label="P9",
    )
    validation = pd.DataFrame(rows)
    set_p7_paths(p6b_panel, retained_features)
    p3d_validation = p7b.run_p3d_validation_baselines(config_path, panel_path, feature_path, selections, p3d_selected, p3d_formulas, horizon)
    p3d_validation.to_csv(output_dir / "validation_p3d_metrics.csv", index=False)
    validation = p7b.annotate_hard_accept(validation, p3d_validation)
    validation = p9.apply_p9_hard_accept(validation, formulas, require_30bps_improvement=True, short_share_limit=0.70)
    validation.to_csv(output_dir / "validation_incremental_metrics.csv", index=False)
    selected = p7b.select_by_fold(validation, p3d_validation, selections, p3d_selected, horizon)
    selected.to_csv(selected_path, index=False)

    set_p7_paths(p6b_panel, retained_features)
    panel = p7.load_panel_p7(config_path, panel_path, feature_path, horizon)
    pf_cfg = p3d.make_portfolio_config(cfg, horizon)
    risk_template = p3d.make_risk_template(cfg)
    result = p7b.replay_selected_tests(
        panel,
        selections,
        selected,
        pf_cfg,
        risk_template,
        {spec.name: spec for spec in p3d.build_committee_specs()},
        {item.name: item for item in formulas},
        {item.name: item for item in p3d_formulas},
    )
    p3d.write_run_result(run_dir, result)
    write_metrics(output_dir / "summary.json", result["metrics"])
    return result, selected, formulas


def run_p15c_stage(
    config_path: Path,
    cfg: dict[str, Any],
    panel_path: Path,
    feature_path: Path,
    p6b_panel: Path,
    retained_features: Path,
    derivatives_panel: Path,
    selections: pd.DataFrame,
    p9_selected: pd.DataFrame,
    p3d_formulas: list[Any],
    deriv_formulas: list[Any],
    output_dir: Path,
    horizon: int,
    validation_days: int,
    force: bool,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    run_dir = ensure_dir(output_dir / "nested_extreme_cluster_run")
    selected_path = output_dir / "selected_extreme_cluster_by_fold.csv"
    if not force and selected_path.exists() and (run_dir / "equity_curve.csv").exists():
        return load_result(run_dir), pd.read_csv(selected_path)

    set_p7_paths(p6b_panel, retained_features)
    panel = p7.load_panel_p7(config_path, panel_path, feature_path, horizon)
    residual_wide = (
        panel[["timestamp", "symbol", "residual_log_ret"]]
        .dropna()
        .pivot(index="timestamp", columns="symbol", values="residual_log_ret")
        .sort_index()
    )
    pf_cfg = p3d.make_portfolio_config(cfg, horizon)
    risk_template = p3d.make_risk_template(cfg)
    spec_by_name = {spec.name: spec for spec in p3d.build_committee_specs()}
    extreme_specs = p15c.build_extreme_specs()
    pd.DataFrame([spec.__dict__ for spec in extreme_specs]).to_csv(output_dir / "extreme_cluster_grid.csv", index=False)

    print("[4/6] P15c allocator validation search", flush=True)
    validation, p9_validation = p15c.run_validation_search(
        panel=panel,
        residual_wide=residual_wide,
        selections=selections,
        selected_p9=p9_selected,
        pf_cfg=pf_cfg,
        risk_template=risk_template,
        spec_by_name=spec_by_name,
        deriv_formulas={item.name: item for item in deriv_formulas},
        p3d_formulas={item.name: item for item in p3d_formulas},
        extreme_specs=extreme_specs,
        validation_days=validation_days,
    )
    validation.to_csv(output_dir / "validation_extreme_cluster_metrics.csv", index=False)
    p9_validation.to_csv(output_dir / "validation_p9_cluster_reference.csv", index=False)
    selected = p15c.select_extreme_spec_by_fold(validation, p9_validation, selections)
    selected.to_csv(selected_path, index=False)

    print("[5/6] P15c nested test replay", flush=True)
    result = p15c.replay_selected_specs(
        panel=panel,
        residual_wide=residual_wide,
        selections=selections,
        selected=selected,
        selected_p9=p9_selected,
        pf_cfg=pf_cfg,
        risk_template=risk_template,
        spec_by_name=spec_by_name,
        deriv_formulas={item.name: item for item in deriv_formulas},
        p3d_formulas={item.name: item for item in p3d_formulas},
        extreme_specs=extreme_specs,
        force_baseline=False,
    )
    write_result(run_dir, result)
    write_metrics(output_dir / "summary.json", result["metrics"])
    return result, selected


def init_long_p7b_worker(
    config_path_str: str,
    panel_path_str: str,
    feature_path_str: str,
    p6b_panel_str: str,
    retained_features_str: str,
    horizon: int,
    deriv_formulas: dict[str, Any],
    p3d_formulas: dict[str, Any],
    p3d_selected: pd.DataFrame,
) -> None:
    set_p7_paths(Path(p6b_panel_str), Path(retained_features_str))
    p7b.init_worker(config_path_str, panel_path_str, feature_path_str, horizon, deriv_formulas, p3d_formulas, p3d_selected)


def set_p7_paths(p6b_panel: Path, retained_features: Path) -> None:
    p7.P6B_PANEL = Path(p6b_panel)
    p7.P6B_RETAINED = Path(retained_features)
    p7._P7_FEATURE_MAP = {}


def run_pool(
    tasks: list[dict[str, Any]],
    initializer: Callable[..., None],
    initargs: tuple[Any, ...],
    worker: Callable[[dict[str, Any]], dict[str, Any]],
    workers: int,
    chunksize: int,
    label: str,
) -> list[dict[str, Any]]:
    start = time.time()
    rows: list[dict[str, Any]] = []
    if workers == 1:
        initializer(*initargs)
        for idx, task in enumerate(tasks, start=1):
            if idx == 1 or idx % 500 == 0 or idx == len(tasks):
                print(f"      {label} task {idx}/{len(tasks)} elapsed={time.time() - start:.1f}s", flush=True)
            rows.append(worker(task))
        return rows
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers, initializer=initializer, initargs=initargs) as pool:
        for idx, row in enumerate(pool.imap_unordered(worker, tasks, chunksize=max(1, chunksize)), start=1):
            rows.append(row)
            if idx == 1 or idx % 500 == 0 or idx == len(tasks):
                print(f"      {label} task {idx}/{len(tasks)} elapsed={time.time() - start:.1f}s", flush=True)
    return rows


def load_result(run_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        "folds": pd.read_csv(run_dir / "walk_forward_folds.csv", parse_dates=["train_start", "train_end", "test_start", "test_end", "test_decision_end"]),
        "equity": pd.read_csv(run_dir / "equity_curve.csv", parse_dates=["timestamp", "exit_timestamp"]),
        "positions": pd.read_csv(run_dir / "positions.csv", parse_dates=["timestamp", "exit_timestamp"]),
        "metrics": json.loads((run_dir / "metrics.json").read_text()) if (run_dir / "metrics.json").exists() else {},
    }


def write_metrics(path: Path, metrics: dict[str, Any]) -> None:
    path.write_text(json.dumps(to_jsonable(metrics), indent=2, ensure_ascii=False), encoding="utf-8")


def write_package(
    output_dir: Path,
    inputs: dict[str, Any],
    results: dict[str, dict[str, pd.DataFrame]],
    selected: dict[str, pd.DataFrame],
    horizon: int,
    cost_bps: list[float],
) -> None:
    summary = build_summary(results, horizon)
    summary.to_csv(output_dir / "summary.csv", index=False)
    cost = build_cost_table(results, horizon, cost_bps)
    cost.to_csv(output_dir / "cost_sensitivity.csv", index=False)
    segments = build_segment_table(results["p15c"]["equity"], horizon, cost_bps)
    segments.to_csv(output_dir / "p15c_segment_cost_summary.csv", index=False)
    asset = build_asset_contribution(results["p15c"]["positions"])
    asset.to_csv(output_dir / "p15c_asset_contribution.csv", index=False)
    plot_curves(output_dir / "nested_oos_pnl_6bps.png", results, cost_bps=6.0)
    plot_curves(output_dir / "nested_oos_pnl_15bps.png", results, cost_bps=15.0)
    write_report(output_dir, inputs, summary, cost, segments, asset, selected)


def build_summary(results: dict[str, dict[str, pd.DataFrame]], horizon: int) -> pd.DataFrame:
    rows = []
    for name, result in results.items():
        metrics = p3d.summarize_nested_result(result, horizon)
        rows.append({"stage": name, **metrics})
    return pd.DataFrame(rows)


def build_cost_table(results: dict[str, dict[str, pd.DataFrame]], horizon: int, cost_bps: list[float]) -> pd.DataFrame:
    rows = []
    for name, result in results.items():
        eq = result["equity"].sort_values("timestamp")
        for bps in cost_bps:
            rows.append({"stage": name, "one_way_cost_bps": float(bps), **calc_metrics_at_cost(eq, horizon, float(bps))})
    return pd.DataFrame(rows)


def build_segment_table(equity: pd.DataFrame, horizon: int, cost_bps: list[float]) -> pd.DataFrame:
    data = equity.sort_values("timestamp").copy()
    original = pd.Timestamp("2024-07-11 08:00:00+00:00")
    segments = {
        "full": data,
        "pre_original_2023_2024H1": data[data["timestamp"] < original],
        "original_window_2024_07_plus": data[data["timestamp"] >= original],
    }
    rows = []
    for segment, frame in segments.items():
        if frame.empty:
            continue
        for bps in cost_bps:
            rows.append({"segment": segment, "one_way_cost_bps": float(bps), **calc_metrics_at_cost(frame, horizon, float(bps))})
    return pd.DataFrame(rows)


def calc_metrics_at_cost(equity: pd.DataFrame, horizon: int, cost_bps: float) -> dict[str, float]:
    gross = equity["gross_period_return"].astype(float)
    net = gross - equity["turnover"].astype(float) * cost_bps / 10000.0
    eq = (1.0 + net).cumprod()
    gross_eq = (1.0 + gross).cumprod()
    ann = 365.0 * 24.0 / float(horizon)
    sharpe = float(net.mean() / net.std(ddof=0) * np.sqrt(ann)) if net.std(ddof=0) > 0 else 0.0
    dd = eq / eq.cummax() - 1.0
    gross_dd = gross_eq / gross_eq.cummax() - 1.0
    return {
        "periods": float(len(equity)),
        "total_return": float(eq.iloc[-1] - 1.0) if len(eq) else np.nan,
        "gross_total_return": float(gross_eq.iloc[-1] - 1.0) if len(gross_eq) else np.nan,
        "sharpe": sharpe,
        "max_drawdown": float(dd.min()) if len(dd) else np.nan,
        "gross_max_drawdown": float(gross_dd.min()) if len(gross_dd) else np.nan,
        "mean_turnover": float(equity["turnover"].mean()) if "turnover" in equity else np.nan,
        "total_turnover": float(equity["turnover"].sum()) if "turnover" in equity else np.nan,
        "positive_period_rate": float((net > 0.0).mean()) if len(net) else np.nan,
    }


def build_asset_contribution(positions: pd.DataFrame) -> pd.DataFrame:
    if positions.empty:
        return pd.DataFrame()
    out = (
        positions.groupby("symbol")
        .agg(
            gross_contribution=("contribution", "sum"),
            avg_abs_weight=("weight", lambda s: s.abs().mean()),
            max_abs_weight=("weight", lambda s: s.abs().max()),
            observations=("weight", "size"),
        )
        .reset_index()
    )
    total_abs = out["gross_contribution"].abs().sum()
    out["abs_contribution_share"] = out["gross_contribution"].abs() / total_abs if total_abs else 0.0
    return out.sort_values("gross_contribution")


def plot_curves(path: Path, results: dict[str, dict[str, pd.DataFrame]], cost_bps: float) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, result in results.items():
        eq = result["equity"].sort_values("timestamp").copy()
        if eq.empty:
            continue
        net = eq["gross_period_return"].astype(float) - eq["turnover"].astype(float) * cost_bps / 10000.0
        curve = (1.0 + net).cumprod() - 1.0
        plot = pd.concat(
            [
                pd.DataFrame({"timestamp": [eq["timestamp"].iloc[0]], "curve": [0.0]}),
                pd.DataFrame({"timestamp": eq["timestamp"], "curve": curve}),
            ],
            ignore_index=True,
        )
        ax.plot(plot["timestamp"], plot["curve"], label=name, linewidth=1.8 if name == "p15c" else 1.2)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title(f"Long-history nested OOS cumulative return, {cost_bps:.0f}bps one-way cost")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Cumulative return")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def write_report(
    output_dir: Path,
    inputs: dict[str, Any],
    summary: pd.DataFrame,
    cost: pd.DataFrame,
    segments: pd.DataFrame,
    asset: pd.DataFrame,
    selected: dict[str, pd.DataFrame],
) -> None:
    p15_cost = cost[cost["stage"] == "p15c"].copy()
    p15_cost_display = p15_cost[p15_cost["one_way_cost_bps"].isin([0.0, 6.0, 15.0, 20.0, 30.0])]
    report = [
        "# Long-History Nested OOS Pipeline",
        "",
        "This package reruns fold-local parameter selection on the long 20-symbol common Binance history.",
        "",
        "Scope:",
        "",
        "- P4 committee parameters are selected inside each fold's training window.",
        "- P3d slow AlphaGPT incremental formulas and weights are selected on fold-local validation windows.",
        "- P9 derivatives incremental formulas and weights are selected on fold-local validation windows.",
        "- P15c allocator/cluster-cap parameters are selected on fold-local validation windows.",
        "- Final PnL is stitched from test windows only.",
        "- The P6/P6b primitive library is treated as a predeclared candidate universe; no full-sample retained-feature filter is used for this run.",
        "",
        "Inputs:",
        "",
        f"- asset panel: `{inputs['asset_panel']}`",
        f"- manual slow features: `{inputs['manual_features']}`",
        f"- P6/P6b features: `{inputs['p6b_features']}`",
        f"- folds: `{inputs['selections_path']}`",
        "",
        "## Stage Summary",
        "",
        "```text",
        summary.to_string(index=False),
        "```",
        "",
        "## P15c Cost Sensitivity",
        "",
        "```text",
        p15_cost_display.to_string(index=False),
        "```",
        "",
        "## P15c Segment Summary",
        "",
        "```text",
        segments[segments["one_way_cost_bps"].isin([6.0, 15.0, 20.0, 30.0])].to_string(index=False),
        "```",
        "",
        "## Selection Counts",
        "",
    ]
    for name, frame in selected.items():
        if frame.empty:
            continue
        preferred_cols = [
            "selected_spec",
            "selected_extreme_spec",
            "deriv_formula_name",
            "formula_name",
            "p3d_formula_name",
            "p4_spec",
        ]
        col = next((candidate for candidate in preferred_cols if candidate in frame.columns), None)
        if col is None:
            report.extend([f"### {name}", "", "```text", "No recognized selection column.", "```", ""])
            continue
        report.extend([f"### {name}", "", "```text", frame[col].value_counts().to_string(), "```", ""])
    report.extend(["## P15c Asset Contribution", "", "```text", asset.to_string(index=False), "```"])
    (output_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
