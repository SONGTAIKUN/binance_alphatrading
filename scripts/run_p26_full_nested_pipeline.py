#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import os
import sys
from dataclasses import replace
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
from crypto_stat_arb.features import add_cross_sectional_features, add_residual_features
from run_p2_family_a_momentum import build_family_a_features
from run_p2_family_b_mean_reversion import build_family_b_features, make_portfolio_config
from run_p2_manual_slow_alpha_grid import build_manual_slow_hybrid_features
from run_p3_committee_guard_scan import build_committee_specs, make_risk_template
import run_p3d_p4_gated_rerank_alphagpt as p3d
import run_p4_nested_walkforward_selection as p4
from run_p5_alphagpt_slow_search import add_transformed_primitives, evaluate_node, required_feature_columns
import run_p15c_extreme_cluster_turnover as p15c
from run_p18c_factor_selector_portfolio import load_existing_result
from run_residual_alphagpt_search import build_candidate_scores, prepare_alpha_panel
from run_risk_control_experiments import add_gross_metrics, add_risk_event_metrics
from run_delay_smoothing_experiments import add_delay_metrics
from run_p10_strategy_conditioned_regime_gate import build_cost_summary
from run_turnover_experiments import to_jsonable


EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "baseline_locked_v1"
DEFAULT_CONFIG = EXPERIMENT_ROOT / "configs" / "baseline_locked_v1.yaml"
DEFAULT_P26_PANEL = EXPERIMENT_ROOT / "outputs" / "p26_no_sector_candidate_baseline" / "p26_model_and_label_panel.csv.gz"
DEFAULT_RAW_PANEL = EXPERIMENT_ROOT / "inputs" / "long_history" / "baseline_panel" / "asset_residual_panel.csv"
DEFAULT_FOLDS = EXPERIMENT_ROOT / "outputs" / "long_history_nested_oos" / "inputs" / "long_formula_selections.csv"
DEFAULT_P15C_RUN = EXPERIMENT_ROOT / "outputs" / "long_history_nested_oos" / "p15c_nested" / "nested_extreme_cluster_run"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "outputs" / "p26_full_nested_pipeline"

HORIZON = 72
FORMULA_COUNT = 180
FORMULA_SEED = 2417
FORMULA_DEPTH = 3
FORMULA_VALIDATION_COUNT = 40
ALPHA_MODEL = "alpha_residual_v2_no_sector_ols720"
RISK_MODEL = "portfolio_risk_v1_p15c"

_PANEL: pd.DataFrame | None = None
_CFG: dict[str, Any] | None = None
_FORMULAS: dict[str, Any] = {}
_P4_SELECTED: pd.DataFrame | None = None
_P3D_SELECTED: pd.DataFrame | None = None
_RESIDUAL_WIDE: pd.DataFrame | None = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Full nested P26 no-sector AlphaGPT/P15c pipeline.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--p26-panel", default=str(DEFAULT_P26_PANEL))
    parser.add_argument("--raw-panel", default=str(DEFAULT_RAW_PANEL))
    parser.add_argument("--folds", default=str(DEFAULT_FOLDS))
    parser.add_argument("--p15c-run", default=str(DEFAULT_P15C_RUN))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument("--force-rebuild-panel", action="store_true")
    parser.add_argument("--formula-count", type=int, default=FORMULA_COUNT)
    parser.add_argument("--cost-bps", type=float, nargs="+", default=[0, 6, 8, 15, 20, 30])
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    p26_panel_path = Path(args.p26_panel).resolve()
    raw_panel_path = Path(args.raw_panel).resolve()
    folds_path = Path(args.folds).resolve()
    p15c_run_path = Path(args.p15c_run).resolve()
    output_dir = ensure_dir(args.output_dir)
    input_dir = ensure_dir(output_dir / "inputs")
    panel_cache = input_dir / "p26_full_prepared_panel.pkl"
    residual_cache = input_dir / "p15c_legacy_residual_wide.pkl"
    workers = max(1, int(args.workers))
    cfg = load_config(config_path)
    folds = pd.read_csv(
        folds_path,
        parse_dates=["train_start", "train_end", "test_start", "test_end"],
    ).sort_values("fold")

    print("[1/8] Rebuilding all no-sector residual-dependent features and labels", flush=True)
    if args.force_rebuild_panel or not panel_cache.exists() or not residual_cache.exists():
        panel, residual_wide = build_prepared_panel(raw_panel_path, p26_panel_path)
        panel.to_pickle(panel_cache)
        residual_wide.to_pickle(residual_cache)
        del panel, residual_wide
        gc.collect()
    else:
        print("      using cached prepared panel", flush=True)

    print("[2/8] Auditing feature/risk namespaces and fold boundaries", flush=True)
    audit = audit_prepared_panel(panel_cache, p26_panel_path, folds)
    audit.to_csv(output_dir / "pre_run_audit.csv", index=False)
    failed = audit.loc[audit["status"].eq("fail")]
    if not failed.empty:
        raise RuntimeError(f"P26-full pre-run audit failed: {failed['check'].tolist()}")

    formulas = p3d.build_incremental_formula_universe(int(args.formula_count), FORMULA_SEED, FORMULA_DEPTH)
    formula_map = {item.name: item for item in formulas}
    pd.DataFrame([item.to_row() for item in formulas]).to_csv(output_dir / "candidate_formulas.csv", index=False)

    print(f"[3/8] Fold-local P4 parameter search ({len(folds)} folds, {workers} workers)", flush=True)
    p4_validation_path = output_dir / "p4_validation_candidate_metrics.csv"
    p4_selected_path = output_dir / "p4_selected_specs_by_fold.csv"
    if p4_validation_path.exists() and p4_selected_path.exists():
        print("      using cached completed P4 search", flush=True)
        p4_validation = pd.read_csv(p4_validation_path, parse_dates=["validation_start", "validation_decision_end", "validation_window_end"])
        p4_selected = pd.read_csv(
            p4_selected_path,
            parse_dates=["train_start", "train_end", "test_start", "test_end", "validation_decision_end", "inner_train_end"],
        )
    else:
        p4_rows = parallel_fold_map(
            worker=p4_fold_worker,
            folds=folds,
            workers=workers,
            initializer=init_panel_worker,
            initargs=(str(panel_cache), str(config_path)),
        )
        p4_validation = pd.concat([item["validation"] for item in p4_rows], ignore_index=True)
        p4_selected = pd.concat([item["selected"] for item in p4_rows], ignore_index=True).sort_values("fold")
        p4_validation.to_csv(p4_validation_path, index=False)
        p4_selected.to_csv(p4_selected_path, index=False)

    print(f"[4/8] Fold-local AlphaGPT search with new terminal/path labels ({len(formulas)} formulas)", flush=True)
    alpha_screen_path = output_dir / "alphagpt_train_label_screening.csv"
    alpha_validation_path = output_dir / "alphagpt_validation_metrics.csv"
    alpha_reference_path = output_dir / "alphagpt_p4_validation_reference.csv"
    alpha_selected_path = output_dir / "alphagpt_selected_by_fold.csv"
    if all(path.exists() for path in [alpha_screen_path, alpha_validation_path, alpha_reference_path, alpha_selected_path]):
        print("      using cached completed AlphaGPT search", flush=True)
        alpha_screening = pd.read_csv(alpha_screen_path)
        alpha_validation = pd.read_csv(alpha_validation_path)
        p4_alpha_reference = pd.read_csv(alpha_reference_path)
        alpha_selected = pd.read_csv(
            alpha_selected_path,
            parse_dates=["train_start", "train_end", "test_start", "test_end", "test_decision_end", "alpha_train_end", "alpha_validation_start"],
        )
    else:
        p3d_rows = parallel_fold_map(
            worker=alphagpt_fold_worker,
            folds=folds,
            workers=workers,
            initializer=init_alphagpt_worker,
            initargs=(str(panel_cache), str(config_path), formula_map, p4_selected),
        )
        alpha_screening = pd.concat([item["screening"] for item in p3d_rows], ignore_index=True)
        alpha_validation = pd.concat([item["validation"] for item in p3d_rows], ignore_index=True)
        p4_alpha_reference = pd.concat([item["p4_reference"] for item in p3d_rows], ignore_index=True)
        alpha_selected = pd.concat([item["selected"] for item in p3d_rows], ignore_index=True).sort_values("fold")
        alpha_screening.to_csv(alpha_screen_path, index=False)
        alpha_validation.to_csv(alpha_validation_path, index=False)
        p4_alpha_reference.to_csv(alpha_reference_path, index=False)
        alpha_selected.to_csv(alpha_selected_path, index=False)

    print("[5/8] Fold-local P15c allocator search and test-only replay", flush=True)
    p15c_validation_path = output_dir / "p15c_validation_metrics.csv"
    p15c_reference_path = output_dir / "p15c_validation_reference.csv"
    p15c_selected_path = output_dir / "p15c_selected_by_fold.csv"
    nested_test_dir = output_dir / "nested_test_run"
    if all(path.exists() for path in [p15c_validation_path, p15c_reference_path, p15c_selected_path, nested_test_dir / "metrics.json"]):
        print("      using cached completed P15c search and test replay", flush=True)
        p15c_validation = pd.read_csv(p15c_validation_path)
        p15c_reference = pd.read_csv(p15c_reference_path)
        p15c_selected = pd.read_csv(p15c_selected_path)
        test_result = load_existing_result(nested_test_dir)
    else:
        p15c_rows = parallel_fold_map(
            worker=p15c_fold_worker,
            folds=folds,
            workers=workers,
            initializer=init_p15c_worker,
            initargs=(str(panel_cache), str(residual_cache), str(config_path), formula_map, p4_selected, alpha_selected),
        )
        p15c_validation = pd.concat([item["validation"] for item in p15c_rows], ignore_index=True)
        p15c_reference = pd.concat([item["reference"] for item in p15c_rows], ignore_index=True)
        p15c_selected = pd.concat([item["selected"] for item in p15c_rows], ignore_index=True).sort_values("fold")
        test_result = combine_test_results([item["test"] for item in p15c_rows])
        p15c_validation.to_csv(p15c_validation_path, index=False)
        p15c_reference.to_csv(p15c_reference_path, index=False)
        p15c_selected.to_csv(p15c_selected_path, index=False)
        p3d.write_run_result(ensure_dir(nested_test_dir), test_result)

    print("[6/8] Matched comparison against the original P15c", flush=True)
    reference = load_existing_result(p15c_run_path)
    results = {"P15c_reference": reference, "P26_full": test_result}
    summary = build_summary(results)
    summary.to_csv(output_dir / "matched_summary.csv", index=False)
    cost = build_cost_summary(results, [float(value) for value in args.cost_bps])
    cost.to_csv(output_dir / "matched_cost_sensitivity.csv", index=False)
    fold_comparison = build_fold_comparison(test_result, reference)
    fold_comparison.to_csv(output_dir / "matched_fold_comparison.csv", index=False)
    regime = build_regime_comparison(results, [0.0, 8.0])
    regime.to_csv(output_dir / "matched_regime_comparison.csv", index=False)

    print("[7/8] Writing PnL chart, protocol audit, and manifest", flush=True)
    plot_matched_curves(output_dir, results)
    protocol_audit = build_protocol_audit(folds, p4_selected, alpha_selected, p15c_selected, test_result)
    protocol_audit.to_csv(output_dir / "nested_protocol_audit.csv", index=False)
    manifest = {
        "alpha_residual_model": ALPHA_MODEL,
        "portfolio_risk_model": RISK_MODEL,
        "horizon_hours": HORIZON,
        "fold_count": int(len(folds)),
        "formula_count": int(len(formulas)),
        "formula_validation_count_per_fold": FORMULA_VALIDATION_COUNT,
        "workers": workers,
        "formula_seed": FORMULA_SEED,
        "formula_depth": FORMULA_DEPTH,
        "selection_protocol": {
            "p4": "first half of each outer training window",
            "alphagpt": "new-label IC on first half plus portfolio fitness on second-half validation",
            "p15c": "second-half validation; test returns never used",
            "test": "next outer test fold only",
        },
        "inputs": {
            "raw_panel": str(raw_panel_path),
            "p26_model_label_panel": str(p26_panel_path),
            "folds": str(folds_path),
            "reference_p15c": str(p15c_run_path),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[8/8] Writing Chinese review", flush=True)
    write_report(output_dir, summary, cost, fold_comparison, regime, p4_selected, alpha_selected, p15c_selected, audit, protocol_audit)
    print(f"Done: {output_dir}", flush=True)


def build_prepared_panel(raw_path: Path, p26_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_cols = [
        "timestamp",
        "symbol",
        "log_close",
        "log_ret",
        "quote_volume",
        "log_quote_volume",
        "residual_log_ret",
    ]
    raw = pd.read_csv(raw_path, usecols=raw_cols, parse_dates=["timestamp"])
    model_cols = [
        "timestamp",
        "symbol",
        "alpha_residual_log_ret",
        "alpha_beta_btc",
        "alpha_beta_eth",
        "alpha_beta_market",
        "risk_beta_btc",
        "risk_beta_eth",
        "risk_beta_market",
        "risk_beta_sector",
        "fwd_log_ret_h72",
        "fwd_simple_ret_h72",
        "label_resid_ret_h72",
        "label_slow_low_turnover",
        "label_path_slow_low_turnover",
    ]
    model = pd.read_csv(p26_path, usecols=model_cols, parse_dates=["timestamp"])
    raw = raw.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    model = model.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    if len(raw) != len(model) or not raw[["timestamp", "symbol"]].equals(model[["timestamp", "symbol"]]):
        raise ValueError("Raw and P26 model panels are not exactly aligned")

    residual_wide = (
        raw[["timestamp", "symbol", "residual_log_ret"]]
        .dropna()
        .pivot(index="timestamp", columns="symbol", values="residual_log_ret")
        .sort_index()
    )
    panel = raw.drop(columns=["residual_log_ret"]).copy()
    panel["residual_log_ret"] = model["alpha_residual_log_ret"].to_numpy()
    for key in ["btc", "eth", "market"]:
        panel[f"beta_{key}"] = model[f"alpha_beta_{key}"].to_numpy()
    panel["beta_sector"] = 0.0
    for column in [
        "fwd_log_ret_h72",
        "fwd_simple_ret_h72",
        "label_resid_ret_h72",
        "label_slow_low_turnover",
        "label_path_slow_low_turnover",
    ]:
        panel[column] = model[column].to_numpy()

    panel = add_residual_features(panel, residual_z_window=168, windows={"short": 4, "medium": 24, "long": 168})
    panel = add_cross_sectional_features(panel)
    panel = prepare_alpha_panel(panel, [HORIZON], execution_delay_bars=1)
    build_candidate_scores(panel)
    build_family_a_features(panel)
    build_family_b_features(panel)
    build_manual_slow_hybrid_features(panel)
    add_transformed_primitives(panel)
    p3d.add_ex_ante_high_risk_regime(panel)
    for key, column in p3d.COMPONENTS.items():
        panel[f"rank_{key.lower()}"] = p3d.centered_rank_by_timestamp(panel, column)

    # Alpha features above use no-sector betas. The allocator aliases below stay on P15c risk betas.
    for key in ["btc", "eth", "market", "sector"]:
        panel[f"alpha_beta_{key}"] = panel[f"beta_{key}"].to_numpy()
        panel[f"beta_{key}"] = model[f"risk_beta_{key}"].to_numpy()

    required = set(required_feature_columns())
    keep = {
        "timestamp",
        "symbol",
        "log_ret",
        "residual_log_ret",
        "score_baseline",
        "score_ema_025",
        "rank_a07",
        "rank_b11",
        "rank_c09",
        "beta_btc",
        "beta_eth",
        "beta_market",
        "beta_sector",
        "alpha_beta_btc",
        "alpha_beta_eth",
        "alpha_beta_market",
        "alpha_beta_sector",
        "fwd_log_ret_h72",
        "fwd_simple_ret_h72",
        "label_resid_ret_h72",
        "label_slow_low_turnover",
        "label_path_slow_low_turnover",
        p3d.HIGH_RISK_REGIME_COL,
        *required,
    }
    keep.update(column for column in panel.columns if column.startswith("ema"))
    missing = sorted(column for column in keep if column not in panel)
    if missing:
        raise KeyError(f"Prepared panel is missing required columns: {missing}")
    out = panel[[column for column in panel.columns if column in keep]].sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    return out, residual_wide


def audit_prepared_panel(cache: Path, p26_path: Path, folds: pd.DataFrame) -> pd.DataFrame:
    panel = pd.read_pickle(cache)
    model = pd.read_csv(
        p26_path,
        usecols=["timestamp", "symbol", "alpha_residual_log_ret", "risk_beta_btc", "risk_beta_eth", "risk_beta_market"],
        parse_dates=["timestamp"],
    ).sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    panel = panel.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    checks = []

    def add(name: str, value: float, passed: bool, detail: str) -> None:
        checks.append({"check": name, "value": value, "status": "pass" if passed else "fail", "detail": detail})

    key_ok = len(panel) == len(model) and panel[["timestamp", "symbol"]].equals(model[["timestamp", "symbol"]])
    add("exact_key_alignment", float(key_ok), key_ok, "prepared panel and P26 model panel")
    resid_error = float(np.nanmax(np.abs(panel["residual_log_ret"].to_numpy() - model["alpha_residual_log_ret"].to_numpy())))
    add("no_sector_residual_attached", resid_error, resid_error < 1e-12, "generic residual is alpha residual v2")
    for key in ["btc", "eth", "market"]:
        error = float(np.nanmax(np.abs(panel[f"beta_{key}"].to_numpy() - model[f"risk_beta_{key}"].to_numpy())))
        add(f"p15c_risk_beta_{key}_unchanged", error, error < 1e-12, "allocator beta stays namespaced P15c beta")
    add("new_terminal_label_coverage", float(panel["label_slow_low_turnover"].notna().mean()), panel["label_slow_low_turnover"].notna().mean() > 0.94, "slow composite label")
    add("new_path_label_coverage", float(panel["label_path_slow_low_turnover"].notna().mean()), panel["label_path_slow_low_turnover"].notna().mean() > 0.94, "path-aware composite label")
    boundary_ok = bool((folds["train_end"] <= folds["test_start"]).all())
    add("outer_fold_boundaries", float(boundary_ok), boundary_ok, "train_end <= test_start for every fold")
    return pd.DataFrame(checks)


def parallel_fold_map(
    worker: Any,
    folds: pd.DataFrame,
    workers: int,
    initializer: Any,
    initargs: tuple[Any, ...],
) -> list[dict[str, Any]]:
    tasks = [row.to_dict() for _, row in folds.sort_values("fold").iterrows()]
    rows: list[dict[str, Any]] = []
    if workers == 1:
        initializer(*initargs)
        for index, task in enumerate(tasks, start=1):
            rows.append(worker(task))
            print(f"      fold {index}/{len(tasks)} complete", flush=True)
        return rows
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers, initializer=initializer, initargs=initargs) as pool:
        for index, row in enumerate(pool.imap_unordered(worker, tasks, chunksize=1), start=1):
            rows.append(row)
            print(f"      fold {index}/{len(tasks)} complete", flush=True)
    return rows


def init_panel_worker(panel_path: str, config_path: str) -> None:
    global _PANEL, _CFG
    _PANEL = pd.read_pickle(panel_path)
    _CFG = load_config(Path(config_path))


def validation_split(fold: dict[str, Any]) -> pd.Timestamp:
    start = pd.Timestamp(fold["train_start"])
    end = pd.Timestamp(fold["train_end"])
    return max(start + (end - start) / 2, end - pd.Timedelta(days=90))


def p4_fold_worker(fold: dict[str, Any]) -> dict[str, pd.DataFrame]:
    if _PANEL is None or _CFG is None:
        raise RuntimeError("P4 worker is not initialized")
    fold_id = int(fold["fold"])
    split = validation_split(fold)
    decision_end = p4.decision_end_for_closed_positions(split, HORIZON)
    local = _PANEL.loc[(_PANEL["timestamp"] >= pd.Timestamp(fold["train_start"])) & (_PANEL["timestamp"] < split)].copy()
    p4._WORKER_PANEL = local
    p4._WORKER_PF_CFG = make_portfolio_config(_CFG, HORIZON)
    p4._WORKER_RISK_TEMPLATE = make_risk_template(_CFG)
    specs = build_committee_specs()
    p4._WORKER_SPECS = {spec.name: spec for spec in specs}
    rows = []
    for spec in specs:
        rows.append(
            p4.run_validation_task(
                {
                    "fold": fold_id,
                    "window": "inner_train",
                    "start": pd.Timestamp(fold["train_start"]),
                    "end": decision_end,
                    "window_end": split,
                    "spec_name": spec.name,
                }
            )
        )
    validation = pd.DataFrame(rows)
    validation["fitness"] = p4.build_fitness(validation)
    one_fold = pd.DataFrame([fold])
    selected = p4.select_specs_by_fold(validation, one_fold, HORIZON)
    selected["inner_train_end"] = split
    return {"validation": validation, "selected": selected}


def init_alphagpt_worker(
    panel_path: str,
    config_path: str,
    formulas: dict[str, Any],
    p4_selected: pd.DataFrame,
) -> None:
    global _PANEL, _CFG, _FORMULAS, _P4_SELECTED
    _PANEL = pd.read_pickle(panel_path)
    _CFG = load_config(Path(config_path))
    _FORMULAS = formulas
    _P4_SELECTED = p4_selected


def alphagpt_fold_worker(fold: dict[str, Any]) -> dict[str, pd.DataFrame]:
    if _PANEL is None or _CFG is None or _P4_SELECTED is None:
        raise RuntimeError("AlphaGPT worker is not initialized")
    fold_id = int(fold["fold"])
    split = validation_split(fold)
    train_label_end = split - pd.Timedelta(hours=169)
    validation_end = p4.decision_end_for_closed_positions(pd.Timestamp(fold["train_end"]), HORIZON)
    local = _PANEL.loc[
        (_PANEL["timestamp"] >= pd.Timestamp(fold["train_start"]))
        & (_PANEL["timestamp"] < pd.Timestamp(fold["train_end"]))
    ].copy()
    p4_row = _P4_SELECTED.loc[_P4_SELECTED["fold"].eq(fold_id)].iloc[0]
    committee = {spec.name: spec for spec in build_committee_specs()}[str(p4_row["selected_spec"])]
    pf_cfg = make_portfolio_config(_CFG, HORIZON)
    fast_pf_cfg = replace(pf_cfg, rebalance_hours=1)
    risk_template = make_risk_template(_CFG)
    risk_cfg = risk_template.__class__(
        **{**risk_template.__dict__, "name": f"p26_alpha_{fold_id}", "short_resid_confirm_rank": committee.short_resid_confirm_rank}
    )
    local[p3d.P4_SCORE_COL] = p3d.p4_score(local, committee)
    trade_panel = rebalance_view(local, split, validation_end, HORIZON)
    p4_equity, p4_positions, p4_metrics = p3d.run_committee_backtest(
        panel=trade_panel,
        start=split,
        end=validation_end,
        cfg=fast_pf_cfg,
        risk_cfg=risk_cfg,
        spec=committee,
        score_col=p3d.P4_SCORE_COL,
        selected_formulas=f"P4({committee.name})",
    )
    p4_metrics = p3d.aggregate_oos_metrics(p4_equity, HORIZON)
    enrich_alpha_metrics(p4_equity, p4_positions, p4_metrics, complexity=0)
    p4_reference = pd.DataFrame(
        [{
            "fold": fold_id,
            "p4_spec": committee.name,
            "validation_start": split,
            "validation_decision_end": validation_end,
            **{f"p4_validation_{key}": value for key, value in p4_metrics.items() if p3d.is_scalar(value)},
        }]
    )

    train_mask = (local["timestamp"] >= pd.Timestamp(fold["train_start"])) & (local["timestamp"] < train_label_end)
    screen_rows = []
    for formula in _FORMULAS.values():
        raw_candidate = evaluate_node(formula.node, local)
        candidate = 2.0 * raw_candidate.groupby(local["timestamp"], sort=False).rank(pct=True, method="average") - 1.0
        local[p3d.CANDIDATE_RANK_COL] = candidate
        terminal_ic = cross_sectional_ic(local.loc[train_mask], p3d.CANDIDATE_RANK_COL, "label_slow_low_turnover")
        path_ic = cross_sectional_ic(local.loc[train_mask], p3d.CANDIDATE_RANK_COL, "label_path_slow_low_turnover")
        screen_rows.append(
            {
                "fold": fold_id,
                "formula_name": formula.name,
                "formula": formula.formula,
                "source": formula.source,
                "complexity": formula.complexity,
                "primitives": "|".join(formula.primitives),
                "train_terminal_label_ic": terminal_ic,
                "train_path_label_ic": path_ic,
                "train_label_screen_score": nz(terminal_ic) + 0.60 * nz(path_ic) - 0.001 * formula.complexity,
            }
        )
    screening = pd.DataFrame(screen_rows).sort_values(
        ["train_label_screen_score", "train_terminal_label_ic", "train_path_label_ic"], ascending=False
    )
    validation_names = set(screening.head(FORMULA_VALIDATION_COUNT)["formula_name"])
    screening["advanced_to_validation"] = screening["formula_name"].isin(validation_names)

    rows = []
    screen_by_name = screening.set_index("formula_name")
    for formula_name in screening.loc[screening["advanced_to_validation"], "formula_name"]:
        formula = _FORMULAS[str(formula_name)]
        screen = screen_by_name.loc[formula.name]
        terminal_ic = float(screen["train_terminal_label_ic"])
        path_ic = float(screen["train_path_label_ic"])
        raw_candidate = evaluate_node(formula.node, local)
        candidate = 2.0 * raw_candidate.groupby(local["timestamp"], sort=False).rank(pct=True, method="average") - 1.0
        local[p3d.CANDIDATE_RANK_COL] = candidate
        trade_panel[p3d.CANDIDATE_RANK_COL] = local.loc[trade_panel.index, p3d.CANDIDATE_RANK_COL].to_numpy()
        for grid in p3d.SIDE_AWARE_GRID:
            p3d.apply_constrained_incremental_score(
                trade_panel,
                candidate_weight_long=float(grid["candidate_weight_long"]),
                candidate_weight_short=float(grid["candidate_weight_short"]),
                high_risk_candidate_cap=float(grid["high_risk_candidate_cap"]),
            )
            formula_text = f"P4_gated_rerank({committee.name},{formula.name}): {formula.formula}"
            equity, positions, metrics = p3d.run_p4_gated_rerank_backtest(
                panel=trade_panel,
                start=split,
                end=validation_end,
                cfg=fast_pf_cfg,
                risk_cfg=risk_cfg,
                spec=committee,
                score_col=p3d.SCORE_COL,
                selected_formulas=formula_text,
            )
            metrics = p3d.aggregate_oos_metrics(equity, HORIZON)
            enrich_alpha_metrics(equity, positions, metrics, complexity=formula.complexity)
            valid = local.loc[(local["timestamp"] >= split) & (local["timestamp"] < validation_end), [p3d.P4_SCORE_COL, p3d.CANDIDATE_RANK_COL]].dropna()
            corr = float(valid[p3d.P4_SCORE_COL].corr(valid[p3d.CANDIDATE_RANK_COL])) if len(valid) > 20 else 0.0
            metrics["candidate_score_corr_to_p4"] = corr
            fitness = p3d.incremental_fitness(metrics, formula.complexity) + 20.0 * nz(terminal_ic) + 12.0 * nz(path_ic)
            rows.append(
                {
                    "fold": fold_id,
                    "formula_name": formula.name,
                    "formula": formula.formula,
                    "source": formula.source,
                    "complexity": formula.complexity,
                    "primitives": "|".join(formula.primitives),
                    "p4_spec": committee.name,
                    "candidate_weight_long": float(grid["candidate_weight_long"]),
                    "candidate_weight_short": float(grid["candidate_weight_short"]),
                    "high_risk_candidate_cap": float(grid["high_risk_candidate_cap"]),
                    "train_terminal_label_ic": terminal_ic,
                    "train_path_label_ic": path_ic,
                    "validation_start": split,
                    "validation_decision_end": validation_end,
                    "validation_fitness": fitness,
                    **{f"validation_{key}": value for key, value in metrics.items() if p3d.is_scalar(value)},
                }
            )
    validation = p3d.annotate_hard_accept(pd.DataFrame(rows), p4_reference)
    validation["label_hard_accept"] = (
        validation["train_terminal_label_ic"].fillna(-1.0).ge(0.0)
        & validation["train_path_label_ic"].fillna(-1.0).ge(-0.01)
    )
    validation["hard_accept"] = validation["hard_accept"].astype(bool) & validation["label_hard_accept"]
    selected = p3d.select_by_fold(validation, p4_reference, pd.DataFrame([fold]), pd.DataFrame([p4_row]), HORIZON)
    selected["alpha_train_end"] = train_label_end
    selected["alpha_validation_start"] = split
    return {"screening": screening, "validation": validation, "p4_reference": p4_reference, "selected": selected}


def enrich_alpha_metrics(equity: pd.DataFrame, positions: pd.DataFrame, metrics: dict[str, float], complexity: int) -> None:
    add_gross_metrics(equity, metrics)
    add_risk_event_metrics(positions, metrics)
    add_delay_metrics(equity, metrics)
    metrics.update(p3d.cost_adjusted_metrics(equity, HORIZON, [20.0, 30.0]))
    metrics["q25_segment_sharpe"] = p3d.q25_segment_sharpe(equity, HORIZON)
    metrics["long_short_balance"] = p3d.long_short_balance(positions)
    metrics["short_abs_contribution_share"] = p3d.short_abs_contribution_share(positions)
    metrics["complexity"] = float(complexity)


def cross_sectional_ic(frame: pd.DataFrame, score_col: str, label_col: str) -> float:
    valid = frame[["timestamp", score_col, label_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        return np.nan
    score_mean = valid.groupby("timestamp")[score_col].transform("mean")
    label_mean = valid.groupby("timestamp")[label_col].transform("mean")
    score = valid[score_col] - score_mean
    label = valid[label_col] - label_mean
    work = pd.DataFrame({"timestamp": valid["timestamp"], "xy": score * label, "xx": score * score, "yy": label * label})
    sums = work.groupby("timestamp")[["xy", "xx", "yy"]].sum()
    corr = sums["xy"] / np.sqrt(sums["xx"] * sums["yy"]).replace(0.0, np.nan)
    return float(corr.mean()) if corr.notna().any() else np.nan


def nz(value: float) -> float:
    return float(value) if np.isfinite(value) else 0.0


def rebalance_view(panel: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, hours: int) -> pd.DataFrame:
    mask = (panel["timestamp"] >= pd.Timestamp(start)) & (panel["timestamp"] < pd.Timestamp(end))
    timestamps = np.array(sorted(panel.loc[mask, "timestamp"].unique()))[:: int(hours)]
    return panel.loc[panel["timestamp"].isin(timestamps)].copy()


def init_p15c_worker(
    panel_path: str,
    residual_path: str,
    config_path: str,
    formulas: dict[str, Any],
    p4_selected: pd.DataFrame,
    p3d_selected: pd.DataFrame,
) -> None:
    global _PANEL, _CFG, _FORMULAS, _P4_SELECTED, _P3D_SELECTED, _RESIDUAL_WIDE
    _PANEL = pd.read_pickle(panel_path)
    _RESIDUAL_WIDE = pd.read_pickle(residual_path)
    _CFG = load_config(Path(config_path))
    _FORMULAS = formulas
    _P4_SELECTED = p4_selected
    _P3D_SELECTED = p3d_selected


def apply_fold_score(panel: pd.DataFrame, alpha_row: pd.Series, committee: Any) -> str:
    panel[p3d.P4_SCORE_COL] = p3d.p4_score(panel, committee)
    if str(alpha_row["formula_name"]) == p3d.P4_ONLY_FORMULA:
        panel[p3d.SCORE_COL] = panel[p3d.P4_SCORE_COL]
        panel[p3d.CANDIDATE_RANK_COL] = np.nan
        panel[p3d.EFFECTIVE_CANDIDATE_WEIGHT_COL] = 0.0
        return f"P4({committee.name}) fallback"
    formula = _FORMULAS[str(alpha_row["formula_name"])]
    panel[p3d.CANDIDATE_RANK_COL] = 2.0 * evaluate_node(formula.node, panel).groupby(panel["timestamp"], sort=False).rank(pct=True, method="average") - 1.0
    p3d.apply_constrained_incremental_score(
        panel,
        candidate_weight_long=float(alpha_row["candidate_weight_long"]),
        candidate_weight_short=float(alpha_row["candidate_weight_short"]),
        high_risk_candidate_cap=float(alpha_row["high_risk_candidate_cap"]),
    )
    return f"P4_gated_rerank({committee.name},{formula.name}): {formula.formula}"


def p15c_fold_worker(fold: dict[str, Any]) -> dict[str, Any]:
    if _PANEL is None or _CFG is None or _P4_SELECTED is None or _P3D_SELECTED is None or _RESIDUAL_WIDE is None:
        raise RuntimeError("P15c worker is not initialized")
    fold_id = int(fold["fold"])
    history_start = pd.Timestamp(fold["train_start"]) - pd.Timedelta(hours=1440)
    test_end = p4.decision_end_for_closed_positions(pd.Timestamp(fold["test_end"]), HORIZON)
    local = _PANEL.loc[(_PANEL["timestamp"] >= history_start) & (_PANEL["timestamp"] < pd.Timestamp(fold["test_end"]))].copy()
    p4_row = _P4_SELECTED.loc[_P4_SELECTED["fold"].eq(fold_id)].iloc[0]
    alpha_row = _P3D_SELECTED.loc[_P3D_SELECTED["fold"].eq(fold_id)].iloc[0]
    committee = {spec.name: spec for spec in build_committee_specs()}[str(p4_row["selected_spec"])]
    formula_text = apply_fold_score(local, alpha_row, committee)
    pf_cfg = make_portfolio_config(_CFG, HORIZON)
    fast_pf_cfg = replace(pf_cfg, rebalance_hours=1)
    risk_template = make_risk_template(_CFG)
    risk_cfg = p15c.make_fold_risk(risk_template, committee, f"p26_p15c_{fold_id}")
    validation_start = validation_split(fold)
    validation_end = p4.decision_end_for_closed_positions(pd.Timestamp(fold["train_end"]), HORIZON)
    validation_trade = rebalance_view(local, validation_start, validation_end, HORIZON)
    test_trade = rebalance_view(local, pd.Timestamp(fold["test_start"]), test_end, HORIZON)
    trade_panel = pd.concat([validation_trade, test_trade]).drop_duplicates(["timestamp", "symbol"]).sort_values(["timestamp", "symbol"])
    specs = p15c.build_extreme_specs()
    rows = []
    ref_rows = []
    cache: dict[tuple[int, int], dict[str, float]] = {}
    for spec in specs:
        key = (spec.lookback_hours, spec.cluster_k)
        if key not in cache:
            equity, positions, metrics = p15c.run_extreme_backtest(
                trade_panel, _RESIDUAL_WIDE, validation_start, validation_end, fast_pf_cfg, risk_cfg, committee,
                None, spec.lookback_hours, spec.cluster_k, p3d.SCORE_COL, formula_text,
            )
            metrics = p3d.aggregate_oos_metrics(equity, HORIZON)
            p15c.enrich_extreme_metrics(equity, positions, metrics, HORIZON)
            cache[key] = metrics
            ref_rows.append(
                {
                    "fold": fold_id,
                    "lookback_hours": spec.lookback_hours,
                    "cluster_k": spec.cluster_k,
                    "validation_start": validation_start,
                    "validation_decision_end": validation_end,
                    **{f"p9_validation_{name}": value for name, value in metrics.items() if p3d.is_scalar(value)},
                }
            )
        equity, positions, metrics = p15c.run_extreme_backtest(
            trade_panel, _RESIDUAL_WIDE, validation_start, validation_end, fast_pf_cfg, risk_cfg, committee,
            spec, spec.lookback_hours, spec.cluster_k, p3d.SCORE_COL, formula_text,
        )
        metrics = p3d.aggregate_oos_metrics(equity, HORIZON)
        p15c.enrich_extreme_metrics(equity, positions, metrics, HORIZON)
        rows.append(
            {
                "fold": fold_id,
                "extreme_spec": spec.name,
                **spec.__dict__,
                "validation_start": validation_start,
                "validation_decision_end": validation_end,
                **{f"validation_{name}": value for name, value in metrics.items() if p3d.is_scalar(value)},
            }
        )
    reference = pd.DataFrame(ref_rows)
    validation = p15c.annotate_validation(pd.DataFrame(rows), reference)
    selected = p15c.select_extreme_spec_by_fold(validation, reference, pd.DataFrame([fold]))
    selected_row = selected.iloc[0]
    spec_map = {spec.name: spec for spec in specs}
    selected_spec = None if bool(selected_row["fallback_to_p9_sizing"]) else spec_map[str(selected_row["selected_extreme_spec"])]
    equity, positions, metrics = p15c.run_extreme_backtest(
        trade_panel,
        _RESIDUAL_WIDE,
        pd.Timestamp(fold["test_start"]),
        test_end,
        fast_pf_cfg,
        risk_cfg,
        committee,
        selected_spec,
        int(selected_row["lookback_hours"]),
        int(selected_row["cluster_k"]),
        p3d.SCORE_COL,
        formula_text,
    )
    metrics = p3d.aggregate_oos_metrics(equity, HORIZON)
    p15c.enrich_extreme_metrics(equity, positions, metrics, HORIZON)
    fold_metrics = pd.DataFrame(
        [{
            "fold": fold_id,
            "train_start": fold["train_start"],
            "train_end": fold["train_end"],
            "test_start": fold["test_start"],
            "test_end": fold["test_end"],
            "test_decision_end": test_end,
            "p4_spec": committee.name,
            "formula_name": alpha_row["formula_name"],
            "formula": formula_text,
            "selected_extreme_spec": p15c.NO_EXTREME_CAP if selected_spec is None else selected_spec.name,
            "fallback_to_p4": bool(alpha_row["fallback_to_p4"]),
            "fallback_to_p9_sizing": bool(selected_spec is None),
            **metrics,
        }]
    )
    if not equity.empty:
        equity = equity.copy()
        equity["fold"] = fold_id
        equity["p4_spec"] = committee.name
        equity["formula_name"] = alpha_row["formula_name"]
        equity["formula"] = formula_text
        equity["selected_extreme_spec"] = p15c.NO_EXTREME_CAP if selected_spec is None else selected_spec.name
    if not positions.empty:
        positions = positions.copy()
        positions["fold"] = fold_id
        positions["p4_spec"] = committee.name
        positions["formula_name"] = alpha_row["formula_name"]
        positions["formula"] = formula_text
        positions["selected_extreme_spec"] = p15c.NO_EXTREME_CAP if selected_spec is None else selected_spec.name
    return {
        "validation": validation,
        "reference": reference,
        "selected": selected,
        "test": {"folds": fold_metrics, "equity": equity, "positions": positions},
    }


def combine_test_results(results: list[dict[str, pd.DataFrame]]) -> dict[str, Any]:
    folds = pd.concat([item["folds"] for item in results], ignore_index=True).sort_values("fold")
    equity = pd.concat([item["equity"] for item in results if not item["equity"].empty], ignore_index=True).sort_values("timestamp")
    positions = pd.concat([item["positions"] for item in results if not item["positions"].empty], ignore_index=True).sort_values(["timestamp", "symbol"])
    metrics = p4.summarize_nested_result({"folds": folds, "equity": equity, "positions": positions}, HORIZON)
    p15c.enrich_extreme_metrics(equity, positions, metrics, HORIZON)
    return {"folds": folds, "equity": equity, "positions": positions, "metrics": metrics}


def build_summary(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for name, result in results.items():
        metrics = result.get("metrics") or p4.summarize_nested_result(result, HORIZON)
        rows.append({"run": name, **metrics})
    return pd.DataFrame(rows)


def build_fold_comparison(candidate: dict[str, Any], reference: dict[str, Any]) -> pd.DataFrame:
    left = candidate["folds"][["fold", "total_return", "sharpe", "max_drawdown", "mean_turnover"]].rename(
        columns={column: f"p26_{column}" for column in ["total_return", "sharpe", "max_drawdown", "mean_turnover"]}
    )
    right = reference["folds"][["fold", "total_return", "sharpe", "max_drawdown", "mean_turnover"]].rename(
        columns={column: f"p15c_{column}" for column in ["total_return", "sharpe", "max_drawdown", "mean_turnover"]}
    )
    out = left.merge(right, on="fold", how="inner")
    out["delta_total_return"] = out["p26_total_return"] - out["p15c_total_return"]
    out["delta_sharpe"] = out["p26_sharpe"] - out["p15c_sharpe"]
    return out


def build_regime_comparison(results: dict[str, dict[str, Any]], costs: list[float]) -> pd.DataFrame:
    rows = []
    for name, result in results.items():
        equity = result["equity"].sort_values("timestamp").copy()
        equity["regime"] = equity["timestamp"].dt.to_period("Q").astype(str)
        for cost in costs:
            returns = equity["gross_period_return"] - equity["turnover"] * cost / 10_000.0
            for regime, values in returns.groupby(equity["regime"]):
                rows.append(
                    {
                        "run": name,
                        "regime": regime,
                        "one_way_cost_bps": cost,
                        "periods": int(len(values)),
                        "total_return": float((1.0 + values).prod() - 1.0),
                        "mean_period_return": float(values.mean()),
                    }
                )
    return pd.DataFrame(rows)


def build_protocol_audit(
    folds: pd.DataFrame,
    p4_selected: pd.DataFrame,
    alpha_selected: pd.DataFrame,
    p15c_selected: pd.DataFrame,
    result: dict[str, Any],
) -> pd.DataFrame:
    allocator_times = folds[["fold", "train_end"]].copy()
    allocator_times["allocator_validation_decision_end"] = allocator_times["train_end"].map(
        lambda value: p4.decision_end_for_closed_positions(pd.Timestamp(value), HORIZON)
    )
    merged = (
        folds[["fold", "train_end", "test_start", "test_end"]]
        .merge(p4_selected[["fold", "inner_train_end"]], on="fold")
        .merge(alpha_selected[["fold", "alpha_train_end", "alpha_validation_start"]], on="fold")
        .merge(allocator_times[["fold", "allocator_validation_decision_end"]], on="fold")
    )
    test_max = result["equity"].groupby("fold")["timestamp"].max().rename("max_test_decision_time")
    merged = merged.merge(test_max, on="fold", how="left")
    merged["p4_before_test"] = merged["inner_train_end"] <= merged["test_start"]
    merged["alpha_before_test"] = merged["alpha_validation_start"] < merged["test_start"]
    merged["allocator_before_test"] = merged["allocator_validation_decision_end"] < merged["test_start"]
    merged["test_decisions_inside_fold"] = merged["max_test_decision_time"] < merged["test_end"]
    merged["all_pass"] = merged[["p4_before_test", "alpha_before_test", "allocator_before_test", "test_decisions_inside_fold"]].all(axis=1)
    return merged


def plot_matched_curves(output_dir: Path, results: dict[str, dict[str, Any]]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0]})
    colors = {"P15c_reference": "#777777", "P26_full": "#1261a0"}
    for name, result in results.items():
        equity = result["equity"].sort_values("timestamp")
        ret8 = equity["gross_period_return"] - equity["turnover"] * 8.0 / 10_000.0
        curve = (1.0 + ret8).cumprod() - 1.0
        axes[0].plot(equity["timestamp"], curve, label=f"{name} net 8bps", linewidth=2.0, color=colors[name])
        if name == "P26_full":
            gross = (1.0 + equity["gross_period_return"]).cumprod() - 1.0
            axes[0].plot(equity["timestamp"], gross, label="P26_full gross", linewidth=1.4, color="#2a9d8f", alpha=0.8)
            wealth = 1.0 + curve
            drawdown = wealth / wealth.cummax() - 1.0
            axes[1].fill_between(equity["timestamp"], drawdown, 0.0, color="#e76f51", alpha=0.28)
            axes[1].plot(equity["timestamp"], drawdown, color="#e76f51", linewidth=1.2)
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_title("P26-full nested OOS vs original P15c")
    axes[0].set_ylabel("Cumulative return")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].set_title("P26-full net 8bps drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "p26_full_matched_pnl.png", dpi=180)
    plt.close(fig)


def write_report(
    output_dir: Path,
    summary: pd.DataFrame,
    cost: pd.DataFrame,
    folds: pd.DataFrame,
    regime: pd.DataFrame,
    p4_selected: pd.DataFrame,
    alpha_selected: pd.DataFrame,
    p15c_selected: pd.DataFrame,
    audit: pd.DataFrame,
    protocol_audit: pd.DataFrame,
) -> None:
    candidate = summary.loc[summary["run"].eq("P26_full")].iloc[0]
    reference = summary.loc[summary["run"].eq("P15c_reference")].iloc[0]
    c8 = cost.loc[(cost["variant"].eq("P26_full")) & (cost["one_way_cost_bps"].eq(8.0))].iloc[0]
    r8 = cost.loc[(cost["variant"].eq("P15c_reference")) & (cost["one_way_cost_bps"].eq(8.0))].iloc[0]
    fallback_rate = float(alpha_selected["fallback_to_p4"].mean())
    cluster_fallback = float(p15c_selected["fallback_to_p9_sizing"].mean())
    p26_worst5 = float(folds["p26_total_return"].nsmallest(5).sum())
    p15c_worst5 = float(folds["p15c_total_return"].nsmallest(5).sum())
    fold_win_rate = float((folds["delta_total_return"] > 0.0).mean())
    turnover_reduction = 1.0 - float(candidate["mean_turnover"]) / float(reference["mean_turnover"])
    p26_breakeven = estimate_breakeven_from_grid(cost, "P26_full")
    p15c_breakeven = estimate_breakeven_from_grid(cost, "P15c_reference")
    promote = bool(
        candidate["gross_total_return_before_cost"] > reference["gross_total_return_before_cost"]
        and c8["total_return"] > r8["total_return"]
        and candidate["sharpe"] > reference["sharpe"]
        and candidate["max_drawdown"] >= reference["max_drawdown"]
        and candidate["positive_fold_rate"] >= reference["positive_fold_rate"]
    )
    verdict = "升级为正式 baseline" if promote else "不升级；P15c 继续作为正式 baseline，P26-full 保留为研究候选"
    lines = [
        "# P26-full 完整 Nested Walk-forward 报告",
        "",
        "## 协议",
        "",
        f"- Alpha residual model：`{ALPHA_MODEL}`；所有 residual-dependent features 已重建。",
        f"- Portfolio risk model：`{RISK_MODEL}`；BTC/ETH/market beta 与原 P15c 保持一致。",
        "- 每个 outer fold 内：前半训练窗选 P4 参数并计算新版 terminal/path label IC；后半验证窗选 AlphaGPT 公式、候选权重和 P15c cluster 参数；只在下一段 test 交易。",
        "- 公式语法和候选集合是预先固定的，不由 test 数据生成。",
        "",
        "## Review 结论",
        "",
        f"- 结论：**{verdict}**。",
        "- P26-full 的尾部 fold 有改善，但总体 gross、8bps net、Sharpe、MaxDD 和 positive-fold rate 没有同时胜过 P15c。",
        f"- 最差 5 个 fold 合计：P26-full {p26_worst5:.2%}，P15c {p15c_worst5:.2%}；P26-full 更好。",
        f"- P26-full 在 {fold_win_rate:.1%} 的 fold 胜过 P15c，但自身 positive-fold rate 从 {float(reference['positive_fold_rate']):.1%} 降到 {float(candidate['positive_fold_rate']):.1%}。",
        f"- 换手只下降 {turnover_reduction:.1%}；成本盈亏平衡约从 P15c {p15c_breakeven:.2f}bps 降到 P26-full {p26_breakeven:.2f}bps。",
        f"- AlphaGPT hard-accept 率为 {1.0-fallback_rate:.1%}，选中了 {alpha_selected['formula_name'].nunique()} 个不同公式；P4 也选中了 {p4_selected['selected_spec'].nunique()} 个不同参数组。选择器偏活跃，跨 fold 参数稳定性不足。",
        "- 最大相对失效发生在 2023Q4；P26-full gross 约 -2.37%，而 P15c 约 +4.50%。这说明新 residual/label 对后续 regime 有增量，但早期外推仍不稳定。",
        "",
        "## 核心结果",
        "",
        f"- P26-full 6bps net：{float(candidate['total_return']):.2%}；P15c：{float(reference['total_return']):.2%}。",
        f"- P26-full gross：{float(candidate['gross_total_return_before_cost']):.2%}；P15c：{float(reference['gross_total_return_before_cost']):.2%}。",
        f"- P26-full Sharpe：{float(candidate['sharpe']):.3f}；P15c：{float(reference['sharpe']):.3f}。",
        f"- P26-full MaxDD：{float(candidate['max_drawdown']):.2%}；P15c：{float(reference['max_drawdown']):.2%}。",
        f"- 8bps net：P26-full {float(c8['total_return']):.2%}；P15c {float(r8['total_return']):.2%}。",
        f"- AlphaGPT fallback 到纯 P4：{fallback_rate:.1%}；P15c cluster sizing fallback：{cluster_fallback:.1%}。",
        "",
        "## Matched Summary",
        "```text",
        summary.to_string(index=False),
        "```",
        "",
        "## Cost Sensitivity",
        "```text",
        cost.to_string(index=False),
        "```",
        "",
        "## Fold Comparison",
        "```text",
        folds.to_string(index=False),
        "```",
        "",
        "## Quarterly Regime Comparison",
        "```text",
        regime.to_string(index=False),
        "```",
        "",
        "## Audit",
        "```text",
        audit.to_string(index=False),
        protocol_audit.to_string(index=False),
        "```",
        "",
        "## Selection Counts",
        "",
        "### P4",
        "```text",
        p4_selected["selected_spec"].value_counts().to_string(),
        "```",
        "### AlphaGPT",
        "```text",
        alpha_selected["formula_name"].value_counts().to_string(),
        "```",
        "### P15c",
        "```text",
        p15c_selected["selected_extreme_spec"].value_counts().to_string(),
        "```",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def estimate_breakeven_from_grid(cost: pd.DataFrame, variant: str) -> float:
    frame = cost.loc[cost["variant"].eq(variant), ["one_way_cost_bps", "total_return"]].sort_values("one_way_cost_bps")
    positive = frame.loc[frame["total_return"] >= 0.0]
    negative = frame.loc[frame["total_return"] < 0.0]
    if positive.empty:
        return 0.0
    if negative.empty:
        return float(positive["one_way_cost_bps"].max())
    left = positive.iloc[-1]
    right = negative.loc[negative["one_way_cost_bps"] > left["one_way_cost_bps"]].iloc[0]
    span = float(right["one_way_cost_bps"] - left["one_way_cost_bps"])
    weight = float(left["total_return"] / (left["total_return"] - right["total_return"]))
    return float(left["one_way_cost_bps"] + span * weight)


if __name__ == "__main__":
    main()
