#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import os
import sys
from dataclasses import dataclass, replace
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
from run_p2_family_b_mean_reversion import make_portfolio_config
from run_p3_committee_guard_scan import CommitteeSpec, make_risk_template
import run_p3d_p4_gated_rerank_alphagpt as p3d
import run_p4_nested_walkforward_selection as p4
import run_p15c_extreme_cluster_turnover as p15c
from run_p18c_factor_selector_portfolio import load_existing_result
from run_p26_full_nested_pipeline import (
    combine_test_results,
    parallel_fold_map,
    rebalance_view,
    validation_split,
)
from run_risk_control_experiments import add_gross_metrics, add_risk_event_metrics
from run_delay_smoothing_experiments import add_delay_metrics
from run_turnover_experiments import to_jsonable


EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "baseline_locked_v1"
DEFAULT_CONFIG = EXPERIMENT_ROOT / "configs" / "baseline_locked_v1.yaml"
DEFAULT_PREPARED = EXPERIMENT_ROOT / "outputs" / "p26_full_nested_pipeline" / "inputs" / "p26_full_prepared_panel.pkl"
DEFAULT_MODEL_LABEL = EXPERIMENT_ROOT / "outputs" / "p26_no_sector_candidate_baseline" / "p26_model_and_label_panel.csv.gz"
DEFAULT_RAW = EXPERIMENT_ROOT / "inputs" / "long_history" / "baseline_panel" / "asset_residual_panel.csv"
DEFAULT_FOLDS = EXPERIMENT_ROOT / "outputs" / "long_history_nested_oos" / "inputs" / "long_formula_selections.csv"
DEFAULT_RISK_RESIDUAL = EXPERIMENT_ROOT / "outputs" / "p26_full_nested_pipeline" / "inputs" / "p15c_legacy_residual_wide.pkl"
DEFAULT_P15C = EXPERIMENT_ROOT / "outputs" / "long_history_nested_oos" / "p15c_nested" / "nested_extreme_cluster_run"
DEFAULT_P26 = EXPERIMENT_ROOT / "outputs" / "p26_full_nested_pipeline" / "nested_test_run"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "outputs" / "p27_no_sector_native_factor_remine"

HORIZON = 72
MAX_LABEL_HORIZON = 168
FORMULA_SEED = 2727
FORMULA_COUNT = 240
FORMULA_VALIDATION_COUNT = 24
SMOOTHING_GRID = (1.0, 0.5, 0.25)


@dataclass(frozen=True)
class NativeFormula:
    name: str
    op: str
    left: str
    right: str | None
    gate: str | None
    weight: float
    complexity: int
    expression: str
    primitives: tuple[str, ...]

    def to_row(self) -> dict[str, Any]:
        return {
            "formula_name": self.name,
            "op": self.op,
            "left": self.left,
            "right": self.right or "",
            "gate": self.gate or "",
            "weight": self.weight,
            "complexity": self.complexity,
            "formula": self.expression,
            "primitives": "|".join(self.primitives),
        }


_PANEL: pd.DataFrame | None = None
_CFG: dict[str, Any] | None = None
_FORMULAS: dict[str, NativeFormula] = {}
_RISK_RESIDUAL: pd.DataFrame | None = None


def main() -> None:
    parser = argparse.ArgumentParser(description="P27 no-sector native factor re-mining and nested OOS portfolio replay.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--prepared-panel", default=str(DEFAULT_PREPARED))
    parser.add_argument("--model-label-panel", default=str(DEFAULT_MODEL_LABEL))
    parser.add_argument("--raw-panel", default=str(DEFAULT_RAW))
    parser.add_argument("--folds", default=str(DEFAULT_FOLDS))
    parser.add_argument("--risk-residual", default=str(DEFAULT_RISK_RESIDUAL))
    parser.add_argument("--p15c-run", default=str(DEFAULT_P15C))
    parser.add_argument("--p26-run", default=str(DEFAULT_P26))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument("--formula-count", type=int, default=FORMULA_COUNT)
    parser.add_argument("--force-rebuild-panel", action="store_true")
    parser.add_argument("--cost-bps", type=float, nargs="+", default=[0, 6, 8, 15, 20, 30])
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    output_dir = ensure_dir(Path(args.output_dir).resolve())
    input_dir = ensure_dir(output_dir / "inputs")
    native_panel_path = input_dir / "p27_no_sector_native_feature_panel.pkl"
    folds = pd.read_csv(
        Path(args.folds).resolve(),
        parse_dates=["train_start", "train_end", "test_start", "test_end"],
    ).sort_values("fold")

    print("[1/8] Building no-sector-native slow residual feature panel", flush=True)
    if args.force_rebuild_panel or not native_panel_path.exists():
        panel, primitive_rows = build_native_panel(
            Path(args.prepared_panel).resolve(),
            Path(args.model_label_panel).resolve(),
            Path(args.raw_panel).resolve(),
        )
        panel.to_pickle(native_panel_path)
        pd.DataFrame(primitive_rows).to_csv(output_dir / "native_primitives.csv", index=False)
        del panel
        gc.collect()
    else:
        print("      using cached native feature panel", flush=True)
        panel = pd.read_pickle(native_panel_path)
        primitive_rows = describe_primitives(panel)
        pd.DataFrame(primitive_rows).to_csv(output_dir / "native_primitives.csv", index=False)
        del panel
        gc.collect()

    print("[2/8] Building fixed symbolic formula universe", flush=True)
    formulas = build_formula_universe(primitive_rows, int(args.formula_count), FORMULA_SEED)
    formula_map = {item.name: item for item in formulas}
    pd.DataFrame([item.to_row() for item in formulas]).to_csv(output_dir / "native_formula_universe.csv", index=False)

    workers = max(1, int(args.workers))
    print(f"[3/8] Computing test-only factor diagnostics ({len(folds)} folds, {workers} workers)", flush=True)
    factor_fold_path = output_dir / "factor_oos_fold_metrics.csv"
    factor_summary_path = output_dir / "factor_oos_summary.csv"
    if factor_fold_path.exists() and factor_summary_path.exists():
        print("      using cached factor diagnostics", flush=True)
        factor_fold = pd.read_csv(factor_fold_path)
        factor_summary = pd.read_csv(factor_summary_path)
    else:
        factor_results = parallel_fold_map(
            worker=factor_diagnostic_worker,
            folds=folds,
            workers=workers,
            initializer=init_basic_worker,
            initargs=(str(native_panel_path), str(config_path), formula_map),
        )
        factor_fold = pd.concat(factor_results, ignore_index=True)
        factor_summary = summarize_factor_diagnostics(factor_fold)
        factor_fold.to_csv(factor_fold_path, index=False)
        factor_summary.to_csv(factor_summary_path, index=False)

    print("[4/8] Fold-local native formula and execution-parameter search", flush=True)
    screening_path = output_dir / "formula_train_screening.csv"
    validation_path = output_dir / "formula_validation_metrics.csv"
    execution_path = output_dir / "execution_validation_metrics.csv"
    selected_path = output_dir / "selected_native_formula_by_fold.csv"
    if all(path.exists() for path in [screening_path, validation_path, execution_path, selected_path]):
        print("      using cached formula search", flush=True)
        screening = pd.read_csv(screening_path)
        validation = pd.read_csv(validation_path)
        execution_validation = pd.read_csv(execution_path)
        selected = pd.read_csv(selected_path, parse_dates=["validation_start", "validation_decision_end", "train_label_end"])
    else:
        search_results = parallel_fold_map(
            worker=formula_search_worker,
            folds=folds,
            workers=workers,
            initializer=init_basic_worker,
            initargs=(str(native_panel_path), str(config_path), formula_map),
        )
        screening = pd.concat([item["screening"] for item in search_results], ignore_index=True)
        validation = pd.concat([item["validation"] for item in search_results], ignore_index=True)
        execution_validation = pd.concat([item["execution"] for item in search_results], ignore_index=True)
        selected = pd.concat([item["selected"] for item in search_results], ignore_index=True).sort_values("fold")
        screening.to_csv(screening_path, index=False)
        validation.to_csv(validation_path, index=False)
        execution_validation.to_csv(execution_path, index=False)
        selected.to_csv(selected_path, index=False)

    print("[5/8] Fold-local P15c cluster selection and test-only replay", flush=True)
    p15c_validation_path = output_dir / "p15c_validation_metrics.csv"
    p15c_reference_path = output_dir / "p15c_validation_reference.csv"
    p15c_selected_path = output_dir / "p15c_selected_by_fold.csv"
    run_dir = output_dir / "nested_test_run"
    if all(path.exists() for path in [p15c_validation_path, p15c_reference_path, p15c_selected_path, run_dir / "metrics.json"]):
        print("      using cached nested portfolio replay", flush=True)
        cluster_validation = pd.read_csv(p15c_validation_path)
        cluster_reference = pd.read_csv(p15c_reference_path)
        cluster_selected = pd.read_csv(p15c_selected_path)
        result = load_existing_result(run_dir)
    else:
        p15c_results = parallel_fold_map(
            worker=p15c_replay_worker,
            folds=folds,
            workers=workers,
            initializer=init_replay_worker,
            initargs=(
                str(native_panel_path),
                str(config_path),
                formula_map,
                selected,
                str(Path(args.risk_residual).resolve()),
            ),
        )
        cluster_validation = pd.concat([item["validation"] for item in p15c_results], ignore_index=True)
        cluster_reference = pd.concat([item["reference"] for item in p15c_results], ignore_index=True)
        cluster_selected = pd.concat([item["selected"] for item in p15c_results], ignore_index=True).sort_values("fold")
        result = combine_test_results([item["test"] for item in p15c_results])
        cluster_validation.to_csv(p15c_validation_path, index=False)
        cluster_reference.to_csv(p15c_reference_path, index=False)
        cluster_selected.to_csv(p15c_selected_path, index=False)
        p3d.write_run_result(ensure_dir(run_dir), result)

    print("[6/8] Building matched P15c/P26/P27 comparisons", flush=True)
    selected_formula_oos = build_selected_formula_oos_diagnostics(native_panel_path, folds, selected, formula_map)
    selected_formula_oos.to_csv(output_dir / "selected_formula_oos_fold_metrics.csv", index=False)
    selected_formula_oos_summary = summarize_selected_formula_oos(selected_formula_oos)
    selected_formula_oos_summary.to_csv(output_dir / "selected_formula_oos_summary.csv", index=False)
    p15c_result = load_existing_result(Path(args.p15c_run).resolve())
    p26_result = load_existing_result(Path(args.p26_run).resolve())
    results = {"P15c": p15c_result, "P26_full": p26_result, "P27_native": result}
    summary = build_trading_summary(results)
    cost = p3d.build_cost_sensitivity(results, [float(value) for value in args.cost_bps])
    folds_out = build_fold_comparison(results)
    regime = build_regime_comparison(results, [0.0, 8.0])
    summary.to_csv(output_dir / "trading_matched_summary.csv", index=False)
    cost.to_csv(output_dir / "trading_cost_sensitivity.csv", index=False)
    folds_out.to_csv(output_dir / "trading_fold_comparison.csv", index=False)
    regime.to_csv(output_dir / "trading_quarterly_regime.csv", index=False)

    print("[7/8] Auditing chronology and writing charts", flush=True)
    audit = build_protocol_audit(folds, selected, cluster_selected, result)
    audit.to_csv(output_dir / "nested_protocol_audit.csv", index=False)
    if not audit["all_pass"].all():
        raise RuntimeError("P27 nested protocol audit failed")
    plot_factor_diagnostics(output_dir, factor_summary)
    plot_trading_curves(output_dir, results)

    print("[8/8] Writing Chinese review and manifest", flush=True)
    write_report(
        output_dir,
        factor_summary,
        summary,
        cost,
        folds_out,
        regime,
        selected,
        selected_formula_oos_summary,
        cluster_selected,
        audit,
    )
    manifest = {
        "experiment": "P27 no-sector native factor re-mining",
        "alpha_residual_model": "no-sector OLS720 residualizer",
        "portfolio_risk_model": "P15c legacy risk beta and strict-neutral allocator",
        "formula_count": len(formulas),
        "formula_seed": FORMULA_SEED,
        "formula_validation_count": FORMULA_VALIDATION_COUNT,
        "smoothing_grid": list(SMOOTHING_GRID),
        "execution_grid": [spec.name for spec in native_execution_specs()],
        "outer_folds": int(len(folds)),
        "workers": workers,
        "selection": "train label screen -> validation formula/smoothing -> validation execution and cluster -> next test only",
        "factor_oos_diagnostics_are_selection_free": True,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Done: {output_dir}", flush=True)


def build_native_panel(prepared_path: Path, model_path: Path, raw_path: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    panel = pd.read_pickle(prepared_path).sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    raw = pd.read_csv(
        raw_path,
        usecols=["timestamp", "symbol", "quote_volume", "log_quote_volume"],
        parse_dates=["timestamp"],
    ).sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    label_cols = [
        "timestamp",
        "symbol",
        *[f"label_resid_rank_centered_h{h}" for h in [48, 72, 96, 168]],
        *[f"label_path_rank_centered_h{h}" for h in [48, 72, 96, 168]],
        "label_slow_low_turnover_rank_centered",
        "label_path_slow_low_turnover_rank_centered",
    ]
    labels = pd.read_csv(model_path, usecols=label_cols, parse_dates=["timestamp"]).sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    keys = ["timestamp", "symbol"]
    if len(panel) != len(raw) or not panel[keys].equals(raw[keys]):
        raise ValueError("P26 prepared panel and raw quote-volume panel are not aligned")
    if len(panel) != len(labels) or not panel[keys].equals(labels[keys]):
        raise ValueError("P26 prepared panel and new label panel are not aligned")
    panel["quote_volume"] = raw["quote_volume"].to_numpy(dtype=float)
    panel["log_quote_volume"] = raw["log_quote_volume"].to_numpy(dtype=float)
    for column in label_cols[2:]:
        panel[column] = labels[column].to_numpy(dtype=float)

    n = len(panel)
    work: dict[str, np.ndarray] = {}
    for name in native_raw_feature_names():
        work[name] = np.full(n, np.nan, dtype=np.float64)

    residual = panel["residual_log_ret"].astype(float)
    cross_sum = residual.groupby(panel["timestamp"], sort=False).transform("sum")
    cross_count = residual.groupby(panel["timestamp"], sort=False).transform("count")
    panel["_common_residual_ex_self"] = (cross_sum - residual) / (cross_count - 1.0).replace(0.0, np.nan)
    dispersion = residual.groupby(panel["timestamp"], sort=False).std()
    dispersion_z = rolling_z(dispersion.shift(1), 720, 168)
    panel["_market_calm_quality"] = panel["timestamp"].map(1.0 / (1.0 + np.exp(dispersion_z.clip(-8.0, 8.0))))

    for _, indexes in panel.groupby("symbol", sort=False).groups.items():
        idx = np.asarray(indexes, dtype=int)
        rows = panel.loc[idx]
        r = rows["residual_log_ret"].astype(float).reset_index(drop=True)
        e = r.fillna(0.0).cumsum()
        abs_r = r.abs()
        rv336 = r.rolling(336, min_periods=168).std() * np.sqrt(336.0)

        for total, skip in [(72, 12), (168, 24), (336, 24), (504, 48)]:
            span = total - skip
            mom = r.shift(skip).rolling(span, min_periods=max(48, span // 2)).sum()
            scale = r.shift(skip).rolling(max(span, 168), min_periods=max(72, span // 2)).std() * np.sqrt(span)
            work[f"mom_{total}_{skip}"][idx] = (mom / scale.replace(0.0, np.nan)).to_numpy()

        for window in [168, 336]:
            shifted = r.shift(24)
            num = shifted.rolling(window - 24, min_periods=(window - 24) // 2).sum()
            den = shifted.abs().rolling(window - 24, min_periods=(window - 24) // 2).sum()
            work[f"trend_eff_{window}"][idx] = (num / den.replace(0.0, np.nan)).to_numpy()
            work[f"sign_persist_{window}"][idx] = shifted.apply(np.sign).rolling(
                window - 24, min_periods=(window - 24) // 2
            ).mean().to_numpy()

        for window in [168, 336, 504]:
            lo = e.rolling(window, min_periods=window // 2).min()
            hi = e.rolling(window, min_periods=window // 2).max()
            work[f"highloc_{window}"][idx] = (2.0 * (e - lo) / (hi - lo).replace(0.0, np.nan) - 1.0).to_numpy()
            anchor = e - e.ewm(span=window, adjust=False, min_periods=window // 2).mean()
            anchor_scale = anchor.rolling(window, min_periods=window // 2).std()
            work[f"reversion_{window}"][idx] = (-anchor / anchor_scale.replace(0.0, np.nan)).to_numpy()

        for window in [168, 336]:
            lag = e.shift(1)
            delta = e.diff()
            slope = lag.rolling(window, min_periods=window // 2).cov(delta) / lag.rolling(
                window, min_periods=window // 2
            ).var().replace(0.0, np.nan)
            speed = (-slope).clip(lower=0.0)
            halflife = np.log(2.0) / speed.replace(0.0, np.nan)
            upper = 240.0 if window == 168 else 336.0
            gate = np.exp(-np.abs(np.log(halflife / np.sqrt(48.0 * upper))))
            gate = gate.where((slope < 0.0) & (halflife >= 48.0) & (halflife <= upper), 0.0)
            work[f"ou_speed_{window}"][idx] = speed.to_numpy()
            work[f"stationarity_{window}"][idx] = gate.to_numpy()

        common = rows["_common_residual_ex_self"].astype(float).reset_index(drop=True)
        x = common.shift(1)
        y = r.shift(1)
        for window in [168, 336]:
            cov = x.rolling(window, min_periods=window // 2).cov(y)
            xvar = x.rolling(window, min_periods=window // 2).var()
            beta = cov / xvar.replace(0.0, np.nan)
            corr = x.rolling(window, min_periods=window // 2).corr(y)
            adjusted = r - beta * common
            adj_mom = adjusted.shift(24).rolling(window - 24, min_periods=(window - 24) // 2).sum()
            adj_scale = adjusted.shift(24).rolling(window, min_periods=window // 2).std() * np.sqrt(window - 24)
            work[f"common_corr_{window}"][idx] = corr.to_numpy()
            work[f"common_purity_{window}"][idx] = (1.0 - corr.abs()).to_numpy()
            work[f"common_adj_mom_{window}"][idx] = (adj_mom / adj_scale.replace(0.0, np.nan)).to_numpy()

        short_rv = r.rolling(72, min_periods=36).std()
        long_rv = r.rolling(336, min_periods=168).std()
        work["rv_ratio_72_336"][idx] = (short_rv / long_rv.replace(0.0, np.nan)).to_numpy()
        work["vol_of_vol_168"][idx] = r.rolling(24, min_periods=12).std().rolling(168, min_periods=84).std().to_numpy()
        downside = r.clip(upper=0.0).pow(2).rolling(168, min_periods=84).mean()
        total_var = r.pow(2).rolling(168, min_periods=84).mean()
        work["downside_share_168"][idx] = (downside / total_var.replace(0.0, np.nan)).to_numpy()

        beta_frame = rows[["alpha_beta_btc", "alpha_beta_eth", "alpha_beta_market"]].astype(float).reset_index(drop=True)
        for window in [168, 336]:
            beta_std = beta_frame.shift(1).rolling(window, min_periods=window // 2).std().mean(axis=1)
            work[f"beta_instability_{window}"][idx] = beta_std.to_numpy()

        qv = rows["quote_volume"].astype(float).clip(lower=1.0).reset_index(drop=True)
        share = qv / panel.groupby("timestamp", sort=False)["quote_volume"].transform("sum").loc[idx].reset_index(drop=True)
        share_fast = share.rolling(168, min_periods=84).mean()
        share_slow = share.rolling(720, min_periods=336).mean()
        work["volume_share_trend"][idx] = np.log((share_fast + 1e-12) / (share_slow + 1e-12)).to_numpy()
        amihud = rows["log_ret"].astype(float).abs().reset_index(drop=True) / qv
        work["amihud_168"][idx] = amihud.rolling(168, min_periods=84).mean().to_numpy()
        work["liquidity_168"][idx] = np.log(qv).rolling(168, min_periods=84).mean().to_numpy()

    for name, values in work.items():
        panel[f"raw_{name}"] = values

    directional = directional_raw_features()
    quality = quality_raw_features()
    for name, source, sign in directional:
        panel[name] = centered_rank(panel, panel[source] * sign)
    for name, source, sign in quality:
        panel[name] = percentile_rank(panel, panel[source] * sign)
    panel["nf_q_market_calm"] = panel["_market_calm_quality"].clip(0.0, 1.0)
    panel["nf_q_trend_quality_168"] = percentile_rank(panel, panel["raw_trend_eff_168"].abs())
    panel["nf_q_trend_quality_336"] = percentile_rank(panel, panel["raw_trend_eff_336"].abs())

    keep = [
        column
        for column in panel.columns
        if not column.startswith("raw_") and column not in {"_common_residual_ex_self", "_market_calm_quality", "quote_volume", "log_quote_volume"}
    ]
    panel = panel[keep].replace([np.inf, -np.inf], np.nan).sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    primitives = describe_primitives(panel)
    coverage = panel[[row["primitive"] for row in primitives]].notna().mean()
    for row in primitives:
        row["coverage"] = float(coverage[row["primitive"]])
    return panel, primitives


def native_raw_feature_names() -> list[str]:
    return [
        "mom_72_12", "mom_168_24", "mom_336_24", "mom_504_48",
        "trend_eff_168", "trend_eff_336", "sign_persist_168", "sign_persist_336",
        "highloc_168", "highloc_336", "highloc_504",
        "reversion_168", "reversion_336", "reversion_504",
        "ou_speed_168", "ou_speed_336", "stationarity_168", "stationarity_336",
        "common_corr_168", "common_corr_336", "common_purity_168", "common_purity_336",
        "common_adj_mom_168", "common_adj_mom_336",
        "rv_ratio_72_336", "vol_of_vol_168", "downside_share_168",
        "beta_instability_168", "beta_instability_336", "volume_share_trend", "amihud_168", "liquidity_168",
    ]


def directional_raw_features() -> list[tuple[str, str, float]]:
    return [
        ("nf_d_mom_72_12", "raw_mom_72_12", 1.0),
        ("nf_d_mom_168_24", "raw_mom_168_24", 1.0),
        ("nf_d_mom_336_24", "raw_mom_336_24", 1.0),
        ("nf_d_mom_504_48", "raw_mom_504_48", 1.0),
        ("nf_d_trend_eff_168", "raw_trend_eff_168", 1.0),
        ("nf_d_trend_eff_336", "raw_trend_eff_336", 1.0),
        ("nf_d_sign_persist_168", "raw_sign_persist_168", 1.0),
        ("nf_d_sign_persist_336", "raw_sign_persist_336", 1.0),
        ("nf_d_highloc_168", "raw_highloc_168", 1.0),
        ("nf_d_highloc_336", "raw_highloc_336", 1.0),
        ("nf_d_highloc_504", "raw_highloc_504", 1.0),
        ("nf_d_reversion_168", "raw_reversion_168", 1.0),
        ("nf_d_reversion_336", "raw_reversion_336", 1.0),
        ("nf_d_reversion_504", "raw_reversion_504", 1.0),
        ("nf_d_common_adj_mom_168", "raw_common_adj_mom_168", 1.0),
        ("nf_d_common_adj_mom_336", "raw_common_adj_mom_336", 1.0),
        ("nf_d_volume_share_trend", "raw_volume_share_trend", 1.0),
        ("nf_d_low_downside_share", "raw_downside_share_168", -1.0),
    ]


def quality_raw_features() -> list[tuple[str, str, float]]:
    return [
        ("nf_q_stationarity_168", "raw_stationarity_168", 1.0),
        ("nf_q_stationarity_336", "raw_stationarity_336", 1.0),
        ("nf_q_purity_168", "raw_common_purity_168", 1.0),
        ("nf_q_purity_336", "raw_common_purity_336", 1.0),
        ("nf_q_beta_stability_168", "raw_beta_instability_168", -1.0),
        ("nf_q_beta_stability_336", "raw_beta_instability_336", -1.0),
        ("nf_q_liquidity", "raw_liquidity_168", 1.0),
        ("nf_q_low_amihud", "raw_amihud_168", -1.0),
        ("nf_q_low_vov", "raw_vol_of_vol_168", -1.0),
        ("nf_q_low_rv_ratio", "raw_rv_ratio_72_336", -1.0),
    ]


def describe_primitives(panel: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    descriptions = {
        "mom": "skip-recent normalized residual momentum",
        "trend_eff": "residual directional efficiency",
        "sign_persist": "residual sign persistence",
        "highloc": "location in trailing residual range",
        "reversion": "negative deviation from slow residual anchor",
        "common_adj": "momentum after ex-self residual-common-factor adjustment",
        "volume": "slow volume-share trend",
        "downside": "low residual downside semivariance share",
        "stationarity": "OU half-life quality gate",
        "purity": "low loading to ex-self residual common factor",
        "beta_stability": "stable no-sector alpha beta",
        "liquidity": "high trailing quote liquidity",
        "amihud": "low Amihud illiquidity",
        "vov": "low residual volatility-of-volatility",
        "rv_ratio": "low short/long residual volatility ratio",
        "market_calm": "low market-wide residual dispersion state",
    }
    for column in panel.columns:
        if not column.startswith("nf_"):
            continue
        key = next((token for token in descriptions if token in column), "native residual primitive")
        rows.append(
            {
                "primitive": column,
                "kind": "directional" if column.startswith("nf_d_") else "quality_gate",
                "description": descriptions.get(key, "native no-sector residual primitive"),
                "coverage": float(panel[column].notna().mean()),
                "minimum_lookback_hours": infer_minimum_lookback(column),
                "past_only": True,
            }
        )
    return rows


def infer_minimum_lookback(name: str) -> int:
    for value in [504, 336, 168, 72, 48]:
        if str(value) in name:
            return value
    return 168


def rolling_z(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std()
    return (series - mean) / std.replace(0.0, np.nan)


def centered_rank(panel: pd.DataFrame, values: pd.Series) -> pd.Series:
    return 2.0 * values.groupby(panel["timestamp"], sort=False).rank(pct=True, method="average") - 1.0


def percentile_rank(panel: pd.DataFrame, values: pd.Series) -> pd.Series:
    return values.groupby(panel["timestamp"], sort=False).rank(pct=True, method="average")


def build_formula_universe(primitive_rows: list[dict[str, Any]], count: int, seed: int) -> list[NativeFormula]:
    directional = sorted(row["primitive"] for row in primitive_rows if row["kind"] == "directional")
    quality = sorted(row["primitive"] for row in primitive_rows if row["kind"] == "quality_gate")
    formulas: list[NativeFormula] = []
    seen: set[str] = set()

    def add(op: str, left: str, right: str | None = None, gate: str | None = None, weight: float = 1.0) -> None:
        if op == "single":
            expression = left
            primitives = (left,)
            complexity = 1
        elif op == "neg":
            expression = f"NEG({left})"
            primitives = (left,)
            complexity = 2
        elif op in {"add", "sub"}:
            symbol = "ADD" if op == "add" else "SUB"
            expression = f"{symbol}({left},{weight:.2f}*{right})"
            primitives = (left, str(right))
            complexity = 4
        elif op == "gate":
            expression = f"GATE({left},{gate})"
            primitives = (left, str(gate))
            complexity = 4
        else:
            expression = f"GATED_ADD({left},{weight:.2f}*{right},{gate})"
            primitives = (left, str(right), str(gate))
            complexity = 7
        if expression in seen or len(formulas) >= count:
            return
        seen.add(expression)
        formulas.append(
            NativeFormula(
                name=f"NSF{len(formulas)+1:03d}",
                op=op,
                left=left,
                right=right,
                gate=gate,
                weight=float(weight),
                complexity=complexity,
                expression=expression,
                primitives=primitives,
            )
        )

    for primitive in directional:
        add("single", primitive)
        add("neg", primitive)

    economic_seeds = [
        ("gate", "nf_d_mom_168_24", None, "nf_q_purity_168", 1.0),
        ("gate", "nf_d_mom_336_24", None, "nf_q_beta_stability_336", 1.0),
        ("gate", "nf_d_common_adj_mom_168", None, "nf_q_liquidity", 1.0),
        ("gate", "nf_d_reversion_168", None, "nf_q_stationarity_168", 1.0),
        ("gate", "nf_d_reversion_336", None, "nf_q_stationarity_336", 1.0),
        ("gated_add", "nf_d_mom_168_24", "nf_d_highloc_168", "nf_q_purity_168", 0.5),
        ("gated_add", "nf_d_mom_336_24", "nf_d_trend_eff_336", "nf_q_beta_stability_336", 0.5),
        ("gated_add", "nf_d_reversion_336", "nf_d_reversion_504", "nf_q_stationarity_336", 0.5),
        ("add", "nf_d_common_adj_mom_168", "nf_d_volume_share_trend", None, 0.3),
        ("sub", "nf_d_mom_168_24", "nf_d_reversion_168", None, 0.5),
    ]
    for op, left, right, gate, weight in economic_seeds:
        add(op, left, right, gate, weight)

    rng = np.random.default_rng(seed)
    while len(formulas) < count:
        op = str(rng.choice(["add", "sub", "gate", "gated_add"], p=[0.28, 0.22, 0.30, 0.20]))
        left = str(rng.choice(directional))
        right = str(rng.choice(directional)) if op in {"add", "sub", "gated_add"} else None
        gate = str(rng.choice(quality)) if op in {"gate", "gated_add"} else None
        weight = float(rng.choice([0.25, 0.50, 0.75]))
        add(op, left, right, gate, weight)
    return formulas


def evaluate_formula(formula: NativeFormula, panel: pd.DataFrame) -> pd.Series:
    left = panel[formula.left].astype(float)
    if formula.op == "single":
        raw = left
    elif formula.op == "neg":
        raw = -left
    elif formula.op == "add":
        raw = left + formula.weight * panel[str(formula.right)].astype(float)
    elif formula.op == "sub":
        raw = left - formula.weight * panel[str(formula.right)].astype(float)
    elif formula.op == "gate":
        raw = left * (0.25 + 0.75 * panel[str(formula.gate)].astype(float).clip(0.0, 1.0))
    elif formula.op == "gated_add":
        combined = left + formula.weight * panel[str(formula.right)].astype(float)
        raw = combined * (0.25 + 0.75 * panel[str(formula.gate)].astype(float).clip(0.0, 1.0))
    else:
        raise ValueError(f"Unknown formula operation: {formula.op}")
    return centered_rank(panel, raw)


def formula_score(formula: NativeFormula, panel: pd.DataFrame, smoothing_alpha: float) -> pd.Series:
    score = evaluate_formula(formula, panel)
    if smoothing_alpha < 1.0:
        score = score.groupby(panel["symbol"], sort=False).transform(
            lambda values: values.ewm(alpha=smoothing_alpha, adjust=False, min_periods=1).mean()
        )
        score = centered_rank(panel, score)
    return score


def init_basic_worker(panel_path: str, config_path: str, formulas: dict[str, NativeFormula]) -> None:
    global _PANEL, _CFG, _FORMULAS
    _PANEL = pd.read_pickle(panel_path)
    _CFG = load_config(Path(config_path))
    _FORMULAS = formulas


def init_replay_worker(
    panel_path: str,
    config_path: str,
    formulas: dict[str, NativeFormula],
    selected: pd.DataFrame,
    residual_path: str,
) -> None:
    global _PANEL, _CFG, _FORMULAS, _RISK_RESIDUAL
    _PANEL = pd.read_pickle(panel_path)
    _CFG = load_config(Path(config_path))
    _FORMULAS = formulas
    _RISK_RESIDUAL = pd.read_pickle(residual_path)
    globals()["_SELECTED"] = selected


def factor_diagnostic_worker(fold: dict[str, Any]) -> pd.DataFrame:
    if _PANEL is None:
        raise RuntimeError("Factor worker is not initialized")
    fold_id = int(fold["fold"])
    start = pd.Timestamp(fold["test_start"])
    end = p4.decision_end_for_closed_positions(pd.Timestamp(fold["test_end"]), MAX_LABEL_HORIZON)
    local = _PANEL.loc[(_PANEL["timestamp"] >= start) & (_PANEL["timestamp"] < end)].copy()
    labels = [
        *[f"label_resid_rank_centered_h{h}" for h in [48, 72, 96, 168]],
        "label_slow_low_turnover_rank_centered",
        "label_path_slow_low_turnover_rank_centered",
    ]
    rows = []
    for primitive in sorted(column for column in local.columns if column.startswith("nf_d_")):
        row: dict[str, Any] = {
            "fold": fold_id,
            "primitive": primitive,
            "test_start": start,
            "test_decision_end": end,
            "observations": int(local[[primitive, labels[-2]]].dropna().shape[0]),
        }
        for label in labels:
            row[f"ic_{label}"] = mean_timestamp_ic(local, primitive, label)
        row["top_bottom_slow_label"] = mean_top_bottom(local, primitive, labels[-2], 4)
        row["signal_turnover_72h"] = signal_turnover(local, primitive, HORIZON)
        row["signal_autocorr_72h"] = signal_autocorr(local, primitive, HORIZON)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_factor_diagnostics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    target = "ic_label_slow_low_turnover_rank_centered"
    path = "ic_label_path_slow_low_turnover_rank_centered"
    rows = []
    for primitive, frame in fold_metrics.groupby("primitive"):
        mean_ic = float(frame[target].mean())
        direction = 1.0 if mean_ic >= 0.0 else -1.0
        oriented = direction * frame[target]
        row = {
            "primitive": primitive,
            "preferred_direction": "as_is" if direction > 0 else "negated",
            "mean_slow_rank_ic": mean_ic,
            "oriented_mean_slow_rank_ic": float(oriented.mean()),
            "oriented_ic_std": float(oriented.std(ddof=1)),
            "oriented_icir": float(oriented.mean() / oriented.std(ddof=1)) if oriented.std(ddof=1) > 0 else np.nan,
            "oriented_q25_fold_ic": float(oriented.quantile(0.25)),
            "oriented_positive_fold_rate": float((oriented > 0.0).mean()),
            "oriented_path_rank_ic": float(direction * frame[path].mean()),
            "oriented_top_bottom_slow_label": float(direction * frame["top_bottom_slow_label"].mean()),
            "mean_signal_turnover_72h": float(frame["signal_turnover_72h"].mean()),
            "mean_signal_autocorr_72h": float(frame["signal_autocorr_72h"].mean()),
            "folds": int(frame["fold"].nunique()),
        }
        for horizon in [48, 72, 96, 168]:
            row[f"oriented_rank_ic_h{horizon}"] = float(
                direction * frame[f"ic_label_resid_rank_centered_h{horizon}"].mean()
            )
        row["robust_pass"] = bool(
            row["oriented_mean_slow_rank_ic"] >= 0.015
            and row["oriented_q25_fold_ic"] >= -0.010
            and row["oriented_positive_fold_rate"] >= 0.55
            and row["oriented_path_rank_ic"] >= 0.0
            and row["mean_signal_autocorr_72h"] >= 0.50
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["robust_pass", "oriented_mean_slow_rank_ic", "oriented_q25_fold_ic"],
        ascending=[False, False, False],
    )


def formula_search_worker(fold: dict[str, Any]) -> dict[str, pd.DataFrame]:
    if _PANEL is None or _CFG is None:
        raise RuntimeError("Formula search worker is not initialized")
    fold_id = int(fold["fold"])
    validation_start = validation_split(fold)
    validation_end = p4.decision_end_for_closed_positions(pd.Timestamp(fold["train_end"]), HORIZON)
    train_label_end = validation_start - pd.Timedelta(hours=MAX_LABEL_HORIZON + 1)
    history_start = pd.Timestamp(fold["train_start"]) - pd.Timedelta(hours=504)
    local = _PANEL.loc[
        (_PANEL["timestamp"] >= history_start) & (_PANEL["timestamp"] < pd.Timestamp(fold["train_end"]))
    ].copy()
    train_mask = (local["timestamp"] >= pd.Timestamp(fold["train_start"])) & (local["timestamp"] < train_label_end)
    screening_rows = []
    for formula in _FORMULAS.values():
        local["_native_score"] = evaluate_formula(formula, local)
        ic_series = timestamp_ic_series(local.loc[train_mask], "_native_score", "label_slow_low_turnover_rank_centered")
        path_ic = mean_timestamp_ic(local.loc[train_mask], "_native_score", "label_path_slow_low_turnover_rank_centered")
        mean_ic = float(ic_series.mean()) if not ic_series.empty else np.nan
        monthly_key = ic_series.index.tz_localize(None).to_period("M") if not ic_series.empty else []
        monthly = ic_series.groupby(monthly_key).mean() if not ic_series.empty else pd.Series(dtype=float)
        q25_ic = float(monthly.quantile(0.25)) if not monthly.empty else np.nan
        turnover = signal_turnover(local.loc[train_mask], "_native_score", HORIZON)
        screen_score = nz(mean_ic) + 0.60 * nz(path_ic) + 0.40 * nz(q25_ic) - 0.02 * nz(turnover) - 0.001 * formula.complexity
        screening_rows.append(
            {
                "fold": fold_id,
                **formula.to_row(),
                "train_terminal_ic": mean_ic,
                "train_path_ic": path_ic,
                "train_q25_month_ic": q25_ic,
                "train_signal_turnover_72h": turnover,
                "train_screen_score": screen_score,
                "train_hard_pass": bool(
                    np.isfinite(mean_ic) and mean_ic >= 0.0 and nz(path_ic) >= -0.01 and nz(q25_ic) >= -0.03
                ),
            }
        )
    screening = pd.DataFrame(screening_rows)
    preferred = screening.loc[screening["train_hard_pass"]].sort_values("train_screen_score", ascending=False)
    if len(preferred) < FORMULA_VALIDATION_COUNT:
        preferred = screening.sort_values("train_screen_score", ascending=False)
    advanced = set(preferred.head(FORMULA_VALIDATION_COUNT)["formula_name"])
    screening["advanced_to_validation"] = screening["formula_name"].isin(advanced)

    pf_cfg = make_portfolio_config(_CFG, HORIZON)
    fast_cfg = replace(pf_cfg, rebalance_hours=1)
    fixed_spec = fixed_native_spec()
    risk_template = make_risk_template(_CFG)
    risk_cfg = p15c.make_fold_risk(risk_template, fixed_spec, f"p27_formula_{fold_id}")
    trade = rebalance_view(local, validation_start, validation_end, HORIZON)
    validation_rows = []
    screen_by_name = screening.set_index("formula_name")
    for formula_name in sorted(advanced):
        formula = _FORMULAS[formula_name]
        for alpha in SMOOTHING_GRID:
            score = formula_score(formula, local, alpha)
            local["_native_score"] = score
            trade["_native_score"] = local.loc[trade.index, "_native_score"].to_numpy()
            prepare_direct_score(trade, "_native_score")
            equity, positions, metrics = p3d.run_p4_gated_rerank_backtest(
                trade,
                validation_start,
                validation_end,
                fast_cfg,
                risk_cfg,
                fixed_spec,
                p3d.SCORE_COL,
                f"P27({formula.name},alpha={alpha})",
            )
            metrics = p3d.aggregate_oos_metrics(equity, HORIZON)
            enrich_metrics(equity, positions, metrics, formula.complexity)
            validation_ic = mean_timestamp_ic(
                local.loc[(local["timestamp"] >= validation_start) & (local["timestamp"] < validation_end)],
                "_native_score",
                "label_slow_low_turnover_rank_centered",
            )
            screen = screen_by_name.loc[formula_name]
            fitness = validation_fitness(metrics, validation_ic, float(screen["train_terminal_ic"]), float(screen["train_path_ic"]), formula.complexity)
            validation_rows.append(
                {
                    "fold": fold_id,
                    **formula.to_row(),
                    "smoothing_alpha": alpha,
                    "validation_start": validation_start,
                    "validation_decision_end": validation_end,
                    "train_terminal_ic": screen["train_terminal_ic"],
                    "train_path_ic": screen["train_path_ic"],
                    "train_q25_month_ic": screen["train_q25_month_ic"],
                    "train_hard_pass": screen["train_hard_pass"],
                    "validation_slow_rank_ic": validation_ic,
                    "validation_fitness": fitness,
                    **{f"validation_{key}": value for key, value in metrics.items() if p3d.is_scalar(value)},
                }
            )
    validation = pd.DataFrame(validation_rows)
    valid = validation.loc[
        validation["train_hard_pass"].astype(bool)
        & validation["validation_slow_rank_ic"].fillna(-1.0).ge(-0.01)
        & validation["validation_periods"].fillna(0).ge(8)
    ]
    forced = valid.empty
    winner = (validation if forced else valid).sort_values(
        ["validation_fitness", "validation_total_return_8bps", "validation_slow_rank_ic"], ascending=False
    ).iloc[0]

    local["_native_score"] = formula_score(_FORMULAS[str(winner["formula_name"])], local, float(winner["smoothing_alpha"]))
    trade["_native_score"] = local.loc[trade.index, "_native_score"].to_numpy()
    prepare_direct_score(trade, "_native_score")
    execution_rows = []
    for spec in native_execution_specs():
        exec_risk = p15c.make_fold_risk(risk_template, spec, f"p27_exec_{fold_id}_{spec.name}")
        equity, positions, metrics = p3d.run_p4_gated_rerank_backtest(
            trade,
            validation_start,
            validation_end,
            fast_cfg,
            exec_risk,
            spec,
            p3d.SCORE_COL,
            f"P27({winner['formula_name']},alpha={winner['smoothing_alpha']},{spec.name})",
        )
        metrics = p3d.aggregate_oos_metrics(equity, HORIZON)
        enrich_metrics(equity, positions, metrics, int(winner["complexity"]))
        execution_rows.append(
            {
                "fold": fold_id,
                "execution_spec": spec.name,
                "long_fraction": spec.long_fraction,
                "short_guard": spec.short_guard,
                "short_resid_confirm_rank": spec.short_resid_confirm_rank,
                "validation_execution_fitness": execution_fitness(metrics),
                **{f"validation_{key}": value for key, value in metrics.items() if p3d.is_scalar(value)},
            }
        )
    execution = pd.DataFrame(execution_rows)
    reference = execution.loc[execution["execution_spec"].eq(fixed_spec.name)].iloc[0]
    execution["hard_accept"] = (
        execution["validation_total_return_8bps"].ge(float(reference["validation_total_return_8bps"]) - 0.002)
        & execution["validation_gross_total_return_before_cost"].ge(float(reference["validation_gross_total_return_before_cost"]) - 0.002)
        & execution["validation_max_drawdown_8bps"].ge(1.20 * float(reference["validation_max_drawdown_8bps"]))
        & execution["validation_mean_turnover"].le(1.15 * float(reference["validation_mean_turnover"]))
    )
    accepted = execution.loc[execution["hard_accept"]]
    exec_winner = (execution.loc[execution["execution_spec"].eq(fixed_spec.name)] if accepted.empty else accepted).sort_values(
        ["validation_execution_fitness", "validation_total_return_8bps"], ascending=False
    ).iloc[0]
    selected = pd.DataFrame(
        [{
            "fold": fold_id,
            "formula_name": winner["formula_name"],
            "formula": winner["formula"],
            "primitives": winner["primitives"],
            "complexity": int(winner["complexity"]),
            "smoothing_alpha": float(winner["smoothing_alpha"]),
            "execution_spec": exec_winner["execution_spec"],
            "validation_start": validation_start,
            "validation_decision_end": validation_end,
            "train_label_end": train_label_end,
            "forced_formula_selection": forced,
            "validation_formula_fitness": float(winner["validation_fitness"]),
            "validation_execution_fitness": float(exec_winner["validation_execution_fitness"]),
            "validation_slow_rank_ic": float(winner["validation_slow_rank_ic"]),
            "train_terminal_ic": float(winner["train_terminal_ic"]),
            "train_path_ic": float(winner["train_path_ic"]),
        }]
    )
    return {"screening": screening, "validation": validation, "execution": execution, "selected": selected}


def prepare_direct_score(panel: pd.DataFrame, score_col: str) -> None:
    panel[p3d.P4_SCORE_COL] = panel[score_col].astype(float)
    panel[p3d.CANDIDATE_RANK_COL] = panel[score_col].astype(float)
    panel[p3d.SCORE_COL] = panel[score_col].astype(float)
    panel[p3d.EFFECTIVE_CANDIDATE_WEIGHT_COL] = 1.0


def fixed_native_spec() -> CommitteeSpec:
    return CommitteeSpec("native_lf50_ema35", 0.0, 0.0, 1.0, 0.50, "ema35", 0.35)


def native_execution_specs() -> list[CommitteeSpec]:
    return [
        CommitteeSpec(f"native_lf{int(lf*100)}_{guard}", 0.0, 0.0, 1.0, lf, guard, confirm)
        for lf in [0.45, 0.50, 0.55]
        for guard, confirm in [("ema35", 0.35), ("ema30", 0.30)]
    ]


def enrich_metrics(equity: pd.DataFrame, positions: pd.DataFrame, metrics: dict[str, float], complexity: int) -> None:
    add_gross_metrics(equity, metrics)
    add_risk_event_metrics(positions, metrics)
    add_delay_metrics(equity, metrics)
    metrics.update(p3d.cost_adjusted_metrics(equity, HORIZON, [8.0, 20.0, 30.0]))
    metrics["q25_segment_sharpe"] = p3d.q25_segment_sharpe(equity, HORIZON)
    metrics["long_short_balance"] = p3d.long_short_balance(positions)
    metrics["short_abs_contribution_share"] = p3d.short_abs_contribution_share(positions)
    metrics["complexity"] = float(complexity)


def validation_fitness(metrics: dict[str, float], validation_ic: float, train_ic: float, train_path_ic: float, complexity: int) -> float:
    if float(metrics.get("periods", 0.0) or 0.0) < 8:
        return -1000.0
    return (
        120.0 * nz(metrics.get("total_return_8bps", 0.0))
        + 45.0 * nz(metrics.get("gross_total_return_before_cost", 0.0))
        + 2.0 * nz(metrics.get("sharpe_8bps", 0.0))
        - 35.0 * abs(nz(metrics.get("max_drawdown_8bps", 0.0)))
        - 4.0 * nz(metrics.get("mean_turnover", 0.0))
        + 10.0 * nz(validation_ic)
        + 8.0 * nz(train_ic)
        + 5.0 * nz(train_path_ic)
        + 1.0 * nz(metrics.get("q25_segment_sharpe", 0.0))
        - 0.10 * complexity
    )


def execution_fitness(metrics: dict[str, float]) -> float:
    return (
        120.0 * nz(metrics.get("total_return_8bps", 0.0))
        + 40.0 * nz(metrics.get("gross_total_return_before_cost", 0.0))
        + 2.0 * nz(metrics.get("sharpe_8bps", 0.0))
        - 40.0 * abs(nz(metrics.get("max_drawdown_8bps", 0.0)))
        - 4.0 * nz(metrics.get("mean_turnover", 0.0))
        + 2.0 * nz(metrics.get("long_short_balance", 0.0))
    )


def p15c_replay_worker(fold: dict[str, Any]) -> dict[str, Any]:
    if _PANEL is None or _CFG is None or _RISK_RESIDUAL is None:
        raise RuntimeError("P15c replay worker is not initialized")
    selected_all = globals().get("_SELECTED")
    if selected_all is None:
        raise RuntimeError("Selected formula table is unavailable")
    fold_id = int(fold["fold"])
    selected_row = selected_all.loc[selected_all["fold"].eq(fold_id)].iloc[0]
    formula = _FORMULAS[str(selected_row["formula_name"])]
    spec_map = {spec.name: spec for spec in native_execution_specs()}
    execution_spec = spec_map[str(selected_row["execution_spec"])]
    history_start = pd.Timestamp(fold["train_start"]) - pd.Timedelta(hours=504)
    local = _PANEL.loc[
        (_PANEL["timestamp"] >= history_start) & (_PANEL["timestamp"] < pd.Timestamp(fold["test_end"]))
    ].copy()
    local["_native_score"] = formula_score(formula, local, float(selected_row["smoothing_alpha"]))
    prepare_direct_score(local, "_native_score")
    validation_start = pd.Timestamp(selected_row["validation_start"])
    validation_end = pd.Timestamp(selected_row["validation_decision_end"])
    test_end = p4.decision_end_for_closed_positions(pd.Timestamp(fold["test_end"]), HORIZON)
    validation_trade = rebalance_view(local, validation_start, validation_end, HORIZON)
    test_trade = rebalance_view(local, pd.Timestamp(fold["test_start"]), test_end, HORIZON)
    trade = pd.concat([validation_trade, test_trade]).drop_duplicates(["timestamp", "symbol"]).sort_values(["timestamp", "symbol"])
    pf_cfg = make_portfolio_config(_CFG, HORIZON)
    fast_cfg = replace(pf_cfg, rebalance_hours=1)
    risk_cfg = p15c.make_fold_risk(make_risk_template(_CFG), execution_spec, f"p27_p15c_{fold_id}")
    formula_text = f"P27({formula.name},alpha={selected_row['smoothing_alpha']},{execution_spec.name})"

    specs = p15c.build_extreme_specs()
    rows = []
    references = []
    cache: dict[tuple[int, int], dict[str, float]] = {}
    for cluster_spec in specs:
        key = (cluster_spec.lookback_hours, cluster_spec.cluster_k)
        if key not in cache:
            equity, positions, metrics = p15c.run_extreme_backtest(
                trade, _RISK_RESIDUAL, validation_start, validation_end, fast_cfg, risk_cfg, execution_spec,
                None, cluster_spec.lookback_hours, cluster_spec.cluster_k, p3d.SCORE_COL, formula_text,
            )
            metrics = p3d.aggregate_oos_metrics(equity, HORIZON)
            p15c.enrich_extreme_metrics(equity, positions, metrics, HORIZON)
            cache[key] = metrics
            references.append(
                {
                    "fold": fold_id,
                    "lookback_hours": cluster_spec.lookback_hours,
                    "cluster_k": cluster_spec.cluster_k,
                    "validation_start": validation_start,
                    "validation_decision_end": validation_end,
                    **{f"p9_validation_{key_name}": value for key_name, value in metrics.items() if p3d.is_scalar(value)},
                }
            )
        equity, positions, metrics = p15c.run_extreme_backtest(
            trade, _RISK_RESIDUAL, validation_start, validation_end, fast_cfg, risk_cfg, execution_spec,
            cluster_spec, cluster_spec.lookback_hours, cluster_spec.cluster_k, p3d.SCORE_COL, formula_text,
        )
        metrics = p3d.aggregate_oos_metrics(equity, HORIZON)
        p15c.enrich_extreme_metrics(equity, positions, metrics, HORIZON)
        rows.append(
            {
                "fold": fold_id,
                "extreme_spec": cluster_spec.name,
                **cluster_spec.__dict__,
                "validation_start": validation_start,
                "validation_decision_end": validation_end,
                **{f"validation_{key_name}": value for key_name, value in metrics.items() if p3d.is_scalar(value)},
            }
        )
    reference = pd.DataFrame(references)
    validation = p15c.annotate_validation(pd.DataFrame(rows), reference)
    selected_cluster = p15c.select_extreme_spec_by_fold(validation, reference, pd.DataFrame([fold]))
    cluster_row = selected_cluster.iloc[0]
    cluster_map = {item.name: item for item in specs}
    cluster_spec = None if bool(cluster_row["fallback_to_p9_sizing"]) else cluster_map[str(cluster_row["selected_extreme_spec"])]
    equity, positions, metrics = p15c.run_extreme_backtest(
        trade,
        _RISK_RESIDUAL,
        pd.Timestamp(fold["test_start"]),
        test_end,
        fast_cfg,
        risk_cfg,
        execution_spec,
        cluster_spec,
        int(cluster_row["lookback_hours"]),
        int(cluster_row["cluster_k"]),
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
            "formula_name": formula.name,
            "formula": formula.expression,
            "smoothing_alpha": float(selected_row["smoothing_alpha"]),
            "execution_spec": execution_spec.name,
            "selected_extreme_spec": p15c.NO_EXTREME_CAP if cluster_spec is None else cluster_spec.name,
            "fallback_to_p9_sizing": bool(cluster_spec is None),
            **metrics,
        }]
    )
    for frame in [equity, positions]:
        if frame.empty:
            continue
        frame["fold"] = fold_id
        frame["formula_name"] = formula.name
        frame["formula"] = formula.expression
        frame["smoothing_alpha"] = float(selected_row["smoothing_alpha"])
        frame["execution_spec"] = execution_spec.name
        frame["selected_extreme_spec"] = p15c.NO_EXTREME_CAP if cluster_spec is None else cluster_spec.name
    return {
        "validation": validation,
        "reference": reference,
        "selected": selected_cluster,
        "test": {"folds": fold_metrics, "equity": equity, "positions": positions},
    }


def timestamp_ic_series(frame: pd.DataFrame, score_col: str, label_col: str) -> pd.Series:
    valid = frame[["timestamp", score_col, label_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        return pd.Series(dtype=float)
    score = valid[score_col] - valid.groupby("timestamp")[score_col].transform("mean")
    label = valid[label_col] - valid.groupby("timestamp")[label_col].transform("mean")
    values = pd.DataFrame(
        {"timestamp": valid["timestamp"], "xy": score * label, "xx": score * score, "yy": label * label}
    ).groupby("timestamp")[["xy", "xx", "yy"]].sum()
    return values["xy"] / np.sqrt(values["xx"] * values["yy"]).replace(0.0, np.nan)


def mean_timestamp_ic(frame: pd.DataFrame, score_col: str, label_col: str) -> float:
    values = timestamp_ic_series(frame, score_col, label_col)
    return float(values.mean()) if values.notna().any() else np.nan


def mean_top_bottom(frame: pd.DataFrame, score_col: str, label_col: str, k: int) -> float:
    values = []
    for _, rows in frame[["timestamp", score_col, label_col]].dropna().groupby("timestamp", sort=False):
        if len(rows) < 2 * k:
            continue
        ordered = rows.sort_values(score_col)
        values.append(float(ordered[label_col].tail(k).mean() - ordered[label_col].head(k).mean()))
    return float(np.mean(values)) if values else np.nan


def rebalance_matrix(frame: pd.DataFrame, score_col: str, hours: int) -> pd.DataFrame:
    timestamps = np.array(sorted(frame["timestamp"].unique()))[::hours]
    return frame.loc[frame["timestamp"].isin(timestamps)].pivot(index="timestamp", columns="symbol", values=score_col).sort_index()


def signal_turnover(frame: pd.DataFrame, score_col: str, hours: int) -> float:
    matrix = rebalance_matrix(frame, score_col, hours)
    return float(matrix.diff().abs().mean(axis=1).mean()) if len(matrix) > 1 else np.nan


def signal_autocorr(frame: pd.DataFrame, score_col: str, hours: int) -> float:
    matrix = rebalance_matrix(frame, score_col, hours)
    if len(matrix) < 2:
        return np.nan
    left = matrix.iloc[:-1].to_numpy().ravel()
    right = matrix.iloc[1:].to_numpy().ravel()
    valid = np.isfinite(left) & np.isfinite(right)
    return float(np.corrcoef(left[valid], right[valid])[0, 1]) if valid.sum() > 20 else np.nan


def nz(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if np.isfinite(number) else 0.0


def build_trading_summary(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for name, result in results.items():
        metrics = result.get("metrics") or p4.summarize_nested_result(result, HORIZON)
        row = {"run": name, **metrics}
        folds = result["folds"]
        row["worst_5_fold_return_sum"] = float(folds["total_return"].nsmallest(5).sum())
        row["median_fold_return"] = float(folds["total_return"].median())
        row["q25_fold_return"] = float(folds["total_return"].quantile(0.25))
        rows.append(row)
    return pd.DataFrame(rows)


def build_selected_formula_oos_diagnostics(
    panel_path: Path,
    folds: pd.DataFrame,
    selected: pd.DataFrame,
    formulas: dict[str, NativeFormula],
) -> pd.DataFrame:
    panel = pd.read_pickle(panel_path)
    rows = []
    selected_by_fold = selected.set_index("fold")
    for _, fold in folds.sort_values("fold").iterrows():
        fold_id = int(fold["fold"])
        choice = selected_by_fold.loc[fold_id]
        formula = formulas[str(choice["formula_name"])]
        history_start = pd.Timestamp(fold["train_start"]) - pd.Timedelta(hours=504)
        decision_end = p4.decision_end_for_closed_positions(pd.Timestamp(fold["test_end"]), MAX_LABEL_HORIZON)
        local = panel.loc[
            (panel["timestamp"] >= history_start) & (panel["timestamp"] < pd.Timestamp(fold["test_end"]))
        ].copy()
        local["_selected_score"] = formula_score(formula, local, float(choice["smoothing_alpha"]))
        test = local.loc[(local["timestamp"] >= pd.Timestamp(fold["test_start"])) & (local["timestamp"] < decision_end)]
        row = {
            "fold": fold_id,
            "formula_name": formula.name,
            "formula": formula.expression,
            "smoothing_alpha": float(choice["smoothing_alpha"]),
            "test_start": fold["test_start"],
            "test_decision_end": decision_end,
            "test_slow_rank_ic": mean_timestamp_ic(test, "_selected_score", "label_slow_low_turnover_rank_centered"),
            "test_path_rank_ic": mean_timestamp_ic(test, "_selected_score", "label_path_slow_low_turnover_rank_centered"),
            "test_top_bottom_slow_label": mean_top_bottom(test, "_selected_score", "label_slow_low_turnover_rank_centered", 4),
            "test_signal_turnover_72h": signal_turnover(test, "_selected_score", HORIZON),
            "test_signal_autocorr_72h": signal_autocorr(test, "_selected_score", HORIZON),
        }
        for horizon in [48, 72, 96, 168]:
            row[f"test_rank_ic_h{horizon}"] = mean_timestamp_ic(
                test, "_selected_score", f"label_resid_rank_centered_h{horizon}"
            )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_selected_formula_oos(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame["test_slow_rank_ic"]
    std = float(values.std(ddof=1))
    return pd.DataFrame(
        [{
            "folds": int(frame["fold"].nunique()),
            "mean_test_slow_rank_ic": float(values.mean()),
            "test_slow_rank_ic_std": std,
            "test_slow_rank_icir": float(values.mean() / std) if std > 0 else np.nan,
            "q25_test_slow_rank_ic": float(values.quantile(0.25)),
            "positive_test_slow_ic_fold_rate": float((values > 0.0).mean()),
            "mean_test_path_rank_ic": float(frame["test_path_rank_ic"].mean()),
            "mean_test_top_bottom_slow_label": float(frame["test_top_bottom_slow_label"].mean()),
            "mean_test_signal_turnover_72h": float(frame["test_signal_turnover_72h"].mean()),
            "mean_test_signal_autocorr_72h": float(frame["test_signal_autocorr_72h"].mean()),
            **{f"mean_test_rank_ic_h{h}": float(frame[f"test_rank_ic_h{h}"].mean()) for h in [48, 72, 96, 168]},
        }]
    )


def build_fold_comparison(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    output = None
    for name, result in results.items():
        columns = ["fold", "total_return", "gross_total_return_before_cost", "sharpe", "max_drawdown", "mean_turnover"]
        frame = result["folds"][[column for column in columns if column in result["folds"]]].rename(
            columns={column: f"{name}_{column}" for column in columns if column != "fold"}
        )
        output = frame if output is None else output.merge(frame, on="fold", how="inner")
    if output is not None:
        output["P27_minus_P15c_return"] = output["P27_native_total_return"] - output["P15c_total_return"]
        output["P27_minus_P26_return"] = output["P27_native_total_return"] - output["P26_full_total_return"]
    return output if output is not None else pd.DataFrame()


def build_regime_comparison(results: dict[str, dict[str, Any]], costs: list[float]) -> pd.DataFrame:
    rows = []
    for name, result in results.items():
        equity = result["equity"].sort_values("timestamp").copy()
        equity["regime"] = equity["timestamp"].dt.tz_localize(None).dt.to_period("Q").astype(str)
        for cost in costs:
            values = equity["gross_period_return"] - equity["turnover"] * cost / 10_000.0
            for regime, returns in values.groupby(equity["regime"]):
                rows.append(
                    {
                        "run": name,
                        "regime": regime,
                        "one_way_cost_bps": cost,
                        "periods": int(len(returns)),
                        "total_return": float((1.0 + returns).prod() - 1.0),
                    }
                )
    return pd.DataFrame(rows)


def build_protocol_audit(
    folds: pd.DataFrame,
    selected: pd.DataFrame,
    cluster_selected: pd.DataFrame,
    result: dict[str, Any],
) -> pd.DataFrame:
    merged = folds[["fold", "train_end", "test_start", "test_end"]].merge(
        selected[["fold", "validation_decision_end", "train_label_end"]], on="fold", how="left"
    ).merge(cluster_selected[["fold", "selected_extreme_spec"]], on="fold", how="left")
    test_max = result["equity"].groupby("fold")["timestamp"].max().rename("max_test_decision_time")
    merged = merged.merge(test_max, on="fold", how="left")
    merged["labels_close_before_validation"] = (
        merged["train_label_end"] + pd.Timedelta(hours=MAX_LABEL_HORIZON + 1)
    ) <= selected.set_index("fold").loc[merged["fold"], "validation_start"].reset_index(drop=True)
    merged["formula_selection_before_test"] = merged["validation_decision_end"] < merged["test_start"]
    merged["cluster_selection_before_test"] = merged["train_end"] <= merged["test_start"]
    merged["test_decisions_inside_fold"] = merged["max_test_decision_time"] < merged["test_end"]
    merged["all_pass"] = merged[
        ["labels_close_before_validation", "formula_selection_before_test", "cluster_selection_before_test", "test_decisions_inside_fold"]
    ].all(axis=1)
    return merged


def plot_factor_diagnostics(output_dir: Path, summary: pd.DataFrame) -> None:
    view = summary.head(15).sort_values("oriented_mean_slow_rank_ic")
    fig, ax = plt.subplots(figsize=(11, 7))
    colors = np.where(view["robust_pass"], "#2a9d8f", "#8d99ae")
    ax.barh(view["primitive"].str.replace("nf_d_", "", regex=False), view["oriented_mean_slow_rank_ic"], color=colors)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_title("P27 no-sector native primitives: nested OOS slow-label Rank IC")
    ax.set_xlabel("Oriented mean Rank IC")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "factor_oos_rank_ic.png", dpi=180)
    plt.close(fig)


def plot_trading_curves(output_dir: Path, results: dict[str, dict[str, Any]]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0]})
    colors = {"P15c": "#777777", "P26_full": "#457b9d", "P27_native": "#d1495b"}
    for name, result in results.items():
        equity = result["equity"].sort_values("timestamp")
        returns = equity["gross_period_return"] - equity["turnover"] * 8.0 / 10_000.0
        curve = (1.0 + returns).cumprod() - 1.0
        axes[0].plot(equity["timestamp"], curve, label=f"{name} net 8bps", color=colors[name], linewidth=2.0 if name == "P27_native" else 1.4)
        if name == "P27_native":
            gross = (1.0 + equity["gross_period_return"]).cumprod() - 1.0
            axes[0].plot(equity["timestamp"], gross, label="P27_native gross", color="#2a9d8f", linewidth=1.4)
            wealth = 1.0 + curve
            drawdown = wealth / wealth.cummax() - 1.0
            axes[1].fill_between(equity["timestamp"], drawdown, 0.0, color="#d1495b", alpha=0.22)
            axes[1].plot(equity["timestamp"], drawdown, color="#d1495b", linewidth=1.2)
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_title("P27 native no-sector re-mining: matched nested OOS")
    axes[0].set_ylabel("Cumulative return")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].set_title("P27_native net 8bps drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "p27_matched_nested_oos_pnl.png", dpi=180)
    plt.close(fig)


def estimate_breakeven(cost: pd.DataFrame, run: str) -> float:
    frame = cost.loc[cost["experiment"].eq(run), ["one_way_cost_bps", "total_return"]].sort_values("one_way_cost_bps")
    positive = frame.loc[frame["total_return"] >= 0.0]
    negative = frame.loc[frame["total_return"] < 0.0]
    if positive.empty:
        return 0.0
    if negative.empty:
        return float(positive["one_way_cost_bps"].max())
    left = positive.iloc[-1]
    right_candidates = negative.loc[negative["one_way_cost_bps"] > left["one_way_cost_bps"]]
    if right_candidates.empty:
        return float(left["one_way_cost_bps"])
    right = right_candidates.iloc[0]
    fraction = float(left["total_return"] / (left["total_return"] - right["total_return"]))
    return float(left["one_way_cost_bps"] + fraction * (right["one_way_cost_bps"] - left["one_way_cost_bps"]))


def write_report(
    output_dir: Path,
    factors: pd.DataFrame,
    summary: pd.DataFrame,
    cost: pd.DataFrame,
    folds: pd.DataFrame,
    regime: pd.DataFrame,
    selected: pd.DataFrame,
    selected_formula_oos: pd.DataFrame,
    cluster_selected: pd.DataFrame,
    audit: pd.DataFrame,
) -> None:
    p27 = summary.loc[summary["run"].eq("P27_native")].iloc[0]
    p15 = summary.loc[summary["run"].eq("P15c")].iloc[0]
    p26 = summary.loc[summary["run"].eq("P26_full")].iloc[0]
    c8 = cost.loc[(cost["experiment"].eq("P27_native")) & (cost["one_way_cost_bps"].eq(8.0))].iloc[0]
    p15_8 = cost.loc[(cost["experiment"].eq("P15c")) & (cost["one_way_cost_bps"].eq(8.0))].iloc[0]
    robust = factors.loc[factors["robust_pass"]]
    top = factors.head(12)
    selected_oos = selected_formula_oos.iloc[0]
    fold_win_p15 = float((folds["P27_minus_P15c_return"] > 0.0).mean())
    fold_win_p26 = float((folds["P27_minus_P26_return"] > 0.0).mean())
    promote = bool(
        p27["gross_total_return_before_cost"] > p15["gross_total_return_before_cost"]
        and c8["total_return"] > p15_8["total_return"]
        and p27["sharpe"] > p15["sharpe"]
        and p27["max_drawdown"] >= 1.10 * p15["max_drawdown"]
        and p27["positive_fold_rate"] >= p15["positive_fold_rate"] - 0.05
    )
    if robust.empty:
        alpha_verdict = "没有发现同时通过均值 IC、q25 fold IC、path IC、正 fold 比例和慢速自相关门槛的原生 primitive。"
    else:
        alpha_verdict = f"发现 {len(robust)} 个通过统一稳健门槛的原生 primitive，说明 no-sector residual 并非完全没有可预测结构。"
    trade_verdict = "可升级为研究 baseline candidate" if promote else "不覆盖 P15c；保留为原生 re-mining 研究结果"
    lines = [
        "# P27 no-sector 原生因子重挖报告",
        "",
        "## 实验问题",
        "",
        "本实验只回答一个问题：`no-sector residual` 本身缺乏 alpha，还是旧 A/B/C 因子语言与它不匹配。",
        "",
        "- 因子层：从 no-sector residual 原生重建 48h 以上的 momentum、anchor reversion、OU stationarity、common-residual purity、beta stability、liquidity 与 slow quality gates。",
        "- 搜索层：固定 240 个符号公式；每个 outer fold 只用 train label 筛选、validation 选择公式/平滑/执行参数，test 只回放。",
        "- 交易层：P15c 的 portfolio risk beta、严格中性化、symbol cap、short confirmation 与 extreme-cluster allocator 保持不变。",
        "- test 中的因子 IC 仅用于最终诊断，绝不参与该 fold 的公式选择。",
        "",
        "## 总结结论",
        "",
        f"- 因子层：{alpha_verdict}",
        f"- 交易层：**{trade_verdict}**。",
        f"- P27 相对 P15c 的 fold 胜率为 {fold_win_p15:.1%}，相对 P26-full 为 {fold_win_p26:.1%}。",
        f"- 公式 forced selection 比例 {float(selected['forced_formula_selection'].mean()):.1%}；共选择 {selected['formula_name'].nunique()} 个公式。",
        f"- 实际被选公式的 test slow Rank IC 均值 {float(selected_oos['mean_test_slow_rank_ic']):.4f}，q25 fold {float(selected_oos['q25_test_slow_rank_ic']):.4f}，正 IC fold 比例 {float(selected_oos['positive_test_slow_ic_fold_rate']):.1%}。",
        f"- cluster allocator fallback 比例 {float(cluster_selected['fallback_to_p9_sizing'].mean()):.1%}。",
        "",
        "## 1. 因子层结果",
        "",
        "```text",
        top[[
            "primitive", "preferred_direction", "oriented_mean_slow_rank_ic", "oriented_q25_fold_ic",
            "oriented_positive_fold_rate", "oriented_path_rank_ic", "mean_signal_turnover_72h",
            "mean_signal_autocorr_72h", "robust_pass",
        ]].to_string(index=False),
        "```",
        "",
        "## 2. 最终交易层结果",
        "",
        f"- P27 gross：{float(p27['gross_total_return_before_cost']):.2%}；6bps net：{float(p27['total_return']):.2%}；8bps net：{float(c8['total_return']):.2%}。",
        f"- P27 Sharpe：{float(p27['sharpe']):.3f}；MaxDD：{float(p27['max_drawdown']):.2%}；positive fold rate：{float(p27['positive_fold_rate']):.1%}。",
        f"- P27 mean turnover：{float(p27['mean_turnover']):.4f}；估算单边 breakeven：{estimate_breakeven(cost, 'P27_native'):.2f}bps。",
        f"- P15c gross/net6/Sharpe：{float(p15['gross_total_return_before_cost']):.2%} / {float(p15['total_return']):.2%} / {float(p15['sharpe']):.3f}。",
        f"- P26-full gross/net6/Sharpe：{float(p26['gross_total_return_before_cost']):.2%} / {float(p26['total_return']):.2%} / {float(p26['sharpe']):.3f}。",
        "",
        "### Matched summary",
        "```text",
        summary[[
            "run", "gross_total_return_before_cost", "total_return", "sharpe", "max_drawdown",
            "mean_turnover", "positive_fold_rate", "worst_5_fold_return_sum", "median_fold_return",
        ]].to_string(index=False),
        "```",
        "",
        "### 成本敏感性",
        "```text",
        cost.to_string(index=False),
        "```",
        "",
        "### 季度 regime",
        "```text",
        regime.to_string(index=False),
        "```",
        "",
        "## 严格性审计",
        "",
        f"- {int(audit['all_pass'].sum())}/{len(audit)} 个 fold 通过 label closure、selection-before-test 与 test boundary 检查。",
        "- 原生特征只使用 t 时刻及以前数据；交易使用 t+1 执行口径。",
        "- factor test IC 是事后评价表，不参与选择；公式与 operator 集合在全实验前固定。",
        "",
        "```text",
        audit.to_string(index=False),
        "```",
        "",
        "## 公式选择频次",
        "```text",
        selected[["formula_name", "formula", "smoothing_alpha", "execution_spec"]].value_counts().to_string(),
        "```",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
