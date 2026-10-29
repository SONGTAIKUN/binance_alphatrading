#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
from crypto_stat_arb.portfolio import constrained_neutralize_weights, portfolio_exposures
from crypto_stat_arb.residual_model import add_portfolio_risk_aliases
from crypto_stat_arb.slow_labels import LOW_TURNOVER_WEIGHTS, SlowLabelSpec, build_slow_residual_labels
from run_p18_factor_decay_walkforward import centered_rank_matrix
from run_p18c_factor_selector_portfolio import (
    add_p18c_entry_fields,
    build_cost_sensitivity,
    load_existing_result,
    write_run_result,
)
from run_p18d_guarded_factor_selector import add_regime_features
from run_p20_selection_protocol import build_protocols
from run_p21_gross_alpha_family_search import (
    resolve_families,
    run_family_diagnostics,
    select_family_by_fold,
    summarize_families,
    summarize_selected,
)
from run_p22_p21_family_portfolio import replay_protocol
from run_p25_residual_model_bakeoff import (
    add_generic_features,
    build_fold_stability,
    build_generic_factor_library,
    build_generic_families,
    build_portfolio_summary,
    build_q2_q3_summary,
)
from run_p3d_p4_gated_rerank_alphagpt import (
    aggregate_oos_metrics,
    load_panel as load_p3d_panel,
    make_portfolio_config,
    make_risk_template,
)
from run_p4_nested_walkforward_selection import summarize_nested_result
from run_turnover_experiments import to_jsonable


EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "baseline_locked_v1"
DEFAULT_CANDIDATE_CONFIG = EXPERIMENT_ROOT / "configs" / "p26_no_sector_candidate.yaml"
DEFAULT_ALLOCATOR_CONFIG = EXPERIMENT_ROOT / "configs" / "baseline_locked_v1.yaml"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "outputs" / "p26_no_sector_candidate_baseline"
DEFAULT_P25_PRACTICAL = (
    EXPERIMENT_ROOT
    / "outputs"
    / "p25_residual_model_bakeoff"
    / "models"
    / "legacy_no_sector"
    / "p20_practical_q25_m5"
    / "nested_oos_run"
)
PROTOCOL = "p20_practical_q25_m5"
CORE_STAGE = "p26_no_sector_v2_core"
OVERLAY_STAGE = "p26_no_sector_v2_slow_label_overlay"
RISK_V2_STAGE = "p26_no_sector_risk_v2_reprojection"
ALPHA_MODEL_VERSION = "alpha_residual_v2_no_sector_ols720"
RISK_MODEL_VERSION = "portfolio_risk_v1_p15c_namespaced"
FACTOR_KEYS = ("btc", "eth", "market")
RAW_FACTOR_COLS = tuple(f"factor_{key}_ret" for key in FACTOR_KEYS)
HORIZONS = (48, 72, 96, 168)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the P26 namespaced no-sector research baseline candidate.")
    parser.add_argument("--candidate-config", default=str(DEFAULT_CANDIDATE_CONFIG))
    parser.add_argument("--allocator-config", default=str(DEFAULT_ALLOCATOR_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--force-rebuild-panel", action="store_true")
    parser.add_argument("--cost-bps", type=float, nargs="+", default=[0, 6, 8, 15, 20, 30])
    args = parser.parse_args()

    candidate_config_path = Path(args.candidate_config).resolve()
    allocator_config_path = Path(args.allocator_config).resolve()
    candidate_cfg = load_config(candidate_config_path)
    allocator_cfg = load_config(allocator_config_path)
    output_dir = ensure_dir(args.output_dir)
    panel_path = output_dir / "p26_model_and_label_panel.csv.gz"

    raw_asset_path = resolve_config_path(candidate_config_path, candidate_cfg["inputs"]["raw_asset_panel"])
    cached_alpha_path = resolve_config_path(candidate_config_path, candidate_cfg["inputs"]["cached_no_sector_panel"])
    manual_path = resolve_config_path(candidate_config_path, candidate_cfg["inputs"]["manual_feature_panel"])
    folds_path = resolve_config_path(candidate_config_path, candidate_cfg["inputs"]["folds"])
    p15c_run_path = resolve_config_path(candidate_config_path, candidate_cfg["inputs"]["p15c_run"])

    print("[1/8] Building namespaced no-sector alpha/risk panel", flush=True)
    if panel_path.exists() and not args.force_rebuild_panel:
        panel = pd.read_csv(panel_path, parse_dates=["timestamp"])
    else:
        panel = build_candidate_panel(raw_asset_path, cached_alpha_path)
        label_spec = SlowLabelSpec(
            horizons=HORIZONS,
            execution_delay_bars=int(candidate_cfg["labels"]["execution_delay_bars"]),
            volatility_lookback=int(candidate_cfg["labels"]["ex_ante_volatility_lookback_hours"]),
            path_lambda=float(candidate_cfg["labels"]["path_aware_lambda"]),
        )
        labels = build_slow_residual_labels(panel, label_spec)
        panel = merge_aligned(panel, labels, "model panel", "label panel")
        panel.to_csv(panel_path, index=False)

    print("[2/8] Running build-fail residual and label audit gates", flush=True)
    audit, diagnostics = run_audit(panel, candidate_cfg)
    audit.to_csv(output_dir / "audit_gates.csv", index=False)
    diagnostics.to_csv(output_dir / "audit_diagnostics.csv", index=False)
    write_audit_report(output_dir, audit, diagnostics)
    failed = audit[audit["status"].eq("fail")]
    if not failed.empty:
        raise RuntimeError(f"P26 audit gates failed: {failed['check'].tolist()}")

    print("[3/8] Building no-sector slow factor library", flush=True)
    research_panel = build_research_view(panel)
    factor_panel, factor_library = build_generic_factor_library(research_panel)
    standard_labels = research_panel[
        [
            "timestamp",
            "symbol",
            "label_slow_low_turnover",
            "label_path_slow_low_turnover",
            "label_horizon_sign_consistency",
            "label_path_horizon_sign_consistency",
        ]
    ]
    factor_panel = factor_panel.merge(
        standard_labels,
        on=["timestamp", "symbol"],
        how="left",
        validate="one_to_one",
    )
    factor_library.to_csv(output_dir / "factor_library.csv", index=False)
    folds = pd.read_csv(
        folds_path,
        parse_dates=["train_start", "train_end", "test_start", "test_end"],
    ).sort_values("fold")

    base_panel = load_p3d_panel(allocator_config_path, raw_asset_path, manual_path, 72)
    add_regime_features(base_panel)
    regime = (
        base_panel[["timestamp", "_p18d_high_risk_regime"]]
        .drop_duplicates("timestamp")
        .sort_values("timestamp")
        .set_index("timestamp")["_p18d_high_risk_regime"]
        .astype(bool)
    )

    print("[4/8] Running P20 practical fold-local family selection on the standard slow label", flush=True)
    label_rank = centered_rank_matrix(factor_panel, "label_slow_low_turnover", min_symbols=8)
    families = resolve_families(build_generic_families(), set(factor_library["factor"]))
    needed = sorted({component.factor for family in families for component in family.components})
    rank_matrices = {factor: centered_rank_matrix(factor_panel, factor, min_symbols=8) for factor in needed}
    fold_metrics, score_timeseries = run_family_diagnostics(
        families=families,
        folds=folds,
        rank_matrices=rank_matrices,
        label_rank=label_rank,
        regime=regime,
        horizon=72,
        validation_days=90,
        min_symbols=8,
    )
    fold_metrics["model"] = ALPHA_MODEL_VERSION
    fold_metrics["selection_label"] = "label_slow_low_turnover"
    fold_metrics.to_csv(output_dir / "family_fold_metrics.csv", index=False)
    score_timeseries.to_csv(output_dir / "family_score_timeseries.csv.gz", index=False)
    protocol = next(spec for spec in build_protocols() if spec.name == PROTOCOL)
    family_summary = summarize_families(fold_metrics, [protocol])
    family_summary.to_csv(output_dir / "family_summary.csv", index=False)
    selected = select_family_by_fold(fold_metrics, [protocol])
    selected.to_csv(output_dir / "selected_family_by_fold.csv", index=False)
    summarize_selected(selected).to_csv(output_dir / "selection_summary.csv", index=False)

    print("[5/8] Replaying the separated core, slow-label overlay, and risk-v2 diagnostic", flush=True)
    model_base = attach_candidate_model_to_allocator(base_panel, panel)
    add_p18c_entry_fields(model_base)
    p15c = load_existing_result(p15c_run_path)
    pf_cfg = make_portfolio_config(allocator_cfg, 72)
    risk_template = make_risk_template(allocator_cfg)
    replay, portfolio_selection = replay_protocol(
        protocol_name=OVERLAY_STAGE,
        protocol=PROTOCOL,
        base_panel=model_base,
        family_scores=score_timeseries,
        folds=folds,
        selected=selected,
        p15c=p15c,
        pf_cfg=pf_cfg,
        risk_template=risk_template,
        long_fraction=0.50,
        short_confirm_rank=0.35,
    )
    overlay = replay
    write_run_result(ensure_dir(output_dir / "nested_alpha_overlay_run"), overlay)
    candidate = relabel_existing_result(p15c, CORE_STAGE)
    write_run_result(ensure_dir(output_dir / "nested_candidate_run"), candidate)
    model_base_risk_v2 = attach_candidate_model_to_allocator(base_panel, panel, use_alpha_beta_as_risk=True)
    risk_v2_seed = build_all_fallback_seed(p15c)
    risk_v2 = replace_fallback_with_risk_v2(
        risk_v2_seed, p15c, model_base_risk_v2, pf_cfg, risk_template, stage=RISK_V2_STAGE
    )
    write_run_result(ensure_dir(output_dir / "nested_risk_v2_reprojection_run"), risk_v2)
    portfolio_selection.to_csv(output_dir / "portfolio_selection_by_fold.csv", index=False)

    print("[6/8] Building matched P15c/P25/P26 comparisons", flush=True)
    p25_practical = load_existing_result(DEFAULT_P25_PRACTICAL)
    results = {
        "p15c": p15c,
        "p25_no_sector_practical": p25_practical,
        CORE_STAGE: candidate,
        OVERLAY_STAGE: overlay,
        RISK_V2_STAGE: risk_v2,
    }
    comparison = build_portfolio_summary(results)
    comparison.to_csv(output_dir / "portfolio_comparison.csv", index=False)
    cost = build_cost_sensitivity(results, [float(value) for value in args.cost_bps])
    cost.to_csv(output_dir / "cost_sensitivity.csv", index=False)
    regime_summary = build_q2_q3_summary(results, [float(value) for value in args.cost_bps], 72)
    regime_summary.to_csv(output_dir / "regime_2024q2_q3_summary.csv", index=False)
    fold_stability = build_fold_stability(results)
    fold_stability.to_csv(output_dir / "fold_stability.csv", index=False)

    print("[7/8] Writing clean PnL chart and candidate manifest", flush=True)
    plot_candidate_pnl(output_dir, candidate)
    write_manifest(
        output_dir,
        candidate_config_path,
        allocator_config_path,
        raw_asset_path,
        panel_path,
        selected,
        candidate,
        overlay,
        risk_v2,
    )

    print("[8/8] Writing Chinese review", flush=True)
    write_report(output_dir, audit, comparison, cost, regime_summary, fold_stability, selected)
    print(f"Done: {output_dir}", flush=True)


def resolve_config_path(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def build_candidate_panel(raw_asset_path: Path, cached_alpha_path: Path) -> pd.DataFrame:
    raw_cols = [
        "timestamp",
        "symbol",
        "log_close",
        "log_ret",
        *RAW_FACTOR_COLS,
        "factor_sector_ret",
        "beta_intercept",
        "beta_btc",
        "beta_eth",
        "beta_market",
        "beta_sector",
    ]
    raw = pd.read_csv(raw_asset_path, usecols=raw_cols, parse_dates=["timestamp"])
    cached_cols = [
        "timestamp",
        "symbol",
        "residual_log_ret",
        "predicted_log_ret",
        "beta_btc",
        "beta_eth",
        "beta_market",
        "resid_z",
    ]
    cached = pd.read_csv(cached_alpha_path, usecols=cached_cols, parse_dates=["timestamp"])
    raw = raw.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    cached = cached.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    if len(raw) != len(cached):
        raise ValueError(f"Raw/cache row mismatch: {len(raw)} != {len(cached)}")
    if not raw[["timestamp", "symbol"]].equals(cached[["timestamp", "symbol"]]):
        raise ValueError("Raw/cache keys are not exactly aligned")

    out = raw.copy()
    for key, raw_factor in zip(FACTOR_KEYS, RAW_FACTOR_COLS, strict=True):
        out[f"alpha_factor_{key}_ret"] = out[raw_factor]
        out[f"risk_factor_{key}_ret"] = out[raw_factor]
        out[f"alpha_beta_{key}"] = cached[f"beta_{key}"].to_numpy()
        out[f"risk_beta_{key}"] = out[f"beta_{key}"]
    out["risk_factor_sector_ret"] = out["factor_sector_ret"]
    out["risk_beta_sector"] = out["beta_sector"]
    out["alpha_predicted_log_ret"] = cached["predicted_log_ret"].to_numpy()
    out["alpha_residual_log_ret"] = cached["residual_log_ret"].to_numpy()
    out["alpha_resid_z"] = cached["resid_z"].to_numpy()
    explained_without_intercept = sum(
        out[f"alpha_factor_{key}_ret"] * out[f"alpha_beta_{key}"] for key in FACTOR_KEYS
    )
    out["alpha_beta_intercept"] = out["alpha_predicted_log_ret"] - explained_without_intercept
    out["risk_beta_intercept"] = out["beta_intercept"]
    out.drop(columns=["beta_intercept", "beta_btc", "beta_eth", "beta_market", "beta_sector"], inplace=True)
    return out.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def merge_aligned(left: pd.DataFrame, right: pd.DataFrame, left_name: str, right_name: str) -> pd.DataFrame:
    left = left.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    right = right.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    if len(left) != len(right) or not left[["timestamp", "symbol"]].equals(right[["timestamp", "symbol"]]):
        raise ValueError(f"{left_name} and {right_name} are not exactly aligned")
    extra = [column for column in right.columns if column not in {"timestamp", "symbol"}]
    return pd.concat([left, right[extra]], axis=1)


def build_research_view(panel: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "timestamp",
        "symbol",
        "log_ret",
        "alpha_residual_log_ret",
        "alpha_resid_z",
        *[f"alpha_beta_{key}" for key in FACTOR_KEYS],
        *[column for column in panel.columns if column.startswith("label_")],
    ]
    out = panel[keep].rename(
        columns={
            "alpha_residual_log_ret": "residual_log_ret",
            "alpha_resid_z": "resid_z",
            **{f"alpha_beta_{key}": f"beta_{key}" for key in FACTOR_KEYS},
        }
    )
    return add_generic_features(out)


def attach_candidate_model_to_allocator(
    base_panel: pd.DataFrame,
    candidate: pd.DataFrame,
    use_alpha_beta_as_risk: bool = False,
) -> pd.DataFrame:
    columns = [
        "timestamp",
        "symbol",
        *[f"risk_beta_{key}" for key in FACTOR_KEYS],
        "label_resid_ret_h72",
        "fwd_log_ret_h72",
        "fwd_simple_ret_h72",
    ]
    replacement = candidate[columns].copy()
    if use_alpha_beta_as_risk:
        for key in FACTOR_KEYS:
            replacement[f"risk_beta_{key}"] = candidate[f"alpha_beta_{key}"].to_numpy()
    out = base_panel.drop(
        columns=["beta_btc", "beta_eth", "beta_market", "label_resid_ret_h72", "fwd_log_ret_h72", "fwd_simple_ret_h72"],
        errors="ignore",
    ).merge(replacement, on=["timestamp", "symbol"], how="left", validate="one_to_one")
    return add_portfolio_risk_aliases(out)


def run_audit(panel: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    thresholds = cfg["audit_gates"]
    diagnostics: list[dict[str, Any]] = []

    beta_window = int(cfg["alpha_residual_model"]["beta_window_hours"])
    beta_error = direct_beta_error(panel, beta_window=beta_window)
    risk_beta_error = direct_risk_beta_error(panel, beta_window=beta_window)
    predicted = panel["alpha_beta_intercept"].copy()
    for key in FACTOR_KEYS:
        predicted = predicted + panel[f"alpha_beta_{key}"] * panel[f"alpha_factor_{key}_ret"]
    residual_error = (panel["log_ret"] - predicted - panel["alpha_residual_log_ret"]).abs().max()
    label_error = direct_label_error(panel)

    condition_rows = []
    for symbol, frame in panel.groupby("symbol", sort=False):
        factor_frame = frame[[f"alpha_factor_{key}_ret" for key in FACTOR_KEYS]].dropna()
        active_cols = [column for column in factor_frame if float(factor_frame[column].std(ddof=0)) > 1e-12]
        corr = factor_frame[active_cols].corr().to_numpy(dtype=float)
        condition_value = float(np.linalg.cond(corr)) if corr.shape[0] >= 2 and np.isfinite(corr).all() else 1.0
        condition_rows.append({"symbol": symbol, "condition_number": condition_value})
    condition = pd.DataFrame(condition_rows)
    condition_median = float(condition["condition_number"].median())
    bnb_condition = float(condition.loc[condition["symbol"].eq("BNBUSDT"), "condition_number"].iloc[0])
    beta_change = float(
        panel.sort_values(["symbol", "timestamp"])
        .groupby("symbol")[[f"alpha_beta_{key}" for key in FACTOR_KEYS]]
        .diff()
        .abs()
        .stack()
        .mean()
    )
    residual_coverage = float(panel["alpha_residual_log_ret"].notna().mean())
    label_coverages = {h: float(panel[f"label_resid_ret_h{h}"].notna().mean()) for h in HORIZONS}
    min_label_coverage = min(label_coverages.values())
    rank_cols = [f"label_resid_rank_centered_h{h}" for h in HORIZONS]
    rank_mean_error = max(
        float(panel.groupby("timestamp")[column].mean().abs().max()) for column in rank_cols
    )
    composite_expected = sum(
        weight * panel[f"label_resid_rank_centered_h{horizon}"]
        for horizon, weight in LOW_TURNOVER_WEIGHTS.items()
    )
    composite_error = float((panel["label_slow_low_turnover"] - composite_expected).abs().max())
    no_sector_alpha_fields = not any("sector" in column for column in panel.columns if column.startswith("alpha_"))

    values = {
        "past_only_alpha_beta_error": beta_error,
        "past_only_portfolio_risk_beta_error": risk_beta_error,
        "alpha_residual_identity_error": float(residual_error),
        "terminal_label_formula_error": label_error,
        "factor_condition_number_median": condition_median,
        "bnb_factor_condition_number": bnb_condition,
        "alpha_beta_mean_abs_hourly_change": beta_change,
        "alpha_residual_coverage": residual_coverage,
        "minimum_terminal_label_coverage": min_label_coverage,
        "centered_rank_mean_error": rank_mean_error,
        "slow_low_turnover_composite_error": composite_error,
        "no_sector_factor_in_alpha_model": float(no_sector_alpha_fields),
    }
    for name, value in values.items():
        diagnostics.append({"metric": name, "value": value})
    for horizon, value in label_coverages.items():
        diagnostics.append({"metric": f"label_coverage_h{horizon}", "value": value})
    diagnostics.extend(condition.rename(columns={"condition_number": "value"}).assign(metric=lambda x: "condition_number__" + x["symbol"])[["metric", "value"]].to_dict("records"))

    gates = [
        gate("past_only_alpha_beta", beta_error, float(thresholds["max_past_only_beta_error"]), "max"),
        gate("past_only_portfolio_risk_beta", risk_beta_error, float(thresholds["max_past_only_beta_error"]), "max"),
        gate("alpha_residual_identity", residual_error, float(thresholds["max_residual_identity_error"]), "max"),
        gate("terminal_label_formula", label_error, float(thresholds["max_label_formula_error"]), "max"),
        gate("condition_number_median", condition_median, float(thresholds["max_condition_number_median"]), "max"),
        gate("bnb_condition_number", bnb_condition, float(thresholds["max_bnb_condition_number"]), "max"),
        gate("beta_mean_abs_hourly_change", beta_change, float(thresholds["max_beta_mean_abs_hourly_change"]), "max"),
        gate("residual_coverage", residual_coverage, float(thresholds["min_residual_coverage"]), "min"),
        gate("label_coverage", min_label_coverage, float(thresholds["min_label_coverage"]), "min"),
        gate("centered_rank_mean", rank_mean_error, float(thresholds["max_centered_rank_mean_error"]), "max"),
        gate("slow_label_composite_formula", composite_error, 1e-12, "max"),
        gate("alpha_sector_factor_absent", float(no_sector_alpha_fields), 1.0, "min"),
    ]
    return pd.DataFrame(gates), pd.DataFrame(diagnostics)


def gate(check: str, value: float, threshold: float, direction: str) -> dict[str, Any]:
    passed = np.isfinite(value) and (value <= threshold if direction == "max" else value >= threshold)
    return {"check": check, "value": float(value), "threshold": threshold, "direction": direction, "status": "pass" if passed else "fail"}


def direct_beta_error(panel: pd.DataFrame, beta_window: int) -> float:
    errors: list[float] = []
    beta_cols = ["alpha_beta_intercept", *[f"alpha_beta_{key}" for key in FACTOR_KEYS]]
    factor_cols = [f"alpha_factor_{key}_ret" for key in FACTOR_KEYS]
    for _, frame in panel.sort_values(["symbol", "timestamp"]).groupby("symbol", sort=False):
        frame = frame.reset_index(drop=True)
        candidates = np.flatnonzero(frame[beta_cols].notna().all(axis=1).to_numpy())
        if not len(candidates):
            continue
        sample_indices = sorted(set(int(candidates[int(q * (len(candidates) - 1))]) for q in (0.20, 0.50, 0.80)))
        for index in sample_indices:
            history = frame.iloc[max(0, index - beta_window):index][["log_ret", *factor_cols]].dropna()
            design = np.column_stack([np.ones(len(history)), history[factor_cols].to_numpy(dtype=float)])
            target = history["log_ret"].to_numpy(dtype=float)
            coef = np.linalg.pinv(design.T @ design, rcond=1e-10) @ (design.T @ target)
            stored = frame.loc[index, beta_cols].to_numpy(dtype=float)
            errors.append(float(np.max(np.abs(coef - stored))))
    return max(errors) if errors else np.nan


def direct_risk_beta_error(panel: pd.DataFrame, beta_window: int) -> float:
    factor_cols = [
        "risk_factor_btc_ret",
        "risk_factor_eth_ret",
        "risk_factor_market_ret",
        "risk_factor_sector_ret",
    ]
    beta_cols = [
        "risk_beta_intercept",
        "risk_beta_btc",
        "risk_beta_eth",
        "risk_beta_market",
        "risk_beta_sector",
    ]
    errors: list[float] = []
    for _, frame in panel.sort_values(["symbol", "timestamp"]).groupby("symbol", sort=False):
        frame = frame.reset_index(drop=True)
        candidates = np.flatnonzero(frame[beta_cols].notna().all(axis=1).to_numpy())
        if not len(candidates):
            continue
        sample_indices = sorted(set(int(candidates[int(q * (len(candidates) - 1))]) for q in (0.20, 0.50, 0.80)))
        for index in sample_indices:
            history = frame.iloc[max(0, index - beta_window):index][["log_ret", *factor_cols]].dropna()
            design = np.column_stack([np.ones(len(history)), history[factor_cols].to_numpy(dtype=float)])
            target = history["log_ret"].to_numpy(dtype=float)
            coef = np.linalg.pinv(design.T @ design, rcond=1e-10) @ (design.T @ target)
            stored = frame.loc[index, beta_cols].to_numpy(dtype=float)
            errors.append(float(np.max(np.abs(coef - stored))))
    return max(errors) if errors else np.nan


def direct_label_error(panel: pd.DataFrame) -> float:
    errors: list[float] = []
    for _, frame in panel.sort_values(["symbol", "timestamp"]).groupby("symbol", sort=False):
        frame = frame.reset_index(drop=True)
        valid = frame["label_resid_ret_h168"].notna() & frame["alpha_beta_btc"].notna()
        candidates = np.flatnonzero(valid.to_numpy())
        if not len(candidates):
            continue
        for index in sorted(set(int(candidates[int(q * (len(candidates) - 1))]) for q in (0.25, 0.60, 0.85))):
            for horizon in HORIZONS:
                offsets = slice(index + 2, index + 2 + horizon)
                raw = float(frame.iloc[offsets]["log_ret"].sum())
                explained = 0.0
                for key in FACTOR_KEYS:
                    explained += float(frame.loc[index, f"alpha_beta_{key}"]) * float(
                        frame.iloc[offsets][f"alpha_factor_{key}_ret"].sum()
                    )
                errors.append(abs((raw - explained) - float(frame.loc[index, f"label_resid_ret_h{horizon}"])))
    return max(errors) if errors else np.nan


def replace_fallback_with_risk_v2(
    replay: dict[str, Any],
    p15c: dict[str, Any],
    panel: pd.DataFrame,
    pf_cfg: Any,
    risk_template: Any,
    stage: str,
) -> dict[str, Any]:
    fallback_folds = set(
        replay["folds"].loc[replay["folds"]["fallback_to_p15c"].astype(bool), "fold"].astype(int)
    )
    if not fallback_folds:
        return replay
    panel_by_time = {timestamp: frame.set_index("symbol") for timestamp, frame in panel.groupby("timestamp", sort=False)}
    equity_parts = [replay["equity"][~replay["equity"]["fold"].astype(int).isin(fallback_folds)].copy()]
    position_parts = [replay["positions"][~replay["positions"]["fold"].astype(int).isin(fallback_folds)].copy()]
    fold_rows = replay["folds"].copy().set_index("fold")

    for fold in sorted(fallback_folds):
        old_equity = p15c["equity"][p15c["equity"]["fold"].astype(int).eq(fold)].sort_values("timestamp")
        old_positions = p15c["positions"][p15c["positions"]["fold"].astype(int).eq(fold)].copy()
        prev = pd.Series(0.0, index=sorted(panel["symbol"].unique()))
        equity_value = 1.0
        eq_rows: list[dict[str, Any]] = []
        pos_rows: list[dict[str, Any]] = []
        for timestamp, old_group in old_positions.groupby("timestamp", sort=True):
            timestamp = pd.Timestamp(timestamp)
            if timestamp not in panel_by_time:
                continue
            rows = panel_by_time[timestamp]
            old_weights = old_group.set_index("symbol")["weight"].astype(float)
            selected = rows.reindex(old_weights.index).reset_index()
            weights, info = constrained_neutralize_weights(
                raw_weights=old_weights,
                selected=selected,
                exposure_cols=list(pf_cfg.neutralize_exposures),
                gross_exposure=float(old_weights.abs().sum()),
                max_abs_weight=float(risk_template.max_symbol_weight),
                target_gross=float(old_weights.abs().sum()),
                return_info=True,
            )
            weights = weights[weights.abs() > 1e-12]
            if weights.empty:
                continue
            full = pd.Series(0.0, index=prev.index)
            full.loc[weights.index] = weights
            turnover = float((full - prev).abs().sum())
            active = rows.reindex(weights.index)
            gross_return = float((weights * active["fwd_simple_ret_h72"]).sum())
            cost_return = turnover * (pf_cfg.one_way_fee_bps + pf_cfg.one_way_slippage_bps) / 10_000.0
            period_return = gross_return - cost_return
            equity_value *= 1.0 + period_return
            exposure = portfolio_exposures(weights, active, list(pf_cfg.neutralize_exposures))
            source = old_equity[old_equity["timestamp"].eq(timestamp)]
            row = source.iloc[0].to_dict() if len(source) else {}
            row.update(
                {
                    "timestamp": timestamp,
                    "exit_timestamp": timestamp + pd.Timedelta(hours=73),
                    "period_return": period_return,
                    "gross_period_return": gross_return,
                    "cost_return": cost_return,
                    "equity": equity_value,
                    "turnover": turnover,
                    "gross_exposure": float(weights.abs().sum()),
                    "target_gross_exposure": float(old_weights.abs().sum()),
                    "abs_beta_btc": abs(exposure.get("beta_btc", 0.0)),
                    "abs_beta_eth": abs(exposure.get("beta_eth", 0.0)),
                    "abs_beta_market": abs(exposure.get("beta_market", 0.0)),
                    "symbol_concentration": float((weights**2).sum()),
                    "max_abs_weight": float(weights.abs().max()),
                    "optimizer_success": info.get("optimizer_success", np.nan),
                    "optimizer_exposure_error": info.get("optimizer_exposure_error", np.nan),
                    "optimizer_attempts": info.get("optimizer_attempts", np.nan),
                    "selected_formulas": "P15c fallback signal reprojected by portfolio_risk_v2_no_sector",
                    "fold": fold,
                    "stage": stage,
                    "protocol": PROTOCOL,
                    "selected_family": "__P15C_FALLBACK_RISK_V2__",
                    "fallback_to_p15c": True,
                }
            )
            eq_rows.append(row)
            old_by_symbol = old_group.set_index("symbol")
            for symbol, weight in weights.items():
                source_position = old_by_symbol.loc[symbol].to_dict() if symbol in old_by_symbol.index else {}
                source_position.update(
                    {
                        "timestamp": timestamp,
                        "exit_timestamp": timestamp + pd.Timedelta(hours=73),
                        "symbol": symbol,
                        "weight": float(weight),
                        "target_weight": float(weight),
                        "fwd_simple_ret": float(active.loc[symbol, "fwd_simple_ret_h72"]),
                        "label_resid_ret": float(active.loc[symbol, "label_resid_ret_h72"]),
                        "contribution": float(weight * active.loc[symbol, "fwd_simple_ret_h72"]),
                        "selected_score_cols": "P15c fallback signal | risk_v2_no_sector projection",
                        "fold": fold,
                        "stage": stage,
                        "protocol": PROTOCOL,
                        "selected_family": "__P15C_FALLBACK_RISK_V2__",
                        "fallback_to_p15c": True,
                    }
                )
                pos_rows.append(source_position)
            prev = full
        fold_equity = pd.DataFrame(eq_rows)
        fold_positions = pd.DataFrame(pos_rows)
        if not fold_equity.empty:
            metrics = aggregate_oos_metrics(fold_equity, pf_cfg.rebalance_hours)
            for key, value in metrics.items():
                fold_rows.loc[fold, key] = value
            fold_rows.loc[fold, "selected_family"] = "__P15C_FALLBACK_RISK_V2__"
            fold_rows.loc[fold, "fallback_reason"] = "p20_failed_reprojected_with_risk_v2"
            equity_parts.append(fold_equity)
            position_parts.append(fold_positions)

    equity = pd.concat(equity_parts, ignore_index=True).sort_values(["timestamp", "fold"]).reset_index(drop=True)
    positions = pd.concat(position_parts, ignore_index=True).sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    folds = fold_rows.reset_index().sort_values("fold")
    metrics = summarize_nested_result({"equity": equity, "positions": positions, "folds": folds}, pf_cfg.rebalance_hours)
    return {"equity": equity, "positions": positions, "folds": folds, "metrics": metrics}


def build_all_fallback_seed(p15c: dict[str, Any]) -> dict[str, Any]:
    folds = p15c["folds"].copy()
    folds["fallback_to_p15c"] = True
    folds["selected_family"] = "__P15C_SIGNAL_RISK_V2__"
    folds["fallback_reason"] = "structural_candidate_reprojects_all_folds"
    equity = p15c["equity"].copy()
    equity["fallback_to_p15c"] = True
    positions = p15c["positions"].copy()
    positions["fallback_to_p15c"] = True
    return {"folds": folds, "equity": equity, "positions": positions, "metrics": p15c["metrics"]}


def relabel_existing_result(result: dict[str, Any], stage: str) -> dict[str, Any]:
    folds = result["folds"].copy()
    equity = result["equity"].copy()
    positions = result["positions"].copy()
    folds["stage"] = stage
    equity["stage"] = stage
    positions["stage"] = stage
    folds["alpha_residual_model"] = ALPHA_MODEL_VERSION
    folds["portfolio_risk_model"] = RISK_MODEL_VERSION
    return {"folds": folds, "equity": equity, "positions": positions, "metrics": dict(result["metrics"])}


def plot_candidate_pnl(output_dir: Path, candidate: dict[str, Any]) -> None:
    equity = candidate["equity"].sort_values("timestamp").copy()
    gross = equity["gross_period_return"].astype(float)
    turnover = equity["turnover"].astype(float)
    fig, ax = plt.subplots(figsize=(13, 6.5))
    for cost, label, color in [(0.0, "Gross before cost", "#1f77b4"), (8.0, "Net at 8bps one-way", "#2ca02c")]:
        returns = gross - turnover * cost / 10_000.0
        pnl = (1.0 + returns).cumprod() - 1.0
        ax.plot(equity["timestamp"], pnl, label=label, linewidth=2.0, color=color)
    ax.axhline(0.0, color="#444444", linewidth=0.8)
    ax.set_title("P26 No-sector Residual v2 Candidate - Nested OOS PnL")
    ax.set_ylabel("Cumulative return")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "p26_candidate_pnl.png", dpi=170)
    plt.close(fig)


def write_manifest(
    output_dir: Path,
    candidate_config_path: Path,
    allocator_config_path: Path,
    raw_asset_path: Path,
    panel_path: Path,
    selected: pd.DataFrame,
    candidate: dict[str, Any],
    overlay: dict[str, Any],
    risk_v2: dict[str, Any],
) -> None:
    manifest = {
        "name": CORE_STAGE,
        "status": "research_candidate",
        "alpha_residual_model": ALPHA_MODEL_VERSION,
        "portfolio_risk_model": RISK_MODEL_VERSION,
        "selection_protocol": PROTOCOL,
        "selection_label": "label_slow_low_turnover",
        "active_alpha_folds": 0,
        "diagnostic_overlay_active_folds": int((~selected["fallback"].astype(bool)).sum()),
        "total_folds": int(len(selected)),
        "candidate_config": str(candidate_config_path),
        "allocator_config": str(allocator_config_path),
        "raw_asset_panel": str(raw_asset_path),
        "model_and_label_panel": str(panel_path),
        "nested_run": str(output_dir / "nested_candidate_run"),
        "audit": str(output_dir / "audit_gates.csv"),
        "metrics_6bps": candidate["metrics"],
        "rejected_slow_label_overlay_metrics_6bps": overlay["metrics"],
        "rejected_no_sector_risk_v2_metrics_6bps": risk_v2["metrics"],
        "interpretation_warning": (
            "P20 practical and no-sector were promoted after reviewing P25. "
            "P26 is a potential baseline for prospective research, not untouched OOS evidence."
        ),
    }
    (output_dir / "candidate_manifest.json").write_text(
        json.dumps(to_jsonable(manifest), indent=2), encoding="utf-8"
    )


def write_audit_report(output_dir: Path, audit: pd.DataFrame, diagnostics: pd.DataFrame) -> None:
    lines = [
        "# P26 Residual / Label 自动审计",
        "",
        "该审计在组合回放前执行；任一硬门槛失败时脚本立即终止，不生成候选 baseline。",
        "",
        "```text",
        audit.to_string(index=False),
        "```",
        "",
        "完整诊断见 `audit_diagnostics.csv`。",
    ]
    (output_dir / "AUDIT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def write_report(
    output_dir: Path,
    audit: pd.DataFrame,
    comparison: pd.DataFrame,
    cost: pd.DataFrame,
    regime: pd.DataFrame,
    fold_stability: pd.DataFrame,
    selected: pd.DataFrame,
) -> None:
    candidate_metrics = comparison[comparison["stage"].eq(CORE_STAGE)].iloc[0]
    overlay_metrics = comparison[comparison["stage"].eq(OVERLAY_STAGE)].iloc[0]
    risk_v2_metrics = comparison[comparison["stage"].eq(RISK_V2_STAGE)].iloc[0]
    p15c_metrics = comparison[comparison["stage"].eq("p15c")].iloc[0]
    cost8 = cost[(cost["stage"].eq(CORE_STAGE)) & (cost["one_way_cost_bps"].eq(8.0))].iloc[0]
    q8 = regime[(regime["stage"].eq(CORE_STAGE)) & (regime["cost_bps"].eq(8.0))].iloc[0]
    fold = fold_stability[fold_stability["stage"].eq(CORE_STAGE)].iloc[0]
    active = int((~selected["fallback"].astype(bool)).sum())
    lines = [
        "# P26 No-sector Residual v2 潜在新 Baseline",
        "",
        "## 已正式固化的结构",
        "",
        "- Alpha residual model：`alpha_residual_v2_no_sector_ols720`，仅使用 BTC、ETH、ex-self market 三个因子。",
        "- Portfolio risk model：保留已验证的 `portfolio_risk_v1_p15c_namespaced`，字段迁入独立 `risk_beta_*` 命名空间。",
        "- allocator 兼容字段只在入口由 `risk_beta_* -> beta_*` 显式映射，不再让 alpha beta 和 risk beta 静默混用。",
        "- 固定 48/72/96/168h terminal residual labels 均使用 t 时刻 beta；波动率归一化只使用过去 336h。",
        "- 横截面标签使用 [-1, 1] centered rank；标准选择标签为 `label_slow_low_turnover`。",
        "- path-aware labels 与 adverse excursion 字段作为独立标签和诊断保留。",
        "- 所有 residual/label 审计均为 build-fail gate。",
        "",
        "## 自动审计",
        "",
        f"- 通过 {int(audit['status'].eq('pass').sum())}/{len(audit)} 项硬门槛。",
        "- Alpha residual model 对 majors 与其他币统一取消 sector factor，因此 BNB 的结构性奇异问题已从 residual/label 生成层移除。",
        "",
        "## Nested OOS 结果",
        "",
        "- P26-core 不启用新 alpha family：全部 fold 保留 P15c 信号、eligibility 与已验证的 portfolio risk model；结构升级只作用于新的 residual/features/labels 研究入口。",
        f"- 默认 6bps：return {candidate_metrics['total_return']:.2%}，gross {candidate_metrics['gross_total_return_before_cost']:.2%}，Sharpe {candidate_metrics['sharpe']:.3f}，maxDD {candidate_metrics['max_drawdown']:.2%}。",
        f"- 8bps：return {cost8['total_return']:.2%}，Sharpe {cost8['sharpe']:.3f}，maxDD {cost8['max_drawdown']:.2%}。",
        f"- 2024Q2-Q3 @8bps：return {q8['return']:.2%}，maxDD {q8['max_drawdown']:.2%}。",
        f"- positive fold rate {fold['positive_fold_rate']:.2%}，worst-5 folds 平均 {fold['worst_5_mean_return']:.2%}。",
        f"- 对照 P15c 默认 6bps return {p15c_metrics['total_return']:.2%}，Sharpe {p15c_metrics['sharpe']:.3f}。",
        "",
        "## 被拒绝的 slow-label overlay",
        "",
        f"P20 practical 用 `label_slow_low_turnover` 激活 {active}/{len(selected)} folds 后，6bps return {overlay_metrics['total_return']:.2%}，gross {overlay_metrics['gross_total_return_before_cost']:.2%}，Sharpe {overlay_metrics['sharpe']:.3f}。该 overlay 明显劣于 P15c/P26-core，已作为失败实验隔离在 `nested_alpha_overlay_run/`，不会进入 baseline。",
        "",
        "## 被拒绝的 risk-v2 替换",
        "",
        f"把 Portfolio risk model 也替换成 no-sector beta 并重投影全部 P15c 权重后，6bps return {risk_v2_metrics['total_return']:.2%}，gross {risk_v2_metrics['gross_total_return_before_cost']:.2%}，Sharpe {risk_v2_metrics['sharpe']:.3f}。它低于 P15c，因此只保留在 `nested_risk_v2_reprojection_run/` 作为进一步研究备选。",
        "",
        "## 定位与限制",
        "",
        "P26 是后续探索应使用的潜在新 baseline，不覆盖正式 P15c。no-sector 与 P20 practical 是在查看 P25 后确定的，因此这条曲线不能被解释为未经观察的纯 OOS 证据；它的价值是把推荐结构固化成可复现、可审计、可继续做 prospective 验证的研究起点。",
        "",
        "## 复现",
        "",
        "```bash",
        "python scripts/run_p26_no_sector_candidate_baseline.py",
        "```",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
