#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
from run_p3d_p4_gated_rerank_alphagpt import (
    aggregate_oos_metrics,
    decision_end_for_closed_positions,
    load_panel as load_p3d_panel,
    make_portfolio_config,
    make_risk_template,
)
from run_p18_factor_decay_walkforward import (
    DEFAULT_ASSET_PANEL,
    DEFAULT_DERIVATIVES_PANEL,
    DEFAULT_MANUAL_PANEL,
    DEFAULT_P6B_PANEL,
    EXPERIMENT_ROOT,
    centered_rank_matrix,
    load_factor_library,
)
from run_p18c_factor_selector_portfolio import (
    DEFAULT_P15C_RUN,
    P18C_SCORE,
    breakeven_cost,
    build_asset_contribution,
    build_cost_sensitivity,
    build_segment_comparison,
    load_existing_result,
    write_run_result,
)
from run_p18d_guarded_factor_selector import (
    DEFAULT_COST_BPS,
    DEFAULT_P15C_SELECTED,
    DEFAULT_P15C_VALIDATION,
    DEFAULT_P18B,
    LONG_ROOT,
    add_regime_features,
    build_guard_specs,
    enrich_metrics,
    load_selected_p15c_validation,
    replay_selected_guards,
    run_guard_validation,
)
from run_turnover_experiments import to_jsonable


DEFAULT_CONFIG = EXPERIMENT_ROOT / "configs" / "baseline_locked_v1.yaml"
DEFAULT_FOLDS = LONG_ROOT / "inputs" / "long_formula_selections.csv"
DEFAULT_P18C_RUN = EXPERIMENT_ROOT / "outputs" / "p18c_factor_selector_portfolio" / "nested_p18c_run"
DEFAULT_P18D_RUN = EXPERIMENT_ROOT / "outputs" / "p18d_guarded_factor_selector" / "nested_p18d_run"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "outputs" / "p20_selection_protocol"


@dataclass(frozen=True)
class ProtocolSpec:
    name: str
    q25_floor: float
    min_second_half_ic: float
    min_decay_delta: float
    max_abs_pair_corr: float
    min_factor_count: int
    min_validation_high_risk_rate: float
    recent_high_risk_days: int
    recent_high_risk_trigger: float
    description: str


def main() -> None:
    parser = argparse.ArgumentParser(description="P20: stricter selection protocol before allowing P18d incremental factors.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--asset-panel", default=str(DEFAULT_ASSET_PANEL))
    parser.add_argument("--manual-panel", default=str(DEFAULT_MANUAL_PANEL))
    parser.add_argument("--p6b-panel", default=str(DEFAULT_P6B_PANEL))
    parser.add_argument("--derivatives-panel", default=str(DEFAULT_DERIVATIVES_PANEL))
    parser.add_argument("--folds", default=str(DEFAULT_FOLDS))
    parser.add_argument("--p18b-dir", default=str(DEFAULT_P18B))
    parser.add_argument("--p15c-run-dir", default=str(DEFAULT_P15C_RUN))
    parser.add_argument("--p18c-run-dir", default=str(DEFAULT_P18C_RUN))
    parser.add_argument("--p18d-run-dir", default=str(DEFAULT_P18D_RUN))
    parser.add_argument("--p15c-selected", default=str(DEFAULT_P15C_SELECTED))
    parser.add_argument("--p15c-validation", default=str(DEFAULT_P15C_VALIDATION))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--horizon", type=int, default=72)
    parser.add_argument("--validation-days", type=int, default=90)
    parser.add_argument("--cost-bps", type=float, nargs="+", default=DEFAULT_COST_BPS)
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    horizon = int(args.horizon)
    validation_days = int(args.validation_days)

    print("[1/8] Loading folds, selected factors, references", flush=True)
    folds = pd.read_csv(args.folds, parse_dates=["train_start", "train_end", "test_start", "test_end"]).sort_values("fold")
    selected_factors = pd.read_csv(Path(args.p18b_dir) / "selected_factors_by_fold.csv", parse_dates=["test_start"])
    selected_factors = selected_factors[selected_factors["horizon"].astype(int).eq(horizon)].copy()
    if selected_factors.empty:
        raise RuntimeError(f"No selected P18b factors found for horizon={horizon}.")
    p15c_result = load_existing_result(Path(args.p15c_run_dir))
    p18c_result = load_existing_result(Path(args.p18c_run_dir))
    p18d_result = load_existing_result(Path(args.p18d_run_dir))
    p15c_validation = load_selected_p15c_validation(Path(args.p15c_selected), Path(args.p15c_validation))

    print("[2/8] Loading factor ranks and allocator panel", flush=True)
    factor_panel, _ = load_factor_library(
        asset_panel=Path(args.asset_panel),
        manual_panel=Path(args.manual_panel),
        p6b_panel=Path(args.p6b_panel),
        derivatives_panel=Path(args.derivatives_panel),
        max_factors=0,
    )
    needed_factors = sorted(set(selected_factors["factor"]))
    rank_matrices = {factor: centered_rank_matrix(factor_panel, factor, min_symbols=8) for factor in needed_factors}
    base_panel = load_p3d_panel(config_path, Path(args.asset_panel), Path(args.manual_panel), horizon)
    add_regime_features(base_panel)

    pf_cfg = make_portfolio_config(cfg, horizon)
    risk_template = make_risk_template(cfg)
    guard_specs = build_guard_specs()
    protocols = build_protocols()
    pd.DataFrame([spec.__dict__ for spec in protocols]).to_csv(output_dir / "protocol_grid.csv", index=False)

    protocol_results: dict[str, dict[str, Any]] = {}
    summary_rows: list[dict[str, Any]] = []
    print("[3/8] Running protocol variants", flush=True)
    for spec in protocols:
        print(f"      protocol {spec.name}", flush=True)
        protocol_dir = ensure_dir(output_dir / spec.name)
        result, audit, filtered, validation, selected = run_protocol(
            spec=spec,
            base_panel=base_panel,
            folds=folds,
            selected_factors=selected_factors,
            rank_matrices=rank_matrices,
            p15c_validation=p15c_validation,
            p15c_result=p15c_result,
            pf_cfg=pf_cfg,
            risk_template=risk_template,
            guard_specs=guard_specs,
            validation_days=validation_days,
            protocol_dir=protocol_dir,
        )
        protocol_results[spec.name] = result
        write_run_result(protocol_dir / "nested_p20_run", result)
        audit.to_csv(protocol_dir / "factor_protocol_audit.csv", index=False)
        filtered.to_csv(protocol_dir / "filtered_selected_factors.csv", index=False)
        validation.to_csv(protocol_dir / "validation_guard_metrics.csv", index=False)
        selected.to_csv(protocol_dir / "selected_guard_by_fold.csv", index=False)
        comparison = build_comparison({spec.name: result, "p18d": p18d_result, "p18c": p18c_result, "p15c": p15c_result})
        cost = build_cost_sensitivity(
            {spec.name: result, "p18d": p18d_result, "p18c": p18c_result, "p15c": p15c_result},
            [float(x) for x in args.cost_bps],
        )
        segments = build_segment_comparison(
            {spec.name: result, "p18d": p18d_result, "p18c": p18c_result, "p15c": p15c_result},
            [float(x) for x in args.cost_bps],
        )
        fold_delta = build_fold_delta(result, p15c_result, spec.name)
        asset = build_asset_contribution(result["positions"])
        comparison.to_csv(protocol_dir / "comparison_summary.csv", index=False)
        cost.to_csv(protocol_dir / "cost_sensitivity.csv", index=False)
        segments.to_csv(protocol_dir / "segment_cost_summary.csv", index=False)
        fold_delta.to_csv(protocol_dir / "fold_delta_vs_p15c.csv", index=False)
        asset.to_csv(protocol_dir / "asset_contribution.csv", index=False)
        plot_curves(protocol_dir, {spec.name: result, "p18d": p18d_result, "p15c": p15c_result}, [6.0, 15.0, 20.0, 30.0])
        plot_fold_delta(protocol_dir, fold_delta, spec.name)
        write_protocol_report(protocol_dir, spec, comparison, cost, segments, fold_delta, asset, audit, selected)
        summary_rows.append(build_protocol_summary_row(spec, result, cost, selected, audit, fold_delta))

    print("[4/8] Writing combined protocol comparison", flush=True)
    summary = pd.DataFrame(summary_rows).sort_values("total_return", ascending=False)
    summary.to_csv(output_dir / "protocol_summary.csv", index=False)
    refs = {"p18d": p18d_result, "p18c": p18c_result, "p15c": p15c_result, **protocol_results}
    combined = build_comparison(refs)
    combined.to_csv(output_dir / "combined_comparison_summary.csv", index=False)
    combined_cost = build_cost_sensitivity(refs, [float(x) for x in args.cost_bps])
    combined_cost.to_csv(output_dir / "combined_cost_sensitivity.csv", index=False)

    print("[5/8] Writing combined plots", flush=True)
    plot_curves(output_dir, refs, [6.0, 15.0, 20.0, 30.0])
    plot_protocol_summary(output_dir, summary)

    print("[6/8] Writing top-level report", flush=True)
    write_top_report(output_dir, summary, combined, combined_cost)
    metadata = {
        "horizon": horizon,
        "validation_days": validation_days,
        "protocols": [spec.__dict__ for spec in protocols],
        "selection_layer": "factor quality + homogeneity + regime coverage before P18d guard validation; fallback to P15c otherwise",
    }
    (output_dir / "metadata.json").write_text(json.dumps(to_jsonable(metadata), indent=2), encoding="utf-8")
    print(f"Done: {output_dir}", flush=True)


def build_protocols() -> list[ProtocolSpec]:
    return [
        ProtocolSpec(
            name="p20_strict_q25_nonneg",
            q25_floor=0.0,
            min_second_half_ic=0.02,
            min_decay_delta=-0.05,
            max_abs_pair_corr=0.70,
            min_factor_count=2,
            min_validation_high_risk_rate=0.20,
            recent_high_risk_days=14,
            recent_high_risk_trigger=0.50,
            description="Strict interpretation: any negative validation q25 IC is rejected.",
        ),
        ProtocolSpec(
            name="p20_balanced_q25_m3",
            q25_floor=-0.03,
            min_second_half_ic=0.02,
            min_decay_delta=-0.08,
            max_abs_pair_corr=0.75,
            min_factor_count=3,
            min_validation_high_risk_rate=0.20,
            recent_high_risk_days=14,
            recent_high_risk_trigger=0.50,
            description="Balanced protocol: allows small negative q25 noise but rejects fragile lower tails.",
        ),
        ProtocolSpec(
            name="p20_practical_q25_m5",
            q25_floor=-0.05,
            min_second_half_ic=0.015,
            min_decay_delta=-0.10,
            max_abs_pair_corr=0.80,
            min_factor_count=3,
            min_validation_high_risk_rate=0.20,
            recent_high_risk_days=14,
            recent_high_risk_trigger=0.50,
            description="Practical protocol: enough factors may survive, but still blocks strong decay and uncovered risk regimes.",
        ),
        ProtocolSpec(
            name="p20_loose_decay_corr",
            q25_floor=-0.07,
            min_second_half_ic=0.01,
            min_decay_delta=-0.12,
            max_abs_pair_corr=0.85,
            min_factor_count=4,
            min_validation_high_risk_rate=0.15,
            recent_high_risk_days=14,
            recent_high_risk_trigger=0.60,
            description="Loose diagnostic protocol to test whether any filtered P18d edge remains after homogeneity and decay checks.",
        ),
    ]


def run_protocol(
    spec: ProtocolSpec,
    base_panel: pd.DataFrame,
    folds: pd.DataFrame,
    selected_factors: pd.DataFrame,
    rank_matrices: dict[str, pd.DataFrame],
    p15c_validation: pd.DataFrame,
    p15c_result: dict[str, Any],
    pf_cfg: Any,
    risk_template: Any,
    guard_specs: list[Any],
    validation_days: int,
    protocol_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    audit, filtered = apply_selection_protocol(
        spec=spec,
        base_panel=base_panel,
        folds=folds,
        selected_factors=selected_factors,
        rank_matrices=rank_matrices,
        validation_days=validation_days,
        horizon=pf_cfg.primary_horizon,
    )
    if filtered.empty:
        validation = pd.DataFrame()
        selected = build_all_fallback_selection(folds, p15c_validation, audit)
    else:
        validation, selected_partial = run_guard_validation(
            base_panel=base_panel,
            folds=folds,
            selected_factors=filtered,
            rank_matrices=rank_matrices,
            p15c_validation=p15c_validation,
            pf_cfg=pf_cfg,
            risk_template=risk_template,
            guard_specs=guard_specs,
            validation_days=validation_days,
        )
        selected = complete_selection_with_protocol_fallbacks(folds, p15c_validation, audit, selected_partial)
    result = replay_selected_guards(
        base_panel=base_panel,
        folds=folds,
        selected_factors=filtered,
        rank_matrices=rank_matrices,
        selected=selected,
        p15c_result=p15c_result,
        pf_cfg=pf_cfg,
        risk_template=risk_template,
    )
    ensure_fold_column(result, folds)
    tag_result_rows(result, spec.name)
    return result, audit, filtered, validation, selected


def apply_selection_protocol(
    spec: ProtocolSpec,
    base_panel: pd.DataFrame,
    folds: pd.DataFrame,
    selected_factors: pd.DataFrame,
    rank_matrices: dict[str, pd.DataFrame],
    validation_days: int,
    horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_by_fold = {int(k): v.copy() for k, v in selected_factors.groupby("fold", sort=False)}
    audit_rows: list[dict[str, Any]] = []
    keep_parts: list[pd.DataFrame] = []
    for _, fold in folds.sort_values("fold").iterrows():
        fold_id = int(fold["fold"])
        validation_start, validation_end = validation_window(fold, validation_days, horizon)
        train_end = pd.Timestamp(fold["train_end"])
        fold_group = selected_by_fold.get(fold_id, pd.DataFrame()).copy()
        if fold_group.empty:
            audit_rows.append(
                base_audit_row(spec, fold_id, validation_start, validation_end, "no_selected_factors")
            )
            continue

        validation_high, recent_high, current_high = regime_coverage(
            base_panel, validation_start, validation_end, train_end, spec.recent_high_risk_days
        )
        quality = factor_quality_mask(fold_group, spec)
        quality_pass = fold_group[quality].copy()
        quality_reject_count = int((~quality).sum())
        pre_corr = factor_homogeneity(fold_group, rank_matrices, validation_start, validation_end)
        pruned, corr_rejects, post_corr = greedy_decorrelate(
            quality_pass, rank_matrices, validation_start, validation_end, spec.max_abs_pair_corr
        )

        fail_reasons: list[str] = []
        if len(pruned) < spec.min_factor_count:
            fail_reasons.append("too_few_surviving_factors")
        if validation_high < spec.min_validation_high_risk_rate and recent_high >= spec.recent_high_risk_trigger:
            fail_reasons.append("unsupported_current_high_risk_regime")
        protocol_pass = not fail_reasons
        if protocol_pass:
            keep_parts.append(pruned)

        audit_rows.append(
            {
                "protocol": spec.name,
                "fold": fold_id,
                "validation_start": validation_start,
                "validation_decision_end": validation_end,
                "test_start": fold["test_start"],
                "original_factor_count": int(len(fold_group)),
                "quality_pass_count": int(len(quality_pass)),
                "quality_reject_count": quality_reject_count,
                "corr_reject_count": int(corr_rejects),
                "surviving_factor_count": int(len(pruned)),
                "pre_mean_abs_factor_corr": pre_corr,
                "post_mean_abs_factor_corr": post_corr,
                "validation_high_risk_rate": validation_high,
                "recent_high_risk_rate": recent_high,
                "current_high_risk": bool(current_high),
                "protocol_pass": bool(protocol_pass),
                "fallback_reason": "|".join(fail_reasons) if fail_reasons else "",
                "surviving_raw_names": "|".join(pruned["raw_name"].astype(str).tolist()) if not pruned.empty else "",
            }
        )
    audit = pd.DataFrame(audit_rows)
    filtered = pd.concat(keep_parts, ignore_index=True) if keep_parts else pd.DataFrame(columns=selected_factors.columns)
    return audit, filtered


def validation_window(fold: pd.Series, validation_days: int, horizon: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    validation_start = max(pd.Timestamp(fold["train_start"]), pd.Timestamp(fold["train_end"]) - pd.Timedelta(days=validation_days))
    validation_end = min(
        decision_end_for_closed_positions(validation_start + pd.Timedelta(days=validation_days), horizon),
        pd.Timestamp(fold["train_end"]),
    )
    return validation_start, validation_end


def base_audit_row(
    spec: ProtocolSpec,
    fold_id: int,
    validation_start: pd.Timestamp,
    validation_end: pd.Timestamp,
    reason: str,
) -> dict[str, Any]:
    return {
        "protocol": spec.name,
        "fold": fold_id,
        "validation_start": validation_start,
        "validation_decision_end": validation_end,
        "original_factor_count": 0,
        "quality_pass_count": 0,
        "quality_reject_count": 0,
        "corr_reject_count": 0,
        "surviving_factor_count": 0,
        "pre_mean_abs_factor_corr": np.nan,
        "post_mean_abs_factor_corr": np.nan,
        "validation_high_risk_rate": np.nan,
        "recent_high_risk_rate": np.nan,
        "current_high_risk": False,
        "protocol_pass": False,
        "fallback_reason": reason,
        "surviving_raw_names": "",
    }


def factor_quality_mask(fold_group: pd.DataFrame, spec: ProtocolSpec) -> pd.Series:
    q25 = pd.to_numeric(fold_group["validation_ic_q25"], errors="coerce")
    second = pd.to_numeric(fold_group["validation_second_half_ic"], errors="coerce")
    decay = pd.to_numeric(fold_group["validation_decay_delta"], errors="coerce")
    return (q25 >= spec.q25_floor) & (second >= spec.min_second_half_ic) & (decay >= spec.min_decay_delta)


def greedy_decorrelate(
    factors: pd.DataFrame,
    rank_matrices: dict[str, pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
    max_abs_corr: float,
) -> tuple[pd.DataFrame, int, float]:
    if factors.empty:
        return factors.copy(), 0, np.nan
    ordered = factors.sort_values("selection_score", ascending=False).copy()
    kept_rows: list[pd.Series] = []
    kept_factors: list[str] = []
    rejects = 0
    for _, row in ordered.iterrows():
        factor = str(row["factor"])
        if factor not in rank_matrices:
            rejects += 1
            continue
        if not kept_factors:
            kept_rows.append(row)
            kept_factors.append(factor)
            continue
        corr_values = [
            abs(pair_corr(rank_matrices[factor], rank_matrices[kept], start, end))
            for kept in kept_factors
            if kept in rank_matrices
        ]
        max_corr = np.nanmax(corr_values) if corr_values else np.nan
        if np.isfinite(max_corr) and max_corr > max_abs_corr:
            rejects += 1
            continue
        kept_rows.append(row)
        kept_factors.append(factor)
    out = pd.DataFrame(kept_rows).reset_index(drop=True) if kept_rows else factors.iloc[0:0].copy()
    return out, rejects, factor_homogeneity(out, rank_matrices, start, end)


def factor_homogeneity(
    factors: pd.DataFrame,
    rank_matrices: dict[str, pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> float:
    names = [str(x) for x in factors["factor"].tolist() if str(x) in rank_matrices]
    if len(names) < 2:
        return 0.0 if len(names) == 1 else np.nan
    values: list[float] = []
    for idx, left in enumerate(names):
        for right in names[idx + 1 :]:
            corr = abs(pair_corr(rank_matrices[left], rank_matrices[right], start, end))
            if np.isfinite(corr):
                values.append(float(corr))
    return float(np.mean(values)) if values else np.nan


def pair_corr(left: pd.DataFrame, right: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> float:
    x, y = left.loc[start:end].align(right.loc[start:end], join="inner", axis=0)
    x, y = x.align(y, join="inner", axis=1)
    xv = x.to_numpy(dtype=float).ravel()
    yv = y.to_numpy(dtype=float).ravel()
    valid = np.isfinite(xv) & np.isfinite(yv)
    if valid.sum() < 30:
        return np.nan
    return float(np.corrcoef(xv[valid], yv[valid])[0, 1])


def regime_coverage(
    base_panel: pd.DataFrame,
    validation_start: pd.Timestamp,
    validation_end: pd.Timestamp,
    train_end: pd.Timestamp,
    recent_days: int,
) -> tuple[float, float, bool]:
    regime = (
        base_panel[["timestamp", "_p18d_high_risk_regime"]]
        .drop_duplicates("timestamp")
        .sort_values("timestamp")
        .set_index("timestamp")["_p18d_high_risk_regime"]
        .astype(bool)
    )
    validation = regime[(regime.index >= validation_start) & (regime.index <= validation_end)]
    recent_start = train_end - pd.Timedelta(days=recent_days)
    recent = regime[(regime.index >= recent_start) & (regime.index <= train_end)]
    current = regime[regime.index <= train_end].tail(1)
    return (
        float(validation.mean()) if len(validation) else np.nan,
        float(recent.mean()) if len(recent) else np.nan,
        bool(current.iloc[0]) if len(current) else False,
    )


def build_all_fallback_selection(
    folds: pd.DataFrame,
    p15c_validation: pd.DataFrame,
    audit: pd.DataFrame,
) -> pd.DataFrame:
    return complete_selection_with_protocol_fallbacks(folds, p15c_validation, audit, pd.DataFrame())


def complete_selection_with_protocol_fallbacks(
    folds: pd.DataFrame,
    p15c_validation: pd.DataFrame,
    audit: pd.DataFrame,
    selected_partial: pd.DataFrame,
) -> pd.DataFrame:
    p15c_by_fold = p15c_validation.set_index("fold")
    audit_by_fold = audit.set_index("fold")
    partial_by_fold = selected_partial.set_index("fold") if not selected_partial.empty else pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for _, fold in folds.sort_values("fold").iterrows():
        fold_id = int(fold["fold"])
        audit_row = audit_by_fold.loc[fold_id].to_dict() if fold_id in audit_by_fold.index else {}
        if not partial_by_fold.empty and fold_id in partial_by_fold.index:
            row = partial_by_fold.loc[fold_id].to_dict()
            row["fold"] = fold_id
            if bool(row.get("fallback_to_p15c", False)):
                row["fallback_reason"] = "guard_validation_failed"
            else:
                row["fallback_reason"] = ""
        else:
            ref = p15c_by_fold.loc[fold_id]
            row = {
                "fold": fold_id,
                "selected_guard_spec": "__P15C_FALLBACK__",
                "fallback_to_p15c": True,
                "validation_selection_score": float(ref["validation_total_return"]),
                "validation_total_return": float(ref["validation_total_return"]),
                "validation_gross_total_return_before_cost": float(ref["validation_gross_total_return_before_cost"]),
                "validation_total_return_20bps": float(ref["validation_total_return_20bps"]),
                "validation_sharpe": float(ref["validation_sharpe"]),
                "validation_max_drawdown": float(ref["validation_max_drawdown"]),
                "fallback_reason": audit_row.get("fallback_reason", "protocol_failed"),
            }
        for key in [
            "protocol",
            "original_factor_count",
            "quality_pass_count",
            "corr_reject_count",
            "surviving_factor_count",
            "pre_mean_abs_factor_corr",
            "post_mean_abs_factor_corr",
            "validation_high_risk_rate",
            "recent_high_risk_rate",
            "current_high_risk",
            "protocol_pass",
            "surviving_raw_names",
        ]:
            row[key] = audit_row.get(key, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def tag_result_rows(result: dict[str, Any], protocol_name: str) -> None:
    for key in ["equity", "positions", "folds"]:
        if key in result and not result[key].empty:
            result[key] = result[key].copy()
            result[key]["protocol"] = protocol_name


def ensure_fold_column(result: dict[str, Any], folds: pd.DataFrame) -> None:
    if "folds" not in result or result["folds"].empty:
        return
    frame = result["folds"].copy()
    if "fold" not in frame.columns:
        frame.insert(0, "fold", folds.sort_values("fold")["fold"].to_numpy()[: len(frame)])
    result["folds"] = frame


def build_comparison(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame([{"stage": name, **result["metrics"]} for name, result in results.items()])


def build_fold_delta(p20: dict[str, Any], p15c: dict[str, Any], stage_name: str) -> pd.DataFrame:
    left_cols = [
        "fold",
        "test_start",
        "selected_guard_spec",
        "fallback_to_p15c",
        "fallback_reason",
        "surviving_factor_count",
        "validation_high_risk_rate",
        "recent_high_risk_rate",
        "total_return",
        "gross_total_return_before_cost",
        "sharpe",
        "max_drawdown",
        "mean_turnover",
    ]
    left = p20["folds"][[c for c in left_cols if c in p20["folds"].columns]].copy()
    right = p15c["folds"][["fold", "total_return", "gross_total_return_before_cost", "sharpe", "max_drawdown", "mean_turnover"]].copy()
    merged = left.merge(right, on="fold", suffixes=(f"_{stage_name}", "_p15c"))
    for col in ["total_return", "gross_total_return_before_cost", "sharpe", "max_drawdown", "mean_turnover"]:
        merged[f"delta_{col}"] = merged[f"{col}_{stage_name}"] - merged[f"{col}_p15c"]
    return merged


def build_protocol_summary_row(
    spec: ProtocolSpec,
    result: dict[str, Any],
    cost: pd.DataFrame,
    selected: pd.DataFrame,
    audit: pd.DataFrame,
    fold_delta: pd.DataFrame,
) -> dict[str, Any]:
    metrics = result["metrics"]
    return {
        "protocol": spec.name,
        "description": spec.description,
        "accepted_fold_rate": 1.0 - float(selected["fallback_to_p15c"].astype(bool).mean()),
        "protocol_pass_fold_rate": float(audit["protocol_pass"].astype(bool).mean()),
        "mean_surviving_factor_count": float(audit["surviving_factor_count"].mean()),
        "folds_beating_p15c": int((fold_delta["delta_total_return"] > 0).sum()),
        "worst_fold_delta": float(fold_delta["delta_total_return"].min()),
        "breakeven_one_way_bps": breakeven_cost(cost, spec.name),
        **metrics,
    }


def plot_curves(output_dir: Path, results: dict[str, dict[str, Any]], cost_bps: list[float]) -> None:
    for bps in cost_bps:
        fig, ax = plt.subplots(figsize=(14, 7))
        for name, result in results.items():
            equity = result["equity"].sort_values("timestamp")
            gross = equity["gross_period_return"] if "gross_period_return" in equity else equity["period_return"] + equity["cost_return"]
            ret = gross - equity["turnover"] * bps / 10_000.0
            curve = (1.0 + ret).cumprod() - 1.0
            width = 2.4 if name.startswith("p20") else 1.5
            ax.plot(equity["timestamp"], curve, label=name, linewidth=width)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_title(f"P20 selection protocol cumulative return, {bps:g}bps one-way cost")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Cumulative return")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / f"p20_pnl_{int(bps)}bps.png", dpi=160)
        plt.close(fig)


def plot_fold_delta(output_dir: Path, fold_delta: pd.DataFrame, stage_name: str) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(fold_delta["fold"], fold_delta["delta_total_return"])
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title(f"{stage_name} fold net return delta vs P15c")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Delta return")
    fig.tight_layout()
    fig.savefig(output_dir / "fold_delta_vs_p15c.png", dpi=160)
    plt.close(fig)


def plot_protocol_summary(output_dir: Path, summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(summary["protocol"], summary["total_return"])
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("P20 protocol net return")
    ax.set_xlabel("Protocol")
    ax.set_ylabel("Net return")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output_dir / "protocol_total_return.png", dpi=160)
    plt.close(fig)


def write_protocol_report(
    output_dir: Path,
    spec: ProtocolSpec,
    comparison: pd.DataFrame,
    cost: pd.DataFrame,
    segments: pd.DataFrame,
    fold_delta: pd.DataFrame,
    asset: pd.DataFrame,
    audit: pd.DataFrame,
    selected: pd.DataFrame,
) -> None:
    accept_rate = 1.0 - float(selected["fallback_to_p15c"].astype(bool).mean()) if not selected.empty else np.nan
    protocol_pass_rate = float(audit["protocol_pass"].astype(bool).mean()) if not audit.empty else np.nan
    report = [
        f"# {spec.name}",
        "",
        spec.description,
        "",
        f"- accepted fold rate after guard validation: {accept_rate:.2%}",
        f"- protocol pass fold rate before guard validation: {protocol_pass_rate:.2%}",
        f"- breakeven one-way cost: {breakeven_cost(cost, spec.name):.2f}bps",
        "",
        "## Comparison",
        "",
        "```text",
        comparison.to_string(index=False),
        "```",
        "",
        "## Cost Sensitivity",
        "",
        "```text",
        cost[cost["one_way_cost_bps"].isin([6.0, 15.0, 20.0, 30.0])].to_string(index=False),
        "```",
        "",
        "## Segment Summary",
        "",
        "```text",
        segments[segments["one_way_cost_bps"].isin([6.0, 15.0, 20.0, 30.0])].to_string(index=False),
        "```",
        "",
        "## Fallback Reasons",
        "",
        "```text",
        selected["fallback_reason"].fillna("").replace("", "accepted").value_counts().to_string(),
        "```",
        "",
        "## Worst Fold Deltas vs P15c",
        "",
        "```text",
        fold_delta.sort_values("delta_total_return").head(12).to_string(index=False),
        "```",
        "",
        "## Factor Protocol Audit",
        "",
        "```text",
        audit[[
            "fold",
            "protocol_pass",
            "fallback_reason",
            "quality_pass_count",
            "corr_reject_count",
            "surviving_factor_count",
            "validation_high_risk_rate",
            "recent_high_risk_rate",
            "pre_mean_abs_factor_corr",
            "post_mean_abs_factor_corr",
        ]].to_string(index=False),
        "```",
        "",
        "## Asset Contribution",
        "",
        "```text",
        asset.to_string(index=False) if not asset.empty else "None",
        "```",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")


def write_top_report(output_dir: Path, summary: pd.DataFrame, combined: pd.DataFrame, cost: pd.DataFrame) -> None:
    report = [
        "# P20 Selection Protocol",
        "",
        "Purpose: fix the P18d failure mode by adding a pre-trade selection protocol before incremental factors can replace P15c.",
        "",
        "The protocol checks factor lower-tail IC, validation second-half decay, factor homogeneity, and whether the current high-risk regime was represented in validation. Failing folds fall back to P15c.",
        "",
        "## Protocol Summary",
        "",
        "```text",
        summary.to_string(index=False),
        "```",
        "",
        "## Combined Comparison",
        "",
        "```text",
        combined.to_string(index=False),
        "```",
        "",
        "## Combined Cost Sensitivity",
        "",
        "```text",
        cost[cost["one_way_cost_bps"].isin([6.0, 15.0, 20.0, 30.0])].to_string(index=False),
        "```",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
