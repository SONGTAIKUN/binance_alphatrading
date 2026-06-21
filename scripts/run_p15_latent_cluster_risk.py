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
import yaml
from matplotlib.ticker import PercentFormatter
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from crypto_stat_arb.config import ensure_dir, load_config
from crypto_stat_arb.portfolio import portfolio_exposures, spearman_ic, update_state
from run_p3_committee_guard_scan import apply_short_guard, side_fraction_weights
import run_p3d_p4_gated_rerank_alphagpt as p3d
import run_p7_derivatives_gated_rerank_alphagpt as p7
import run_p9_clean_derivatives_baseline as p9
import run_p10_strategy_conditioned_regime_gate as p10
import run_p10d_nested_allocator_short_guard as p10d
from run_p13_turnover_no_trade_zone import calibrate_score_coef, make_fold_risk, write_result
from run_turnover_experiments import to_jsonable


EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "baseline_locked_v1"
DEFAULT_CONFIG = EXPERIMENT_ROOT / "configs" / "baseline_locked_v1.yaml"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "outputs" / "p15_latent_cluster_risk"
P9_DIR = EXPERIMENT_ROOT / "outputs" / "p9_clean_baseline_package" / "nested_incremental_run"
P9_SELECTED = EXPERIMENT_ROOT / "outputs" / "p9_clean_baseline_package" / "selected_incremental_by_fold.csv"
NO_CLUSTER_CAP = "__P9_NO_CLUSTER_CAP__"


@dataclass(frozen=True)
class ClusterCapSpec:
    name: str
    lookback_hours: int
    cluster_k: int
    cluster_cap: float
    min_gross_retention: float = 0.85


@dataclass(frozen=True)
class LatentRiskModel:
    symbols: list[str]
    cluster_by_symbol: dict[str, int]
    pcs: np.ndarray
    corr: np.ndarray
    observations: int


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--horizon", type=int, default=72)
    parser.add_argument("--inner-validation-days", type=int, default=90)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--cost-bps", type=float, nargs="+", default=[0, 4, 6, 8, 10, 12, 15, 20, 30])
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    output_dir = ensure_dir(args.output_dir)
    run_dir = ensure_dir(output_dir / "nested_cluster_cap_run")
    p9_cluster_run_dir = ensure_dir(output_dir / "p9_cluster_reference_run")
    horizon = int(args.horizon)
    cost_grid = [float(x) for x in args.cost_bps]

    print("[1/10] Loading P9 references and formula universes", flush=True)
    p9_result = p9.load_existing_result(P9_DIR)
    selections_path = p3d.resolve_config_path(config_path, cfg["inputs"]["formula_selections"])
    selections = pd.read_csv(selections_path, parse_dates=["train_start", "train_end", "test_start", "test_end"])
    selected_p9 = pd.read_csv(P9_SELECTED, parse_dates=["train_start", "train_end", "test_start", "test_end", "test_decision_end"])
    panel_path = p3d.resolve_config_path(config_path, cfg["inputs"]["panel"])

    retained = p7.load_retained_features()
    raw_deriv_formulas = p7.build_p7_formula_universe(retained, 160, 2607, 3)
    deriv_formulas = {item.name: item for item in p9.filter_p9_formulas(raw_deriv_formulas, exclude_k01=True)}
    p3d_formulas = {item.name: item for item in p3d.build_incremental_formula_universe(180, 2417, 3)}
    spec_by_name = {spec.name: spec for spec in p3d.build_committee_specs()}

    print("[2/10] Loading residual panel and building rolling risk inputs", flush=True)
    panel = p7.load_panel_p7(config_path, panel_path, p3d.MANUAL_FEATURE_PANEL, horizon)
    residual_wide = (
        panel[["timestamp", "symbol", "residual_log_ret"]]
        .dropna()
        .pivot(index="timestamp", columns="symbol", values="residual_log_ret")
        .sort_index()
    )
    pf_cfg = p3d.make_portfolio_config(cfg, horizon)
    risk_template = p3d.make_risk_template(cfg)

    print("[3/10] Running P15a latent cluster diagnostics on P9 positions", flush=True)
    diagnostics, cluster_attr = build_p15a_diagnostics(p9_result, residual_wide)
    diagnostics.to_csv(output_dir / "p15a_latent_diagnostics.csv", index=False)
    cluster_attr.to_csv(output_dir / "p15a_cluster_attribution.csv", index=False)
    latent_summary = build_latent_summary(diagnostics, cluster_attr)
    latent_summary.to_csv(output_dir / "p15a_latent_summary.csv", index=False)

    print("[4/10] Building P15b cluster cap grid", flush=True)
    cluster_specs = build_cluster_specs()
    pd.DataFrame([spec.__dict__ for spec in cluster_specs]).to_csv(output_dir / "cluster_cap_grid.csv", index=False)

    print("[5/10] Running fold-local validation search", flush=True)
    validation, p9_validation = run_validation_search(
        panel=panel,
        residual_wide=residual_wide,
        selections=selections,
        selected_p9=selected_p9,
        pf_cfg=pf_cfg,
        risk_template=risk_template,
        spec_by_name=spec_by_name,
        deriv_formulas=deriv_formulas,
        p3d_formulas=p3d_formulas,
        cluster_specs=cluster_specs,
        validation_days=int(args.inner_validation_days),
    )
    validation.to_csv(output_dir / "validation_cluster_cap_metrics.csv", index=False)
    p9_validation.to_csv(output_dir / "validation_p9_cluster_reference.csv", index=False)

    print("[6/10] Selecting cluster cap spec per fold without test data", flush=True)
    selected = select_cluster_spec_by_fold(validation, p9_validation, selections)
    selected.to_csv(output_dir / "selected_cluster_cap_by_fold.csv", index=False)

    print("[7/10] Replaying selected cluster caps on test windows only", flush=True)
    nested = replay_selected_cluster_specs(
        panel=panel,
        residual_wide=residual_wide,
        selections=selections,
        selected=selected,
        selected_p9=selected_p9,
        pf_cfg=pf_cfg,
        risk_template=risk_template,
        spec_by_name=spec_by_name,
        deriv_formulas=deriv_formulas,
        p3d_formulas=p3d_formulas,
        cluster_specs=cluster_specs,
        force_baseline=False,
    )
    write_result(run_dir, nested)

    print("[8/10] Replaying same fold diagnostics for P9 cluster reference", flush=True)
    p9_cluster_ref = replay_selected_cluster_specs(
        panel=panel,
        residual_wide=residual_wide,
        selections=selections,
        selected=selected,
        selected_p9=selected_p9,
        pf_cfg=pf_cfg,
        risk_template=risk_template,
        spec_by_name=spec_by_name,
        deriv_formulas=deriv_formulas,
        p3d_formulas=p3d_formulas,
        cluster_specs=cluster_specs,
        force_baseline=True,
    )
    write_result(p9_cluster_run_dir, p9_cluster_ref)

    print("[9/10] Building diagnostics, robustness checks, and plots", flush=True)
    results = {"p15b_cluster_cap": nested, "p9_reference": p9_result, "p9_cluster_reference": p9_cluster_ref}
    summary = build_p15_summary(results, horizon)
    summary.to_csv(output_dir / "variant_summary.csv", index=False)
    cost_summary = p10.build_cost_summary({"p15b_cluster_cap": nested, "p9_reference": p9_result}, cost_grid)
    cost_summary.to_csv(output_dir / "cost_sensitivity.csv", index=False)
    fold_delta = p9.build_fold_delta(nested, p9_result, "p15b_cluster_cap", "p9")
    fold_delta.to_csv(output_dir / "fold_delta_vs_p9.csv", index=False)
    bootstrap = p9.bootstrap_fold_checks(fold_delta, "p15b_cluster_cap_minus_p9", int(args.bootstrap_samples))
    bootstrap.to_csv(output_dir / "bootstrap_checks.csv", index=False)
    start_vs_p9 = p9.build_start_fold_sensitivity(nested, p9_result, "p15b_cluster_cap", "p9", horizon)
    start_vs_p9.to_csv(output_dir / "start_fold_sensitivity_vs_p9.csv", index=False)
    dropout_vs_p9 = p9.build_symbol_dropout_checks(nested, p9_result, "p15b_cluster_cap", "p9", horizon)
    dropout_vs_p9.to_csv(output_dir / "symbol_dropout_vs_p9.csv", index=False)
    cap_attr = build_cluster_cap_attribution(nested)
    cap_attr.to_csv(output_dir / "cluster_cap_attribution.csv", index=False)
    cluster_delta = build_cluster_risk_delta(nested, p9_cluster_ref)
    cluster_delta.to_csv(output_dir / "cluster_risk_delta_vs_p9.csv", index=False)
    reality = build_reality_summary(summary, cost_summary, bootstrap, start_vs_p9, dropout_vs_p9, cluster_delta)
    reality.to_csv(output_dir / "reality_summary.csv", index=False)

    plot_curves(output_dir, {"p15b_cluster_cap": nested, "p9_reference": p9_result})
    plot_cost(output_dir, cost_summary)
    plot_selected(output_dir, selected)
    plot_fold_delta(output_dir, fold_delta)
    plot_latent_diagnostics(output_dir, diagnostics)

    print("[10/10] Writing report and config metadata", flush=True)
    write_report(
        output_dir=output_dir,
        latent_summary=latent_summary,
        summary=summary,
        cost_summary=cost_summary,
        selected=selected,
        validation=validation,
        p9_validation=p9_validation,
        fold_delta=fold_delta,
        bootstrap=bootstrap,
        start_vs_p9=start_vs_p9,
        dropout_vs_p9=dropout_vs_p9,
        cap_attr=cap_attr,
        cluster_delta=cluster_delta,
        reality=reality,
    )
    update_config_stage(config_path)
    print("[done] P15 complete", flush=True)
    print(f"Done: {output_dir}", flush=True)


def build_cluster_specs() -> list[ClusterCapSpec]:
    specs: list[ClusterCapSpec] = []
    for lookback in [336, 720, 1440]:
        for cluster_k in [3, 4]:
            for cap in [0.40, 0.50]:
                specs.append(
                    ClusterCapSpec(
                        name=f"p15b_lb{lookback}_k{cluster_k}_cap{int(cap*100)}",
                        lookback_hours=lookback,
                        cluster_k=cluster_k,
                        cluster_cap=cap,
                        min_gross_retention=0.85,
                    )
                )
    return specs


def build_p15a_diagnostics(p9_result: dict[str, pd.DataFrame], residual_wide: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    positions = p9_result["positions"].copy()
    equity = p9_result["equity"][["timestamp", "fold", "period_return", "gross_period_return", "turnover"]].copy()
    timestamps = equity["timestamp"].drop_duplicates().sort_values().tolist()
    rows: list[dict[str, Any]] = []
    attr_rows: list[dict[str, Any]] = []
    pos_by_ts = {pd.Timestamp(k): v.copy() for k, v in positions.groupby("timestamp")}
    eq_by_ts = equity.set_index("timestamp")
    cache: dict[tuple[pd.Timestamp, int, int], LatentRiskModel] = {}
    for lookback in [336, 720, 1440]:
        for cluster_k in [3, 4, 5]:
            for ts in timestamps:
                timestamp = pd.Timestamp(ts)
                pos = pos_by_ts.get(timestamp)
                if pos is None or pos.empty:
                    continue
                weights = pos.set_index("symbol")["weight"].astype(float)
                model = get_risk_model(cache, residual_wide, timestamp, lookback, cluster_k)
                metrics = latent_metrics(weights, model)
                eq = eq_by_ts.loc[timestamp]
                rows.append(
                    {
                        "timestamp": timestamp,
                        "fold": int(eq["fold"]) if "fold" in eq.index else np.nan,
                        "lookback_hours": lookback,
                        "cluster_k": cluster_k,
                        "period_return": float(eq["period_return"]),
                        "gross_period_return": float(eq["gross_period_return"]),
                        "turnover": float(eq["turnover"]),
                        **metrics,
                    }
                )
                labels = pd.Series({symbol: model.cluster_by_symbol.get(symbol, 0) for symbol in weights.index})
                pos_local = pos.copy()
                pos_local["cluster_id"] = pos_local["symbol"].map(labels).fillna(0).astype(int)
                gross = float(pos_local["weight"].abs().sum())
                for cluster_id, group in pos_local.groupby("cluster_id"):
                    attr_rows.append(
                        {
                            "timestamp": timestamp,
                            "fold": int(eq["fold"]) if "fold" in eq.index else np.nan,
                            "lookback_hours": lookback,
                            "cluster_k": cluster_k,
                            "cluster_id": int(cluster_id),
                            "symbols": ",".join(sorted(group["symbol"].astype(str).tolist())),
                            "cluster_gross": float(group["weight"].abs().sum()),
                            "cluster_gross_share": float(group["weight"].abs().sum() / gross) if gross > 0 else np.nan,
                            "cluster_net_weight": float(group["weight"].sum()),
                            "cluster_contribution": float(group.get("contribution", pd.Series(0.0, index=group.index)).sum()),
                            "period_return": float(eq["period_return"]),
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(attr_rows)


def build_latent_summary(diagnostics: pd.DataFrame, cluster_attr: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (lookback, cluster_k), frame in diagnostics.groupby(["lookback_hours", "cluster_k"]):
        neg = frame[frame["period_return"] < 0]
        rows.append(
            {
                "lookback_hours": int(lookback),
                "cluster_k": int(cluster_k),
                "periods": int(len(frame)),
                "mean_max_cluster_gross_share": float(frame["max_cluster_gross_share"].mean()),
                "p90_max_cluster_gross_share": float(frame["max_cluster_gross_share"].quantile(0.90)),
                "mean_cluster_hhi": float(frame["cluster_hhi"].mean()),
                "mean_abs_pc1_exposure": float(frame["abs_pc1_exposure"].mean()),
                "mean_abs_pc2_exposure": float(frame["abs_pc2_exposure"].mean()),
                "neg_mean_max_cluster_gross_share": float(neg["max_cluster_gross_share"].mean()) if not neg.empty else np.nan,
                "neg_mean_cluster_hhi": float(neg["cluster_hhi"].mean()) if not neg.empty else np.nan,
                "corr_cluster_hhi_to_period_return": float(frame["cluster_hhi"].corr(frame["period_return"])),
                "corr_abs_pc1_to_period_return": float(frame["abs_pc1_exposure"].corr(frame["period_return"])),
            }
        )
    summary = pd.DataFrame(rows)
    if cluster_attr.empty:
        return summary
    top = (
        cluster_attr.groupby(["lookback_hours", "cluster_k", "cluster_id"])
        .agg(
            cluster_contribution_sum=("cluster_contribution", "sum"),
            mean_cluster_gross_share=("cluster_gross_share", "mean"),
            periods=("timestamp", "count"),
        )
        .reset_index()
        .sort_values("cluster_contribution_sum")
    )
    top.to_csv(DEFAULT_OUTPUT / "p15a_cluster_contribution_by_cluster.csv", index=False)
    return summary


def run_validation_search(
    panel: pd.DataFrame,
    residual_wide: pd.DataFrame,
    selections: pd.DataFrame,
    selected_p9: pd.DataFrame,
    pf_cfg: Any,
    risk_template: Any,
    spec_by_name: dict[str, Any],
    deriv_formulas: dict[str, Any],
    p3d_formulas: dict[str, Any],
    cluster_specs: list[ClusterCapSpec],
    validation_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    p9_rows: list[dict[str, Any]] = []
    selected_by_fold = selected_p9.set_index("fold")
    p9_cache: dict[tuple[int, int, int], dict[str, float]] = {}
    for _, fold in selections.sort_values("fold").iterrows():
        fold_id = int(fold["fold"])
        base = selected_by_fold.loc[fold_id]
        committee = spec_by_name[str(base["p4_spec"])]
        formula_text = p10d.prepare_fold_score(panel, base, committee, deriv_formulas, p3d_formulas)
        risk_cfg = make_fold_risk(risk_template, committee, f"p15_validation_{fold_id}")
        validation_start = max(pd.Timestamp(fold["train_start"]), pd.Timestamp(fold["train_end"]) - pd.Timedelta(days=validation_days))
        validation_end = min(
            p3d.decision_end_for_closed_positions(validation_start + pd.Timedelta(days=validation_days), pf_cfg.primary_horizon),
            pd.Timestamp(fold["train_end"]),
        )
        for spec in cluster_specs:
            key = (fold_id, spec.lookback_hours, spec.cluster_k)
            if key not in p9_cache:
                p9_equity, p9_positions, p9_metrics = run_cluster_backtest(
                    panel=panel,
                    residual_wide=residual_wide,
                    start=validation_start,
                    end=validation_end,
                    cfg=pf_cfg,
                    risk_cfg=risk_cfg,
                    committee=committee,
                    cluster_spec=None,
                    diagnostic_lookback=spec.lookback_hours,
                    diagnostic_k=spec.cluster_k,
                    score_col=p3d.SCORE_COL,
                    selected_formulas=formula_text,
                )
                enrich_cluster_metrics(p9_equity, p9_positions, p9_metrics, pf_cfg.primary_horizon)
                p9_cache[key] = p9_metrics
                p9_rows.append(
                    {
                        "fold": fold_id,
                        "lookback_hours": spec.lookback_hours,
                        "cluster_k": spec.cluster_k,
                        "validation_start": validation_start,
                        "validation_decision_end": validation_end,
                        **{f"p9_validation_{k}": v for k, v in p9_metrics.items() if p3d.is_scalar(v)},
                    }
                )
            equity, positions, metrics = run_cluster_backtest(
                panel=panel,
                residual_wide=residual_wide,
                start=validation_start,
                end=validation_end,
                cfg=pf_cfg,
                risk_cfg=risk_cfg,
                committee=committee,
                cluster_spec=spec,
                diagnostic_lookback=spec.lookback_hours,
                diagnostic_k=spec.cluster_k,
                score_col=p3d.SCORE_COL,
                selected_formulas=formula_text,
            )
            enrich_cluster_metrics(equity, positions, metrics, pf_cfg.primary_horizon)
            rows.append(
                {
                    "fold": fold_id,
                    "cluster_spec": spec.name,
                    **spec.__dict__,
                    "validation_start": validation_start,
                    "validation_decision_end": validation_end,
                    **{f"validation_{k}": v for k, v in metrics.items() if p3d.is_scalar(v)},
                }
            )
    p9_validation = pd.DataFrame(p9_rows)
    validation = annotate_validation(pd.DataFrame(rows), p9_validation)
    return validation, p9_validation


def annotate_validation(validation: pd.DataFrame, p9_validation: pd.DataFrame) -> pd.DataFrame:
    out = validation.merge(
        p9_validation.drop(columns=["validation_start", "validation_decision_end"], errors="ignore"),
        on=["fold", "lookback_hours", "cluster_k"],
        how="left",
    )
    out["delta_validation_total_return"] = out["validation_total_return"] - out["p9_validation_total_return"]
    out["delta_validation_sharpe"] = out["validation_sharpe"] - out["p9_validation_sharpe"]
    out["delta_validation_max_drawdown"] = out["validation_max_drawdown"] - out["p9_validation_max_drawdown"]
    out["delta_validation_return_20bps"] = out["validation_total_return_20bps"] - out["p9_validation_total_return_20bps"]
    out["delta_validation_return_30bps"] = out["validation_total_return_30bps"] - out["p9_validation_total_return_30bps"]
    out["validation_turnover_ratio_vs_p9"] = out["validation_mean_turnover"] / out["p9_validation_mean_turnover"].replace(0.0, np.nan)
    out["validation_gross_retention_vs_p9"] = out["validation_gross_total_return_before_cost"] / out[
        "p9_validation_gross_total_return_before_cost"
    ].replace(0.0, np.nan)
    out["validation_cluster_hhi_reduction_vs_p9"] = 1.0 - out["validation_mean_cluster_hhi"] / out[
        "p9_validation_mean_cluster_hhi"
    ].replace(0.0, np.nan)
    out["validation_max_cluster_share_reduction_vs_p9"] = 1.0 - out["validation_mean_max_cluster_gross_share"] / out[
        "p9_validation_mean_max_cluster_gross_share"
    ].replace(0.0, np.nan)
    out["hard_accept"] = (
        (out["delta_validation_total_return"] >= -0.005)
        & (out["delta_validation_sharpe"] >= -0.10)
        & (out["validation_max_drawdown"] >= 1.05 * out["p9_validation_max_drawdown"])
        & (out["delta_validation_return_20bps"] >= -0.002)
        & (out["delta_validation_return_30bps"] >= -0.002)
        & (out["validation_turnover_ratio_vs_p9"] <= 1.05)
        & (out["validation_gross_retention_vs_p9"] >= 0.85)
        & (out["validation_cluster_hhi_reduction_vs_p9"] >= 0.15)
        & (out["validation_mean_cluster_cap_success"] >= 0.95)
        & (out["validation_mean_abs_beta_market"] <= 1e-8)
    )
    out["selection_score"] = (
        100.0 * out["delta_validation_total_return"].fillna(-1.0)
        + 1.5 * out["delta_validation_sharpe"].fillna(-5.0)
        + 80.0 * out["delta_validation_return_20bps"].fillna(-1.0)
        + 80.0 * out["delta_validation_return_30bps"].fillna(-1.0)
        + 8.0 * out["validation_cluster_hhi_reduction_vs_p9"].clip(-1.0, 1.0).fillna(0.0)
        + 4.0 * out["validation_max_cluster_share_reduction_vs_p9"].clip(-1.0, 1.0).fillna(0.0)
        - 2.0 * (out["validation_turnover_ratio_vs_p9"].fillna(1.0) - 1.0)
    )
    return out


def select_cluster_spec_by_fold(validation: pd.DataFrame, p9_validation: pd.DataFrame, selections: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    by_fold = {int(k): v.copy() for k, v in validation.groupby("fold")}
    p9_by_fold = p9_validation.groupby("fold").first()
    for _, fold in selections.sort_values("fold").iterrows():
        fold_id = int(fold["fold"])
        accepted = by_fold.get(fold_id, pd.DataFrame())
        accepted = accepted[accepted["hard_accept"].astype(bool)].copy() if not accepted.empty else pd.DataFrame()
        if accepted.empty:
            p9_row = p9_by_fold.loc[fold_id]
            rows.append(
                {
                    "fold": fold_id,
                    "selected_cluster_spec": NO_CLUSTER_CAP,
                    "fallback_to_p9_sizing": True,
                    "lookback_hours": int(p9_row["lookback_hours"]),
                    "cluster_k": int(p9_row["cluster_k"]),
                    "cluster_cap": np.nan,
                    "min_gross_retention": np.nan,
                    "validation_selection_score": float(p9_row.get("p9_validation_total_return", np.nan)),
                    "validation_total_return": float(p9_row.get("p9_validation_total_return", np.nan)),
                    "validation_sharpe": float(p9_row.get("p9_validation_sharpe", np.nan)),
                    "validation_mean_cluster_hhi": float(p9_row.get("p9_validation_mean_cluster_hhi", np.nan)),
                    "validation_cluster_hhi_reduction_vs_p9": 0.0,
                }
            )
            continue
        winner = accepted.sort_values(["selection_score", "delta_validation_return_20bps"], ascending=False).iloc[0]
        rows.append(
            {
                "fold": fold_id,
                "selected_cluster_spec": str(winner["cluster_spec"]),
                "fallback_to_p9_sizing": False,
                "lookback_hours": int(winner["lookback_hours"]),
                "cluster_k": int(winner["cluster_k"]),
                "cluster_cap": float(winner["cluster_cap"]),
                "min_gross_retention": float(winner["min_gross_retention"]),
                "validation_selection_score": float(winner["selection_score"]),
                "validation_total_return": float(winner["validation_total_return"]),
                "validation_sharpe": float(winner["validation_sharpe"]),
                "validation_mean_cluster_hhi": float(winner["validation_mean_cluster_hhi"]),
                "validation_cluster_hhi_reduction_vs_p9": float(winner["validation_cluster_hhi_reduction_vs_p9"]),
            }
        )
    return pd.DataFrame(rows)


def replay_selected_cluster_specs(
    panel: pd.DataFrame,
    residual_wide: pd.DataFrame,
    selections: pd.DataFrame,
    selected: pd.DataFrame,
    selected_p9: pd.DataFrame,
    pf_cfg: Any,
    risk_template: Any,
    spec_by_name: dict[str, Any],
    deriv_formulas: dict[str, Any],
    p3d_formulas: dict[str, Any],
    cluster_specs: list[ClusterCapSpec],
    force_baseline: bool,
) -> dict[str, Any]:
    spec_map = {spec.name: spec for spec in cluster_specs}
    selected_exec = selected.set_index("fold")
    selected_p9_by_fold = selected_p9.set_index("fold")
    fold_rows: list[dict[str, Any]] = []
    equity_parts: list[pd.DataFrame] = []
    position_parts: list[pd.DataFrame] = []
    for _, fold in selections.sort_values("fold").iterrows():
        fold_id = int(fold["fold"])
        base = selected_p9_by_fold.loc[fold_id]
        decision = selected_exec.loc[fold_id]
        selected_name = str(decision["selected_cluster_spec"])
        selected_spec = None if bool(decision["fallback_to_p9_sizing"]) else spec_map[selected_name]
        exec_spec = None if force_baseline else selected_spec
        lookback = int(decision["lookback_hours"])
        cluster_k = int(decision["cluster_k"])
        committee = spec_by_name[str(base["p4_spec"])]
        formula_text = p10d.prepare_fold_score(panel, base, committee, deriv_formulas, p3d_formulas)
        risk_cfg = make_fold_risk(risk_template, committee, f"p15_{selected_name}_{fold_id}")
        start = pd.Timestamp(fold["test_start"])
        end = p3d.decision_end_for_closed_positions(pd.Timestamp(fold["test_end"]), pf_cfg.primary_horizon)
        equity, positions, metrics = run_cluster_backtest(
            panel=panel,
            residual_wide=residual_wide,
            start=start,
            end=end,
            cfg=pf_cfg,
            risk_cfg=risk_cfg,
            committee=committee,
            cluster_spec=exec_spec,
            diagnostic_lookback=lookback,
            diagnostic_k=cluster_k,
            score_col=p3d.SCORE_COL,
            selected_formulas=formula_text,
        )
        enrich_cluster_metrics(equity, positions, metrics, pf_cfg.primary_horizon)
        fold_rows.append(
            {
                "fold": fold_id,
                "train_start": fold["train_start"],
                "train_end": fold["train_end"],
                "test_start": fold["test_start"],
                "test_end": fold["test_end"],
                "test_decision_end": end,
                "p4_spec": committee.name,
                "p3d_formula_name": str(base["p3d_formula_name"]),
                "deriv_formula_name": str(base["deriv_formula_name"]),
                "formula": formula_text,
                "selected_cluster_spec": NO_CLUSTER_CAP if exec_spec is None else exec_spec.name,
                "validation_selected_cluster_spec": selected_name,
                "fallback_to_p9_sizing": bool(exec_spec is None),
                "validation_selection_score": float(decision["validation_selection_score"]),
                **metrics,
            }
        )
        if not equity.empty:
            equity = equity.copy()
            positions = positions.copy()
            equity["fold"] = fold_id
            equity["p4_spec"] = committee.name
            equity["formula_name"] = str(base["deriv_formula_name"])
            equity["formula"] = formula_text
            equity["selected_cluster_spec"] = NO_CLUSTER_CAP if exec_spec is None else exec_spec.name
            equity["validation_selected_cluster_spec"] = selected_name
            positions["fold"] = fold_id
            positions["p4_spec"] = committee.name
            positions["formula_name"] = str(base["deriv_formula_name"])
            positions["formula"] = formula_text
            positions["selected_cluster_spec"] = NO_CLUSTER_CAP if exec_spec is None else exec_spec.name
            positions["validation_selected_cluster_spec"] = selected_name
            equity_parts.append(equity)
            position_parts.append(positions)
    equity_curve = pd.concat(equity_parts, ignore_index=True) if equity_parts else pd.DataFrame()
    positions = pd.concat(position_parts, ignore_index=True) if position_parts else pd.DataFrame()
    folds = pd.DataFrame(fold_rows)
    metrics = p3d.summarize_nested_result({"equity": equity_curve, "positions": positions, "folds": folds}, pf_cfg.rebalance_hours)
    enrich_cluster_metrics(equity_curve, positions, metrics, pf_cfg.primary_horizon)
    return {"folds": folds, "equity": equity_curve, "positions": positions, "metrics": metrics}


def run_cluster_backtest(
    panel: pd.DataFrame,
    residual_wide: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cfg: Any,
    risk_cfg: Any,
    committee: Any,
    cluster_spec: ClusterCapSpec | None,
    diagnostic_lookback: int,
    diagnostic_k: int,
    score_col: str,
    selected_formulas: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    label_col = f"label_resid_ret_h{cfg.primary_horizon}"
    ret_col = f"fwd_simple_ret_h{cfg.primary_horizon}"
    symbols = sorted(panel["symbol"].unique())
    mask = (panel["timestamp"] >= start) & (panel["timestamp"] < end)
    timestamps = np.array(sorted(panel.loc[mask, "timestamp"].unique()))[:: cfg.rebalance_hours]
    prev_weights = pd.Series(0.0, index=symbols)
    current_state: dict[str, tuple[int, int]] = {}
    model_cache: dict[tuple[pd.Timestamp, int, int], LatentRiskModel] = {}
    equity = 1.0
    rows_out: list[dict[str, Any]] = []
    pos_out: list[dict[str, Any]] = []

    for ts in timestamps:
        timestamp = pd.Timestamp(ts)
        rows = panel[panel["timestamp"] == timestamp].copy()
        rows = rows[
            np.isfinite(rows[score_col])
            & np.isfinite(rows[p3d.P4_SCORE_COL])
            & np.isfinite(rows[ret_col])
            & np.isfinite(rows[label_col])
        ]
        if len(rows) < cfg.long_k + cfg.short_k:
            continue
        rows["score_rank"] = rows[score_col].rank(pct=True, method="average")
        rows["p4_rank"] = rows[p3d.P4_SCORE_COL].rank(pct=True, method="average")
        rows["ema_rank"] = rows["score_ema_025"].rank(pct=True, method="average")
        rows["base_rank"] = rows["score_baseline"].rank(pct=True, method="average")
        rows = rows.sort_values(score_col, ascending=False)
        rows_by_symbol = rows.set_index("symbol")

        target_sides = p3d.select_p4_gated_target_sides(rows, current_state, cfg, score_col)
        before_guard = len([side for side in target_sides.values() if side < 0])
        target_sides = apply_short_guard(target_sides, rows_by_symbol, committee)
        after_guard = len([side for side in target_sides.values() if side < 0])
        if not any(side > 0 for side in target_sides.values()) or not any(side < 0 for side in target_sides.values()):
            continue

        selected = rows_by_symbol.reindex(list(target_sides)).reset_index()
        raw_weights = side_fraction_weights(target_sides, cfg.gross_exposure, committee.long_fraction)
        baseline_weights, baseline_info = p3d.build_weights(raw_weights, selected, rows_by_symbol, cfg, risk_cfg)
        baseline_weights = baseline_weights[baseline_weights.abs() > 1e-12]
        if baseline_weights.abs().sum() <= 0:
            continue

        lookback = diagnostic_lookback if cluster_spec is None else cluster_spec.lookback_hours
        cluster_k = diagnostic_k if cluster_spec is None else cluster_spec.cluster_k
        model = get_risk_model(model_cache, residual_wide, timestamp, lookback, cluster_k)
        baseline_latent = latent_metrics(baseline_weights, model)
        if cluster_spec is None:
            weights = baseline_weights
            cap_info = cluster_cap_info(
                success=True,
                active=False,
                attempts=0,
                exposure_error=float(baseline_info.get("optimizer_exposure_error", np.nan)),
                before=baseline_latent,
                after=baseline_latent,
                cluster_cap=np.nan,
            )
            cap_info.update({k: v for k, v in baseline_info.items() if k.startswith("optimizer_")})
        else:
            weights, cap_info = apply_cluster_cap_weights(
                baseline_weights=baseline_weights,
                raw_weights=raw_weights,
                rows_by_symbol=rows_by_symbol,
                model=model,
                cfg=cfg,
                risk_cfg=risk_cfg,
                spec=cluster_spec,
                baseline_latent=baseline_latent,
            )
        weights = weights[weights.abs() > 1e-12]
        if weights.abs().sum() <= 0:
            continue
        latent = latent_metrics(weights, model)

        target_full_weights = pd.Series(0.0, index=symbols)
        target_full_weights.loc[weights.index] = weights
        baseline_full = pd.Series(0.0, index=symbols)
        baseline_full.loc[baseline_weights.index] = baseline_weights
        turnover = float((target_full_weights - prev_weights).abs().sum())
        target_turnover = float((baseline_full - prev_weights).abs().sum())
        cost = turnover * (cfg.one_way_fee_bps + cfg.one_way_slippage_bps) / 10_000.0
        active_rows = rows_by_symbol.reindex(weights.index)
        gross_period_ret = float((weights * active_rows[ret_col]).sum())
        baseline_rows = rows_by_symbol.reindex(baseline_weights.index)
        baseline_gross_ret = float((baseline_weights * baseline_rows[ret_col]).sum())
        period_ret = gross_period_ret - cost
        equity *= 1.0 + period_ret

        rank_ic = spearman_ic(rows[score_col], rows[label_col])
        long_labels = active_rows.loc[weights[weights > 0].index, label_col]
        short_labels = active_rows.loc[weights[weights < 0].index, label_col]
        top_bottom = float(long_labels.mean() - short_labels.mean()) if len(long_labels) and len(short_labels) else np.nan
        exposures = portfolio_exposures(weights, active_rows, cfg.neutralize_exposures)
        rows_out.append(
            {
                "timestamp": timestamp,
                "exit_timestamp": timestamp + pd.Timedelta(hours=cfg.execution_delay_bars + cfg.primary_horizon),
                "period_return": period_ret,
                "gross_period_return": gross_period_ret,
                "baseline_target_gross_period_return": baseline_gross_ret,
                "cost_return": cost,
                "equity": equity,
                "turnover": turnover,
                "target_turnover": target_turnover,
                "turnover_saved": max(0.0, target_turnover - turnover),
                "gross_exposure": float(weights.abs().sum()),
                "target_gross_exposure": float(baseline_weights.abs().sum()),
                "gross_exposure_retention": float(weights.abs().sum() / baseline_weights.abs().sum()),
                "rank_ic": rank_ic,
                "top_bottom_label_spread": top_bottom,
                "abs_beta_btc": abs(exposures.get("beta_btc", 0.0)),
                "abs_beta_eth": abs(exposures.get("beta_eth", 0.0)),
                "abs_beta_market": abs(exposures.get("beta_market", 0.0)),
                "symbol_concentration": float((weights**2).sum()),
                "max_abs_weight": float(weights.abs().max()),
                "delay_tau_days": 0.0,
                "delay_retention": 0.0,
                "delay_new_target_weight": 1.0,
                "short_guard_removed": float(before_guard - after_guard),
                "target_applied": 1.0,
                "cluster_lookback_hours": lookback,
                "cluster_k": cluster_k,
                "selected_cluster_spec": NO_CLUSTER_CAP if cluster_spec is None else cluster_spec.name,
                **latent,
                **cap_info,
                "selected_formulas": selected_formulas,
            }
        )
        labels = pd.Series({symbol: model.cluster_by_symbol.get(symbol, 0) for symbol in weights.index})
        for symbol, weight in weights.items():
            row = rows_by_symbol.loc[symbol]
            baseline_weight = float(baseline_full.loc[symbol])
            pos_out.append(
                {
                    "timestamp": timestamp,
                    "exit_timestamp": timestamp + pd.Timedelta(hours=cfg.execution_delay_bars + cfg.primary_horizon),
                    "symbol": symbol,
                    "weight": float(weight),
                    "target_weight": float(weight),
                    "baseline_target_weight": baseline_weight,
                    "score": float(row[score_col]),
                    "p4_score": float(row[p3d.P4_SCORE_COL]),
                    "candidate_rank": float(row[p3d.CANDIDATE_RANK_COL]) if p3d.CANDIDATE_RANK_COL in row.index else np.nan,
                    "p4_rank": float(row["p4_rank"]),
                    "ema_rank": float(row["ema_rank"]),
                    "base_rank": float(row["base_rank"]),
                    "rank_a07": float(row["rank_a07"]),
                    "rank_b11": float(row["rank_b11"]),
                    "rank_c09": float(row["rank_c09"]),
                    "cluster_id": int(labels.loc[symbol]),
                    "fwd_simple_ret": float(row[ret_col]),
                    "label_resid_ret": float(row[label_col]),
                    "contribution": float(weight * row[ret_col]),
                    "baseline_target_contribution": float(baseline_weight * row[ret_col]),
                    "selected_score_cols": "p4_score | candidate_rerank_score",
                }
            )
        prev_weights = target_full_weights
        current_state = update_state(current_state, target_sides, True)

    equity_curve = pd.DataFrame(rows_out)
    positions = pd.DataFrame(pos_out)
    metrics = p3d.aggregate_oos_metrics(equity_curve, cfg.rebalance_hours)
    return equity_curve, positions, metrics


def get_risk_model(
    cache: dict[tuple[pd.Timestamp, int, int], LatentRiskModel],
    residual_wide: pd.DataFrame,
    timestamp: pd.Timestamp,
    lookback: int,
    cluster_k: int,
) -> LatentRiskModel:
    key = (pd.Timestamp(timestamp), int(lookback), int(cluster_k))
    if key not in cache:
        cache[key] = compute_risk_model(residual_wide, pd.Timestamp(timestamp), int(lookback), int(cluster_k))
    return cache[key]


def compute_risk_model(residual_wide: pd.DataFrame, timestamp: pd.Timestamp, lookback: int, cluster_k: int) -> LatentRiskModel:
    symbols = list(residual_wide.columns)
    hist = residual_wide.loc[residual_wide.index < timestamp, symbols].tail(int(lookback)).copy()
    min_obs = max(72, min(int(lookback) // 4, 240))
    if len(hist) < min_obs:
        labels = {symbol: int(index % max(1, cluster_k)) + 1 for index, symbol in enumerate(symbols)}
        pcs = np.eye(len(symbols), min(3, len(symbols)))
        corr = np.eye(len(symbols))
        return LatentRiskModel(symbols=symbols, cluster_by_symbol=labels, pcs=pcs, corr=corr, observations=int(len(hist)))
    hist = hist.fillna(0.0)
    values = hist.to_numpy(dtype=float)
    corr = np.corrcoef(values, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.clip((corr + corr.T) / 2.0, -0.999, 0.999)
    np.fill_diagonal(corr, 1.0)
    distance = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, 1.0))
    try:
        condensed = squareform(distance, checks=False)
        link = linkage(condensed, method="average")
        raw_labels = fcluster(link, int(cluster_k), criterion="maxclust")
    except Exception:
        raw_labels = np.array([index % max(1, cluster_k) + 1 for index in range(len(symbols))])
    cov = np.cov(values, rowvar=False)
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    cov = (cov + cov.T) / 2.0
    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        pcs = eigvecs[:, order[: min(3, len(symbols))]]
    except np.linalg.LinAlgError:
        pcs = np.eye(len(symbols), min(3, len(symbols)))
    return LatentRiskModel(
        symbols=symbols,
        cluster_by_symbol={symbol: int(label) for symbol, label in zip(symbols, raw_labels)},
        pcs=np.asarray(pcs, dtype=float),
        corr=corr,
        observations=int(len(hist)),
    )


def latent_metrics(weights: pd.Series, model: LatentRiskModel) -> dict[str, float]:
    weights = weights.astype(float)
    gross = float(weights.abs().sum())
    if gross <= 1e-12:
        return {
            "max_cluster_gross_share": np.nan,
            "cluster_hhi": np.nan,
            "active_cluster_count": 0.0,
            "pc1_exposure": np.nan,
            "pc2_exposure": np.nan,
            "pc3_exposure": np.nan,
            "abs_pc1_exposure": np.nan,
            "abs_pc2_exposure": np.nan,
            "abs_pc3_exposure": np.nan,
        }
    labels = pd.Series({symbol: model.cluster_by_symbol.get(symbol, 0) for symbol in weights.index})
    gross_by_cluster = weights.abs().groupby(labels).sum()
    shares = gross_by_cluster / gross
    aligned = weights.reindex(model.symbols).fillna(0.0).to_numpy(dtype=float)
    pc_exposures = []
    for index in range(3):
        if model.pcs.shape[1] > index:
            pc_exposures.append(float(aligned @ model.pcs[:, index] / gross))
        else:
            pc_exposures.append(np.nan)
    return {
        "max_cluster_gross_share": float(shares.max()),
        "cluster_hhi": float((shares**2).sum()),
        "active_cluster_count": float((gross_by_cluster > 1e-12).sum()),
        "pc1_exposure": pc_exposures[0],
        "pc2_exposure": pc_exposures[1],
        "pc3_exposure": pc_exposures[2],
        "abs_pc1_exposure": abs(pc_exposures[0]) if np.isfinite(pc_exposures[0]) else np.nan,
        "abs_pc2_exposure": abs(pc_exposures[1]) if np.isfinite(pc_exposures[1]) else np.nan,
        "abs_pc3_exposure": abs(pc_exposures[2]) if np.isfinite(pc_exposures[2]) else np.nan,
    }


def apply_cluster_cap_weights(
    baseline_weights: pd.Series,
    raw_weights: pd.Series,
    rows_by_symbol: pd.DataFrame,
    model: LatentRiskModel,
    cfg: Any,
    risk_cfg: Any,
    spec: ClusterCapSpec,
    baseline_latent: dict[str, float],
) -> tuple[pd.Series, dict[str, float]]:
    if baseline_latent["max_cluster_gross_share"] <= spec.cluster_cap + 1e-10:
        info = cluster_cap_info(True, False, 0, 0.0, baseline_latent, baseline_latent, spec.cluster_cap)
        info.update({"optimizer_success": 1.0, "optimizer_exposure_error": 0.0, "optimizer_attempts": 0.0})
        return baseline_weights, info
    symbols = list(baseline_weights.index)
    selected = rows_by_symbol.loc[symbols]
    signs = np.sign(raw_weights.reindex(symbols).fillna(baseline_weights).to_numpy(dtype=float))
    if not np.any(signs > 0) or not np.any(signs < 0):
        info = cluster_cap_info(False, True, 0, np.nan, baseline_latent, baseline_latent, spec.cluster_cap)
        info.update({"optimizer_success": 0.0, "optimizer_exposure_error": np.nan, "optimizer_attempts": 0.0})
        return baseline_weights, info
    baseline = baseline_weights.reindex(symbols).to_numpy(dtype=float)
    gross_limit = float(np.sum(np.abs(baseline)))
    max_abs = float(risk_cfg.max_symbol_weight or cfg.gross_exposure)
    bounds = [(0.0, max_abs) if sign > 0 else (-max_abs, 0.0) for sign in signs]
    exposure_matrix = exposure_matrix_for(selected, cfg.neutralize_exposures)
    cluster_ids = np.array([model.cluster_by_symbol.get(symbol, 0) for symbol in symbols], dtype=int)
    unique_clusters = sorted(set(cluster_ids.tolist()))

    def objective(x: np.ndarray) -> float:
        return float(np.sum((x - baseline) ** 2) + 1e-5 * np.sum(x**2))

    def gradient(x: np.ndarray) -> np.ndarray:
        return 2.0 * (x - baseline) + 2e-5 * x

    constraints: list[dict[str, Any]] = [
        {"type": "eq", "fun": lambda x, a=exposure_matrix: a @ x, "jac": lambda x, a=exposure_matrix: a},
        {"type": "ineq", "fun": lambda x, s=signs, limit=gross_limit: float(limit - s @ x), "jac": lambda x, s=signs: -s},
        {
            "type": "ineq",
            "fun": lambda x, s=signs, floor=spec.min_gross_retention, limit=gross_limit: float(s @ x - floor * limit),
            "jac": lambda x, s=signs: s,
        },
    ]
    for cluster_id in unique_clusters:
        mask = (cluster_ids == cluster_id).astype(float)
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda x, m=mask, s=signs, cap=spec.cluster_cap: float(cap * (s @ x) - np.sum(m * s * x)),
                "jac": lambda x, m=mask, s=signs, cap=spec.cluster_cap: cap * s - m * s,
            }
        )
    result = minimize(
        objective,
        baseline,
        jac=gradient,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 120, "ftol": 1e-9, "disp": False},
    )
    x = result.x if result.x is not None else baseline
    exposure_error = max_constraint_error(exposure_matrix, x)
    gross = float(signs @ x)
    bounds_ok = all(value >= lo - 1e-8 and value <= hi + 1e-8 for value, (lo, hi) in zip(x, bounds))
    after = latent_metrics(pd.Series(x, index=symbols), model)
    constraints_ok = (
        exposure_error <= 1e-7
        and gross <= gross_limit + 1e-7
        and gross >= spec.min_gross_retention * gross_limit - 1e-7
        and after["max_cluster_gross_share"] <= spec.cluster_cap + 1e-6
        and bounds_ok
    )
    if not constraints_ok:
        x = baseline
        after = baseline_latent
    weights = pd.Series(x, index=symbols)
    info = cluster_cap_info(
        bool(constraints_ok),
        True,
        1,
        exposure_error,
        baseline_latent,
        after,
        spec.cluster_cap,
    )
    info.update(
        {
            "optimizer_success": float(constraints_ok),
            "optimizer_exposure_error": float(exposure_error),
            "optimizer_attempts": 1.0,
        }
    )
    return weights, info


def exposure_matrix_for(selected: pd.DataFrame, exposure_cols: list[str]) -> np.ndarray:
    arrays = []
    for col in exposure_cols:
        if col == "intercept":
            arrays.append(np.ones(len(selected)))
        elif col in selected.columns:
            arrays.append(selected[col].fillna(0.0).to_numpy(dtype=float))
    return np.vstack(arrays) if arrays else np.empty((0, len(selected)))


def max_constraint_error(exposure_matrix: np.ndarray, weights: np.ndarray) -> float:
    if exposure_matrix.size == 0:
        return 0.0
    return float(np.max(np.abs(exposure_matrix @ weights)))


def cluster_cap_info(
    success: bool,
    active: bool,
    attempts: int,
    exposure_error: float,
    before: dict[str, float],
    after: dict[str, float],
    cluster_cap: float,
) -> dict[str, float]:
    return {
        "cluster_cap_success": float(success),
        "cluster_cap_active": float(active),
        "cluster_cap_attempts": float(attempts),
        "cluster_cap": float(cluster_cap),
        "cluster_cap_exposure_error": float(exposure_error),
        "baseline_max_cluster_gross_share": float(before.get("max_cluster_gross_share", np.nan)),
        "baseline_cluster_hhi": float(before.get("cluster_hhi", np.nan)),
        "cluster_share_reduction": float(before.get("max_cluster_gross_share", np.nan) - after.get("max_cluster_gross_share", np.nan)),
        "cluster_hhi_reduction_period": float(before.get("cluster_hhi", np.nan) - after.get("cluster_hhi", np.nan)),
    }


def enrich_cluster_metrics(equity: pd.DataFrame, positions: pd.DataFrame, metrics: dict[str, float], horizon: int) -> None:
    p3d.add_gross_metrics(equity, metrics)
    p3d.add_risk_event_metrics(positions, metrics)
    p3d.add_delay_metrics(equity, metrics)
    metrics.update(p3d.cost_adjusted_metrics(equity, horizon, [20.0, 30.0]))
    metrics["q25_segment_sharpe"] = p3d.q25_segment_sharpe(equity, horizon)
    metrics["long_short_balance"] = p3d.long_short_balance(positions)
    metrics["short_abs_contribution_share"] = p3d.short_abs_contribution_share(positions)
    if equity.empty:
        return
    target_turnover = float(equity["target_turnover"].sum()) if "target_turnover" in equity else np.nan
    turnover = float(equity["turnover"].sum()) if "turnover" in equity else np.nan
    metrics["mean_target_turnover"] = float(equity["target_turnover"].mean()) if "target_turnover" in equity else np.nan
    metrics["total_target_turnover"] = target_turnover
    metrics["mean_turnover_saved"] = float(equity["turnover_saved"].mean()) if "turnover_saved" in equity else np.nan
    metrics["total_turnover_saved"] = float(equity["turnover_saved"].sum()) if "turnover_saved" in equity else np.nan
    metrics["turnover_reduction"] = 1.0 - turnover / target_turnover if target_turnover and target_turnover > 0 else 0.0
    for col in [
        "max_cluster_gross_share",
        "cluster_hhi",
        "active_cluster_count",
        "abs_pc1_exposure",
        "abs_pc2_exposure",
        "abs_pc3_exposure",
        "baseline_max_cluster_gross_share",
        "baseline_cluster_hhi",
        "cluster_cap_success",
        "cluster_cap_active",
        "gross_exposure_retention",
    ]:
        if col in equity:
            metrics[f"mean_{col}"] = float(equity[col].mean())
    if "cluster_hhi" in equity and "baseline_cluster_hhi" in equity:
        base = float(equity["baseline_cluster_hhi"].mean())
        metrics["cluster_hhi_reduction_vs_baseline"] = 1.0 - float(equity["cluster_hhi"].mean()) / base if base > 0 else 0.0
    if "max_cluster_gross_share" in equity and "baseline_max_cluster_gross_share" in equity:
        base = float(equity["baseline_max_cluster_gross_share"].mean())
        metrics["max_cluster_share_reduction_vs_baseline"] = 1.0 - float(equity["max_cluster_gross_share"].mean()) / base if base > 0 else 0.0


def build_p15_summary(results: dict[str, dict[str, Any]], horizon: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    p9_ref = p3d.summarize_nested_result(results["p9_reference"], horizon)
    for name, result in results.items():
        row = p3d.summarize_nested_result(result, horizon)
        enrich_cluster_metrics(result["equity"], result["positions"], row, horizon)
        row["variant"] = name
        row["delta_vs_p9_total_return"] = row["total_return"] - p9_ref["total_return"]
        row["delta_vs_p9_gross_total_return_before_cost"] = row["gross_total_return_before_cost"] - p9_ref["gross_total_return_before_cost"]
        row["delta_vs_p9_sharpe"] = row["sharpe"] - p9_ref["sharpe"]
        row["delta_vs_p9_max_drawdown"] = row["max_drawdown"] - p9_ref["max_drawdown"]
        row["worst_5_fold_return_sum"] = float(result["folds"]["total_return"].sort_values().head(5).sum()) if "total_return" in result["folds"] else np.nan
        row["median_fold_return"] = float(result["folds"]["total_return"].median()) if "total_return" in result["folds"] else np.nan
        row["q25_fold_return"] = float(result["folds"]["total_return"].quantile(0.25)) if "total_return" in result["folds"] else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["total_return", "sharpe"], ascending=False)


def build_cluster_cap_attribution(result: dict[str, Any]) -> pd.DataFrame:
    eq = result["equity"].copy()
    if eq.empty:
        return pd.DataFrame()
    return (
        eq.groupby("selected_cluster_spec")
        .agg(
            periods=("timestamp", "count"),
            period_return_sum=("period_return", "sum"),
            gross_return_sum=("gross_period_return", "sum"),
            cost_sum=("cost_return", "sum"),
            turnover_sum=("turnover", "sum"),
            mean_cluster_hhi=("cluster_hhi", "mean"),
            mean_baseline_cluster_hhi=("baseline_cluster_hhi", "mean"),
            mean_max_cluster_gross_share=("max_cluster_gross_share", "mean"),
            mean_baseline_max_cluster_gross_share=("baseline_max_cluster_gross_share", "mean"),
            mean_cluster_cap_success=("cluster_cap_success", "mean"),
            mean_gross_exposure_retention=("gross_exposure_retention", "mean"),
        )
        .reset_index()
        .sort_values("period_return_sum", ascending=False)
    )


def build_cluster_risk_delta(candidate: dict[str, Any], reference: dict[str, Any]) -> pd.DataFrame:
    cand = candidate["equity"].copy()
    ref = reference["equity"].copy()
    if cand.empty or ref.empty:
        return pd.DataFrame()
    cols = ["timestamp", "cluster_hhi", "max_cluster_gross_share", "abs_pc1_exposure", "abs_pc2_exposure", "gross_exposure"]
    merged = cand[cols].merge(ref[cols], on="timestamp", suffixes=("_p15b", "_p9"))
    rows = []
    for metric in ["cluster_hhi", "max_cluster_gross_share", "abs_pc1_exposure", "abs_pc2_exposure", "gross_exposure"]:
        c = f"{metric}_p15b"
        r = f"{metric}_p9"
        rows.append(
            {
                "metric": metric,
                "p15b_mean": float(merged[c].mean()),
                "p9_mean": float(merged[r].mean()),
                "delta": float(merged[c].mean() - merged[r].mean()),
                "reduction_pct": float(1.0 - merged[c].mean() / merged[r].mean()) if float(merged[r].mean()) != 0 else np.nan,
                "periods": int(len(merged)),
            }
        )
    return pd.DataFrame(rows)


def build_reality_summary(
    summary: pd.DataFrame,
    cost: pd.DataFrame,
    bootstrap: pd.DataFrame,
    start_vs_p9: pd.DataFrame,
    dropout_vs_p9: pd.DataFrame,
    cluster_delta: pd.DataFrame,
) -> pd.DataFrame:
    cand = summary[summary["variant"] == "p15b_cluster_cap"].iloc[0]
    p9_row = summary[summary["variant"] == "p9_reference"].iloc[0]
    c20 = cost[(cost["variant"] == "p15b_cluster_cap") & (cost["one_way_cost_bps"] == 20.0)].iloc[0]
    p20 = cost[(cost["variant"] == "p9_reference") & (cost["one_way_cost_bps"] == 20.0)].iloc[0]
    c30 = cost[(cost["variant"] == "p15b_cluster_cap") & (cost["one_way_cost_bps"] == 30.0)].iloc[0]
    p30 = cost[(cost["variant"] == "p9_reference") & (cost["one_way_cost_bps"] == 30.0)].iloc[0]
    boot = bootstrap.iloc[0]
    start_positive = float((start_vs_p9["delta_total_return"] > 0).mean()) if "delta_total_return" in start_vs_p9 else np.nan
    dropout_positive = float((dropout_vs_p9["delta_total_return"] > 0).mean()) if "delta_total_return" in dropout_vs_p9 else np.nan
    hhi_delta = cluster_delta[cluster_delta["metric"] == "cluster_hhi"].iloc[0] if not cluster_delta.empty else pd.Series(dtype=float)
    max_share_delta = cluster_delta[cluster_delta["metric"] == "max_cluster_gross_share"].iloc[0] if not cluster_delta.empty else pd.Series(dtype=float)
    gross_retention = cand["gross_total_return_before_cost"] / p9_row["gross_total_return_before_cost"]
    return pd.DataFrame(
        [
            {"check": "net_not_worse_than_p9_minus_50bps", "value": cand["total_return"] - p9_row["total_return"], "pass": bool(cand["total_return"] >= p9_row["total_return"] - 0.005)},
            {"check": "sharpe_not_worse_than_p9_minus_010", "value": cand["sharpe"] - p9_row["sharpe"], "pass": bool(cand["sharpe"] >= p9_row["sharpe"] - 0.10)},
            {"check": "maxdd_not_worse_than_105x_p9", "value": cand["max_drawdown"] - p9_row["max_drawdown"], "pass": bool(cand["max_drawdown"] >= 1.05 * p9_row["max_drawdown"])},
            {"check": "return_20bps_not_worse_than_p9", "value": c20["total_return"] - p20["total_return"], "pass": bool(c20["total_return"] >= p20["total_return"] - 0.002)},
            {"check": "return_30bps_not_worse_than_p9", "value": c30["total_return"] - p30["total_return"], "pass": bool(c30["total_return"] >= p30["total_return"] - 0.002)},
            {"check": "turnover_not_higher_than_105pct_p9", "value": cand["mean_turnover"] / p9_row["mean_turnover"], "pass": bool(cand["mean_turnover"] <= 1.05 * p9_row["mean_turnover"])},
            {"check": "gross_retention_85pct", "value": gross_retention, "pass": bool(gross_retention >= 0.85)},
            {"check": "cluster_hhi_reduction_15pct", "value": hhi_delta.get("reduction_pct", np.nan), "pass": bool(hhi_delta.get("reduction_pct", np.nan) >= 0.15)},
            {"check": "max_cluster_share_reduction_positive", "value": max_share_delta.get("reduction_pct", np.nan), "pass": bool(max_share_delta.get("reduction_pct", np.nan) > 0.0)},
            {"check": "worst5_delta_vs_p9", "value": cand["worst_5_fold_return_sum"] - p9_row["worst_5_fold_return_sum"], "pass": bool(cand["worst_5_fold_return_sum"] >= p9_row["worst_5_fold_return_sum"])},
            {"check": "bootstrap_delta_vs_p9_q05", "value": boot["bootstrap_q05"], "pass": bool(boot["bootstrap_q05"] > -0.0025)},
            {"check": "start_fold_positive_vs_p9", "value": start_positive, "pass": bool(start_positive >= 0.45)},
            {"check": "symbol_dropout_positive_vs_p9", "value": dropout_positive, "pass": bool(dropout_positive >= 0.45)},
            {"check": "strict_mean_abs_market_beta", "value": cand["mean_abs_beta_market"], "pass": bool(cand["mean_abs_beta_market"] < 1e-8)},
            {"check": "strict_max_abs_market_beta", "value": cand["max_abs_beta_market"], "pass": bool(cand["max_abs_beta_market"] < 1e-7)},
        ]
    )


def plot_curves(output_dir: Path, results: dict[str, dict[str, Any]]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0]})
    for name, result in results.items():
        eq = result["equity"].sort_values("timestamp")
        if eq.empty:
            continue
        net = (1.0 + eq["period_return"]).cumprod() - 1.0
        gross = (1.0 + eq["gross_period_return"]).cumprod() - 1.0
        axes[0].plot(eq["timestamp"], net, label=f"{name} net", linewidth=2.0)
        axes[0].plot(eq["timestamp"], gross, label=f"{name} gross", linewidth=1.2, alpha=0.45)
        if "cluster_hhi" in eq:
            axes[1].plot(eq["timestamp"], eq["cluster_hhi"].rolling(8, min_periods=1).mean(), label=name, linewidth=1.6)
    axes[0].axhline(0, color="#222222", linewidth=0.8)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].set_title("P15b cluster cap vs P9")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.25)
    axes[1].set_title("Rolling mean latent cluster HHI")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir / "p15b_curves.png", dpi=160)
    plt.close(fig)


def plot_cost(output_dir: Path, cost: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, frame in cost.groupby("variant"):
        ax.plot(frame["one_way_cost_bps"], frame["total_return"], marker="o", label=name)
    ax.axhline(0, color="#222222", linewidth=0.8)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("one-way cost bps")
    ax.set_ylabel("total return")
    ax.set_title("Cost sensitivity")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "cost_sensitivity.png", dpi=160)
    plt.close(fig)


def plot_selected(output_dir: Path, selected: pd.DataFrame) -> None:
    counts = selected["selected_cluster_spec"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(counts))))
    ax.barh(counts.index, counts.values)
    ax.set_title("Selected cluster cap spec by fold")
    ax.set_xlabel("fold count")
    fig.tight_layout()
    fig.savefig(output_dir / "selected_cluster_cap_counts.png", dpi=160)
    plt.close(fig)


def plot_fold_delta(output_dir: Path, fold_delta: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    colors = np.where(fold_delta["delta_total_return"] >= 0, "#2b8a3e", "#c92a2a")
    ax.bar(fold_delta["fold"], fold_delta["delta_total_return"], color=colors)
    ax.axhline(0, color="#222222", linewidth=0.8)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("P15b minus P9 fold return")
    ax.set_xlabel("fold")
    ax.set_ylabel("delta return")
    fig.tight_layout()
    fig.savefig(output_dir / "fold_delta_vs_p9.png", dpi=160)
    plt.close(fig)


def plot_latent_diagnostics(output_dir: Path, diagnostics: pd.DataFrame) -> None:
    if diagnostics.empty:
        return
    frame = diagnostics[(diagnostics["lookback_hours"] == 720) & (diagnostics["cluster_k"] == 4)].sort_values("timestamp")
    if frame.empty:
        frame = diagnostics.sort_values("timestamp")
    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    axes[0].plot(frame["timestamp"], frame["max_cluster_gross_share"], linewidth=1.5)
    axes[0].set_title("P15a P9 max latent cluster gross share")
    axes[1].plot(frame["timestamp"], frame["cluster_hhi"], linewidth=1.5)
    axes[1].set_title("P15a P9 latent cluster HHI")
    axes[2].plot(frame["timestamp"], frame["abs_pc1_exposure"], linewidth=1.5, label="abs PC1")
    axes[2].plot(frame["timestamp"], frame["abs_pc2_exposure"], linewidth=1.2, label="abs PC2")
    axes[2].set_title("P15a P9 latent PC exposure")
    axes[2].legend(loc="best")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir / "p15a_latent_diagnostics.png", dpi=160)
    plt.close(fig)


def write_report(
    output_dir: Path,
    latent_summary: pd.DataFrame,
    summary: pd.DataFrame,
    cost_summary: pd.DataFrame,
    selected: pd.DataFrame,
    validation: pd.DataFrame,
    p9_validation: pd.DataFrame,
    fold_delta: pd.DataFrame,
    bootstrap: pd.DataFrame,
    start_vs_p9: pd.DataFrame,
    dropout_vs_p9: pd.DataFrame,
    cap_attr: pd.DataFrame,
    cluster_delta: pd.DataFrame,
    reality: pd.DataFrame,
) -> None:
    accepted = bool(reality["pass"].all())
    lines = [
        "# P15 Latent Cluster Risk",
        "",
        "## Design",
        "",
        "- P15a diagnoses P9 latent residual clusters and rolling PCA exposures with ex-ante rolling windows.",
        "- P15b keeps P9 signal and P9 eligible long/short pool fixed, then adds only an internal rolling cluster gross cap.",
        "- PCA is diagnostic only in this run; no PCA exposure cap is traded.",
        "- Rolling cluster/PCA models use only residual returns before each rebalance timestamp.",
        f"- Accepted as new baseline: `{accepted}`.",
        "",
        "## P15a Latent Summary",
        "```text",
        latent_summary.to_string(index=False),
        "```",
        "",
        "## Variant Summary",
        "```text",
        summary.to_string(index=False),
        "```",
        "",
        "## Reality Summary",
        "```text",
        reality.to_string(index=False),
        "```",
        "",
        "## Cluster Risk Delta Vs P9",
        "```text",
        cluster_delta.to_string(index=False),
        "```",
        "",
        "## Cost Sensitivity",
        "```text",
        cost_summary.to_string(index=False),
        "```",
        "",
        "## Selected Cluster Cap By Fold",
        "```text",
        selected.to_string(index=False),
        "```",
        "",
        "## Cluster Cap Attribution",
        "```text",
        cap_attr.to_string(index=False),
        "```",
        "",
        "## Fold Delta Vs P9",
        "```text",
        fold_delta.to_string(index=False),
        "```",
        "",
        "## Bootstrap",
        "```text",
        bootstrap.to_string(index=False),
        "```",
        "",
        "## Start Fold Sensitivity",
        "```text",
        start_vs_p9.to_string(index=False),
        "```",
        "",
        "## Symbol Dropout",
        "```text",
        dropout_vs_p9.to_string(index=False),
        "```",
        "",
        "## Validation Metrics",
        "```text",
        validation.to_string(index=False),
        "```",
        "",
        "## P9 Validation Cluster Reference",
        "```text",
        p9_validation.to_string(index=False),
        "```",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def update_config_stage(config_path: Path) -> None:
    cfg = load_config(config_path)
    stages = cfg.setdefault("stages", {})
    stages["p15_latent_cluster_risk"] = {
        "script": "../../../scripts/run_p15_latent_cluster_risk.py",
        "output_dir": "../outputs/p15_latent_cluster_risk",
        "report": "../outputs/p15_latent_cluster_risk/REPORT.md",
        "summary": "../outputs/p15_latent_cluster_risk/variant_summary.csv",
        "latent_diagnostics": "../outputs/p15_latent_cluster_risk/p15a_latent_diagnostics.csv",
        "selected_cluster_cap_by_fold": "../outputs/p15_latent_cluster_risk/selected_cluster_cap_by_fold.csv",
    }
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False), encoding="utf-8")


if __name__ == "__main__":
    main()
