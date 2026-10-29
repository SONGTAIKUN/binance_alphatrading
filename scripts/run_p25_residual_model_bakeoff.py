#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
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
from crypto_stat_arb.features import rolling_zscore
from run_p18_factor_decay_walkforward import centered_rank_matrix, rowwise_corr
from run_p18c_factor_selector_portfolio import (
    add_p18c_entry_fields,
    build_cost_sensitivity,
    load_existing_result,
    write_run_result,
)
from run_p18d_guarded_factor_selector import LONG_ROOT, add_regime_features
from run_p20_selection_protocol import build_protocols
from run_p21_gross_alpha_family_search import (
    Component,
    Family,
    resolve_families,
    run_family_diagnostics,
    select_family_by_fold,
    summarize_families,
    summarize_selected,
)
from run_p22_p21_family_portfolio import replay_protocol
from run_p23_crypto_ff_residualizer import FACTOR_NAMES, forward_sum, rolling_ols_predict
from run_p24_ff_residual_factor_selection import summarize_result
from run_p3d_p4_gated_rerank_alphagpt import (
    load_panel as load_p3d_panel,
    make_portfolio_config,
    make_risk_template,
)
from run_turnover_experiments import to_jsonable


EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "baseline_locked_v1"
DEFAULT_CONFIG = EXPERIMENT_ROOT / "configs" / "baseline_locked_v1.yaml"
DEFAULT_LONG_CONFIG = EXPERIMENT_ROOT / "inputs" / "long_history" / "configs" / "baseline_long_history.yaml"
DEFAULT_ASSET_PANEL = EXPERIMENT_ROOT / "inputs" / "long_history" / "baseline_panel" / "asset_residual_panel.csv"
DEFAULT_MANUAL_PANEL = LONG_ROOT / "inputs" / "manual_slow_feature_panel_long.csv.gz"
DEFAULT_FOLDS = LONG_ROOT / "inputs" / "long_formula_selections.csv"
DEFAULT_P15C_RUN = LONG_ROOT / "p15c_nested" / "nested_extreme_cluster_run"
DEFAULT_FF_PANEL = EXPERIMENT_ROOT / "outputs" / "p23_crypto_ff_residualizer" / "crypto_ff_residual_panel.csv.gz"
DEFAULT_FF_FACTORS = EXPERIMENT_ROOT / "outputs" / "p23_crypto_ff_residualizer" / "crypto_ff_rolling_factor_returns.csv"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "outputs" / "p25_residual_model_bakeoff"

MODEL_ORDER = [
    "legacy_full",
    "legacy_no_sector",
    "legacy_orthogonalized",
    "legacy_ridge",
    "crypto_ff",
    "crypto_ff_beta_shrink",
]
MODEL_LABELS = {
    "legacy_full": "Legacy 4F",
    "legacy_no_sector": "Legacy no-sector",
    "legacy_orthogonalized": "Orthogonalized 4F",
    "legacy_ridge": "Legacy ridge",
    "crypto_ff": "Crypto FF-like",
    "crypto_ff_beta_shrink": "FF-like + shrink",
    "p15c": "P15c reference",
}
LEGACY_FACTORS = ["factor_btc_ret", "factor_eth_ret", "factor_market_ret", "factor_sector_ret"]
LEGACY_BETAS = ["beta_btc", "beta_eth", "beta_market", "beta_sector"]
HORIZONS = [48, 72, 96, 168]
PROTOCOL_NAMES = [
    "p20_strict_q25_nonneg",
    "p20_balanced_q25_m3",
    "p20_practical_q25_m5",
    "p20_loose_decay_corr",
]
PRIMARY_PROTOCOL = "p20_balanced_q25_m3"


def main() -> None:
    parser = argparse.ArgumentParser(description="P25 residual-model bake-off with P20/P21 nested OOS portfolio replay.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--long-config", default=str(DEFAULT_LONG_CONFIG))
    parser.add_argument("--asset-panel", default=str(DEFAULT_ASSET_PANEL))
    parser.add_argument("--manual-panel", default=str(DEFAULT_MANUAL_PANEL))
    parser.add_argument("--folds", default=str(DEFAULT_FOLDS))
    parser.add_argument("--p15c-run-dir", default=str(DEFAULT_P15C_RUN))
    parser.add_argument("--ff-panel", default=str(DEFAULT_FF_PANEL))
    parser.add_argument("--ff-factors", default=str(DEFAULT_FF_FACTORS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--horizon", type=int, default=72)
    parser.add_argument("--beta-window", type=int, default=720)
    parser.add_argument("--min-beta-obs", type=int, default=240)
    parser.add_argument("--ridge-strength", type=float, default=0.10)
    parser.add_argument("--validation-days", type=int, default=90)
    parser.add_argument("--min-symbols", type=int, default=8)
    parser.add_argument("--cost-bps", type=float, nargs="+", default=[0, 6, 8, 15, 20, 30])
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    panel_dir = ensure_dir(output_dir / "residual_panels")
    model_dir = ensure_dir(output_dir / "models")
    horizon = int(args.horizon)
    folds = pd.read_csv(
        args.folds,
        parse_dates=["train_start", "train_end", "test_start", "test_end"],
    ).sort_values("fold")
    long_cfg = load_config(Path(args.long_config))

    print("[1/7] Building or loading six past-only residual panels", flush=True)
    collinearity_parts: list[pd.DataFrame] = []
    diagnostic_parts: list[pd.DataFrame] = []
    for model in MODEL_ORDER:
        path = panel_dir / f"{model}.csv.gz"
        diag_path = model_dir / model / "residual_diagnostics.csv"
        coll_path = model_dir / model / "factor_collinearity.csv"
        if path.exists() and diag_path.exists() and coll_path.exists() and not args.force_rebuild:
            print(f"      {model}: cached", flush=True)
            diagnostics = pd.read_csv(diag_path)
            collinearity = pd.read_csv(coll_path)
        else:
            print(f"      {model}: generating", flush=True)
            panel, factor_sets = build_model_panel(
                model=model,
                asset_path=Path(args.asset_panel),
                ff_panel_path=Path(args.ff_panel),
                ff_factor_path=Path(args.ff_factors),
                beta_window=int(args.beta_window),
                min_beta_obs=int(args.min_beta_obs),
                ridge_strength=float(args.ridge_strength),
            )
            panel = add_generic_features(panel)
            panel.to_csv(path, index=False)
            diagnostics = residual_diagnostics(model, panel)
            collinearity = collinearity_diagnostics(model, factor_sets)
            ensure_dir(model_dir / model)
            diagnostics.to_csv(diag_path, index=False)
            collinearity.to_csv(coll_path, index=False)
            del panel, factor_sets
            gc.collect()
        diagnostic_parts.append(diagnostics)
        collinearity_parts.append(collinearity)

    diagnostics_all = pd.concat(diagnostic_parts, ignore_index=True)
    collinearity_all = pd.concat(collinearity_parts, ignore_index=True)
    diagnostics_all.to_csv(output_dir / "residual_diagnostics.csv", index=False)
    collinearity_all.to_csv(output_dir / "factor_collinearity.csv", index=False)

    print("[2/7] Loading common regime state and allocator inputs", flush=True)
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    base_panel = load_p3d_panel(config_path, Path(args.asset_panel), Path(args.manual_panel), horizon)
    add_regime_features(base_panel)
    regime = (
        base_panel[["timestamp", "_p18d_high_risk_regime"]]
        .drop_duplicates("timestamp")
        .sort_values("timestamp")
        .set_index("timestamp")["_p18d_high_risk_regime"]
        .astype(bool)
    )
    add_p18c_entry_fields(base_panel)
    pf_cfg = make_portfolio_config(cfg, horizon)
    risk_template = make_risk_template(cfg)
    p15c = load_existing_result(Path(args.p15c_run_dir))
    protocols = [spec for spec in build_protocols() if spec.name in PROTOCOL_NAMES]

    print("[3/7] Running identical P20/P21 factor selection for every residualizer", flush=True)
    fold_metric_parts: list[pd.DataFrame] = []
    family_summary_parts: list[pd.DataFrame] = []
    selected_parts: list[pd.DataFrame] = []
    selected_summary_parts: list[pd.DataFrame] = []
    simple_ic_parts: list[pd.DataFrame] = []
    results: dict[str, dict[str, Any]] = {"p15c": p15c}

    for model in MODEL_ORDER:
        print(f"      {model}: factor diagnostics + selection", flush=True)
        one_dir = ensure_dir(model_dir / model)
        cached_runs = {
            protocol: one_dir / protocol / "nested_oos_run"
            for protocol in PROTOCOL_NAMES
        }
        cached_eval = all(
            [
                (one_dir / "family_fold_metrics.csv").exists(),
                (one_dir / "family_summary.csv").exists(),
                (one_dir / "selected_family_by_fold.csv").exists(),
                (one_dir / "selected_protocol_summary.csv").exists(),
                (one_dir / "simple_label_ic.csv").exists(),
                *[(path / "equity_curve.csv").exists() for path in cached_runs.values()],
            ]
        )
        if cached_eval and not args.force_rebuild:
            print(f"      {model}: cached factor selection and portfolio replays", flush=True)
            fold_metrics = pd.read_csv(one_dir / "family_fold_metrics.csv")
            family_summary = pd.read_csv(one_dir / "family_summary.csv")
            selected = pd.read_csv(one_dir / "selected_family_by_fold.csv")
            selected_summary = pd.read_csv(one_dir / "selected_protocol_summary.csv")
            simple_ic = pd.read_csv(one_dir / "simple_label_ic.csv")
            fold_metric_parts.append(fold_metrics)
            family_summary_parts.append(family_summary)
            selected_parts.append(selected)
            selected_summary_parts.append(selected_summary)
            simple_ic_parts.append(simple_ic)
            for protocol, run_dir in cached_runs.items():
                stage = f"{model}__{protocol.replace('p20_', '')}"
                results[stage] = load_existing_result(run_dir)
            continue
        slim = pd.read_csv(panel_dir / f"{model}.csv.gz", parse_dates=["timestamp"])
        factor_panel, factor_meta = build_generic_factor_library(slim)
        factor_meta.to_csv(one_dir / "factor_library.csv", index=False)
        simple_ic = simple_label_ic(model, factor_panel, horizon, int(args.min_symbols))
        simple_ic.to_csv(one_dir / "simple_label_ic.csv", index=False)
        simple_ic_parts.append(simple_ic)
        label_rank = centered_rank_matrix(factor_panel, f"label_resid_ret_h{horizon}", min_symbols=int(args.min_symbols))
        families = resolve_families(build_generic_families(), set(factor_meta["factor"]))
        needed = sorted({component.factor for family in families for component in family.components})
        rank_matrices = {
            factor: centered_rank_matrix(factor_panel, factor, min_symbols=int(args.min_symbols))
            for factor in needed
        }
        fold_metrics, score_ts = run_family_diagnostics(
            families=families,
            folds=folds,
            rank_matrices=rank_matrices,
            label_rank=label_rank,
            regime=regime,
            horizon=horizon,
            validation_days=int(args.validation_days),
            min_symbols=int(args.min_symbols),
        )
        fold_metrics["model"] = model
        fold_metrics.to_csv(one_dir / "family_fold_metrics.csv", index=False)
        fold_metric_parts.append(fold_metrics)
        family_summary = summarize_families(fold_metrics, protocols)
        family_summary["model"] = model
        family_summary.to_csv(one_dir / "family_summary.csv", index=False)
        family_summary_parts.append(family_summary)
        selected = select_family_by_fold(fold_metrics, protocols)
        selected["model"] = model
        selected.to_csv(one_dir / "selected_family_by_fold.csv", index=False)
        selected_parts.append(selected)
        selected_summary = summarize_selected(selected)
        selected_summary["model"] = model
        selected_summary.to_csv(one_dir / "selected_protocol_summary.csv", index=False)
        selected_summary_parts.append(selected_summary)

        label_repl = slim[["timestamp", "symbol", f"label_resid_ret_h{horizon}"]].copy()
        model_base = replace_model_label(base_panel, label_repl, horizon)
        for protocol in PROTOCOL_NAMES:
            stage = f"{model}__{protocol.replace('p20_', '')}"
            protocol_selected = selected[selected["protocol"].eq(protocol)].copy()
            result, selection = replay_protocol(
                protocol_name=stage,
                protocol=protocol,
                base_panel=model_base,
                family_scores=score_ts,
                folds=folds,
                selected=protocol_selected,
                p15c=p15c,
                pf_cfg=pf_cfg,
                risk_template=risk_template,
                long_fraction=0.50,
                short_confirm_rank=0.35,
            )
            results[stage] = result
            run_dir = ensure_dir(one_dir / protocol / "nested_oos_run")
            write_run_result(run_dir, result)
            selection.to_csv(one_dir / protocol / "portfolio_selection_by_fold.csv", index=False)
        del slim, factor_panel, label_rank, rank_matrices, model_base, score_ts
        gc.collect()

    pd.concat(fold_metric_parts, ignore_index=True).to_csv(output_dir / "family_fold_metrics.csv", index=False)
    pd.concat(family_summary_parts, ignore_index=True).to_csv(output_dir / "family_summary.csv", index=False)
    pd.concat(selected_parts, ignore_index=True).to_csv(output_dir / "selected_family_by_fold.csv", index=False)
    selected_summary_all = pd.concat(selected_summary_parts, ignore_index=True)
    selected_summary_all.to_csv(output_dir / "selected_protocol_summary.csv", index=False)
    simple_ic_all = pd.concat(simple_ic_parts, ignore_index=True)
    simple_ic_all.to_csv(output_dir / "simple_label_ic.csv", index=False)

    print("[4/7] Building OOS, cost, and 2024Q2-Q3 comparisons", flush=True)
    portfolio_summary = build_portfolio_summary(results)
    portfolio_summary.to_csv(output_dir / "portfolio_summary.csv", index=False)
    cost = build_cost_sensitivity(results, [float(value) for value in args.cost_bps])
    cost.to_csv(output_dir / "cost_sensitivity.csv", index=False)
    regime_summary = build_q2_q3_summary(results, [0.0, 6.0, 8.0, 15.0, 20.0, 30.0], horizon)
    regime_summary.to_csv(output_dir / "regime_2024q2_q3_summary.csv", index=False)
    fold_stability = build_fold_stability(results)
    fold_stability.to_csv(output_dir / "fold_stability.csv", index=False)

    print("[5/7] Building fixed-protocol bake-off scorecard", flush=True)
    scorecard = build_scorecard(
        diagnostics=diagnostics_all,
        collinearity=collinearity_all,
        simple_ic=simple_ic_all,
        selected_summary=selected_summary_all,
        portfolio=portfolio_summary,
        cost=cost,
        regime=regime_summary,
        fold_stability=fold_stability,
        primary_protocol=PRIMARY_PROTOCOL,
    )
    scorecard.to_csv(output_dir / "bakeoff_scorecard.csv", index=False)

    print("[6/7] Writing comparison plots", flush=True)
    plot_scorecard(output_dir, scorecard)
    plot_primary_curves(output_dir, results, PRIMARY_PROTOCOL, [0.0, 8.0])
    plot_regime_comparison(output_dir, regime_summary, PRIMARY_PROTOCOL)

    print("[7/7] Writing report", flush=True)
    metadata = {
        "models": MODEL_ORDER,
        "primary_protocol": PRIMARY_PROTOCOL,
        "horizon": horizon,
        "beta_window": int(args.beta_window),
        "min_beta_obs": int(args.min_beta_obs),
        "ridge_strength": float(args.ridge_strength),
        "cost_bps": [float(value) for value in args.cost_bps],
        "q2_q3_window": ["2024-04-01T00:00:00Z", "2024-10-01T00:00:00Z"],
        "past_only_rules": [
            "All regression betas at t use observations ending at t-1.",
            "Orthogonalization coefficients at t use observations ending at t-1.",
            "Ridge strengths are fixed before test evaluation and are not selected on OOS results.",
            "Future residual labels use beta known at t and future factor returns only in the label.",
            "P20/P21 formula family and sign choices use train/validation only; portfolio rows are stitched test only.",
        ],
    }
    (output_dir / "metadata.json").write_text(json.dumps(to_jsonable(metadata), indent=2), encoding="utf-8")
    write_report(output_dir, scorecard, diagnostics_all, collinearity_all, selected_summary_all, fold_stability)
    print(f"Done: {output_dir}", flush=True)


def read_base_panel(path: Path) -> pd.DataFrame:
    needed = [
        "timestamp",
        "symbol",
        "log_ret",
        *LEGACY_FACTORS,
        "residual_log_ret",
        "resid_z",
        "score_baseline",
        *LEGACY_BETAS,
        *[f"fwd_log_ret_h{h}" for h in HORIZONS],
        *[f"label_resid_ret_h{h}" for h in HORIZONS],
    ]
    frame = pd.read_csv(path, usecols=needed, parse_dates=["timestamp"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["symbol"] = frame["symbol"].astype(str)
    return frame.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def build_model_panel(
    model: str,
    asset_path: Path,
    ff_panel_path: Path,
    ff_factor_path: Path,
    beta_window: int,
    min_beta_obs: int,
    ridge_strength: float,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    if model in {"crypto_ff", "crypto_ff_beta_shrink"}:
        return build_ff_model_panel(
            model,
            ff_panel_path,
            ff_factor_path,
            beta_window,
            min_beta_obs,
            ridge_strength,
        )
    base = read_base_panel(asset_path)
    if model == "legacy_full":
        out = base[
            [
                "timestamp",
                "symbol",
                "log_ret",
                "residual_log_ret",
                "resid_z",
                "score_baseline",
                *LEGACY_BETAS,
                *[f"fwd_log_ret_h{h}" for h in HORIZONS],
                *[f"label_resid_ret_h{h}" for h in HORIZONS],
            ]
        ].copy()
        out = out.rename(columns={"score_baseline": "score_neg_resid_z"})
        factors = {
            symbol: group.set_index("timestamp")[LEGACY_FACTORS].sort_index()
            for symbol, group in base.groupby("symbol", sort=True)
        }
        return out, factors

    pieces: list[pd.DataFrame] = []
    factor_sets: dict[str, pd.DataFrame] = {}
    for symbol, group in base.groupby("symbol", sort=True):
        g = group.sort_values("timestamp").copy()
        y = g.set_index("timestamp")["log_ret"]
        raw_x = g.set_index("timestamp")[LEGACY_FACTORS].astype(float)
        if model == "legacy_no_sector":
            x = raw_x[LEGACY_FACTORS[:3]].copy()
            beta, pred, resid = rolling_ols_predict(y, x, beta_window, min_beta_obs)
        elif model == "legacy_orthogonalized":
            x = rolling_past_orthogonalize(raw_x, beta_window, min_beta_obs)
            beta, pred, resid = rolling_ols_predict(y, x, beta_window, min_beta_obs)
        elif model == "legacy_ridge":
            x = raw_x.copy()
            beta, pred, resid = rolling_ridge_predict(y, x, beta_window, min_beta_obs, ridge_strength)
        else:
            raise ValueError(f"Unknown model: {model}")
        factor_sets[symbol] = x
        part = pd.DataFrame({"timestamp": g["timestamp"].to_numpy(), "symbol": symbol, "log_ret": y.to_numpy()})
        part["residual_log_ret"] = resid.to_numpy()
        part["predicted_log_ret"] = pred.to_numpy()
        for name in x.columns:
            part[f"beta_{safe_name(name)}"] = beta[name].to_numpy()
        for horizon in HORIZONS:
            part[f"fwd_log_ret_h{horizon}"] = g[f"fwd_log_ret_h{horizon}"].to_numpy()
            part[f"label_resid_ret_h{horizon}"] = residual_label(
                part[f"fwd_log_ret_h{horizon}"],
                beta,
                x,
                horizon,
                execution_delay=1,
            ).to_numpy()
        pieces.append(part)
    out = pd.concat(pieces, ignore_index=True).sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    return out, factor_sets


def build_ff_model_panel(
    model: str,
    ff_panel_path: Path,
    ff_factor_path: Path,
    beta_window: int,
    min_beta_obs: int,
    ridge_strength: float,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    factor_returns = pd.read_csv(ff_factor_path, parse_dates=["timestamp"]).set_index("timestamp")[FACTOR_NAMES]
    factor_returns.index = pd.to_datetime(factor_returns.index, utc=True)
    needed = [
        "timestamp",
        "symbol",
        "log_ret",
        *[f"fwd_log_ret_h{h}" for h in HORIZONS],
        "ff_residual_log_ret",
        "ff_resid_z",
        "ff_score_neg_resid_z",
        *[f"ff_beta_{name}" for name in FACTOR_NAMES],
        *[f"ff_label_resid_ret_h{h}" for h in HORIZONS],
    ]
    ff = pd.read_csv(ff_panel_path, usecols=needed, parse_dates=["timestamp"])
    ff["timestamp"] = pd.to_datetime(ff["timestamp"], utc=True)
    ff["symbol"] = ff["symbol"].astype(str)
    if model == "crypto_ff":
        rename = {
            "ff_residual_log_ret": "residual_log_ret",
            "ff_resid_z": "resid_z",
            "ff_score_neg_resid_z": "score_neg_resid_z",
            **{f"ff_beta_{name}": f"beta_{name.lower()}" for name in FACTOR_NAMES},
            **{f"ff_label_resid_ret_h{h}": f"label_resid_ret_h{h}" for h in HORIZONS},
        }
        out = ff.rename(columns=rename)
        keep = [
            "timestamp",
            "symbol",
            "log_ret",
            "residual_log_ret",
            "resid_z",
            "score_neg_resid_z",
            *[f"beta_{name.lower()}" for name in FACTOR_NAMES],
            *[f"fwd_log_ret_h{h}" for h in HORIZONS],
            *[f"label_resid_ret_h{h}" for h in HORIZONS],
        ]
        return out[keep].sort_values(["symbol", "timestamp"]).reset_index(drop=True), {"__GLOBAL__": factor_returns}

    pieces: list[pd.DataFrame] = []
    for symbol, group in ff.groupby("symbol", sort=True):
        g = group.sort_values("timestamp").copy()
        y = g.set_index("timestamp")["log_ret"]
        x = factor_returns.reindex(y.index)
        beta, pred, resid = rolling_ridge_predict(y, x, beta_window, min_beta_obs, ridge_strength)
        part = pd.DataFrame({"timestamp": y.index, "symbol": symbol, "log_ret": y.to_numpy()})
        part["residual_log_ret"] = resid.to_numpy()
        part["predicted_log_ret"] = pred.to_numpy()
        for name in FACTOR_NAMES:
            part[f"beta_{name.lower()}"] = beta[name].to_numpy()
        for horizon in HORIZONS:
            part[f"fwd_log_ret_h{horizon}"] = g[f"fwd_log_ret_h{horizon}"].to_numpy()
            part[f"label_resid_ret_h{horizon}"] = residual_label(
                part[f"fwd_log_ret_h{horizon}"], beta, x, horizon, execution_delay=1
            ).to_numpy()
        pieces.append(part)
    out = pd.concat(pieces, ignore_index=True).sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    return out, {"__GLOBAL__": factor_returns}


def safe_name(name: str) -> str:
    return name.removeprefix("factor_").removesuffix("_ret")


def rolling_ridge_predict(
    y: pd.Series,
    factors: pd.DataFrame,
    beta_window: int,
    min_beta_obs: int,
    ridge_strength: float,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    y = y.astype(float).reindex(factors.index)
    x = factors.astype(float)
    names = list(x.columns)
    y_arr = y.to_numpy(dtype=float)
    x_arr = x.to_numpy(dtype=float)
    n, p = x_arr.shape
    valid = np.isfinite(y_arr) & np.isfinite(x_arr).all(axis=1)
    valid_f = valid.astype(float)
    count = pd.Series(valid_f, index=y.index).rolling(beta_window, min_periods=min_beta_obs).sum().shift(1).to_numpy()
    sum_y = pd.Series(np.where(valid, y_arr, np.nan), index=y.index).rolling(beta_window, min_periods=min_beta_obs).sum().shift(1).to_numpy()
    sum_x = np.column_stack(
        [
            pd.Series(np.where(valid, x_arr[:, j], np.nan), index=y.index)
            .rolling(beta_window, min_periods=min_beta_obs)
            .sum()
            .shift(1)
            .to_numpy()
            for j in range(p)
        ]
    )
    sum_xy = np.column_stack(
        [
            pd.Series(np.where(valid, x_arr[:, j] * y_arr, np.nan), index=y.index)
            .rolling(beta_window, min_periods=min_beta_obs)
            .sum()
            .shift(1)
            .to_numpy()
            for j in range(p)
        ]
    )
    sum_xx: dict[tuple[int, int], np.ndarray] = {}
    for a in range(p):
        for b in range(a, p):
            sum_xx[(a, b)] = (
                pd.Series(np.where(valid, x_arr[:, a] * x_arr[:, b], np.nan), index=y.index)
                .rolling(beta_window, min_periods=min_beta_obs)
                .sum()
                .shift(1)
                .to_numpy()
            )
    coefs = np.full((n, p + 1), np.nan)
    pred = np.full(n, np.nan)
    resid = np.full(n, np.nan)
    for i in range(n):
        obs = count[i]
        if not np.isfinite(obs) or obs < min_beta_obs or not np.isfinite(x_arr[i]).all():
            continue
        x_mean = sum_x[i] / obs
        y_mean = sum_y[i] / obs
        sxx = np.empty((p, p), dtype=float)
        for a in range(p):
            for b in range(p):
                key = (a, b) if a <= b else (b, a)
                sxx[a, b] = sum_xx[key][i] - obs * x_mean[a] * x_mean[b]
        sxy = sum_xy[i] - obs * x_mean * y_mean
        if not np.isfinite(sxx).all() or not np.isfinite(sxy).all():
            continue
        scale = np.maximum(np.diag(sxx), 1e-14)
        penalty = ridge_strength * np.diag(scale)
        beta = np.linalg.pinv(sxx + penalty, rcond=1e-10) @ sxy
        intercept = y_mean - float(x_mean @ beta)
        coefs[i, 0] = intercept
        coefs[i, 1:] = beta
        pred[i] = intercept + float(x_arr[i] @ beta)
        resid[i] = y_arr[i] - pred[i] if np.isfinite(y_arr[i]) else np.nan
    beta_frame = pd.DataFrame(coefs, index=y.index, columns=["intercept", *names])
    return beta_frame, pd.Series(pred, index=y.index), pd.Series(resid, index=y.index)


def rolling_past_orthogonalize(factors: pd.DataFrame, beta_window: int, min_beta_obs: int) -> pd.DataFrame:
    out = pd.DataFrame(index=factors.index)
    for index, name in enumerate(factors.columns):
        target = factors[name].astype(float)
        if index == 0:
            out[f"orth_{safe_name(name)}"] = target
            continue
        beta, pred, resid = rolling_ols_predict(target, out.copy(), beta_window, min_beta_obs)
        del beta, pred
        out[f"orth_{safe_name(name)}"] = resid
    return out


def residual_label(
    fwd_log_ret: pd.Series,
    beta: pd.DataFrame,
    factors: pd.DataFrame,
    horizon: int,
    execution_delay: int,
) -> pd.Series:
    explained = pd.Series(0.0, index=beta.index, dtype=float)
    valid = pd.Series(pd.to_numeric(fwd_log_ret, errors="coerce").notna().to_numpy(), index=beta.index)
    for name in factors.columns:
        future = forward_sum(factors[name], execution_delay + 1, horizon)
        explained = explained + beta[name] * future
        valid = valid & beta[name].notna() & future.notna()
    raw = pd.Series(pd.to_numeric(fwd_log_ret, errors="coerce").to_numpy(), index=beta.index)
    return (raw - explained).where(valid)


def add_generic_features(panel: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    beta_cols = [col for col in panel.columns if col.startswith("beta_") and col != "beta_intercept"]
    for _, group in panel.groupby("symbol", sort=False):
        g = group.sort_values("timestamp").copy()
        resid = g["residual_log_ret"].astype(float)
        resid_cum = resid.fillna(0.0).cumsum()
        if "resid_z" not in g or g["resid_z"].isna().all():
            g["resid_z"] = rolling_zscore(resid, 168, 56)
        g["score_neg_resid_z"] = -g["resid_z"]
        g["resid_reversal_z"] = -g["resid_z"]
        g["resid_abs_z"] = -g["resid_z"].abs()
        g["resid_mom_168_24"] = resid.shift(24).rolling(144, min_periods=72).sum()
        g["resid_mom_336_24"] = resid.shift(24).rolling(312, min_periods=120).sum()
        g["resid_mom_504_48"] = resid.shift(48).rolling(456, min_periods=168).sum()
        g["resid_high_mom_168"] = resid_cum - resid_cum.rolling(168, min_periods=56).max()
        g["resid_high_mom_336"] = resid_cum - resid_cum.rolling(336, min_periods=120).max()
        g["resid_ma_slope_48_168"] = resid_cum.ewm(span=48, adjust=False, min_periods=24).mean() - resid_cum.ewm(
            span=168, adjust=False, min_periods=56
        ).mean()
        g["idio_vol_long"] = resid.rolling(168, min_periods=56).std(ddof=0)
        g["low_idio_vol"] = -g["idio_vol_long"]
        g["resid_trend_quality_168_24"] = g["resid_mom_168_24"] / g["idio_vol_long"].replace(0.0, np.nan)
        vol336 = resid.rolling(336, min_periods=120).std(ddof=0)
        g["resid_trend_quality_336_24"] = g["resid_mom_336_24"] / vol336.replace(0.0, np.nan)
        available_betas = [col for col in beta_cols if col in g]
        if available_betas:
            g["beta_abs_sum"] = -g[available_betas].abs().sum(axis=1)
            delta = g[available_betas].diff().abs()
            g["beta_stability_168"] = -delta.rolling(168, min_periods=56).mean().mean(axis=1)
            g["beta_stability_720"] = -delta.rolling(720, min_periods=240).mean().mean(axis=1)
        else:
            g["beta_abs_sum"] = np.nan
            g["beta_stability_168"] = np.nan
            g["beta_stability_720"] = np.nan
        g["quality_lowrisk_combo"] = (
            g["low_idio_vol"] + g["beta_stability_168"] + g["beta_abs_sum"]
        ) / 3.0
        pieces.append(g)
    return pd.concat(pieces, ignore_index=True).sort_values(["timestamp", "symbol"]).reset_index(drop=True)


GENERIC_FACTORS = [
    "score_neg_resid_z",
    "resid_z",
    "resid_reversal_z",
    "resid_abs_z",
    "resid_mom_168_24",
    "resid_mom_336_24",
    "resid_mom_504_48",
    "resid_high_mom_168",
    "resid_high_mom_336",
    "resid_ma_slope_48_168",
    "resid_trend_quality_168_24",
    "resid_trend_quality_336_24",
    "idio_vol_long",
    "low_idio_vol",
    "beta_abs_sum",
    "beta_stability_168",
    "beta_stability_720",
    "quality_lowrisk_combo",
]


def build_generic_factor_library(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    factor_cols = [col for col in GENERIC_FACTORS if col in panel.columns]
    rename = {col: f"rm__{col}" for col in factor_cols}
    labels = [f"label_resid_ret_h{h}" for h in HORIZONS if f"label_resid_ret_h{h}" in panel]
    out = panel[["timestamp", "symbol", *labels, *factor_cols]].rename(columns=rename)
    rows = []
    for raw, safe in rename.items():
        finite = np.isfinite(out[safe].to_numpy(dtype=float, copy=False))
        if finite.mean() < 0.25:
            continue
        by_ts = out.loc[finite, ["timestamp", safe]].groupby("timestamp")[safe].std(ddof=0)
        if by_ts.replace([np.inf, -np.inf], np.nan).dropna().gt(1e-12).mean() < 0.20:
            continue
        rows.append({"factor": safe, "source": "residual_model", "raw_name": raw})
    meta = pd.DataFrame(rows)
    keep = ["timestamp", "symbol", *labels, *meta["factor"].tolist()]
    return out[keep].copy(), meta


def c(name: str, weight: float = 1.0) -> Component:
    return Component(f"rm__{name}", weight)


def build_generic_families() -> list[Family]:
    return [
        Family(
            "rm_mean_reversion_quality",
            "Residual mean reversion with low-risk and beta-stability quality.",
            (c("score_neg_resid_z"), c("resid_reversal_z"), c("resid_abs_z", 0.5), c("low_idio_vol", 0.5), c("beta_stability_168", 0.5)),
        ),
        Family(
            "rm_slow_residual_momentum",
            "Slow residual continuation over 168-504h with recent bars skipped.",
            (c("resid_mom_168_24"), c("resid_mom_336_24"), c("resid_mom_504_48", 0.8), c("resid_trend_quality_168_24", 0.7), c("resid_trend_quality_336_24", 0.7)),
        ),
        Family(
            "rm_high_residual_momentum",
            "Distance to residual highs and slow residual MA slope.",
            (c("resid_high_mom_168"), c("resid_high_mom_336"), c("resid_ma_slope_48_168"), c("resid_mom_336_24", 0.5)),
        ),
        Family(
            "rm_beta_stability_quality",
            "Stable common-factor loadings and low idiosyncratic volatility.",
            (c("beta_stability_168"), c("beta_stability_720"), c("beta_abs_sum"), c("low_idio_vol", 0.7)),
        ),
        Family(
            "rm_residual_quality_combo",
            "Mean reversion, slow trend, and residual-model quality committee.",
            (c("score_neg_resid_z", 0.8), c("resid_mom_336_24", 0.6), c("resid_trend_quality_336_24", 0.6), c("quality_lowrisk_combo"), c("beta_stability_720", 0.6)),
        ),
    ]


def collinearity_diagnostics(model: str, factor_sets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for symbol, factors in factor_sets.items():
        data = factors.replace([np.inf, -np.inf], np.nan).dropna()
        active_cols = [col for col in data.columns if float(data[col].std(ddof=0)) > 1e-12]
        data = data[active_cols]
        if len(data) < 100 or data.shape[1] < 2:
            continue
        corr = data.corr().to_numpy(dtype=float)
        if not np.isfinite(corr).all():
            continue
        off = np.abs(corr - np.eye(corr.shape[0]))
        eig = np.linalg.eigvalsh(corr)
        rows.append(
            {
                "model": model,
                "symbol": symbol,
                "factor_count": int(corr.shape[0]),
                "complete_rows": int(len(data)),
                "max_abs_factor_corr": float(off.max()),
                "condition_number": float(np.linalg.cond(corr)),
                "min_eigenvalue": float(eig.min()),
            }
        )
    return pd.DataFrame(rows)


def residual_diagnostics(model: str, panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    beta_cols = [col for col in panel.columns if col.startswith("beta_") and col != "beta_intercept"]
    for symbol, group in panel.groupby("symbol", sort=True):
        g = group.sort_values("timestamp")
        resid = g["residual_log_ret"].replace([np.inf, -np.inf], np.nan).astype(float)
        rolling_vol = resid.rolling(720, min_periods=240).std(ddof=0).dropna()
        beta_delta = g[beta_cols].diff().abs().replace([np.inf, -np.inf], np.nan) if beta_cols else pd.DataFrame()
        rows.append(
            {
                "model": model,
                "symbol": symbol,
                "residual_valid_rate": float(resid.notna().mean()),
                "residual_std": float(resid.std(ddof=0)),
                "residual_vol_cv": float(rolling_vol.std(ddof=0) / rolling_vol.mean()) if len(rolling_vol) and rolling_vol.mean() > 0 else np.nan,
                "residual_vol_p95_to_median": float(rolling_vol.quantile(0.95) / rolling_vol.median()) if len(rolling_vol) and rolling_vol.median() > 0 else np.nan,
                "residual_autocorr_1h": float(resid.autocorr(1)),
                "beta_mean_abs_1h_change": float(beta_delta.stack().mean()) if not beta_delta.empty else np.nan,
                "beta_median_abs_1h_change": float(beta_delta.stack().median()) if not beta_delta.empty else np.nan,
                "beta_mean_7d_std": float(g[beta_cols].rolling(168, min_periods=56).std(ddof=0).stack().mean()) if beta_cols else np.nan,
                "beta_mean_30d_std": float(g[beta_cols].rolling(720, min_periods=240).std(ddof=0).stack().mean()) if beta_cols else np.nan,
            }
        )
    dispersion = panel.pivot(index="timestamp", columns="symbol", values="residual_log_ret").std(axis=1, ddof=0).dropna()
    for row in rows:
        row["xs_dispersion_mean"] = float(dispersion.mean())
        row["xs_dispersion_cv"] = float(dispersion.std(ddof=0) / dispersion.mean()) if dispersion.mean() > 0 else np.nan
        row["xs_dispersion_p95_to_median"] = float(dispersion.quantile(0.95) / dispersion.median()) if dispersion.median() > 0 else np.nan
    return pd.DataFrame(rows)


def simple_label_ic(model: str, panel: pd.DataFrame, horizon: int, min_symbols: int) -> pd.DataFrame:
    label = centered_rank_matrix(panel, f"label_resid_ret_h{horizon}", min_symbols=min_symbols)
    rows = []
    for factor in ["rm__score_neg_resid_z", "rm__resid_mom_336_24", "rm__quality_lowrisk_combo"]:
        if factor not in panel:
            continue
        score = centered_rank_matrix(panel, factor, min_symbols=min_symbols)
        ic = rowwise_corr(score, label, min_symbols=min_symbols).dropna()
        rows.append(
            {
                "model": model,
                "factor": factor,
                "observations": int(len(ic)),
                "mean_ic": float(ic.mean()),
                "median_ic": float(ic.median()),
                "q25_ic": float(ic.quantile(0.25)),
                "positive_ic_rate": float((ic > 0).mean()),
                "ic_ir": float(ic.mean() / ic.std(ddof=0)) if ic.std(ddof=0) > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def replace_model_label(base_panel: pd.DataFrame, labels: pd.DataFrame, horizon: int) -> pd.DataFrame:
    col = f"label_resid_ret_h{horizon}"
    repl = labels.rename(columns={col: f"{col}_p25"})
    out = base_panel.drop(columns=[f"{col}_p25"], errors="ignore").merge(repl, on=["timestamp", "symbol"], how="left")
    out[col] = out[f"{col}_p25"]
    out.drop(columns=[f"{col}_p25"], inplace=True)
    return out


def build_portfolio_summary(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for stage, result in results.items():
        metrics = result.get("metrics") or summarize_result(result["equity"])
        rows.append({"stage": stage, **metrics})
    return pd.DataFrame(rows)


def adjusted_return_series(equity: pd.DataFrame, cost_bps: float) -> pd.Series:
    gross = equity["gross_period_return"].astype(float)
    turnover = equity["turnover"].astype(float)
    return gross - turnover * float(cost_bps) / 10000.0


def segment_metrics(equity: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, cost_bps: float, horizon: int) -> dict[str, float]:
    frame = equity.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame[(frame["timestamp"] >= start) & (frame["timestamp"] < end)].sort_values("timestamp")
    if frame.empty:
        return {"periods": 0, "return": np.nan, "sharpe": np.nan, "max_drawdown": np.nan, "turnover": np.nan}
    ret = adjusted_return_series(frame, cost_bps)
    curve = (1.0 + ret).cumprod()
    dd = curve / curve.cummax() - 1.0
    std = ret.std(ddof=0)
    sharpe = float(ret.mean() / std * math.sqrt(365 * 24 / horizon)) if std > 0 else np.nan
    return {
        "periods": int(len(frame)),
        "return": float(curve.iloc[-1] - 1.0),
        "sharpe": sharpe,
        "max_drawdown": float(dd.min()),
        "turnover": float(frame["turnover"].sum()),
    }


def build_q2_q3_summary(results: dict[str, dict[str, Any]], costs: list[float], horizon: int) -> pd.DataFrame:
    start = pd.Timestamp("2024-04-01", tz="UTC")
    end = pd.Timestamp("2024-10-01", tz="UTC")
    rows = []
    for stage, result in results.items():
        for cost in costs:
            metrics = segment_metrics(result["equity"], start, end, cost, horizon)
            rows.append({"stage": stage, "cost_bps": cost, **metrics})
    return pd.DataFrame(rows)


def build_fold_stability(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for stage, result in results.items():
        folds = result.get("folds", pd.DataFrame())
        if folds.empty or "total_return" not in folds:
            continue
        ret = folds["total_return"].astype(float)
        rows.append(
            {
                "stage": stage,
                "fold_count": int(len(ret)),
                "positive_fold_rate": float((ret > 0).mean()),
                "median_fold_return": float(ret.median()),
                "q25_fold_return": float(ret.quantile(0.25)),
                "worst_fold_return": float(ret.min()),
                "worst_5_mean_return": float(ret.nsmallest(min(5, len(ret))).mean()),
            }
        )
    return pd.DataFrame(rows)


def model_from_stage(stage: str) -> str:
    return stage.split("__", 1)[0]


def protocol_from_stage(stage: str) -> str:
    if "__" not in stage:
        return ""
    return "p20_" + stage.split("__", 1)[1]


def build_scorecard(
    diagnostics: pd.DataFrame,
    collinearity: pd.DataFrame,
    simple_ic: pd.DataFrame,
    selected_summary: pd.DataFrame,
    portfolio: pd.DataFrame,
    cost: pd.DataFrame,
    regime: pd.DataFrame,
    fold_stability: pd.DataFrame,
    primary_protocol: str,
) -> pd.DataFrame:
    rows = []
    p15c_q = regime[(regime["stage"] == "p15c") & (regime["cost_bps"] == 8.0)]
    p15c_q_ret = float(p15c_q["return"].iloc[0]) if len(p15c_q) else np.nan
    for model in MODEL_ORDER:
        stage = f"{model}__{primary_protocol.replace('p20_', '')}"
        diag = diagnostics[diagnostics["model"] == model]
        coll = collinearity[collinearity["model"] == model]
        ic = simple_ic[(simple_ic["model"] == model) & (simple_ic["factor"] == "rm__score_neg_resid_z")]
        sel = selected_summary[(selected_summary["model"] == model) & (selected_summary["protocol"] == primary_protocol)]
        pf = portfolio[portfolio["stage"] == stage]
        cost_col = "one_way_cost_bps" if "one_way_cost_bps" in cost.columns else "cost_bps"
        c8 = cost[(cost["stage"] == stage) & (cost[cost_col] == 8.0)]
        c20 = cost[(cost["stage"] == stage) & (cost[cost_col] == 20.0)]
        q8 = regime[(regime["stage"] == stage) & (regime["cost_bps"] == 8.0)]
        fold = fold_stability[fold_stability["stage"] == stage]
        rows.append(
            {
                "model": model,
                "label": MODEL_LABELS[model],
                "factor_count": float(coll["factor_count"].median()) if len(coll) else np.nan,
                "max_factor_corr_median": float(coll["max_abs_factor_corr"].median()) if len(coll) else np.nan,
                "max_factor_corr_worst": float(coll["max_abs_factor_corr"].max()) if len(coll) else np.nan,
                "condition_number_median": float(coll["condition_number"].median()) if len(coll) else np.nan,
                "condition_number_worst": float(coll["condition_number"].max()) if len(coll) else np.nan,
                "bnb_condition_number": float(coll.loc[coll["symbol"] == "BNBUSDT", "condition_number"].iloc[0]) if (coll["symbol"] == "BNBUSDT").any() else np.nan,
                "beta_mean_abs_1h_change": float(diag["beta_mean_abs_1h_change"].median()) if len(diag) else np.nan,
                "residual_vol_cv": float(diag["residual_vol_cv"].median()) if len(diag) else np.nan,
                "xs_dispersion_cv": float(diag["xs_dispersion_cv"].median()) if len(diag) else np.nan,
                "simple_mean_reversion_ic": float(ic["mean_ic"].iloc[0]) if len(ic) else np.nan,
                "active_fold_rate": float(sel["active_fold_rate"].iloc[0]) if len(sel) else np.nan,
                "selected_test_ic_mean": float(sel["selected_test_ic_mean"].iloc[0]) if len(sel) else np.nan,
                "oos_return_6bps": float(pf["total_return"].iloc[0]) if len(pf) else np.nan,
                "oos_gross_return": float(pf["gross_total_return_before_cost"].iloc[0]) if len(pf) else np.nan,
                "oos_sharpe_6bps": float(pf["sharpe"].iloc[0]) if len(pf) else np.nan,
                "oos_max_drawdown_6bps": float(pf["max_drawdown"].iloc[0]) if len(pf) else np.nan,
                "oos_return_8bps": float(c8["total_return"].iloc[0]) if len(c8) else np.nan,
                "oos_return_20bps": float(c20["total_return"].iloc[0]) if len(c20) else np.nan,
                "q2_q3_return_8bps": float(q8["return"].iloc[0]) if len(q8) else np.nan,
                "q2_q3_delta_vs_p15c_8bps": float(q8["return"].iloc[0] - p15c_q_ret) if len(q8) else np.nan,
                "q2_q3_max_drawdown_8bps": float(q8["max_drawdown"].iloc[0]) if len(q8) else np.nan,
                "positive_fold_rate": float(fold["positive_fold_rate"].iloc[0]) if len(fold) else np.nan,
                "worst_5_mean_return": float(fold["worst_5_mean_return"].iloc[0]) if len(fold) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def plot_scorecard(output_dir: Path, scorecard: pd.DataFrame) -> None:
    metrics = [
        ("condition_number_median", "Median condition number", False),
        ("beta_mean_abs_1h_change", "Beta 1h change", False),
        ("residual_vol_cv", "Residual vol CV", False),
        ("selected_test_ic_mean", "Selected test IC", True),
        ("oos_return_8bps", "OOS return @8bps", True),
        ("q2_q3_return_8bps", "2024Q2-Q3 return @8bps", True),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    labels = scorecard["label"].tolist()
    colors = ["#2f6f8f", "#4f8b66", "#9a6b38", "#875f8f", "#4776a8", "#a64f4f"]
    for ax, (col, title, percentage) in zip(axes.flat, metrics):
        values = scorecard[col].to_numpy(dtype=float)
        ax.barh(labels, values, color=colors)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.2)
        if percentage:
            ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    fig.suptitle("P25 Residual Model Bake-off — fixed P20 balanced protocol", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_dir / "bakeoff_scorecard.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_primary_curves(output_dir: Path, results: dict[str, dict[str, Any]], primary_protocol: str, costs: list[float]) -> None:
    for cost in costs:
        fig, ax = plt.subplots(figsize=(14, 7))
        for model in MODEL_ORDER:
            stage = f"{model}__{primary_protocol.replace('p20_', '')}"
            equity = results[stage]["equity"].sort_values("timestamp")
            curve = (1.0 + adjusted_return_series(equity, cost)).cumprod() - 1.0
            ax.plot(pd.to_datetime(equity["timestamp"], utc=True), curve, label=MODEL_LABELS[model], linewidth=1.6)
        p15c_equity = results["p15c"]["equity"].sort_values("timestamp")
        p15c_curve = (1.0 + adjusted_return_series(p15c_equity, cost)).cumprod() - 1.0
        ax.plot(pd.to_datetime(p15c_equity["timestamp"], utc=True), p15c_curve, label="P15c reference", linewidth=2.0, color="black", alpha=0.7)
        ax.axvspan(pd.Timestamp("2024-04-01", tz="UTC"), pd.Timestamp("2024-10-01", tz="UTC"), color="#d99a3a", alpha=0.12, label="2024Q2-Q3")
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_title(f"P25 nested OOS equity — P20 balanced — {cost:g}bps one-way cost")
        ax.set_ylabel("Cumulative return")
        ax.grid(alpha=0.2)
        ax.legend(ncol=2, fontsize=9)
        fig.tight_layout()
        fig.savefig(output_dir / f"primary_oos_curves_{cost:g}bps.png", dpi=170, bbox_inches="tight")
        plt.close(fig)


def plot_regime_comparison(output_dir: Path, regime: pd.DataFrame, primary_protocol: str) -> None:
    suffix = primary_protocol.replace("p20_", "")
    stages = [f"{model}__{suffix}" for model in MODEL_ORDER] + ["p15c"]
    data = regime[(regime["stage"].isin(stages)) & (regime["cost_bps"] == 8.0)].copy()
    data["label"] = data["stage"].map(lambda value: MODEL_LABELS.get(model_from_stage(value), MODEL_LABELS.get(value, value)))
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#2f6f8f" if value >= 0 else "#a64f4f" for value in data["return"]]
    ax.barh(data["label"], data["return"], color=colors)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("2024Q2-Q3 nested OOS return — 8bps")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "regime_2024q2_q3_8bps.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def write_report(
    output_dir: Path,
    scorecard: pd.DataFrame,
    diagnostics: pd.DataFrame,
    collinearity: pd.DataFrame,
    selected_summary: pd.DataFrame,
    fold_stability: pd.DataFrame,
) -> None:
    ranked = scorecard.sort_values(["oos_return_8bps", "oos_sharpe_6bps"], ascending=False)
    winner = ranked.iloc[0]
    best_condition = scorecard.sort_values("condition_number_median").iloc[0]
    best_beta = scorecard.sort_values("beta_mean_abs_1h_change").iloc[0]
    best_regime = scorecard.sort_values("q2_q3_return_8bps", ascending=False).iloc[0]
    cost = pd.read_csv(output_dir / "cost_sensitivity.csv")
    cost_col = "one_way_cost_bps" if "one_way_cost_bps" in cost.columns else "cost_bps"
    sensitivity_models = ["legacy_no_sector", "legacy_orthogonalized", "crypto_ff_beta_shrink"]
    sensitivity_protocols = ["balanced_q25_m3", "practical_q25_m5", "loose_decay_corr"]
    lines = [
        "# P25 Residual Model Bake-off 报告",
        "",
        "## 实验口径",
        "",
        "六套 residualizer 统一使用 720h 滚动窗口、240 条最小观测、ex-ante labels、同一套 P20/P21 family library、strict-neutral allocator 和仅拼接 test 的 nested OOS 流程。P20 balanced 是查看结果前固定的主比较协议，其他协议只用于敏感性检查。",
        "",
        "## 核心结果",
        "",
        f"- 固定 balanced 协议下，8bps OOS 收益最高：`{winner['model']}`（{winner['oos_return_8bps']:.2%}）。",
        f"- factor condition number 中位数最低：`{best_condition['model']}`（{best_condition['condition_number_median']:.2f}）。",
        f"- beta 每小时变动最低：`{best_beta['model']}`（{best_beta['beta_mean_abs_1h_change']:.6f}）。",
        f"- 2024Q2-Q3 的 8bps 收益最好：`{best_regime['model']}`（{best_regime['q2_q3_return_8bps']:.2%}）。",
        "",
        "## 主比较结果",
        "",
        "| 模型 | 条件数中位数 | beta 变动 | residual vol CV | selected IC | active folds | OOS gross | OOS @8bps | Sharpe @6bps | maxDD @6bps | 2024Q2-Q3 @8bps |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in scorecard.iterrows():
        lines.append(
            f"| {row['model']} | {row['condition_number_median']:.2f} | {row['beta_mean_abs_1h_change']:.6f} | "
            f"{row['residual_vol_cv']:.3f} | {row['selected_test_ic_mean']:.4f} | {row['active_fold_rate']:.1%} | "
            f"{row['oos_gross_return']:.2%} | {row['oos_return_8bps']:.2%} | {row['oos_sharpe_6bps']:.3f} | "
            f"{row['oos_max_drawdown_6bps']:.2%} | {row['q2_q3_return_8bps']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## 协议敏感性：8bps OOS return",
            "",
            "| 模型 | Balanced | Practical | Loose |",
            "|---|---:|---:|---:|",
        ]
    )
    for model in sensitivity_models:
        values = []
        for protocol in sensitivity_protocols:
            stage = f"{model}__{protocol}"
            row = cost[(cost["stage"] == stage) & (cost[cost_col] == 8.0)]
            values.append(float(row["total_return"].iloc[0]) if len(row) else np.nan)
        lines.append(f"| {model} | {values[0]:.2%} | {values[1]:.2%} | {values[2]:.2%} |")
    lines.extend(
        [
            "",
            "## Review 结论",
            "",
            "- `legacy_orthogonalized` 是结构诊断赢家：condition number 从 legacy 的 155.36 降到 1.07，并在 fixed balanced 下把 2024Q2-Q3 的 8bps 收益从 P15c 的 -4.54% 改善到 +1.08%。但它对 P20 协议较敏感，放宽到 practical/loose 后全样本表现明显退化。",
            "- `crypto_ff_beta_shrink` 是 balanced 协议下的收益赢家，beta 和 residual volatility 也更稳定；但只激活 3/32 folds，且 2024Q2-Q3 仍为 -1.85%，证据宽度不足，不能仅凭 +14.10% 升级 baseline。",
            "- `legacy_no_sector` 是最值得继续验证的简化候选。它消除了 BNB 的近奇异 sector 结构，condition number 降到约 20.62；在 practical/loose 下 8bps OOS 分别为 +12.96%/+13.01%，且 2024Q2-Q3 分别为 +6.15%/+10.19%。不过 balanced 结果较弱，下一步必须预注册协议后再做独立验证。",
            "- 单一 `-resid_z` 的 mean-reversion IC 在六套模型里都接近 0。真正产生区分度的是 residual quality、beta stability 和 slow residual family，经 P20/P21 选择后再接 allocator。",
            "- `legacy_ridge` 证明 estimator shrinkage 确实能稳定 beta，但并未修复 factor 共线性，也没有转化成更好的 OOS alpha；ridge 不能替代 factor design。",
            "",
            "## 解读约束",
            "",
            "- Ridge 可以稳定 beta，但不会降低原始 factor correlation；factor design 和 estimator design 必须分开评价。",
            "- condition number 低不是充分条件，仍需同时改善 selected-label IC 和 portfolio OOS。",
            "- active-fold rate 低意味着大部分时间仍是 P15c fallback，因此组合提升的证据覆盖面有限。",
            "- 2024Q2-Q3 是事先固定的压力窗口，只用于评估，没有参与参数选择。",
            "",
            "## 主要文件",
            "",
            "- `bakeoff_scorecard.csv`：固定主协议的总比较。",
            "- `factor_collinearity.csv`：逐币/全局 factor correlation 与 condition number。",
            "- `residual_diagnostics.csv`：beta、residual volatility 与横截面 dispersion 稳定性。",
            "- `simple_label_ic.csv`、`family_fold_metrics.csv`：label IC 与 P20/P21 诊断。",
            "- `portfolio_summary.csv`、`cost_sensitivity.csv`、`fold_stability.csv`：nested OOS 组合诊断。",
            "- `regime_2024q2_q3_summary.csv`：固定压力窗口结果。",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
