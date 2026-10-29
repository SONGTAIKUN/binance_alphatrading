#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from scipy.stats import norm, spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from crypto_stat_arb.config import ensure_dir, load_config
import run_p3d_p4_gated_rerank_alphagpt as p3d
import run_p4_nested_walkforward_selection as p4
import run_p7_derivatives_gated_rerank_alphagpt as p7
import run_p9_clean_derivatives_baseline as p9
import run_p10d_nested_allocator_short_guard as p10d
import run_p15c_extreme_cluster_turnover as p15c
import run_p27_no_sector_native_factor_remine as p27
from run_p18c_factor_selector_portfolio import load_existing_result
from run_p26_full_nested_pipeline import combine_test_results, parallel_fold_map, validation_split
from run_turnover_experiments import to_jsonable


EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "baseline_locked_v1"
DEFAULT_CONFIG = EXPERIMENT_ROOT / "configs" / "baseline_locked_v1.yaml"
LONG_ROOT = EXPERIMENT_ROOT / "outputs" / "long_history_nested_oos"
P27_ROOT = EXPERIMENT_ROOT / "outputs" / "p27_no_sector_native_factor_remine"
DEFAULT_NATIVE_PANEL = P27_ROOT / "inputs" / "p27_no_sector_native_feature_panel.pkl"
DEFAULT_FOLDS = LONG_ROOT / "inputs" / "long_formula_selections.csv"
DEFAULT_P9_SELECTED = LONG_ROOT / "p9_nested" / "selected_incremental_by_fold.csv"
DEFAULT_P15C_SELECTED = LONG_ROOT / "p15c_nested" / "selected_extreme_cluster_by_fold.csv"
DEFAULT_P15C_RUN = LONG_ROOT / "p15c_nested" / "nested_extreme_cluster_run"
DEFAULT_RISK_RESIDUAL = (
    EXPERIMENT_ROOT / "outputs" / "p26_full_nested_pipeline" / "inputs" / "p15c_legacy_residual_wide.pkl"
)
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "outputs" / "p28_stable_family_ensemble"
LONG_CONFIG = LONG_ROOT / "inputs" / "long_nested_config.yaml"
LONG_ASSET = EXPERIMENT_ROOT / "inputs" / "long_history" / "baseline_panel" / "asset_residual_panel.csv"
LONG_MANUAL = LONG_ROOT / "inputs" / "manual_slow_feature_panel_long.csv.gz"
LONG_P6B = LONG_ROOT / "inputs" / "p6b_feature_panel_long.csv.gz"
LONG_RETAINED = LONG_ROOT / "inputs" / "retained_features_all_predefined.csv"

HORIZON = 72
MAX_LABEL_HORIZON = 168
FAMILY_SMOOTHING_ALPHA = 0.25
TRAIN_BLOCK_DAYS = 28
VALIDATION_BLOCK_DAYS = 14
TRAIN_POSITIVE_RATE_FLOOR = 0.60
TRAIN_Q25_FLOOR = -0.03
TRAIN_MEAN_IC_FLOOR = 0.005
VALIDATION_Q25_FLOOR = -0.05
VALIDATION_MEAN_IC_FLOOR = 0.0
FAMILY_WEIGHT_CAP = 0.10


@dataclass(frozen=True)
class FamilyDefinition:
    name: str
    description: str
    members: tuple[p27.NativeFormula, ...]


@dataclass(frozen=True)
class BlendVariant:
    name: str
    method: str
    target_native_weight: float
    minimum_baseline_weight: float


_PANEL: pd.DataFrame | None = None
_CFG: dict[str, Any] | None = None
_FAMILIES: dict[str, FamilyDefinition] = {}
_P9_SELECTED: pd.DataFrame | None = None
_P15C_SELECTED: pd.DataFrame | None = None
_RISK_RESIDUAL: pd.DataFrame | None = None
_FAMILY_SELECTION: pd.DataFrame | None = None
_FAMILY_VALIDATION: pd.DataFrame | None = None
_FAMILY_OOS: pd.DataFrame | None = None


def main() -> None:
    parser = argparse.ArgumentParser(description="P28 stable family ensemble and nested OOS selector.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--native-panel", default=str(DEFAULT_NATIVE_PANEL))
    parser.add_argument("--folds", default=str(DEFAULT_FOLDS))
    parser.add_argument("--p9-selected", default=str(DEFAULT_P9_SELECTED))
    parser.add_argument("--p15c-selected", default=str(DEFAULT_P15C_SELECTED))
    parser.add_argument("--p15c-run", default=str(DEFAULT_P15C_RUN))
    parser.add_argument("--risk-residual", default=str(DEFAULT_RISK_RESIDUAL))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument("--audit-bootstrap", type=int, default=2000)
    parser.add_argument("--cost-bps", type=float, nargs="+", default=[0, 6, 8, 15, 20, 30])
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_dir = ensure_dir(Path(args.output_dir).resolve())
    input_dir = ensure_dir(output_dir / "inputs")
    config_path = Path(args.config).resolve()
    folds = pd.read_csv(
        Path(args.folds).resolve(),
        parse_dates=["train_start", "train_end", "test_start", "test_end"],
    ).sort_values("fold")
    selected_p9 = pd.read_csv(Path(args.p9_selected).resolve())
    selected_p15c = pd.read_csv(Path(args.p15c_selected).resolve())
    families = build_family_definitions()
    variants = build_variants()
    write_family_definitions(output_dir, families)
    pd.DataFrame([item.__dict__ for item in variants]).to_csv(output_dir / "blend_variants.csv", index=False)

    prepared_path = input_dir / "p28_prepared_panel.pkl"
    print("[1/7] Preparing exact P15c fold scores on the no-sector native panel", flush=True)
    if args.force_prepare or not prepared_path.exists():
        build_prepared_panel(
            native_path=Path(args.native_panel).resolve(),
            output_path=prepared_path,
            selected_p9=selected_p9,
        )
    else:
        print("      using cached prepared panel", flush=True)

    print("[2/7] P28a: selection-overfit and effective-dimension audit", flush=True)
    run_p28a_audit(
        panel_path=prepared_path,
        folds=folds,
        output_dir=ensure_dir(output_dir / "p28a_selection_overfit_audit"),
        bootstrap_samples=int(args.audit_bootstrap),
    )

    workers = max(1, int(args.workers))
    family_selection_path = output_dir / "p28b_family_selection_by_fold.csv"
    family_validation_path = output_dir / "p28b_family_validation_metrics.csv"
    family_oos_path = output_dir / "p28b_family_test_oos_metrics.csv"
    member_path = output_dir / "p28b_member_stability.csv"
    print(f"[3/7] P28b: train-only family stability committees ({len(folds)} folds, {workers} workers)", flush=True)
    if not args.force and all(path.exists() for path in [family_selection_path, family_validation_path, family_oos_path, member_path]):
        print("      using cached family diagnostics", flush=True)
        family_selection = pd.read_csv(family_selection_path)
        family_validation = pd.read_csv(family_validation_path)
        family_oos = pd.read_csv(family_oos_path)
        member_stability = pd.read_csv(member_path)
    else:
        family_results = parallel_fold_map(
            worker=family_diagnostic_worker,
            folds=folds,
            workers=workers,
            initializer=init_family_worker,
            initargs=(str(prepared_path), families),
        )
        member_stability = pd.concat([item["members"] for item in family_results], ignore_index=True)
        family_selection = pd.concat([item["selection"] for item in family_results], ignore_index=True)
        family_validation = pd.concat([item["validation"] for item in family_results], ignore_index=True)
        family_oos = pd.concat([item["test"] for item in family_results], ignore_index=True)
        member_stability.to_csv(member_path, index=False)
        family_selection.to_csv(family_selection_path, index=False)
        family_validation.to_csv(family_validation_path, index=False)
        family_oos.to_csv(family_oos_path, index=False)
    family_summary = summarize_family_oos(family_oos, family_selection)
    family_summary.to_csv(output_dir / "p28b_family_oos_summary.csv", index=False)

    print("[4/7] P28c/P28d: nested validation and fixed-variant OOS replay", flush=True)
    trade_cache = output_dir / "p28_trade_worker_complete.flag"
    selected_variant_path = output_dir / "p28e_selected_variant_by_fold.csv"
    validation_path = output_dir / "p28cd_validation_metrics.csv"
    weight_path = output_dir / "p28cd_family_weights_by_fold.csv"
    result_dirs = {item.name: output_dir / item.name / "nested_test_run" for item in variants}
    result_dirs["P28e_nested"] = output_dir / "P28e_nested" / "nested_test_run"
    replay_dir = output_dir / "P15c_replay_check" / "nested_test_run"
    cached = (
        not args.force
        and trade_cache.exists()
        and selected_variant_path.exists()
        and validation_path.exists()
        and weight_path.exists()
        and all((path / "metrics.json").exists() for path in result_dirs.values())
        and (replay_dir / "metrics.json").exists()
    )
    if cached:
        print("      using cached P28c-e trading results", flush=True)
        validation = pd.read_csv(validation_path)
        selected_variant = pd.read_csv(selected_variant_path)
        family_weights = pd.read_csv(weight_path)
        results = {name: load_existing_result(path) for name, path in result_dirs.items()}
        baseline_replay = load_existing_result(replay_dir)
    else:
        trade_results = parallel_fold_map(
            worker=trade_fold_worker,
            folds=folds,
            workers=workers,
            initializer=init_trade_worker,
            initargs=(
                str(prepared_path),
                str(config_path),
                families,
                selected_p9,
                selected_p15c,
                str(Path(args.risk_residual).resolve()),
                family_selection,
                family_validation,
                family_oos,
            ),
        )
        validation = pd.concat([item["validation"] for item in trade_results], ignore_index=True)
        selected_variant = pd.concat([item["selected"] for item in trade_results], ignore_index=True).sort_values("fold")
        family_weights = pd.concat([item["weights"] for item in trade_results], ignore_index=True)
        validation.to_csv(validation_path, index=False)
        selected_variant.to_csv(selected_variant_path, index=False)
        family_weights.to_csv(weight_path, index=False)
        results = {}
        for variant in variants:
            result = combine_test_results([item["tests"][variant.name] for item in trade_results])
            results[variant.name] = result
            p3d.write_run_result(ensure_dir(result_dirs[variant.name]), result)
        nested = combine_test_results([item["nested"] for item in trade_results])
        results["P28e_nested"] = nested
        p3d.write_run_result(ensure_dir(result_dirs["P28e_nested"]), nested)
        baseline_replay = combine_test_results([item["baseline_replay"] for item in trade_results])
        p3d.write_run_result(ensure_dir(replay_dir), baseline_replay)
        trade_cache.write_text("complete\n", encoding="utf-8")

    print("[5/7] P28e: matched OOS summaries, chronology audit, and acceptance checks", flush=True)
    p15c_result = load_existing_result(Path(args.p15c_run).resolve())
    replay_check = compare_baseline_replay(p15c_result, baseline_replay)
    replay_check.to_csv(output_dir / "p15c_replay_equivalence_check.csv", index=False)
    if not replay_check["pass"].all():
        raise RuntimeError("P28 P15c replay is not equivalent to the locked baseline")
    all_results = {"P15c": p15c_result, **results}
    summary = p27.build_trading_summary(all_results)
    cost = p3d.build_cost_sensitivity(all_results, [float(value) for value in args.cost_bps])
    regime = p27.build_regime_comparison(all_results, [0.0, 8.0])
    folds_out = build_fold_comparison(all_results)
    summary.to_csv(output_dir / "p28_matched_trading_summary.csv", index=False)
    cost.to_csv(output_dir / "p28_cost_sensitivity.csv", index=False)
    regime.to_csv(output_dir / "p28_quarterly_regime.csv", index=False)
    folds_out.to_csv(output_dir / "p28_fold_comparison.csv", index=False)
    audit = build_protocol_audit(folds, family_selection, family_validation, selected_variant, results["P28e_nested"])
    audit.to_csv(output_dir / "p28_nested_protocol_audit.csv", index=False)
    if not audit["all_pass"].all():
        raise RuntimeError("P28 nested chronology audit failed")
    acceptance = build_acceptance_table(
        summary, cost, folds_out, regime, family_oos, selected_variant,
        output_dir / "p28a_selection_overfit_audit" / "pbo_summary.csv",
    )
    acceptance.to_csv(output_dir / "p28e_acceptance_checks.csv", index=False)

    print("[6/7] Writing charts", flush=True)
    plot_family_ic(output_dir, family_summary)
    plot_pnl(output_dir, all_results)
    plot_variant_selection(output_dir, selected_variant)

    print("[7/7] Writing Chinese review and manifest", flush=True)
    write_report(
        output_dir=output_dir,
        family_summary=family_summary,
        member_stability=member_stability,
        validation=validation,
        selected_variant=selected_variant,
        family_weights=family_weights,
        summary=summary,
        cost=cost,
        regime=regime,
        folds_out=folds_out,
        audit=audit,
        acceptance=acceptance,
    )
    manifest = {
        "experiment": "P28 stable economic-family ensemble",
        "stages": ["P28a", "P28b", "P28c", "P28d", "P28e"],
        "outer_folds": int(len(folds)),
        "workers": workers,
        "family_smoothing_alpha": FAMILY_SMOOTHING_ALPHA,
        "train_block_days": TRAIN_BLOCK_DAYS,
        "validation_block_days": VALIDATION_BLOCK_DAYS,
        "family_weight_cap": FAMILY_WEIGHT_CAP,
        "baseline_anchor_floor": 0.70,
        "factor_label_closure_hours": MAX_LABEL_HORIZON + 1,
        "portfolio_horizon_hours": HORIZON,
        "selection": "train stability -> closed-label validation family weighting -> validation variant/fallback -> next test",
        "important_scope": "P28 family hypotheses were designed after observing P27 and therefore require a future untouched sample before live promotion.",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(to_jsonable(manifest), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Done: {output_dir}", flush=True)


def make_formula(
    name: str,
    op: str,
    left: str,
    right: str | None = None,
    gate: str | None = None,
    weight: float = 1.0,
) -> p27.NativeFormula:
    if op == "single":
        expression = left
        primitives = (left,)
        complexity = 1
    elif op == "neg":
        expression = f"NEG({left})"
        primitives = (left,)
        complexity = 2
    elif op in {"add", "sub"}:
        expression = f"{'ADD' if op == 'add' else 'SUB'}({left},{weight:.2f}*{right})"
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
    return p27.NativeFormula(name, op, left, right, gate, weight, complexity, expression, primitives)


def build_family_definitions() -> dict[str, FamilyDefinition]:
    definitions = [
        FamilyDefinition(
            "R1_slow_anchor_reversion",
            "Slow anchor reversion conditioned by observable stationarity.",
            (
                make_formula("R1_01", "gate", "nf_d_reversion_168", gate="nf_q_stationarity_168"),
                make_formula("R1_02", "gate", "nf_d_reversion_336", gate="nf_q_stationarity_336"),
                make_formula("R1_03", "gated_add", "nf_d_reversion_336", "nf_d_reversion_504", "nf_q_stationarity_336", 0.50),
                make_formula("R1_04", "add", "nf_d_reversion_168", "nf_d_reversion_336", weight=0.50),
            ),
        ),
        FamilyDefinition(
            "M1_common_adjusted_momentum",
            "Slow residual continuation after removing the ex-self residual common component.",
            (
                make_formula("M1_01", "gate", "nf_d_common_adj_mom_168", gate="nf_q_purity_168"),
                make_formula("M1_02", "gate", "nf_d_common_adj_mom_336", gate="nf_q_purity_336"),
                make_formula("M1_03", "gate", "nf_d_mom_168_24", gate="nf_q_beta_stability_168"),
                make_formula("M1_04", "gated_add", "nf_d_common_adj_mom_336", "nf_d_mom_336_24", "nf_q_purity_336", 0.50),
            ),
        ),
        FamilyDefinition(
            "T1_residual_trend_quality",
            "Range location and directional efficiency at 7d-14d horizons.",
            (
                make_formula("T1_01", "add", "nf_d_highloc_168", "nf_d_trend_eff_168", weight=0.50),
                make_formula("T1_02", "add", "nf_d_highloc_336", "nf_d_trend_eff_336", weight=0.50),
                make_formula("T1_03", "gated_add", "nf_d_highloc_336", "nf_d_sign_persist_336", "nf_q_beta_stability_336", 0.50),
                make_formula("T1_04", "gated_add", "nf_d_mom_336_24", "nf_d_trend_eff_336", "nf_q_purity_336", 0.50),
            ),
        ),
        FamilyDefinition(
            "S1_downside_snapback",
            "Slow snapback after asymmetric residual downside, without a 1h reversal primitive.",
            (
                make_formula("S1_01", "neg", "nf_d_low_downside_share"),
                make_formula("S1_02", "sub", "nf_d_reversion_168", "nf_d_low_downside_share", weight=0.50),
                make_formula("S1_03", "sub", "nf_d_reversion_336", "nf_d_low_downside_share", weight=0.50),
            ),
        ),
        FamilyDefinition(
            "L1_liquidity_conditioned",
            "Slow volume-share migration and liquid common-adjusted momentum.",
            (
                make_formula("L1_01", "single", "nf_d_volume_share_trend"),
                make_formula("L1_02", "gate", "nf_d_volume_share_trend", gate="nf_q_liquidity"),
                make_formula("L1_03", "gate", "nf_d_common_adj_mom_168", gate="nf_q_low_amihud"),
                make_formula("L1_04", "gated_add", "nf_d_common_adj_mom_168", "nf_d_volume_share_trend", "nf_q_liquidity", 0.30),
            ),
        ),
    ]
    return {item.name: item for item in definitions}


def build_variants() -> list[BlendVariant]:
    return [
        BlendVariant("P28c_equal_30", "equal", 0.30, 0.70),
        BlendVariant("P28c_equal_20", "equal", 0.20, 0.80),
        BlendVariant("P28d_shrink_25", "shrink", 0.25, 0.75),
        BlendVariant("P28d_dma_30", "dma", 0.30, 0.70),
    ]


def write_family_definitions(output_dir: Path, families: dict[str, FamilyDefinition]) -> None:
    rows = []
    for family in families.values():
        for member in family.members:
            rows.append({"family": family.name, "description": family.description, **member.to_row()})
    pd.DataFrame(rows).to_csv(output_dir / "p28b_fixed_family_definitions.csv", index=False)


def configure_long_p7_paths() -> None:
    p7.P6B_PANEL = LONG_P6B
    p7.P6B_RETAINED = LONG_RETAINED
    p7._P7_FEATURE_MAP = {}


def build_prepared_panel(native_path: Path, output_path: Path, selected_p9: pd.DataFrame) -> None:
    native = pd.read_pickle(native_path).sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    configure_long_p7_paths()
    full = p7.load_panel_p7(LONG_CONFIG, LONG_ASSET, LONG_MANUAL, HORIZON)
    full = full.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    if len(native) != len(full) or not native[["timestamp", "symbol"]].equals(full[["timestamp", "symbol"]]):
        raise ValueError("P28 native and long P7 panels are not exactly aligned")
    # The locked P15c execution path must keep every shared trading/risk/guard
    # field from its original long-history P7 panel.  Only no-sector-native
    # features and their namespaced centered labels are retained from P27.
    preserve_native = {
        column
        for column in native.columns
        if column.startswith("nf_")
        or column.startswith("label_resid_rank_centered_h")
        or column.startswith("label_path_rank_centered_h")
        or column in {
            "label_slow_low_turnover_rank_centered",
            "label_path_slow_low_turnover_rank_centered",
        }
    }
    baseline_columns = [
        column
        for column in native.columns
        if column in full.columns and column not in preserve_native and column not in {"timestamp", "symbol"}
    ]
    native.loc[:, baseline_columns] = full[baseline_columns].to_numpy()
    retained = p7.load_retained_features()
    deriv_formulas = {
        item.name: item
        for item in p9.filter_p9_formulas(p7.build_p7_formula_universe(retained, 160, 2607, 3), exclude_k01=True)
    }
    p3d_formulas = {item.name: item for item in p3d.build_incremental_formula_universe(180, 2417, 3)}
    spec_by_name = {item.name: item for item in p3d.build_committee_specs()}
    for row in selected_p9.sort_values("fold").itertuples(index=False):
        fold_id = int(row.fold)
        series = pd.Series(row._asdict())
        committee = spec_by_name[str(series["p4_spec"])]
        p10d.prepare_fold_score(full, series, committee, deriv_formulas, p3d_formulas)
        native[fold_score_column("p15c", fold_id)] = full[p3d.SCORE_COL].to_numpy(dtype=np.float64)
        native[fold_score_column("p4", fold_id)] = full[p3d.P4_SCORE_COL].to_numpy(dtype=np.float64)
        print(f"      baseline score fold {fold_id}/{len(selected_p9)}", flush=True)
    native.to_pickle(output_path)


def fold_score_column(kind: str, fold: int) -> str:
    return f"_p28_{kind}_score_f{fold:02d}"


def run_p28a_audit(panel_path: Path, folds: pd.DataFrame, output_dir: Path, bootstrap_samples: int) -> None:
    required = [
        output_dir / "grammar_effective_dimension.csv",
        output_dir / "pbo_summary.csv",
        output_dir / "formula_oos_ic_matrix.csv",
    ]
    if all(path.exists() for path in required):
        print("      using cached P28a audit", flush=True)
        return
    panel = pd.read_pickle(panel_path)
    primitive_rows = pd.read_csv(P27_ROOT / "native_primitives.csv").to_dict("records")
    formulas = p27.build_formula_universe(primitive_rows, p27.FORMULA_COUNT, p27.FORMULA_SEED)
    formula_map = {item.name: item for item in formulas}

    timestamps = np.array(sorted(panel["timestamp"].unique()))[::MAX_LABEL_HORIZON]
    sampled = panel.loc[panel["timestamp"].isin(timestamps)].copy()
    signal_matrix = {}
    for formula in formulas:
        signal_matrix[formula.name] = p27.evaluate_formula(formula, sampled).to_numpy(dtype=np.float32)
    signal_frame = pd.DataFrame(signal_matrix)
    corr = signal_frame.corr().fillna(0.0)
    corr.to_csv(output_dir / "grammar_signal_correlation.csv")
    eigenvalues = np.linalg.eigvalsh(corr.to_numpy(dtype=float))
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    effective_count = float(eigenvalues.sum() ** 2 / np.square(eigenvalues).sum())
    high_corr_pairs = int((np.triu(np.abs(corr.to_numpy()), 1) >= 0.90).sum())
    pd.DataFrame(
        [{
            "nominal_formula_count": len(formulas),
            "effective_independent_formula_count": effective_count,
            "effective_fraction": effective_count / len(formulas),
            "abs_corr_ge_090_pairs": high_corr_pairs,
            "sample_timestamps": len(timestamps),
            "diagnostic_only": True,
        }]
    ).to_csv(output_dir / "grammar_effective_dimension.csv", index=False)

    selected = pd.read_csv(P27_ROOT / "selected_native_formula_by_fold.csv")
    names = sorted(set(selected["formula_name"]))
    oos = pd.DataFrame(index=folds["fold"].astype(int), columns=names, dtype=float)
    for name in names:
        score = p27.formula_score(formula_map[name], panel, FAMILY_SMOOTHING_ALPHA)
        panel["_audit_score"] = score
        for fold in folds.itertuples(index=False):
            start = pd.Timestamp(fold.test_start)
            end = p4.decision_end_for_closed_positions(pd.Timestamp(fold.test_end), MAX_LABEL_HORIZON)
            scoped = panel.loc[(panel["timestamp"] >= start) & (panel["timestamp"] < end)]
            oos.loc[int(fold.fold), name] = p27.mean_timestamp_ic(
                scoped, "_audit_score", "label_slow_low_turnover_rank_centered"
            )
    oos.to_csv(output_dir / "formula_oos_ic_matrix.csv")
    pbo = probability_of_backtest_overfit(oos, bootstrap_samples, seed=2828)
    pd.DataFrame([pbo]).to_csv(output_dir / "pbo_summary.csv", index=False)
    dsr = deflated_ic_table(oos, effective_count)
    dsr.to_csv(output_dir / "deflated_ic_table.csv", index=False)
    mcs = model_confidence_set_table(oos, bootstrap_samples, seed=2829)
    mcs.to_csv(output_dir / "model_confidence_set_90.csv", index=False)
    rank_decay = validation_test_rank_decay(oos)
    rank_decay.to_csv(output_dir / "validation_test_rank_correlation.csv", index=False)


def probability_of_backtest_overfit(matrix: pd.DataFrame, samples: int, seed: int) -> dict[str, Any]:
    values = matrix.to_numpy(dtype=float)
    n_folds, n_models = values.shape
    rng = np.random.default_rng(seed)
    logits = []
    percentiles = []
    winners = []
    half = n_folds // 2
    for _ in range(samples):
        train_idx = np.sort(rng.choice(n_folds, size=half, replace=False))
        test_idx = np.setdiff1d(np.arange(n_folds), train_idx)
        train_mean = np.nanmean(values[train_idx], axis=0)
        winner = int(np.nanargmax(train_mean))
        test_mean = np.nanmean(values[test_idx], axis=0)
        order = pd.Series(test_mean).rank(pct=True, method="average").to_numpy()
        percentile = float(order[winner])
        percentiles.append(percentile)
        logits.append(float(np.log(np.clip(percentile, 1e-6, 1 - 1e-6) / np.clip(1 - percentile, 1e-6, 1))))
        winners.append(winner)
    return {
        "models": n_models,
        "folds": n_folds,
        "bootstrap_splits": samples,
        "pbo_probability_test_rank_below_median": float(np.mean(np.asarray(percentiles) <= 0.50)),
        "median_test_rank_percentile_of_train_winner": float(np.median(percentiles)),
        "mean_logit": float(np.mean(logits)),
        "distinct_train_winners": int(len(set(winners))),
    }


def deflated_ic_table(matrix: pd.DataFrame, effective_trials: float) -> pd.DataFrame:
    gamma = 0.5772156649015329
    trials = max(2.0, float(effective_trials))
    expected_max = (1.0 - gamma) * norm.ppf(1.0 - 1.0 / trials) + gamma * norm.ppf(1.0 - 1.0 / (trials * np.e))
    rows = []
    for name in matrix.columns:
        values = matrix[name].dropna().to_numpy(dtype=float)
        std = float(np.std(values, ddof=1))
        icir = float(np.mean(values) / std) if std > 0 else np.nan
        skew = float(pd.Series(values).skew()) if len(values) > 2 else 0.0
        kurt = float(pd.Series(values).kurt() + 3.0) if len(values) > 3 else 3.0
        denom = np.sqrt(max(1e-12, 1.0 - skew * icir + (kurt - 1.0) * icir * icir / 4.0))
        z = (icir - expected_max) * np.sqrt(max(1, len(values) - 1)) / denom
        rows.append({
            "formula_name": name,
            "mean_oos_ic": float(np.mean(values)),
            "oos_icir": icir,
            "effective_trials": trials,
            "expected_max_icir_under_multiple_testing": expected_max,
            "deflated_ic_probability": float(norm.cdf(z)),
        })
    return pd.DataFrame(rows).sort_values("deflated_ic_probability", ascending=False)


def moving_block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    out: list[int] = []
    while len(out) < n:
        start = int(rng.integers(0, max(1, n - block + 1)))
        out.extend(range(start, min(n, start + block)))
    return np.asarray(out[:n], dtype=int)


def model_confidence_set_table(matrix: pd.DataFrame, samples: int, seed: int) -> pd.DataFrame:
    values = matrix.to_numpy(dtype=float)
    means = np.nanmean(values, axis=0)
    best = int(np.nanargmax(means))
    rng = np.random.default_rng(seed)
    rows = []
    for idx, name in enumerate(matrix.columns):
        diff = values[:, best] - values[:, idx]
        boot = []
        for _ in range(samples):
            pick = moving_block_indices(len(diff), 3, rng)
            boot.append(float(np.nanmean(diff[pick])))
        lower, upper = np.nanquantile(boot, [0.05, 0.95])
        rows.append({
            "formula_name": name,
            "best_formula": matrix.columns[best],
            "mean_oos_ic": means[idx],
            "mean_ic_gap_to_best": float(np.nanmean(diff)),
            "gap_bootstrap_q05": float(lower),
            "gap_bootstrap_q95": float(upper),
            "mcs90_included": bool(lower <= 0.0),
        })
    return pd.DataFrame(rows).sort_values(["mcs90_included", "mean_oos_ic"], ascending=[False, False])


def validation_test_rank_decay(oos: pd.DataFrame) -> pd.DataFrame:
    validation = pd.read_csv(P27_ROOT / "formula_validation_metrics.csv")
    rows = []
    for fold, frame in validation.groupby("fold"):
        values = frame.groupby("formula_name")["validation_fitness"].max()
        common = values.index.intersection(oos.columns)
        if len(common) < 4 or int(fold) not in oos.index:
            continue
        corr = spearmanr(values.loc[common].to_numpy(), oos.loc[int(fold), common].to_numpy(), nan_policy="omit").statistic
        rows.append({"fold": int(fold), "formulas": len(common), "validation_test_rank_correlation": float(corr)})
    return pd.DataFrame(rows)


def init_family_worker(panel_path: str, families: dict[str, FamilyDefinition]) -> None:
    global _PANEL, _FAMILIES
    _PANEL = pd.read_pickle(panel_path)
    _FAMILIES = families


def family_diagnostic_worker(fold: dict[str, Any]) -> dict[str, pd.DataFrame]:
    if _PANEL is None:
        raise RuntimeError("P28 family worker is not initialized")
    fold_id = int(fold["fold"])
    validation_start = validation_split(fold)
    validation_label_end = p4.decision_end_for_closed_positions(pd.Timestamp(fold["train_end"]), MAX_LABEL_HORIZON)
    train_label_end = validation_start - pd.Timedelta(hours=MAX_LABEL_HORIZON + 1)
    test_label_end = p4.decision_end_for_closed_positions(pd.Timestamp(fold["test_end"]), MAX_LABEL_HORIZON)
    history_start = pd.Timestamp(fold["train_start"]) - pd.Timedelta(hours=504)
    local = _PANEL.loc[
        (_PANEL["timestamp"] >= history_start) & (_PANEL["timestamp"] < pd.Timestamp(fold["test_end"]))
    ].copy()
    train_mask = (local["timestamp"] >= pd.Timestamp(fold["train_start"])) & (local["timestamp"] < train_label_end)
    validation_mask = (local["timestamp"] >= validation_start) & (local["timestamp"] < validation_label_end)
    test_mask = (local["timestamp"] >= pd.Timestamp(fold["test_start"])) & (local["timestamp"] < test_label_end)
    member_rows = []
    selection_rows = []
    validation_rows = []
    test_rows = []
    for family in _FAMILIES.values():
        member_scores: dict[str, pd.Series] = {}
        survivors = []
        for formula in family.members:
            score = p27.evaluate_formula(formula, local)
            member_scores[formula.name] = score
            local["_member_score"] = score
            train_ic = p27.timestamp_ic_series(
                local.loc[train_mask], "_member_score", "label_slow_low_turnover_rank_centered"
            )
            blocks = block_means(train_ic, pd.Timestamp(fold["train_start"]), TRAIN_BLOCK_DAYS)
            path_ic = p27.mean_timestamp_ic(
                local.loc[train_mask], "_member_score", "label_path_slow_low_turnover_rank_centered"
            )
            mean_ic = float(blocks.mean()) if not blocks.empty else np.nan
            q25 = float(blocks.quantile(0.25)) if not blocks.empty else np.nan
            positive_rate = float((blocks > 0.0).mean()) if not blocks.empty else 0.0
            passed = bool(
                len(blocks) >= 3
                and mean_ic >= TRAIN_MEAN_IC_FLOOR
                and q25 >= TRAIN_Q25_FLOOR
                and positive_rate >= TRAIN_POSITIVE_RATE_FLOOR
                and p27.nz(path_ic) >= -0.01
            )
            if passed:
                survivors.append(formula.name)
            member_rows.append({
                "fold": fold_id,
                "family": family.name,
                **formula.to_row(),
                "train_blocks": len(blocks),
                "train_block_mean_ic": mean_ic,
                "train_block_q25_ic": q25,
                "train_positive_block_rate": positive_rate,
                "train_path_ic": path_ic,
                "stability_pass": passed,
                "train_label_end": train_label_end,
            })
        selection_rows.append({
            "fold": fold_id,
            "family": family.name,
            "selected_members": "|".join(survivors),
            "selected_member_count": len(survivors),
            "family_active": bool(survivors),
            "validation_start": validation_start,
            "validation_label_end": validation_label_end,
            "test_start": fold["test_start"],
            "test_label_end": test_label_end,
            "train_label_end": train_label_end,
        })
        if not survivors:
            validation_rows.append(empty_family_metric(fold_id, family.name, "validation"))
            test_rows.append(empty_family_metric(fold_id, family.name, "test"))
            continue
        family_score = family_committee_score(local, [member_scores[name] for name in survivors])
        local["_family_score"] = family_score
        validation_rows.append(
            family_period_metrics(local.loc[validation_mask], fold_id, family.name, "validation", validation_start)
        )
        test_rows.append(
            family_period_metrics(local.loc[test_mask], fold_id, family.name, "test", pd.Timestamp(fold["test_start"]))
        )
    return {
        "members": pd.DataFrame(member_rows),
        "selection": pd.DataFrame(selection_rows),
        "validation": pd.DataFrame(validation_rows),
        "test": pd.DataFrame(test_rows),
    }


def block_means(series: pd.Series, start: pd.Timestamp, days: int) -> pd.Series:
    if series.empty:
        return pd.Series(dtype=float)
    elapsed = (series.index - start).total_seconds() / 86400.0
    block = np.floor(elapsed / days).astype(int)
    return series.groupby(block).mean()


def family_committee_score(local: pd.DataFrame, scores: list[pd.Series]) -> pd.Series:
    matrix = pd.concat(scores, axis=1)
    median = matrix.median(axis=1, skipna=True)
    ranked = p27.centered_rank(local, median)
    smoothed = ranked.groupby(local["symbol"], sort=False).transform(
        lambda values: values.ewm(alpha=FAMILY_SMOOTHING_ALPHA, adjust=False, min_periods=1).mean()
    )
    return p27.centered_rank(local, smoothed)


def family_period_metrics(
    frame: pd.DataFrame,
    fold_id: int,
    family: str,
    prefix: str,
    block_start: pd.Timestamp,
) -> dict[str, Any]:
    ic_series = p27.timestamp_ic_series(frame, "_family_score", "label_slow_low_turnover_rank_centered")
    blocks = block_means(ic_series, block_start, VALIDATION_BLOCK_DAYS)
    mean_ic = float(blocks.mean()) if not blocks.empty else np.nan
    std = float(blocks.std(ddof=1)) if len(blocks) > 1 else np.nan
    se = std / np.sqrt(len(blocks)) if np.isfinite(std) and len(blocks) > 1 else np.nan
    lcb = mean_ic - 0.674 * se if np.isfinite(se) else np.nan
    return {
        "fold": fold_id,
        "family": family,
        f"{prefix}_blocks": len(blocks),
        f"{prefix}_slow_rank_ic": p27.mean_timestamp_ic(frame, "_family_score", "label_slow_low_turnover_rank_centered"),
        f"{prefix}_block_mean_ic": mean_ic,
        f"{prefix}_block_q25_ic": float(blocks.quantile(0.25)) if not blocks.empty else np.nan,
        f"{prefix}_positive_block_rate": float((blocks > 0.0).mean()) if not blocks.empty else 0.0,
        f"{prefix}_path_rank_ic": p27.mean_timestamp_ic(frame, "_family_score", "label_path_slow_low_turnover_rank_centered"),
        f"{prefix}_lcb75_ic": lcb,
        f"{prefix}_signal_turnover_72h": p27.signal_turnover(frame, "_family_score", HORIZON),
        f"{prefix}_signal_autocorr_72h": p27.signal_autocorr(frame, "_family_score", HORIZON),
    }


def empty_family_metric(fold_id: int, family: str, prefix: str) -> dict[str, Any]:
    return {
        "fold": fold_id,
        "family": family,
        f"{prefix}_blocks": 0,
        f"{prefix}_slow_rank_ic": np.nan,
        f"{prefix}_block_mean_ic": np.nan,
        f"{prefix}_block_q25_ic": np.nan,
        f"{prefix}_positive_block_rate": 0.0,
        f"{prefix}_path_rank_ic": np.nan,
        f"{prefix}_lcb75_ic": np.nan,
        f"{prefix}_signal_turnover_72h": np.nan,
        f"{prefix}_signal_autocorr_72h": np.nan,
    }


def summarize_family_oos(oos: pd.DataFrame, selection: pd.DataFrame) -> pd.DataFrame:
    merged = oos.merge(selection[["fold", "family", "family_active", "selected_member_count"]], on=["fold", "family"])
    rows = []
    for family, frame in merged.groupby("family"):
        active = frame.loc[frame["family_active"].astype(bool)]
        values = active["test_slow_rank_ic"].dropna()
        rows.append({
            "family": family,
            "active_folds": int(len(values)),
            "active_fold_rate": float(len(values) / frame["fold"].nunique()),
            "mean_selected_members": float(active["selected_member_count"].mean()) if not active.empty else 0.0,
            "mean_test_slow_rank_ic": float(values.mean()) if not values.empty else np.nan,
            "q25_test_slow_rank_ic": float(values.quantile(0.25)) if not values.empty else np.nan,
            "positive_test_ic_fold_rate": float((values > 0.0).mean()) if not values.empty else 0.0,
            "mean_test_path_rank_ic": float(active["test_path_rank_ic"].mean()) if not active.empty else np.nan,
            "mean_test_turnover_72h": float(active["test_signal_turnover_72h"].mean()) if not active.empty else np.nan,
            "mean_test_autocorr_72h": float(active["test_signal_autocorr_72h"].mean()) if not active.empty else np.nan,
        })
    return pd.DataFrame(rows).sort_values("mean_test_slow_rank_ic", ascending=False)


def init_trade_worker(
    panel_path: str,
    config_path: str,
    families: dict[str, FamilyDefinition],
    selected_p9: pd.DataFrame,
    selected_p15c: pd.DataFrame,
    residual_path: str,
    family_selection: pd.DataFrame,
    family_validation: pd.DataFrame,
    family_oos: pd.DataFrame,
) -> None:
    global _PANEL, _CFG, _FAMILIES, _P9_SELECTED, _P15C_SELECTED, _RISK_RESIDUAL
    global _FAMILY_SELECTION, _FAMILY_VALIDATION, _FAMILY_OOS
    _PANEL = pd.read_pickle(panel_path)
    _CFG = load_config(Path(config_path))
    _FAMILIES = families
    _P9_SELECTED = selected_p9.set_index("fold")
    _P15C_SELECTED = selected_p15c.set_index("fold")
    _RISK_RESIDUAL = pd.read_pickle(residual_path)
    _FAMILY_SELECTION = family_selection
    _FAMILY_VALIDATION = family_validation
    _FAMILY_OOS = family_oos


def trade_fold_worker(fold: dict[str, Any]) -> dict[str, Any]:
    if any(item is None for item in [_PANEL, _CFG, _P9_SELECTED, _P15C_SELECTED, _RISK_RESIDUAL]):
        raise RuntimeError("P28 trade worker is not initialized")
    fold_id = int(fold["fold"])
    history_start = pd.Timestamp(fold["train_start"]) - pd.Timedelta(hours=504)
    local = _PANEL.loc[
        (_PANEL["timestamp"] >= history_start) & (_PANEL["timestamp"] < pd.Timestamp(fold["test_end"]))
    ].copy()
    local[p3d.P4_SCORE_COL] = local[fold_score_column("p4", fold_id)].astype(float)
    local["_p15c_anchor_score"] = local[fold_score_column("p15c", fold_id)].astype(float)
    local[p3d.P4_SCORE_COL] = local[p3d.P4_SCORE_COL].astype(float)

    selection = _FAMILY_SELECTION.loc[_FAMILY_SELECTION["fold"].astype(int).eq(fold_id)]
    family_scores: dict[str, pd.Series] = {}
    for row in selection.itertuples(index=False):
        names = [name for name in str(row.selected_members).split("|") if name and name != "nan"]
        if not names:
            continue
        formula_map = {member.name: member for member in _FAMILIES[str(row.family)].members}
        member_scores = [p27.evaluate_formula(formula_map[name], local) for name in names]
        family_scores[str(row.family)] = family_committee_score(local, member_scores)

    variant_weights = build_variant_weights(fold_id, build_variants())
    score_by_variant: dict[str, pd.Series] = {}
    weight_rows = []
    for variant in build_variants():
        weights = variant_weights[variant.name]
        score_by_variant[variant.name] = blended_score(local, family_scores, weights)
        for family in _FAMILIES:
            weight_rows.append({
                "fold": fold_id,
                "variant": variant.name,
                "family": family,
                "family_weight": float(weights.get(family, 0.0)),
                "baseline_weight": float(1.0 - sum(weights.values())),
            })

    base_row = _P9_SELECTED.loc[fold_id]
    committee = {item.name: item for item in p3d.build_committee_specs()}[str(base_row["p4_spec"])]
    risk_cfg = p15c.make_fold_risk(p3d.make_risk_template(_CFG), committee, f"p28_fold_{fold_id}")
    pf_cfg = p3d.make_portfolio_config(_CFG, HORIZON)
    cluster_row = _P15C_SELECTED.loc[fold_id]
    cluster_map = {item.name: item for item in p15c.build_extreme_specs()}
    cluster_spec = None if bool(cluster_row["fallback_to_p9_sizing"]) else cluster_map[str(cluster_row["selected_extreme_spec"])]
    validation_start = validation_split(fold)
    validation_trade_end = p4.decision_end_for_closed_positions(pd.Timestamp(fold["train_end"]), HORIZON)
    test_trade_end = p4.decision_end_for_closed_positions(pd.Timestamp(fold["test_end"]), HORIZON)

    validation_rows = []
    validation_runs: dict[str, dict[str, Any]] = {}
    all_scores = {"P15c": local["_p15c_anchor_score"], **score_by_variant}
    for name, score in all_scores.items():
        local[p3d.SCORE_COL] = score
        local[p3d.CANDIDATE_RANK_COL] = score
        local[p3d.EFFECTIVE_CANDIDATE_WEIGHT_COL] = 1.0 - (
            1.0 if name == "P15c" else next(v.minimum_baseline_weight for v in build_variants() if v.name == name)
        )
        run = run_window(
            local, validation_start, validation_trade_end, fold, name, pf_cfg, risk_cfg, committee,
            cluster_spec, cluster_row, _RISK_RESIDUAL, score,
        )
        validation_runs[name] = run
        metrics = run["folds"].iloc[0].to_dict()
        validation_rows.append({"fold": fold_id, "variant": name, **{f"validation_{k}": v for k, v in metrics.items() if scalar(v)}})
    validation_frame = pd.DataFrame(validation_rows)
    validation_frame = annotate_variant_acceptance(validation_frame)
    selected_row = select_nested_variant(validation_frame, fold)
    selected_name = str(selected_row["selected_variant"])

    test_runs: dict[str, dict[str, Any]] = {}
    for name, score in all_scores.items():
        local[p3d.SCORE_COL] = score
        local[p3d.CANDIDATE_RANK_COL] = score
        run = run_window(
            local, pd.Timestamp(fold["test_start"]), test_trade_end, fold, name, pf_cfg, risk_cfg, committee,
            cluster_spec, cluster_row, _RISK_RESIDUAL, score,
        )
        test_runs[name] = run
    nested = clone_run_with_name(test_runs[selected_name], "P28e_nested", selected_name)
    return {
        "validation": validation_frame,
        "selected": pd.DataFrame([selected_row]),
        "weights": pd.DataFrame(weight_rows),
        "tests": {variant.name: test_runs[variant.name] for variant in build_variants()},
        "nested": nested,
        "baseline_replay": test_runs["P15c"],
    }


def build_variant_weights(fold_id: int, variants: list[BlendVariant]) -> dict[str, dict[str, float]]:
    validation = _FAMILY_VALIDATION.loc[_FAMILY_VALIDATION["fold"].astype(int).eq(fold_id)].copy()
    selection = _FAMILY_SELECTION.loc[_FAMILY_SELECTION["fold"].astype(int).eq(fold_id)]
    validation = validation.merge(selection[["family", "family_active"]], on="family", how="left")
    validation["eligible"] = (
        validation["family_active"].fillna(False).astype(bool)
        & validation["validation_block_mean_ic"].fillna(-1.0).ge(VALIDATION_MEAN_IC_FLOOR)
        & validation["validation_block_q25_ic"].fillna(-1.0).ge(VALIDATION_Q25_FLOOR)
        & validation["validation_positive_block_rate"].fillna(0.0).ge(0.50)
        & validation["validation_path_rank_ic"].fillna(-1.0).ge(-0.02)
    )
    active = list(validation.loc[validation["eligible"], "family"])
    outputs: dict[str, dict[str, float]] = {}
    for variant in variants:
        if not active:
            outputs[variant.name] = {}
            continue
        if variant.method == "equal":
            raw = {name: 1.0 for name in active}
        elif variant.method == "shrink":
            lcb = validation.set_index("family")["validation_lcb75_ic"].fillna(-1.0)
            positive = {name: max(0.0, float(lcb.loc[name])) for name in active}
            if sum(positive.values()) <= 0.0:
                outputs[variant.name] = {}
                continue
            normalized = normalize_dict(positive)
            equal = 1.0 / len(active)
            raw = {name: 0.50 * normalized[name] + 0.50 * equal for name in active}
        else:
            raw = dma_raw_weights(fold_id, active, validation)
        outputs[variant.name] = capped_allocation(raw, variant.target_native_weight, FAMILY_WEIGHT_CAP)
    return outputs


def dma_raw_weights(fold_id: int, active: list[str], validation: pd.DataFrame) -> dict[str, float]:
    history = _FAMILY_OOS.loc[_FAMILY_OOS["fold"].astype(int).lt(fold_id)].copy()
    current_val = validation.set_index("family")
    raw = {}
    for family in active:
        past = history.loc[history["family"].eq(family), ["fold", "test_slow_rank_ic"]].dropna().sort_values("fold")
        if past.empty:
            ewma = 0.0
        else:
            age = fold_id - past["fold"].to_numpy(dtype=float)
            weights = np.exp(-np.log(2.0) * age / 6.0)
            ewma = float(np.average(past["test_slow_rank_ic"].to_numpy(dtype=float), weights=weights))
        val_mean = float(current_val.loc[family, "validation_block_mean_ic"])
        val_quality = max(0.05, float(current_val.loc[family, "validation_positive_block_rate"]))
        raw[family] = float(np.exp(np.clip(8.0 * ewma, -3.0, 3.0)) * np.exp(np.clip(4.0 * val_mean, -2.0, 2.0)) * val_quality)
    return raw


def normalize_dict(values: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, value) for value in values.values())
    return {key: max(0.0, value) / total for key, value in values.items()} if total > 0 else {}


def capped_allocation(raw: dict[str, float], target: float, cap: float) -> dict[str, float]:
    if not raw:
        return {}
    proportions = normalize_dict(raw)
    allocated = {key: 0.0 for key in raw}
    remaining = min(target, cap * len(raw))
    free = set(raw)
    while free and remaining > 1e-12:
        denom = sum(proportions[key] for key in free)
        if denom <= 0:
            share = remaining / len(free)
            proposed = {key: share for key in free}
        else:
            proposed = {key: remaining * proportions[key] / denom for key in free}
        capped = [key for key, value in proposed.items() if allocated[key] + value > cap + 1e-12]
        if not capped:
            for key, value in proposed.items():
                allocated[key] += value
            remaining = 0.0
            break
        for key in capped:
            add = cap - allocated[key]
            allocated[key] = cap
            remaining -= add
            free.remove(key)
    return {key: value for key, value in allocated.items() if value > 1e-12}


def blended_score(local: pd.DataFrame, family_scores: dict[str, pd.Series], weights: dict[str, float]) -> pd.Series:
    baseline_weight = 1.0 - sum(weights.values())
    combined = baseline_weight * local["_p15c_anchor_score"].astype(float)
    for family, weight in weights.items():
        if family in family_scores:
            combined = combined + weight * family_scores[family]
    return p27.centered_rank(local, combined)


def run_window(
    panel: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    fold: dict[str, Any],
    name: str,
    pf_cfg: Any,
    risk_cfg: Any,
    committee: Any,
    cluster_spec: Any,
    cluster_row: pd.Series,
    risk_residual: pd.DataFrame,
    score: pd.Series,
) -> dict[str, Any]:
    panel[p3d.SCORE_COL] = score
    formula_text = f"P28 {name} anchored to P15c"
    equity, positions, metrics = p15c.run_extreme_backtest(
        panel=panel,
        residual_wide=risk_residual,
        start=start,
        end=end,
        cfg=pf_cfg,
        risk_cfg=risk_cfg,
        committee=committee,
        spec=cluster_spec,
        diagnostic_lookback=int(cluster_row["lookback_hours"]),
        diagnostic_k=int(cluster_row["cluster_k"]),
        score_col=p3d.SCORE_COL,
        selected_formulas=formula_text,
    )
    p27.enrich_metrics(equity, positions, metrics, 0)
    p15c.enrich_extreme_metrics(equity, positions, metrics, HORIZON)
    factor_end = p4.decision_end_for_closed_positions(
        pd.Timestamp(fold["train_end"] if start < pd.Timestamp(fold["test_start"]) else fold["test_end"]),
        MAX_LABEL_HORIZON,
    )
    factor_scope = panel.loc[(panel["timestamp"] >= start) & (panel["timestamp"] < factor_end)].copy()
    factor_scope["_window_score"] = score.loc[factor_scope.index]
    metrics["slow_rank_ic"] = p27.mean_timestamp_ic(
        factor_scope, "_window_score", "label_slow_low_turnover_rank_centered"
    )
    metrics["path_rank_ic"] = p27.mean_timestamp_ic(
        factor_scope, "_window_score", "label_path_slow_low_turnover_rank_centered"
    )
    metrics["signal_turnover_72h"] = p27.signal_turnover(factor_scope, "_window_score", HORIZON)
    metrics["signal_autocorr_72h"] = p27.signal_autocorr(factor_scope, "_window_score", HORIZON)
    fold_row = {
        "fold": int(fold["fold"]),
        "train_start": fold["train_start"],
        "train_end": fold["train_end"],
        "test_start": fold["test_start"],
        "test_end": fold["test_end"],
        "test_decision_end": end,
        "variant": name,
        "selected_extreme_spec": p15c.NO_EXTREME_CAP if cluster_spec is None else cluster_spec.name,
        **metrics,
    }
    for frame in [equity, positions]:
        if frame.empty:
            continue
        frame["fold"] = int(fold["fold"])
        frame["variant"] = name
        frame["formula"] = formula_text
    return {"folds": pd.DataFrame([fold_row]), "equity": equity, "positions": positions}


def scalar(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating, bool)) and not isinstance(value, str)


def annotate_variant_acceptance(frame: pd.DataFrame) -> pd.DataFrame:
    base = frame.loc[frame["variant"].eq("P15c")].iloc[0]
    out = frame.copy()
    out["delta_net8"] = out["validation_total_return_8bps"] - float(base["validation_total_return_8bps"])
    out["delta_net20"] = out["validation_total_return_20bps"] - float(base["validation_total_return_20bps"])
    out["delta_gross"] = out["validation_gross_total_return_before_cost"] - float(base["validation_gross_total_return_before_cost"])
    out["delta_sharpe8"] = out["validation_sharpe_8bps"] - float(base["validation_sharpe_8bps"])
    out["delta_maxdd8"] = out["validation_max_drawdown_8bps"] - float(base["validation_max_drawdown_8bps"])
    turnover_limit = max(float(base["validation_mean_turnover"]) * 1.05, float(base["validation_mean_turnover"]) + 0.01)
    out["hard_accept"] = (
        ~out["variant"].eq("P15c")
        & out["delta_net8"].gt(0.0)
        & out["delta_net20"].gt(0.0)
        & out["delta_gross"].ge(-0.001)
        & out["delta_sharpe8"].ge(-0.05)
        & out["validation_max_drawdown_8bps"].ge(1.10 * float(base["validation_max_drawdown_8bps"]))
        & out["validation_mean_turnover"].le(turnover_limit)
    )
    out["selection_score"] = (
        120.0 * out["delta_net8"]
        + 80.0 * out["delta_net20"]
        + 40.0 * out["delta_gross"]
        + 0.8 * out["delta_sharpe8"]
        + 25.0 * out["delta_maxdd8"]
        - 3.0 * (out["validation_mean_turnover"] - float(base["validation_mean_turnover"])).clip(lower=0.0)
    )
    return out


def select_nested_variant(validation: pd.DataFrame, fold: dict[str, Any]) -> dict[str, Any]:
    accepted = validation.loc[validation["hard_accept"].astype(bool)]
    if accepted.empty:
        winner = validation.loc[validation["variant"].eq("P15c")].iloc[0]
        fallback = True
    else:
        winner = accepted.sort_values(["selection_score", "delta_net20", "delta_gross"], ascending=False).iloc[0]
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
        "validation_net8": float(winner["validation_total_return_8bps"]),
        "validation_net20": float(winner["validation_total_return_20bps"]),
        "validation_gross": float(winner["validation_gross_total_return_before_cost"]),
        "validation_sharpe8": float(winner["validation_sharpe_8bps"]),
        "validation_maxdd8": float(winner["validation_max_drawdown_8bps"]),
        "validation_mean_turnover": float(winner["validation_mean_turnover"]),
    }


def clone_run_with_name(run: dict[str, Any], name: str, selected_variant: str) -> dict[str, Any]:
    cloned = {key: value.copy() for key, value in run.items()}
    cloned["folds"]["variant"] = name
    cloned["folds"]["selected_validation_variant"] = selected_variant
    for key in ["equity", "positions"]:
        if not cloned[key].empty:
            cloned[key]["variant"] = name
            cloned[key]["selected_validation_variant"] = selected_variant
    return cloned


def build_fold_comparison(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    output: pd.DataFrame | None = None
    for name, result in results.items():
        columns = ["fold", "total_return", "gross_total_return_before_cost", "sharpe", "max_drawdown", "mean_turnover", "slow_rank_ic"]
        frame = result["folds"][[column for column in columns if column in result["folds"]]].copy()
        frame = frame.rename(columns={column: f"{name}_{column}" for column in frame if column != "fold"})
        output = frame if output is None else output.merge(frame, on="fold", how="inner")
    if output is not None:
        output["P28e_minus_P15c_return"] = output["P28e_nested_total_return"] - output["P15c_total_return"]
        output["P28e_minus_P15c_gross"] = (
            output["P28e_nested_gross_total_return_before_cost"] - output["P15c_gross_total_return_before_cost"]
        )
    return output if output is not None else pd.DataFrame()


def compare_baseline_replay(
    locked: dict[str, Any], replay: dict[str, Any], tolerance: float = 2e-6
) -> pd.DataFrame:
    left = locked["equity"][["timestamp", "fold", "gross_period_return", "turnover"]].sort_values(
        ["fold", "timestamp"]
    ).reset_index(drop=True)
    right = replay["equity"][["timestamp", "fold", "gross_period_return", "turnover"]].sort_values(
        ["fold", "timestamp"]
    ).reset_index(drop=True)
    key_pass = len(left) == len(right) and left[["timestamp", "fold"]].equals(right[["timestamp", "fold"]])
    gross_error = (
        float(np.nanmax(np.abs(left["gross_period_return"].to_numpy() - right["gross_period_return"].to_numpy())))
        if key_pass and len(left)
        else np.inf
    )
    turnover_error = (
        float(np.nanmax(np.abs(left["turnover"].to_numpy() - right["turnover"].to_numpy())))
        if key_pass and len(left)
        else np.inf
    )
    return pd.DataFrame([
        {"check": "exact_period_keys", "value": float(key_pass), "pass": bool(key_pass)},
        {"check": "max_abs_gross_return_error", "value": gross_error, "pass": bool(gross_error <= tolerance)},
        {"check": "max_abs_turnover_error", "value": turnover_error, "pass": bool(turnover_error <= tolerance)},
    ])


def build_protocol_audit(
    folds: pd.DataFrame,
    family_selection: pd.DataFrame,
    family_validation: pd.DataFrame,
    selected_variant: pd.DataFrame,
    result: dict[str, Any],
) -> pd.DataFrame:
    selection = family_selection.groupby("fold").agg(
        train_label_end=("train_label_end", "max"),
        validation_start=("validation_start", "min"),
        validation_label_end=("validation_label_end", "max"),
    ).reset_index()
    for column in ["train_label_end", "validation_start", "validation_label_end"]:
        selection[column] = pd.to_datetime(selection[column], utc=True)
    merged = folds.merge(selection, on="fold", how="left").merge(
        selected_variant[["fold", "selected_variant"]], on="fold", how="left"
    )
    test_max = result["equity"].groupby("fold")["timestamp"].max().rename("max_test_decision_time")
    merged = merged.merge(test_max, on="fold", how="left")
    merged["train_labels_closed_before_validation"] = (
        merged["train_label_end"] + pd.Timedelta(hours=MAX_LABEL_HORIZON + 1)
    ) <= merged["validation_start"]
    merged["validation_labels_closed_before_test"] = (
        merged["validation_label_end"] + pd.Timedelta(hours=MAX_LABEL_HORIZON + 1)
    ) <= merged["train_end"]
    merged["variant_selected_before_test"] = merged["train_end"] <= merged["test_start"]
    merged["test_trades_close_inside_fold"] = (
        merged["max_test_decision_time"] + pd.Timedelta(hours=HORIZON + 1)
    ) <= merged["test_end"]
    merged["all_pass"] = merged[[
        "train_labels_closed_before_validation",
        "validation_labels_closed_before_test",
        "variant_selected_before_test",
        "test_trades_close_inside_fold",
    ]].all(axis=1)
    return merged


def build_acceptance_table(
    summary: pd.DataFrame,
    cost: pd.DataFrame,
    folds_out: pd.DataFrame,
    regime: pd.DataFrame,
    family_oos: pd.DataFrame,
    selected_variant: pd.DataFrame,
    pbo_path: Path,
) -> pd.DataFrame:
    base = summary.loc[summary["run"].eq("P15c")].iloc[0]
    cand = summary.loc[summary["run"].eq("P28e_nested")].iloc[0]
    base8 = cost.loc[cost["experiment"].eq("P15c") & cost["one_way_cost_bps"].eq(8.0)].iloc[0]
    cand8 = cost.loc[cost["experiment"].eq("P28e_nested") & cost["one_way_cost_bps"].eq(8.0)].iloc[0]
    pbo = (
        pd.read_csv(pbo_path).iloc[0]
        if pbo_path.exists()
        else pd.Series({"pbo_probability_test_rank_below_median": np.nan})
    )
    checks = [
        ("gross_not_below_p15c", cand["gross_total_return_before_cost"] - base["gross_total_return_before_cost"], cand["gross_total_return_before_cost"] >= base["gross_total_return_before_cost"] - 1e-6),
        ("net8_not_below_p15c", cand8["total_return"] - base8["total_return"], cand8["total_return"] >= base8["total_return"] - 1e-6),
        ("sharpe_not_below_p15c", cand["sharpe"] - base["sharpe"], cand["sharpe"] >= base["sharpe"] - 1e-5),
        ("maxdd_within_110pct_p15c", cand["max_drawdown"] - base["max_drawdown"], cand["max_drawdown"] >= 1.10 * base["max_drawdown"]),
        ("worst5_not_below_p15c", cand["worst_5_fold_return_sum"] - base["worst_5_fold_return_sum"], cand["worst_5_fold_return_sum"] >= base["worst_5_fold_return_sum"] - 1e-6),
        ("turnover_not_above_p15c", cand["mean_turnover"] / base["mean_turnover"], cand["mean_turnover"] <= base["mean_turnover"] + 2e-6),
        ("positive_fold_rate_at_least_65pct", cand["positive_fold_rate"], cand["positive_fold_rate"] >= 0.65),
        ("legacy_formula_selector_pbo_below_20pct", pbo.get("pbo_probability_test_rank_below_median", np.nan), pbo.get("pbo_probability_test_rank_below_median", 1.0) < 0.20),
        ("fallback_rate_not_excessive", float(selected_variant["fallback_to_p15c"].mean()), float(selected_variant["fallback_to_p15c"].mean()) < 0.90),
    ]
    return pd.DataFrame([{"check": name, "value": value, "pass": bool(passed)} for name, value, passed in checks])


def plot_family_ic(output_dir: Path, summary: pd.DataFrame) -> None:
    view = summary.sort_values("mean_test_slow_rank_ic")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = np.where(view["q25_test_slow_rank_ic"].fillna(-1.0) >= -0.01, "#2a9d8f", "#8d99ae")
    ax.barh(view["family"], view["mean_test_slow_rank_ic"], color=colors)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_title("P28b fixed family committees: nested OOS slow-label Rank IC")
    ax.set_xlabel("Mean OOS Rank IC on active folds")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "p28b_family_oos_ic.png", dpi=180)
    plt.close(fig)


def plot_pnl(output_dir: Path, results: dict[str, dict[str, Any]]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0]})
    colors = {
        "P15c": "#777777",
        "P28c_equal_30": "#5b8e7d",
        "P28c_equal_20": "#6c8ebf",
        "P28d_shrink_25": "#c49a5a",
        "P28d_dma_30": "#9a6fb0",
        "P28e_nested": "#d1495b",
    }
    for name, result in results.items():
        equity = result["equity"].sort_values("timestamp")
        returns = equity["gross_period_return"] - equity["turnover"] * 8.0 / 10_000.0
        curve = (1.0 + returns).cumprod() - 1.0
        axes[0].plot(equity["timestamp"], curve, label=f"{name} net 8bps", color=colors[name], linewidth=2.2 if name == "P28e_nested" else 1.2)
        if name == "P28e_nested":
            wealth = 1.0 + curve
            drawdown = wealth / wealth.cummax() - 1.0
            axes[1].fill_between(equity["timestamp"], drawdown, 0.0, color=colors[name], alpha=0.20)
            axes[1].plot(equity["timestamp"], drawdown, color=colors[name], linewidth=1.2)
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_title("P28 matched nested OOS portfolio results")
    axes[0].set_ylabel("Cumulative return")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].legend(ncol=2)
    axes[0].grid(alpha=0.25)
    axes[1].set_title("P28e nested selector net 8bps drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "p28e_matched_nested_oos_pnl.png", dpi=180)
    plt.close(fig)


def plot_variant_selection(output_dir: Path, selected: pd.DataFrame) -> None:
    counts = selected["selected_variant"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(counts.index, counts.values, color="#457b9d")
    ax.set_title("P28e validation-selected variant frequency")
    ax.set_xlabel("Outer folds")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "p28e_variant_selection.png", dpi=180)
    plt.close(fig)


def write_report(
    output_dir: Path,
    family_summary: pd.DataFrame,
    member_stability: pd.DataFrame,
    validation: pd.DataFrame,
    selected_variant: pd.DataFrame,
    family_weights: pd.DataFrame,
    summary: pd.DataFrame,
    cost: pd.DataFrame,
    regime: pd.DataFrame,
    folds_out: pd.DataFrame,
    audit: pd.DataFrame,
    acceptance: pd.DataFrame,
) -> None:
    effective = pd.read_csv(output_dir / "p28a_selection_overfit_audit" / "grammar_effective_dimension.csv").iloc[0]
    pbo = pd.read_csv(output_dir / "p28a_selection_overfit_audit" / "pbo_summary.csv").iloc[0]
    val_rank = pd.read_csv(output_dir / "p28a_selection_overfit_audit" / "validation_test_rank_correlation.csv")
    base = summary.loc[summary["run"].eq("P15c")].iloc[0]
    nested = summary.loc[summary["run"].eq("P28e_nested")].iloc[0]
    base8 = cost.loc[cost["experiment"].eq("P15c") & cost["one_way_cost_bps"].eq(8.0)].iloc[0]
    nested8 = cost.loc[cost["experiment"].eq("P28e_nested") & cost["one_way_cost_bps"].eq(8.0)].iloc[0]
    winner = summary.sort_values("total_return", ascending=False).iloc[0]
    pass_count = int(acceptance["pass"].sum())
    verdict = "通过全部升级门槛" if pass_count == len(acceptance) else "不覆盖 P15c；保留为研究结果"
    fold_delta = folds_out["P28e_minus_P15c_return"]
    meaningful_fold_win = float((fold_delta > 1e-6).mean())
    numerical_tie_rate = float((fold_delta.abs() <= 1e-6).mean())
    lines = [
        "# P28 稳定经济 family ensemble 报告",
        "",
        "## 结论",
        "",
        f"- 最终结论：**{verdict}**。",
        f"- P28e nested gross/net6/net8/Sharpe/MaxDD：{nested['gross_total_return_before_cost']:.2%} / {nested['total_return']:.2%} / {nested8['total_return']:.2%} / {nested['sharpe']:.3f} / {nested['max_drawdown']:.2%}。",
        f"- P15c 对照 gross/net6/net8/Sharpe/MaxDD：{base['gross_total_return_before_cost']:.2%} / {base['total_return']:.2%} / {base8['total_return']:.2%} / {base['sharpe']:.3f} / {base['max_drawdown']:.2%}。",
        f"- 固定变体中净收益最高的是 {winner['run']}，但只有 P28e 是在 validation 内选择变体后拼接 test 的最终 selector 结果。",
        f"- P28e fallback 到 P15c 的 fold 比例为 {selected_variant['fallback_to_p15c'].mean():.1%}；因此 P28e 实质上就是 P15c 的数值等价重放。",
        f"- {int(audit['all_pass'].sum())}/{len(audit)} folds 通过 chronology / label closure 审计。",
        "",
        "## P28a：选择过拟合审计",
        "",
        f"- 240 个名义公式的有效独立维度只有 {effective['effective_independent_formula_count']:.1f}（{effective['effective_fraction']:.1%}）。",
        f"- PBO 为 {pbo['pbo_probability_test_rank_below_median']:.1%}；train winner 在另一半样本中的中位排名分位为 {pbo['median_test_rank_percentile_of_train_winner']:.1%}。",
        f"- P27 validation fitness 与同公式 test IC 的平均 rank correlation 为 {val_rank['validation_test_rank_correlation'].mean():.3f}。",
        "- 这部分是 post-hoc 诊断，不参与任何 P28 fold 的选择。",
        "",
        "## P28b：固定 family committee",
        "",
        "每个 family 的成员和方向在实验前固定；每折只用 train 中 28 日 block 稳定性决定成员是否保留。",
        "",
        "```text",
        family_summary.to_string(index=False),
        "```",
        "",
        "## P28c/P28d/P28e：交易结果",
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
        "### P28e 选择频次",
        "```text",
        selected_variant["selected_variant"].value_counts().to_string(),
        "```",
        "",
        "### 升级门槛",
        "```text",
        acceptance.to_string(index=False),
        "```",
        "",
        "## Regime 与 fold 检查",
        "",
        "```text",
        regime.loc[regime["run"].isin(["P15c", "P28e_nested"])].to_string(index=False),
        "```",
        "",
        f"- P28e 相对 P15c 的有意义 fold 胜率（差值 > 1e-6）：{meaningful_fold_win:.1%}；数值等价 fold 比例：{numerical_tie_rate:.1%}。",
        f"- P28e 最差 5 folds 合计：{nested['worst_5_fold_return_sum']:.2%}；P15c：{base['worst_5_fold_return_sum']:.2%}。",
        "",
        "## 严格性与限制",
        "",
        "- family member selection 只读 train；validation factor IC 只用在 train_end 前完整闭合的 168h label。",
        "- variant/weight/fallback 只读 train、closed-label validation 和已经结束的历史 test folds；当前 test 完全不参与选择。",
        "- 交易 validation 只使用在 train_end 前闭合的 72h return；test 交易全部在 test_end 前闭合。",
        "- P28 family 假设是在观察 P27 后提出，存在研究序列层面的 meta-overfitting；即使 nested OOS 好，也仍需要未来 untouched sample。",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
