#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
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

from crypto_stat_arb.config import ensure_dir
from run_p2_family_a_momentum import build_family_a_features
from run_p2_family_b_mean_reversion import build_family_b_features
from run_p2_manual_slow_alpha_grid import build_manual_slow_hybrid_features
from run_p3_committee_guard_scan import COMPONENTS
from run_p3d_p4_gated_rerank_alphagpt import add_ex_ante_high_risk_regime, centered_rank_by_timestamp
from run_p5_alphagpt_slow_search import add_transformed_primitives
from run_p6_feature_expansion import build_p6_features
from run_p6b_derivatives_feature_expansion import build_derivatives_candidates
from run_residual_alphagpt_search import build_candidate_scores, prepare_alpha_panel


DEFAULT_ROOT = PROJECT_ROOT / "experiments" / "baseline_locked_v1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--asset-panel",
        default=str(DEFAULT_ROOT / "inputs" / "long_history" / "baseline_panel" / "asset_residual_panel.csv"),
    )
    parser.add_argument(
        "--derivatives-panel",
        default=str(
            DEFAULT_ROOT
            / "inputs"
            / "long_history"
            / "binance_derivatives"
            / "derivatives_hourly_panel.csv.gz"
        ),
    )
    parser.add_argument(
        "--required-columns",
        default=str(DEFAULT_ROOT / "inputs" / "p16_replay_required_columns.csv"),
    )
    parser.add_argument(
        "--retained-features",
        default=str(DEFAULT_ROOT / "outputs" / "p6b_derivatives_feature_expansion" / "retained_features.csv"),
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_ROOT / "inputs" / "long_history" / "p16_long_replay_panel.csv.gz"),
    )
    parser.add_argument("--horizon", type=int, default=72)
    args = parser.parse_args()

    output_path = Path(args.output).resolve()
    ensure_dir(output_path.parent)

    print("[1/8] Loading long asset residual panel", flush=True)
    panel = pd.read_csv(args.asset_panel, parse_dates=["timestamp"])
    panel = panel.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    print("[2/8] Preparing base AlphaGPT/replay primitives", flush=True)
    panel = prepare_alpha_panel(panel, [int(args.horizon)], execution_delay_bars=1)
    build_candidate_scores(panel)

    print("[3/8] Building P2 slow residual feature families A/B/C", flush=True)
    build_family_a_features(panel)
    build_family_b_features(panel)
    build_manual_slow_hybrid_features(panel)
    add_transformed_primitives(panel)

    print("[4/8] Building P4 committee ranks and high-risk regime", flush=True)
    for key, col in COMPONENTS.items():
        panel[f"rank_{key.lower()}"] = centered_rank_by_timestamp(panel, col)
    add_ex_ante_high_risk_regime(panel)

    print("[5/8] Building P6 OHLCV feature expansion", flush=True)
    build_p6_features(panel)

    print("[6/8] Merging derivatives and building P6b/P7 retained features", flush=True)
    derivatives = pd.read_csv(args.derivatives_panel, parse_dates=["timestamp"])
    panel = panel.merge(derivatives, on=["timestamp", "symbol"], how="left")
    build_derivatives_candidates(panel)
    retained = pd.read_csv(args.retained_features)
    retained = retained[retained["retained"].astype(bool)].copy()
    retained = retained.sort_values(["diagnostic_score", "primary_mean_ic"], ascending=[False, False])
    grouped = panel.groupby("symbol", group_keys=False)
    for row in retained.itertuples(index=False):
        score_col = str(row.score_col)
        if score_col not in panel:
            continue
        rank_col = f"p7_rank_{score_col}"
        panel[rank_col] = centered_rank_by_timestamp(panel, score_col)
        for span in [168, 336]:
            panel[f"p7_ema{span}_{score_col}"] = grouped[rank_col].transform(
                lambda s, sp=span: s.ewm(span=sp, adjust=False, min_periods=max(48, sp // 3)).mean()
            )

    print("[7/8] Selecting required P16 replay columns", flush=True)
    required = pd.read_csv(args.required_columns)["column"].astype(str).tolist()
    missing = sorted(col for col in required if col not in panel.columns)
    if missing:
        raise KeyError(f"Missing required replay columns: {missing}")
    replay = panel[required].sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    replay.to_csv(output_path, index=False, compression="gzip")

    print("[8/8] Writing diagnostics", flush=True)
    coverage = build_coverage(replay)
    coverage.to_csv(output_path.with_suffix("").with_suffix(".coverage.csv"), index=False)
    meta = {
        "asset_panel": str(Path(args.asset_panel).resolve()),
        "derivatives_panel": str(Path(args.derivatives_panel).resolve()),
        "output": str(output_path),
        "rows": int(len(replay)),
        "symbols": int(replay["symbol"].nunique()),
        "timestamps": int(replay["timestamp"].nunique()),
        "first_timestamp": str(replay["timestamp"].min()),
        "last_timestamp": str(replay["timestamp"].max()),
        "required_column_count": len(required),
        "missing_required_columns": missing,
    }
    output_path.with_suffix("").with_suffix(".metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)


def build_coverage(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for col in panel.columns:
        if col in {"timestamp", "symbol"}:
            continue
        values = panel[col]
        finite = np.isfinite(pd.to_numeric(values, errors="coerce"))
        rows.append(
            {
                "column": col,
                "notna_rate": float(values.notna().mean()),
                "finite_rate": float(finite.mean()),
                "first_valid_timestamp": str(panel.loc[values.notna(), "timestamp"].min())
                if values.notna().any()
                else "",
            }
        )
    return pd.DataFrame(rows).sort_values("column")


if __name__ == "__main__":
    main()
