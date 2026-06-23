#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crypto_stat_arb.config import ensure_dir, load_config
from crypto_stat_arb.data import load_universe_1h
from crypto_stat_arb.factors import ResidualConfig, add_factor_inputs, rolling_residualize
from crypto_stat_arb.features import add_baseline_score, add_cross_sectional_features, add_forward_labels, add_residual_features
from crypto_stat_arb.portfolio import PortfolioConfig
from crypto_stat_arb.walk_forward import WalkForwardConfig, run_walk_forward


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_name = "baseline_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = ensure_dir(args.output_dir or PROJECT_ROOT / "outputs" / run_name)

    print(f"[1/6] Loading 15m klines and resampling to 1h: {cfg['data']['klines_dir']}", flush=True)
    data = load_universe_1h(cfg["data"]["klines_dir"], cfg["data"]["symbols"])
    print(f"      rows={len(data):,}, timestamps={data['timestamp'].nunique():,}, symbols={data['symbol'].nunique()}", flush=True)

    print("[2/6] Building past-only rolling factor residuals", flush=True)
    data = add_factor_inputs(data, cfg["sectors"])
    residual_cfg = ResidualConfig(
        beta_window=int(cfg["factor_model"]["beta_window"]),
        min_beta_obs=int(cfg["factor_model"]["min_beta_obs"]),
    )
    panel = rolling_residualize(data, residual_cfg)

    print("[3/6] Building residual features, cross-sectional ranks, and delayed labels", flush=True)
    panel = add_residual_features(
        panel,
        residual_z_window=int(cfg["factor_model"]["residual_z_window"]),
        windows=cfg["factor_model"]["feature_windows"],
    )
    panel = add_cross_sectional_features(panel)
    panel = add_baseline_score(panel)
    panel = add_forward_labels(
        panel,
        horizons=[int(x) for x in cfg["labels"]["horizons"]],
        execution_delay_bars=int(cfg["labels"]["execution_delay_bars"]),
    )

    print("[4/6] Saving asset panel", flush=True)
    panel_path = output_dir / "asset_residual_panel.csv"
    panel.to_csv(panel_path, index=False)

    print("[5/6] Running strict walk-forward baseline", flush=True)
    pf_cfg = PortfolioConfig(
        rebalance_hours=int(cfg["portfolio"]["rebalance_hours"]),
        long_k=int(cfg["portfolio"]["long_k"]),
        short_k=int(cfg["portfolio"]["short_k"]),
        gross_exposure=float(cfg["portfolio"]["gross_exposure"]),
        one_way_fee_bps=float(cfg["portfolio"]["one_way_fee_bps"]),
        one_way_slippage_bps=float(cfg["portfolio"]["one_way_slippage_bps"]),
        neutralize_exposures=list(cfg["portfolio"]["neutralize_exposures"]),
        primary_horizon=int(cfg["labels"]["primary_horizon"]),
        execution_delay_bars=int(cfg["labels"]["execution_delay_bars"]),
    )
    wf_cfg = WalkForwardConfig(
        train_days=int(cfg["walk_forward"]["train_days"]),
        test_days=int(cfg["walk_forward"]["test_days"]),
        step_days=int(cfg["walk_forward"]["step_days"]),
        mode=str(cfg["walk_forward"].get("mode", "rolling")),
    )
    fold_summary, equity_curve, positions, metrics = run_walk_forward(panel, wf_cfg, pf_cfg)

    fold_summary.to_csv(output_dir / "walk_forward_folds.csv", index=False)
    equity_curve.to_csv(output_dir / "equity_curve.csv", index=False)
    positions.to_csv(output_dir / "positions.csv", index=False)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(to_jsonable(metrics), fh, indent=2, ensure_ascii=False)

    print("[6/6] Writing report", flush=True)
    write_report(output_dir, cfg, panel, fold_summary, metrics)
    print(f"Done: {output_dir}", flush=True)


def write_report(
    output_dir: Path,
    cfg: dict[str, Any],
    panel: pd.DataFrame,
    fold_summary: pd.DataFrame,
    metrics: dict[str, float],
) -> None:
    primary_horizon = int(cfg["labels"]["primary_horizon"])
    valid_score = panel["score_baseline"].notna().sum()
    valid_label = panel[f"label_resid_ret_h{primary_horizon}"].notna().sum()

    lines = [
        "# Crypto Residual Stat Arb Baseline",
        "",
        "## Design",
        "",
        "- Unit: `timestamp x symbol` asset panel.",
        "- Factor model: rolling OLS with BTC, ETH, ex-self market, and ex-self sector returns.",
        "- Lookahead guard: rolling betas are shifted by one bar before residuals are computed.",
        "- Execution guard: labels and portfolio returns start after a one-bar execution delay.",
        "- Baseline score: `score = -cs_zscore_resid_z`.",
        "- Portfolio: long top K, short bottom K, then project weights to neutralize configured exposures.",
        "",
        "## Data",
        "",
        f"- Rows: {len(panel):,}",
        f"- Symbols: {panel['symbol'].nunique():,}",
        f"- Timestamps: {panel['timestamp'].nunique():,}",
        f"- First timestamp: {panel['timestamp'].min()}",
        f"- Last timestamp: {panel['timestamp'].max()}",
        f"- Valid score rows: {valid_score:,}",
        f"- Valid primary-label rows: {valid_label:,}",
        "",
        "## Walk-Forward Metrics",
        "",
    ]

    for key, value in metrics.items():
        lines.append(f"- {key}: {value}")

    lines += [
        "",
        "## Fold Count",
        "",
        f"- Folds generated: {len(fold_summary):,}",
        "",
        "## Output Files",
        "",
        "- `asset_residual_panel.csv`",
        "- `walk_forward_folds.csv`",
        "- `equity_curve.csv`",
        "- `positions.csv`",
        "- `metrics.json`",
    ]

    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(key): to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(value) for value in obj]
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


if __name__ == "__main__":
    main()
