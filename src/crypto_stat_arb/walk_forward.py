from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace

import numpy as np
import pandas as pd

from .portfolio import PortfolioConfig, run_long_short_backtest, summarize_equity_curve


@dataclass(frozen=True)
class WalkForwardConfig:
    train_days: int
    test_days: int
    step_days: int
    mode: str = "rolling"


def make_walk_forward_folds(
    panel: pd.DataFrame,
    cfg: WalkForwardConfig,
    score_col: str,
    primary_horizon: int,
) -> list[dict[str, pd.Timestamp]]:
    label_col = f"label_resid_ret_h{primary_horizon}"
    ret_col = f"fwd_simple_ret_h{primary_horizon}"
    valid = panel[np.isfinite(panel[score_col]) & np.isfinite(panel[label_col]) & np.isfinite(panel[ret_col])]
    counts = valid.groupby("timestamp")["symbol"].nunique()
    valid_ts = counts[counts >= 12].index.sort_values()
    if len(valid_ts) == 0:
        return []

    first = pd.Timestamp(valid_ts.min())
    last = pd.Timestamp(valid_ts.max())
    test_start = first + pd.Timedelta(days=cfg.train_days)
    folds: list[dict[str, pd.Timestamp]] = []

    while test_start + pd.Timedelta(days=cfg.test_days) <= last:
        train_start = first if cfg.mode == "expanding" else test_start - pd.Timedelta(days=cfg.train_days)
        train_end = test_start
        test_end = test_start + pd.Timedelta(days=cfg.test_days)
        folds.append(
            {
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
            }
        )
        test_start = test_start + pd.Timedelta(days=cfg.step_days)

    return folds


def run_walk_forward(
    panel: pd.DataFrame,
    wf_cfg: WalkForwardConfig,
    pf_cfg: PortfolioConfig,
    score_col: str = "score_baseline",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    folds = make_walk_forward_folds(panel, wf_cfg, score_col, pf_cfg.primary_horizon)
    fold_rows = []
    all_equity = []
    all_positions = []

    for fold_id, fold in enumerate(folds, start=1):
        fold_pf_cfg = pf_cfg
        if pf_cfg.cost_gate_buffer > 0:
            coef = calibrate_score_to_return_coef(
                panel=panel,
                start=fold["train_start"],
                end=fold["train_end"],
                score_col=score_col,
                primary_horizon=pf_cfg.primary_horizon,
            )
            fold_pf_cfg = replace(pf_cfg, score_to_return_coef=coef)

        equity, positions, metrics = run_long_short_backtest(
            panel=panel,
            start=fold["test_start"],
            end=fold["test_end"],
            cfg=fold_pf_cfg,
            score_col=score_col,
        )
        if not equity.empty:
            equity = equity.copy()
            equity["fold"] = fold_id
            positions = positions.copy()
            positions["fold"] = fold_id
            all_equity.append(equity)
            all_positions.append(positions)

        fold_rows.append(
            {
                "fold": fold_id,
                **fold,
                **metrics,
            }
        )

    fold_summary = pd.DataFrame(fold_rows)
    equity_curve = pd.concat(all_equity, ignore_index=True) if all_equity else pd.DataFrame()
    positions = pd.concat(all_positions, ignore_index=True) if all_positions else pd.DataFrame()

    if not equity_curve.empty:
        equity_curve = equity_curve.sort_values("timestamp").reset_index(drop=True)
        equity_curve["aggregate_equity"] = (1.0 + equity_curve["period_return"]).cumprod()
        aggregate_for_metrics = equity_curve.copy()
        aggregate_for_metrics["equity"] = aggregate_for_metrics["aggregate_equity"]
        aggregate_metrics = summarize_equity_curve(aggregate_for_metrics, pf_cfg.rebalance_hours)
    else:
        aggregate_metrics = summarize_equity_curve(equity_curve, pf_cfg.rebalance_hours)

    aggregate_metrics["fold_count"] = int(len(folds))
    return fold_summary, equity_curve, positions, aggregate_metrics


def calibrate_score_to_return_coef(
    panel: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    score_col: str,
    primary_horizon: int,
) -> float:
    label_col = f"label_resid_ret_h{primary_horizon}"
    train = panel[(panel["timestamp"] >= start) & (panel["timestamp"] < end)]
    x = train[score_col].astype(float)
    y = train[label_col].astype(float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 100:
        return 0.0
    x = x[valid]
    y = y[valid]
    var = float(x.var(ddof=0))
    if var <= 0 or not np.isfinite(var):
        return 0.0
    cov = float(((x - x.mean()) * (y - y.mean())).mean())
    coef = cov / var
    return float(np.clip(coef, -0.01, 0.01))
