"""Read existing results only; do not replay, tune, or modify any strategy."""

from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "binance_alphatrading_clean/experiments/baseline_locked_v1/outputs/long_history_nested_oos"
OUTPUT = Path(__file__).resolve().parent
STAGES = {
    "P4": "p4_nested/nested_test_run",
    "P3d": "p3d_nested/nested_incremental_run",
    "P9": "p9_nested/nested_incremental_run",
    "P15c": "p15c_nested/nested_extreme_cluster_run",
}


def recorded_metrics(frame, bps):
    returns = frame["gross_period_return"] - frame["turnover"] * bps / 10000
    equity = (1 + returns).cumprod()
    std = returns.std(ddof=0)
    # This intentionally reproduces the published 72h-record convention.
    return {
        "one_way_bps": bps,
        "recorded_return": float(equity.iloc[-1] - 1),
        "recorded_sharpe_72h_convention": float(returns.mean() / std * np.sqrt(365 * 24 / 72)),
        "recorded_max_drawdown": float((equity / equity.cummax() - 1).min()),
        "recorded_max_drawdown_including_initial_nav": float((equity / equity.cummax().clip(lower=1) - 1).min()),
    }


def main():
    details, metrics, summaries, validation = [], [], {}, {}
    for name, relative in STAGES.items():
        directory = SOURCE / relative
        frame = pd.read_csv(directory / "equity_curve.csv", parse_dates=["timestamp", "exit_timestamp"])
        folds = pd.read_csv(directory / "walk_forward_folds.csv")
        for column in ("test_start", "test_end", "test_decision_end"):
            folds[column] = pd.to_datetime(folds[column], utc=True)
        missing_total = planned_total = unexpected_total = 0
        for fold in folds.itertuples():
            planned = pd.date_range(fold.test_start, fold.test_decision_end, freq="72h", inclusive="left")
            observed = pd.DatetimeIndex(frame.loc[frame["fold"] == fold.fold, "timestamp"])
            missing = planned.difference(observed)
            unexpected = observed.difference(planned)
            planned_total += len(planned)
            missing_total += len(missing)
            unexpected_total += len(unexpected)
            details.append({
                "stage": name, "fold": int(fold.fold),
                "test_start": fold.test_start, "test_end": fold.test_end,
                "planned_decisions": len(planned), "recorded_decisions": len(observed),
                "missing_decisions": len(missing), "unexpected_decisions": len(unexpected),
                "missing_timestamps": " | ".join(str(t) for t in missing),
            })
        horizon_hours = (frame["exit_timestamp"] - frame["timestamp"]).dt.total_seconds() / 3600 - 1
        calendar_hours = (folds["test_end"].max() - folds["test_start"].min()).total_seconds() / 3600
        summaries[name] = {
            "folds": len(folds), "planned_decisions": planned_total,
            "recorded_decisions": len(frame), "missing_decisions": missing_total,
            "unexpected_decisions": unexpected_total,
            "calendar_start": str(folds["test_start"].min()),
            "calendar_end": str(folds["test_end"].max()),
            "calendar_hours": calendar_hours,
            "represented_holding_hours": float(horizon_hours.sum()),
            "represented_holding_hours_fraction": float(horizon_hours.sum() / calendar_hours),
            "mean_recorded_gross_exposure": float(frame["gross_exposure"].mean()),
            "mean_recorded_rank_ic": float(frame["rank_ic"].mean()),
            "total_recorded_turnover": float(frame["turnover"].sum()),
            "mean_short_guard_removed": float(frame["short_guard_removed"].mean()) if "short_guard_removed" in frame else None,
            "duplicate_timestamps": int(frame["timestamp"].duplicated().sum()),
        }
        for bps in (0, 6, 8, 15, 20, 30):
            metrics.append({"stage": name, **recorded_metrics(frame, bps)})

    for relative in (
        "p4_nested/validation_candidate_metrics.csv",
        "p3d_nested/validation_incremental_metrics.csv",
        "p9_nested/validation_incremental_metrics.csv",
        "p15c_nested/validation_extreme_cluster_metrics.csv",
    ):
        frame = pd.read_csv(SOURCE / relative)
        period_cols = [c for c in ("periods", "validation_periods") if c in frame]
        record = {
            "rows": len(frame), "folds": int(frame["fold"].nunique()),
            "evaluations_per_fold_min": int(frame.groupby("fold").size().min()),
            "evaluations_per_fold_median": float(frame.groupby("fold").size().median()),
            "evaluations_per_fold_max": int(frame.groupby("fold").size().max()),
        }
        for column in period_cols:
            record[column] = {"min": float(frame[column].min()), "median": float(frame[column].median()), "max": float(frame[column].max())}
        validation[relative] = record

    evidence = {
        "scope": "Existing output audit only. No corrected PnL, funding ledger, intraperiod prices, or new backtest.",
        "cadence_assumption": "72h cadence, 1h entry delay, complete hourly panel; expected dates use each stored test_decision_end.",
        "important": "Missing records do not establish the sign of performance bias. Cash, holding, and liquidation policies require a continuous ledger.",
        "stages": summaries, "validation_search_sizes": validation,
    }
    (OUTPUT / "existing_output_audit.json").write_text(json.dumps(evidence, indent=2) + "\n")
    pd.DataFrame(details).to_csv(OUTPUT / "decision_calendar_audit.csv", index=False)
    pd.DataFrame(metrics).to_csv(OUTPUT / "recorded_metrics_cost_grid.csv", index=False)
    print(json.dumps(evidence, indent=2))
    print(pd.DataFrame(metrics).query("one_way_bps == 8").to_string(index=False))


if __name__ == "__main__":
    main()
