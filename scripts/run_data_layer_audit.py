#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "baseline_locked_v1"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "outputs" / "data_layer_audit"
DEFAULT_KLINES_DIR = EXPERIMENT_ROOT / "inputs" / "long_history" / "klines"
DEFAULT_DERIV_PANEL = EXPERIMENT_ROOT / "inputs" / "long_history" / "binance_derivatives" / "derivatives_hourly_panel.csv.gz"
DEFAULT_ASSET_PANEL = EXPERIMENT_ROOT / "inputs" / "long_history" / "baseline_panel" / "asset_residual_panel.csv"
DEFAULT_REPLAY_PANEL = EXPERIMENT_ROOT / "inputs" / "long_history" / "p16_long_replay_panel.csv.gz"
DEFAULT_FOLDS = EXPERIMENT_ROOT / "outputs" / "long_history_nested_oos" / "inputs" / "long_formula_selections.csv"


RAW_DERIVATIVE_COLS = [
    "deriv_funding_rate",
    "deriv_funding_annualized",
    "deriv_oi_value",
    "deriv_log_oi_value",
    "deriv_open_interest",
    "deriv_account_long_short_ratio",
    "deriv_top_count_long_short_ratio",
    "deriv_top_sum_long_short_ratio",
    "deriv_taker_long_short_ratio",
    "deriv_mark_index_basis",
    "deriv_premium_index",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit data layer alignment, coverage, leakage boundaries, and preprocessing choices.")
    parser.add_argument("--klines-dir", default=str(DEFAULT_KLINES_DIR))
    parser.add_argument("--derivatives-panel", default=str(DEFAULT_DERIV_PANEL))
    parser.add_argument("--asset-panel", default=str(DEFAULT_ASSET_PANEL))
    parser.add_argument("--replay-panel", default=str(DEFAULT_REPLAY_PANEL))
    parser.add_argument("--folds", default=str(DEFAULT_FOLDS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/6] Auditing 15m kline files", flush=True)
    kline_summary, kline_gaps, kline_extremes = audit_klines(Path(args.klines_dir))
    kline_summary.to_csv(output_dir / "kline_symbol_summary.csv", index=False)
    kline_gaps.to_csv(output_dir / "kline_gap_summary.csv", index=False)
    kline_extremes.to_csv(output_dir / "kline_extreme_returns.csv", index=False)

    print("[2/6] Auditing asset residual panel", flush=True)
    asset = read_panel(Path(args.asset_panel))
    asset_summary, asset_ts, asset_columns = audit_panel(asset, "asset_panel")
    asset_summary.to_csv(output_dir / "asset_panel_summary.csv", index=False)
    asset_ts.to_csv(output_dir / "asset_panel_timestamp_alignment.csv", index=False)
    asset_columns.to_csv(output_dir / "asset_panel_column_quality.csv", index=False)

    print("[3/6] Auditing derivatives panel", flush=True)
    deriv = read_panel(Path(args.derivatives_panel))
    deriv_summary, deriv_ts, deriv_columns = audit_panel(deriv, "derivatives_panel")
    deriv_summary.to_csv(output_dir / "derivatives_panel_summary.csv", index=False)
    deriv_ts.to_csv(output_dir / "derivatives_panel_timestamp_alignment.csv", index=False)
    deriv_columns.to_csv(output_dir / "derivatives_panel_column_quality.csv", index=False)
    deriv_symbol_quality = audit_derivatives_by_symbol(deriv)
    deriv_symbol_quality.to_csv(output_dir / "derivatives_symbol_quality.csv", index=False)
    deriv_freshness = audit_derivative_freshness(deriv)
    deriv_freshness.to_csv(output_dir / "derivatives_freshness_summary.csv", index=False)

    print("[4/6] Auditing replay panel", flush=True)
    replay = read_panel(Path(args.replay_panel))
    replay_summary, replay_ts, replay_columns = audit_panel(replay, "replay_panel")
    replay_summary.to_csv(output_dir / "replay_panel_summary.csv", index=False)
    replay_ts.to_csv(output_dir / "replay_panel_timestamp_alignment.csv", index=False)
    replay_columns.to_csv(output_dir / "replay_panel_column_quality.csv", index=False)

    print("[5/6] Auditing fold boundaries and source alignment", flush=True)
    folds = pd.read_csv(args.folds, parse_dates=["train_start", "train_end", "test_start", "test_end"])
    fold_audit = audit_folds(folds, replay)
    fold_audit.to_csv(output_dir / "fold_boundary_audit.csv", index=False)
    source_alignment = audit_source_alignment(asset, deriv, replay)
    source_alignment.to_csv(output_dir / "source_alignment.csv", index=False)

    print("[6/6] Writing preprocessing review", flush=True)
    transform_review = build_transform_review(asset, deriv, replay)
    transform_review.to_csv(output_dir / "preprocessing_transform_review.csv", index=False)
    robust_preprocessing = audit_robust_preprocessing(asset)
    robust_preprocessing.to_csv(output_dir / "robust_preprocessing_summary.csv", index=False)
    summary = build_topline_summary(
        kline_summary=kline_summary,
        kline_gaps=kline_gaps,
        asset_summary=asset_summary,
        deriv_summary=deriv_summary,
        replay_summary=replay_summary,
        fold_audit=fold_audit,
        source_alignment=source_alignment,
        transform_review=transform_review,
        deriv_freshness=deriv_freshness,
        robust_preprocessing=robust_preprocessing,
    )
    summary.to_csv(output_dir / "topline_findings.csv", index=False)
    metadata = {
        "klines_dir": str(Path(args.klines_dir).resolve()),
        "derivatives_panel": str(Path(args.derivatives_panel).resolve()),
        "asset_panel": str(Path(args.asset_panel).resolve()),
        "replay_panel": str(Path(args.replay_panel).resolve()),
        "folds": str(Path(args.folds).resolve()),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(
        output_dir,
        summary,
        kline_summary,
        kline_gaps,
        asset_summary,
        deriv_summary,
        replay_summary,
        source_alignment,
        transform_review,
        deriv_freshness,
        robust_preprocessing,
    )
    print(f"Done: {output_dir}", flush=True)


def read_panel(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    if "symbol" in df:
        df["symbol"] = df["symbol"].astype(str)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def audit_klines(klines_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    gap_rows: list[dict[str, Any]] = []
    extreme_rows: list[dict[str, Any]] = []
    for file in sorted(klines_dir.glob("*/15m.csv")):
        symbol = file.parent.name
        df = pd.read_csv(file, parse_dates=["timestamp", "close_time"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], utc=True)
        df = df.sort_values("timestamp")
        dup_count = int(df["timestamp"].duplicated().sum())
        expected = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="15min", tz="UTC")
        observed = pd.DatetimeIndex(df["timestamp"])
        missing = expected.difference(observed)
        extra = observed.difference(expected)
        deltas = df["timestamp"].diff().dropna().dt.total_seconds() / 60.0
        bad_ohlc = (
            (df["open"] <= 0)
            | (df["high"] <= 0)
            | (df["low"] <= 0)
            | (df["close"] <= 0)
            | (df["high"] < df[["open", "close", "low"]].max(axis=1))
            | (df["low"] > df[["open", "close", "high"]].min(axis=1))
        )
        close_lag_ms = (df["close_time"] - df["timestamp"]).dt.total_seconds() * 1000.0
        close = df["close"].astype(float)
        log_ret = np.log(close).diff()
        abs_ret = log_ret.abs()
        summary_rows.append(
            {
                "symbol": symbol,
                "rows": int(len(df)),
                "start": df["timestamp"].min(),
                "end": df["timestamp"].max(),
                "duplicate_timestamps": dup_count,
                "missing_15m_bars": int(len(missing)),
                "extra_15m_bars": int(len(extra)),
                "bad_ohlc_rows": int(bad_ohlc.sum()),
                "nonpositive_volume_rows": int((df["volume"].astype(float) < 0).sum()),
                "nonpositive_quote_volume_rows": int((df["quote_volume"].astype(float) < 0).sum()),
                "close_time_lag_ms_median": float(np.nanmedian(close_lag_ms)),
                "close_time_lag_ms_min": float(np.nanmin(close_lag_ms)),
                "close_time_lag_ms_max": float(np.nanmax(close_lag_ms)),
                "interval_min": float(deltas.min()) if len(deltas) else np.nan,
                "interval_max": float(deltas.max()) if len(deltas) else np.nan,
                "abs_log_ret_p99": float(abs_ret.quantile(0.99)),
                "abs_log_ret_max": float(abs_ret.max()),
                "zero_volume_rows": int((df["volume"].astype(float) == 0).sum()),
            }
        )
        if len(missing):
            gap_rows.append({"symbol": symbol, "gap_type": "missing_15m", "count": int(len(missing)), "first": missing.min(), "last": missing.max()})
        if len(extra):
            gap_rows.append({"symbol": symbol, "gap_type": "extra_15m", "count": int(len(extra)), "first": extra.min(), "last": extra.max()})
        top = df.loc[abs_ret.nlargest(5).index, ["timestamp", "open", "high", "low", "close", "volume", "quote_volume"]].copy()
        top["symbol"] = symbol
        top["abs_log_ret"] = abs_ret.loc[top.index].to_numpy()
        extreme_rows.append(top)
    return (
        pd.DataFrame(summary_rows).sort_values("symbol"),
        pd.DataFrame(gap_rows),
        pd.concat(extreme_rows, ignore_index=True) if extreme_rows else pd.DataFrame(),
    )


def audit_panel(df: pd.DataFrame, panel_name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    symbols = sorted(df["symbol"].dropna().astype(str).unique()) if "symbol" in df else []
    timestamps = pd.DatetimeIndex(sorted(df["timestamp"].dropna().unique()))
    expected_ts = pd.date_range(timestamps.min(), timestamps.max(), freq="1h", tz="UTC") if len(timestamps) else pd.DatetimeIndex([])
    expected_rows = len(expected_ts) * len(symbols)
    duplicates = int(df.duplicated(["timestamp", "symbol"]).sum()) if "symbol" in df else int(df["timestamp"].duplicated().sum())
    counts = df.groupby("timestamp")["symbol"].nunique() if "symbol" in df else pd.Series(dtype=float)
    ts_alignment = pd.DataFrame(
        {
            "panel": panel_name,
            "timestamp": counts.index,
            "symbol_count": counts.to_numpy(),
            "missing_symbol_count": len(symbols) - counts.to_numpy(),
        }
    )
    if len(expected_ts):
        missing_ts = expected_ts.difference(timestamps)
        extra_ts = timestamps.difference(expected_ts)
    else:
        missing_ts = pd.DatetimeIndex([])
        extra_ts = pd.DatetimeIndex([])
    summary = pd.DataFrame(
        [
            {
                "panel": panel_name,
                "rows": int(len(df)),
                "symbols": int(len(symbols)),
                "timestamps": int(len(timestamps)),
                "start": timestamps.min() if len(timestamps) else pd.NaT,
                "end": timestamps.max() if len(timestamps) else pd.NaT,
                "expected_hourly_timestamps": int(len(expected_ts)),
                "missing_hourly_timestamps": int(len(missing_ts)),
                "extra_hourly_timestamps": int(len(extra_ts)),
                "expected_balanced_rows": int(expected_rows),
                "missing_balanced_rows": int(max(expected_rows - len(df), 0)),
                "duplicate_timestamp_symbol_rows": duplicates,
                "min_symbols_per_timestamp": int(counts.min()) if len(counts) else 0,
                "q01_symbols_per_timestamp": float(counts.quantile(0.01)) if len(counts) else np.nan,
                "median_symbols_per_timestamp": float(counts.median()) if len(counts) else np.nan,
            }
        ]
    )
    column_rows: list[dict[str, Any]] = []
    for col in df.columns:
        if col in {"timestamp", "symbol"}:
            continue
        s = df[col]
        numeric = pd.to_numeric(s, errors="coerce")
        if numeric.dtype == bool:
            numeric = numeric.astype(float)
        notna = s.notna()
        finite = np.isfinite(numeric)
        first_valid = df.loc[notna, "timestamp"].min() if notna.any() else pd.NaT
        last_valid = df.loc[notna, "timestamp"].max() if notna.any() else pd.NaT
        column_rows.append(
            {
                "panel": panel_name,
                "column": col,
                "notna_rate": float(notna.mean()),
                "finite_rate": float(finite.mean()),
                "first_valid_timestamp": first_valid,
                "last_valid_timestamp": last_valid,
                "zero_rate": float((numeric == 0).mean()) if numeric.notna().any() else np.nan,
                "negative_rate": float((numeric < 0).mean()) if numeric.notna().any() else np.nan,
                "p01": float(numeric.quantile(0.01)) if numeric.notna().any() else np.nan,
                "p50": float(numeric.quantile(0.50)) if numeric.notna().any() else np.nan,
                "p99": float(numeric.quantile(0.99)) if numeric.notna().any() else np.nan,
            }
        )
    return summary, ts_alignment, pd.DataFrame(column_rows)


def audit_derivatives_by_symbol(deriv: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for symbol, group in deriv.groupby("symbol", sort=True):
        group = group.sort_values("timestamp")
        for col in [c for c in RAW_DERIVATIVE_COLS if c in group.columns]:
            s = pd.to_numeric(group[col], errors="coerce")
            notna = s.notna()
            first = group.loc[notna, "timestamp"].min() if notna.any() else pd.NaT
            after_first = group[group["timestamp"] >= first] if pd.notna(first) else group.iloc[0:0]
            miss_after_first = float(after_first[col].isna().mean()) if len(after_first) else np.nan
            rows.append(
                {
                    "symbol": symbol,
                    "column": col,
                    "notna_rate": float(notna.mean()),
                    "first_valid_timestamp": first,
                    "missing_rate_after_first_valid": miss_after_first,
                    "p01": float(s.quantile(0.01)) if s.notna().any() else np.nan,
                    "p50": float(s.quantile(0.50)) if s.notna().any() else np.nan,
                    "p99": float(s.quantile(0.99)) if s.notna().any() else np.nan,
                }
            )
    return pd.DataFrame(rows)


def audit_derivative_freshness(deriv: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    age_cols = sorted(
        [
            col
            for col in deriv.columns
            if col.endswith("_source_age_hours") and not col.startswith("deriv_data_quality_")
        ]
    )
    for age_col in age_cols:
        prefix = age_col.removesuffix("_source_age_hours")
        stale_col = f"{prefix}_is_stale"
        freshness_col = f"{prefix}_freshness_weight"
        age = pd.to_numeric(deriv[age_col], errors="coerce")
        stale = (
            pd.to_numeric(deriv[stale_col], errors="coerce").fillna(0.0).gt(0.0)
            if stale_col in deriv
            else pd.Series(False, index=deriv.index)
        )
        freshness = pd.to_numeric(deriv[freshness_col], errors="coerce") if freshness_col in deriv else pd.Series(np.nan, index=deriv.index)
        rows.append(
            {
                "field": prefix,
                "age_notna_rate": float(age.notna().mean()),
                "age_p50_hours": float(age.quantile(0.50)) if age.notna().any() else np.nan,
                "age_p95_hours": float(age.quantile(0.95)) if age.notna().any() else np.nan,
                "age_max_hours": float(age.max()) if age.notna().any() else np.nan,
                "stale_rate": float(stale.mean()) if stale_col in deriv else np.nan,
                "freshness_weight_p01": float(freshness.quantile(0.01)) if freshness.notna().any() else np.nan,
                "freshness_weight_p50": float(freshness.quantile(0.50)) if freshness.notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def audit_robust_preprocessing(asset: pd.DataFrame) -> pd.DataFrame:
    expected = {
        "log_volume": "volume log scale",
        "log_volume_diff_24": "volume acceleration",
        "log_quote_volume_diff_24": "quote-volume acceleration",
        "volume_share": "cross-sectional volume share",
        "quote_volume_share": "cross-sectional quote-volume share",
        "quote_volume_share_change_24": "relative liquidity share trend",
        "taker_buy_base_ratio": "normalized taker base buy share",
        "taker_buy_quote_ratio": "normalized taker quote buy share",
        "taker_imbalance_quote": "normalized taker imbalance",
        "taker_imbalance_quote_ewm_24": "slow taker imbalance",
        "taker_imbalance_quote_delta_24": "taker pressure acceleration",
        "log_high_low_range": "range-based volatility proxy",
        "close_location_value": "bar close location",
        "amihud_1h": "liquidity cost proxy",
        "log_ret_winsor_720": "past-only winsorized return",
        "log_ret_robust_z_720": "past-vol standardized return",
        "abs_log_ret_outlier_720": "past-vol outlier flag",
    }
    rows: list[dict[str, Any]] = []
    for col, purpose in expected.items():
        if col not in asset:
            rows.append({"column": col, "purpose": purpose, "present": False})
            continue
        s = asset[col]
        numeric = pd.to_numeric(s, errors="coerce")
        rows.append(
            {
                "column": col,
                "purpose": purpose,
                "present": True,
                "notna_rate": float(s.notna().mean()),
                "finite_rate": float(np.isfinite(numeric).mean()),
                "p01": float(numeric.quantile(0.01)) if numeric.notna().any() else np.nan,
                "p50": float(numeric.quantile(0.50)) if numeric.notna().any() else np.nan,
                "p99": float(numeric.quantile(0.99)) if numeric.notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def audit_folds(folds: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    start = panel["timestamp"].min()
    end = panel["timestamp"].max()
    rows: list[dict[str, Any]] = []
    for _, row in folds.sort_values("fold").iterrows():
        train_start = pd.Timestamp(row["train_start"])
        train_end = pd.Timestamp(row["train_end"])
        test_start = pd.Timestamp(row["test_start"])
        test_end = pd.Timestamp(row["test_end"])
        rows.append(
            {
                "fold": int(row["fold"]),
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "train_before_test": bool(train_end <= test_start),
                "test_inside_panel": bool(test_start >= start and test_end <= end),
                "train_inside_panel": bool(train_start >= start and train_end <= end),
                "gap_hours_train_to_test": float((test_start - train_end).total_seconds() / 3600.0),
                "test_rows": int(panel[(panel["timestamp"] >= test_start) & (panel["timestamp"] < test_end)].shape[0]),
                "train_rows": int(panel[(panel["timestamp"] >= train_start) & (panel["timestamp"] < train_end)].shape[0]),
            }
        )
    return pd.DataFrame(rows)


def audit_source_alignment(asset: pd.DataFrame, deriv: pd.DataFrame, replay: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    sources = {"asset": asset, "derivatives": deriv, "replay": replay}
    for name, df in sources.items():
        key = df[["timestamp", "symbol"]].drop_duplicates()
        rows.append(
            {
                "source": name,
                "rows": int(len(df)),
                "unique_timestamp_symbol": int(len(key)),
                "symbols": int(key["symbol"].nunique()),
                "timestamps": int(key["timestamp"].nunique()),
                "start": key["timestamp"].min(),
                "end": key["timestamp"].max(),
            }
        )
    asset_keys = set(map(tuple, asset[["timestamp", "symbol"]].drop_duplicates().to_numpy()))
    deriv_keys = set(map(tuple, deriv[["timestamp", "symbol"]].drop_duplicates().to_numpy()))
    replay_keys = set(map(tuple, replay[["timestamp", "symbol"]].drop_duplicates().to_numpy()))
    rows.extend(
        [
            {"source": "asset_minus_derivatives_keys", "rows": len(asset_keys - deriv_keys)},
            {"source": "derivatives_minus_asset_keys", "rows": len(deriv_keys - asset_keys)},
            {"source": "replay_minus_asset_keys", "rows": len(replay_keys - asset_keys)},
            {"source": "asset_minus_replay_keys", "rows": len(asset_keys - replay_keys)},
        ]
    )
    return pd.DataFrame(rows)


def build_transform_review(asset: pd.DataFrame, deriv: pd.DataFrame, replay: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    add = rows.append
    add({"field_family": "price", "current_state": "asset panel has log_close/log_ret/simple_ret; residual model uses log_ret.", "recommended_input": "log price for level features; log return for beta/residual; never raw price across symbols.", "action": "keep"})
    add({"field_family": "volume/quote_volume/trades", "current_state": "asset panel has log_quote_volume, log-volume diffs, liquidity share, and liquidity_z.", "recommended_input": "log1p volume/quote_volume/trades, rolling share, rolling z, and log-diff for liquidity acceleration.", "action": "done"})
    add({"field_family": "taker buy/sell flow", "current_state": "asset panel has normalized taker buy ratio, taker imbalance, EWMA, and delta.", "recommended_input": "use imbalance = 2*taker_buy/volume - 1, then EWMA/delta/z; raw taker quantities should be log-scaled or normalized.", "action": "done"})
    add({"field_family": "open_interest", "current_state": "derivatives panel has OI level, log OI value, EWM, delta, z, OI growth.", "recommended_input": "log OI value level for leverage stock; log-diff/delta for leverage flow; OI/volume for crowdedness.", "action": "mostly ok; add OI/volume if not present in active panel"})
    add({"field_family": "long_short_ratios", "current_state": "derivatives panel stores log(ratio), EWM, deltas, rolling z, and freshness metadata.", "recommended_input": "keep log-ratio convention documented; raw ratio is skewed and asymmetric around 1.", "action": "done"})
    add({"field_family": "funding", "current_state": "funding level, annualized, EWM, delta, z, acceleration are present.", "recommended_input": "level is economically meaningful; use EWM/delta/z and carry-adjusted return. Do not only diff it.", "action": "keep"})
    add({"field_family": "premium/basis", "current_state": "premium and mark-index basis level, EWM, delta, z are present.", "recommended_input": "level for rich/cheap state, delta for worsening/normalization, z for cross-time comparability.", "action": "keep"})
    add({"field_family": "beta/residual", "current_state": "beta, residual_log_ret, residual z/momentum/quality features are present.", "recommended_input": "residual return should be modeled in log-return units; residual level anchors should use cumulative residual index/log residual equity.", "action": "check residual index construction"})
    add({"field_family": "cross_sectional_features", "current_state": "many features are already CS rank/zscore or EMA of ranks.", "recommended_input": "preferred for ranking alpha; keep raw transformed values only for diagnostics and calibration.", "action": "keep"})
    add({"field_family": "missing_data", "current_state": "warmup NaNs exist; derivatives raw fields are shifted by 1h and now include source age/stale/freshness metadata.", "recommended_input": "never fill alpha features cross-sectionally; if forward-fill derivatives, add freshness age and cap max stale hours.", "action": "done"})
    return pd.DataFrame(rows)


def build_topline_summary(
    kline_summary: pd.DataFrame,
    kline_gaps: pd.DataFrame,
    asset_summary: pd.DataFrame,
    deriv_summary: pd.DataFrame,
    replay_summary: pd.DataFrame,
    fold_audit: pd.DataFrame,
    source_alignment: pd.DataFrame,
    transform_review: pd.DataFrame,
    deriv_freshness: pd.DataFrame,
    robust_preprocessing: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows.append({"check": "kline_missing_15m_bars_total", "value": int(kline_summary["missing_15m_bars"].sum()), "status": "pass" if int(kline_summary["missing_15m_bars"].sum()) == 0 else "review"})
    rows.append({"check": "kline_duplicate_timestamps_total", "value": int(kline_summary["duplicate_timestamps"].sum()), "status": "pass" if int(kline_summary["duplicate_timestamps"].sum()) == 0 else "fail"})
    rows.append({"check": "kline_bad_ohlc_rows_total", "value": int(kline_summary["bad_ohlc_rows"].sum()), "status": "pass" if int(kline_summary["bad_ohlc_rows"].sum()) == 0 else "fail"})
    for name, summary in [("asset", asset_summary), ("derivatives", deriv_summary), ("replay", replay_summary)]:
        row = summary.iloc[0]
        rows.append({"check": f"{name}_missing_hourly_timestamps", "value": int(row["missing_hourly_timestamps"]), "status": "pass" if int(row["missing_hourly_timestamps"]) == 0 else "review"})
        rows.append({"check": f"{name}_min_symbols_per_timestamp", "value": int(row["min_symbols_per_timestamp"]), "status": "pass" if int(row["min_symbols_per_timestamp"]) >= 20 else "review"})
        rows.append({"check": f"{name}_duplicate_timestamp_symbol_rows", "value": int(row["duplicate_timestamp_symbol_rows"]), "status": "pass" if int(row["duplicate_timestamp_symbol_rows"]) == 0 else "fail"})
    rows.append({"check": "fold_train_before_test_all", "value": bool(fold_audit["train_before_test"].all()), "status": "pass" if bool(fold_audit["train_before_test"].all()) else "fail"})
    rows.append({"check": "fold_test_inside_panel_all", "value": bool(fold_audit["test_inside_panel"].all()), "status": "pass" if bool(fold_audit["test_inside_panel"].all()) else "fail"})
    key_diffs = source_alignment[source_alignment["source"].astype(str).str.contains("minus")]
    for _, row in key_diffs.iterrows():
        rows.append({"check": row["source"], "value": int(row["rows"]), "status": "pass" if int(row["rows"]) == 0 else "review"})
    freshness_count = int(len(deriv_freshness)) if deriv_freshness is not None else 0
    rows.append({"check": "derivatives_freshness_field_count", "value": freshness_count, "status": "pass" if freshness_count >= 10 else "review"})
    if freshness_count:
        stale_rate = float(deriv_freshness["stale_rate"].fillna(0.0).max())
        rows.append({"check": "derivatives_max_stale_rate", "value": stale_rate, "status": "pass" if stale_rate < 0.01 else "review"})
    missing_preprocess = int((~robust_preprocessing["present"].fillna(False).astype(bool)).sum()) if robust_preprocessing is not None else 999
    rows.append({"check": "robust_preprocessing_missing_columns", "value": missing_preprocess, "status": "pass" if missing_preprocess == 0 else "review"})
    rows.append({"check": "transform_review_action_count", "value": int((transform_review["action"] != "keep").sum()), "status": "review"})
    return pd.DataFrame(rows)


def write_report(
    output_dir: Path,
    summary: pd.DataFrame,
    kline_summary: pd.DataFrame,
    kline_gaps: pd.DataFrame,
    asset_summary: pd.DataFrame,
    deriv_summary: pd.DataFrame,
    replay_summary: pd.DataFrame,
    source_alignment: pd.DataFrame,
    transform_review: pd.DataFrame,
    deriv_freshness: pd.DataFrame,
    robust_preprocessing: pd.DataFrame,
) -> None:
    report = [
        "# Data Layer Audit",
        "",
        "This audit checks raw 15m Binance klines, hourly derivatives, residual/replay panels, fold boundaries, and preprocessing choices.",
        "",
        "## Topline Findings",
        "",
        "```text",
        summary.to_string(index=False),
        "```",
        "",
        "## Kline Summary",
        "",
        "```text",
        kline_summary.to_string(index=False),
        "```",
        "",
        "## Kline Gaps",
        "",
        "```text",
        kline_gaps.to_string(index=False) if not kline_gaps.empty else "No gaps.",
        "```",
        "",
        "## Panel Summaries",
        "",
        "```text",
        pd.concat([asset_summary, deriv_summary, replay_summary], ignore_index=True).to_string(index=False),
        "```",
        "",
        "## Source Alignment",
        "",
        "```text",
        source_alignment.to_string(index=False),
        "```",
        "",
        "## Preprocessing Transform Review",
        "",
        "```text",
        transform_review.to_string(index=False),
        "```",
        "",
        "## Derivatives Freshness",
        "",
        "```text",
        deriv_freshness.to_string(index=False) if not deriv_freshness.empty else "No freshness fields found.",
        "```",
        "",
        "## Robust Preprocessing",
        "",
        "```text",
        robust_preprocessing.to_string(index=False),
        "```",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
