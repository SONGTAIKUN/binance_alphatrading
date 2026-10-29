#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from crypto_stat_arb.config import ensure_dir, load_config
from crypto_stat_arb.data import load_universe_1h
from crypto_stat_arb.factors import ResidualConfig, add_factor_inputs, rolling_residualize
from crypto_stat_arb.features import add_baseline_score, add_cross_sectional_features, add_forward_labels, add_residual_features
from run_latest_baseline import evaluate_formula_text
from run_p2_family_a_momentum import build_family_a_features
from run_p2_family_b_mean_reversion import centered_rank_by_timestamp, make_portfolio_config
from run_p2_family_b_mean_reversion import build_family_b_features
from run_p2_manual_slow_alpha_grid import build_manual_slow_hybrid_features
from run_p3_committee_guard_scan import COMPONENTS, add_market_regime_features, make_risk_template
import run_p3d_p4_gated_rerank_alphagpt as p3d
import run_p15c_extreme_cluster_turnover as p15c
from run_residual_alphagpt_search import build_candidate_scores, prepare_alpha_panel
from run_turnover_experiments import to_jsonable


EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "baseline_locked_v1"
BASE_CONFIG = PROJECT_ROOT / "configs" / "baseline.yaml"
LOCKED_CONFIG = EXPERIMENT_ROOT / "configs" / "baseline_locked_v1.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "live_p16_cs72h_1month"
DEFAULT_P9_SELECTED = EXPERIMENT_ROOT / "outputs" / "p9_clean_baseline_package" / "selected_incremental_by_fold.csv"
DEFAULT_P16_SELECTED = EXPERIMENT_ROOT / "outputs" / "p16_clean_candidate_package" / "fixed_selected_extreme_cluster_by_fold.csv"
DEFAULT_P3D_CANDIDATES = EXPERIMENT_ROOT / "outputs" / "p3d_p4_gated_rerank_alphagpt" / "candidate_formulas.csv"
BINANCE_FAPI_BASE = "https://fapi.binance.com"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a local P16-CS-72H live-cost paper-trading monitor.")
    parser.add_argument("--start-mode", choices=["rolling_window", "forward"], default="forward")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--history-days", type=float, default=420.0)
    parser.add_argument("--window-days", type=float, default=30.0)
    parser.add_argument("--refresh-seconds", type=int, default=900)
    parser.add_argument("--initial-capital", type=float, default=10000.0)
    parser.add_argument("--skip-download", action="store_true", help="use existing data/klines CSV files only")
    parser.add_argument("--complete-lag-seconds", type=float, default=90.0)
    parser.add_argument("--execution-cost-model", choices=["fixed_bps", "orderbook"], default="orderbook")
    parser.add_argument("--taker-fee-bps", type=float, default=4.0)
    parser.add_argument("--fallback-slippage-bps", type=float, default=2.0)
    parser.add_argument("--orderbook-limit", type=int, default=20)
    parser.add_argument("--disable-funding", action="store_true")
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    state_path = output_dir / "live_state.json"
    if args.reset and state_path.exists():
        state_path.unlink()

    while True:
        try:
            summary = run_refresh(args, output_dir, state_path)
            status = build_status(summary, output_dir, args)
            (output_dir / "STATUS.txt").write_text(status, encoding="utf-8")
            print(status)
            print("")
            print(f"Next refresh in {int(args.refresh_seconds)} seconds. Press Ctrl+C to stop.")
        except Exception as exc:  # pragma: no cover - operational path
            output_dir.mkdir(parents=True, exist_ok=True)
            error_status = build_error_status(exc)
            (output_dir / "STATUS.txt").write_text(error_status, encoding="utf-8")
            (output_dir / "ERROR.log").write_text(traceback.format_exc(), encoding="utf-8")
            print(error_status)
            if args.once:
                return 1

        if args.once:
            break
        if args.start_mode == "forward" and forward_finished(state_path):
            print("Forward paper duration reached. Exiting.")
            break
        time.sleep(max(30, int(args.refresh_seconds)))
    return 0


def run_refresh(args: argparse.Namespace, output_dir: Path, state_path: Path) -> dict[str, Any]:
    base_cfg = load_config(BASE_CONFIG)
    locked_cfg = load_config(LOCKED_CONFIG)
    symbols = list(base_cfg["data"]["symbols"])
    horizon = int(locked_cfg["locked_baseline"]["horizon_hours"])

    if not args.skip_download:
        download_klines(symbols, args.history_days, float(args.complete_lag_seconds))

    panel = build_live_panel(base_cfg, locked_cfg, symbols, horizon, complete_lag_seconds=float(args.complete_lag_seconds))
    p9_row, p16_row, formula_text = apply_latest_locked_score(panel)
    latest_ts = pd.Timestamp(panel["timestamp"].max()).tz_convert("UTC")
    residual_wide = panel.pivot(index="timestamp", columns="symbol", values="residual_log_ret").sort_index()
    pf_cfg = make_portfolio_config(locked_cfg, horizon)
    signal_ts = latest_signal_timestamp(panel, latest_ts, int(pf_cfg.execution_delay_bars))
    risk_template = make_risk_template(locked_cfg)
    committee = {spec.name: spec for spec in p3d.build_committee_specs()}[str(p9_row["p4_spec"])]
    spec = selected_extreme_spec(p16_row)

    rolling = run_rolling_window(
        panel=panel,
        residual_wide=residual_wide,
        latest_ts=latest_ts,
        window_days=float(args.window_days),
        horizon=horizon,
        pf_cfg=pf_cfg,
        risk_template=risk_template,
        committee=committee,
        spec=spec,
        p16_row=p16_row,
        formula_text=formula_text,
    )

    forward = None
    if args.start_mode == "forward":
        state = load_or_init_state(state_path, args, horizon)
        forward = update_forward_state(
            state=state,
            state_path=state_path,
            panel=panel,
            residual_wide=residual_wide,
            latest_ts=latest_ts,
            signal_ts=signal_ts,
            pf_cfg=pf_cfg,
            risk_template=risk_template,
            committee=committee,
            spec=spec,
            p16_row=p16_row,
            formula_text=formula_text,
            output_dir=output_dir,
            args=args,
        )

    write_outputs(output_dir, panel, rolling, forward, p9_row, p16_row, formula_text, latest_ts)
    return {
        "mode": args.start_mode,
        "latest_ts": str(latest_ts),
        "signal_ts": str(signal_ts),
        "symbols": symbols,
        "horizon": horizon,
        "rolling": rolling,
        "forward": forward,
        "p9_selected": dict(p9_row),
        "p16_selected": dict(p16_row),
        "formula_text": formula_text,
    }


def download_klines(symbols: list[str], history_days: float, complete_lag_seconds: float) -> None:
    script = SCRIPT_ROOT / "download_binance_klines.py"
    cmd = [
        sys.executable,
        str(script),
        "--symbols",
        *symbols,
        "--history-days",
        str(float(history_days)),
        "--complete-lag-seconds",
        str(float(complete_lag_seconds)),
        "--output-dir",
        str(PROJECT_ROOT / "data" / "klines"),
    ]
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def build_live_panel(
    base_cfg: dict[str, Any],
    locked_cfg: dict[str, Any],
    symbols: list[str],
    horizon: int,
    complete_lag_seconds: float,
) -> pd.DataFrame:
    data = load_universe_1h(PROJECT_ROOT / base_cfg["data"]["klines_dir"], symbols)
    data = keep_completed_hourly_rows(data, complete_lag_seconds)
    if data.empty:
        raise RuntimeError("No completed hourly rows are available after live completeness filtering.")
    data = add_factor_inputs(data, base_cfg["sectors"])
    residual_cfg = ResidualConfig(
        beta_window=int(base_cfg["factor_model"]["beta_window"]),
        min_beta_obs=int(base_cfg["factor_model"]["min_beta_obs"]),
    )
    panel = rolling_residualize(data, residual_cfg)
    panel = add_residual_features(
        panel,
        residual_z_window=int(base_cfg["factor_model"]["residual_z_window"]),
        windows=base_cfg["factor_model"]["feature_windows"],
    )
    panel = add_cross_sectional_features(panel)
    panel = add_baseline_score(panel)
    panel = add_forward_labels(panel, horizons=[horizon], execution_delay_bars=1)
    panel = prepare_p4_panel(panel, horizon)
    panel = panel.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    required = ["rank_a07", "rank_b11", "rank_c09", "score_ema_025", "score_baseline", "market_vol_14d", "residual_dispersion_7d"]
    missing = [col for col in required if col not in panel.columns]
    if missing:
        raise KeyError(f"Live panel missing required columns: {missing}")
    return panel


def keep_completed_hourly_rows(data: pd.DataFrame, complete_lag_seconds: float) -> pd.DataFrame:
    out = data.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    now = pd.Timestamp.now(tz="UTC")
    latest_allowed_open = (now - pd.Timedelta(hours=1, seconds=max(0.0, float(complete_lag_seconds)))).floor("h")
    return out[out["timestamp"] <= latest_allowed_open].copy()


def latest_signal_timestamp(panel: pd.DataFrame, latest_data_ts: pd.Timestamp, execution_delay_bars: int) -> pd.Timestamp:
    cutoff = pd.Timestamp(latest_data_ts).tz_convert("UTC") - pd.Timedelta(hours=max(0, int(execution_delay_bars)))
    timestamps = pd.to_datetime(panel["timestamp"], utc=True).drop_duplicates().sort_values()
    eligible = timestamps[timestamps <= cutoff]
    if eligible.empty:
        raise RuntimeError(f"No signal timestamp is available at or before {cutoff}.")
    return pd.Timestamp(eligible.iloc[-1]).tz_convert("UTC")


def prepare_p4_panel(panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    panel = prepare_alpha_panel(panel, [horizon, 48, 96, 168], execution_delay_bars=1)
    build_candidate_scores(panel)
    build_family_a_features(panel)
    build_family_b_features(panel)
    build_manual_slow_hybrid_features(panel)
    add_market_regime_features(panel)
    for key, col in COMPONENTS.items():
        panel[f"rank_{key.lower()}"] = centered_rank_by_timestamp(panel, col)
    p3d.add_ex_ante_high_risk_regime(panel)
    return panel


def apply_latest_locked_score(panel: pd.DataFrame) -> tuple[pd.Series, pd.Series, str]:
    p9 = pd.read_csv(DEFAULT_P9_SELECTED, parse_dates=["train_start", "train_end", "test_start", "test_end", "test_decision_end"])
    p16 = pd.read_csv(DEFAULT_P16_SELECTED)
    p3d_candidates = pd.read_csv(DEFAULT_P3D_CANDIDATES)
    p3d_formula_text = dict(zip(p3d_candidates["name"].astype(str), p3d_candidates["formula"].astype(str)))
    p9_row = p9.sort_values("fold").iloc[-1]
    p16_row = p16.sort_values("fold").iloc[-1]
    committee = {spec.name: spec for spec in p3d.build_committee_specs()}[str(p9_row["p4_spec"])]

    panel[p3d.P4_SCORE_COL] = p3d.p4_score(panel, committee)
    p3d_name = str(p9_row["p3d_formula_name"])
    if p3d_name == p3d.P4_ONLY_FORMULA:
        panel[p3d.SCORE_COL] = panel[p3d.P4_SCORE_COL]
        panel[p3d.CANDIDATE_RANK_COL] = np.nan
        panel[p3d.EFFECTIVE_CANDIDATE_WEIGHT_COL] = 0.0
        formula_text = f"P4 only ({committee.name})"
    else:
        formula = p3d_formula_text[p3d_name]
        candidate_rank = evaluate_formula_text(formula, panel, namespace="p3d").groupby(
            panel["timestamp"], sort=False
        ).rank(pct=True, method="average")
        panel[p3d.CANDIDATE_RANK_COL] = 2.0 * candidate_rank - 1.0
        p3d.apply_constrained_incremental_score(
            panel,
            candidate_weight_long=float(p9_row["p3d_candidate_weight_long"]),
            candidate_weight_short=float(p9_row["p3d_candidate_weight_short"]),
            high_risk_candidate_cap=float(p9_row["p3d_high_risk_candidate_cap"]),
        )
        formula_text = f"P3d_base({p3d_name}, {formula})"

    deriv_name = str(p9_row.get("deriv_formula_name", p3d.P4_ONLY_FORMULA))
    if deriv_name != p3d.P4_ONLY_FORMULA:
        raise RuntimeError(
            "Latest locked fold uses a derivative increment. This paper runner currently supports "
            "the latest v2 fold-23 setting, where derivative increment is P4-only/fallback."
        )
    return p9_row, p16_row, formula_text


def selected_extreme_spec(p16_row: pd.Series) -> p15c.ExtremeClusterSpec | None:
    if bool(p16_row.get("fallback_to_p9_sizing", False)):
        return None
    spec_map = {spec.name: spec for spec in p15c.build_extreme_specs()}
    return spec_map[str(p16_row["selected_extreme_spec"])]


def run_rolling_window(
    panel: pd.DataFrame,
    residual_wide: pd.DataFrame,
    latest_ts: pd.Timestamp,
    window_days: float,
    horizon: int,
    pf_cfg: Any,
    risk_template: Any,
    committee: Any,
    spec: Any,
    p16_row: pd.Series,
    formula_text: str,
) -> dict[str, Any]:
    end = latest_ts - pd.Timedelta(hours=horizon + 1)
    start = end - pd.Timedelta(days=window_days)
    if start >= end:
        return empty_rolling(start, end)
    equity, positions, metrics = p15c.run_extreme_backtest(
        panel=panel,
        residual_wide=residual_wide,
        start=start,
        end=end,
        cfg=pf_cfg,
        risk_cfg=risk_template,
        committee=committee,
        spec=spec,
        diagnostic_lookback=int(p16_row["lookback_hours"]),
        diagnostic_k=int(p16_row["cluster_k"]),
        score_col=p3d.SCORE_COL,
        selected_formulas=formula_text,
    )
    p15c.enrich_extreme_metrics(equity, positions, metrics, pf_cfg.primary_horizon)
    return {
        "paper_start": str(start),
        "paper_end": str(end),
        "equity": equity,
        "positions": positions,
        "metrics": metrics,
    }


def empty_rolling(start: pd.Timestamp, end: pd.Timestamp) -> dict[str, Any]:
    return {"paper_start": str(start), "paper_end": str(end), "equity": pd.DataFrame(), "positions": pd.DataFrame(), "metrics": {}}


def load_or_init_state(state_path: Path, args: argparse.Namespace, horizon: int) -> dict[str, Any]:
    if state_path.exists():
        return json.loads(state_path.read_text(encoding="utf-8"))
    now = pd.Timestamp.now(tz="UTC")
    state = {
        "created_at_utc": str(now),
        "forward_start_utc": str(now),
        "forward_end_utc": str(now + pd.Timedelta(days=float(args.window_days))),
        "initial_capital": float(args.initial_capital),
        "entry_equity": float(args.initial_capital),
        "weights": {},
        "entry_prices": {},
        "current_state": {},
        "last_rebalance_timestamp": None,
        "last_rebalance_signal_timestamp": None,
        "last_funding_timestamp": None,
        "rebalance_hours": int(horizon),
        "execution_metadata": {},
    }
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state


def forward_finished(state_path: Path) -> bool:
    if not state_path.exists():
        return False
    state = json.loads(state_path.read_text(encoding="utf-8"))
    return pd.Timestamp.now(tz="UTC") >= pd.Timestamp(state["forward_end_utc"]).tz_convert("UTC")


def binance_get_json(path: str, params: dict[str, Any]) -> Any:
    query = urlencode(params)
    url = BINANCE_FAPI_BASE + path + ("?" + query if query else "")
    req = Request(url, headers={"User-Agent": "p16-cs72h-live-cost-paper/1.0"})
    with urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_live_mark_prices(symbols: list[str], fallback_prices: dict[str, float]) -> dict[str, float]:
    prices = {symbol: float(fallback_prices[symbol]) for symbol in symbols if symbol in fallback_prices}
    try:
        data = binance_get_json("/fapi/v1/premiumIndex", {})
        rows = data if isinstance(data, list) else [data]
        mark_map = {
            str(row["symbol"]): float(row["markPrice"])
            for row in rows
            if "symbol" in row and "markPrice" in row and np.isfinite(float(row["markPrice"]))
        }
        for symbol in symbols:
            if symbol in mark_map:
                prices[symbol] = mark_map[symbol]
    except Exception:
        pass
    return prices


def fetch_orderbook(symbol: str, limit: int) -> dict[str, Any]:
    data = binance_get_json("/fapi/v1/depth", {"symbol": symbol, "limit": int(limit)})
    bids = [(float(price), float(qty)) for price, qty in data.get("bids", [])]
    asks = [(float(price), float(qty)) for price, qty in data.get("asks", [])]
    if not bids or not asks:
        raise RuntimeError(f"Empty orderbook for {symbol}")
    best_bid = bids[0][0]
    best_ask = asks[0][0]
    mid = (best_bid + best_ask) / 2.0
    return {
        "symbol": symbol,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "mid": mid,
        "spread_bps": (best_ask - best_bid) / mid * 10_000.0 if mid > 0 else np.nan,
        "bids": bids,
        "asks": asks,
    }


def average_market_fill(levels: list[tuple[float, float]], quote_notional: float) -> tuple[float, float, int]:
    remaining = float(max(0.0, quote_notional))
    quote_filled = 0.0
    base_filled = 0.0
    levels_used = 0
    for price, base_qty in levels:
        if remaining <= 1e-12:
            break
        level_quote = float(price) * float(base_qty)
        take_quote = min(remaining, level_quote)
        if take_quote <= 0:
            continue
        quote_filled += take_quote
        base_filled += take_quote / float(price)
        remaining -= take_quote
        levels_used += 1
    avg_price = quote_filled / base_filled if base_filled > 0 else np.nan
    fill_ratio = quote_filled / quote_notional if quote_notional > 0 else 1.0
    return avg_price, min(1.0, fill_ratio), levels_used


def estimate_execution_costs(
    old_weights: pd.Series,
    new_weights: pd.Series,
    equity: float,
    latest_ts: pd.Timestamp,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], pd.DataFrame]:
    delta = (new_weights - old_weights).astype(float)
    rows: list[dict[str, Any]] = []
    fee_return = 0.0
    slippage_return = 0.0
    fallback_count = 0
    for symbol, delta_weight in delta[delta.abs() > 1e-12].items():
        side = "buy" if delta_weight > 0 else "sell"
        abs_delta = abs(float(delta_weight))
        notional = abs_delta * float(equity)
        fee_bps = float(args.taker_fee_bps)
        slippage_bps = float(args.fallback_slippage_bps)
        spread_bps = np.nan
        best_bid = np.nan
        best_ask = np.nan
        mid = np.nan
        avg_fill_price = np.nan
        fill_ratio = 0.0
        levels_used = 0
        fallback_used = False
        error = ""
        if args.execution_cost_model == "orderbook" and notional > 0:
            try:
                book = fetch_orderbook(symbol, int(args.orderbook_limit))
                best_bid = book["best_bid"]
                best_ask = book["best_ask"]
                mid = book["mid"]
                spread_bps = book["spread_bps"]
                levels = book["asks"] if side == "buy" else book["bids"]
                avg_fill_price, fill_ratio, levels_used = average_market_fill(levels, notional)
                if np.isfinite(avg_fill_price) and np.isfinite(mid) and mid > 0:
                    if side == "buy":
                        slippage_bps = max(0.0, (avg_fill_price / mid - 1.0) * 10_000.0)
                    else:
                        slippage_bps = max(0.0, (mid / avg_fill_price - 1.0) * 10_000.0)
                if fill_ratio < 0.999:
                    fallback_used = True
                    slippage_bps = max(slippage_bps, float(args.fallback_slippage_bps)) + (1.0 - fill_ratio) * 10.0
            except Exception as exc:  # operational fallback
                fallback_used = True
                error = f"{type(exc).__name__}: {exc}"
        else:
            fallback_used = True

        if fallback_used:
            fallback_count += 1
        fee_cost_return = abs_delta * fee_bps / 10_000.0
        slippage_cost_return = abs_delta * slippage_bps / 10_000.0
        total_cost_return = fee_cost_return + slippage_cost_return
        fee_return += fee_cost_return
        slippage_return += slippage_cost_return
        rows.append(
            {
                "timestamp": latest_ts,
                "symbol": symbol,
                "side": side,
                "old_weight": float(old_weights.get(symbol, 0.0)),
                "new_weight": float(new_weights.get(symbol, 0.0)),
                "delta_weight": float(delta_weight),
                "abs_delta_weight": abs_delta,
                "estimated_notional_usdt": notional,
                "best_bid": best_bid,
                "best_ask": best_ask,
                "mid": mid,
                "spread_bps": spread_bps,
                "avg_fill_price": avg_fill_price,
                "estimated_slippage_bps": slippage_bps,
                "taker_fee_bps": fee_bps,
                "total_cost_bps": fee_bps + slippage_bps,
                "fee_cost_return": fee_cost_return,
                "slippage_cost_return": slippage_cost_return,
                "total_cost_return": total_cost_return,
                "orderbook_fill_ratio": fill_ratio,
                "orderbook_levels_used": levels_used,
                "fallback_used": fallback_used,
                "error": error,
            }
        )
        time.sleep(0.03)
    frame = pd.DataFrame(rows)
    cost_return = float(fee_return + slippage_return)
    weighted_slip = float((frame["abs_delta_weight"] * frame["estimated_slippage_bps"]).sum() / frame["abs_delta_weight"].sum()) if not frame.empty and frame["abs_delta_weight"].sum() > 0 else 0.0
    weighted_spread = float((frame["abs_delta_weight"] * frame["spread_bps"].fillna(0.0)).sum() / frame["abs_delta_weight"].sum()) if not frame.empty and frame["abs_delta_weight"].sum() > 0 else 0.0
    summary = {
        "cost_model": str(args.execution_cost_model),
        "cost_return": cost_return,
        "fee_return": float(fee_return),
        "slippage_return": float(slippage_return),
        "weighted_slippage_bps": weighted_slip,
        "weighted_spread_bps": weighted_spread,
        "fallback_orderbook_count": int(fallback_count),
        "orders": int(len(frame)),
    }
    return summary, frame


def append_frame_csv(path: Path, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    if path.exists() and path.stat().st_size > 0:
        old = pd.read_csv(path)
        frame = pd.concat([old, frame], ignore_index=True)
    frame.to_csv(path, index=False)


def fetch_funding_rates(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> list[dict[str, Any]]:
    start_ms = int((start + pd.Timedelta(milliseconds=1)).timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    if start_ms >= end_ms:
        return []
    data = binance_get_json("/fapi/v1/fundingRate", {"symbol": symbol, "startTime": start_ms, "endTime": end_ms, "limit": 1000})
    out: list[dict[str, Any]] = []
    for item in data:
        out.append(
            {
                "funding_time": pd.to_datetime(int(item["fundingTime"]), unit="ms", utc=True),
                "symbol": symbol,
                "funding_rate": float(item["fundingRate"]),
            }
        )
    return out


def apply_funding(
    state: dict[str, Any],
    active_weights: pd.Series,
    latest_ts: pd.Timestamp,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if args.disable_funding:
        return {"funding_return": 0.0, "funding_events": 0, "funding_error_count": 0}
    last_raw = state.get("last_funding_timestamp")
    if not last_raw:
        state["last_funding_timestamp"] = str(latest_ts)
        return {"funding_return": 0.0, "funding_events": 0, "funding_error_count": 0}
    start = pd.Timestamp(last_raw).tz_convert("UTC")
    end = pd.Timestamp(latest_ts).tz_convert("UTC")
    if end <= start or active_weights.abs().sum() <= 1e-12:
        state["last_funding_timestamp"] = str(latest_ts)
        return {"funding_return": 0.0, "funding_events": 0, "funding_error_count": 0}

    rows: list[dict[str, Any]] = []
    funding_return = 0.0
    error_count = 0
    for symbol, weight in active_weights[active_weights.abs() > 1e-12].items():
        try:
            events = fetch_funding_rates(symbol, start, end)
        except Exception as exc:
            error_count += 1
            rows.append(
                {
                    "funding_time": end,
                    "symbol": symbol,
                    "weight": float(weight),
                    "funding_rate": np.nan,
                    "funding_return": 0.0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        for event in events:
            # USD-M funding convention: positive funding is paid by longs and received by shorts.
            event_return = -float(weight) * float(event["funding_rate"])
            funding_return += event_return
            rows.append(
                {
                    "funding_time": event["funding_time"],
                    "symbol": symbol,
                    "weight": float(weight),
                    "funding_rate": float(event["funding_rate"]),
                    "funding_return": event_return,
                    "error": "",
                }
            )
        time.sleep(0.03)

    if rows:
        append_frame_csv(output_dir / "paper_funding.csv", pd.DataFrame(rows))
    state["last_funding_timestamp"] = str(latest_ts)
    if abs(funding_return) > 0:
        state["entry_equity"] = float(state.get("entry_equity", state.get("initial_capital", 10000.0))) * (1.0 + funding_return)
    funding_events = sum(1 for row in rows if not row.get("error"))
    return {"funding_return": float(funding_return), "funding_events": int(funding_events), "funding_error_count": int(error_count)}


def append_mark_snapshot(
    output_dir: Path,
    latest_ts: pd.Timestamp,
    kline_prices: dict[str, float],
    live_prices: dict[str, float],
    active_weights: pd.Series,
) -> None:
    rows = []
    for symbol, weight in active_weights[active_weights.abs() > 1e-12].items():
        kline_close = float(kline_prices.get(symbol, np.nan))
        mark_price = float(live_prices.get(symbol, kline_close))
        rows.append(
            {
                "updated_utc": pd.Timestamp.now(tz="UTC"),
                "latest_kline_timestamp": latest_ts,
                "symbol": symbol,
                "weight": float(weight),
                "live_mark_price": mark_price,
                "latest_kline_close": kline_close,
                "mark_vs_kline_return": mark_price / kline_close - 1.0 if kline_close > 0 else np.nan,
            }
        )
    append_frame_csv(output_dir / "paper_mark_prices.csv", pd.DataFrame(rows))


def update_forward_state(
    state: dict[str, Any],
    state_path: Path,
    panel: pd.DataFrame,
    residual_wide: pd.DataFrame,
    latest_ts: pd.Timestamp,
    signal_ts: pd.Timestamp,
    pf_cfg: Any,
    risk_template: Any,
    committee: Any,
    spec: Any,
    p16_row: pd.Series,
    formula_text: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    kline_prices = prices_at(panel, latest_ts)
    symbols = sorted(panel["symbol"].unique())
    live_prices = fetch_live_mark_prices(symbols, kline_prices)
    old_weights = pd.Series({sym: float(w) for sym, w in state.get("weights", {}).items()}, dtype=float)
    old_weights = old_weights.reindex(symbols, fill_value=0.0)
    funding_summary = apply_funding(state, old_weights, latest_ts, output_dir, args)
    entry_equity = float(state.get("entry_equity", state.get("initial_capital", 10000.0)))
    unrealized_return = mark_unrealized_return(old_weights, state.get("entry_prices", {}), live_prices)
    mark_equity = entry_equity * (1.0 + unrealized_return)
    current_weights = mark_current_weights(old_weights, state.get("entry_prices", {}), live_prices)
    current_weights = current_weights.reindex(symbols, fill_value=0.0)

    should_rebalance = state.get("last_rebalance_signal_timestamp") is None and state.get("last_rebalance_timestamp") is None
    if not should_rebalance:
        last_raw = state.get("last_rebalance_signal_timestamp") or state.get("last_rebalance_timestamp")
        last = pd.Timestamp(last_raw).tz_convert("UTC")
        should_rebalance = signal_ts >= last + pd.Timedelta(hours=int(pf_cfg.rebalance_hours))

    rebalance_row: dict[str, Any] | None = None
    target = None
    if should_rebalance:
        target = build_current_target(
            panel=panel,
            residual_wide=residual_wide,
            timestamp=signal_ts,
            prev_weights=old_weights,
            current_state=decode_current_state(state.get("current_state", {})),
            pf_cfg=pf_cfg,
            risk_template=risk_template,
            committee=committee,
            spec=spec,
            p16_row=p16_row,
        )
        new_weights = target["weights"].reindex(old_weights.index, fill_value=0.0)
        turnover = float((new_weights - current_weights).abs().sum())
        model_turnover = float((new_weights - old_weights).abs().sum())
        execution_summary, execution_frame = estimate_execution_costs(current_weights, new_weights, mark_equity, latest_ts, args)
        append_frame_csv(output_dir / "paper_execution_estimates.csv", execution_frame)
        cost_return = float(execution_summary["cost_return"])
        equity_after = mark_equity * (1.0 - cost_return)
        state["entry_equity"] = float(equity_after)
        state["weights"] = {sym: float(w) for sym, w in new_weights[new_weights.abs() > 1e-12].items()}
        state["entry_prices"] = {sym: float(live_prices.get(sym, kline_prices[sym])) for sym in state["weights"]}
        state["current_state"] = encode_current_state(target["next_state"])
        state["execution_metadata"] = target["execution_metadata"]
        state["last_rebalance_timestamp"] = str(latest_ts)
        state["last_rebalance_signal_timestamp"] = str(signal_ts)
        state["last_funding_timestamp"] = str(latest_ts)
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        rebalance_row = {
            "timestamp": latest_ts,
            "signal_timestamp": signal_ts,
            "execution_timestamp": latest_ts,
            "equity_before_cost": mark_equity,
            "equity_after_cost": equity_after,
            "unrealized_return_before_rebalance": unrealized_return,
            "turnover": turnover,
            "model_turnover": model_turnover,
            "cost_return": cost_return,
            "cost_model": execution_summary["cost_model"],
            "fee_return": execution_summary["fee_return"],
            "slippage_return": execution_summary["slippage_return"],
            "weighted_slippage_bps": execution_summary["weighted_slippage_bps"],
            "weighted_spread_bps": execution_summary["weighted_spread_bps"],
            "fallback_orderbook_count": execution_summary["fallback_orderbook_count"],
            "execution_orders": execution_summary["orders"],
            "long_symbols": ",".join(new_weights[new_weights > 0].index.tolist()),
            "short_symbols": ",".join(new_weights[new_weights < 0].index.tolist()),
            "formula": formula_text,
        }
        append_rebalance(output_dir / "paper_rebalances.csv", rebalance_row)
        mark_equity = equity_after
        unrealized_return = 0.0

    active_weights = pd.Series({sym: float(w) for sym, w in state.get("weights", {}).items()}, dtype=float)
    active_weights = active_weights.reindex(symbols, fill_value=0.0)
    latest_unrealized = mark_unrealized_return(active_weights, state.get("entry_prices", {}), live_prices)
    current_equity = float(state["entry_equity"]) * (1.0 + latest_unrealized)
    append_mark_snapshot(output_dir, latest_ts, kline_prices, live_prices, active_weights)
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return {
        "state": state,
        "latest_timestamp": str(latest_ts),
        "signal_timestamp": str(signal_ts),
        "current_equity": current_equity,
        "total_return": current_equity / float(state["initial_capital"]) - 1.0,
        "unrealized_return": latest_unrealized,
        "active_weights": active_weights[active_weights.abs() > 1e-12],
        "last_rebalance": rebalance_row,
        "funding": funding_summary,
        "price_source": "Binance USD-M markPrice; latest 1h kline close fallback",
        "execution_metadata": state.get("execution_metadata", {}),
        "target_debug": target or {},
    }


def build_current_target(
    panel: pd.DataFrame,
    residual_wide: pd.DataFrame,
    timestamp: pd.Timestamp,
    prev_weights: pd.Series,
    current_state: dict[str, tuple[int, int]],
    pf_cfg: Any,
    risk_template: Any,
    committee: Any,
    spec: Any,
    p16_row: pd.Series,
) -> dict[str, Any]:
    symbols = sorted(panel["symbol"].unique())
    rows = panel[panel["timestamp"] == timestamp].copy()
    rows = rows[np.isfinite(rows[p3d.SCORE_COL]) & np.isfinite(rows[p3d.P4_SCORE_COL])]
    if len(rows) < pf_cfg.long_k + pf_cfg.short_k:
        raise RuntimeError(f"Not enough valid symbols at {timestamp}: {len(rows)}")
    rows["score_rank"] = rows[p3d.SCORE_COL].rank(pct=True, method="average")
    rows["p4_rank"] = rows[p3d.P4_SCORE_COL].rank(pct=True, method="average")
    rows["ema_rank"] = rows["score_ema_025"].rank(pct=True, method="average")
    rows["base_rank"] = rows["score_baseline"].rank(pct=True, method="average")
    rows = rows.sort_values(p3d.SCORE_COL, ascending=False)
    rows_by_symbol = rows.set_index("symbol")

    target_sides = p3d.select_p4_gated_target_sides(rows, current_state, pf_cfg, p3d.SCORE_COL)
    target_sides = p15c.apply_short_guard(target_sides, rows_by_symbol, committee)
    if not any(side > 0 for side in target_sides.values()) or not any(side < 0 for side in target_sides.values()):
        raise RuntimeError("Target construction produced no balanced long/short book.")

    selected = rows_by_symbol.reindex(list(target_sides)).reset_index()
    raw_weights = p15c.side_fraction_weights(target_sides, pf_cfg.gross_exposure, committee.long_fraction)
    baseline_weights, baseline_info = p3d.build_weights(raw_weights, selected, rows_by_symbol, pf_cfg, risk_template)
    baseline_weights = baseline_weights[baseline_weights.abs() > 1e-12]
    if baseline_weights.abs().sum() <= 0:
        raise RuntimeError("Optimizer produced zero baseline weights.")

    baseline_full = pd.Series(0.0, index=symbols)
    baseline_full.loc[baseline_weights.index] = baseline_weights
    baseline_turnover = float((baseline_full - prev_weights.reindex(symbols, fill_value=0.0)).abs().sum())

    model_cache: dict[tuple[pd.Timestamp, int, int], Any] = {}
    model = p15c.p15.get_risk_model(model_cache, residual_wide, timestamp, int(p16_row["lookback_hours"]), int(p16_row["cluster_k"]))
    baseline_latent = p15c.p15.latent_metrics(baseline_weights, model)
    if spec is None:
        weights = baseline_weights
    else:
        weights, _ = p15c.apply_extreme_cluster_cap(
            baseline_weights=baseline_weights,
            raw_weights=raw_weights,
            rows_by_symbol=rows_by_symbol,
            model=model,
            cfg=pf_cfg,
            risk_cfg=risk_template,
            spec=spec,
            baseline_latent=baseline_latent,
            prev_weights=prev_weights.reindex(symbols, fill_value=0.0),
            all_symbols=symbols,
            baseline_turnover=baseline_turnover,
        )
    weights = weights[weights.abs() > 1e-12]
    next_state = p3d.update_state(current_state, target_sides, True)
    exposure_cols = list(pf_cfg.neutralize_exposures)
    exposure_loadings: dict[str, dict[str, float]] = {}
    for symbol, row in rows_by_symbol.iterrows():
        exposure_loadings[str(symbol)] = {
            col: 1.0 if col == "intercept" else float(row.get(col, 0.0) or 0.0)
            for col in exposure_cols
        }
    execution_metadata = {
        "signal_timestamp": str(timestamp),
        "neutralize_exposures": exposure_cols,
        "exposure_loadings": exposure_loadings,
    }
    return {
        "weights": weights,
        "target_sides": target_sides,
        "next_state": next_state,
        "optimizer": baseline_info,
        "execution_metadata": execution_metadata,
    }


def prices_at(panel: pd.DataFrame, timestamp: pd.Timestamp) -> dict[str, float]:
    rows = panel.loc[panel["timestamp"] == timestamp, ["symbol", "close"]]
    return {str(row.symbol): float(row.close) for row in rows.itertuples(index=False)}


def mark_unrealized_return(weights: pd.Series, entry_prices: dict[str, Any], latest_prices: dict[str, float]) -> float:
    total = 0.0
    for symbol, weight in weights.items():
        if abs(float(weight)) <= 1e-12:
            continue
        if symbol not in entry_prices or symbol not in latest_prices:
            continue
        entry = float(entry_prices[symbol])
        latest = float(latest_prices[symbol])
        if entry > 0 and np.isfinite(entry) and np.isfinite(latest):
            total += float(weight) * (latest / entry - 1.0)
    return float(total)


def mark_current_weights(weights: pd.Series, entry_prices: dict[str, Any], latest_prices: dict[str, float]) -> pd.Series:
    weights = weights.astype(float).copy()
    unrealized = mark_unrealized_return(weights, entry_prices, latest_prices)
    denominator = 1.0 + unrealized
    if denominator <= 1e-12 or not np.isfinite(denominator):
        return weights
    current = pd.Series(0.0, index=weights.index, dtype=float)
    for symbol, weight in weights.items():
        if abs(float(weight)) <= 1e-12:
            continue
        if symbol not in entry_prices or symbol not in latest_prices:
            current.loc[symbol] = float(weight)
            continue
        entry = float(entry_prices[symbol])
        latest = float(latest_prices[symbol])
        if entry > 0 and np.isfinite(entry) and np.isfinite(latest):
            current.loc[symbol] = float(weight) * latest / entry / denominator
        else:
            current.loc[symbol] = float(weight)
    return current


def encode_current_state(state: dict[str, tuple[int, int]]) -> dict[str, list[int]]:
    return {symbol: [int(side), int(age)] for symbol, (side, age) in state.items()}


def decode_current_state(raw: dict[str, Any]) -> dict[str, tuple[int, int]]:
    out = {}
    for symbol, value in raw.items():
        if isinstance(value, (list, tuple)) and len(value) == 2:
            out[str(symbol)] = (int(value[0]), int(value[1]))
    return out


def append_rebalance(path: Path, row: dict[str, Any]) -> None:
    frame = pd.DataFrame([row])
    if path.exists() and path.stat().st_size > 0:
        old = pd.read_csv(path, parse_dates=["timestamp"])
        frame = pd.concat([old, frame], ignore_index=True)
    frame.to_csv(path, index=False)


def write_outputs(
    output_dir: Path,
    panel: pd.DataFrame,
    rolling: dict[str, Any],
    forward: dict[str, Any] | None,
    p9_row: pd.Series,
    p16_row: pd.Series,
    formula_text: str,
    latest_ts: pd.Timestamp,
) -> None:
    equity = rolling["equity"]
    positions = rolling["positions"]
    if isinstance(equity, pd.DataFrame):
        equity.to_csv(output_dir / "paper_equity.csv", index=False)
        plot_equity(output_dir / "paper_equity.png", equity)
    if isinstance(positions, pd.DataFrame):
        positions.to_csv(output_dir / "paper_positions.csv", index=False)

    summary = {
        "updated_utc": str(pd.Timestamp.now(tz="UTC")),
        "latest_data_timestamp": str(latest_ts),
        "signal_timestamp": str(forward.get("signal_timestamp")) if isinstance(forward, dict) else None,
        "strategy": "P16-CS-72H",
        "rolling_metrics": to_jsonable(rolling.get("metrics", {})),
        "rolling_window": {"start": rolling.get("paper_start"), "end": rolling.get("paper_end")},
        "forward": summarize_forward_json(forward),
        "latest_p9_selection": to_jsonable(dict(p9_row)),
        "latest_p16_selection": to_jsonable(dict(p16_row)),
        "formula_text": formula_text,
        "panel_rows": int(len(panel)),
        "panel_symbols": int(panel["symbol"].nunique()),
    }
    (output_dir / "paper_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def summarize_forward_json(forward: dict[str, Any] | None) -> Any:
    if forward is None:
        return None
    out = {key: value for key, value in forward.items() if key not in {"active_weights", "target_debug"}}
    out["active_weights"] = {sym: float(w) for sym, w in forward["active_weights"].items()}
    return to_jsonable(out)


def plot_equity(path: Path, equity: pd.DataFrame) -> None:
    if equity.empty or "period_return" not in equity:
        return
    data = equity.sort_values("timestamp")
    curve = (1.0 + data["period_return"].astype(float)).cumprod() - 1.0
    gross = (1.0 + data["gross_period_return"].astype(float)).cumprod() - 1.0
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data["timestamp"], curve, label="net")
    ax.plot(data["timestamp"], gross, label="gross", alpha=0.7)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_title("P16-CS-72H Rolling Paper Equity")
    ax.set_ylabel("Cumulative return")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_status(summary: dict[str, Any], output_dir: Path, args: argparse.Namespace) -> str:
    rolling = summary["rolling"]
    metrics = rolling.get("metrics", {}) or {}
    positions = rolling.get("positions")
    periods = int(metrics.get("periods", 0) or 0)
    total_return = float(metrics.get("total_return", 0.0) or 0.0)
    initial = float(args.initial_capital)
    lines = [
        "P16-CS-72H Live-Cost Paper Trading Status",
        "=" * 52,
        f"Updated UTC: {pd.Timestamp.now(tz='UTC')}",
        f"Mode: {summary['mode']}",
        f"Latest data timestamp: {summary['latest_ts']}",
        f"Signal timestamp: {summary.get('signal_ts')}",
    ]
    if summary.get("forward"):
        fwd = summary["forward"]
        lines.extend(
            [
                "",
                "Forward Paper Account",
                f"  Current equity: {float(fwd['current_equity']):.2f}",
                f"  Total return: {float(fwd['total_return']):.2%}",
                f"  Unrealized return since entry: {float(fwd['unrealized_return']):.2%}",
                f"  Last rebalance: {fwd['state'].get('last_rebalance_timestamp')}",
                f"  Last rebalance signal: {fwd['state'].get('last_rebalance_signal_timestamp')}",
                f"  Active positions: {len(fwd['active_weights'])}",
                f"  Price source: {fwd.get('price_source')}",
            ]
        )
        funding = fwd.get("funding") or {}
        lines.extend(
            [
                f"  Funding return since last check: {float(funding.get('funding_return', 0.0) or 0.0):+.4%}",
                f"  Funding events checked: {int(funding.get('funding_events', 0) or 0)}",
            ]
        )
        if fwd.get("last_rebalance"):
            reb = fwd["last_rebalance"]
            lines.extend(
                [
                    "",
                    "Latest Rebalance Execution Estimate",
                    f"  Cost model: {reb.get('cost_model')}",
                    f"  Turnover: {float(reb.get('turnover', 0.0) or 0.0):.4f}",
                    f"  Model turnover: {float(reb.get('model_turnover', 0.0) or 0.0):.4f}",
                    f"  Total cost: {float(reb.get('cost_return', 0.0) or 0.0):.4%}",
                    f"  Fee cost: {float(reb.get('fee_return', 0.0) or 0.0):.4%}",
                    f"  Slippage cost: {float(reb.get('slippage_return', 0.0) or 0.0):.4%}",
                    f"  Weighted spread: {float(reb.get('weighted_spread_bps', 0.0) or 0.0):.3f} bps",
                    f"  Weighted slippage: {float(reb.get('weighted_slippage_bps', 0.0) or 0.0):.3f} bps",
                    f"  Orderbook fallbacks: {int(reb.get('fallback_orderbook_count', 0) or 0)}",
                ]
            )
        if len(fwd["active_weights"]):
            lines.extend(["", "Active Weights"])
            for symbol, weight in fwd["active_weights"].sort_values(ascending=False).items():
                lines.append(f"  {symbol}: {float(weight):+.4f}")

        lines.extend(
            [
                "",
                "Rolling Window Reference Only",
                f"  Window: {rolling.get('paper_start')} -> {rolling.get('paper_end')}",
                f"  Equity: {initial * (1.0 + total_return):.2f}",
                f"  Return: {total_return:.2%}",
                f"  Sharpe: {float(metrics.get('sharpe', 0.0) or 0.0):.3f}",
                f"  Max drawdown: {float(metrics.get('max_drawdown', 0.0) or 0.0):.2%}",
                f"  Effective 72h periods: {periods}",
                f"  Mean turnover: {float(metrics.get('mean_turnover', 0.0) or 0.0):.3f}",
            ]
        )
    elif isinstance(positions, pd.DataFrame) and not positions.empty:
        lines.extend(
            [
                "",
                "Rolling Window Performance",
                f"  Window: {rolling.get('paper_start')} -> {rolling.get('paper_end')}",
                f"  Equity: {initial * (1.0 + total_return):.2f}",
                f"  Return: {total_return:.2%}",
                f"  Sharpe: {float(metrics.get('sharpe', 0.0) or 0.0):.3f}",
                f"  Max drawdown: {float(metrics.get('max_drawdown', 0.0) or 0.0):.2%}",
                f"  Effective 72h periods: {periods}",
                f"  Mean turnover: {float(metrics.get('mean_turnover', 0.0) or 0.0):.3f}",
            ]
        )
        latest_pos_ts = positions["timestamp"].max()
        latest_positions = positions[positions["timestamp"] == latest_pos_ts].sort_values("weight", ascending=False)
        lines.extend(["", f"Latest Rolling Positions at {latest_pos_ts}"])
        for row in latest_positions.itertuples(index=False):
            lines.append(f"  {row.symbol}: {float(row.weight):+.4f}")

    lines.extend(
        [
            "",
            "Locked Live Selection",
            f"  P9/P3d: {summary['p9_selected'].get('p3d_formula_name')} | {summary['formula_text']}",
            f"  P16 allocator: {summary['p16_selected'].get('selected_extreme_spec')}",
            "",
            "Output Files",
            f"  {output_dir / 'STATUS.txt'}",
            f"  {output_dir / 'paper_summary.json'}",
            f"  {output_dir / 'paper_equity.csv'}",
            f"  {output_dir / 'paper_positions.csv'}",
            f"  {output_dir / 'paper_rebalances.csv'}",
            f"  {output_dir / 'paper_execution_estimates.csv'}",
            f"  {output_dir / 'paper_funding.csv'}",
            f"  {output_dir / 'paper_mark_prices.csv'}",
            f"  {output_dir / 'paper_equity.png'}",
        ]
    )
    return "\n".join(lines)


def build_error_status(exc: Exception) -> str:
    return "\n".join(
        [
            "P16-CS-72H Live-Cost Paper Trading Status",
            "=" * 52,
            f"Updated UTC: {pd.Timestamp.now(tz='UTC')}",
            "Status: ERROR",
            f"Error: {type(exc).__name__}: {exc}",
            "The full traceback was written to ERROR.log.",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
