from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PortfolioConfig:
    rebalance_hours: int
    long_k: int
    short_k: int
    gross_exposure: float
    one_way_fee_bps: float
    one_way_slippage_bps: float
    neutralize_exposures: list[str]
    primary_horizon: int
    execution_delay_bars: int
    use_hysteresis: bool = False
    long_entry_rank: float = 0.80
    long_exit_rank: float = 0.55
    short_entry_rank: float = 0.20
    short_exit_rank: float = 0.45
    min_hold_rebalances: int = 0
    cost_gate_buffer: float = 0.0
    score_to_return_coef: float = 0.0


def run_long_short_backtest(
    panel: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cfg: PortfolioConfig,
    score_col: str = "score_baseline",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    label_col = f"label_resid_ret_h{cfg.primary_horizon}"
    ret_col = f"fwd_simple_ret_h{cfg.primary_horizon}"
    symbols = sorted(panel["symbol"].unique())

    mask = (panel["timestamp"] >= start) & (panel["timestamp"] < end)
    timestamps = np.array(sorted(panel.loc[mask, "timestamp"].unique()))
    timestamps = timestamps[:: cfg.rebalance_hours]

    prev_weights = pd.Series(0.0, index=symbols)
    current_state: dict[str, tuple[int, int]] = {}
    equity = 1.0
    rows_out: list[dict[str, float | str | pd.Timestamp]] = []
    pos_out: list[dict[str, float | str | pd.Timestamp]] = []

    for ts in timestamps:
        rows = panel[panel["timestamp"] == ts].copy()
        rows = rows[np.isfinite(rows[score_col]) & np.isfinite(rows[ret_col]) & np.isfinite(rows[label_col])]
        if len(rows) < cfg.long_k + cfg.short_k:
            continue

        rows["score_rank"] = rows[score_col].rank(pct=True, method="average")
        rows = rows.sort_values(score_col, ascending=False)
        rows_by_symbol = rows.set_index("symbol")
        target_sides = select_target_sides(rows, current_state, cfg, score_col)
        if not any(side > 0 for side in target_sides.values()) or not any(side < 0 for side in target_sides.values()):
            continue

        selected_symbols = list(target_sides)
        selected = rows_by_symbol.reindex(selected_symbols).reset_index()
        raw_weights = sides_to_weights(target_sides, cfg.gross_exposure)
        weights = neutralize_weights(raw_weights, selected, cfg.neutralize_exposures, cfg.gross_exposure)
        if weights.abs().sum() <= 0:
            continue

        target_full_weights = pd.Series(0.0, index=symbols)
        target_full_weights.loc[weights.index] = weights
        full_weights = apply_cost_gate(prev_weights, target_full_weights, rows_by_symbol, cfg, score_col)
        target_applied = not np.allclose(full_weights.to_numpy(), prev_weights.to_numpy(), atol=1e-12, rtol=1e-12)
        turnover = float((full_weights - prev_weights).abs().sum())
        cost = turnover * (cfg.one_way_fee_bps + cfg.one_way_slippage_bps) / 10_000.0

        active_weights = full_weights[full_weights.abs() > 1e-12]
        active_rows = rows_by_symbol.reindex(active_weights.index)
        gross_period_ret = float((active_weights * active_rows[ret_col]).sum())
        period_ret = gross_period_ret - cost
        equity *= 1.0 + period_ret

        rank_ic = spearman_ic(rows[score_col], rows[label_col])
        long_labels = active_rows.loc[active_weights[active_weights > 0].index, label_col]
        short_labels = active_rows.loc[active_weights[active_weights < 0].index, label_col]
        top_bottom = float(long_labels.mean() - short_labels.mean())
        exposures = portfolio_exposures(active_weights, active_rows, cfg.neutralize_exposures)

        exit_ts = pd.Timestamp(ts) + pd.Timedelta(hours=cfg.execution_delay_bars + cfg.primary_horizon)
        rows_out.append(
            {
                "timestamp": pd.Timestamp(ts),
                "exit_timestamp": exit_ts,
                "period_return": period_ret,
                "gross_period_return": gross_period_ret,
                "cost_return": cost,
                "equity": equity,
                "turnover": turnover,
                "gross_exposure": float(weights.abs().sum()),
                "rank_ic": rank_ic,
                "top_bottom_label_spread": top_bottom,
                "abs_beta_btc": abs(exposures.get("beta_btc", 0.0)),
                "abs_beta_eth": abs(exposures.get("beta_eth", 0.0)),
                "abs_beta_market": abs(exposures.get("beta_market", 0.0)),
                "symbol_concentration": float((active_weights**2).sum()),
                "target_applied": float(target_applied),
            }
        )

        for symbol, weight in active_weights.items():
            pos_out.append(
                {
                    "timestamp": pd.Timestamp(ts),
                    "exit_timestamp": exit_ts,
                    "symbol": symbol,
                    "weight": float(weight),
                    "score": float(rows_by_symbol.loc[symbol, score_col]),
                    "fwd_simple_ret": float(rows_by_symbol.loc[symbol, ret_col]),
                    "label_resid_ret": float(rows_by_symbol.loc[symbol, label_col]),
                }
            )

        prev_weights = full_weights
        current_state = update_state(current_state, target_sides, target_applied)

    equity_curve = pd.DataFrame(rows_out)
    positions = pd.DataFrame(pos_out)
    metrics = summarize_equity_curve(equity_curve, cfg.rebalance_hours)
    return equity_curve, positions, metrics


def select_target_sides(
    rows: pd.DataFrame,
    current_state: dict[str, tuple[int, int]],
    cfg: PortfolioConfig,
    score_col: str,
) -> dict[str, int]:
    rows_by_symbol = rows.set_index("symbol")
    valid_symbols = set(rows_by_symbol.index)
    forced_longs = [
        symbol
        for symbol, (side, age) in current_state.items()
        if side > 0 and age < cfg.min_hold_rebalances and symbol in valid_symbols
    ]
    forced_shorts = [
        symbol
        for symbol, (side, age) in current_state.items()
        if side < 0 and age < cfg.min_hold_rebalances and symbol in valid_symbols
    ]

    longs = _dedupe_keep_order(forced_longs)
    shorts = _dedupe_keep_order(forced_shorts)

    if cfg.use_hysteresis:
        held_longs = [
            symbol
            for symbol, (side, _) in current_state.items()
            if side > 0
            and symbol in valid_symbols
            and rows_by_symbol.loc[symbol, "score_rank"] >= cfg.long_exit_rank
            and symbol not in longs
        ]
        held_shorts = [
            symbol
            for symbol, (side, _) in current_state.items()
            if side < 0
            and symbol in valid_symbols
            and rows_by_symbol.loc[symbol, "score_rank"] <= cfg.short_exit_rank
            and symbol not in shorts
        ]
        longs.extend(_sort_symbols_by_score(held_longs, rows_by_symbol, score_col, ascending=False))
        shorts.extend(_sort_symbols_by_score(held_shorts, rows_by_symbol, score_col, ascending=True))

        long_pool = rows[
            (rows["score_rank"] >= cfg.long_entry_rank)
            & ~rows["symbol"].isin(longs)
            & ~rows["symbol"].isin(shorts)
        ].sort_values(score_col, ascending=False)
        short_pool = rows[
            (rows["score_rank"] <= cfg.short_entry_rank)
            & ~rows["symbol"].isin(longs)
            & ~rows["symbol"].isin(shorts)
        ].sort_values(score_col, ascending=True)
    else:
        long_pool = rows[~rows["symbol"].isin(longs) & ~rows["symbol"].isin(shorts)].sort_values(score_col, ascending=False)
        short_pool = rows[~rows["symbol"].isin(longs) & ~rows["symbol"].isin(shorts)].sort_values(score_col, ascending=True)

    for symbol in long_pool["symbol"]:
        if len(longs) >= cfg.long_k:
            break
        if symbol not in shorts:
            longs.append(symbol)

    for symbol in short_pool["symbol"]:
        if len(shorts) >= cfg.short_k:
            break
        if symbol not in longs:
            shorts.append(symbol)

    longs = _sort_symbols_by_score(longs, rows_by_symbol, score_col, ascending=False)[: cfg.long_k]
    shorts = _sort_symbols_by_score(shorts, rows_by_symbol, score_col, ascending=True)[: cfg.short_k]

    sides = {symbol: 1 for symbol in longs}
    sides.update({symbol: -1 for symbol in shorts if symbol not in sides})
    return sides


def sides_to_weights(target_sides: dict[str, int], gross_exposure: float) -> pd.Series:
    longs = [symbol for symbol, side in target_sides.items() if side > 0]
    shorts = [symbol for symbol, side in target_sides.items() if side < 0]
    weights = pd.Series(0.0, index=list(target_sides), dtype=float)
    if longs:
        weights.loc[longs] = gross_exposure * 0.5 / len(longs)
    if shorts:
        weights.loc[shorts] = -gross_exposure * 0.5 / len(shorts)
    return weights


def apply_cost_gate(
    prev_weights: pd.Series,
    target_weights: pd.Series,
    rows_by_symbol: pd.DataFrame,
    cfg: PortfolioConfig,
    score_col: str,
) -> pd.Series:
    if cfg.cost_gate_buffer <= 0 or prev_weights.abs().sum() <= 0:
        return target_weights

    turnover = float((target_weights - prev_weights).abs().sum())
    if turnover <= 1e-12:
        return target_weights

    cost = turnover * (cfg.one_way_fee_bps + cfg.one_way_slippage_bps) / 10_000.0
    expected = rows_by_symbol[score_col].reindex(prev_weights.index).fillna(0.0) * cfg.score_to_return_coef
    current_alpha = float((prev_weights * expected).sum())
    target_alpha = float((target_weights * expected).sum())
    if target_alpha - current_alpha <= cfg.cost_gate_buffer * cost:
        return prev_weights.copy()
    return target_weights


def update_state(
    current_state: dict[str, tuple[int, int]],
    target_sides: dict[str, int],
    target_applied: bool,
) -> dict[str, tuple[int, int]]:
    if not target_applied:
        return {symbol: (side, age + 1) for symbol, (side, age) in current_state.items()}

    next_state: dict[str, tuple[int, int]] = {}
    for symbol, side in target_sides.items():
        old = current_state.get(symbol)
        if old is not None and old[0] == side:
            next_state[symbol] = (side, old[1] + 1)
        else:
            next_state[symbol] = (side, 1)
    return next_state


def _sort_symbols_by_score(
    symbols: list[str],
    rows_by_symbol: pd.DataFrame,
    score_col: str,
    ascending: bool,
) -> list[str]:
    if not symbols:
        return []
    values = rows_by_symbol.loc[symbols, score_col].sort_values(ascending=ascending)
    return list(values.index)


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for value in values:
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out


def neutralize_weights(
    raw_weights: pd.Series,
    selected: pd.DataFrame,
    exposure_cols: list[str],
    gross_exposure: float,
) -> pd.Series:
    weights = raw_weights.astype(float).copy()
    if not exposure_cols:
        return _scale_gross(weights, gross_exposure)

    selected = selected.set_index("symbol").reindex(weights.index)
    exposure_arrays = []
    for col in exposure_cols:
        if col == "intercept":
            exposure_arrays.append(np.ones(len(weights)))
        elif col in selected.columns:
            exposure_arrays.append(selected[col].fillna(0.0).to_numpy(dtype=float))

    if not exposure_arrays:
        return _scale_gross(weights, gross_exposure)

    b = np.column_stack(exposure_arrays)
    w = weights.to_numpy(dtype=float)
    correction = b @ (np.linalg.pinv(b.T @ b, rcond=1e-10) @ (b.T @ w))
    adjusted = pd.Series(w - correction, index=weights.index)
    return _scale_gross(adjusted, gross_exposure)


def constrained_neutralize_weights(
    raw_weights: pd.Series,
    selected: pd.DataFrame,
    exposure_cols: list[str],
    gross_exposure: float,
    max_abs_weight: float,
    target_gross: float | None = None,
    min_gross: float = 0.05,
    return_info: bool = False,
) -> pd.Series | tuple[pd.Series, dict[str, float]]:
    from scipy.optimize import minimize

    weights = raw_weights.astype(float).copy()
    if weights.empty or max_abs_weight <= 0:
        out = weights * 0.0
        info = _optimizer_info(False, 0.0, 0.0, np.nan, 0)
        return (out, info) if return_info else out

    signs = np.sign(weights.to_numpy(dtype=float))
    if not np.any(signs > 0) or not np.any(signs < 0):
        out = weights * 0.0
        info = _optimizer_info(False, 0.0, 0.0, np.nan, 0)
        return (out, info) if return_info else out

    selected = selected.set_index("symbol").reindex(weights.index)
    exposure_arrays = []
    for col in exposure_cols:
        if col == "intercept":
            exposure_arrays.append(np.ones(len(weights)))
        elif col in selected.columns:
            exposure_arrays.append(selected[col].fillna(0.0).to_numpy(dtype=float))
    exposure_matrix = np.vstack(exposure_arrays) if exposure_arrays else np.empty((0, len(weights)))

    bounds = [(0.0, max_abs_weight) if sign > 0 else (-max_abs_weight, 0.0) for sign in signs]
    long_capacity = max_abs_weight * float(np.sum(signs > 0))
    short_capacity = max_abs_weight * float(np.sum(signs < 0))
    capacity_gross = 2.0 * min(long_capacity, short_capacity)
    requested_gross = gross_exposure if target_gross is None else max(0.0, min(float(target_gross), gross_exposure))
    start_gross = min(requested_gross, capacity_gross)
    if start_gross < min_gross:
        out = weights * 0.0
        info = _optimizer_info(False, start_gross, 0.0, np.nan, 0)
        return (out, info) if return_info else out

    reference = weights.to_numpy(dtype=float)
    reference_gross = float(np.abs(reference).sum())
    if reference_gross > 0:
        reference = reference * (start_gross / reference_gross)

    gross_grid = [start_gross]
    gross_grid.extend(start_gross * value for value in (0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20))
    gross_grid = [value for value in gross_grid if value >= min_gross]

    best_x: np.ndarray | None = None
    best_error = np.inf
    attempts = 0
    for gross_target in gross_grid:
        attempts += 1
        x0 = _side_balanced_start(signs, gross_target)
        constraints = [
            {"type": "eq", "fun": lambda x, a=exposure_matrix: a @ x},
            {"type": "eq", "fun": lambda x, s=signs, g=gross_target: float(s @ x - g)},
        ]
        result = minimize(
            lambda x, ref=reference: float(np.sum((x - ref) ** 2)),
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-12, "disp": False},
        )
        candidate = result.x if result.x is not None else x0
        exposure_error = _max_constraint_error(exposure_matrix, candidate)
        gross_error = abs(float(signs @ candidate - gross_target))
        error = max(exposure_error, gross_error)
        if error < best_error:
            best_x = candidate
            best_error = error
        if error <= 1e-7 and _bounds_ok(candidate, bounds):
            out = pd.Series(candidate, index=weights.index)
            info = _optimizer_info(True, gross_target, float(out.abs().sum()), exposure_error, attempts)
            return (out, info) if return_info else out

    if best_x is not None and best_error <= 1e-5 and _bounds_ok(best_x, bounds):
        out = pd.Series(best_x, index=weights.index)
        info = _optimizer_info(True, float(np.abs(best_x).sum()), float(out.abs().sum()), best_error, attempts)
        return (out, info) if return_info else out

    out = weights * 0.0
    info = _optimizer_info(False, start_gross, 0.0, best_error if np.isfinite(best_error) else np.nan, attempts)
    return (out, info) if return_info else out


def constrained_project_weights(
    target_weights: pd.Series,
    universe: pd.DataFrame,
    exposure_cols: list[str],
    max_abs_weight: float,
    gross_limit: float,
    fixed_signs: dict[str, int] | None = None,
    hedge_penalty: float = 25.0,
    return_info: bool = False,
) -> pd.Series | tuple[pd.Series, dict[str, float]]:
    from scipy.optimize import minimize

    fixed_signs = fixed_signs or {}
    universe = universe.set_index("symbol")
    symbols = list(universe.index)
    target = target_weights.reindex(symbols).fillna(0.0).to_numpy(dtype=float)
    if len(symbols) == 0 or max_abs_weight <= 0 or gross_limit <= 0:
        out = pd.Series(0.0, index=symbols)
        info = _optimizer_info(False, gross_limit, 0.0, np.nan, 0)
        return (out, info) if return_info else out

    exposure_arrays = []
    for col in exposure_cols:
        if col == "intercept":
            exposure_arrays.append(np.ones(len(symbols)))
        elif col in universe.columns:
            exposure_arrays.append(universe[col].fillna(0.0).to_numpy(dtype=float))
    exposure_matrix = np.vstack(exposure_arrays) if exposure_arrays else np.empty((0, len(symbols)))

    bounds = []
    penalties = np.full(len(symbols), hedge_penalty, dtype=float)
    for index, symbol in enumerate(symbols):
        side = fixed_signs.get(symbol, 0)
        if side > 0:
            bounds.append((0.0, max_abs_weight))
            penalties[index] = 1.0
        elif side < 0:
            bounds.append((-max_abs_weight, 0.0))
            penalties[index] = 1.0
        else:
            bounds.append((-max_abs_weight, max_abs_weight))

    x0 = np.clip(target, -max_abs_weight, max_abs_weight)
    if np.abs(x0).sum() > gross_limit:
        x0 = x0 * (gross_limit / np.abs(x0).sum())

    constraints = [
        {"type": "eq", "fun": lambda x, a=exposure_matrix: a @ x},
        {"type": "ineq", "fun": lambda x, limit=gross_limit: float(limit - np.abs(x).sum())},
    ]
    result = minimize(
        lambda x, ref=target, weights=penalties: float(np.sum(weights * (x - ref) ** 2)),
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-12, "disp": False},
    )

    candidate = result.x if result.x is not None else x0
    exposure_error = _max_constraint_error(exposure_matrix, candidate)
    gross = float(np.abs(candidate).sum())
    constraints_ok = exposure_error <= 1e-7 and gross <= gross_limit + 1e-7 and _bounds_ok(candidate, bounds)
    out = pd.Series(candidate if constraints_ok else np.zeros(len(symbols)), index=symbols)
    info = _optimizer_info(bool(constraints_ok), gross_limit, float(out.abs().sum()), exposure_error, 1)
    return (out, info) if return_info else out


def _side_balanced_start(signs: np.ndarray, gross_target: float) -> np.ndarray:
    out = np.zeros(len(signs), dtype=float)
    long_count = float(np.sum(signs > 0))
    short_count = float(np.sum(signs < 0))
    if long_count > 0:
        out[signs > 0] = gross_target * 0.5 / long_count
    if short_count > 0:
        out[signs < 0] = -gross_target * 0.5 / short_count
    return out


def _max_constraint_error(exposure_matrix: np.ndarray, weights: np.ndarray) -> float:
    if exposure_matrix.size == 0:
        return 0.0
    return float(np.max(np.abs(exposure_matrix @ weights)))


def _bounds_ok(values: np.ndarray, bounds: list[tuple[float, float]]) -> bool:
    for value, (lower, upper) in zip(values, bounds):
        if value < lower - 1e-8 or value > upper + 1e-8:
            return False
    return True


def _optimizer_info(
    success: bool,
    target_gross: float,
    actual_gross: float,
    exposure_error: float,
    attempts: int,
) -> dict[str, float]:
    return {
        "optimizer_success": float(success),
        "optimizer_target_gross": float(target_gross),
        "optimizer_actual_gross": float(actual_gross),
        "optimizer_exposure_error": float(exposure_error),
        "optimizer_attempts": float(attempts),
    }


def _scale_gross(weights: pd.Series, gross_exposure: float) -> pd.Series:
    gross = float(weights.abs().sum())
    if gross <= 0 or not np.isfinite(gross):
        return weights * 0.0
    return weights * (gross_exposure / gross)


def portfolio_exposures(weights: pd.Series, selected: pd.DataFrame, exposure_cols: list[str]) -> dict[str, float]:
    exposures: dict[str, float] = {}
    for col in exposure_cols:
        if col == "intercept":
            exposures[col] = float(weights.sum())
        elif col in selected.columns:
            exposures[col] = float((weights * selected[col].fillna(0.0)).sum())
    return exposures


def spearman_ic(score: pd.Series, label: pd.Series) -> float:
    valid = np.isfinite(score) & np.isfinite(label)
    if valid.sum() < 4:
        return np.nan
    return float(score[valid].rank().corr(label[valid].rank()))


def summarize_equity_curve(equity_curve: pd.DataFrame, rebalance_hours: int) -> dict[str, float]:
    if equity_curve.empty:
        return {
            "periods": 0,
            "total_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "mean_rank_ic": np.nan,
            "mean_turnover": np.nan,
            "total_turnover": 0.0,
            "cost_drag": 0.0,
        }

    returns = equity_curve["period_return"].astype(float)
    gross_returns = equity_curve["gross_period_return"].astype(float)
    periods_per_year = 365.0 * 24.0 / rebalance_hours
    ret_std = returns.std(ddof=0)
    sharpe = 0.0 if ret_std == 0 or not np.isfinite(ret_std) else float(returns.mean() / ret_std * np.sqrt(periods_per_year))
    equity = equity_curve["equity"].astype(float)
    drawdown = equity / equity.cummax() - 1.0

    return {
        "periods": int(len(equity_curve)),
        "total_return": float(equity.iloc[-1] - 1.0),
        "gross_total_return_before_cost": float((1.0 + gross_returns).prod() - 1.0),
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "mean_rank_ic": float(equity_curve["rank_ic"].mean()),
        "mean_top_bottom_label_spread": float(equity_curve["top_bottom_label_spread"].mean()),
        "mean_period_return": float(returns.mean()),
        "mean_gross_period_return": float(gross_returns.mean()),
        "positive_period_rate": float((returns > 0).mean()),
        "positive_gross_period_rate": float((gross_returns > 0).mean()),
        "mean_turnover": float(equity_curve["turnover"].mean()),
        "total_turnover": float(equity_curve["turnover"].sum()),
        "cost_drag": float(equity_curve["cost_return"].sum()),
        "mean_period_cost": float(equity_curve["cost_return"].mean()),
        "target_apply_rate": float(equity_curve["target_applied"].mean()) if "target_applied" in equity_curve else 1.0,
        "mean_abs_beta_btc": float(equity_curve["abs_beta_btc"].mean()),
        "mean_abs_beta_eth": float(equity_curve["abs_beta_eth"].mean()),
        "mean_abs_beta_market": float(equity_curve["abs_beta_market"].mean()),
        "mean_symbol_concentration": float(equity_curve["symbol_concentration"].mean()),
    }
