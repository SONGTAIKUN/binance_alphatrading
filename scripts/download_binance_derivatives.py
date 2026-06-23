#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crypto_stat_arb.config import load_config


BASE_URL = "https://data.binance.vision/data/futures/um"
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "baseline_locked_v1"
DEFAULT_PANEL = EXPERIMENT_ROOT / "inputs" / "asset_residual_panel.csv"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "baseline.yaml"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "inputs" / "binance_derivatives"

MONTHLY_SERIES = {
    "fundingRate": "{base}/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{month}.zip",
    "indexPriceKlines": "{base}/monthly/indexPriceKlines/{symbol}/1h/{symbol}-1h-{month}.zip",
    "markPriceKlines": "{base}/monthly/markPriceKlines/{symbol}/1h/{symbol}-1h-{month}.zip",
    "premiumIndexKlines": "{base}/monthly/premiumIndexKlines/{symbol}/1h/{symbol}-1h-{month}.zip",
}
DAILY_SERIES = {
    "metrics": "{base}/daily/metrics/{symbol}/{symbol}-metrics-{day}.zip",
}


@dataclass(frozen=True)
class DownloadTask:
    symbol: str
    series: str
    period: str
    url: str
    output_path: Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--panel", default=str(DEFAULT_PANEL))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument(
        "--series",
        nargs="*",
        default=["fundingRate", "indexPriceKlines", "markPriceKlines", "premiumIndexKlines", "metrics"],
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    symbols = list(args.symbols or cfg["data"]["symbols"])
    start_dt, end_dt = infer_date_range(args.panel, args.start_date, args.end_date)
    selected_series = validate_series(args.series)

    print(
        f"Downloading Binance USD-M derivatives data: {len(symbols)} symbols, "
        f"{start_dt} to {end_dt}, series={','.join(selected_series)}",
        flush=True,
    )

    manifest_path = output_dir / "download_manifest.csv"
    all_rows: list[dict[str, object]] = []
    started = time.time()
    for idx, symbol in enumerate(symbols, start=1):
        symbol_tasks = build_tasks(symbol, start_dt, end_dt, selected_series, raw_dir)
        rows = run_symbol_tasks(symbol_tasks, workers=args.workers, timeout=args.timeout, retries=args.retries)
        all_rows.extend(rows)
        counts = pd.Series([row["status"] for row in rows]).value_counts().to_dict()
        elapsed = time.time() - started
        print(
            f"[{idx:02d}/{len(symbols):02d}] {symbol} done "
            f"downloaded={counts.get('downloaded', 0)} existing={counts.get('existing', 0)} "
            f"missing={counts.get('missing', 0)} failed={counts.get('failed', 0)} "
            f"elapsed={elapsed/60:.1f}m",
            flush=True,
        )

    write_manifest(manifest_path, all_rows)
    print(f"Manifest: {manifest_path}", flush=True)
    print(f"Raw data: {raw_dir}", flush=True)


def infer_date_range(panel_path: str, start_arg: str | None, end_arg: str | None) -> tuple[date, date]:
    if start_arg and end_arg:
        return pd.Timestamp(start_arg).date(), pd.Timestamp(end_arg).date()

    panel = pd.read_csv(panel_path, usecols=["timestamp"], parse_dates=["timestamp"])
    min_ts = pd.Timestamp(panel["timestamp"].min()).tz_convert("UTC")
    max_ts = pd.Timestamp(panel["timestamp"].max()).tz_convert("UTC")
    start_dt = pd.Timestamp(start_arg).date() if start_arg else min_ts.date()
    end_dt = pd.Timestamp(end_arg).date() if end_arg else max_ts.date()
    return start_dt, end_dt


def validate_series(series: Iterable[str]) -> list[str]:
    valid = set(MONTHLY_SERIES) | set(DAILY_SERIES)
    out = []
    for name in series:
        if name not in valid:
            raise ValueError(f"Unknown series {name!r}; valid={sorted(valid)}")
        out.append(name)
    return out


def build_tasks(symbol: str, start_dt: date, end_dt: date, series: list[str], raw_dir: Path) -> list[DownloadTask]:
    tasks: list[DownloadTask] = []
    months = pd.period_range(start=start_dt, end=end_dt, freq="M")
    days = pd.date_range(start=start_dt, end=end_dt, freq="D")

    for name in series:
        if name in MONTHLY_SERIES:
            for month in months:
                period = str(month)
                url = MONTHLY_SERIES[name].format(base=BASE_URL, symbol=symbol, month=period)
                out = raw_dir / "monthly" / name / symbol / ("1h" if name.endswith("Klines") else "") / f"{Path(url).name}"
                tasks.append(DownloadTask(symbol, name, period, url, out))
        elif name in DAILY_SERIES:
            for ts in days:
                period = ts.strftime("%Y-%m-%d")
                url = DAILY_SERIES[name].format(base=BASE_URL, symbol=symbol, day=period)
                out = raw_dir / "daily" / name / symbol / f"{Path(url).name}"
                tasks.append(DownloadTask(symbol, name, period, url, out))
    return tasks


def run_symbol_tasks(tasks: list[DownloadTask], workers: int, timeout: float, retries: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    workers = max(1, min(int(workers), len(tasks) if tasks else 1))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(download_one, task, timeout, retries) for task in tasks]
        for future in as_completed(futures):
            rows.append(future.result())
    return rows


def download_one(task: DownloadTask, timeout: float, retries: int) -> dict[str, object]:
    task.output_path.parent.mkdir(parents=True, exist_ok=True)
    if task.output_path.exists() and task.output_path.stat().st_size > 0:
        return row_for(task, "existing", task.output_path.stat().st_size, "")

    tmp = task.output_path.with_suffix(task.output_path.suffix + ".part")
    last_error = ""
    for attempt in range(retries + 1):
        try:
            req = Request(task.url, headers={"User-Agent": "crypto-stat-arb-research/1.0"})
            with urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            if not data:
                last_error = "empty response"
                continue
            tmp.write_bytes(data)
            tmp.replace(task.output_path)
            return row_for(task, "downloaded", len(data), "")
        except HTTPError as exc:
            if exc.code == 404:
                if tmp.exists():
                    tmp.unlink()
                return row_for(task, "missing", 0, "404")
            last_error = f"HTTP {exc.code}"
        except (TimeoutError, URLError, OSError) as exc:
            last_error = repr(exc)
        time.sleep(min(0.5 * (attempt + 1), 3.0))

    if tmp.exists():
        tmp.unlink()
    return row_for(task, "failed", 0, last_error)


def row_for(task: DownloadTask, status: str, size: int, error: str) -> dict[str, object]:
    return {
        "symbol": task.symbol,
        "series": task.series,
        "period": task.period,
        "status": status,
        "size": int(size),
        "path": str(task.output_path),
        "url": task.url,
        "error": error,
    }


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["symbol", "series", "period", "status", "size", "path", "url", "error"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
