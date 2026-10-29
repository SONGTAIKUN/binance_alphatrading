#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "experiments"
    / "baseline_locked_v1"
    / "inputs"
    / "cross_exchange"
    / "okx"
)
BASE_URL = "https://www.okx.com"
UTC = timezone.utc


@dataclass(frozen=True)
class ArrayDataset:
    name: str
    path: str
    params: dict[str, str]
    columns: tuple[str, ...]
    timestamp_column: str
    history_days: int | None = None
    page_limit: int = 300


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download one OKX perpetual symbol from public, unauthenticated endpoints."
    )
    parser.add_argument("--inst-id", default="BTC-USDT-SWAP")
    parser.add_argument("--index-id", default="BTC-USDT")
    parser.add_argument("--ccy", default="BTC")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default=None, help="UTC timestamp; default is the current UTC hour")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--sleep", type=float, default=0.12)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


class OkxClient:
    def __init__(self, sleep_seconds: float, timeout: float, retries: int) -> None:
        self.sleep_seconds = sleep_seconds
        self.timeout = timeout
        self.retries = retries
        self.request_count = 0

    def get(self, path: str, params: dict[str, Any]) -> list[Any]:
        query = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        url = f"{BASE_URL}{path}?{query}"
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                request = urllib.request.Request(
                    url,
                    headers={"User-Agent": "binance-alphatrading-research/1.0"},
                )
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                self.request_count += 1
                if str(payload.get("code")) != "0":
                    raise RuntimeError(
                        f"OKX error code={payload.get('code')} msg={payload.get('msg')} url={url}"
                    )
                time.sleep(self.sleep_seconds)
                return payload.get("data", [])
            except (OSError, ValueError, urllib.error.URLError) as exc:
                last_error = exc
                if attempt == self.retries:
                    break
                time.sleep(min(2 ** (attempt - 1), 8))
        raise RuntimeError(
            f"request failed after {self.retries} attempts: {url}; last_error={last_error}"
        ) from last_error


def as_utc(value: str | None, default: datetime | None = None) -> pd.Timestamp:
    if value is None:
        if default is None:
            raise ValueError("timestamp is required")
        return pd.Timestamp(default).tz_convert("UTC")
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp


def ms(timestamp: pd.Timestamp) -> int:
    return int(timestamp.timestamp() * 1000)


def load_partial(path: Path) -> list[Any]:
    if not path.exists():
        return []
    rows: list[Any] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_partial(path: Path, rows: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":"), ensure_ascii=True))
            handle.write("\n")


def finalize_array_dataset(
    rows: list[list[Any]], columns: tuple[str, ...], timestamp_column: str, output: Path
) -> pd.DataFrame:
    valid_rows = [row for row in rows if isinstance(row, list) and len(row) >= len(columns)]
    frame = pd.DataFrame([row[: len(columns)] for row in valid_rows], columns=columns)
    if frame.empty:
        frame.to_csv(output, index=False, compression="gzip")
        return frame
    frame[timestamp_column] = pd.to_datetime(
        pd.to_numeric(frame[timestamp_column], errors="coerce"), unit="ms", utc=True
    )
    frame = frame.dropna(subset=[timestamp_column]).drop_duplicates(timestamp_column)
    frame = frame.sort_values(timestamp_column).reset_index(drop=True)
    for column in frame.columns:
        if column == timestamp_column:
            continue
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.notna().sum() == frame[column].notna().sum():
            frame[column] = converted
    frame.to_csv(output, index=False, compression="gzip")
    return frame


def download_cursor_array(
    client: OkxClient,
    dataset: ArrayDataset,
    start: pd.Timestamp,
    end: pd.Timestamp,
    output_dir: Path,
    force: bool,
) -> pd.DataFrame:
    output = output_dir / f"{dataset.name}.csv.gz"
    partial = output_dir / ".partial" / f"{dataset.name}.jsonl"
    if output.exists() and not force:
        frame = pd.read_csv(output, parse_dates=[dataset.timestamp_column])
        print(f"[done] {dataset.name}: existing rows={len(frame):,}", flush=True)
        return frame
    if force:
        partial.unlink(missing_ok=True)

    rows = load_partial(partial)
    seen_timestamps = {
        int(row[0]) for row in rows if isinstance(row, list) and row and str(row[0]).isdigit()
    }
    cursor = min(seen_timestamps) if seen_timestamps else ms(end) + 1
    page = 0
    while cursor > ms(start):
        params = dict(dataset.params)
        params.update({"bar": "1H", "limit": str(dataset.page_limit), "after": str(cursor)})
        batch = client.get(dataset.path, params)
        if not batch:
            break
        fresh = [row for row in batch if int(row[0]) not in seen_timestamps]
        if fresh:
            append_partial(partial, fresh)
            rows.extend(fresh)
            seen_timestamps.update(int(row[0]) for row in fresh)
        oldest = min(int(row[0]) for row in batch)
        page += 1
        if page == 1 or page % 20 == 0:
            stamp = pd.to_datetime(oldest, unit="ms", utc=True)
            print(f"  {dataset.name}: pages={page:,} rows={len(rows):,} oldest={stamp}", flush=True)
        if oldest >= cursor or oldest <= ms(start):
            break
        cursor = oldest

    frame = finalize_array_dataset(rows, dataset.columns, dataset.timestamp_column, output)
    frame = frame.loc[
        (frame[dataset.timestamp_column] >= start) & (frame[dataset.timestamp_column] <= end)
    ].copy()
    frame.to_csv(output, index=False, compression="gzip")
    partial.unlink(missing_ok=True)
    print(f"[done] {dataset.name}: rows={len(frame):,} coverage={coverage_text(frame, dataset.timestamp_column)}", flush=True)
    return frame


def download_funding(
    client: OkxClient,
    inst_id: str,
    requested_start: pd.Timestamp,
    end: pd.Timestamp,
    output_dir: Path,
    force: bool,
) -> pd.DataFrame:
    name = "funding_rate"
    output = output_dir / f"{name}.csv.gz"
    partial = output_dir / ".partial" / f"{name}.jsonl"
    if output.exists() and not force:
        frame = pd.read_csv(output, parse_dates=["timestamp"])
        print(f"[done] {name}: existing rows={len(frame):,}", flush=True)
        return frame
    if force:
        partial.unlink(missing_ok=True)

    # OKX documents a three-month retention limit for this endpoint.
    start = max(requested_start, end - pd.Timedelta(days=92))
    rows = load_partial(partial)
    seen = {str(row.get("fundingTime")) for row in rows if isinstance(row, dict)}
    cursor = min((int(value) for value in seen if value.isdigit()), default=ms(end) + 1)
    page = 0
    while cursor > ms(start):
        batch = client.get(
            "/api/v5/public/funding-rate-history",
            {"instId": inst_id, "limit": "100", "after": str(cursor)},
        )
        if not batch:
            break
        fresh = [row for row in batch if str(row.get("fundingTime")) not in seen]
        if fresh:
            append_partial(partial, fresh)
            rows.extend(fresh)
            seen.update(str(row.get("fundingTime")) for row in fresh)
        oldest = min(int(row["fundingTime"]) for row in batch)
        page += 1
        if oldest >= cursor or oldest <= ms(start):
            break
        cursor = oldest

    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(
            columns=["timestamp", "inst_id", "funding_rate", "realized_rate", "formula_type", "method"]
        )
    else:
        frame["timestamp"] = pd.to_datetime(
            pd.to_numeric(frame["fundingTime"], errors="coerce"), unit="ms", utc=True
        )
        frame = frame.rename(
            columns={
                "instId": "inst_id",
                "fundingRate": "funding_rate",
                "realizedRate": "realized_rate",
                "formulaType": "formula_type",
            }
        )
        frame["funding_rate"] = pd.to_numeric(frame["funding_rate"], errors="coerce")
        frame["realized_rate"] = pd.to_numeric(frame["realized_rate"], errors="coerce")
        frame = frame[
            ["timestamp", "inst_id", "funding_rate", "realized_rate", "formula_type", "method"]
        ]
        frame = frame.drop_duplicates("timestamp").sort_values("timestamp")
        frame = frame.loc[(frame["timestamp"] >= start) & (frame["timestamp"] <= end)]
    frame.to_csv(output, index=False, compression="gzip")
    partial.unlink(missing_ok=True)
    print(f"[done] {name}: rows={len(frame):,} coverage={coverage_text(frame, 'timestamp')}", flush=True)
    return frame


def download_rubik_array(
    client: OkxClient,
    dataset: ArrayDataset,
    requested_start: pd.Timestamp,
    end: pd.Timestamp,
    output_dir: Path,
    force: bool,
) -> pd.DataFrame:
    output = output_dir / f"{dataset.name}.csv.gz"
    partial = output_dir / ".partial" / f"{dataset.name}.jsonl"
    if output.exists() and not force:
        frame = pd.read_csv(output, parse_dates=[dataset.timestamp_column])
        print(f"[done] {dataset.name}: existing rows={len(frame):,}", flush=True)
        return frame
    if force:
        partial.unlink(missing_ok=True)

    retention_days = dataset.history_days or 92
    start = max(requested_start, end - pd.Timedelta(days=retention_days))
    rows = load_partial(partial)
    # Small windows avoid endpoint-specific response caps that are not consistently documented.
    window = pd.Timedelta(hours=72)
    cursor_end = end
    page = 0
    while cursor_end > start:
        cursor_start = max(start, cursor_end - window)
        params = dict(dataset.params)
        params.update({"period": "1H", "begin": str(ms(cursor_start)), "end": str(ms(cursor_end))})
        try:
            batch = client.get(dataset.path, params)
        except RuntimeError as exc:
            # Some Rubik endpoints expose slightly less than the documented retention window.
            if "50030" in str(exc):
                cursor_end = cursor_start
                continue
            raise
        if batch:
            append_partial(partial, batch)
            rows.extend(batch)
        page += 1
        if page == 1 or page % 10 == 0:
            print(f"  {dataset.name}: windows={page:,} raw_rows={len(rows):,} at={cursor_start}", flush=True)
        cursor_end = cursor_start

    frame = finalize_array_dataset(rows, dataset.columns, dataset.timestamp_column, output)
    frame = frame.loc[
        (frame[dataset.timestamp_column] >= start) & (frame[dataset.timestamp_column] <= end)
    ].copy()
    frame.to_csv(output, index=False, compression="gzip")
    partial.unlink(missing_ok=True)
    print(f"[done] {dataset.name}: rows={len(frame):,} coverage={coverage_text(frame, dataset.timestamp_column)}", flush=True)
    return frame


def coverage_text(frame: pd.DataFrame, timestamp_column: str) -> str:
    if frame.empty:
        return "empty"
    timestamps = pd.to_datetime(frame[timestamp_column], utc=True)
    return f"{timestamps.min().isoformat()} .. {timestamps.max().isoformat()}"


def dataset_metadata(frame: pd.DataFrame, timestamp_column: str, requested_start: pd.Timestamp) -> dict[str, Any]:
    if frame.empty:
        return {
            "rows": 0,
            "first_timestamp": None,
            "last_timestamp": None,
            "requested_start_reached": False,
        }
    timestamps = pd.to_datetime(frame[timestamp_column], utc=True)
    return {
        "rows": int(len(frame)),
        "first_timestamp": timestamps.min().isoformat(),
        "last_timestamp": timestamps.max().isoformat(),
        "requested_start_reached": bool(timestamps.min() <= requested_start + pd.Timedelta(hours=1)),
    }


def main() -> None:
    args = parse_args()
    now = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    start = as_utc(args.start)
    end = as_utc(args.end, default=now)
    if start >= end:
        raise ValueError("--start must be earlier than --end")

    output_dir = Path(args.output_root).resolve() / args.inst_id
    output_dir.mkdir(parents=True, exist_ok=True)
    client = OkxClient(args.sleep, args.timeout, args.retries)
    print(
        f"OKX one-symbol download: {args.inst_id} requested={start.isoformat()} .. {end.isoformat()}",
        flush=True,
    )

    cursor_datasets = [
        ArrayDataset(
            "swap_price_1h",
            "/api/v5/market/history-candles",
            {"instId": args.inst_id},
            (
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume_contracts",
                "volume_base",
                "volume_quote",
                "confirmed",
            ),
            "timestamp",
        ),
        ArrayDataset(
            "mark_price_1h",
            "/api/v5/market/history-mark-price-candles",
            {"instId": args.inst_id},
            ("timestamp", "open", "high", "low", "close", "confirmed"),
            "timestamp",
            page_limit=100,
        ),
        ArrayDataset(
            "index_price_1h",
            "/api/v5/market/history-index-candles",
            {"instId": args.index_id},
            ("timestamp", "open", "high", "low", "close", "confirmed"),
            "timestamp",
            page_limit=100,
        ),
    ]
    frames: dict[str, pd.DataFrame] = {}
    for dataset in cursor_datasets:
        frames[dataset.name] = download_cursor_array(
            client, dataset, start, end, output_dir, args.force
        )

    frames["funding_rate"] = download_funding(
        client, args.inst_id, start, end, output_dir, args.force
    )

    rubik_datasets = [
        ArrayDataset(
            "open_interest_1h",
            "/api/v5/rubik/stat/contracts/open-interest-history",
            {"instId": args.inst_id},
            ("timestamp", "open_interest_contracts", "open_interest_ccy", "open_interest_usd"),
            "timestamp",
            history_days=92,
        ),
        ArrayDataset(
            "long_short_account_ratio_1h",
            "/api/v5/rubik/stat/contracts/long-short-account-ratio",
            {"ccy": args.ccy},
            ("timestamp", "long_short_account_ratio"),
            "timestamp",
            history_days=92,
        ),
        ArrayDataset(
            "taker_volume_1h",
            "/api/v5/rubik/stat/taker-volume",
            {"ccy": args.ccy, "instType": "CONTRACTS"},
            ("timestamp", "sell_volume", "buy_volume"),
            "timestamp",
            history_days=92,
        ),
    ]
    for dataset in rubik_datasets:
        frames[dataset.name] = download_rubik_array(
            client, dataset, start, end, output_dir, args.force
        )

    mark = frames["mark_price_1h"][["timestamp", "close"]].rename(columns={"close": "mark_close"})
    index = frames["index_price_1h"][["timestamp", "close"]].rename(columns={"close": "index_close"})
    basis = mark.merge(index, on="timestamp", how="inner")
    basis["mark_index_basis"] = basis["mark_close"] / basis["index_close"] - 1.0
    basis.to_csv(output_dir / "mark_index_basis_1h.csv.gz", index=False, compression="gzip")
    frames["mark_index_basis_1h"] = basis
    print(
        f"[done] mark_index_basis_1h: rows={len(basis):,} coverage={coverage_text(basis, 'timestamp')}",
        flush=True,
    )

    manifest = {
        "exchange": "OKX",
        "instrument_id": args.inst_id,
        "index_id": args.index_id,
        "currency": args.ccy,
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "requested_start": start.isoformat(),
        "requested_end": end.isoformat(),
        "api_base_url": BASE_URL,
        "authentication": "none; public endpoints",
        "transform_policy": "raw endpoint values only; no forward fill; mark-index basis is same-timestamp arithmetic",
        "known_source_limits": {
            "funding_rate": "OKX documents up to three months",
            "rubik_statistics": "finite recent history; requested in 72-hour windows and actual coverage recorded",
            "price_candles": "OKX history-candle endpoints; requested back to configured start",
        },
        "datasets": {
            name: dataset_metadata(frame, "timestamp", start) for name, frame in frames.items()
        },
        "http_request_count": client.request_count,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"Manifest: {manifest_path}", flush=True)
    print(f"HTTP requests: {client.request_count}", flush=True)


if __name__ == "__main__":
    main()
