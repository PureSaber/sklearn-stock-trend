"""OHLCV data fetching and local caching."""

from __future__ import annotations

import os
import time
from pathlib import Path

import akshare as ak
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

OHLCV_COLUMNS = ["date", "open", "high", "low", "close", "volume"]

_AKSHARE_COLUMN_MAP = {
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
}


def parse_symbol(symbol: str) -> str:
    """Convert '000001.SZ' to akshare code '000001'."""
    return symbol.split(".")[0]


def symbol_to_index_code(symbol: str) -> str:
    """Convert '000300.SH' to akshare index code 'sh000300'."""
    code, market = symbol.split(".")
    prefix = "sh" if market.upper() == "SH" else "sz"
    return f"{prefix}{code}"


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.rename(columns=_AKSHARE_COLUMN_MAP)
    missing = [col for col in OHLCV_COLUMNS if col not in renamed.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns after rename: {missing}")

    out = renamed[OHLCV_COLUMNS].copy()
    out["date"] = pd.to_datetime(out["date"])
    for col in ("open", "high", "low", "close", "volume"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna().sort_values("date").reset_index(drop=True)
    return out


def _to_akshare_date(date_str: str) -> str:
    return date_str.replace("-", "")


def fetch_ohlcv(symbol: str, start_date: str, end_date: str, retries: int = 3) -> pd.DataFrame:
    """Fetch daily OHLCV from akshare (forward-adjusted)."""
    code = parse_symbol(symbol)
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            raw = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=_to_akshare_date(start_date),
                end_date=_to_akshare_date(end_date),
                adjust="qfq",
            )
            if raw.empty:
                raise ValueError(f"No data returned for {symbol} ({start_date} to {end_date})")
            return _normalize_ohlcv(raw)
        except Exception as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
    raise last_error  # type: ignore[misc]


def _cache_path(cache_dir: Path, symbol: str, start_date: str, end_date: str) -> Path:
    safe_symbol = symbol.replace(".", "_")
    filename = f"{safe_symbol}_{start_date}_{end_date}.csv"
    return cache_dir / filename


def load_or_fetch(
    symbol: str,
    start_date: str,
    end_date: str,
    cache_dir: Path | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load OHLCV from cache or fetch remotely."""
    cache_root = cache_dir or Path(os.getenv("DATA_DIR", "./data"))
    cache_root.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_root, symbol, start_date, end_date)

    if use_cache and path.exists():
        df = pd.read_csv(path, parse_dates=["date"])
        return df.sort_values("date").reset_index(drop=True)

    df = fetch_ohlcv(symbol, start_date, end_date)
    df.to_csv(path, index=False)
    return df


def fetch_benchmark(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily index close series for benchmark comparison."""
    index_code = symbol_to_index_code(symbol)
    raw = ak.stock_zh_index_daily(symbol=index_code)
    if raw.empty:
        raise ValueError(f"No benchmark data returned for {symbol}")

    raw = raw.rename(columns={"date": "date", "close": "close"})
    raw["date"] = pd.to_datetime(raw["date"])
    raw["close"] = pd.to_numeric(raw["close"], errors="coerce")
    raw = raw.dropna(subset=["date", "close"]).sort_values("date")

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    filtered = raw[(raw["date"] >= start) & (raw["date"] <= end)]
    if filtered.empty:
        raise ValueError(f"No benchmark data for {symbol} in {start_date} to {end_date}")
    return filtered[["date", "close"]].reset_index(drop=True)


def load_or_fetch_benchmark(
    symbol: str,
    start_date: str,
    end_date: str,
    cache_dir: Path | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load benchmark index from cache or fetch remotely."""
    cache_root = cache_dir or Path(os.getenv("DATA_DIR", "./data"))
    cache_root.mkdir(parents=True, exist_ok=True)
    safe_symbol = symbol.replace(".", "_")
    path = cache_root / f"benchmark_{safe_symbol}_{start_date}_{end_date}.csv"

    if use_cache and path.exists():
        return pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

    df = fetch_benchmark(symbol, start_date, end_date)
    df.to_csv(path, index=False)
    return df
