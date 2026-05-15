"""
data_loader.py
==============
Intelligent data ingestion layer with local CSV caching.
- First run: downloads full history from yfinance and persists to data/{ticker}.csv
- Subsequent runs: reads cached CSV, detects the latest stored date, and appends only missing rows
- Retry / exponential back-off for transient yfinance failures
- Inter-ticker delay to respect rate limits
"""

from __future__ import annotations

import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

DEFAULT_TICKERS: List[str] = ["SPY", "QQQ", "TLT", "GLD", "XLF", "XLK", "BTC-USD"]
FULL_HISTORY_START = "2005-01-01"

RETRY_ATTEMPTS = 4
RETRY_BASE_DELAY = 2.0      # seconds (doubles each retry)
INTER_TICKER_DELAY = 1.2    # seconds between ticker downloads

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _csv_path(ticker: str) -> Path:
    safe = ticker.replace("-", "_")
    return DATA_DIR / f"{safe}.csv"


def _download_with_retry(
    ticker: str,
    start: str,
    end: Optional[str] = None,
    attempts: int = RETRY_ATTEMPTS,
) -> pd.DataFrame:
    """Download OHLCV data with exponential back-off on failure."""
    delay = RETRY_BASE_DELAY
    for attempt in range(1, attempts + 1):
        try:
            kw: dict = dict(start=start, auto_adjust=True, progress=False)
            if end:
                kw["end"] = end
            df = yf.download(ticker, **kw)
            if df is not None and not df.empty:
                # Flatten MultiIndex columns produced by newer yfinance versions
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.index = pd.to_datetime(df.index)
                df.index.name = "Date"
                return df
            log.warning("Empty response for %s (attempt %d/%d)", ticker, attempt, attempts)
        except Exception as exc:
            log.warning("Error fetching %s (attempt %d/%d): %s", ticker, attempt, attempts, exc)
        if attempt < attempts:
            log.info("Retrying %s in %.1fs …", ticker, delay)
            time.sleep(delay)
            delay *= 2
    return pd.DataFrame()


def _load_cached(ticker: str) -> pd.DataFrame:
    path = _csv_path(ticker)
    if path.exists():
        try:
            df = pd.read_csv(path, index_col="Date", parse_dates=True)
            return df
        except Exception as exc:
            log.warning("Cache read failed for %s: %s — re-downloading.", ticker, exc)
    return pd.DataFrame()


def _save_csv(ticker: str, df: pd.DataFrame) -> None:
    path = _csv_path(ticker)
    df.to_csv(path)


def fetch_ticker(ticker: str) -> pd.DataFrame:
    """
    Load or incrementally update OHLCV data for a single ticker.

    1. Reads cached CSV if it exists.
    2. Determines the next required date (last cached date + 1 day).
    3. Downloads only the missing range from yfinance.
    4. Deduplicates, sorts, and saves the merged result back to CSV.
    """
    cached = _load_cached(ticker)

    if cached.empty:
        log.info("[%s] No cache found — downloading full history from %s", ticker, FULL_HISTORY_START)
        fresh = _download_with_retry(ticker, start=FULL_HISTORY_START)
        if fresh.empty:
            log.error("[%s] Download failed — returning empty DataFrame.", ticker)
            return pd.DataFrame()
        _save_csv(ticker, fresh)
        return fresh

    latest_date: pd.Timestamp = cached.index.max()
    next_start: str = (latest_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    today: str = pd.Timestamp.today().strftime("%Y-%m-%d")

    if next_start >= today:
        log.info("[%s] Cache is up-to-date (latest: %s).", ticker, latest_date.date())
        return cached

    log.info("[%s] Fetching new rows from %s to %s", ticker, next_start, today)
    incremental = _download_with_retry(ticker, start=next_start)

    if incremental.empty:
        log.info("[%s] No new data returned — using existing cache.", ticker)
        return cached

    merged = pd.concat([cached, incremental])
    merged = merged[~merged.index.duplicated(keep="last")]
    merged.sort_index(inplace=True)
    _save_csv(ticker, merged)
    log.info("[%s] Cache updated — %d rows total.", ticker, len(merged))
    return merged


def load_all_tickers(
    tickers: List[str] = DEFAULT_TICKERS,
    delay: float = INTER_TICKER_DELAY,
) -> Dict[str, pd.DataFrame]:
    """
    Load/update data for all tickers with inter-ticker delay.
    Returns a dict mapping ticker -> DataFrame.
    """
    results: Dict[str, pd.DataFrame] = {}
    for i, ticker in enumerate(tickers):
        log.info("Processing ticker %d/%d: %s", i + 1, len(tickers), ticker)
        results[ticker] = fetch_ticker(ticker)
        if i < len(tickers) - 1:
            time.sleep(delay)
    return results


# ---------------------------------------------------------------------------
# Derived series helpers
# ---------------------------------------------------------------------------

def compute_returns(price_series: pd.Series, log_returns: bool = True) -> pd.Series:
    """Compute daily log or simple returns from a price series."""
    if log_returns:
        return np.log(price_series / price_series.shift(1)).dropna()
    return price_series.pct_change().dropna()


def build_returns_matrix(
    data: Dict[str, pd.DataFrame],
    price_col: str = "Close",
    log_returns: bool = True,
) -> pd.DataFrame:
    """
    Stack all ticker return series into a single aligned DataFrame.
    Dates with missing tickers are forward-filled then dropped.
    """
    series_map = {}
    for ticker, df in data.items():
        if df.empty or price_col not in df.columns:
            continue
        series_map[ticker] = compute_returns(df[price_col], log_returns=log_returns)

    if not series_map:
        return pd.DataFrame()

    returns = pd.DataFrame(series_map)
    returns.dropna(how="all", inplace=True)
    return returns


def build_price_matrix(
    data: Dict[str, pd.DataFrame],
    price_col: str = "Close",
) -> pd.DataFrame:
    prices = {t: df[price_col] for t, df in data.items() if not df.empty and price_col in df.columns}
    return pd.DataFrame(prices).dropna(how="all")
