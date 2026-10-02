"""
Node: fetch_market_data

Pulls price history via yfinance and computes basic technical indicators:
  - SMA-20 (20-day Simple Moving Average)
  - SMA-50 (50-day Simple Moving Average)
  - RSI-14 (14-day Relative Strength Index)
  - Volume trend (increasing / decreasing / stable)
  - Price change % (5-day and 30-day)

Graceful degradation: if yfinance fails after retries, the node logs the
error into error_log and the graph proceeds with price_data=None and
technical_signals=None. Downstream nodes handle the missing data.

Rate limiting: uses tenacity retry with exponential backoff.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime

import numpy as np
import pandas as pd
from tenacity import (
    retry,
    retry_if_exception_type,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.state import (
    LogEntry,
    PriceBar,
    PriceData,
    ResearchState,
    TechnicalSignals,
)

logger = logging.getLogger("market_research_agent.nodes.market_data")


# ---------------------------------------------------------------------------
# Technical indicator calculations
# ---------------------------------------------------------------------------

def compute_rsi(closes: pd.Series, window: int = 14) -> float | None:
    """
    Compute RSI (Relative Strength Index) using the standard Wilder method.

    Returns None if there isn't enough data.
    """
    if len(closes) < window + 1:
        return None

    delta = closes.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta.where(delta < 0, 0.0))

    # Wilder's smoothing (exponential moving average)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    last_rsi = rsi.iloc[-1]
    return round(float(last_rsi), 2) if not np.isnan(last_rsi) else None


def compute_volume_trend(volumes: pd.Series, lookback: int = 10) -> str:
    """
    Compare average volume over the recent `lookback` days vs the prior
    `lookback` days to determine trend direction.
    """
    if len(volumes) < lookback * 2:
        return "stable"

    recent = volumes.iloc[-lookback:].mean()
    prior = volumes.iloc[-lookback * 2 : -lookback].mean()

    if prior == 0:
        return "stable"

    change = (recent - prior) / prior
    if change > 0.10:
        return "increasing"
    elif change < -0.10:
        return "decreasing"
    return "stable"


def compute_price_change_pct(closes: pd.Series, days: int) -> float:
    """Compute percentage price change over the last N trading days."""
    if len(closes) < days + 1:
        return 0.0
    current = closes.iloc[-1]
    past = closes.iloc[-(days + 1)]
    if past == 0:
        return 0.0
    return round(((current - past) / past) * 100, 2)


# ---------------------------------------------------------------------------
# yfinance data pull with retry
# ---------------------------------------------------------------------------

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    # Don't back off on "no data for this ticker" — it won't fix itself.
    retry=retry_if_exception_type(Exception) & retry_if_not_exception_type(ValueError),
    reraise=True,
)
def _fetch_yfinance_data(ticker: str, period: str = "3mo") -> pd.DataFrame:
    """Pull OHLCV data from yfinance with retry/backoff."""
    import yfinance as yf

    stock = yf.Ticker(ticker)
    df = stock.history(period=period)

    if df.empty:
        raise ValueError(f"yfinance returned no data for ticker '{ticker}'")

    return df


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------

def fetch_market_data(state: ResearchState) -> dict:
    """
    Fetch price history via yfinance and compute technical indicators.

    On failure: logs error, sets price_data=None and technical_signals=None.
    The graph continues — downstream nodes check for missing data.
    """
    start = time.time()
    ticker = state["ticker"]
    period = "3mo"

    try:
        logger.info("Fetching market data for %s (period=%s)", ticker, period)
        df = _fetch_yfinance_data(ticker, period)

        # Build PriceBar list from DataFrame
        bars = []
        for idx, row in df.iterrows():
            bars.append(
                PriceBar(
                    date=idx.strftime("%Y-%m-%d"),
                    open=round(float(row["Open"]), 2),
                    high=round(float(row["High"]), 2),
                    low=round(float(row["Low"]), 2),
                    close=round(float(row["Close"]), 2),
                    volume=int(row["Volume"]),
                )
            )

        latest_close = round(float(df["Close"].iloc[-1]), 2)
        latest_volume = int(df["Volume"].iloc[-1])

        price_data = PriceData(
            ticker=ticker,
            period=period,
            bars=bars,
            latest_close=latest_close,
            latest_volume=latest_volume,
        )

        # Compute technical indicators
        closes = df["Close"]
        volumes = df["Volume"]

        sma_20 = round(float(closes.rolling(20).mean().iloc[-1]), 2) if len(closes) >= 20 else None
        sma_50 = round(float(closes.rolling(50).mean().iloc[-1]), 2) if len(closes) >= 50 else None
        rsi_14 = compute_rsi(closes, 14)
        vol_trend = compute_volume_trend(volumes, 10)
        pct_5d = compute_price_change_pct(closes, 5)
        pct_30d = compute_price_change_pct(closes, 30)

        # Build signal summary
        summary_parts = []
        if sma_20 and sma_50:
            if latest_close > sma_20 > sma_50:
                summary_parts.append("Price above both SMAs (bullish structure)")
            elif latest_close < sma_20 < sma_50:
                summary_parts.append("Price below both SMAs (bearish structure)")
            elif latest_close > sma_20:
                summary_parts.append("Price above SMA-20 but below SMA-50 (mixed)")
            else:
                summary_parts.append("Price below SMA-20 but above SMA-50 (mixed)")
        elif sma_20:
            pos = "above" if latest_close > sma_20 else "below"
            summary_parts.append(f"Price {pos} SMA-20")

        if rsi_14 is not None:
            if rsi_14 > 70:
                summary_parts.append(f"RSI at {rsi_14} (OVERBOUGHT)")
            elif rsi_14 < 30:
                summary_parts.append(f"RSI at {rsi_14} (OVERSOLD)")
            else:
                summary_parts.append(f"RSI at {rsi_14} (neutral)")

        summary_parts.append(f"Volume {vol_trend}")
        summary_parts.append(f"5d change: {pct_5d:+.1f}%, 30d change: {pct_30d:+.1f}%")

        technical_signals = TechnicalSignals(
            sma_20=sma_20,
            sma_50=sma_50,
            rsi_14=rsi_14,
            volume_trend=vol_trend,
            price_change_pct_5d=pct_5d,
            price_change_pct_30d=pct_30d,
            signal_summary=f"{ticker}: " + ". ".join(summary_parts),
        )

        elapsed_ms = (time.time() - start) * 1000
        logger.info(
            "Market data fetched for %s: close=%.2f, RSI=%.1f, %d bars",
            ticker, latest_close, rsi_14 or 0, len(bars),
        )

        return {
            "price_data": price_data.model_dump(),
            "technical_signals": technical_signals.model_dump(),
            "run_log": [
                LogEntry(
                    node="fetch_market_data",
                    status="completed",
                    duration_ms=elapsed_ms,
                    message=(
                        f"Fetched {len(bars)} bars for {ticker}. "
                        f"Close: ${latest_close}, RSI: {rsi_14}, "
                        f"Volume trend: {vol_trend}"
                    ),
                    details={
                        "bars_count": len(bars),
                        "latest_close": latest_close,
                        "sma_20": sma_20,
                        "sma_50": sma_50,
                        "rsi_14": rsi_14,
                    },
                ).model_dump()
            ],
        }

    except Exception as e:
        elapsed_ms = (time.time() - start) * 1000
        logger.error("Failed to fetch market data for %s: %s", ticker, e)

        return {
            "price_data": None,
            "technical_signals": None,
            "run_log": [
                LogEntry(
                    node="fetch_market_data",
                    status="failed",
                    duration_ms=elapsed_ms,
                    message=f"Failed to fetch market data for {ticker}: {e}",
                ).model_dump()
            ],
            "error_log": [
                f"[fetch_market_data] yfinance failed for {ticker}: {e}"
            ],
        }
