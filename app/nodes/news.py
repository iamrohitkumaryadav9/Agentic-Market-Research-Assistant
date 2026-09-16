"""
Node: fetch_news

Pulls recent news articles scoped to the given ticker.

Primary source: Massive.com (formerly Polygon.io) /v2/reference/news endpoint.
  - Endpoint: GET https://api.massive.com/v2/reference/news?ticker={TICKER}
  - Response: { results: [ { title, description, published_utc, article_url,
                              tickers, publisher { name }, insights [ { ticker,
                              sentiment, sentiment_reasoning } ] } ] }

Fallback: feedparser RSS pull from Google News finance search.
  - URL: https://news.google.com/rss/search?q={TICKER}+stock

Graceful degradation:
  - If Massive API key missing or rate-limited → fall back to RSS
  - If both fail → proceed with news_articles=[], log the failure
  - The final thesis will note that news data was unavailable
"""

from __future__ import annotations

import logging
import time
from datetime import datetime

import feedparser
import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import get_settings
from app.state import LogEntry, NewsArticle, ResearchState

logger = logging.getLogger("market_research_agent.nodes.news")


# ---------------------------------------------------------------------------
# Massive.com (Polygon.io) news fetch
# ---------------------------------------------------------------------------

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((requests.RequestException, requests.Timeout)),
    reraise=True,
)
def _fetch_massive_news(ticker: str, limit: int = 10) -> list[NewsArticle]:
    """
    Fetch ticker-scoped news from Massive.com /v2/reference/news.

    Parses the response based on Massive's actual schema (not a generic shape).
    Raises on HTTP errors or missing data.
    """
    settings = get_settings()

    if not settings.MASSIVE_API_KEY:
        raise ValueError("MASSIVE_API_KEY not configured")

    url = f"{settings.MASSIVE_BASE_URL}/v2/reference/news"
    params = {
        "ticker": ticker.upper(),
        "limit": limit,
        "order": "desc",
        "sort": "published_utc",
        "apiKey": settings.MASSIVE_API_KEY,
    }

    logger.info("Fetching news from Massive.com for %s", ticker)
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()

    data = resp.json()

    if data.get("status") != "OK":
        raise ValueError(
            f"Massive API returned non-OK status: {data.get('status')} "
            f"(request_id: {data.get('request_id', 'N/A')})"
        )

    results = data.get("results", [])
    articles = []

    for item in results:
        # Parse insights (sentiment tags) — Massive-specific field
        insights = None
        raw_insights = item.get("insights")
        if raw_insights and isinstance(raw_insights, list):
            insights = [
                {
                    "ticker": ins.get("ticker", ""),
                    "sentiment": ins.get("sentiment", ""),
                    "sentiment_reasoning": ins.get("sentiment_reasoning", ""),
                }
                for ins in raw_insights
            ]

        # Publisher name from nested publisher object
        publisher = item.get("publisher", {})
        source_name = publisher.get("name", "Massive.com") if isinstance(publisher, dict) else "Massive.com"

        articles.append(
            NewsArticle(
                title=item.get("title", "Untitled"),
                description=item.get("description"),
                source=source_name,
                published_utc=item.get("published_utc", ""),
                url=item.get("article_url", ""),
                tickers=item.get("tickers", []),
                insights=insights,
            )
        )

    return articles


# ---------------------------------------------------------------------------
# RSS fallback via feedparser
# ---------------------------------------------------------------------------

def _fetch_rss_news(ticker: str, limit: int = 10) -> list[NewsArticle]:
    """
    Fallback: fetch news via Google News RSS search for the ticker.

    This provides less structured data (no sentiment tags, limited metadata)
    but works without any API key.
    """
    rss_url = (
        f"https://news.google.com/rss/search?"
        f"q={ticker.upper()}+stock&hl=en-US&gl=US&ceid=US:en"
    )

    logger.info("Falling back to RSS for %s: %s", ticker, rss_url)
    feed = feedparser.parse(rss_url)

    if feed.bozo and not feed.entries:
        logger.warning("RSS feed parse error for %s: %s", ticker, feed.bozo_exception)
        return []

    articles = []
    for entry in feed.entries[:limit]:
        # Parse published date
        published = ""
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            try:
                published = datetime(*entry.published_parsed[:6]).isoformat() + "Z"
            except Exception:
                published = entry.get("published", "")

        # Extract source from title (Google News format: "Title - Source")
        title = entry.get("title", "Untitled")
        source = "Google News RSS"
        if " - " in title:
            parts = title.rsplit(" - ", 1)
            title = parts[0].strip()
            source = parts[1].strip()

        articles.append(
            NewsArticle(
                title=title,
                description=entry.get("summary", None),
                source=f"RSS-{source}",
                published_utc=published,
                url=entry.get("link", ""),
                tickers=[ticker.upper()],
                insights=None,  # RSS doesn't provide sentiment tags
            )
        )

    return articles


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------

def fetch_news(state: ResearchState) -> dict:
    """
    Fetch recent news for the ticker. Tries Massive.com first, falls back
    to RSS, and degrades gracefully if both fail.
    """
    start = time.time()
    ticker = state["ticker"]
    articles: list[NewsArticle] = []
    source_used = "none"
    errors: list[str] = []

    # --- Try Massive.com (primary) ---
    try:
        articles = _fetch_massive_news(ticker, limit=10)
        source_used = "Massive.com"
        logger.info("Massive.com returned %d articles for %s", len(articles), ticker)
    except Exception as e:
        error_msg = f"Massive.com news fetch failed for {ticker}: {e}"
        logger.warning(error_msg)
        errors.append(f"[fetch_news] {error_msg}")

        # --- Try RSS fallback ---
        try:
            articles = _fetch_rss_news(ticker, limit=10)
            source_used = "RSS-GoogleNews"
            logger.info("RSS fallback returned %d articles for %s", len(articles), ticker)
        except Exception as e2:
            error_msg2 = f"RSS fallback also failed for {ticker}: {e2}"
            logger.error(error_msg2)
            errors.append(f"[fetch_news] {error_msg2}")

    elapsed_ms = (time.time() - start) * 1000

    # Build status
    if articles:
        status = "completed"
        message = (
            f"Fetched {len(articles)} articles for {ticker} via {source_used}"
        )
    else:
        status = "degraded" if errors else "completed"
        message = (
            f"No news articles found for {ticker}. "
            f"Source attempted: {source_used or 'Massive.com + RSS'}. "
            "Graph will proceed with technical-only analysis."
        )

    log_entry = LogEntry(
        node="fetch_news",
        status=status,
        duration_ms=elapsed_ms,
        message=message,
        details={
            "source": source_used,
            "article_count": len(articles),
            "has_insights": any(
                a.insights for a in articles if hasattr(a, "insights")
            ),
        },
    ).model_dump()

    return {
        "news_articles": [a.model_dump() for a in articles],
        "run_log": [log_entry],
        "error_log": errors,
    }
