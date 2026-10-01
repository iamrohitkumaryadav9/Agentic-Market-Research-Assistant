"""
Node: analyze_sentiment

LLM-based per-article sentiment classification using Claude (or OpenAI fallback).

For each news article, the LLM produces a structured verdict in one batch call:
  - sentiment: "positive" | "negative" | "neutral"
  - confidence: 0.0–1.0
  - rationale: brief explanation of the classification

If Massive.com already provided sentiment insights, they're included as
context for the LLM (but the LLM makes its own independent assessment).

Results are aggregated into a SentimentSummary.

Graceful degradation:
    - If no news articles are available → skip with unavailable sentiment
    - If the batch LLM call fails → report unavailable sentiment and the error
"""

from __future__ import annotations

import logging
import time

from pydantic import BaseModel, Field

from app.config import get_llm
from app.state import LogEntry, ResearchState, SentimentResult, SentimentSummary

logger = logging.getLogger("market_research_agent.nodes.sentiment")


# ---------------------------------------------------------------------------
# Structured output schema for LLM sentiment classification
# ---------------------------------------------------------------------------

class ArticleSentimentOutput(BaseModel):
    """LLM output schema for one indexed article in a sentiment batch."""
    article_index: int = Field(
        ge=0,
        description="Zero-based index of the input article being classified",
    )
    sentiment: str = Field(
        description="The overall sentiment: 'positive', 'negative', or 'neutral'"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in the sentiment classification (0.0 to 1.0)"
    )
    rationale: str = Field(
        description=(
            "Brief 1-2 sentence explanation of why this sentiment was assigned, "
            "referencing specific claims or language in the article"
        )
    )


class SentimentBatchOutput(BaseModel):
    """Structured sentiment classifications for a batch of articles."""
    article_sentiments: list[ArticleSentimentOutput]


# ---------------------------------------------------------------------------
# Sentiment classification prompt
# ---------------------------------------------------------------------------

_SENTIMENT_SYSTEM_PROMPT = """\
You are a financial news sentiment analyst. Your job is to classify the sentiment \
of a news article as it relates to a specific stock ticker.

Rules:
- Return exactly one result for every input article, preserving its article_index.
- Do not omit, duplicate, or invent article indexes.
- Focus on how the article's content would likely affect investor sentiment \
toward the SPECIFIC TICKER, not the market in general.
- "positive" = the article contains information likely to increase investor \
confidence or the stock price (earnings beat, new product, analyst upgrade, etc.)
- "negative" = the article contains information likely to decrease investor \
confidence or the stock price (earnings miss, lawsuit, analyst downgrade, etc.)
- "neutral" = the article is informational without a clear directional bias, \
or the ticker is mentioned only tangentially.
- Set confidence high (>0.8) only when the sentiment signal is strong and \
unambiguous. Mixed articles should get lower confidence.
- Be concise in your rationale — cite specific facts from the article.
- You are classifying sentiment, NOT giving investment advice. Do not recommend \
any action."""


def _build_article_prompt(ticker: str, article: dict) -> str:
    """Build the user prompt for classifying a single article."""
    parts = [
        f"Ticker under analysis: {ticker}",
        f"\nArticle title: {article.get('title', 'N/A')}",
    ]

    desc = article.get("description")
    if desc:
        parts.append(f"Article description: {desc}")

    source = article.get("source", "Unknown")
    parts.append(f"Source: {source}")

    published = article.get("published_utc", "N/A")
    parts.append(f"Published: {published}")

    # Include Massive.com's pre-existing sentiment as context (not ground truth)
    insights = article.get("insights")
    if insights:
        for ins in insights:
            if ins.get("ticker", "").upper() == ticker.upper():
                parts.append(
                    f"\nPre-existing sentiment tag from data provider: "
                    f"{ins.get('sentiment', 'N/A')} — "
                    f"{ins.get('sentiment_reasoning', 'N/A')}"
                )
                parts.append(
                    "(Use this as context but make your own independent assessment.)"
                )

    parts.append(
        "\nClassify the sentiment of this article for the specified ticker."
    )

    return "\n".join(parts)


def _build_batch_prompt(ticker: str, articles: list[dict]) -> str:
    """Build one indexed request containing the articles to classify."""
    parts = [
        f"Classify the following {len(articles)} articles for ticker {ticker}.",
        "Return one result per article, using the exact zero-based article_index.",
    ]
    for index, article in enumerate(articles):
        parts.append(f"\n--- article_index: {index} ---")
        parts.append(_build_article_prompt(ticker, article))
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------

def analyze_sentiment(state: ResearchState) -> dict:
    """
    Classify all news articles in one LLM structured-output request.

    On failure: returns unavailable sentiment and marks the node failed so the
    API and reviewer can identify a degraded run.
    """
    start = time.time()
    ticker = state["ticker"]
    articles = state.get("news_articles") or []

    # --- Handle no articles ---
    if not articles:
        elapsed_ms = (time.time() - start) * 1000
        empty_summary = SentimentSummary(
            overall_sentiment="unavailable",
            average_confidence=0.0,
        )
        return {
            "sentiment_summary": empty_summary.model_dump(),
            "run_log": [
                LogEntry(
                    node="analyze_sentiment",
                    status="skipped",
                    duration_ms=elapsed_ms,
                    message="No news articles to analyze — sentiment unavailable",
                ).model_dump()
            ],
        }

    # --- Initialize LLM with structured output ---
    try:
        llm = get_llm()
        structured_llm = llm.with_structured_output(SentimentBatchOutput)
    except Exception as e:
        logger.error("Failed to initialize LLM for sentiment: %s", e)
        elapsed_ms = (time.time() - start) * 1000
        fallback_summary = SentimentSummary(
            overall_sentiment="unavailable",
            average_confidence=0.0,
        )
        return {
            "sentiment_summary": fallback_summary.model_dump(),
            "run_log": [
                LogEntry(
                    node="analyze_sentiment",
                    status="failed",
                    duration_ms=elapsed_ms,
                    message=f"LLM initialization failed: {e}",
                ).model_dump()
            ],
            "error_log": [f"[analyze_sentiment] LLM init failed: {e}"],
        }

    # --- Classify the batch in one request ---
    try:
        result: SentimentBatchOutput = structured_llm.invoke(
            [
                {"role": "system", "content": _SENTIMENT_SYSTEM_PROMPT},
                {"role": "user", "content": _build_batch_prompt(ticker, articles)},
            ]
        )
        results_by_index = {}
        for item in result.article_sentiments:
            if item.article_index >= len(articles) or item.article_index in results_by_index:
                raise ValueError(f"Invalid or duplicate article_index: {item.article_index}")
            if item.sentiment.lower() not in {"positive", "negative", "neutral"}:
                raise ValueError(f"Invalid sentiment for article {item.article_index}")
            results_by_index[item.article_index] = item

        expected_indexes = set(range(len(articles)))
        if set(results_by_index) != expected_indexes:
            raise ValueError("Sentiment response did not classify every article")

        article_sentiments = [
            SentimentResult(
                article_title=article.get("title", f"Article {index + 1}"),
                article_url=article.get("url", ""),
                sentiment=results_by_index[index].sentiment.lower(),
                confidence=results_by_index[index].confidence,
                rationale=results_by_index[index].rationale,
            )
            for index, article in enumerate(articles)
        ]
    except Exception as e:
        elapsed_ms = (time.time() - start) * 1000
        message = f"[analyze_sentiment] Batch classification unavailable: {e}"
        logger.warning("Sentiment batch failed for %s: %s", ticker, e)
        unavailable_summary = SentimentSummary(
            overall_sentiment="unavailable",
            average_confidence=0.0,
        )
        return {
            "sentiment_summary": unavailable_summary.model_dump(),
            "run_log": [
                LogEntry(
                    node="analyze_sentiment",
                    status="failed",
                    duration_ms=elapsed_ms,
                    message=message,
                    details={"article_count": len(articles), "degraded": True},
                ).model_dump()
            ],
            "error_log": [message],
        }

    # --- Aggregate ---
    pos = sum(1 for s in article_sentiments if s.sentiment == "positive")
    neg = sum(1 for s in article_sentiments if s.sentiment == "negative")
    neu = sum(1 for s in article_sentiments if s.sentiment == "neutral")
    avg_conf = (
        sum(s.confidence for s in article_sentiments) / len(article_sentiments)
        if article_sentiments
        else 0.0
    )

    # Determine overall sentiment
    if pos > neg and pos > neu:
        overall = "positive"
    elif neg > pos and neg > neu:
        overall = "negative"
    elif pos == neg and pos > 0:
        overall = "mixed"
    else:
        overall = "neutral"

    summary = SentimentSummary(
        overall_sentiment=overall,
        positive_count=pos,
        negative_count=neg,
        neutral_count=neu,
        average_confidence=round(avg_conf, 3),
        article_sentiments=article_sentiments,
    )

    elapsed_ms = (time.time() - start) * 1000
    logger.info(
        "Sentiment analysis complete for %s: %s (pos=%d, neg=%d, neu=%d, avg_conf=%.2f) in %.0fms",
        ticker, overall, pos, neg, neu, avg_conf, elapsed_ms,
    )

    return {
        "sentiment_summary": summary.model_dump(),
        "run_log": [
            LogEntry(
                node="analyze_sentiment",
                status="completed",
                duration_ms=elapsed_ms,
                message=(
                    f"Sentiment for {ticker}: {overall} "
                    f"(+{pos}/-{neg}/~{neu}, avg confidence {avg_conf:.0%}). "
                    f"Analyzed {len(articles)} articles."
                ),
                details={
                    "overall": overall,
                    "positive": pos,
                    "negative": neg,
                    "neutral": neu,
                    "avg_confidence": round(avg_conf, 3),
                    "failed_classifications": 0,
                },
            ).model_dump()
        ],
        "error_log": [],
    }
