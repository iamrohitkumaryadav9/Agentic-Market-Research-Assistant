"""
Node: analyze_sentiment

LLM-based per-article sentiment classification using Claude (or OpenAI fallback).

For each news article, the LLM produces a structured verdict:
  - sentiment: "positive" | "negative" | "neutral"
  - confidence: 0.0–1.0
  - rationale: brief explanation of the classification

If Massive.com already provided sentiment insights, they're included as
context for the LLM (but the LLM makes its own independent assessment).

Results are aggregated into a SentimentSummary.

Graceful degradation:
  - If no news articles available → skip with neutral summary
  - If LLM call fails for an article → mark it neutral with low confidence
  - If all LLM calls fail → proceed with neutral summary
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
    """LLM output schema for a single article's sentiment classification."""
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


# ---------------------------------------------------------------------------
# Sentiment classification prompt
# ---------------------------------------------------------------------------

_SENTIMENT_SYSTEM_PROMPT = """\
You are a financial news sentiment analyst. Your job is to classify the sentiment \
of a news article as it relates to a specific stock ticker.

Rules:
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


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------

def analyze_sentiment(state: ResearchState) -> dict:
    """
    Classify sentiment for each news article via LLM structured output.

    On failure for individual articles: marks them neutral with low confidence.
    On total failure: returns a neutral summary so the graph can proceed.
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
        structured_llm = llm.with_structured_output(ArticleSentimentOutput)
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

    # --- Classify each article ---
    article_sentiments: list[SentimentResult] = []
    errors: list[str] = []

    for i, article in enumerate(articles):
        try:
            user_prompt = _build_article_prompt(ticker, article)
            result: ArticleSentimentOutput = structured_llm.invoke(
                [
                    {"role": "system", "content": _SENTIMENT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]
            )

            article_sentiments.append(
                SentimentResult(
                    article_title=article.get("title", f"Article {i + 1}"),
                    article_url=article.get("url", ""),
                    sentiment=result.sentiment.lower(),
                    confidence=result.confidence,
                    rationale=result.rationale,
                )
            )
            logger.debug(
                "Article %d/%d: %s (%.0f%%) — %s",
                i + 1, len(articles), result.sentiment,
                result.confidence * 100, result.rationale[:60],
            )

        except Exception as e:
            logger.warning(
                "Sentiment classification failed for article %d: %s", i + 1, e
            )
            errors.append(
                f"[analyze_sentiment] Failed for article {i + 1} "
                f"('{article.get('title', '?')[:40]}'): {e}"
            )
            # Graceful fallback: mark as neutral with low confidence
            article_sentiments.append(
                SentimentResult(
                    article_title=article.get("title", f"Article {i + 1}"),
                    article_url=article.get("url", ""),
                    sentiment="neutral",
                    confidence=0.2,
                    rationale=f"Classification failed ({e}); defaulting to neutral",
                )
            )

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
                    "failed_classifications": len(errors),
                },
            ).model_dump()
        ],
        "error_log": errors,
    }
