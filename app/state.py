"""
State schema for the Market Research Agent.

Defines:
- Pydantic models for structured, typed data passed between graph nodes.
- ResearchState TypedDict consumed by LangGraph's StateGraph.

Design notes:
- Pydantic models validate data at node boundaries (each node serializes
  its output via .model_dump() before writing to state).
- ResearchState uses TypedDict (not a Pydantic BaseModel) because LangGraph's
  StateGraph channel system expects TypedDict or Annotated fields.
- run_log and error_log use operator.add as their reducer so every node's
  entries are appended, never overwritten.
"""

from __future__ import annotations

import operator
from datetime import datetime, timezone
from typing import Annotated, Optional

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# Pydantic models — structured data exchanged between nodes
# ---------------------------------------------------------------------------

class PriceBar(BaseModel):
    """Single OHLCV bar."""
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int


class PriceData(BaseModel):
    """Raw price history returned by fetch_market_data."""
    ticker: str
    period: str = "3mo"
    bars: list[PriceBar] = Field(default_factory=list)
    latest_close: float = 0.0
    latest_volume: int = 0
    currency: str = "USD"


class TechnicalSignals(BaseModel):
    """Computed technical indicators."""
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    rsi_14: Optional[float] = None
    volume_trend: str = "stable"          # "increasing" | "decreasing" | "stable"
    price_change_pct_5d: float = 0.0      # 5-day price change %
    price_change_pct_30d: float = 0.0     # 30-day price change %
    signal_summary: str = ""              # Brief human-readable summary


class NewsArticle(BaseModel):
    """
    A single news article.

    Schema aligned with Massive.com (formerly Polygon.io) ticker-news
    endpoint fields: ticker, title, description, published_utc, insights.
    Falls back gracefully from RSS/feedparser which provides a subset.
    """
    title: str
    description: Optional[str] = None
    source: str = "unknown"
    published_utc: str = ""
    url: str = ""
    tickers: list[str] = Field(default_factory=list)
    # Massive-specific: sentiment/insight tags returned by their API
    insights: Optional[list[dict]] = None


class SentimentResult(BaseModel):
    """Per-article sentiment classification (LLM-generated)."""
    article_title: str
    article_url: str = ""
    sentiment: str = "neutral"            # "positive" | "negative" | "neutral"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    rationale: str = ""


class SentimentSummary(BaseModel):
    """Aggregated sentiment across all articles."""
    overall_sentiment: str = "neutral"
    positive_count: int = 0
    negative_count: int = 0
    neutral_count: int = 0
    average_confidence: float = 0.0
    article_sentiments: list[SentimentResult] = Field(default_factory=list)


class DraftThesis(BaseModel):
    """
    Draft research thesis produced by the synthesis node.
    Subject to critique and possible revision before finalization.
    """
    ticker: str
    direction: str = "neutral"            # "bullish" | "bearish" | "neutral"
    confidence_score: float = Field(default=50.0, ge=0.0, le=100.0)
    summary: str = ""
    technical_evidence: list[str] = Field(default_factory=list)
    sentiment_evidence: list[str] = Field(default_factory=list)
    cited_sources: list[str] = Field(default_factory=list)
    risks_and_caveats: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    revision_number: int = 0


class CritiqueResult(BaseModel):
    """Output of the adversarial critique node."""
    approved: bool = False
    issues: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    suggested_revisions: list[str] = Field(default_factory=list)
    confidence_adjustment: Optional[float] = None  # Suggested delta


class FinalThesis(BaseModel):
    """
    Finalized research thesis — the terminal output of a successful run.
    The disclaimer field is populated programmatically and MUST NOT be removed.
    """
    ticker: str
    direction: str
    confidence_score: float
    summary: str
    technical_evidence: list[str]
    sentiment_evidence: list[str]
    cited_sources: list[str]
    risks_and_caveats: list[str]
    contradictions: list[str]
    critique_history: list[dict] = Field(default_factory=list)
    human_approved: bool = False
    disclaimer: str = (
        "NOT FINANCIAL ADVICE — This output is a research artifact produced "
        "by an automated system. It does not constitute investment advice, a "
        "recommendation to buy or sell any security, or an offer to transact. "
        "All conclusions require independent verification by a qualified analyst. "
        "Use at your own risk."
    )
    generated_at: str = Field(default_factory=lambda: datetime.now(tz=timezone.utc).isoformat())
    data_sources_used: list[str] = Field(default_factory=list)
    data_sources_failed: list[str] = Field(default_factory=list)


class LogEntry(BaseModel):
    """Structured log entry for the run trace."""
    node: str
    status: str                           # "started" | "completed" | "failed" | "skipped"
    timestamp: str = Field(default_factory=lambda: datetime.now(tz=timezone.utc).isoformat())
    duration_ms: Optional[float] = None
    message: str = ""
    details: Optional[dict] = None


# ---------------------------------------------------------------------------
# LangGraph state — the single source of truth flowing through the graph
# ---------------------------------------------------------------------------

class ResearchState(TypedDict):
    """
    Typed state schema for the LangGraph StateGraph.

    Fields without Annotated use "last writer wins" (overwrite) semantics.
    run_log and error_log use operator.add so every node appends entries.
    """
    # Input
    ticker: str

    # Data-fetch outputs (serialized Pydantic dicts)
    price_data: Optional[dict]
    technical_signals: Optional[dict]
    news_articles: Optional[list]
    sentiment_summary: Optional[dict]

    # Synthesis / critique
    draft_thesis: Optional[dict]
    critique_notes: Optional[dict]
    critique_count: int

    # Human-in-the-loop
    human_decision: Optional[str]         # "approve" | "reject"

    # Final output
    final_thesis: Optional[dict]

    # Observability — append-only
    run_log: Annotated[list, operator.add]
    error_log: Annotated[list, operator.add]
