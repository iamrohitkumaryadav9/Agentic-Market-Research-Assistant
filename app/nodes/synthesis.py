"""
Node: synthesize_draft

LLM node that combines technical signals + sentiment analysis into a
structured draft research thesis.

Produces:
  - Direction lean: bullish / bearish / neutral
  - Confidence score: 0-100
  - Technical evidence with specifics
  - Sentiment evidence with citations
  - Cross-check: contradictions between technical and sentiment reads
  - Risks and caveats
  - Cited sources (article URLs)

On revision passes (critique_count > 0), the node receives critique_notes
in state and must address the specific issues raised.

Guardrail: checks input for action-oriented requests before synthesis.
"""

from __future__ import annotations

import logging
import time

from pydantic import BaseModel, Field

from app.config import get_llm
from app.guardrails import check_guardrails
from app.state import DraftThesis, LogEntry, ResearchState

logger = logging.getLogger("market_research_agent.nodes.synthesis")


# ---------------------------------------------------------------------------
# Structured output schema for LLM thesis generation
# ---------------------------------------------------------------------------

class ThesisOutput(BaseModel):
    """LLM output schema for the draft thesis."""
    direction: str = Field(
        description="Direction lean: 'bullish', 'bearish', or 'neutral'"
    )
    confidence_score: float = Field(
        ge=0.0, le=100.0,
        description=(
            "Confidence in the thesis direction, 0-100. "
            "Below 40 = low confidence, 40-70 = moderate, above 70 = high. "
            "Be calibrated — high confidence requires strong, corroborating signals."
        )
    )
    summary: str = Field(
        description=(
            "2-4 sentence research summary explaining the thesis. "
            "Reference specific data points (price levels, RSI, sentiment scores)."
        )
    )
    technical_evidence: list[str] = Field(
        description="List of technical evidence points supporting the thesis"
    )
    sentiment_evidence: list[str] = Field(
        description=(
            "List of sentiment evidence points, each citing a specific article "
            "or data source"
        )
    )
    risks_and_caveats: list[str] = Field(
        description=(
            "List of risks, caveats, and factors that could invalidate the thesis"
        )
    )
    contradictions: list[str] = Field(
        description=(
            "List of contradictions between technical and sentiment signals. "
            "If signals are aligned, state that explicitly."
        )
    )


# ---------------------------------------------------------------------------
# Synthesis prompt
# ---------------------------------------------------------------------------

_SYNTHESIS_SYSTEM_PROMPT = """\
You are a senior equity research analyst producing a structured research thesis. \
Your output is a RESEARCH ARTIFACT for human review — it is NOT investment advice \
and must NOT contain buy/sell recommendations or trade instructions.

Rules:
- Synthesize the technical indicators and news sentiment into a coherent thesis.
- Be specific: cite actual numbers (RSI values, SMA levels, price changes, \
sentiment scores) rather than vague qualitative statements.
- Cross-check: explicitly identify contradictions between technical and sentiment \
signals. If the technicals say one thing and the news says another, call it out.
- Calibrate confidence: a 90+ confidence requires overwhelming, corroborating \
signals across both technicals and sentiment. Mixed signals should produce \
40-65 confidence. Contradictory signals should produce <40.
- List risks and caveats — every thesis has them.
- Never use language like "you should buy/sell" or "I recommend." This is analysis, \
not advice."""


def _build_synthesis_prompt(
    ticker: str,
    technical_signals: dict | None,
    sentiment_summary: dict | None,
    critique_notes: dict | None,
    critique_count: int,
) -> str:
    """Build the user prompt for thesis synthesis."""
    parts = [f"## Research Thesis Request: {ticker}\n"]

    # --- Technical signals ---
    if technical_signals:
        parts.append("### Technical Indicators")
        parts.append(f"- SMA-20: {technical_signals.get('sma_20', 'N/A')}")
        parts.append(f"- SMA-50: {technical_signals.get('sma_50', 'N/A')}")
        parts.append(f"- RSI-14: {technical_signals.get('rsi_14', 'N/A')}")
        parts.append(f"- Volume trend: {technical_signals.get('volume_trend', 'N/A')}")
        parts.append(f"- 5-day price change: {technical_signals.get('price_change_pct_5d', 'N/A')}%")
        parts.append(f"- 30-day price change: {technical_signals.get('price_change_pct_30d', 'N/A')}%")
        sig_summary = technical_signals.get("signal_summary", "")
        if sig_summary:
            parts.append(f"- Signal summary: {sig_summary}")
        parts.append("")
    else:
        parts.append("### Technical Indicators")
        parts.append("**UNAVAILABLE** — Technical data could not be fetched. ")
        parts.append("Base the thesis on sentiment only and note this limitation.\n")

    # --- Sentiment summary ---
    if sentiment_summary:
        parts.append("### News Sentiment")
        parts.append(f"- Overall sentiment: {sentiment_summary.get('overall_sentiment', 'N/A')}")
        parts.append(f"- Positive articles: {sentiment_summary.get('positive_count', 0)}")
        parts.append(f"- Negative articles: {sentiment_summary.get('negative_count', 0)}")
        parts.append(f"- Neutral articles: {sentiment_summary.get('neutral_count', 0)}")
        parts.append(f"- Average confidence: {sentiment_summary.get('average_confidence', 0):.0%}")

        # Individual article sentiments
        article_sents = sentiment_summary.get("article_sentiments", [])
        if article_sents:
            parts.append("\n#### Per-Article Breakdown")
            for i, art_sent in enumerate(article_sents, 1):
                parts.append(
                    f"{i}. [{art_sent.get('sentiment', '?').upper()}] "
                    f"({art_sent.get('confidence', 0):.0%}) "
                    f"{art_sent.get('article_title', 'N/A')}"
                )
                if art_sent.get("rationale"):
                    parts.append(f"   Rationale: {art_sent['rationale']}")
                if art_sent.get("article_url"):
                    parts.append(f"   Source: {art_sent['article_url']}")
        parts.append("")
    else:
        parts.append("### News Sentiment")
        parts.append("**UNAVAILABLE** — News data could not be fetched. ")
        parts.append("Base the thesis on technicals only and note this limitation.\n")

    # --- Critique feedback (on revision passes) ---
    if critique_notes and critique_count > 0:
        parts.append("### REVISION REQUIRED — Critique Feedback")
        parts.append(f"This is revision #{critique_count}. Address these issues:\n")

        issues = critique_notes.get("issues", [])
        if issues:
            parts.append("**Issues identified:**")
            for issue in issues:
                parts.append(f"- {issue}")

        unsupported = critique_notes.get("unsupported_claims", [])
        if unsupported:
            parts.append("\n**Unsupported claims to fix or remove:**")
            for claim in unsupported:
                parts.append(f"- {claim}")

        revisions = critique_notes.get("suggested_revisions", [])
        if revisions:
            parts.append("\n**Suggested revisions:**")
            for rev in revisions:
                parts.append(f"- {rev}")

        conf_adj = critique_notes.get("confidence_adjustment")
        if conf_adj is not None:
            parts.append(f"\n**Suggested confidence adjustment:** {conf_adj:+.0f} points")

        parts.append("")

    parts.append(
        "Produce a structured research thesis based on the above data. "
        "Remember: this is analysis for human review, not investment advice."
    )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------

def synthesize_draft(state: ResearchState) -> dict:
    """
    Generate a draft research thesis combining technical signals + sentiment.

    On revision passes, incorporates critique feedback.
    Guardrail check on input before synthesis.
    """
    start = time.time()
    ticker = state["ticker"]
    critique_count = state.get("critique_count", 0)
    critique_notes = state.get("critique_notes")
    technical_signals = state.get("technical_signals")
    sentiment_summary = state.get("sentiment_summary")

    # --- Guardrail check ---
    is_safe, refusal_msg = check_guardrails(ticker)
    if not is_safe:
        elapsed_ms = (time.time() - start) * 1000
        return {
            "draft_thesis": DraftThesis(
                ticker=ticker,
                direction="refused",
                confidence_score=0.0,
                summary=refusal_msg,
                risks_and_caveats=["Request refused by guardrails"],
            ).model_dump(),
            "run_log": [
                LogEntry(
                    node="synthesize_draft",
                    status="refused",
                    duration_ms=elapsed_ms,
                    message=f"Guardrail triggered for {ticker}: {refusal_msg[:80]}",
                ).model_dump()
            ],
        }

    # --- Build prompt ---
    user_prompt = _build_synthesis_prompt(
        ticker, technical_signals, sentiment_summary,
        critique_notes, critique_count,
    )

    # --- LLM call with structured output ---
    try:
        llm = get_llm()
        structured_llm = llm.with_structured_output(ThesisOutput)

        logger.info(
            "Synthesizing draft thesis for %s (revision %d)...",
            ticker, critique_count,
        )

        result: ThesisOutput = structured_llm.invoke(
            [
                {"role": "system", "content": _SYNTHESIS_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        )

        # Collect cited sources from sentiment data
        cited_sources = []
        if sentiment_summary:
            for art_sent in sentiment_summary.get("article_sentiments", []):
                url = art_sent.get("article_url", "")
                if url and url not in cited_sources:
                    cited_sources.append(url)

        draft = DraftThesis(
            ticker=ticker,
            direction=result.direction.lower(),
            confidence_score=result.confidence_score,
            summary=result.summary,
            technical_evidence=result.technical_evidence,
            sentiment_evidence=result.sentiment_evidence,
            cited_sources=cited_sources,
            risks_and_caveats=result.risks_and_caveats,
            contradictions=result.contradictions,
            revision_number=critique_count,
        )

        elapsed_ms = (time.time() - start) * 1000
        logger.info(
            "Draft thesis for %s: %s (confidence: %.0f) in %.0fms",
            ticker, draft.direction, draft.confidence_score, elapsed_ms,
        )

        return {
            "draft_thesis": draft.model_dump(),
            "run_log": [
                LogEntry(
                    node="synthesize_draft",
                    status="completed",
                    duration_ms=elapsed_ms,
                    message=(
                        f"Draft thesis for {ticker}: {draft.direction} "
                        f"(confidence {draft.confidence_score:.0f}/100, "
                        f"revision {critique_count})"
                    ),
                    details={
                        "direction": draft.direction,
                        "confidence": draft.confidence_score,
                        "revision": critique_count,
                        "evidence_count": (
                            len(draft.technical_evidence)
                            + len(draft.sentiment_evidence)
                        ),
                        "contradictions_count": len(draft.contradictions),
                    },
                ).model_dump()
            ],
        }

    except Exception as e:
        elapsed_ms = (time.time() - start) * 1000
        logger.error("Synthesis failed for %s: %s", ticker, e)

        # Fallback: produce a minimal draft so the graph can proceed
        fallback = DraftThesis(
            ticker=ticker,
            direction="neutral",
            confidence_score=10.0,
            summary=(
                f"Synthesis failed due to LLM error: {e}. "
                "This is a fallback thesis with minimal confidence."
            ),
            risks_and_caveats=[
                "LLM synthesis failed — this thesis is a placeholder",
                f"Error: {str(e)[:200]}",
            ],
            revision_number=critique_count,
        )

        return {
            "draft_thesis": fallback.model_dump(),
            "run_log": [
                LogEntry(
                    node="synthesize_draft",
                    status="failed",
                    duration_ms=elapsed_ms,
                    message=f"Synthesis failed for {ticker}: {e}",
                ).model_dump()
            ],
            "error_log": [f"[synthesize_draft] LLM call failed: {e}"],
        }
