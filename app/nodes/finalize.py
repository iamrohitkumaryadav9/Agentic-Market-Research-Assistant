"""
Node: finalize + discard

Terminal nodes of the research graph.

finalize: Formats the approved draft into a FinalThesis with all citations,
confidence score, critique history, and the baked-in "NOT FINANCIAL ADVICE"
disclaimer. The disclaimer is part of the FinalThesis Pydantic model itself,
not just a display concern.

discard: Marks the thesis as rejected and records the decision.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

from app.state import FinalThesis, LogEntry, ResearchState


def finalize(state: ResearchState) -> dict:
    """
    Produces the final thesis output after human approval.

    The disclaimer is baked into the FinalThesis model default and
    cannot be removed without modifying the schema — this is intentional.
    """
    start = time.time()
    ticker = state["ticker"]
    draft = state.get("draft_thesis", {})
    critique_notes = state.get("critique_notes")

    # Build critique history from the run log
    critique_history = []
    if critique_notes:
        critique_history.append(critique_notes)

    # Determine which data sources were used vs failed
    data_sources_used = []
    data_sources_failed = []

    if state.get("price_data"):
        data_sources_used.append("yfinance (price/technicals)")
    else:
        data_sources_failed.append("yfinance (price/technicals)")

    if state.get("news_articles"):
        data_sources_used.append("Massive.com (news)")
    else:
        data_sources_failed.append("Massive.com (news)")

    final = FinalThesis(
        ticker=ticker,
        direction=draft.get("direction", "neutral"),
        confidence_score=draft.get("confidence_score", 0.0),
        summary=draft.get("summary", ""),
        technical_evidence=draft.get("technical_evidence", []),
        sentiment_evidence=draft.get("sentiment_evidence", []),
        cited_sources=draft.get("cited_sources", []),
        risks_and_caveats=draft.get("risks_and_caveats", []),
        contradictions=draft.get("contradictions", []),
        critique_history=critique_history,
        human_approved=True,
        generated_at=datetime.now(tz=timezone.utc).isoformat(),
        data_sources_used=data_sources_used,
        data_sources_failed=data_sources_failed,
        # disclaimer is set by FinalThesis model default — always present
    )

    elapsed_ms = (time.time() - start) * 1000

    return {
        "final_thesis": final.model_dump(),
        "run_log": [
            LogEntry(
                node="finalize",
                status="completed",
                duration_ms=elapsed_ms,
                message=f"Finalized thesis for {ticker} (human approved)",
            ).model_dump()
        ],
    }


def discard(state: ResearchState) -> dict:
    """
    Marks the thesis as rejected. No final thesis is produced.
    """
    start = time.time()
    ticker = state["ticker"]
    elapsed_ms = (time.time() - start) * 1000

    return {
        "final_thesis": None,
        "run_log": [
            LogEntry(
                node="discard",
                status="completed",
                duration_ms=elapsed_ms,
                message=f"Thesis for {ticker} was REJECTED by human reviewer",
            ).model_dump()
        ],
    }
