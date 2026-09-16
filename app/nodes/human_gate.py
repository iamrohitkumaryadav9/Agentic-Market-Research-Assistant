"""
Node: human_approval_gate

Uses LangGraph's interrupt() to halt graph execution and wait for
explicit human sign-off before a thesis can be finalized.

This is NOT simulated with a sleep, flag, or polling loop. The graph
truly pauses via the checkpointer, and resumes only when the caller
invokes graph.invoke(Command(resume=...), config) with either
"approve" or "reject".

The conditional edge after this node (defined in graph.py) routes to
finalize or discard based on the human_decision value.
"""

from __future__ import annotations

import time

from langgraph.types import interrupt

from app.state import LogEntry, ResearchState


def human_approval_gate(state: ResearchState) -> dict:
    """
    Pauses execution via interrupt() and waits for a human decision.

    The interrupt() call:
    1. Serializes the current state to the checkpointer.
    2. Raises a GraphInterrupt exception, halting the graph.
    3. Returns the value passed to Command(resume=...) when the graph
       is re-invoked.

    Returns the human decision ("approve" or "reject") into state.
    """
    start = time.time()
    ticker = state["ticker"]
    draft = state.get("draft_thesis", {})

    # Emit a pre-interrupt log entry
    pre_log = LogEntry(
        node="human_approval_gate",
        status="waiting_for_human",
        message=(
            f"Thesis for {ticker} ready for review. "
            f"Direction: {draft.get('direction', 'N/A')}, "
            f"Confidence: {draft.get('confidence_score', 'N/A')}. "
            f"Graph execution paused — awaiting human decision."
        ),
    ).model_dump()

    # ⏸ THIS IS THE REAL PAUSE POINT ⏸
    # interrupt() halts the graph and persists state via the checkpointer.
    # It returns whatever value the human provides via Command(resume=...).
    human_decision = interrupt(
        {
            "message": "Please review the draft thesis and submit your decision.",
            "ticker": ticker,
            "draft_thesis": draft,
            "options": ["approve", "reject"],
        }
    )

    elapsed_ms = (time.time() - start) * 1000

    post_log = LogEntry(
        node="human_approval_gate",
        status="completed",
        duration_ms=elapsed_ms,
        message=f"Human decision received: {human_decision}",
    ).model_dump()

    return {
        "human_decision": human_decision,
        "run_log": [pre_log, post_log],
    }
