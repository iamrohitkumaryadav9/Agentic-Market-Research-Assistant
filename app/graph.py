"""
Graph definition for the Market Research Agent.

This is the core deliverable — a LangGraph StateGraph with:
  - 7 functional nodes + 1 discard node
  - 2 conditional edges (critique retry loop, human decision routing)
  - Real interrupt/resume via interrupt() + MemorySaver checkpointer
  - Full state typing via ResearchState TypedDict

Design rationale — why LangGraph over a plain chain:
  1. The critique → synthesize_draft retry loop is a *conditional cycle*,
     not expressible in a linear chain without ugly hacks.
  2. The human_approval_gate uses interrupt() for true execution pause/resume
     with checkpointed state — LangChain chains have no equivalent.
  3. StateGraph makes the execution topology explicit and inspectable,
     which matters for observability and debugging.
  4. The graph can be visualized (graph.get_graph().draw_mermaid()) and
     its execution traced node-by-node through LangSmith or the run log.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

from langgraph.graph import END, START, StateGraph

from app.config import get_checkpointer, get_settings
from app.nodes.critique import critique
from app.nodes.finalize import discard, finalize
from app.nodes.human_gate import human_approval_gate
from app.nodes.market_data import fetch_market_data
from app.nodes.news import fetch_news
from app.nodes.sentiment import analyze_sentiment
from app.nodes.synthesis import synthesize_draft
from app.state import ResearchState

logger = logging.getLogger("market_research_agent.graph")


# ---------------------------------------------------------------------------
# Conditional edge functions
# ---------------------------------------------------------------------------

def should_retry_or_proceed(state: ResearchState) -> str:
    """
    After the critique node, decide whether to:
      - "revise": send back to synthesize_draft (if critique rejected AND
        we haven't hit the max retry limit)
      - "proceed": advance to human_approval_gate (if critique approved
        OR we've exhausted retries)

    When max retries are exhausted without approval, the draft proceeds
    with a flagged low-confidence note — this is by design, to avoid
    infinite loops while still surfacing the critique concerns.
    """
    settings = get_settings()
    max_retries = settings.CRITIQUE_MAX_RETRIES
    critique_count = state.get("critique_count", 0)
    critique_notes = state.get("critique_notes", {})

    approved = critique_notes.get("approved", False) if critique_notes else False

    if approved:
        logger.info("Critique APPROVED — proceeding to human approval gate")
        return "proceed"

    if critique_count < max_retries:
        logger.info(
            "Critique REJECTED (attempt %d/%d) — sending back for revision",
            critique_count, max_retries,
        )
        return "revise"

    logger.warning(
        "Critique still not satisfied after %d retries — proceeding with "
        "low-confidence flag to human approval gate",
        max_retries,
    )
    return "proceed"


def route_human_decision(state: ResearchState) -> str:
    """
    After the human_approval_gate resumes, route based on the decision.
    """
    # The gate only returns a validated "approve"/"reject"; anything else
    # (should be impossible) is treated as a reject, never as an approval.
    decision = state.get("human_decision")

    if decision == "approve":
        logger.info("Human APPROVED thesis — routing to finalize")
        return "finalize"

    logger.info("Human REJECTED thesis — routing to discard")
    return "discard"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """
    Construct the StateGraph with all nodes and edges.
    Returns the uncompiled builder — call compile_graph() for the runnable.
    """
    builder = StateGraph(ResearchState)

    # --- Add nodes ---
    builder.add_node("fetch_market_data", fetch_market_data)
    builder.add_node("fetch_news", fetch_news)
    builder.add_node("analyze_sentiment", analyze_sentiment)
    builder.add_node("synthesize_draft", synthesize_draft)
    builder.add_node("critique", critique)
    builder.add_node("human_approval_gate", human_approval_gate)
    builder.add_node("finalize", finalize)
    builder.add_node("discard", discard)

    # --- Linear edges ---
    builder.add_edge(START, "fetch_market_data")
    builder.add_edge("fetch_market_data", "fetch_news")
    builder.add_edge("fetch_news", "analyze_sentiment")
    builder.add_edge("analyze_sentiment", "synthesize_draft")
    builder.add_edge("synthesize_draft", "critique")

    # --- Conditional edge: critique → revise or proceed ---
    builder.add_conditional_edges(
        "critique",
        should_retry_or_proceed,
        {
            "revise": "synthesize_draft",
            "proceed": "human_approval_gate",
        },
    )

    # --- Conditional edge: human gate → finalize or discard ---
    builder.add_conditional_edges(
        "human_approval_gate",
        route_human_decision,
        {
            "finalize": "finalize",
            "discard": "discard",
        },
    )

    # --- Terminal edges ---
    builder.add_edge("finalize", END)
    builder.add_edge("discard", END)

    return builder


def compile_graph(checkpointer=None):
    """
    Build and compile the graph with a checkpointer.

    The checkpointer is required for interrupt/resume to work.
    Defaults to the configured backend (MemorySaver for dev).
    """
    if checkpointer is None:
        checkpointer = get_checkpointer()

    builder = build_graph()
    graph = builder.compile(checkpointer=checkpointer)

    logger.info("Graph compiled successfully with checkpointer: %s", type(checkpointer).__name__)
    return graph


# ---------------------------------------------------------------------------
# Run trace persistence
# ---------------------------------------------------------------------------

def save_run_trace(state: ResearchState, thread_id: str) -> str:
    """
    Save the complete run trace (run_log + error_log) to a JSON file.

    Returns the path to the saved trace file.
    """
    settings = get_settings()
    traces_dir = settings.RUN_TRACES_DIR
    os.makedirs(traces_dir, exist_ok=True)

    trace_data = {
        "thread_id": thread_id,
        "ticker": state.get("ticker", "UNKNOWN"),
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "run_log": state.get("run_log", []),
        "error_log": state.get("error_log", []),
        "final_state": {
            "has_price_data": state.get("price_data") is not None,
            "has_news": bool(state.get("news_articles")),
            "has_sentiment": state.get("sentiment_summary") is not None,
            "has_draft": state.get("draft_thesis") is not None,
            "critique_count": state.get("critique_count", 0),
            "human_decision": state.get("human_decision"),
            "has_final_thesis": state.get("final_thesis") is not None,
        },
    }

    filename = f"trace_{state.get('ticker', 'UNKNOWN')}_{thread_id}.json"
    filepath = os.path.join(traces_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(trace_data, f, indent=2, default=str, ensure_ascii=False)

    logger.info("Run trace saved to %s", filepath)
    return filepath


# ---------------------------------------------------------------------------
# Convenience: get the default compiled graph
# ---------------------------------------------------------------------------

_default_graph = None


def get_graph():
    """Return a module-level singleton compiled graph."""
    global _default_graph
    if _default_graph is None:
        _default_graph = compile_graph()
    return _default_graph
