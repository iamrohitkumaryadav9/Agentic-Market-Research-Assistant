"""
FastAPI wrapper for the Market Research Agent.

Endpoints:
  POST /run              — Start a new research run for a ticker
  GET  /status/{run_id}  — Check run progress (which node is running)
  POST /approve/{run_id} — Submit human approval/rejection decision
  GET  /thesis/{run_id}  — Fetch the final thesis (after approval)

The graph runs in background threads. The checkpointer (MemorySaver)
persists state across the interrupt/resume cycle.
"""

from __future__ import annotations

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langgraph.types import Command
from pydantic import BaseModel, Field

from app.config import configure_logging, get_checkpointer
from app.graph import build_graph, save_run_trace
from app.guardrails import validate_ticker

configure_logging()
logger = logging.getLogger("market_research_agent.api")

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Market Research Agent",
    description=(
        "Agentic market research assistant using LangGraph. "
        "Produces analyst-grade research theses on stocks. "
        "NOT a trading system — all output is for research purposes only."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared checkpointer and graph — singleton for the process
_checkpointer = get_checkpointer()
_graph = build_graph().compile(checkpointer=_checkpointer)
_executor = ThreadPoolExecutor(max_workers=4)

# In-memory run tracking (production: use a database)
_runs: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class RunRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=10, description="Stock ticker symbol")


class RunResponse(BaseModel):
    run_id: str
    ticker: str
    status: str
    message: str


class ApproveRequest(BaseModel):
    decision: str = Field(
        ...,
        description="Human decision: 'approve' or 'reject'",
        pattern="^(approve|reject)$",
    )
    comment: Optional[str] = Field(None, description="Optional reviewer comment")


class StatusResponse(BaseModel):
    run_id: str
    ticker: str
    status: str
    degraded: bool = False
    degradation_reasons: list[str] = Field(default_factory=list)
    current_node: Optional[str] = None
    draft_thesis: Optional[dict] = None
    critique_count: int = 0
    run_log: list = []
    error_log: list = []


class ThesisResponse(BaseModel):
    run_id: str
    ticker: str
    status: str
    final_thesis: Optional[dict] = None


# ---------------------------------------------------------------------------
# Background run execution
# ---------------------------------------------------------------------------

def _run_graph(run_id: str, ticker: str):
    """Execute the graph in a background thread."""
    try:
        _runs[run_id]["status"] = "running"
        config = {"configurable": {"thread_id": run_id}}

        initial_state = {
            "ticker": ticker,
            "price_data": None,
            "technical_signals": None,
            "news_articles": None,
            "sentiment_summary": None,
            "draft_thesis": None,
            "critique_notes": None,
            "critique_count": 0,
            "human_decision": None,
            "final_thesis": None,
            "run_log": [],
            "error_log": [],
        }

        logger.info("Starting graph run %s for %s", run_id, ticker)
        _graph.invoke(initial_state, config)

        # If we get here, the graph is paused at interrupt (human gate)
        _runs[run_id]["status"] = "awaiting_approval"
        logger.info("Run %s paused at human approval gate", run_id)

    except Exception as e:
        logger.error("Run %s failed: %s", run_id, e)
        _runs[run_id]["status"] = "failed"
        _runs[run_id]["error"] = str(e)

def _degradation_details(state: dict) -> tuple[bool, list[str]]:
    """Summarize failed steps and error-log entries for API/UI consumers."""
    reasons = [str(error) for error in state.get("error_log", []) if error]
    for entry in state.get("run_log", []):
        if entry.get("status") in {"failed", "degraded"} and entry.get("message"):
            reasons.append(str(entry["message"]))

    unique_reasons = list(dict.fromkeys(reasons))
    return bool(unique_reasons), [reason[:500] for reason in unique_reasons[:5]]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/run", response_model=RunResponse)
def start_run(req: RunRequest):
    """Start a new research run for a ticker."""
    ticker = req.ticker.upper().strip()

    # Guardrail check (also rejects anything that doesn't look like a ticker)
    is_safe, refusal = validate_ticker(ticker)
    if not is_safe:
        raise HTTPException(status_code=400, detail=refusal)

    run_id = str(uuid.uuid4())
    _runs[run_id] = {
        "ticker": ticker,
        "status": "starting",
        "error": None,
    }

    # Launch graph in background
    _executor.submit(_run_graph, run_id, ticker)

    return RunResponse(
        run_id=run_id,
        ticker=ticker,
        status="starting",
        message=f"Research run started for {ticker}. Use /status/{run_id} to track progress.",
    )


@app.get("/status/{run_id}", response_model=StatusResponse)
def get_status(run_id: str):
    """Check the current status of a research run."""
    if run_id not in _runs:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    run_info = _runs[run_id]
    config = {"configurable": {"thread_id": run_id}}

    try:
        snapshot = _graph.get_state(config)
        state = snapshot.values or {}

        # Determine current node
        current_node = None
        if snapshot.next:
            current_node = snapshot.next[0]

        # Determine status
        status = run_info["status"]
        if current_node == "human_approval_gate":
            status = "awaiting_approval"
        elif not snapshot.next and state.get("final_thesis"):
            status = "completed"
        elif not snapshot.next and state.get("human_decision") == "reject":
            status = "rejected"

        degraded, degradation_reasons = _degradation_details(state)

        return StatusResponse(
            run_id=run_id,
            ticker=run_info["ticker"],
            status=status,
            degraded=degraded,
            degradation_reasons=degradation_reasons,
            current_node=current_node,
            draft_thesis=state.get("draft_thesis"),
            critique_count=state.get("critique_count", 0),
            run_log=state.get("run_log", []),
            error_log=state.get("error_log", []),
        )
    except Exception:
        return StatusResponse(
            run_id=run_id,
            ticker=run_info["ticker"],
            status=run_info["status"],
        )


@app.post("/approve/{run_id}", response_model=RunResponse)
def submit_approval(run_id: str, req: ApproveRequest):
    """Submit a human approval or rejection decision for a paused run."""
    if run_id not in _runs:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    run_info = _runs[run_id]
    config = {"configurable": {"thread_id": run_id}}

    # Verify the graph is actually paused at the human gate
    try:
        snapshot = _graph.get_state(config)
        if not snapshot.next or "human_approval_gate" not in snapshot.next:
            raise HTTPException(
                status_code=409,
                detail=f"Run {run_id} is not awaiting approval (status: {run_info['status']})",
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to check run state: {e}")

    # A placeholder draft (synthesis failed) is not a research thesis: never approvable.
    if req.decision == "approve" and any(
        e.get("node") == "synthesize_draft" and e.get("status") == "failed"
        for e in (snapshot.values or {}).get("run_log", [])
    ):
        raise HTTPException(
            status_code=409,
            detail="Synthesis failed; the draft is a placeholder and cannot be approved.",
        )

    # Resume the graph with the human decision
    try:
        logger.info("Resuming run %s with decision: %s", run_id, req.decision)
        result = _graph.invoke(Command(resume=req.decision), config)

        if req.decision == "approve":
            _runs[run_id]["status"] = "completed"
            # Save the run trace
            try:
                save_run_trace(result, run_id)
            except Exception as e:
                logger.warning("Failed to save run trace: %s", e)
        else:
            _runs[run_id]["status"] = "rejected"

        return RunResponse(
            run_id=run_id,
            ticker=run_info["ticker"],
            status=_runs[run_id]["status"],
            message=(
                f"Thesis {'approved and finalized' if req.decision == 'approve' else 'rejected'}. "
                f"{'Use /thesis/' + run_id + ' to view.' if req.decision == 'approve' else ''}"
            ),
        )

    except Exception as e:
        logger.error("Failed to resume run %s: %s", run_id, e)
        raise HTTPException(status_code=500, detail=f"Failed to resume: {e}")


@app.get("/thesis/{run_id}", response_model=ThesisResponse)
def get_thesis(run_id: str):
    """Fetch the final thesis for a completed run."""
    if run_id not in _runs:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    config = {"configurable": {"thread_id": run_id}}

    try:
        snapshot = _graph.get_state(config)
        state = snapshot.values or {}
        final_thesis = state.get("final_thesis")

        if not final_thesis:
            status = _runs[run_id].get("status", "unknown")
            raise HTTPException(
                status_code=404,
                detail=f"No final thesis for run {run_id} (status: {status})",
            )

        return ThesisResponse(
            run_id=run_id,
            ticker=_runs[run_id]["ticker"],
            status="completed",
            final_thesis=final_thesis,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch thesis: {e}")


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "market-research-agent"}
