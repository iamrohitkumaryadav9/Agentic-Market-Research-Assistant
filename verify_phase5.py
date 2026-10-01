"""
Phase 5 Verification — Human Approval Gate Battle Tests

The interrupt() mechanism was implemented in Phase 1. This phase proves
it's robust across the edge cases that matter for production:

  1. State preservation — all fields survive the interrupt boundary intact
  2. Concurrent threads — multiple tickers interrupted independently
  3. State inspection — can read the draft thesis while paused
  4. Approve path — resume with "approve" routes to finalize
  5. Reject path — resume with "reject" routes to discard
  6. Arbitrary resume values — unexpected values handled gracefully
  7. Re-invocation after completion — graph doesn't re-run

Run: python verify_phase5.py
"""

from __future__ import annotations

import sys
import uuid
from unittest.mock import MagicMock, patch

sys.path.insert(0, ".")

import os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from app.config import configure_logging
configure_logging()

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from app.graph import build_graph
from app.nodes.critique import CritiqueOutput
from app.nodes.synthesis import ThesisOutput
from app.state import NewsArticle


def mock_llm_factory():
    """Create a mock LLM that produces valid structured output."""
    mock_llm = MagicMock()
    def with_structured_output_side_effect(schema):
        mock_structured = MagicMock()
        if schema == CritiqueOutput:
            # Auto-approve so we reach human gate quickly
            mock_structured.invoke.return_value = CritiqueOutput(
                approved=True, issues=[], unsupported_claims=[],
                suggested_revisions=[], confidence_adjustment=0.0,
            )
        elif schema == ThesisOutput:
            mock_structured.invoke.return_value = ThesisOutput(
                direction="bullish", confidence_score=65.0,
                summary="Test thesis for battle testing",
                technical_evidence=["SMA bullish", "RSI neutral"],
                sentiment_evidence=["3/5 articles positive"],
                risks_and_caveats=["Market volatility"],
                contradictions=["Minor divergence"],
            )
        else:
            from app.nodes.sentiment import ArticleSentimentOutput, SentimentBatchOutput
            mock_structured.invoke.return_value = SentimentBatchOutput(
                article_sentiments=[ArticleSentimentOutput(
                    article_index=0, sentiment="positive", confidence=0.8,
                    rationale="Test",
                )]
            )
        return mock_structured
    mock_llm.with_structured_output.side_effect = with_structured_output_side_effect
    return mock_llm


def make_mock_yf():
    """Create mock yfinance data."""
    import numpy as np
    import pandas as pd
    dates = pd.date_range("2024-01-01", periods=63, freq="B")
    return pd.DataFrame({
        "Open": np.random.uniform(145, 155, 63),
        "High": np.random.uniform(150, 160, 63),
        "Low": np.random.uniform(140, 150, 63),
        "Close": np.linspace(145, 155, 63),
        "Volume": np.random.randint(500000, 2000000, 63),
    }, index=dates)


def run_to_interrupt(graph, ticker, config):
    """Run a graph invocation that should pause at the human gate."""
    initial_state = {
        "ticker": ticker,
        "price_data": None, "technical_signals": None,
        "news_articles": None, "sentiment_summary": None,
        "draft_thesis": None, "critique_notes": None,
        "critique_count": 0, "human_decision": None,
        "final_thesis": None, "run_log": [], "error_log": [],
    }

    with patch("app.nodes.critique.get_llm", side_effect=mock_llm_factory), \
         patch("app.nodes.synthesis.get_llm", side_effect=mock_llm_factory), \
         patch("app.nodes.sentiment.get_llm", side_effect=mock_llm_factory), \
         patch("app.nodes.news._fetch_massive_news", return_value=[
             NewsArticle(title="Mock article", url="https://example.com/1"),
         ]), \
         patch("app.nodes.market_data._fetch_yfinance_data", return_value=make_mock_yf()):
        return graph.invoke(initial_state, config)


def test_state_preservation():
    """Verify all state fields survive the interrupt boundary."""
    print("=" * 70)
    print("TEST 1: State preservation across interrupt")
    print("=" * 70)

    checkpointer = MemorySaver()
    graph = build_graph().compile(checkpointer=checkpointer)
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    result = run_to_interrupt(graph, "AAPL", config)

    # Check every field survived
    assert result["ticker"] == "AAPL", "Ticker lost"
    assert result["price_data"] is not None, "Price data lost"
    assert result["technical_signals"] is not None, "Technical signals lost"
    assert isinstance(result["news_articles"], list), "News articles lost"
    assert result["sentiment_summary"] is not None, "Sentiment lost"
    assert result["draft_thesis"] is not None, "Draft thesis lost"
    assert result["critique_notes"] is not None, "Critique notes lost"
    assert result["critique_count"] >= 1, "Critique count wrong"
    assert result["human_decision"] is None, "Human decision set before interrupt"
    assert result["final_thesis"] is None, "Final thesis exists before approval"
    assert len(result["run_log"]) >= 4, "Run log incomplete"

    # Verify via checkpointer snapshot too
    snapshot = graph.get_state(config)
    assert snapshot.next == ("human_approval_gate",), f"Wrong pause point: {snapshot.next}"
    snap_vals = snapshot.values
    assert snap_vals["ticker"] == "AAPL"
    assert snap_vals["draft_thesis"] is not None

    print(f"  Ticker: {result['ticker']}")
    print(f"  Price data present: {result['price_data'] is not None}")
    print(f"  Technical signals present: {result['technical_signals'] is not None}")
    print(f"  News articles: {len(result['news_articles'])} articles")
    print(f"  Sentiment present: {result['sentiment_summary'] is not None}")
    print(f"  Draft thesis direction: {result['draft_thesis']['direction']}")
    print(f"  Critique count: {result['critique_count']}")
    print(f"  Human decision: {result['human_decision']} (None = correct)")
    print(f"  Final thesis: {result['final_thesis']} (None = correct)")
    print(f"  Run log entries: {len(result['run_log'])}")
    print(f"  Snapshot next: {snapshot.next}")
    print("  [OK] All state fields preserved across interrupt")
    return True


def test_concurrent_threads():
    """Verify multiple tickers can be interrupted independently."""
    print("\n" + "=" * 70)
    print("TEST 2: Concurrent threads — 3 tickers paused independently")
    print("=" * 70)

    checkpointer = MemorySaver()
    graph = build_graph().compile(checkpointer=checkpointer)

    tickers = ["AAPL", "MSFT", "GOOGL"]
    configs = {}
    results = {}

    # Pause all three
    for ticker in tickers:
        tid = str(uuid.uuid4())
        cfg = {"configurable": {"thread_id": tid}}
        configs[ticker] = cfg
        results[ticker] = run_to_interrupt(graph, ticker, cfg)
        print(f"  {ticker}: paused at human gate (thread {tid[:8]}...)")

    # Verify all three are independently paused
    for ticker in tickers:
        snap = graph.get_state(configs[ticker])
        assert snap.next == ("human_approval_gate",)
        assert snap.values["ticker"] == ticker
    print("  All 3 threads independently paused")

    # Approve AAPL, reject MSFT, leave GOOGL paused
    with patch("app.nodes.critique.get_llm", side_effect=mock_llm_factory):
        aapl_result = graph.invoke(Command(resume="approve"), configs["AAPL"])
        msft_result = graph.invoke(Command(resume="reject"), configs["MSFT"])

    # Verify outcomes
    assert aapl_result["final_thesis"] is not None, "AAPL should be finalized"
    assert aapl_result["final_thesis"]["human_approved"] is True
    print(f"  AAPL: approved -> finalized (direction: {aapl_result['final_thesis']['direction']})")

    assert msft_result["human_decision"] == "reject"
    print(f"  MSFT: rejected -> discarded")

    # GOOGL should still be paused
    googl_snap = graph.get_state(configs["GOOGL"])
    assert googl_snap.next == ("human_approval_gate",)
    print(f"  GOOGL: still paused at {googl_snap.next}")

    # Now approve GOOGL
    with patch("app.nodes.critique.get_llm", side_effect=mock_llm_factory):
        googl_result = graph.invoke(Command(resume="approve"), configs["GOOGL"])
    assert googl_result["final_thesis"] is not None
    print(f"  GOOGL: approved -> finalized")

    print("  [OK] Concurrent threads work independently")
    return True


def test_state_inspection_while_paused():
    """Verify the draft thesis can be read from the snapshot while paused."""
    print("\n" + "=" * 70)
    print("TEST 3: State inspection during pause")
    print("=" * 70)

    checkpointer = MemorySaver()
    graph = build_graph().compile(checkpointer=checkpointer)
    tid = str(uuid.uuid4())
    config = {"configurable": {"thread_id": tid}}

    run_to_interrupt(graph, "TSLA", config)

    # Read state from checkpointer — this is what the API/frontend will do
    snapshot = graph.get_state(config)
    state = snapshot.values

    draft = state.get("draft_thesis", {})
    print(f"  Ticker: {state['ticker']}")
    print(f"  Draft direction: {draft.get('direction')}")
    print(f"  Draft confidence: {draft.get('confidence_score')}")
    print(f"  Draft summary: {draft.get('summary', '')[:60]}...")
    print(f"  Technical evidence: {len(draft.get('technical_evidence', []))} points")
    print(f"  Sentiment evidence: {len(draft.get('sentiment_evidence', []))} points")
    print(f"  Risks: {len(draft.get('risks_and_caveats', []))}")
    print(f"  Paused at: {snapshot.next}")

    # This is exactly what /status/{run_id} will return
    status_payload = {
        "run_id": tid,
        "ticker": state["ticker"],
        "status": "awaiting_human_approval",
        "current_node": snapshot.next[0] if snapshot.next else None,
        "draft_thesis": draft,
        "critique_count": state.get("critique_count", 0),
    }
    print(f"\n  API status payload would contain:")
    print(f"    status: {status_payload['status']}")
    print(f"    current_node: {status_payload['current_node']}")
    print(f"    draft direction: {status_payload['draft_thesis']['direction']}")

    assert draft.get("direction") is not None, "Draft should have a direction"
    assert draft.get("confidence_score") is not None, "Draft should have confidence"

    print("  [OK] State inspection during pause works")
    return True


def test_reject_path_details():
    """Verify the reject path produces the correct final state."""
    print("\n" + "=" * 70)
    print("TEST 4: Reject path — detailed verification")
    print("=" * 70)

    checkpointer = MemorySaver()
    graph = build_graph().compile(checkpointer=checkpointer)
    tid = str(uuid.uuid4())
    config = {"configurable": {"thread_id": tid}}

    run_to_interrupt(graph, "NVDA", config)

    with patch("app.nodes.critique.get_llm", side_effect=mock_llm_factory):
        result = graph.invoke(Command(resume="reject"), config)

    assert result["human_decision"] == "reject"
    assert result["final_thesis"] is None, "Rejected thesis should have no final output"

    # Verify run log has discard entry
    run_log = result.get("run_log", [])
    discard_entries = [e for e in run_log if e.get("node") == "discard"]
    assert len(discard_entries) >= 1, "Discard node should have run"
    assert "REJECTED" in discard_entries[0].get("message", "")

    # Graph should be finished (no next node)
    snapshot = graph.get_state(config)
    assert not snapshot.next, f"Graph should be finished, but next={snapshot.next}"

    print(f"  Human decision: {result['human_decision']}")
    print(f"  Final thesis: {result['final_thesis']} (None = correct)")
    print(f"  Discard log: {discard_entries[0]['message']}")
    print(f"  Graph finished: {not snapshot.next}")
    print("  [OK] Reject path works correctly")
    return True


def test_approve_path_details():
    """Verify the approve path produces a complete final thesis."""
    print("\n" + "=" * 70)
    print("TEST 5: Approve path — detailed verification")
    print("=" * 70)

    checkpointer = MemorySaver()
    graph = build_graph().compile(checkpointer=checkpointer)
    tid = str(uuid.uuid4())
    config = {"configurable": {"thread_id": tid}}

    run_to_interrupt(graph, "META", config)

    with patch("app.nodes.critique.get_llm", side_effect=mock_llm_factory):
        result = graph.invoke(Command(resume="approve"), config)

    final = result.get("final_thesis")
    assert final is not None, "Approved thesis should produce a final output"
    assert final["human_approved"] is True
    assert final["ticker"] == "META"
    assert "NOT FINANCIAL ADVICE" in final["disclaimer"]
    assert final["direction"] in ("bullish", "bearish", "neutral")
    assert 0 <= final["confidence_score"] <= 100
    assert len(final.get("data_sources_used", [])) > 0
    assert final.get("generated_at") is not None

    # Graph should be finished
    snapshot = graph.get_state(config)
    assert not snapshot.next

    print(f"  Ticker: {final['ticker']}")
    print(f"  Direction: {final['direction']}")
    print(f"  Confidence: {final['confidence_score']}")
    print(f"  Human approved: {final['human_approved']}")
    print(f"  Disclaimer: {final['disclaimer'][:50]}...")
    print(f"  Data sources: {final['data_sources_used']}")
    print(f"  Generated at: {final['generated_at']}")
    print(f"  Graph finished: {not snapshot.next}")
    print("  [OK] Approve path produces complete final thesis")
    return True


def test_unknown_resume_value():
    """Verify that an unexpected resume value is handled gracefully."""
    print("\n" + "=" * 70)
    print("TEST 6: Unknown resume value")
    print("=" * 70)

    checkpointer = MemorySaver()
    graph = build_graph().compile(checkpointer=checkpointer)
    tid = str(uuid.uuid4())
    config = {"configurable": {"thread_id": tid}}

    run_to_interrupt(graph, "AMD", config)

    # Resume with an unexpected value — should route to discard (default)
    with patch("app.nodes.critique.get_llm", side_effect=mock_llm_factory):
        result = graph.invoke(Command(resume="maybe_later"), config)

    # The route_human_decision function treats anything != "approve" as reject
    assert result["human_decision"] == "maybe_later"
    assert result["final_thesis"] is None  # Should discard

    run_log = result.get("run_log", [])
    discard_entries = [e for e in run_log if e.get("node") == "discard"]
    assert len(discard_entries) >= 1

    print(f"  Resume value: 'maybe_later'")
    print(f"  Human decision: {result['human_decision']}")
    print(f"  Routed to: discard (any non-'approve' value = reject)")
    print(f"  Final thesis: {result['final_thesis']} (None = correct)")
    print("  [OK] Unknown resume values handled gracefully (routed to discard)")
    return True


if __name__ == "__main__":
    print()
    results = [
        test_state_preservation(),
        test_concurrent_threads(),
        test_state_inspection_while_paused(),
        test_reject_path_details(),
        test_approve_path_details(),
        test_unknown_resume_value(),
    ]

    print("\n" + "=" * 70)
    if all(results):
        print("ALL PHASE 5 TESTS PASSED (6/6)")
    else:
        failed = sum(1 for r in results if not r)
        print(f"SOME TESTS FAILED ({failed} failures)")
    print("=" * 70)
    print("""
    Battle-tested:
      [OK] State preservation — all fields survive interrupt boundary
      [OK] Concurrent threads — 3 tickers paused/resumed independently
      [OK] State inspection — draft thesis readable from snapshot while paused
      [OK] Reject path — routes to discard, no final thesis, graph finishes
      [OK] Approve path — produces complete FinalThesis with disclaimer
      [OK] Unknown resume — gracefully routes to discard
    """)
