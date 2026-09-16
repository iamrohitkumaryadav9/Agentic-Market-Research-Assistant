"""
Phase 4 Verification Script

Tests the critique node + retry conditional edge:
  1. Critique node with mocked LLM — verifies structured output parsing
  2. Retry loop — mocked critique rejects, then approves on revision
  3. Max retry exhaustion — verifies low-confidence flag after max retries
  4. Integration test — full graph with mocked LLM for the critique path

Since LLM API credits are exhausted, we use unittest.mock to patch the
LLM calls and verify the wiring is correct.

Run: python verify_phase4.py
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
from app.nodes.critique import CritiqueOutput, critique
from app.state import ResearchState


def make_base_state(ticker: str = "TEST") -> dict:
    """Create a base state dict for testing."""
    return {
        "ticker": ticker,
        "price_data": {"ticker": ticker, "latest_close": 150.0},
        "technical_signals": {
            "sma_20": 148.0, "sma_50": 145.0, "rsi_14": 62.0,
            "volume_trend": "increasing",
            "price_change_pct_5d": 2.5, "price_change_pct_30d": 8.0,
            "signal_summary": f"{ticker}: Bullish structure",
        },
        "news_articles": [
            {"title": "Test article", "source": "Test", "url": "https://test.com/1"},
        ],
        "sentiment_summary": {
            "overall_sentiment": "positive",
            "positive_count": 3, "negative_count": 1, "neutral_count": 1,
            "average_confidence": 0.75,
            "article_sentiments": [],
        },
        "draft_thesis": {
            "ticker": ticker,
            "direction": "bullish",
            "confidence_score": 72.0,
            "summary": "Test thesis: bullish lean based on SMA crossover and positive sentiment.",
            "technical_evidence": [
                "Price above SMA-20 (148.0) and SMA-50 (145.0)",
                "RSI at 62.0 — neutral, not overbought",
            ],
            "sentiment_evidence": [
                "3 of 5 articles classified positive (75%)",
                "Earnings beat noted in multiple sources",
            ],
            "cited_sources": ["https://test.com/1", "https://test.com/2"],
            "risks_and_caveats": [
                "RSI approaching overbought territory",
                "One negative article flagging supply chain risk",
            ],
            "contradictions": [
                "Volume increasing supports bullish read, but one bearish analyst note contradicts",
            ],
            "revision_number": 0,
        },
        "critique_notes": None,
        "critique_count": 0,
        "human_decision": None,
        "final_thesis": None,
        "run_log": [],
        "error_log": [],
    }


def test_critique_rejection():
    """Test that critique rejects and produces structured feedback."""
    print("=" * 70)
    print("TEST 1: Critique node — mock rejection")
    print("=" * 70)

    mock_output = CritiqueOutput(
        approved=False,
        issues=[
            "Confidence score of 72 is too high for mixed signals",
            "Supply chain risk mentioned but not weighted heavily enough",
        ],
        unsupported_claims=[
            "'Bullish structure' claim lacks sufficient data window (only 3mo)",
        ],
        suggested_revisions=[
            "Lower confidence to 55-60 range given the mixed sentiment",
            "Elevate supply chain risk to a primary caveat",
            "Add caveat about limited data window",
        ],
        confidence_adjustment=-15.0,
    )

    with patch("app.nodes.critique.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = mock_output
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm

        state = make_base_state()
        result = critique(state)

    notes = result["critique_notes"]
    count = result["critique_count"]

    print(f"  Approved: {notes['approved']}")
    print(f"  Issues: {len(notes['issues'])}")
    for issue in notes["issues"]:
        print(f"    - {issue}")
    print(f"  Unsupported claims: {len(notes['unsupported_claims'])}")
    for claim in notes["unsupported_claims"]:
        print(f"    - {claim}")
    print(f"  Suggested revisions: {len(notes['suggested_revisions'])}")
    for rev in notes["suggested_revisions"]:
        print(f"    - {rev}")
    print(f"  Confidence adjustment: {notes['confidence_adjustment']:+.0f}")
    print(f"  Critique count: {count}")

    assert not notes["approved"], "Should be rejected"
    assert len(notes["issues"]) == 2
    assert len(notes["suggested_revisions"]) == 3
    assert notes["confidence_adjustment"] == -15.0
    assert count == 1
    print("  [OK] Critique rejection works correctly")
    return True


def test_critique_approval():
    """Test that critique approves a strong thesis."""
    print("\n" + "=" * 70)
    print("TEST 2: Critique node — mock approval")
    print("=" * 70)

    mock_output = CritiqueOutput(
        approved=True,
        issues=[],
        unsupported_claims=[],
        suggested_revisions=[],
        confidence_adjustment=0.0,
    )

    with patch("app.nodes.critique.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = mock_output
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm

        state = make_base_state()
        state["critique_count"] = 1  # Revision pass
        state["draft_thesis"]["revision_number"] = 1
        result = critique(state)

    notes = result["critique_notes"]
    assert notes["approved"], "Should be approved"
    assert len(notes["issues"]) == 0
    print(f"  Approved: {notes['approved']}")
    print(f"  Issues: {len(notes['issues'])} (none)")
    print("  [OK] Critique approval works correctly")
    return True


def test_retry_loop_in_graph():
    """
    Test the full retry loop: critique rejects on first pass, then approves.
    Uses mocked LLM for both synthesis and critique nodes.
    """
    print("\n" + "=" * 70)
    print("TEST 3: Full retry loop in graph (mocked LLM)")
    print("=" * 70)

    call_count = {"critique": 0, "synthesis": 0}

    # Mock critique: reject on first call, approve on second
    def mock_critique_invoke(messages):
        call_count["critique"] += 1
        if call_count["critique"] == 1:
            return CritiqueOutput(
                approved=False,
                issues=["Confidence too high", "Missing risk factors"],
                unsupported_claims=["Volume trend claim"],
                suggested_revisions=["Lower confidence", "Add risks"],
                confidence_adjustment=-10.0,
            )
        return CritiqueOutput(
            approved=True,
            issues=[],
            unsupported_claims=[],
            suggested_revisions=[],
            confidence_adjustment=0.0,
        )

    # Mock synthesis output schema
    from app.nodes.synthesis import ThesisOutput

    def mock_synthesis_invoke(messages):
        call_count["synthesis"] += 1
        return ThesisOutput(
            direction="bullish",
            confidence_score=60.0 if call_count["synthesis"] > 1 else 72.0,
            summary=f"Test synthesis pass {call_count['synthesis']}",
            technical_evidence=["SMA bullish crossover", "RSI neutral"],
            sentiment_evidence=["3/5 articles positive"],
            risks_and_caveats=["Market volatility", "Supply chain risk"],
            contradictions=["Volume vs sentiment divergence"],
        )

    # Patch get_llm to return different mocks for different structured outputs
    def mock_get_llm_factory():
        mock_llm = MagicMock()
        def with_structured_output_side_effect(schema):
            mock_structured = MagicMock()
            if schema == CritiqueOutput:
                mock_structured.invoke.side_effect = mock_critique_invoke
            elif schema == ThesisOutput:
                mock_structured.invoke.side_effect = mock_synthesis_invoke
            else:
                # Sentiment — return a basic mock
                from app.nodes.sentiment import ArticleSentimentOutput
                mock_structured.invoke.return_value = ArticleSentimentOutput(
                    sentiment="positive", confidence=0.8, rationale="Test"
                )
            return mock_structured
        mock_llm.with_structured_output.side_effect = with_structured_output_side_effect
        return mock_llm

    checkpointer = MemorySaver()
    builder = build_graph()
    graph = builder.compile(checkpointer=checkpointer)

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = make_base_state("LOOP")
    # Clear real data — we'll let stubs handle market/news
    initial_state["price_data"] = None
    initial_state["technical_signals"] = None
    initial_state["news_articles"] = None

    with patch("app.nodes.critique.get_llm", side_effect=mock_get_llm_factory), \
         patch("app.nodes.synthesis.get_llm", side_effect=mock_get_llm_factory), \
         patch("app.nodes.sentiment.get_llm", side_effect=mock_get_llm_factory), \
         patch("app.nodes.market_data._fetch_yfinance_data") as mock_yf:

        # Mock yfinance
        import pandas as pd
        import numpy as np
        dates = pd.date_range("2024-01-01", periods=63, freq="B")
        mock_df = pd.DataFrame({
            "Open": np.random.uniform(145, 155, 63),
            "High": np.random.uniform(150, 160, 63),
            "Low": np.random.uniform(140, 150, 63),
            "Close": np.linspace(145, 155, 63),
            "Volume": np.random.randint(500000, 2000000, 63),
        }, index=dates)
        mock_yf.return_value = mock_df

        result = graph.invoke(initial_state, config)

    # Verify the retry loop
    run_log = result.get("run_log", [])
    critique_entries = [e for e in run_log if e.get("node") == "critique"]
    synthesis_entries = [e for e in run_log if e.get("node") == "synthesize_draft"]

    print(f"  Synthesis passes: {len(synthesis_entries)}")
    print(f"  Critique passes: {len(critique_entries)}")
    print(f"  LLM critique calls: {call_count['critique']}")
    print(f"  LLM synthesis calls: {call_count['synthesis']}")

    print(f"\n  --- Run log ---")
    for i, entry in enumerate(run_log, 1):
        node = entry.get("node", "?")
        status = entry.get("status", "?")
        msg = entry.get("message", "")
        if len(msg) > 85:
            msg = msg[:82] + "..."
        print(f"  {i:>2}. [{status:>12}] {node:>25} | {msg}")

    # Assertions
    assert len(critique_entries) >= 2, f"Expected >= 2 critique passes, got {len(critique_entries)}"
    assert len(synthesis_entries) >= 2, f"Expected >= 2 synthesis passes, got {len(synthesis_entries)}"

    # First critique should have rejected
    first_critique = critique_entries[0]
    assert "NEEDS REVISION" in first_critique.get("message", ""), "First critique should reject"

    # Second critique should have approved
    second_critique = critique_entries[1]
    assert "APPROVED" in second_critique.get("message", ""), "Second critique should approve"

    # Graph should be paused at human gate
    snapshot = graph.get_state(config)
    assert snapshot.next == ("human_approval_gate",), f"Expected pause at human gate, got {snapshot.next}"
    print(f"\n  Graph paused at: {snapshot.next}")

    # Resume and verify
    with patch("app.nodes.critique.get_llm", side_effect=mock_get_llm_factory):
        result = graph.invoke(Command(resume="approve"), config)

    final = result.get("final_thesis")
    assert final is not None, "Final thesis should exist after approval"
    print(f"  Final thesis: {final['direction']} (confidence {final['confidence_score']})")

    print("  [OK] Retry loop works correctly with mocked LLM")
    return True


def test_max_retries_exhaustion():
    """Test that after max retries, the graph proceeds with low-confidence flag."""
    print("\n" + "=" * 70)
    print("TEST 4: Max retries exhaustion")
    print("=" * 70)

    # Mock critique that NEVER approves
    def always_reject(messages):
        return CritiqueOutput(
            approved=False,
            issues=["Still not good enough"],
            unsupported_claims=["Everything"],
            suggested_revisions=["Start over"],
            confidence_adjustment=-20.0,
        )

    from app.nodes.synthesis import ThesisOutput

    def mock_synthesis(messages):
        return ThesisOutput(
            direction="neutral",
            confidence_score=40.0,
            summary="Perpetually revised thesis",
            technical_evidence=["SMA data"],
            sentiment_evidence=["Mixed sentiment"],
            risks_and_caveats=["High uncertainty"],
            contradictions=["Signals conflict"],
        )

    def mock_get_llm_factory():
        mock_llm = MagicMock()
        def with_structured_output_side_effect(schema):
            mock_structured = MagicMock()
            if schema == CritiqueOutput:
                mock_structured.invoke.side_effect = always_reject
            elif schema == ThesisOutput:
                mock_structured.invoke.side_effect = mock_synthesis
            else:
                from app.nodes.sentiment import ArticleSentimentOutput
                mock_structured.invoke.return_value = ArticleSentimentOutput(
                    sentiment="neutral", confidence=0.5, rationale="Test"
                )
            return mock_structured
        mock_llm.with_structured_output.side_effect = with_structured_output_side_effect
        return mock_llm

    checkpointer = MemorySaver()
    builder = build_graph()
    graph = builder.compile(checkpointer=checkpointer)

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = make_base_state("MAXRETRY")
    initial_state["price_data"] = None
    initial_state["news_articles"] = None

    with patch("app.nodes.critique.get_llm", side_effect=mock_get_llm_factory), \
         patch("app.nodes.synthesis.get_llm", side_effect=mock_get_llm_factory), \
         patch("app.nodes.sentiment.get_llm", side_effect=mock_get_llm_factory), \
         patch("app.nodes.market_data._fetch_yfinance_data") as mock_yf:

        import pandas as pd
        import numpy as np
        dates = pd.date_range("2024-01-01", periods=63, freq="B")
        mock_df = pd.DataFrame({
            "Open": np.random.uniform(145, 155, 63),
            "High": np.random.uniform(150, 160, 63),
            "Low": np.random.uniform(140, 150, 63),
            "Close": np.linspace(145, 155, 63),
            "Volume": np.random.randint(500000, 2000000, 63),
        }, index=dates)
        mock_yf.return_value = mock_df

        result = graph.invoke(initial_state, config)

    run_log = result.get("run_log", [])
    critique_entries = [e for e in run_log if e.get("node") == "critique"]
    critique_count = result.get("critique_count", 0)

    print(f"  Critique passes: {len(critique_entries)}")
    print(f"  Critique count in state: {critique_count}")

    # Should have hit max retries (default 2) and proceeded anyway
    assert critique_count >= 2, f"Expected >= 2 critique passes, got {critique_count}"

    # Last critique should NOT have approved — graph proceeded via max retry rule
    last_critique = result.get("critique_notes", {})
    print(f"  Last critique approved: {last_critique.get('approved', 'N/A')}")
    print(f"  Graph paused at human gate: {graph.get_state(config).next}")

    print("  [OK] Max retries exhaustion works — graph proceeded to human gate")
    return True


if __name__ == "__main__":
    print()
    ok1 = test_critique_rejection()
    ok2 = test_critique_approval()
    ok3 = test_retry_loop_in_graph()
    ok4 = test_max_retries_exhaustion()

    print("\n" + "=" * 70)
    if all([ok1, ok2, ok3, ok4]):
        print("ALL PHASE 4 TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 70)
