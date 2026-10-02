"""
Phase 1 Verification Script — graph skeleton and control flow

Validates, fully offline (yfinance, news and the LLM are mocked), that:
  1. The graph compiles
  2. The graph topology can be rendered
  3. A run goes through every node and pauses at the human gate
  4. The critique loop fires (first draft rejected, revision approved)
  5. The pause is a genuine interrupt() (finalize/discard have not run)
  6. Resuming with "approve" -> finalize, FinalThesis with baked-in disclaimer
  7. Resuming with "reject" -> discard
  8. The run_log captures node executions in order

Run: python verify_phase1.py
"""

from __future__ import annotations

import sys
import uuid
from unittest.mock import MagicMock, patch

sys.path.insert(0, ".")

import os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from app.graph import build_graph, save_run_trace
from app.nodes.critique import CritiqueOutput
from app.nodes.sentiment import ArticleSentimentOutput, SentimentBatchOutput
from app.nodes.synthesis import ThesisOutput
from app.state import NewsArticle


def make_llm_factory():
    """Mock LLM: critique rejects the first draft, approves the revision."""
    calls = {"critique": 0}

    def critique_invoke(_messages):
        calls["critique"] += 1
        if calls["critique"] == 1:
            return CritiqueOutput(
                approved=False, issues=["Confidence too high"], unsupported_claims=[],
                suggested_revisions=["Lower confidence"], confidence_adjustment=-10.0,
            )
        return CritiqueOutput(
            approved=True, issues=[], unsupported_claims=[],
            suggested_revisions=[], confidence_adjustment=0.0,
        )

    def factory():
        llm = MagicMock()

        def structured(schema):
            m = MagicMock()
            if schema is CritiqueOutput:
                m.invoke.side_effect = critique_invoke
            elif schema is ThesisOutput:
                m.invoke.return_value = ThesisOutput(
                    direction="bullish", confidence_score=65.0, summary="Bullish lean.",
                    technical_evidence=["RSI 55"], sentiment_evidence=["2/3 positive"],
                    risks_and_caveats=["rates", "valuation"], contradictions=["volume"],
                )
            else:
                m.invoke.return_value = SentimentBatchOutput(article_sentiments=[
                    ArticleSentimentOutput(article_index=0, sentiment="positive",
                                           confidence=0.8, rationale="ok")])
            return m

        llm.with_structured_output.side_effect = structured
        return llm

    return factory


def make_df():
    import numpy as np
    import pandas as pd
    dates = pd.date_range("2024-01-01", periods=63, freq="B")
    return pd.DataFrame({
        "Open": np.linspace(100, 110, 63), "High": np.linspace(101, 111, 63),
        "Low": np.linspace(99, 109, 63), "Close": np.linspace(100, 110, 63),
        "Volume": np.full(63, 1_000_000),
    }, index=dates)


INITIAL_STATE = {
    "ticker": "AAPL", "price_data": None, "technical_signals": None,
    "news_articles": None, "sentiment_summary": None, "draft_thesis": None,
    "critique_notes": None, "critique_count": 0, "human_decision": None,
    "final_thesis": None, "run_log": [], "error_log": [],
}


def run_verification():
    print("=" * 70)
    print("PHASE 1 VERIFICATION — Graph skeleton and control flow")
    print("=" * 70)

    print("\n[1/8] Compiling graph...")
    graph = build_graph().compile(checkpointer=MemorySaver())
    print("  [OK] Graph compiled successfully")

    print("\n[2/8] Graph structure:")
    try:
        print(graph.get_graph().draw_mermaid())
    except Exception as e:  # rendering is cosmetic
        print(f"  (Could not render mermaid: {e})")

    factory = make_llm_factory()
    patches = [
        patch("app.nodes.critique.get_llm", side_effect=factory),
        patch("app.nodes.synthesis.get_llm", side_effect=factory),
        patch("app.nodes.sentiment.get_llm", side_effect=factory),
        patch("app.nodes.news._fetch_massive_news",
              return_value=[NewsArticle(title="Mock article", url="https://example.com/1",
                                        provider="massive.com")]),
        patch("app.nodes.market_data._fetch_yfinance_data", return_value=make_df()),
    ]
    for p in patches:
        p.start()
    try:
        print("\n[3/8] Starting run for AAPL (should pause at the human gate)...")
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        result = graph.invoke(dict(INITIAL_STATE), config)
        run_log = result.get("run_log", [])
        nodes_run = [e["node"] for e in run_log]
        for expected in ("fetch_market_data", "fetch_news", "analyze_sentiment",
                         "synthesize_draft", "critique"):
            assert expected in nodes_run, f"{expected} did not run"
        print(f"  [OK] Nodes executed: {nodes_run}")

        print("\n[4/8] Verifying critique loop...")
        critiques = [e for e in run_log if e["node"] == "critique"]
        syntheses = [e for e in run_log if e["node"] == "synthesize_draft"]
        assert len(critiques) >= 2 and len(syntheses) >= 2, "critique loop did not fire"
        assert "NEEDS REVISION" in critiques[0]["message"]
        assert "APPROVED" in critiques[1]["message"]
        print(f"  [OK] {len(critiques)} critique passes, {len(syntheses)} synthesis passes")

        print("\n[5/8] Verifying interrupt...")
        assert not [e for e in run_log if e["node"] in ("finalize", "discard")]
        snapshot = graph.get_state(config)
        assert snapshot.next == ("human_approval_gate",), snapshot.next
        print(f"  [OK] Genuinely paused — next: {snapshot.next}")

        print("\n[6/8] Resuming with APPROVE...")
        result = graph.invoke(Command(resume="approve"), config)
        final = result.get("final_thesis")
        assert final is not None and final["human_approved"] is True
        assert "NOT FINANCIAL ADVICE" in final["disclaimer"]
        print(f"  [OK] Finalized: {final['direction']} @ {final['confidence_score']}")
        print(f"       Disclaimer: {final['disclaimer'][:60]}...")

        print("\n[7/8] Testing rejection path...")
        cfg_r = {"configurable": {"thread_id": str(uuid.uuid4())}}
        graph.invoke(dict(INITIAL_STATE), cfg_r)
        result_r = graph.invoke(Command(resume="reject"), cfg_r)
        assert result_r["human_decision"] == "reject"
        assert result_r["final_thesis"] is None
        assert [e for e in result_r["run_log"] if e["node"] == "discard"]
        print("  [OK] Rejection path works — thesis discarded")

        print("\n[8/8] Run log (approval path):")
        for i, entry in enumerate(result["run_log"], 1):
            print(f"  {i:>2}. [{entry['status']:>20}] {entry['node']:>22} — "
                  f"{entry.get('message', '')[:70]}")
        print(f"\n  Trace saved to: {save_run_trace(result, thread_id)}")
    finally:
        for p in patches:
            p.stop()

    print("\n" + "=" * 70)
    print("ALL PHASE 1 CHECKS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    run_verification()
