"""
Phase 3 Verification Script

Tests the LLM nodes with real Claude API calls:
  1. analyze_sentiment — per-article sentiment classification
  2. synthesize_draft — thesis generation from real data
  3. Full graph run with real data + real LLM through to interrupt

Run: python verify_phase3.py
"""

from __future__ import annotations

import json
import sys
import uuid

sys.path.insert(0, ".")

import os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from app.config import configure_logging
configure_logging()

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from app.graph import build_graph, save_run_trace


def test_full_graph_with_llm():
    print("=" * 70)
    print("PHASE 3: Full graph run — real data + real LLM (AAPL)")
    print("=" * 70)

    checkpointer = MemorySaver()
    builder = build_graph()
    graph = builder.compile(checkpointer=checkpointer)

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "ticker": "AAPL",
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

    print("\n  Running graph for AAPL (real data + real LLM)...")
    print("  This will take ~30-60s (multiple LLM calls)...\n")

    result = graph.invoke(initial_state, config)

    # --- Print run log ---
    run_log = result.get("run_log", [])
    errors = result.get("error_log", [])

    print("  --- Run log ---")
    for i, entry in enumerate(run_log, 1):
        status = entry.get("status", "?")
        node = entry.get("node", "?")
        msg = entry.get("message", "")
        dur = entry.get("duration_ms")
        dur_str = f" ({dur:.0f}ms)" if dur else ""
        if len(msg) > 90:
            msg = msg[:87] + "..."
        print(f"  {i:>2}. [{status:>12}] {node:>25}{dur_str} | {msg}")

    if errors:
        print(f"\n  --- Errors ---")
        for e in errors:
            print(f"  ! {e[:100]}")

    # --- Sentiment results ---
    sentiment = result.get("sentiment_summary", {})
    if sentiment:
        print(f"\n  --- Sentiment Analysis ---")
        print(f"  Overall: {sentiment.get('overall_sentiment', 'N/A')}")
        print(f"  Positive: {sentiment.get('positive_count', 0)}, "
              f"Negative: {sentiment.get('negative_count', 0)}, "
              f"Neutral: {sentiment.get('neutral_count', 0)}")
        print(f"  Avg confidence: {sentiment.get('average_confidence', 0):.0%}")

        arts = sentiment.get("article_sentiments", [])
        if arts:
            print(f"\n  Per-article (first 5):")
            for a in arts[:5]:
                print(f"    [{a.get('sentiment', '?'):>8}] ({a.get('confidence', 0):.0%}) "
                      f"{a.get('article_title', '?')[:55]}...")
                print(f"             {a.get('rationale', '')[:70]}...")

    # --- Draft thesis ---
    draft = result.get("draft_thesis", {})
    if draft:
        print(f"\n  --- Draft Thesis ---")
        print(f"  Direction: {draft.get('direction', 'N/A')}")
        print(f"  Confidence: {draft.get('confidence_score', 'N/A')}/100")
        print(f"  Revision: {draft.get('revision_number', 0)}")
        print(f"\n  Summary:")
        summary = draft.get("summary", "")
        # Word-wrap summary
        words = summary.split()
        line = "    "
        for w in words:
            if len(line) + len(w) > 78:
                print(line)
                line = "    " + w
            else:
                line += " " + w if line.strip() else "    " + w
        if line.strip():
            print(line)

        tech_ev = draft.get("technical_evidence", [])
        if tech_ev:
            print(f"\n  Technical evidence ({len(tech_ev)} points):")
            for t in tech_ev[:4]:
                print(f"    - {t[:75]}...")

        sent_ev = draft.get("sentiment_evidence", [])
        if sent_ev:
            print(f"\n  Sentiment evidence ({len(sent_ev)} points):")
            for s in sent_ev[:4]:
                print(f"    - {s[:75]}...")

        contras = draft.get("contradictions", [])
        if contras:
            print(f"\n  Contradictions ({len(contras)}):")
            for c in contras:
                print(f"    - {c[:75]}...")

        risks = draft.get("risks_and_caveats", [])
        if risks:
            print(f"\n  Risks ({len(risks)}):")
            for r in risks[:4]:
                print(f"    - {r[:75]}...")

    # --- Verify interrupt ---
    snapshot = graph.get_state(config)
    print(f"\n  Graph paused at: {snapshot.next}")

    # --- Resume and finalize ---
    print("\n  Resuming with approval...")
    result = graph.invoke(Command(resume="approve"), config)

    final = result.get("final_thesis")
    if final:
        print(f"\n  --- Final Thesis ---")
        print(f"  Direction: {final['direction']}")
        print(f"  Confidence: {final['confidence_score']}")
        print(f"  Human approved: {final['human_approved']}")
        print(f"  Disclaimer: {final.get('disclaimer', '')[:60]}...")
        print(f"  Sources used: {final.get('data_sources_used', [])}")

    # Save trace
    trace_path = save_run_trace(result, thread_id)
    print(f"\n  Trace saved to: {trace_path}")

    print("\n" + "=" * 70)
    print("PHASE 3 VERIFICATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    print()
    test_full_graph_with_llm()
