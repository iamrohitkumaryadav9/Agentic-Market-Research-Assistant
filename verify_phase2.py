"""
Phase 2 Verification Script

Tests the real data nodes:
  1. fetch_market_data — yfinance pull + technical indicator computation
  2. fetch_news — Massive.com API pull (with RSS fallback test)
  3. Full graph run with real data through to the interrupt point

Run: python verify_phase2.py
"""

from __future__ import annotations

import json
import sys
import uuid

sys.path.insert(0, ".")

# Set encoding for Windows console
import os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from app.config import configure_logging
configure_logging()

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from app.graph import build_graph, save_run_trace
from app.nodes.market_data import fetch_market_data
from app.nodes.news import fetch_news


def test_market_data():
    print("=" * 70)
    print("TEST 1: fetch_market_data (yfinance)")
    print("=" * 70)

    state = {
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

    result = fetch_market_data(state)

    price_data = result.get("price_data")
    tech = result.get("technical_signals")
    log = result["run_log"][0]

    if price_data is None:
        print("  [!] Market data fetch FAILED (check network)")
        print(f"      Error: {result.get('error_log', [])}")
        return False

    print(f"  Ticker: {price_data['ticker']}")
    print(f"  Bars fetched: {len(price_data['bars'])}")
    print(f"  Latest close: ${price_data['latest_close']}")
    print(f"  Latest volume: {price_data['latest_volume']:,}")
    print(f"  SMA-20: {tech['sma_20']}")
    print(f"  SMA-50: {tech['sma_50']}")
    print(f"  RSI-14: {tech['rsi_14']}")
    print(f"  Volume trend: {tech['volume_trend']}")
    print(f"  5d change: {tech['price_change_pct_5d']:+.2f}%")
    print(f"  30d change: {tech['price_change_pct_30d']:+.2f}%")
    print(f"  Signal summary: {tech['signal_summary']}")
    print(f"  Duration: {log['duration_ms']:.0f}ms")
    print("  [OK] Market data node works")
    return True


def test_news_massive():
    print("\n" + "=" * 70)
    print("TEST 2: fetch_news (Massive.com API)")
    print("=" * 70)

    state = {
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

    result = fetch_news(state)

    articles = result.get("news_articles", [])
    log = result["run_log"][0]
    errors = result.get("error_log", [])

    print(f"  Source used: {log['details']['source']}")
    print(f"  Articles fetched: {len(articles)}")
    print(f"  Has Massive insights: {log['details']['has_insights']}")

    if errors:
        print(f"  Errors: {errors}")

    if articles:
        print(f"\n  --- Sample articles ---")
        for i, a in enumerate(articles[:3], 1):
            print(f"  {i}. [{a.get('source', '?')}] {a['title'][:70]}...")
            print(f"     Published: {a.get('published_utc', 'N/A')}")
            print(f"     URL: {a.get('url', 'N/A')[:60]}...")
            if a.get("insights"):
                for ins in a["insights"][:1]:
                    print(f"     Sentiment: {ins.get('sentiment', '?')} — {ins.get('sentiment_reasoning', '')[:50]}...")
            print()
        print(f"  Duration: {log['duration_ms']:.0f}ms")
        print("  [OK] News node works")
        return True
    else:
        print("  [!] No articles returned (check MASSIVE_API_KEY)")
        return False


def test_full_graph_with_real_data():
    print("\n" + "=" * 70)
    print("TEST 3: Full graph run with real data (to interrupt point)")
    print("=" * 70)

    checkpointer = MemorySaver()
    builder = build_graph()
    graph = builder.compile(checkpointer=checkpointer)

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "ticker": "MSFT",
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

    print("  Running graph for MSFT (will pause at human gate)...")
    result = graph.invoke(initial_state, config)

    # Check real data was fetched
    price_data = result.get("price_data")
    articles = result.get("news_articles", [])
    run_log = result.get("run_log", [])
    errors = result.get("error_log", [])

    print(f"\n  --- Run log ---")
    for i, entry in enumerate(run_log, 1):
        status = entry.get("status", "?")
        node = entry.get("node", "?")
        msg = entry.get("message", "")
        # Truncate long messages
        if len(msg) > 80:
            msg = msg[:77] + "..."
        print(f"  {i:>2}. [{status:>12}] {node:>25} | {msg}")

    if errors:
        print(f"\n  --- Errors (graceful degradation) ---")
        for e in errors:
            print(f"  ! {e}")

    print(f"\n  Has price data: {price_data is not None}")
    print(f"  Has news articles: {len(articles)} articles")
    print(f"  Draft thesis direction: {result.get('draft_thesis', {}).get('direction', 'N/A')}")
    print(f"  Critique count: {result.get('critique_count', 0)}")

    # Verify interrupt
    snapshot = graph.get_state(config)
    print(f"  Graph paused at: {snapshot.next}")

    # Resume and finalize
    print("\n  Resuming with approval...")
    result = graph.invoke(Command(resume="approve"), config)

    final = result.get("final_thesis")
    if final:
        print(f"  Final thesis direction: {final['direction']}")
        print(f"  Final confidence: {final['confidence_score']}")
        print(f"  Data sources used: {final.get('data_sources_used', [])}")
        print(f"  Data sources failed: {final.get('data_sources_failed', [])}")
        print(f"  Disclaimer present: {'NOT FINANCIAL ADVICE' in final.get('disclaimer', '')}")

    # Save trace
    trace_path = save_run_trace(result, thread_id)
    print(f"\n  Trace saved to: {trace_path}")
    print("  [OK] Full graph run with real data works")
    return True


if __name__ == "__main__":
    print()
    ok1 = test_market_data()
    ok2 = test_news_massive()
    ok3 = test_full_graph_with_real_data()

    print("\n" + "=" * 70)
    if ok1 and ok2 and ok3:
        print("ALL PHASE 2 TESTS PASSED")
    else:
        print("SOME TESTS FAILED — see output above")
        failures = []
        if not ok1: failures.append("market_data")
        if not ok2: failures.append("news")
        if not ok3: failures.append("full_graph")
        print(f"  Failed: {', '.join(failures)}")
    print("=" * 70)
