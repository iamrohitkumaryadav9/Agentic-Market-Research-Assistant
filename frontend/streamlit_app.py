"""
Streamlit frontend for the Market Research Agent.

Features:
  - Ticker input
  - Live graph-progress display (which node is currently running)
  - Approve/reject UI for the human gate
  - Final thesis display with citations and disclaimer

Connects to the FastAPI backend via HTTP.
"""

from __future__ import annotations

import time

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Market Research Agent",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .node-step {
        padding: 8px 16px;
        margin: 4px 0;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        font-size: 14px;
    }
    .node-completed {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    .node-running {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
    }
    .node-failed {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
    }
    .node-waiting {
        background-color: #cce5ff;
        border-left: 4px solid #007bff;
    }
    .disclaimer-box {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        border-radius: 8px;
        padding: 16px;
        margin: 16px 0;
        font-weight: bold;
    }
    .thesis-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 12px;
        padding: 24px;
        margin: 16px 0;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "run_id" not in st.session_state:
    st.session_state.run_id = None
if "ticker" not in st.session_state:
    st.session_state.ticker = ""
if "status" not in st.session_state:
    st.session_state.status = None


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def start_run(ticker: str):
    """Start a new research run via the API."""
    try:
        resp = requests.post(f"{API_BASE}/run", json={"ticker": ticker}, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            st.session_state.run_id = data["run_id"]
            st.session_state.ticker = ticker.upper()
            return data
        else:
            st.error(f"Error: {resp.json().get('detail', resp.text)}")
            return None
    except requests.ConnectionError:
        st.error("Cannot connect to API server. Make sure it's running on localhost:8000")
        return None


def get_status(run_id: str):
    """Check run status via the API."""
    try:
        resp = requests.get(f"{API_BASE}/status/{run_id}", timeout=10)
        if resp.status_code == 200:
            return resp.json()
        return None
    except Exception:
        return None


def submit_decision(run_id: str, decision: str):
    """Submit approval/rejection via the API."""
    try:
        resp = requests.post(
            f"{API_BASE}/approve/{run_id}",
            json={"decision": decision},
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json()
        else:
            st.error(f"Error: {resp.json().get('detail', resp.text)}")
            return None
    except Exception as e:
        st.error(f"Error: {e}")
        return None


def get_thesis(run_id: str):
    """Fetch the final thesis via the API."""
    try:
        resp = requests.get(f"{API_BASE}/thesis/{run_id}", timeout=10)
        if resp.status_code == 200:
            return resp.json()
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Node progress display
# ---------------------------------------------------------------------------

NODE_LABELS = {
    "fetch_market_data": "📈 Fetch Market Data",
    "fetch_news": "📰 Fetch News",
    "analyze_sentiment": "🧠 Analyze Sentiment",
    "synthesize_draft": "📝 Synthesize Draft",
    "critique": "🔍 Critique Review",
    "human_approval_gate": "👤 Human Approval",
    "finalize": "✅ Finalize",
    "discard": "❌ Discard",
}

NODE_ORDER = [
    "fetch_market_data", "fetch_news", "analyze_sentiment",
    "synthesize_draft", "critique", "human_approval_gate", "finalize",
]


def render_progress(status_data: dict):
    """Render the graph progress visualization."""
    run_log = status_data.get("run_log", [])
    current_node = status_data.get("current_node")
    status = status_data.get("status", "")

    completed_nodes = set()
    failed_nodes = set()
    for entry in run_log:
        node = entry.get("node", "")
        if entry.get("status") == "completed":
            completed_nodes.add(node)
        elif entry.get("status") == "failed":
            failed_nodes.add(node)

    st.markdown("### Graph Progress")

    for node_id in NODE_ORDER:
        label = NODE_LABELS.get(node_id, node_id)

        if node_id in completed_nodes:
            css_class = "node-completed"
            icon = "✅"
        elif node_id in failed_nodes:
            css_class = "node-failed"
            icon = "❌"
        elif node_id == current_node:
            css_class = "node-waiting" if "awaiting" in status else "node-running"
            icon = "⏸️" if "awaiting" in status else "🔄"
        else:
            css_class = ""
            icon = "⬜"
            label = f"<span style='opacity:0.5'>{label}</span>"

        st.markdown(
            f'<div class="node-step {css_class}">{icon} {label}</div>',
            unsafe_allow_html=True,
        )

    # Show critique loop iterations
    critique_count = status_data.get("critique_count", 0)
    if critique_count > 1:
        st.info(f"🔄 Critique loop ran {critique_count} times (draft was revised)")


def render_thesis(thesis: dict):
    """Render the final thesis with formatting."""
    # Disclaimer first
    st.markdown(
        f'<div class="disclaimer-box">⚠️ {thesis.get("disclaimer", "")}</div>',
        unsafe_allow_html=True,
    )

    # Main thesis card
    col1, col2, col3 = st.columns(3)
    with col1:
        direction = thesis.get("direction", "N/A")
        emoji = {"bullish": "🟢", "bearish": "🔴", "neutral": "🟡"}.get(direction, "⚪")
        st.metric("Direction", f"{emoji} {direction.upper()}")
    with col2:
        conf = thesis.get("confidence_score", 0)
        st.metric("Confidence", f"{conf:.0f}/100")
    with col3:
        st.metric("Human Approved", "✅ Yes" if thesis.get("human_approved") else "❌ No")

    # Summary
    st.markdown("### Summary")
    st.write(thesis.get("summary", "N/A"))

    # Evidence columns
    col_tech, col_sent = st.columns(2)

    with col_tech:
        st.markdown("### 📈 Technical Evidence")
        for ev in thesis.get("technical_evidence", []):
            st.markdown(f"- {ev}")

    with col_sent:
        st.markdown("### 📰 Sentiment Evidence")
        for ev in thesis.get("sentiment_evidence", []):
            st.markdown(f"- {ev}")

    # Contradictions
    contras = thesis.get("contradictions", [])
    if contras:
        st.markdown("### ⚡ Contradictions")
        for c in contras:
            st.warning(c)

    # Risks
    risks = thesis.get("risks_and_caveats", [])
    if risks:
        st.markdown("### ⚠️ Risks & Caveats")
        for r in risks:
            st.markdown(f"- {r}")

    # Sources
    sources = thesis.get("cited_sources", [])
    if sources:
        st.markdown("### 📎 Cited Sources")
        for s in sources:
            st.markdown(f"- [{s}]({s})")

    # Data quality
    used = thesis.get("data_sources_used", [])
    failed = thesis.get("data_sources_failed", [])
    if used or failed:
        st.markdown("### 🔌 Data Sources")
        for u in used:
            st.markdown(f"- ✅ {u}")
        for f_src in failed:
            st.markdown(f"- ❌ {f_src}")

    st.caption(f"Generated at: {thesis.get('generated_at', 'N/A')}")


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

st.title("📊 Market Research Agent")
st.caption("LangGraph-powered multi-agent research assistant • NOT a trading system")

# --- Ticker input ---
with st.sidebar:
    st.header("New Research Run")
    ticker_input = st.text_input(
        "Stock Ticker",
        placeholder="e.g., AAPL, MSFT, GOOGL",
        max_chars=10,
    )

    if st.button("🚀 Start Research", type="primary", disabled=not ticker_input):
        result = start_run(ticker_input)
        if result:
            st.success(f"Run started: {result['run_id'][:8]}...")

    st.divider()
    st.header("Existing Run")
    manual_run_id = st.text_input("Run ID", placeholder="Paste a run ID")
    if st.button("Load Run") and manual_run_id:
        st.session_state.run_id = manual_run_id
        st.rerun()

# --- Main content ---
if st.session_state.run_id:
    run_id = st.session_state.run_id

    st.markdown(f"**Run ID:** `{run_id}`")
    st.markdown(f"**Ticker:** `{st.session_state.ticker}`")

    # Poll for status
    status_data = get_status(run_id)

    if status_data:
        status = status_data.get("status", "unknown")

        # Progress visualization
        render_progress(status_data)

        if status_data.get("degraded"):
            st.warning(
                "This run is degraded: one or more data or model steps failed. "
                "Review the run log and errors before interpreting the thesis."
            )

        # --- Awaiting approval ---
        if status == "awaiting_approval":
            st.markdown("---")
            st.markdown("## 👤 Human Review Required")

            draft = status_data.get("draft_thesis", {})
            if draft:
                st.markdown(f"**Direction:** {draft.get('direction', 'N/A')}")
                st.markdown(f"**Confidence:** {draft.get('confidence_score', 'N/A')}/100")
                st.markdown(f"**Summary:** {draft.get('summary', 'N/A')}")

                with st.expander("View full draft"):
                    st.json(draft)

            col_approve, col_reject = st.columns(2)
            with col_approve:
                if st.button("✅ Approve Thesis", type="primary", use_container_width=True):
                    result = submit_decision(run_id, "approve")
                    if result:
                        st.success("Thesis approved and finalized!")
                        time.sleep(1)
                        st.rerun()

            with col_reject:
                if st.button("❌ Reject Thesis", type="secondary", use_container_width=True):
                    result = submit_decision(run_id, "reject")
                    if result:
                        st.warning("Thesis rejected.")
                        time.sleep(1)
                        st.rerun()

        # --- Completed ---
        elif status == "completed":
            st.markdown("---")
            thesis_data = get_thesis(run_id)
            if thesis_data and thesis_data.get("final_thesis"):
                render_thesis(thesis_data["final_thesis"])
            else:
                st.info("Thesis finalized. Fetching...")

        # --- Rejected ---
        elif status == "rejected":
            st.markdown("---")
            st.error("This thesis was rejected by the human reviewer.")

        # --- Running ---
        elif status in ("starting", "running"):
            st.info("Research is in progress... Refresh to check status.")
            if st.button("🔄 Refresh"):
                st.rerun()

        # --- Failed ---
        elif status == "failed":
            st.error("Run failed. Check error log below.")

        # Error log
        errors = status_data.get("error_log", [])
        if errors:
            with st.expander(f"⚠️ Errors ({len(errors)})"):
                for e in errors:
                    st.code(e)

        # Run log
        run_log = status_data.get("run_log", [])
        if run_log:
            with st.expander(f"📋 Run Log ({len(run_log)} entries)"):
                for entry in run_log:
                    node = entry.get("node", "?")
                    sts = entry.get("status", "?")
                    msg = entry.get("message", "")
                    dur = entry.get("duration_ms")
                    dur_str = f" ({dur:.0f}ms)" if dur else ""
                    st.text(f"[{sts}] {node}{dur_str} — {msg}")

else:
    st.info("Enter a stock ticker in the sidebar and click 'Start Research' to begin.")

    st.markdown("""
    ### How it works

    1. **Enter a ticker** — The system fetches price data and news
    2. **AI analysis** — Sentiment classification and thesis synthesis
    3. **Adversarial critique** — A second AI pass checks for weaknesses
    4. **You decide** — Approve or reject the thesis
    5. **Final output** — Cited research thesis with disclaimer

    > ⚠️ This is a research tool, not a trading system.
    > All output requires independent verification.
    """)
