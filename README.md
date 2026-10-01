# Market Research Agent

A multi-agent market research assistant built with **LangGraph** that produces analyst-grade research theses on stocks. Given a ticker, the system fetches price/technical data, pulls news, analyzes sentiment, synthesizes a draft thesis, adversarially critiques it, and waits for human approval before finalizing.

> **⚠️ NOT A TRADING SYSTEM** — This is a research assistant. It does not place trades, connect to brokerages, or give direct buy/sell instructions. Every output is a research artifact requiring independent human review. This constraint is enforced architecturally (guardrails module + baked-in disclaimer in the output schema), not just via prompts.

## Why that boundary matters

Financial advice has regulatory implications (SEC, FINRA). By treating all output as research artifacts — never as actionable instructions — this system can be used as an analytical tool without crossing into regulated territory. The guardrails module uses pattern matching (not just LLM instructions) to detect and refuse trade-oriented requests, including prompt injection attempts.

---

## Graph Architecture

```mermaid
graph TD
    START([START]) --> fetch_market_data
    fetch_market_data --> fetch_news
    fetch_news --> analyze_sentiment
    analyze_sentiment --> synthesize_draft
    synthesize_draft --> critique
    critique -->|"needs_revision ≤2 retries"| synthesize_draft
    critique -->|"approved / max retries"| human_approval_gate
    human_approval_gate -->|"⏸ interrupt — waits for human"| PAUSE["⏸ PAUSED"]
    PAUSE -->|"approve"| finalize
    PAUSE -->|"reject"| discard
    finalize --> END([END])
    discard --> END
```

### Nodes

| Node | Purpose | Data Source |
|------|---------|-------------|
| `fetch_market_data` | Price history + SMA-20/50, RSI-14, volume trend | yfinance |
| `fetch_news` | Ticker-scoped news articles | Massive.com (Polygon.io) → RSS fallback |
| `analyze_sentiment` | Batched sentiment classification for fetched articles | LLM (structured output) |
| `synthesize_draft` | Combines technicals + sentiment into thesis | LLM (structured output) |
| `critique` | Adversarial review for weaknesses | LLM (different prompt) |
| `human_approval_gate` | Pauses execution via `interrupt()` | Human decision |
| `finalize` | Formats final thesis with disclaimer | — |
| `discard` | Records rejection | — |

### Conditional Edges

- **Critique loop**: `critique → synthesize_draft` if rejected AND `critique_count < 2`, else `critique → human_approval_gate`
- **Human gate**: `human_approval_gate → finalize` on approve, `→ discard` on reject

---

## Design Rationale: Why LangGraph over a plain chain

1. **Conditional cycles** — The critique → synthesize_draft retry loop is a *conditional cycle*, not expressible in a linear LangChain chain without hacks.
2. **True interrupt/resume** — `interrupt()` genuinely pauses graph execution via the checkpointer. The process can be stopped and restarted; state survives. LangChain chains have no equivalent.
3. **Inspectable topology** — The graph can be visualized (`graph.get_graph().draw_mermaid()`), and execution is traced node-by-node through LangSmith or the structured run log.
4. **Typed state** — The `ResearchState` TypedDict with Pydantic validation at node boundaries gives compile-time and runtime safety that dict-passing chains lack.

---

## Setup

### Prerequisites
- Python 3.11+
- API keys: at least one of Anthropic, OpenAI, or Google Gemini (for LLM nodes)
- Massive.com API key (optional — falls back to RSS)

### Local Setup

```bash
# Clone and install
cd market-research-agent
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your API keys:
#   ANTHROPIC_API_KEY=sk-ant-...   (or)
#   OPENAI_API_KEY=sk-proj-...
#   GOOGLE_API_KEY=...             (or)
#   LLM_PROVIDER=anthropic         (or openai or gemini)
#   MASSIVE_API_KEY=your-key       (optional)
#   LANGSMITH_API_KEY=lsv2_...     (optional)

# For Gemini, set GOOGLE_API_KEY and optionally GEMINI_MODEL (default: gemini-3.8-flash).

# Run tests
pytest tests/ -v

# Start the API server
uvicorn app.api:app --reload --port 8000

# In another terminal, start the Streamlit frontend
streamlit run frontend/streamlit_app.py --server.port 8501
```

### Docker

```bash
docker build -t market-research-agent .
docker run -p 8000:8000 -p 8501:8501 --env-file .env market-research-agent
```

The API is at `http://localhost:8000` and the Streamlit UI at `http://localhost:8501`.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/run` | Start a research run. Body: `{"ticker": "AAPL"}` |
| `GET` | `/status/{run_id}` | Check progress — returns current node, draft thesis, run log |
| `POST` | `/approve/{run_id}` | Submit human decision. Body: `{"decision": "approve"}` or `{"decision": "reject"}` |
| `GET` | `/thesis/{run_id}` | Fetch the final thesis (after approval) |
| `GET` | `/health` | Health check |

---

## Worked Example: Full Run Trace (MSFT)

This trace is from a real run with live yfinance data and Massive.com news (LLM nodes used stubs during this particular run due to API credits):

```
Step  Status           Node                    Detail
────  ──────           ────                    ──────
 1    completed        fetch_market_data       63 bars, Close: $497.12, RSI: 56.18, Volume: decreasing
 2    completed        fetch_news              10 articles from Massive.com
 3    completed        analyze_sentiment       Sentiment: neutral (+0/-0/~10, avg conf 20%)
 4    completed        synthesize_draft        Draft: bullish (confidence 65/100, revision 0)
 5    completed        critique                Pass 1: NEEDS REVISION — 2 issues
 6    completed        synthesize_draft        Draft: bullish (confidence 70/100, revision 1)
 7    completed        critique                Pass 2: APPROVED — 0 issues
 8    ⏸ PAUSED        human_approval_gate     Awaiting human decision
      ↓ (resume with "approve")
 9    completed        human_approval_gate     Human decision: approve
10    completed        finalize                Final thesis: bullish, confidence 70.0
```

**Key observations:**
- The **critique loop fired** — draft rejected on pass 1, revised, then approved on pass 2
- The **interrupt genuinely paused** — verified via `graph.get_state()` showing `next == ('human_approval_gate',)`
- The **resume** correctly routed to finalize after approval
- The **disclaimer** is baked into the FinalThesis object: `"NOT FINANCIAL ADVICE — This output is a research artifact..."`

See `run_traces/` for saved JSON traces.

---

## Checkpointer Swap Guide

The system defaults to `MemorySaver` (in-process, dev only). For production persistence:

```bash
# SQLite
pip install langgraph-checkpoint-sqlite
# In .env:
CHECKPOINTER_BACKEND=sqlite
SQLITE_DB_PATH=./checkpoints.db

# PostgreSQL
pip install langgraph-checkpoint-postgres
# In .env:
CHECKPOINTER_BACKEND=postgres
POSTGRES_CONN_STRING=postgresql://user:pass@host:5432/dbname
```

That's it — `config.py` picks up the new backend automatically.

---

## Known Limitations

| Limitation | Detail |
|-----------|--------|
| Data source reliability | yfinance can be rate-limited; Massive.com free tier has request limits |
| LLM sentiment accuracy | Sentiment classification is LLM-based, not fine-tuned — accuracy varies |
| LLM quota limits | Articles are classified in one batch request; if a model quota is exhausted, sentiment is marked unavailable and the run is flagged as degraded for reviewer attention |
| No backtesting | The system produces point-in-time theses, not backtested strategies |
| Single ticker | One ticker per run — no batch/portfolio analysis |
| In-memory runs | API run tracking uses an in-memory dict (lost on restart) |
| No streaming | Frontend polls for status rather than receiving streaming updates |

### What I'd add for production

- **Streaming updates** — WebSocket or SSE for real-time node progress to the frontend
- **Persistent checkpointer** — PostgresSaver for interrupt/resume across deployments
- **Multi-ticker batch** — Run multiple tickers in parallel with a graph-per-ticker pattern
- **Audit logging** — Compliance-grade logging of every LLM input/output and human decision
- **Fine-tuned sentiment** — FinBERT or domain-specific model instead of general LLM
- **Backtesting framework** — Historical accuracy tracking of thesis directions
- **Rate limit pooling** — Rotating API keys for Massive.com and LLM providers
- **CI/CD** — GitHub Actions for test/lint/build on every PR

---

## Project Structure

```
market-research-agent/
├── app/
│   ├── state.py             # TypedDict state + 10 Pydantic models
│   ├── nodes/
│   │   ├── market_data.py   # yfinance + SMA/RSI/volume
│   │   ├── news.py          # Massive.com API + RSS fallback
│   │   ├── sentiment.py     # LLM per-article classification
│   │   ├── synthesis.py     # LLM thesis generation
│   │   ├── critique.py      # LLM adversarial review
│   │   ├── human_gate.py    # interrupt() pause/resume
│   │   └── finalize.py      # Final output + discard
│   ├── graph.py             # StateGraph definition + conditional edges
│   ├── guardrails.py        # Pattern-based action-request refusal
│   ├── api.py               # FastAPI: /run, /status, /approve, /thesis
│   └── config.py            # Env loading, LLM/checkpointer factories
├── frontend/
│   └── streamlit_app.py     # Ticker input, progress display, approve/reject UI
├── tests/
│   └── test_agent.py        # 10 pytest tests (nodes, loop, interrupt, guardrails)
├── run_traces/              # Saved execution traces (gitignored except example)
├── .env.example
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Tests

```bash
pytest tests/ -v
```

10 tests covering:
1. Graph compilation
2. Pydantic schema validation
3. Market data node (success)
4. Market data graceful degradation (failure)
5. Critique retry loop (reject → revise → approve)
6. Interrupt/resume (approve path)
7. Interrupt/resume (reject path)
8. Guardrail blocks trade requests
9. Guardrail allows research requests
10. Finalize output structure completeness
