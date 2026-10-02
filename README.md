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
- Python 3.11+ (developed on 3.14; the Docker image uses 3.12)
- LLM provider: an API key for Anthropic, OpenAI, or Google Gemini, or a local Ollama model
- Massive.com API key (optional — falls back to RSS)

### Local Setup

```bash
# Clone and install
cd Agentic-Market-Research-Assistant
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
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
# For local inference, install Ollama, run `ollama pull qwen3:4b`, then set
# LLM_PROVIDER=ollama and OLLAMA_MODEL=qwen3:4b. On CPU, the default local
# sentiment cap is one article; increase OLLAMA_MAX_SENTIMENT_ARTICLES if your
# hardware can handle more. OLLAMA_NUM_PREDICT controls output length
# (default: 768). OLLAMA_BASE_URL defaults to http://localhost:11434.
# Market/news data still require their configured sources.

# Run tests (offline: no network, no API keys, no LLM needed)
pytest tests/ -v

# Start the API server
uvicorn app.api:app --reload --port 8000

# In another terminal, start the Streamlit frontend
streamlit run frontend/streamlit_app.py --server.port 8501
```

### Verification scripts (one per build phase)

| Script | What it proves | Needs |
|--------|----------------|-------|
| `python verify_phase1.py` | Graph compiles, critique loop fires, interrupt pauses, approve/reject routing | nothing (mocked) |
| `python verify_phase2.py` | Real yfinance + news (Massive.com or RSS fallback), full graph on live data | internet + a working LLM |
| `python verify_phase3.py` | Real LLM sentiment/synthesis/critique, full run, resume, trace | internet + a working LLM |
| `python verify_phase4.py` | Critique node, retry loop, max-retry exhaustion | nothing (mocked) |
| `python verify_phase5.py` | Human gate: state preservation, concurrent threads, approve/reject, invalid input | nothing (mocked) |

Phases 2 and 3 use whatever `LLM_PROVIDER` is configured. With a small local
Ollama model on CPU they take several minutes.

### Docker

```bash
docker build -t market-research-agent .
docker run -p 8000:8000 -p 8501:8501 --env-file .env market-research-agent
```

Using Ollama from inside the container? Set `OLLAMA_BASE_URL=http://host.docker.internal:11434`
in your `.env` (the container cannot reach the host's `localhost`).

The API is at `http://localhost:8000` and the Streamlit UI at `http://localhost:8501`.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/run` | Start a research run. Body: `{"ticker": "AAPL"}` |
| `GET` | `/status/{run_id}` | Check progress — status, current node, draft thesis, critique, run log, `degraded` + `degradation_reasons` |
| `POST` | `/approve/{run_id}` | Submit human decision. Body: `{"decision": "approve"}` or `{"decision": "reject"}` |
| `GET` | `/thesis/{run_id}` | Fetch the final thesis (after approval) |
| `GET` | `/health` | Health check |

`/run` returns immediately (the graph runs in a background thread). Poll `/status/{run_id}` until
`status` is `awaiting_approval`, then `POST /approve/{run_id}`. Statuses: `starting`, `running`,
`awaiting_approval`, `completed`, `rejected`, `failed`. A decision is case-insensitive; anything
other than approve/reject is refused (HTTP 422). A placeholder draft (synthesis failed) cannot be
approved (HTTP 409).

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

The system defaults to `MemorySaver` (in-process, dev only: a run paused at the human gate is lost
on restart). For persistence:

```bash
# SQLite (langgraph-checkpoint-sqlite is already in requirements.txt)
# In .env:
CHECKPOINTER_BACKEND=sqlite
SQLITE_DB_PATH=./checkpoints.db

# PostgreSQL
pip install langgraph-checkpoint-postgres "psycopg[binary]"
# In .env:
CHECKPOINTER_BACKEND=postgres
POSTGRES_CONN_STRING=postgresql://user:pass@host:5432/dbname
```

`config.py` picks up the new backend automatically; if the package for a persistent backend is
missing it logs a warning and falls back to memory.

---

## Known Limitations

| Limitation | Detail |
|-----------|--------|
| Data source reliability | yfinance can be rate-limited; Massive.com free tier has request limits |
| LLM sentiment accuracy | Sentiment classification is LLM-based, not fine-tuned — accuracy varies |
| LLM quota limits | Articles are classified in one batch request; if a model quota is exhausted, sentiment is marked unavailable and the run is flagged as degraded for reviewer attention |
| No backtesting | The system produces point-in-time theses, not backtested strategies |
| Single ticker | One ticker per run — no batch/portfolio analysis |
| In-memory runs | The API's run registry (run_id -> status) is an in-memory dict, lost on restart even with a persistent checkpointer |
| Small local models | A 3-8B Ollama model produces weak theses (e.g. inconsistent SMA reasoning). The critique node and the human gate are there to catch this; use a stronger model for real work |
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
│   └── config.py            # Env loading, LLM (4 providers) + checkpointer factories
├── frontend/
│   └── streamlit_app.py     # Ticker input, progress display, approve/reject UI
├── tests/
│   ├── conftest.py          # Pins settings so tests ignore your local .env
│   ├── test_agent.py        # Original suite: schema, market data, sentiment, critique loop, interrupt, guardrails, finalize
│   ├── test_fixes.py        # Retry loop, human-gate input validation, auto-approval flags, UTF-8 traces
│   └── test_pipeline.py     # Real nodes with mocked yfinance/news/LLM; full API flow
├── run_traces/              # Saved execution traces (gitignored except example)
├── verify_phase1.py ... verify_phase5.py   # per-phase verification scripts
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

All tests run offline: yfinance, both news providers and the LLM are mocked (see
`tests/conftest.py`, which also pins settings so a local `.env` can't affect results). They cover:

- **Graph:** compilation and node set; retry loop (reject → revise → approve); max-retry exhaustion
  proceeds with a flag; reject path; LLM failure everywhere still reaches the human gate, flagged
- **Human gate:** state preserved across the interrupt; decisions are case-insensitive; an invalid
  decision re-prompts instead of silently discarding
- **Nodes:** market data success and graceful failure; RSS fallback; sentiment/synthesis/critique with
  mocked structured output; prompt-injection in news is dropped before it reaches the LLM
- **Guardrails:** trade/injection requests blocked (including zero-width and fullwidth evasion),
  research questions allowed, ticker validation
- **Output:** `FinalThesis` always carries the disclaimer; full critique history; accurate data-source
  reporting; UTF-8 trace files
- **API:** full run → approve → thesis flow, reject flow, guardrail 400s, 404/409/422 handling,
  placeholder drafts cannot be approved