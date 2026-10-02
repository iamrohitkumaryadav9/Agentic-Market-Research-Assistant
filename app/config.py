"""
Configuration module for the Market Research Agent.

Handles:
- Environment variable loading (.env)
- LLM provider factory (Anthropic primary, OpenAI fallback, Gemini, local Ollama)
- Checkpointer factory (MemorySaver for dev, swap to SqliteSaver/PostgresSaver)
- API key validation
- Logging configuration

All secrets come from environment variables; nothing is hardcoded.
"""

from __future__ import annotations

import logging
import os
from enum import Enum
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("market_research_agent")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class LLMProvider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GEMINI = "gemini"
    OLLAMA = "ollama"


class CheckpointerBackend(str, Enum):
    MEMORY = "memory"
    SQLITE = "sqlite"
    POSTGRES = "postgres"


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

class Settings:
    """Centralised settings read from environment variables."""

    # LLM
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", LLMProvider.ANTHROPIC.value)
    ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))

    # Local Ollama (no hosted API key needed)
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen3:4b")
    # Small CPU-bound models: keep the sentiment batch tiny and outputs capped
    OLLAMA_MAX_SENTIMENT_ARTICLES: int = int(os.getenv("OLLAMA_MAX_SENTIMENT_ARTICLES", "1"))
    OLLAMA_NUM_PREDICT: int = int(os.getenv("OLLAMA_NUM_PREDICT", "1024"))

    # Data sources
    MASSIVE_API_KEY: str = os.getenv("MASSIVE_API_KEY", "")
    MASSIVE_BASE_URL: str = os.getenv(
        "MASSIVE_BASE_URL", "https://api.massive.com"
    )

    # Observability
    LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY", "")
    LANGSMITH_PROJECT: str = os.getenv(
        "LANGSMITH_PROJECT", "market-research-agent"
    )
    LANGSMITH_TRACING: bool = os.getenv(
        "LANGSMITH_TRACING_V2", "false"
    ).lower() == "true"

    # Checkpointer
    CHECKPOINTER_BACKEND: str = os.getenv(
        "CHECKPOINTER_BACKEND", CheckpointerBackend.MEMORY.value
    )
    SQLITE_DB_PATH: str = os.getenv("SQLITE_DB_PATH", "checkpoints.db")
    POSTGRES_CONN_STRING: str = os.getenv("POSTGRES_CONN_STRING", "")

    # Operational
    CRITIQUE_MAX_RETRIES: int = int(os.getenv("CRITIQUE_MAX_RETRIES", "2"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    RUN_TRACES_DIR: str = os.getenv("RUN_TRACES_DIR", "run_traces")


@lru_cache()
def get_settings() -> Settings:
    return Settings()


# ---------------------------------------------------------------------------
# Factory: LLM
# ---------------------------------------------------------------------------

def get_llm(
    provider: str | None = None,
    temperature: float | None = None,
):
    """
    Return a LangChain chat model instance.

    Uses Anthropic by default; falls back to OpenAI if configured.
    Raises ValueError if no valid API key is found.
    """
    settings = get_settings()
    provider = provider or settings.LLM_PROVIDER
    temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE

    if provider == LLMProvider.OLLAMA:
        import httpx
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=temperature,
            reasoning=False,  # qwen3 "thinking" mode is slow and breaks JSON output
            num_ctx=8192,
            num_predict=settings.OLLAMA_NUM_PREDICT,
            # Read timeout applies per streamed chunk: a stalled stream fails
            # (and the node degrades gracefully) instead of hanging forever.
            client_kwargs={"timeout": httpx.Timeout(connect=10, read=180, write=30, pool=10)},
        )

    if provider == LLMProvider.OLLAMA:
        import httpx
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=temperature,
            reasoning=False,  # qwen3 "thinking" mode is slow and breaks JSON output
            num_ctx=8192,
            num_predict=settings.OLLAMA_NUM_PREDICT,
            # Read timeout applies per streamed chunk: a stalled stream fails
            # (and the node degrades gracefully) instead of hanging forever.
            client_kwargs={"timeout": httpx.Timeout(connect=10, read=180, write=30, pool=10)},
        )

    if provider == LLMProvider.GEMINI:
        if not settings.GOOGLE_API_KEY:
            raise ValueError("No GOOGLE_API_KEY configured for Gemini.")
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=temperature,
            retries=1,
            request_timeout=30,
        )

    if provider == LLMProvider.ANTHROPIC:
        if not settings.ANTHROPIC_API_KEY:
            logger.warning("No ANTHROPIC_API_KEY found, falling back to OpenAI")
            provider = LLMProvider.OPENAI
        else:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=settings.ANTHROPIC_MODEL,
                api_key=settings.ANTHROPIC_API_KEY,
                temperature=temperature,
                max_tokens=4096,
            )

    if provider == LLMProvider.OPENAI:
        if not settings.OPENAI_API_KEY:
            raise ValueError(
                "No OPENAI_API_KEY configured for OpenAI."
            )
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.OPENAI_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=temperature,
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")


# ---------------------------------------------------------------------------
# Factory: Checkpointer
# ---------------------------------------------------------------------------

def get_checkpointer(backend: str | None = None):
    """
    Return a LangGraph checkpointer instance.

    Defaults to MemorySaver (in-process, dev only).
    For production persistence, set CHECKPOINTER_BACKEND to "sqlite" or "postgres".

    Swap guide:
      1. pip install langgraph-checkpoint-sqlite   (or langgraph-checkpoint-postgres)
      2. Set CHECKPOINTER_BACKEND=sqlite and SQLITE_DB_PATH=./checkpoints.db
         — or —
         Set CHECKPOINTER_BACKEND=postgres and POSTGRES_CONN_STRING=postgresql://...
      3. Restart. That's it — the graph.compile() call picks up the new checkpointer.
    """
    settings = get_settings()
    backend = backend or settings.CHECKPOINTER_BACKEND

    if backend == CheckpointerBackend.MEMORY:
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver()

    if backend == CheckpointerBackend.SQLITE:
        try:
            import sqlite3
            from langgraph.checkpoint.sqlite import SqliteSaver
            # SqliteSaver needs a live connection, not a path string
            conn = sqlite3.connect(settings.SQLITE_DB_PATH, check_same_thread=False)
            return SqliteSaver(conn)
        except ImportError:
            logger.warning(
                "langgraph-checkpoint-sqlite not installed; falling back to MemorySaver"
            )
            from langgraph.checkpoint.memory import MemorySaver
            return MemorySaver()

    if backend == CheckpointerBackend.POSTGRES:
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            from psycopg import Connection
            from psycopg.rows import dict_row
            conn = Connection.connect(
                settings.POSTGRES_CONN_STRING,
                autocommit=True, prepare_threshold=0, row_factory=dict_row,
            )
            saver = PostgresSaver(conn)
            saver.setup()
            return saver
        except ImportError:
            logger.warning(
                "langgraph-checkpoint-postgres/psycopg not installed; falling back to MemorySaver"
            )
            from langgraph.checkpoint.memory import MemorySaver
            return MemorySaver()

    raise ValueError(f"Unsupported checkpointer backend: {backend}")


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def configure_logging():
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # Enable LangSmith tracing if key is present
    if settings.LANGSMITH_API_KEY:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.LANGSMITH_API_KEY
        os.environ["LANGCHAIN_PROJECT"] = settings.LANGSMITH_PROJECT
        logger.info("LangSmith tracing enabled (project: %s)", settings.LANGSMITH_PROJECT)
    else:
        logger.info("LangSmith tracing disabled — using structured JSON logging fallback")
