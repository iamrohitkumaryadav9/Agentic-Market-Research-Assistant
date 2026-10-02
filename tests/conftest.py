"""
Test isolation: pin settings *before* app.config is imported so a developer's
local .env (python-dotenv never overrides variables that are already set)
can't change test behaviour, and so no test ever reaches a real LLM.
"""

import os
import tempfile

os.environ["LLM_PROVIDER"] = "anthropic"
os.environ["CRITIQUE_MAX_RETRIES"] = "2"
os.environ["CHECKPOINTER_BACKEND"] = "memory"
os.environ["MASSIVE_API_KEY"] = ""
os.environ["OLLAMA_MAX_SENTIMENT_ARTICLES"] = "1"
os.environ["RUN_TRACES_DIR"] = tempfile.mkdtemp(prefix="mra_traces_")
