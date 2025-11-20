"""
Provider utilities for LLM and embedding models with multi-provider support.
Supports: OpenAI, Ollama, Anthropic, and OpenAI-compatible APIs.

This version INSTRUMENTS token usage globally:
- Wraps AsyncOpenAI chat.completions.create and embeddings.create to capture `usage`
- Falls back to tiktoken estimation when providers don't return usage
- Exposes get_global_token_metrics()/reset_global_token_metrics()/record_reranker_tokens()

You can read these counters anywhere (e.g., for pricing in ingest.py).
"""

from __future__ import annotations

import os
import logging
import asyncio
from typing import Optional, Dict, Any
from types import SimpleNamespace

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

# LLM Models and costs
MODEL_CONFIGS = {
    # OpenAI
    "gpt-4o": {
        "provider": "openai",
        "context_window": 128000,
        "max_output_tokens": 16384,
        "supports_streaming": True,
        "cost_per_1k_input": 0.0025,
        "cost_per_1k_output": 0.010,
    },
    "gpt-4o-mini": {
        "provider": "openai",
        "context_window": 128000,
        "max_output_tokens": 16384,
        "supports_streaming": True,
        "cost_per_1k_input": 0.00015,
        "cost_per_1k_output": 0.0006,
    },
    "gpt-4-turbo": {
        "provider": "openai",
        "context_window": 128000,
        "max_output_tokens": 4096,
        "supports_streaming": True,
        "cost_per_1k_input": 0.01,
        "cost_per_1k_output": 0.03,
    },
    "gpt-4": {
        "provider": "openai",
        "context_window": 8192,
        "max_output_tokens": 4096,
        "supports_streaming": True,
        "cost_per_1k_input": 0.03,
        "cost_per_1k_output": 0.06,
    },
    "gpt-3.5-turbo": {
        "provider": "openai",
        "context_window": 16385,
        "max_output_tokens": 4096,
        "supports_streaming": True,
        "cost_per_1k_input": 0.0005,
        "cost_per_1k_output": 0.0015,
    },

    # Ollama (local, free)
    "mistral:7b-instruct-q4_K_M": {
        "provider": "ollama",
        "context_window": 32000,
        "max_output_tokens": 8192,
        "supports_streaming": False,
        "cost_per_1k_input": 0.0,
        "cost_per_1k_output": 0.0,
    },
    "llama2:13b": {
        "provider": "ollama",
        "context_window": 4096,
        "max_output_tokens": 4096,
        "supports_streaming": False,
        "cost_per_1k_input": 0.0,
        "cost_per_1k_output": 0.0,
    },
    "phi:latest": {
        "provider": "ollama",
        "context_window": 2048,
        "max_output_tokens": 1024,
        "supports_streaming": False,
        "cost_per_1k_input": 0.0,
        "cost_per_1k_output": 0.0,
    },

    # Anthropic
    "claude-3-5-sonnet-20241022": {
        "provider": "anthropic",
        "context_window": 200000,
        "max_output_tokens": 8192,
        "supports_streaming": True,
        "cost_per_1k_input": 0.003,
        "cost_per_1k_output": 0.015,
    },
    "claude-3-opus-20240229": {
        "provider": "anthropic",
        "context_window": 200000,
        "max_output_tokens": 4096,
        "supports_streaming": True,
        "cost_per_1k_input": 0.015,
        "cost_per_1k_output": 0.075,
    },
    "claude-3-haiku-20240307": {
        "provider": "anthropic",
        "context_window": 200000,
        "max_output_tokens": 4096,
        "supports_streaming": True,
        "cost_per_1k_input": 0.00025,
        "cost_per_1k_output": 0.00125,
    },
    "openai/gpt-oss-120b": {
        "provider": "groq",
        "context_window": 131072,
        "max_output_tokens": 65536,
        "supports_streaming": True,
        "cost_per_1k_input": 0.0,
        "cost_per_1k_output": 0.0,
    },
    "groq/compound": {
        "provider": "groq",
        "context_window": 131072,
        "max_output_tokens": 8192,
        "supports_streaming": True,
        "cost_per_1k_input": 0.0,
        "cost_per_1k_output": 0.0,
    },
}

# Embedding models
EMBEDDING_MODELS = {
    "text-embedding-3-small": {
        "provider": "openai",
        "dimensions": 1536,
        "cost_per_1k_tokens": 0.00002,
    },
    "text-embedding-3-large": {
        "provider": "openai",
        "dimensions": 3072,
        "cost_per_1k_tokens": 0.00013,
    },
    "text-embedding-ada-002": {
        "provider": "openai",
        "dimensions": 1536,
        "cost_per_1k_tokens": 0.0001,
    },
    "nomic-embed-text": {
        "provider": "ollama",
        "dimensions": 768,
        "cost_per_1k_tokens": 0.0,
    },
}

# ============================================================================
# GLOBAL TOKEN COUNTER (singleton)
# ============================================================================

class GlobalTokenCounter:
    """Thread/async-safe global token aggregator."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._data = {
            "tokens_llm_prompt": 0,
            "tokens_llm_completion": 0,
            "tokens_embeddings": 0,
            "tokens_reranker": 0,
            "llm_calls": 0,
            "embedding_calls": 0,
            "reranker_calls": 0,
        }

    async def add_llm(self, prompt: int, completion: int) -> None:
        async with self._lock:
            self._data["tokens_llm_prompt"] += int(prompt or 0)
            self._data["tokens_llm_completion"] += int(completion or 0)
            self._data["llm_calls"] += 1

    async def add_embeddings(self, total: int) -> None:
        async with self._lock:
            self._data["tokens_embeddings"] += int(total or 0)
            self._data["embedding_calls"] += 1

    async def add_reranker(self, total: int) -> None:
        async with self._lock:
            self._data["tokens_reranker"] += int(total or 0)
            self._data["reranker_calls"] += 1

    async def snapshot(self) -> Dict[str, int]:
        async with self._lock:
            return dict(self._data)

    async def reset(self) -> None:
        async with self._lock:
            for k in self._data:
                self._data[k] = 0

_GLOBAL_COUNTER = GlobalTokenCounter()

def get_global_token_metrics_sync() -> Dict[str, int]:
    """Synchronous helper to read counters (safe to call from sync contexts)."""
    loop = asyncio.get_event_loop() if asyncio.get_event_loop().is_running() else None
    if loop and loop.is_running():
        # Not ideal to block; return last-known snapshot if needed.
        # For simplicity, we run create_task and gather; but many callers are async.
        # Prefer async get_global_token_metrics() when available.
        # Here we just return zeros to avoid blocking event loop misuse.
        return {}  # encourage async usage
    return asyncio.run(_GLOBAL_COUNTER.snapshot())  # type: ignore

async def get_global_token_metrics() -> Dict[str, int]:
    return await _GLOBAL_COUNTER.snapshot()

async def reset_global_token_metrics() -> None:
    await _GLOBAL_COUNTER.reset()

async def record_reranker_tokens(total_tokens: int) -> None:
    await _GLOBAL_COUNTER.add_reranker(int(total_tokens or 0))

# ============================================================================
# UTIL: fallback token estimation
# ============================================================================

def _estimate_tokens_from_messages(messages: Any) -> int:
    try:
        import tiktoken  # type: ignore
        enc = tiktoken.get_encoding("cl100k_base")
        txt = ""
        for m in messages or []:
            # OpenAI v1 uses dicts like {"role":"user","content":"..."}
            content = m.get("content", "")
            if isinstance(content, list):
                # If content is a list of parts, join their 'text' fields
                for part in content:
                    if isinstance(part, dict):
                        txt += str(part.get("text", "")) + "\n"
                    else:
                        txt += str(part) + "\n"
            else:
                txt += str(content) + "\n"
        return len(enc.encode(txt))
    except Exception:
        # ~4 chars per token fallback
        text = ""
        for m in messages or []:
            text += str(m.get("content", "")) + "\n"
        return max(len(text) // 4, 1)

def _estimate_tokens_from_inputs(inputs: Any) -> int:
    try:
        import tiktoken  # type: ignore
        enc = tiktoken.get_encoding("cl100k_base")
        if isinstance(inputs, list):
            return sum(len(enc.encode(str(x))) for x in inputs)
        return len(enc.encode(str(inputs)))
    except Exception:
        if isinstance(inputs, list):
            return max(sum(len(str(x)) for x in inputs) // 4, 1)
        return max(len(str(inputs)) // 4, 1)

# ============================================================================
# MODEL GETTERS
# ============================================================================

def get_model(model_name: Optional[str] = None) -> str:
    """Get LLM model name to use."""
    model = os.getenv("LLM_CHOICE")
    if model:
        return model

    model = os.getenv("MODEL_NAME")
    if model:
        return model

    if model_name:
        return model_name

    return DEFAULT_MODEL


def get_embedding_model(model_name: Optional[str] = None) -> str:
    """Get embedding model name to use."""
    model = os.getenv("EMBEDDING_MODEL")
    if model:
        return model

    if model_name:
        return model_name

    return DEFAULT_EMBEDDING_MODEL


def get_ingestion_model(model_name: Optional[str] = None) -> str:
    """Get ingestion model (for Graphiti extraction)."""
    model = os.getenv("INGESTION_LLM_CHOICE")
    if model:
        return model
    return get_model(model_name)


def get_model_config(model_name: Optional[str] = None) -> dict:
    """Get configuration for a model."""
    model = get_model(model_name)
    if model in MODEL_CONFIGS:
        return MODEL_CONFIGS[model]
    logger.warning(f"Model {model} not in config, using defaults")
    return {
        "provider": "openai",
        "context_window": 8192,
        "max_output_tokens": 4096,
        "supports_streaming": True,
        "cost_per_1k_input": 0.0,
        "cost_per_1k_output": 0.0,
    }


def get_embedding_config(model_name: Optional[str] = None) -> dict:
    """Get configuration for embedding model."""
    model = get_embedding_model(model_name)
    if model in EMBEDDING_MODELS:
        return EMBEDDING_MODELS[model]
    logger.warning(f"Embedding model {model} not in config, using defaults")
    return {
        "provider": "openai",
        "dimensions": 1536,
        "cost_per_1k_tokens": 0.0,
    }

# ============================================================================
# AsyncOpenAI INSTRUMENTED WRAPPERS
# ============================================================================

class _ChatCompletionsCreateWrapper:
    def __init__(self, inner_create):
        self._create = inner_create

    async def create(self, *args, **kwargs):
        resp = await self._create(*args, **kwargs)
        # OpenAI v1: resp.usage may exist with prompt_tokens/completion_tokens
        usage = getattr(resp, "usage", None)
        if usage:
            prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion = int(getattr(usage, "completion_tokens", 0) or 0)
        else:
            # No usage returned -> estimate from messages
            msgs = kwargs.get("messages") or {}
            prompt = _estimate_tokens_from_messages(msgs)
            # Can't know completion_tokens without usage; leave 0
            completion = 0
        await _GLOBAL_COUNTER.add_llm(prompt, completion)
        return resp

class _ChatProxy:
    def __init__(self, inner_chat):
        self.completions = SimpleNamespace(create=_ChatCompletionsCreateWrapper(inner_chat.completions.create).create)

class _EmbeddingsCreateWrapper:
    def __init__(self, inner_create):
        self._create = inner_create

    async def create(self, *args, **kwargs):
        resp = await self._create(*args, **kwargs)
        usage = getattr(resp, "usage", None)
        if usage:
            # OpenAI embeddings usage often exposes total_tokens or prompt_tokens
            total = int(
                getattr(usage, "total_tokens", None)
                or getattr(usage, "prompt_tokens", 0)
                or 0
            )
        else:
            inputs = kwargs.get("input")
            total = _estimate_tokens_from_inputs(inputs)
        await _GLOBAL_COUNTER.add_embeddings(total)
        return resp

class _EmbeddingsProxy:
    def __init__(self, inner_embeddings):
        self.create = _EmbeddingsCreateWrapper(inner_embeddings.create).create

class _AsyncOpenAIInstrumented:
    """Thin proxy around AsyncOpenAI that instruments chat+embeddings."""

    def __init__(self, inner):
        self._inner = inner
        self.chat = _ChatProxy(inner.chat)
        self.embeddings = _EmbeddingsProxy(inner.embeddings)

    # Allow passthrough for any attribute not wrapped
    def __getattr__(self, item):
        return getattr(self._inner, item)

# ============================================================================
# CLIENT GETTERS
# ============================================================================

def get_embedding_client():
    """Get embedding client (OpenAI-compatible) wrapped for token capture."""
    from openai import AsyncOpenAI

    provider = os.getenv("EMBEDDING_PROVIDER", "openai").lower()
    base_url = os.getenv("EMBEDDING_BASE_URL", "https://api.openai.com/v1")
    api_key = os.getenv("EMBEDDING_API_KEY", "")

    if provider == "ollama":
        api_key = api_key or "ollama"
        logger.info(f"Using Ollama embeddings at {base_url}")
        # Ollama client is not OpenAI-compatible via openai lib; skip wrapping here.
        return AsyncOpenAI(api_key=api_key, base_url=base_url)

    if not api_key:
        raise ValueError("EMBEDDING_API_KEY not set for non-Ollama provider")

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    return _AsyncOpenAIInstrumented(client)


def get_llm_client(provider: Optional[str] = None):
    """Get LLM client (OpenAI, Ollama, or Anthropic). OpenAI/Ollama are instrumented."""
    if provider is None:
        provider = os.getenv("LLM_PROVIDER", "openai").lower()
    else:
        provider = provider.lower()

    if provider in ["openai", "ollama", "groq"]:
        from openai import AsyncOpenAI

        base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
        api_key = os.getenv("LLM_API_KEY", "")

        if provider == "ollama":
            api_key = api_key or "ollama"
            logger.info(f"Using Ollama LLM at {base_url}")
            # If your Ollama endpoint is OpenAI-compatible, the wrapper still works.
            client = AsyncOpenAI(api_key=api_key, base_url=base_url)
            return _AsyncOpenAIInstrumented(client)

        if provider == "groq":
            api_key = api_key or "groq"
            logger.info(f"Using Groq LLM at {base_url}")
            client = AsyncOpenAI(api_key=api_key, base_url=base_url)
            return _AsyncOpenAIInstrumented(client)

        if not api_key:
            raise ValueError("LLM_API_KEY not set for OpenAI")

        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        return _AsyncOpenAIInstrumented(client)

    elif provider == "anthropic":
        # Anthropic SDK doesn't return OpenAI-style usage; leave unwrapped.
        from anthropic import AsyncAnthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        return AsyncAnthropic(api_key=api_key)

    else:
        raise ValueError(f"Unsupported provider: {provider}")

# ============================================================================
# COST CALCULATION
# ============================================================================

def calculate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    model_name: Optional[str] = None,
) -> float:
    """Calculate cost for model call."""
    config = get_model_config(model_name)
    input_cost = (prompt_tokens / 1000) * config.get("cost_per_1k_input", 0)
    output_cost = (completion_tokens / 1000) * config.get("cost_per_1k_output", 0)
    return input_cost + output_cost


def calculate_embedding_cost(
    token_count: int,
    model_name: Optional[str] = None,
) -> float:
    """Calculate cost for embedding."""
    config = get_embedding_config(model_name)
    return (token_count / 1000) * config.get("cost_per_1k_tokens", 0)

# ============================================================================
# MODEL VALIDATION / LISTING
# ============================================================================

def validate_model(model_name: str) -> bool:
    return model_name in MODEL_CONFIGS

def validate_embedding_model(model_name: str) -> bool:
    return model_name in EMBEDDING_MODELS

def list_available_models() -> dict:
    return {
        "llm_models": list(MODEL_CONFIGS.keys()),
        "embedding_models": list(EMBEDDING_MODELS.keys()),
        "current_llm": get_model(),
        "current_embedding": get_embedding_model(),
        "current_ingestion_llm": get_ingestion_model(),
        "llm_provider": os.getenv("LLM_PROVIDER", "openai"),
        "embedding_provider": os.getenv("EMBEDDING_PROVIDER", "openai"),
    }

# ============================================================================
# PROVIDER DETECTION
# ============================================================================

def get_provider_from_model(model_name: str) -> str:
    return get_model_config(model_name)["provider"]

def get_api_key(provider: str) -> str:
    provider = provider.lower()
    if provider == "openai":
        api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("LLM_API_KEY or OPENAI_API_KEY not set")
        return api_key
    elif provider == "ollama":
        return "ollama"
    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        return api_key
    else:
        raise ValueError(f"Unknown provider: {provider}")

def get_provider_info() -> dict:
    return {
        "llm_provider": os.getenv("LLM_PROVIDER", "openai"),
        "llm_base_url": os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
        "llm_model": get_model(),
        "embedding_provider": os.getenv("EMBEDDING_PROVIDER", "openai"),
        "embedding_base_url": os.getenv("EMBEDDING_BASE_URL", "https://api.openai.com/v1"),
        "embedding_model": get_embedding_model(),
        "ingestion_model": get_ingestion_model(),
    }

# ============================================================================
# EXAMPLE
# ============================================================================

def example_usage():
    """Example showing provider utilities."""
    info = get_provider_info()
    print(f"Provider config: {info}")

    model = get_model()
    config = get_model_config(model)
    print(f"LLM: {model} - {config}")

    cost = calculate_cost(1000, 500, model)
    print(f"Cost: ${cost:.6f}")

    models = list_available_models()
    print(f"Available models: {len(models['llm_models'])} LLM + {len(models['embedding_models'])} embedding")

if __name__ == "__main__":
    example_usage()
