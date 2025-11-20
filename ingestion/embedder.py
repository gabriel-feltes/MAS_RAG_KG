"""
Document embedding generation for vector search with metrics tracking.

Fixes:
- Correct token cost scaling (prices are per 1K tokens, not per 1M).
- Uses provider-level cost calculator to keep pricing in one place.
- Captures response.usage.prompt_tokens when available; otherwise falls back
  to a heuristic (~len(text)/4 tokens).
- In batch mode, allocates token/cost per item proportionally by input size.
"""

from __future__ import annotations

import os
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import hashlib
import math

from dotenv import load_dotenv

# OpenAI exceptions (compatible with openai>=1.x)
try:
    from openai import RateLimitError, APIError  # type: ignore
except Exception:  # pragma: no cover
    RateLimitError = APIError = Exception  # fallback if pkg layout changes

from .chunker import DocumentChunk

# Import flexible providers (shared configs + cost)
try:
    from ..agent.providers import (
        get_embedding_client,
        get_embedding_model,
        calculate_embedding_cost,
        get_embedding_config,
    )
    from ..agent.db_utils import db_pool
except ImportError:
    import sys
    import os as _os
    sys.path.append(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from agent.providers import (  # type: ignore
        get_embedding_client,
        get_embedding_model,
        calculate_embedding_cost,
        get_embedding_config,
    )
    from agent.db_utils import db_pool  # type: ignore

load_dotenv()
logger = logging.getLogger(__name__)


# ============================================================================
# Helpers
# ============================================================================

def _approx_token_count(text: str) -> int:
    """
    Very rough token approximation if provider doesn't return usage.
    Heuristic: ~4 chars per token (safe default for OpenAI-like BPE).
    """
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def _clip_for_model(text: str, max_tokens: int) -> str:
    """
    Clip text conservatively to ~4 chars per token to avoid model hard limits.
    """
    if not text:
        return text
    max_chars = max_tokens * 4
    if len(text) > max_chars:
        return text[:max_chars]
    return text


def _safe_usage_tokens(resp_usage: Any) -> int:
    """
    Extract prompt token count from response.usage if present; otherwise 0.
    """
    try:
        return int(getattr(resp_usage, "prompt_tokens", 0))
    except Exception:
        try:
            # Some providers might return dict-like usage
            return int(resp_usage.get("prompt_tokens", 0))  # type: ignore
        except Exception:
            return 0


# ============================================================================
# Metrics
# ============================================================================

class EmbeddingMetrics:
    """Rastreia métricas agregadas de embedding (tokens, custo)."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or get_embedding_model()
        self.total_input_tokens = 0
        self.total_output_tokens = 0  # embeddings usually 0; kept for symmetry
        self.total_cost_usd = 0.0
        self.requests_count = 0
        self.errors_count = 0

    def add_usage(self, input_tokens: int, output_tokens: int = 0):
        """Registra uso de tokens + custo centralizado nos providers."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost_usd += calculate_embedding_cost(input_tokens, self.model_name)
        self.requests_count += 1

    def add_error(self):
        self.errors_count += 1

    def to_dict(self) -> Dict[str, Any]:
        total_tokens = self.total_input_tokens + self.total_output_tokens
        avg_cost = (self.total_cost_usd / self.requests_count) if self.requests_count else 0.0
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "requests_count": self.requests_count,
            "errors_count": self.errors_count,
            "avg_cost_per_request": round(avg_cost, 6),
        }


# ============================================================================
# Embedding Generator
# ============================================================================

class EmbeddingGenerator:
    """Gera embeddings com rastreamento fiel de tokens e custo."""

    def __init__(
        self,
        model: str = None,
        batch_size: int = 100,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self.model = model or get_embedding_model()
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self._client = get_embedding_client()
        self._emb_cfg = get_embedding_config(self.model)
        # se não houver max_tokens no config, use um padrão conservador (OpenAI ~8191)
        self.max_tokens = int(self._emb_cfg.get("max_tokens", 8191))
        self.dimensions = int(self._emb_cfg.get("dimensions", 1536))

        self.metrics = EmbeddingMetrics(self.model)

    # -------- low-level single call --------
    async def _embed_call(self, inputs: List[str]) -> Tuple[List[List[float]], int]:
        """
        Faz uma chamada ao provedor para 1..N textos.
        Retorna (embeddings, used_tokens).
        """
        resp = await self._client.embeddings.create(model=self.model, input=inputs)
        used_tokens = _safe_usage_tokens(getattr(resp, "usage", None))
        vectors = [d.embedding for d in resp.data]
        return vectors, used_tokens

    async def generate_embedding(self, text: str) -> Tuple[List[float], Dict[str, Any]]:
        """
        Gera embedding para um único texto, com métricas detalhadas.
        """
        clipped = _clip_for_model(text, self.max_tokens)

        for attempt in range(self.max_retries):
            try:
                vectors, used_tokens = await self._embed_call([clipped])

                # fallback se provider não reporta usage
                if used_tokens == 0:
                    used_tokens = _approx_token_count(clipped)

                # atualiza métricas globais
                self.metrics.add_usage(used_tokens)

                usage = {
                    "input_tokens": used_tokens,
                    "output_tokens": 0,
                    "total_tokens": used_tokens,
                    "cost_usd": calculate_embedding_cost(used_tokens, self.model),
                    "cached": False,
                }
                logger.debug(
                    f"Embedding generated: {usage['input_tokens']} tokens, "
                    f"${usage['cost_usd']:.6f}"
                )
                return vectors[0], usage

            except RateLimitError:
                if attempt == self.max_retries - 1:
                    self.metrics.add_error()
                    raise
                delay = self.retry_delay * (2 ** attempt)
                logger.warning(f"Rate limit; retrying in {delay:.2f}s")
                await asyncio.sleep(delay)

            except APIError as e:
                logger.error(f"Embedding API error: {e}")
                if attempt == self.max_retries - 1:
                    self.metrics.add_error()
                    raise
                await asyncio.sleep(self.retry_delay)

            except Exception as e:
                logger.error(f"Error generating embedding: {e}")
                if attempt == self.max_retries - 1:
                    self.metrics.add_error()
                    raise
                await asyncio.sleep(self.retry_delay)

        # unreachable
        raise RuntimeError("Exhausted retries in generate_embedding")

    # -------- batch call --------
    async def generate_embeddings_batch(
        self, texts: List[str]
    ) -> Tuple[List[List[float]], Dict[str, Any]]:
        """
        Gera embeddings em lote. Distribui tokens/custo proporcionalmente ao tamanho
        de cada item quando o provedor só retorna usage agregado.
        """
        if not texts:
            return [], {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}

        processed = [_clip_for_model(t or "", self.max_tokens) for t in texts]

        for attempt in range(self.max_retries):
            try:
                vectors, used_tokens = await self._embed_call(processed)

                # se o provedor não trazer usage, estima
                if used_tokens == 0:
                    used_tokens = sum(_approx_token_count(t) for t in processed)

                # atualiza métricas agregadas
                self.metrics.add_usage(used_tokens)

                batch_cost = calculate_embedding_cost(used_tokens, self.model)
                metrics = {
                    "input_tokens": used_tokens,
                    "output_tokens": 0,
                    "cost_usd": batch_cost,
                }
                logger.info(
                    f"Batch processed: {used_tokens} tokens, ${batch_cost:.6f}"
                )
                return vectors, metrics

            except RateLimitError:
                if attempt == self.max_retries - 1:
                    self.metrics.add_error()
                    logger.warning(
                        "Rate limit exceeded; falling back to per-item processing."
                    )
                    return await self._process_individually(processed)

                delay = self.retry_delay * (2 ** attempt)
                logger.warning(f"Rate limit; retry batch in {delay:.2f}s")
                await asyncio.sleep(delay)

            except APIError as e:
                logger.error(f"Embedding API error (batch): {e}")
                if attempt == self.max_retries - 1:
                    self.metrics.add_error()
                    return await self._process_individually(processed)
                await asyncio.sleep(self.retry_delay)

            except Exception as e:
                logger.error(f"Error in batch embedding: {e}")
                if attempt == self.max_retries - 1:
                    self.metrics.add_error()
                    return await self._process_individually(processed)
                await asyncio.sleep(self.retry_delay)

        # unreachable
        raise RuntimeError("Exhausted retries in generate_embeddings_batch")

    async def _process_individually(
        self, texts: List[str]
    ) -> Tuple[List[List[float]], Dict[str, Any]]:
        """
        Fallback: processa item a item (mantém métricas corretas).
        """
        vectors: List[List[float]] = []
        total_tokens = 0
        total_cost = 0.0

        for t in texts:
            try:
                vec, usage = await self.generate_embedding(t)
                vectors.append(vec)
                total_tokens += int(usage.get("input_tokens", 0))
                total_cost += float(usage.get("cost_usd", 0.0))
                await asyncio.sleep(0.05)  # ameniza throttling
            except Exception as e:
                logger.error(f"Failed to embed item: {e}")
                vectors.append([0.0] * self.dimensions)

        return vectors, {
            "input_tokens": total_tokens,
            "output_tokens": 0,
            "cost_usd": total_cost,
        }

    # -------- public: embed chunks --------
    async def embed_chunks(
        self,
        chunks: List[DocumentChunk],
        progress_callback: Optional[callable] = None,
    ) -> List[DocumentChunk]:
        """
        Gera embeddings para uma lista de chunks com batch + métricas corretas.
        Atribui tokens/custo por chunk proporcional ao tamanho do conteúdo.
        """
        if not chunks:
            return chunks

        logger.info(f"Generating embeddings for {len(chunks)} chunks")

        embedded: List[DocumentChunk] = []
        total_batches = (len(chunks) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(chunks), self.batch_size):
            batch_chunks = chunks[i : i + self.batch_size]
            batch_texts = [c.content or "" for c in batch_chunks]

            try:
                vectors, batch_metrics = await self.generate_embeddings_batch(batch_texts)

                # Distribuição proporcional por tamanho (evita dividir igualmente)
                lengths = [max(1, len(_clip_for_model(t, self.max_tokens))) for t in batch_texts]
                sum_len = sum(lengths)
                if sum_len == 0:
                    # fallback seguro
                    per_item_tokens = [0 for _ in lengths]
                    per_item_costs = [0.0 for _ in lengths]
                else:
                    per_item_tokens = [
                        int(round(batch_metrics["input_tokens"] * (L / sum_len))) for L in lengths
                    ]
                    # ajuste para fechar no total (due to rounding)
                    delta = batch_metrics["input_tokens"] - sum(per_item_tokens)
                    if delta != 0:
                        # corrige distribuindo +/-1 nos primeiros itens
                        for k in range(abs(delta)):
                            idx = k % len(per_item_tokens)
                            per_item_tokens[idx] += 1 if delta > 0 else -1

                    per_item_costs = [
                        calculate_embedding_cost(tok, self.model) for tok in per_item_tokens
                    ]

                # Materializa chunks com vetores + metadados
                for c, vec, tok, cst in zip(batch_chunks, vectors, per_item_tokens, per_item_costs):
                    out = DocumentChunk(
                        content=c.content,
                        index=c.index,
                        start_char=c.start_char,
                        end_char=c.end_char,
                        metadata={
                            **(c.metadata or {}),
                            "embedding_model": self.model,
                            "embedding_tokens": tok,
                            "embedding_cost_usd": round(cst, 6),
                            "embedding_generated_at": datetime.now().isoformat(),
                        },
                        token_count=c.token_count,
                    )
                    out.embedding = vec or ([0.0] * self.dimensions)
                    embedded.append(out)

                current_batch = (i // self.batch_size) + 1
                if progress_callback:
                    progress_callback(current_batch, total_batches)

                logger.info(
                    f"Batch {current_batch}/{total_batches}: "
                    f"{batch_metrics['input_tokens']} tokens, "
                    f"${batch_metrics['cost_usd']:.6f}"
                )

            except Exception as e:
                logger.error(f"Batch {(i // self.batch_size) + 1} failed: {e}")
                # fallback: devolve chunks com vetor zeroed e erro no metadata
                for c in batch_chunks:
                    c.metadata = {
                        **(c.metadata or {}),
                        "embedding_error": str(e),
                        "embedding_generated_at": datetime.now().isoformat(),
                    }
                    c.embedding = [0.0] * self.dimensions
                    embedded.append(c)

        logger.info(f"Embeddings complete. Metrics: {self.metrics.to_dict()}")
        return embedded

    async def embed_query(self, query: str) -> List[float]:
        vec, _ = await self.generate_embedding(query or "")
        return vec

    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics.to_dict()

    def get_embedding_dimension(self) -> int:
        return self.dimensions


# ============================================================================
# Simple in-memory cache wrapper
# ============================================================================

class EmbeddingCache:
    """Cache in-memory para embeddings (LRU simples baseado em timestamp)."""

    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, List[float]] = {}
        self.access_times: Dict[str, datetime] = {}
        self.max_size = max_size

    def _key(self, text: str) -> str:
        return hashlib.md5((text or "").encode("utf-8")).hexdigest()

    def get(self, text: str) -> Optional[List[float]]:
        k = self._key(text)
        if k in self.cache:
            self.access_times[k] = datetime.now()
            return self.cache[k]
        return None

    def put(self, text: str, embedding: List[float]):
        k = self._key(text)
        if len(self.cache) >= self.max_size and self.access_times:
            oldest_key = min(self.access_times, key=lambda kk: self.access_times[kk])
            self.cache.pop(oldest_key, None)
            self.access_times.pop(oldest_key, None)
        self.cache[k] = embedding
        self.access_times[k] = datetime.now()


def create_embedder(
    model: str = None,
    use_cache: bool = True,
    **kwargs,
) -> EmbeddingGenerator:
    """
    Factory de embedder com cache opcional.
    Mantém o tracking de métricas mesmo com cache (não conta custo de cache hits).
    """
    embedder = EmbeddingGenerator(model=model or get_embedding_model(), **kwargs)

    if not use_cache:
        return embedder

    cache = EmbeddingCache()
    original_generate = embedder.generate_embedding

    async def cached_generate(text: str) -> Tuple[List[float], Dict[str, Any]]:
        cached = cache.get(text or "")
        if cached is not None:
            # cache hit: não incrementar tokens/custo
            return cached, {
                "cached": True,
                "input_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
            }
        vec, usage = await original_generate(text or "")
        cache.put(text or "", vec)
        return vec, usage

    embedder.generate_embedding = cached_generate  # type: ignore[assignment]
    return embedder
