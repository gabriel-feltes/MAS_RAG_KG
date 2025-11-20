# agent/graph_utils.py
"""
Graph utilities for Neo4j/Graphiti integration.
Tracks token usage (LLM, embeddings, reranker) and can compute a cost breakdown.

Environment overrides for pricing (USD per 1K tokens):
- PRICE_CHAT_PROMPT_PER_1K (default 0.005)
- PRICE_CHAT_COMPLETION_PER_1K (default 0.015)
- PRICE_EMBEDDING_PER_1K (default 0.00002)
- PRICE_RERANKER_PER_1K (default 0.0005)

Set LOG_USAGE_DEBUG=1 to print detailed usage logs from wrappers.

IMPORTANT: This version monkey-patches openai.AsyncOpenAI globally so ANY client
created inside graphiti_core will be instrumented automatically.

CACHE-AWARE MODE: Accounts for prompt caching - only counts tokens actually charged.
This should match the OpenAI dashboard exactly.
"""

from __future__ import annotations

import os
import logging
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime, timezone

from dotenv import load_dotenv

# ------------------------------------------------------------------------------------
# Global monkey-patch for OpenAI Async client (before importing graphiti_core clients)
# ------------------------------------------------------------------------------------
load_dotenv()
logger = logging.getLogger(__name__)
LOG_USAGE_DEBUG = os.getenv("LOG_USAGE_DEBUG", "1") == "1"

_active_usage_sink: Callable[[int, int, int], None] | None = None
# signature: (prompt_tokens, completion_tokens, embedding_tokens) -> None


def _set_usage_sink(fn: Callable[[int, int, int], None] | None) -> None:
    global _active_usage_sink
    _active_usage_sink = fn


def _emit_usage(prompt: int = 0, completion: int = 0, embedding: int = 0) -> None:
    if _active_usage_sink:
        try:
            _active_usage_sink(int(prompt or 0), int(completion or 0), int(embedding or 0))
        except Exception as e:
            logger.debug(f"[USAGE] sink error: {e}")


def _coerce_usage(usage: Any) -> tuple[int, int, int]:
    """
    Returns (prompt, completion, embedding).
    CACHE-AWARE: Only counts tokens that were actually CHARGED.
    Subtracts cached tokens to match OpenAI dashboard billing.
    
    For Responses API: input_tokens, output_tokens, cache_read_input_tokens
    For Chat API: prompt_tokens, completion_tokens, cached_tokens
    For Embeddings API: total_tokens
    """
    if usage is None:
        return 0, 0, 0
    
    # dict-like
    if isinstance(usage, dict):
        # Get base token counts
        p = int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0)
        c = int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0)
        
        # SUBTRACT cached tokens (they weren't charged)
        cache_read = int(usage.get("cache_read_input_tokens", 0) or 0)
        cached_tokens = int(usage.get("cached_tokens", 0) or 0)
        total_cached = cache_read + cached_tokens
        
        # Only subtract from prompt tokens
        p = max(0, p - total_cached)
        
        # Embeddings: only count if no LLM tokens
        e = int(usage.get("total_tokens", 0) or 0) if (p == 0 and c == 0 and total_cached == 0) else 0
        
        if LOG_USAGE_DEBUG and total_cached > 0:
            logger.debug(f"[USAGE] Cache hit: {total_cached} tokens (not charged)")
        
        return p, c, e
    
    # attr-like
    p = int(getattr(usage, "input_tokens", getattr(usage, "prompt_tokens", 0)) or 0)
    c = int(getattr(usage, "output_tokens", getattr(usage, "completion_tokens", 0)) or 0)
    
    cache_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
    cached_tokens = int(getattr(usage, "cached_tokens", 0) or 0)
    total_cached = cache_read + cached_tokens
    
    p = max(0, p - total_cached)
    
    e = int(getattr(usage, "total_tokens", 0) or 0) if (p == 0 and c == 0 and total_cached == 0) else 0
    
    if LOG_USAGE_DEBUG and total_cached > 0:
        logger.debug(f"[USAGE] Cache hit: {total_cached} tokens (not charged)")
    
    return p, c, e


def _monkey_patch_openai_async_client() -> None:
    try:
        from openai import AsyncOpenAI as _AsyncOpenAI  # type: ignore

        # Already patched?
        if getattr(_AsyncOpenAI, "_graphiti_usage_patched", False):
            return

        _orig_init = _AsyncOpenAI.__init__

        def _wrap_create(create_fn, kind: str):
            async def _wrapped_create(*args, **kwargs):
                resp = await create_fn(*args, **kwargs)
                usage = getattr(resp, "usage", None)
                
                # Check for cache indicators (before coercion)
                if usage:
                    cache_creation = 0
                    cache_read = 0
                    
                    if isinstance(usage, dict):
                        cache_creation = int(usage.get("cache_creation_input_tokens", 0) or 0)
                        cache_read = int(usage.get("cache_read_input_tokens", 0) or 0)
                        cache_read += int(usage.get("cached_tokens", 0) or 0)
                    else:
                        cache_creation = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
                        cache_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
                        cache_read += int(getattr(usage, "cached_tokens", 0) or 0)
                    
                    if LOG_USAGE_DEBUG and (cache_creation or cache_read):
                        logger.debug(
                            f"[USAGE/{kind}] CACHE - creation={cache_creation}, read={cache_read}"
                        )
                
                p, c, e = _coerce_usage(usage)
                
                if LOG_USAGE_DEBUG:
                    if (p + c + e) == 0:
                        logger.debug(f"[USAGE/{kind}] Zero charged tokens (fully cached or unlimited)")
                    else:
                        logger.debug(f"[USAGE/{kind}] Charged tokens -> p={p} c={c} e={e}")

                _emit_usage(prompt=p, completion=c, embedding=e)
                return resp
            return _wrapped_create

        def __init__(self, *args, **kwargs):
            _orig_init(self, *args, **kwargs)

            # chat.completions.create
            chat = getattr(self, "chat", None)
            if chat and hasattr(chat, "completions"):
                create = chat.completions.create
                chat.completions.create = _wrap_create(create, "chat")

            # responses.parse / responses.create
            responses = getattr(self, "responses", None)
            if responses:
                if hasattr(responses, "parse"):
                    orig_parse = responses.parse
                    async def _wrapped_parse(*a, **kw):
                        # ✅ FIX: Strip incompatible parameters for non-o1/o3 models
                        model = kw.get("model", getattr(self, "_default_model", ""))
                        
                        if not model.startswith(("o1-", "o3-")):
                            if "reasoning" in kw:
                                if LOG_USAGE_DEBUG:
                                    logger.debug(
                                        f"[USAGE/responses.parse] Removing 'reasoning' parameter "
                                        f"(not supported by {model})"
                                    )
                                kw.pop("reasoning")
                            
                            if "text" in kw and isinstance(kw["text"], dict):
                                if "verbosity" in kw["text"]:
                                    original_verbosity = kw["text"]["verbosity"]
                                    if original_verbosity == "low":
                                        if LOG_USAGE_DEBUG:
                                            logger.debug(
                                                f"[USAGE/responses.parse] Changing text.verbosity from 'low' to 'medium' "
                                                f"(gpt-4o-mini only supports 'medium')"
                                            )
                                        kw["text"]["verbosity"] = "medium"
                        
                        resp = await orig_parse(*a, **kw)
                        usage = getattr(resp, "usage", None)
                        
                        # Check for cache before coercion
                        if usage:
                            cache_creation = 0
                            cache_read = 0
                            
                            if isinstance(usage, dict):
                                cache_creation = int(usage.get("cache_creation_input_tokens", 0) or 0)
                                cache_read = int(usage.get("cache_read_input_tokens", 0) or 0)
                                cache_read += int(usage.get("cached_tokens", 0) or 0)
                            else:
                                cache_creation = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
                                cache_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
                                cache_read += int(getattr(usage, "cached_tokens", 0) or 0)
                            
                            if LOG_USAGE_DEBUG and (cache_creation or cache_read):
                                logger.debug(
                                    f"[USAGE/responses.parse] CACHE - creation={cache_creation}, read={cache_read}"
                                )
                        
                        p, c, e = _coerce_usage(usage)
                        
                        if LOG_USAGE_DEBUG:
                            if (p + c + e) == 0:
                                logger.debug(f"[USAGE/responses.parse] Zero charged tokens (fully cached)")
                            else:
                                logger.debug(f"[USAGE/responses.parse] Charged tokens -> p={p} c={c}")
                        
                        _emit_usage(prompt=p, completion=c, embedding=0)
                        return resp
                    responses.parse = _wrapped_parse  # type: ignore

                if hasattr(responses, "create"):
                    responses.create = _wrap_create(responses.create, "responses")  # type: ignore

            # embeddings.create
            embeddings = getattr(self, "embeddings", None)
            if embeddings and hasattr(embeddings, "create"):
                embeddings.create = _wrap_create(embeddings.create, "embed")  # type: ignore

        _AsyncOpenAI.__init__ = __init__
        _AsyncOpenAI._graphiti_usage_patched = True  # type: ignore
        if LOG_USAGE_DEBUG:
            logger.debug("[USAGE] AsyncOpenAI monkey-patched successfully (cache-aware mode)")
    except Exception as e:
        logger.warning(f"[USAGE] Failed to monkey-patch AsyncOpenAI: {e}")


_monkey_patch_openai_async_client()

# ----------------------------- Pricing (env) --------------------------------
PRICES = {
    "chat_prompt_per_1k": float(os.getenv("PRICE_CHAT_PROMPT_PER_1K", "0.005")),
    "chat_completion_per_1k": float(os.getenv("PRICE_CHAT_COMPLETION_PER_1K", "0.015")),
    "embedding_per_1k": float(os.getenv("PRICE_EMBEDDING_PER_1K", "0.00002")),
    "reranker_per_1k": float(os.getenv("PRICE_RERANKER_PER_1K", "0.0005")),
}

# ============================================================================
# Metrics
# ============================================================================

class GraphMetrics:
    """Tracks graph ops and token usage across LLM / embeddings / reranker."""

    def __init__(self) -> None:
        # Graph ops
        self.total_episodes_added = 0
        self.total_tokens_estimated = 0
        self.failed_episodes = 0
        self.search_count = 0
        self.last_error: Optional[str] = None

        # Usage accounting
        self.llm_calls = 0
        self.embedding_calls = 0
        self.reranker_calls = 0
        self.tokens_llm_prompt = 0
        self.tokens_llm_completion = 0
        self.tokens_embeddings = 0
        self.tokens_reranker = 0

    # ----- mutation helpers -----
    def add_episode(self, tokens_estimated: int, *, success: bool = True) -> None:
        if success:
            self.total_episodes_added += 1
            self.total_tokens_estimated += tokens_estimated
        else:
            self.failed_episodes += 1

    def record_search(self) -> None:
        self.search_count += 1

    def record_error(self, error: str) -> None:
        self.last_error = error

    def add_usage(
        self,
        *,
        llm_prompt: int = 0,
        llm_completion: int = 0,
        embeddings: int = 0,
        reranker: int = 0,
        llm_inc: int = 0,
        emb_inc: int = 0,
        rerank_inc: int = 0,
    ) -> None:
        self.tokens_llm_prompt += int(llm_prompt or 0)
        self.tokens_llm_completion += int(llm_completion or 0)
        self.tokens_embeddings += int(embeddings or 0)
        self.tokens_reranker += int(reranker or 0)
        self.llm_calls += int(llm_inc or 0)
        self.embedding_calls += int(emb_inc or 0)
        self.reranker_calls += int(rerank_inc or 0)
        if LOG_USAGE_DEBUG:
            logger.debug(
                f"[USAGE] +prompt={llm_prompt} +completion={llm_completion} "
                f"+emb={embeddings} +rerank={reranker} | "
                f"calls(l/e/r)={self.llm_calls}/{self.embedding_calls}/{self.reranker_calls}"
            )

    # ----- exports -----
    def to_dict(self) -> Dict[str, Any]:
        return {
            "episodes_added": self.total_episodes_added,
            "failed_episodes": self.failed_episodes,
            "tokens_estimated": self.total_tokens_estimated,
            "tokens_llm_prompt": self.tokens_llm_prompt,
            "tokens_llm_completion": self.tokens_llm_completion,
            "tokens_embeddings": self.tokens_embeddings,
            "tokens_reranker": self.tokens_reranker,
            "llm_calls": self.llm_calls,
            "embedding_calls": self.embedding_calls,
            "reranker_calls": self.reranker_calls,
            "search_count": self.search_count,
            "last_error": self.last_error,
        }

# ============================================================================
# Usage-capturing helpers & wrappers (local estimation utils)
# ============================================================================

def _estimate_tokens_from_text(text: str) -> int:
    """Best-effort token estimate using tiktoken if available, else ~4 chars/token fallback."""
    try:
        import tiktoken  # type: ignore
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text or ""))
    except Exception:
        t = text or ""
        return max(len(t) // 4, 1)


def _jsonify_props(obj: Any) -> Any:
    """Converte recursivamente valores Neo4j (DateTime etc.) para JSON-safe."""
    try:
        from neo4j.time import DateTime, Date, Time, LocalTime, LocalDateTime, Duration
        temporal_types = (DateTime, Date, Time, LocalTime, LocalDateTime, Duration)
    except Exception:
        temporal_types = tuple()

    if isinstance(obj, dict):
        return {k: _jsonify_props(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify_props(v) for v in obj]
    if temporal_types and isinstance(obj, temporal_types):
        for m in ("iso_format", "isoformat"):
            if hasattr(obj, m):
                try:
                    return getattr(obj, m)()
                except Exception:
                    pass
        return str(obj)
    if isinstance(obj, (bytes, bytearray)):
        return obj.decode("utf-8", errors="replace")
    return obj

# ------------------------------------------------------------------------------------
# Graphiti imports (AFTER the monkey-patch, so their clients are instrumented)
# ------------------------------------------------------------------------------------
from graphiti_core import Graphiti
from graphiti_core.utils.maintenance.graph_data_operations import clear_data
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient

__all__ = [
    "GraphMetrics",
    "GraphitiClient",
    "graph_client",
    # public API helpers
    "initialize_graph",
    "close_graph",
    "add_episode",
    "search_graph",
    "get_entity_relationships",
    "get_embedding",
    "get_graph_stats",
    "clear_graph",
    "test_graph_connection",
    "get_graph_metrics",
]

# ============================================================================
# Graphiti client
# ============================================================================

class GraphitiClient:
    """Manages Graphiti + Neo4j operations with usage tracking."""

    def __init__(
        self,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
    ) -> None:
        # Neo4j configuration
        self.neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = neo4j_user or os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD")
        if not self.neo4j_password:
            raise ValueError("NEO4J_PASSWORD environment variable not set")

        # LLM configuration
        self.llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()
        self.llm_base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
        self.llm_api_key = os.getenv("LLM_API_KEY", "")
        self.llm_choice = os.getenv("LLM_CHOICE", "gpt-4o-mini")

        # Ollama friendly defaults
        if self.llm_provider == "ollama":
            self.llm_api_key = self.llm_api_key or "ollama"
            logger.info(f"Using Ollama LLM at {self.llm_base_url}")
        elif not self.llm_api_key:
            raise ValueError("LLM_API_KEY environment variable not set for non-Ollama providers")

        # Embedding configuration
        self.embedding_base_url = os.getenv("EMBEDDING_BASE_URL", "https://api.openai.com/v1")
        self.embedding_api_key = os.getenv("EMBEDDING_API_KEY")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.embedding_dimensions = int(os.getenv("VECTOR_DIMENSION", "1536"))
        if self.llm_provider != "ollama" and not self.embedding_api_key:
            raise ValueError("EMBEDDING_API_KEY environment variable not set")

        self.graphiti: Optional[Graphiti] = None
        self.metrics = GraphMetrics()
        self._initialized = False

        # Connect the monkey-patch sink to this instance's metrics
        def sink(prompt: int, completion: int, embedding: int) -> None:
            # Any chat/responses counts as an LLM call; embeddings as embedding call
            llm_inc = 1 if (prompt or completion) else 0
            emb_inc = 1 if embedding else 0
            self.metrics.add_usage(
                llm_prompt=prompt,
                llm_completion=completion,
                embeddings=embedding,
                llm_inc=llm_inc,
                emb_inc=emb_inc,
            )
        _set_usage_sink(sink)

    # ----- lifecycle -----
    async def initialize(self) -> None:
        if self._initialized:
            return
        try:
            llm_config = LLMConfig(
                api_key=self.llm_api_key,
                model=self.llm_choice,
                small_model=self.llm_choice,
                base_url=self.llm_base_url,
            )

            # Use vanilla clients; they are globally instrumented by our monkey-patch.
            llm_client = OpenAIClient(config=llm_config)
            embedder = OpenAIEmbedder(
                config=OpenAIEmbedderConfig(
                    api_key=self.embedding_api_key,
                    embedding_model=self.embedding_model,
                    embedding_dim=self.embedding_dimensions,
                    base_url=self.embedding_base_url,
                )
            )
            reranker = OpenAIRerankerClient(client=llm_client, config=llm_config)

            self.graphiti = Graphiti(
                self.neo4j_uri,
                self.neo4j_user,
                self.neo4j_password,
                llm_client=llm_client,
                embedder=embedder,
                cross_encoder=reranker,
            )

            await self.graphiti.build_indices_and_constraints()
            self._initialized = True
            logger.info(
                f"Graphiti initialized: LLM={self.llm_choice} ({self.llm_provider}), "
                f"Embedder={self.embedding_model}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Graphiti: {e}", exc_info=True)
            raise

    async def close(self) -> None:
        if self.graphiti:
            await self.graphiti.close()
            self.graphiti = None
            self._initialized = False
            logger.info("Graphiti client closed")

    # ----- operations -----
    async def add_episode(
        self,
        episode_id: str,
        content: str,
        source: str,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Add an episode to the graph (tracks token usage via wrappers)."""
        if not self._initialized:
            await self.initialize()

        # minimum 100 tokens for Graphiti processing; estimate with tiktoken fallback
        estimated_tokens = max(_estimate_tokens_from_text(content), 100)
        episode_timestamp = timestamp or datetime.now(timezone.utc)

        try:
            from graphiti_core.nodes import EpisodeType

            await self.graphiti.add_episode(
                name=episode_id,
                episode_body=content,
                source=EpisodeType.text,
                source_description=source,
                reference_time=episode_timestamp,
            )

            self.metrics.add_episode(estimated_tokens, success=True)
            logger.info(f"✓ Added episode {episode_id} (~{estimated_tokens} tokens estimated)")
            return {"episode_id": episode_id, "estimated_tokens": estimated_tokens, "status": "success"}

        except Exception as e:
            error_msg = f"Failed to add episode {episode_id}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.metrics.add_episode(estimated_tokens, success=False)
            self.metrics.record_error(error_msg)
            return {"episode_id": episode_id, "estimated_tokens": estimated_tokens, "status": "failed", "error": str(e)}

    async def search(self, query: str, limit: int = 10, **kwargs) -> List[Dict[str, Any]]:
        if not self._initialized:
            await self.initialize()
        
        # Log if we get the unexpected parameter, but don't crash
        if "center_node_distance" in kwargs:
            logger.debug(
                f"Ignoring unused 'center_node_distance' parameter: {kwargs['center_node_distance']}"
            )
            
        try:
            self.metrics.record_search()
            # The underlying graphiti.search() only takes query
            results = await self.graphiti.search(query) 
            return [
                {
                    "fact": getattr(r, "fact", None),
                    "uuid": str(getattr(r, "uuid", "")),
                    "valid_at": str(getattr(r, "valid_at", None)) if getattr(r, "valid_at", None) else None,
                    "invalid_at": str(getattr(r, "invalid_at", None)) if getattr(r, "invalid_at", None) else None,
                    "source_node_uuid": str(getattr(r, "source_node_uuid", None)) if getattr(r, "source_node_uuid", None) else None,
                }
                for r in (results or [])[:limit]
            ]
        except Exception as e:
            logger.error(f"Graph search failed: {e}")
            self.metrics.record_error(f"Search failed: {str(e)}")
            return []

    async def get_embedding(self, text: str) -> List[float]:
        """Generates an embedding for a single string of text."""
        if not self._initialized:
            await self.initialize()
        
        if not self.graphiti or not self.graphiti.embedder:
            logger.error("Embedder not initialized, cannot get embedding")
            raise RuntimeError("Graphiti client or embedder not available")
            
        try:
            # Access the underlying OpenAI client and config
            client = self.graphiti.embedder.client
            model = self.graphiti.embedder.config.embedding_model

            # This call will be tracked by our monkey-patch
            response = await client.embeddings.create(input=[text], model=model)
            
            if not response.data or not response.data[0].embedding:
                raise ValueError("Embedding generation returned no data or embedding")
            
            # Return the first (and only) embedding vector
            return response.data[0].embedding
        
        except Exception as e:
            logger.error(f"Failed to get embedding: {e}", exc_info=True)
            self.metrics.record_error(f"Embedding failed: {str(e)}")
            # Return empty list on failure
            return []
        
    async def get_graph_statistics(self) -> Dict[str, Any]:
        if not self._initialized:
            await self.initialize()
        try:
            driver = self.graphiti.driver

            # node count
            recs, _, _ = await driver.execute_query("MATCH (n) RETURN count(n) AS count")
            node_count = recs[0]["count"] if recs else 0

            # relationship count
            recs, _, _ = await driver.execute_query("MATCH ()-[r]->() RETURN count(r) AS count")
            rel_count = recs[0]["count"] if recs else 0

            # per-label counts (multi-label safe)
            recs, _, _ = await driver.execute_query(
                """
                MATCH (n)
                UNWIND labels(n) AS label
                RETURN label, count(*) AS count
                ORDER BY count DESC
                """
            )
            node_types = {r["label"]: r["count"] for r in (recs or [])}

            # per-relationship-type counts
            recs, _, _ = await driver.execute_query(
                """
                MATCH ()-[r]->()
                RETURN type(r) AS rel_type, count(*) AS count
                ORDER BY count DESC
                """
            )
            rel_types = {r["rel_type"]: r["count"] for r in (recs or [])}

            return {
                "total_nodes": node_count,
                "total_relationships": rel_count,
                "node_types": node_types,
                "relationship_types": rel_types,
                "graphiti_initialized": True,
                "status": "healthy",
            }
        except Exception as e:
            logger.error(f"Failed to get graph statistics: {e}", exc_info=True)
            return {"error": str(e), "graphiti_initialized": self._initialized, "status": "unhealthy"}

    async def get_entity_relationships(self, entity_name: str, depth: int = 1) -> Dict[str, Any]:
        if not self._initialized:
            await self.initialize()
        try:
            driver = self.graphiti.driver
            cypher_query = (
                """
                MATCH (e:Entity {name: $entity_name})
                OPTIONAL MATCH (e)-[r]-(related:Entity)
                RETURN e.name AS entity, type(r) AS relationship, related.name AS related_entity
                LIMIT 100
                """
            )
            recs, _, _ = await driver.execute_query(cypher_query, entity_name=entity_name)
            relationships = [
                {"entity": r["entity"], "relationship": r["relationship"], "related_entity": r["related_entity"]}
                for r in (recs or [])
                if r.get("related_entity") is not None
            ]
            return {
                "central_entity": entity_name,
                "relationships": relationships,
                "count": len(relationships),
                "status": "success",
            }
        except Exception as e:
            logger.error(f"Failed to get entity relationships: {e}")
            return {"central_entity": entity_name, "relationships": [], "error": str(e), "status": "failed"}

    async def export_snapshot(self) -> Dict[str, Any]:
        """
        Exporta snapshot leve do KG (nós e arestas) com IDs estáveis.
        """
        if not self._initialized:
            await self.initialize()

        driver = self.graphiti.driver

        # NÓS: evita UnknownPropertyKeyWarning usando keys(n) + n[k]
        nodes_q = """
        MATCH (n)
        WITH n, [k IN ['name','title','id'] WHERE k IN keys(n) AND n[k] IS NOT NULL] AS ks
        RETURN
        elementId(n) AS neo4j_id,
        labels(n)    AS labels,
        CASE WHEN size(ks) > 0 THEN toString(n[ks[0]]) ELSE elementId(n) END AS name,
        properties(n) AS properties
        """

        recs, _, _ = await driver.execute_query(nodes_q)

        entities = []
        for r in (recs or []):
            props = _jsonify_props(r.get("properties") or {})
            entities.append({
                "neo4j_id": r.get("neo4j_id"),
                "labels":   r.get("labels") or [],
                "name":     r.get("name"),
                "properties": props,
            })

        # ARESTAS
        rels_q = """
        MATCH (a)-[r]->(b)
        RETURN
        elementId(r) AS rel_id,
        type(r)      AS relation,
        elementId(a) AS src_id,
        elementId(b) AS dst_id,
        properties(r) AS properties
        """
        rrecs, _, _ = await driver.execute_query(rels_q)

        edges = []
        for r in (rrecs or []):
            props = _jsonify_props(r.get("properties") or {})
            edges.append({
                "rel_id":     r.get("rel_id"),
                "relation":   r.get("relation"),
                "src_id":     r.get("src_id"),
                "dst_id":     r.get("dst_id"),
                "properties": props,
            })

        return {"entities": entities, "edges": edges}

    # ----- cost helpers -----
    def get_cost_breakdown(self) -> Dict[str, float]:
        """Compute costs using the internal token meters and env pricing."""
        prompt = int(self.metrics.tokens_llm_prompt or 0)
        completion = int(self.metrics.tokens_llm_completion or 0)
        emb = int(self.metrics.tokens_embeddings or 0)
        rerank = int(self.metrics.tokens_reranker or 0)

        chat_prompt_cost = (prompt / 1000.0) * PRICES["chat_prompt_per_1k"]
        chat_completion_cost = (completion / 1000.0) * PRICES["chat_completion_per_1k"]
        emb_cost = (emb / 1000.0) * PRICES["embedding_per_1k"]
        rerank_cost = (rerank / 1000.0) * PRICES["reranker_per_1k"]

        total = chat_prompt_cost + chat_completion_cost + emb_cost + rerank_cost
        return {
            "chat_prompt_cost": chat_prompt_cost,
            "chat_completion_cost": chat_completion_cost,
            "embedding_cost": emb_cost,
            "reranker_cost": rerank_cost,
            "total_cost_usd": total,
        }

    def get_metrics_with_cost(self) -> Dict[str, Any]:
        """Return metrics + per-bucket costs + totals."""
        m = self.get_metrics()
        costs = self.get_cost_breakdown()
        return {
            **m,
            **costs,
            "total_tokens": int(m["tokens_llm_prompt"]) + int(m["tokens_llm_completion"]) +
                            int(m["tokens_embeddings"]) + int(m["tokens_reranker"]),
        }

    async def clear_graph(self) -> Dict[str, Any]:
        if not self._initialized:
            await self.initialize()
        try:
            await clear_data(self.graphiti.driver)
            logger.warning("Cleared all data from knowledge graph")
            return {"status": "success", "message": "Graph cleared"}
        except Exception as e:
            logger.error(f"Failed to clear graph: {e}")
            return {"status": "failed", "error": str(e)}

    # ----- metrics -----
    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics.to_dict()


# Global instance
graph_client = GraphitiClient()


# ============================================================================
# PUBLIC API (thin wrappers around the singleton)
# ============================================================================

async def initialize_graph() -> None:
    await graph_client.initialize()


async def close_graph() -> None:
    await graph_client.close()


async def add_episode(
    episode_id: str,
    content: str,
    source: str,
    timestamp: Optional[datetime] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return await graph_client.add_episode(episode_id, content, source, timestamp, metadata)


async def search_graph(query: str, limit: int = 10, **kwargs) -> List[Dict[str, Any]]:
    return await graph_client.search(query, limit, **kwargs)


async def get_embedding(text: str) -> List[float]:
    """Generates an embedding for a single string of text."""
    return await graph_client.get_embedding(text)


async def get_entity_relationships(entity: str, depth: int = 1) -> Dict[str, Any]:
    return await graph_client.get_entity_relationships(entity, depth)


async def get_graph_stats() -> Dict[str, Any]:
    return await graph_client.get_graph_statistics()


async def clear_graph() -> Dict[str, Any]:
    return await graph_client.clear_graph()


async def test_graph_connection() -> bool:
    try:
        await graph_client.initialize()
        stats = await graph_client.get_graph_statistics()
        logger.info(f"Graph connection successful. Stats: {stats}")
        return stats.get("status") == "healthy"
    except Exception as e:
        logger.error(f"Graph connection test failed: {e}")
        return False


def get_graph_metrics() -> Dict[str, Any]:
    return graph_client.get_metrics()
