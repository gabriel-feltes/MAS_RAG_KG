# ingestion/graph_builder.py
"""
Knowledge graph builder for extracting entities and relationships from scientific papers.
Aligned with GraphitiClient metrics: collects per-document token/cost deltas and graph deltas.

NEW: Syncs KG to PostgreSQL with embeddings and calculated edge weights.

Changes vs. previous:
- After adding all episodes, waits briefly and reads GraphitiClient.get_metrics() again.
- Computes per-document token deltas; if zero, falls back to tokens_estimated delta.
- Computes cost from detailed buckets when available; else uses fallback blended price.
- Syncs entities with generated embeddings and edges with calculated weights.
"""

from __future__ import annotations

import os
import logging
import asyncio
import re
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timezone

from dotenv import load_dotenv
from .chunker import DocumentChunk

# Import graph utilities with fallback
try:
    from ..agent.db_utils import db_pool  # noqa: F401 (compat)
    from ..agent.graph_utils import GraphitiClient, get_embedding
except ImportError:  # pragma: no cover
    import sys
    import os as _os
    sys.path.append(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from agent.db_utils import db_pool  # type: ignore  # noqa: F401
    from agent.graph_utils import GraphitiClient, get_embedding  # type: ignore

load_dotenv()
logger = logging.getLogger(__name__)

# ------------------------------ Pricing -------------------------------------
PRICES = {
    "chat_prompt_per_1k": float(os.getenv("PRICE_CHAT_PROMPT_PER_1K", "0.00015")),
    "chat_completion_per_1k": float(os.getenv("PRICE_CHAT_COMPLETION_PER_1K", "0.0006")),
    "embedding_per_1k": float(os.getenv("PRICE_EMBEDDING_PER_1K", "0.00002")),
    "reranker_per_1k": float(os.getenv("PRICE_RERANKER_PER_1K", "0.0005")),
    "graph_fallback_per_1k": float(os.getenv("PRICE_GRAPH_FALLBACK_PER_1K", "0.01")),
}


def _safe_int(v: Any) -> int:
    try:
        return int(v or 0)
    except Exception:
        return 0


def _metrics_delta(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, int]:
    """Delta ONLY for real buckets reported by GraphitiClient wrappers."""
    keys = ["tokens_llm_prompt", "tokens_llm_completion", "tokens_embeddings", "tokens_reranker"]
    out: Dict[str, int] = {}
    for k in keys:
        out[k] = _safe_int(after.get(k)) - _safe_int(before.get(k))
        if out[k] < 0:
            # protect against counter resets
            out[k] = _safe_int(after.get(k))
    out["total_tokens"] = sum(out[k] for k in keys)
    # also carry estimated tokens delta (for fallback logic)
    out["tokens_estimated_delta"] = _safe_int(after.get("tokens_estimated")) - _safe_int(before.get("tokens_estimated"))
    if out["tokens_estimated_delta"] < 0:
        out["tokens_estimated_delta"] = _safe_int(after.get("tokens_estimated"))
    return out


def _compute_cost_from_tokens(delta: Dict[str, int]) -> Dict[str, float]:
    """Compute cost from real buckets; no fallback here."""
    prompt = _safe_int(delta.get("tokens_llm_prompt"))
    completion = _safe_int(delta.get("tokens_llm_completion"))
    emb = _safe_int(delta.get("tokens_embeddings"))
    rerank = _safe_int(delta.get("tokens_reranker"))

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


def _calculate_edge_weight(edge_props: Dict[str, Any]) -> float:
    """
    Calculate edge weight based on recency.
    Base weight = 0.7 for all meaningful edges.
    Recent edges get slightly higher weight.
    """
    base_weight = 0.7
    
    created_at = edge_props.get('created_at')
    if not created_at:
        return base_weight
    
    try:
        created = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
        age_days = (datetime.now(timezone.utc) - created).days
        
        # Recency adjustment
        if age_days < 1:
            return 0.8  # Very recent
        elif age_days < 7:
            return 0.75  # Recent
        else:
            return 0.6  # Older
    except Exception:
        return base_weight


# -------------------------- Local metrics tracker ----------------------------

class GraphMetricsTracker:
    """Local builder metrics (not GraphitiClient's internal counters)."""

    def __init__(self) -> None:
        self.episodes_created = 0
        self.relationships_discovered = 0
        self.entities_extracted = 0

        self.errors: List[str] = []
        self.retry_attempts = 0
        self.max_retry_attempts = 0

        # Filled from deltas (per document)
        self.tokens_llm_prompt = 0
        self.tokens_llm_completion = 0
        self.tokens_embeddings = 0
        self.tokens_reranker = 0
        self.total_llm_tokens = 0
        self.total_llm_cost_usd = 0.0

        # Fallback reporting flag
        self.used_estimated_fallback = False

    def add_error(self, error: str) -> None:
        self.errors.append(error)
        logger.error(f"Graph error: {error}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episodes_created": self.episodes_created,
            "relationships_discovered": self.relationships_discovered,
            "entities_extracted": self.entities_extracted,
            "retry_attempts": self.retry_attempts,
            "max_retry_attempts": self.max_retry_attempts,
            "errors_count": len(self.errors),
            "tokens_llm_prompt": self.tokens_llm_prompt,
            "tokens_llm_completion": self.tokens_llm_completion,
            "tokens_embeddings": self.tokens_embeddings,
            "tokens_reranker": self.tokens_reranker,
            "total_llm_tokens": self.total_llm_tokens,
            "total_llm_cost_usd": round(self.total_llm_cost_usd, 6),
            "used_estimated_fallback": self.used_estimated_fallback,
        }


# ------------------------ Heuristic entity extractor -------------------------

class AcademicEntityExtractor:
    def __init__(self) -> None:
        self.ml_techniques = {
            "Transformer", "LSTM", "GRU", "CNN", "RNN", "MLP",
            "Attention", "Self-Attention", "Multi-Head Attention",
            "ResNet", "VGG", "BERT", "GPT", "Vision Transformer", "ViT",
            "Fine-tuning", "Transfer Learning", "Few-shot Learning",
            "Zero-shot Learning", "Prompt Engineering", "In-context Learning",
            "Retrieval Augmented Generation", "RAG", "Knowledge Graph",
            "Graph Neural Network", "GNN", "Message Passing",
            "Gradient Descent", "Adam", "SGD", "Learning Rate Scheduling",
            "Backpropagation", "Federated Learning", "Distributed Training",
            "Machine Translation", "Question Answering",
            "Text Summarization", "Sentiment Analysis", "NER",
            "Semantic Similarity", "Relation Extraction",
        }
        self.academic_concepts = {
            "Multi-Agent System", "Agent", "Reinforcement Learning",
            "Reward Function", "Policy", "Value Function",
            "Coordination", "Communication Protocol", "Consensus",
            "Swarm Intelligence", "Emergent Behavior",
            "Knowledge Representation", "Ontology", "Semantic Web",
            "Belief-Desire-Intention", "BDI", "Temporal Logic",
            "Neuro-Symbolic", "Hybrid Intelligence",
        }
        self.datasets = {
            "ImageNet", "COCO", "Pascal VOC", "MNIST", "CIFAR",
            "SQuAD", "GLUE", "SuperGLUE", "WIKITEXT", "BookCorpus",
            "Wikipedia", "Common Crawl", "The Pile", "Stack",
        }
        self.venues = {
            "NeurIPS", "ICML", "ICLR", "IJCAI", "AAAI",
            "ACL", "EMNLP", "NAACL", "ICCV", "ECCV", "CVPR",
            "IEEE", "Nature", "Science", "JMLR", "TPAMI",
        }

    def _find_word(self, text: str, word: str) -> bool:
        return bool(re.search(r"\b" + re.escape(word) + r"\b", text, re.IGNORECASE))

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        entities: Dict[str, Set[str]] = {
            "techniques": set(), "concepts": set(), "datasets": set(),
            "venues": set(), "metrics": set(), "people": set(),
        }
        tl = text.lower()

        for t in self.ml_techniques:
            if self._find_word(tl, t.lower()):
                entities["techniques"].add(t)
        for c in self.academic_concepts:
            if self._find_word(tl, c.lower()):
                entities["concepts"].add(c)
        for d in self.datasets:
            if self._find_word(tl, d.lower()):
                entities["datasets"].add(d)
        for v in self.venues:
            if self._find_word(tl, v.lower()):
                entities["venues"].add(v)

        # metrics & percents
        for pat in [r"\b(accuracy|precision|recall|f1(?:-score)?|bleu|rouge|perplexity|auc|rmse)\b",
                    r"\b\d+(?:\.\d+)?%\b"]:
            for m in re.findall(pat, tl):
                if isinstance(m, tuple):
                    entities["metrics"].add(m[0])
                else:
                    entities["metrics"].add(m)

        for person in {"Yann LeCun", "Yoshua Bengio", "Geoffrey Hinton",
                       "Andrew Ng", "Fei-Fei Li", "Demis Hassabis",
                       "Goodfellow", "Kingma", "Rezende", "Karpathy"}:
            if person in text:
                entities["people"].add(person)

        return {k: sorted(v) for k, v in entities.items()}


# ------------------------------- GraphBuilder --------------------------------

class GraphBuilder:
    """Builds KG with retries and returns accurate token/cost deltas per document."""

    def __init__(self, max_retries: int = 5, retry_delay: float = 2.0) -> None:
        self.graph_client = GraphitiClient()
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.entity_extractor = AcademicEntityExtractor()
        self.metrics = GraphMetricsTracker()
        self._initialized = False
        self.db = None  # Will be set from ingestion pipeline

    async def initialize(self) -> None:
        if not self._initialized:
            await self.graph_client.initialize()
            self._initialized = True
            logger.info("Graph builder initialized")

    async def close(self) -> None:
        if self._initialized:
            await self.graph_client.close()
            self._initialized = False

    async def add_document_to_graph(
        self,
        chunks: List[DocumentChunk],
        document_title: str,
        document_source: str,
        document_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self._initialized:
            await self.initialize()
        if not chunks:
            return {"episodes_created": 0, "nodes_created": 0, "relationships_created": 0, "errors": []}

        logger.info(f"Adding {len(chunks)} chunks to graph for: {document_title}")

        # Graph stats snapshot (for deltas)
        try:
            before_stats = await self.graph_client.get_graph_statistics()
            before_nodes = int(before_stats.get("total_nodes", 0))
            before_rels = int(before_stats.get("total_relationships", 0))
        except Exception:
            before_nodes = before_rels = 0

        # Graphiti metrics snapshot (for token/cost deltas)
        before_metrics = dict(self.graph_client.get_metrics() or {})

        episodes_created = 0
        errors: List[str] = []

        for i, chunk in enumerate(chunks):
            for attempt in range(self.max_retries):
                try:
                    self.metrics.max_retry_attempts = max(self.metrics.max_retry_attempts, attempt)

                    episode_id = f"{document_source}_{chunk.index}_{int(datetime.now().timestamp())}"
                    episode_content = self._prepare_episode_for_academic_context(
                        chunk, document_title, document_metadata
                    )

                    await self.graph_client.add_episode(
                        episode_id=episode_id,
                        content=episode_content,
                        source=f"Paper: {document_title} (Chunk {chunk.index})",
                        timestamp=datetime.now(timezone.utc),
                        metadata={
                            "document_title": document_title,
                            "document_source": document_source,
                            "chunk_index": chunk.index,
                            "content_length": len(chunk.content),
                            "chunk_topic": self._infer_chunk_topic(chunk.content),
                            **(document_metadata or {}),
                        },
                    )

                    episodes_created += 1
                    logger.info(f"✓ Chunk {i}/{len(chunks)-1} added (attempt {attempt + 1})")
                    if i < len(chunks) - 1:
                        await asyncio.sleep(0.3)
                    break

                except Exception as e:  # pragma: no cover
                    if attempt == self.max_retries - 1:
                        msg = f"Chunk {chunk.index} (attempt {attempt + 1}/{self.max_retries}): {e}"
                        logger.error(f"✗ {msg}")
                        errors.append(msg)
                        self.metrics.add_error(msg)
                        break
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Retry chunk {chunk.index} in {delay}s ({e})")
                    await asyncio.sleep(delay)

        logger.info(f"Graph building complete: {episodes_created} episodes, {len(errors)} errors")

        # Give Graphiti/clients a moment to flush internal counters
        await asyncio.sleep(0.75)

        # Final snapshots
        try:
            after_stats = await self.graph_client.get_graph_statistics()
            after_nodes = int(after_stats.get("total_nodes", 0))
            after_rels = int(after_stats.get("total_relationships", 0))
        except Exception:
            after_nodes, after_rels = before_nodes, before_rels

        nodes_delta = max(after_nodes - before_nodes, 0)
        rels_delta = max(after_rels - before_rels, 0)

        after_metrics = dict(self.graph_client.get_metrics() or {})
        delta_tokens = _metrics_delta(before_metrics, after_metrics)

        # Compute cost from real buckets first
        delta_costs = _compute_cost_from_tokens(delta_tokens)

        # If everything is still zero, FALL BACK to estimated tokens
        used_estimated_fallback = False
        if (delta_tokens["total_tokens"] == 0) and (delta_tokens["tokens_estimated_delta"] > 0):
            used_estimated_fallback = True
            est = delta_tokens["tokens_estimated_delta"]
            # attribute all to prompt bucket for reporting
            self.metrics.tokens_llm_prompt = est
            self.metrics.tokens_llm_completion = 0
            self.metrics.tokens_embeddings = 0
            self.metrics.tokens_reranker = 0
            self.metrics.total_llm_tokens = est
            self.metrics.total_llm_cost_usd = (est / 1000.0) * PRICES["graph_fallback_per_1k"]
        else:
            # use real buckets
            self.metrics.tokens_llm_prompt = delta_tokens["tokens_llm_prompt"]
            self.metrics.tokens_llm_completion = delta_tokens["tokens_llm_completion"]
            self.metrics.tokens_embeddings = delta_tokens["tokens_embeddings"]
            self.metrics.tokens_reranker = delta_tokens["tokens_reranker"]
            self.metrics.total_llm_tokens = delta_tokens["total_tokens"]
            self.metrics.total_llm_cost_usd = delta_costs["total_cost_usd"]

        self.metrics.episodes_created = episodes_created
        self.metrics.entities_extracted = nodes_delta
        self.metrics.relationships_discovered = rels_delta
        self.metrics.used_estimated_fallback = used_estimated_fallback

        # Terminal-friendly log (keeps your existing message shape)
        logger.info(
            (
                "Graph: %d relationships, %d nodes. LLM tokens: %d | cost: $%.6f "
                "(prompt=%d, completion=%d, emb=%d, rerank=%d)%s"
            ),
            rels_delta,
            nodes_delta,
            self.metrics.total_llm_tokens,
            self.metrics.total_llm_cost_usd,
            self.metrics.tokens_llm_prompt,
            self.metrics.tokens_llm_completion,
            self.metrics.tokens_embeddings,
            self.metrics.tokens_reranker,
            " [estimated]" if used_estimated_fallback else "",
        )

        # Build return payload
        payload = {
            "episodes_created": episodes_created,
            "total_chunks": len(chunks),
            "errors": errors,
            "nodes_created": nodes_delta,
            "relationships_created": rels_delta,
            "tokens_llm_prompt": self.metrics.tokens_llm_prompt,
            "tokens_llm_completion": self.metrics.tokens_llm_completion,
            "tokens_embeddings": self.metrics.tokens_embeddings,
            "tokens_reranker": self.metrics.tokens_reranker,
            "total_llm_tokens": self.metrics.total_llm_tokens,
            "total_cost_usd": self.metrics.total_llm_cost_usd,
            "metrics": self.metrics.to_dict(),
        }

        if not used_estimated_fallback:
            payload["graph_cost_breakdown"] = {
                "chat_prompt_cost": (self.metrics.tokens_llm_prompt / 1000.0) * PRICES["chat_prompt_per_1k"],
                "chat_completion_cost": (self.metrics.tokens_llm_completion / 1000.0) * PRICES["chat_completion_per_1k"],
                "embedding_cost": (self.metrics.tokens_embeddings / 1000.0) * PRICES["embedding_per_1k"],
                "reranker_cost": (self.metrics.tokens_reranker / 1000.0) * PRICES["reranker_per_1k"],
            }
        else:
            payload["graph_cost_breakdown"] = {
                "fallback_graph_cost": self.metrics.total_llm_cost_usd
            }

        return payload

    async def sync_kg_to_postgres(self) -> Dict[str, int]:
        """
        Sync Neo4j KG to PostgreSQL with:
        - Generated embeddings for entities
        - Calculated weights for edges
        
        Returns counts of synced entities and edges.
        """
        if not self.db:
            logger.warning("Database connection not set - skipping KG sync")
            return {"entities": 0, "edges": 0}
        
        logger.info("Syncing knowledge graph to PostgreSQL...")
        
        # Export snapshot from Neo4j
        snapshot = await self.graph_client.export_snapshot()
        
        # ENTITIES: Generate embeddings
        entities_data = []
        for entity in snapshot['entities']:
            try:
                # Generate embedding for entity name
                embedding = await get_embedding(entity['name'])
            except Exception as e:
                logger.warning(f"Failed to generate embedding for '{entity['name']}': {e}")
                embedding = None
            
            entities_data.append({
                'neo4j_id': entity['neo4j_id'],
                'name': entity['name'],
                'labels': entity['labels'],
                'properties': entity['properties'],
                'embedding': embedding  # ✅ Generated
            })
        
        # EDGES: Calculate weights
        edges_data = []
        for edge in snapshot['edges']:
            props = edge.get('properties', {})
            weight = _calculate_edge_weight(props)  # ✅ Calculated
            
            edges_data.append({
                'neo4j_id': edge['rel_id'],
                'source_id': edge['src_id'],
                'target_id': edge['dst_id'],
                'relation': edge['relation'],
                'properties': props,
                'weight': weight
            })
        
        # Bulk insert to PostgreSQL
        await self.db.bulk_insert_entities(entities_data)
        await self.db.bulk_insert_edges(edges_data)
        
        logger.info(f"Synced {len(entities_data)} entities (with embeddings) and {len(edges_data)} edges (with weights)")
        
        return {
            "entities": len(entities_data),
            "edges": len(edges_data)
        }

    def _prepare_episode_for_academic_context(
        self,
        chunk: DocumentChunk,
        document_title: str,
        document_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        max_length = 5500
        content = chunk.content
        if len(content) > max_length:
            truncated = content[:max_length]
            last_period = max(truncated.rfind(". "), truncated.rfind(".\n"))
            content = truncated[: last_period + 1] if last_period > int(max_length * 0.7) else truncated
            content += "\n\n[... content truncated for processing]"

        key_entities = self.entity_extractor.extract_entities(chunk.content)
        header = f"# {document_title}\n"
        if key_entities.get("techniques"):
            header += f"**Techniques**: {', '.join(key_entities['techniques'][:5])}\n"
        if key_entities.get("concepts"):
            header += f"**Concepts**: {', '.join(key_entities['concepts'][:5])}\n"
        header += "\n---\n\n"
        return header + content

    def _infer_chunk_topic(self, content: str) -> str:
        content_lower = content.lower()
        topics = {
            "abstract": ["abstract", "summary", "overview"],
            "introduction": ["introduction", "motivation", "background"],
            "methodology": ["method", "approach", "propose", "algorithm"],
            "experiments": ["experiment", "evaluation", "results", "benchmark"],
            "conclusion": ["conclusion", "future work", "discussion"],
            "related_work": ["related", "prior", "existing", "previous"],
        }
        for topic, kws in topics.items():
            for kw in kws:
                if kw in content_lower[:500]:
                    return topic
        return "content"

    async def extract_entities_from_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        logger.info(f"Extracting academic entities from {len(chunks)} chunks")
        enriched: List[DocumentChunk] = []
        for chunk in chunks:
            entities = self.entity_extractor.extract_entities(chunk.content)
            self.metrics.entities_extracted += sum(len(v) for v in entities.values())
            enriched_chunk = DocumentChunk(
                content=chunk.content,
                index=chunk.index,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                metadata={
                    **(chunk.metadata or {}),
                    "academic_entities": entities,
                    "entity_extraction_date": datetime.now().isoformat(),
                },
                token_count=getattr(chunk, "token_count", None),
            )
            if hasattr(chunk, "embedding"):
                enriched_chunk.embedding = chunk.embedding  # type: ignore[attr-defined]
            enriched.append(enriched_chunk)
        logger.info(f"Entity extraction complete: {self.metrics.entities_extracted} entities (cumulative)")
        return enriched

    async def clear_graph(self) -> None:
        if not self._initialized:
            await self.initialize()
        logger.warning("Clearing knowledge graph...")
        await self.graph_client.clear_graph()
        logger.info("Knowledge graph cleared")

    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics.to_dict()


def create_graph_builder(max_retries: int = 5, retry_delay: float = 2.0) -> GraphBuilder:
    return GraphBuilder(max_retries=max_retries, retry_delay=retry_delay)
