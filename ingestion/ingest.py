# ingestion/ingest.py
from __future__ import annotations

"""
Main ingestion script for processing markdown documents into vector DB and knowledge graph.
Includes comprehensive metrics tracking for tokens, costs, and performance.
With resumable checkpoint support to avoid re-processing and wasting money.

This version relies on per-document token/cost deltas returned by GraphBuilder.add_document_to_graph(),
so logs and totals show real usage for each file.
It also exports a Neo4j snapshot and mirrors it into Postgres (kg_entities / kg_edges).
"""

import os
import asyncio
import logging
import json
import glob
import uuid
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
import argparse

from dotenv import load_dotenv
import asyncpg

from .chunker import ChunkingConfig, create_chunker, DocumentChunk
from .embedder import create_embedder
from .graph_builder import create_graph_builder

# Import agent utilities
try:
    from ..agent.db_utils import initialize_database, close_database, db_pool  # type: ignore
    from ..agent.graph_utils import initialize_graph, close_graph  # type: ignore
    from ..agent.models import IngestionConfig, IngestionResult  # type: ignore
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from agent.db_utils import initialize_database, close_database, db_pool  # type: ignore
    from agent.graph_utils import initialize_graph, close_graph  # type: ignore
    from agent.models import IngestionConfig, IngestionResult  # type: ignore

load_dotenv()
logger = logging.getLogger(__name__)


# ============================================================================
# Ingestion metrics collector
# ============================================================================

class IngestionMetricsCollector:
    """Coleta e rastreia métricas de ingestão (tokens, custo, tempo)."""

    def __init__(self, ingestion_session_id: str):
        self.session_id = ingestion_session_id
        self.start_time = datetime.now()

        # Métricas agregadas
        self.documents_count = 0
        self.documents_success = 0
        self.documents_failed = 0
        self.chunks_count = 0
        self.entities_count = 0
        self.relationships_count = 0

        # Métricas de LLM
        self.embedding_tokens = 0
        self.embedding_cost_usd = 0.0
        self.graph_tokens = 0
        self.graph_cost_usd = 0.0

        self.errors: List[str] = []

    def add_document_metrics(
        self,
        chunks_count: int,
        entities_count: int,
        relationships_count: int,
        embedding_tokens: int,
        embedding_cost: float,
        graph_tokens: int = 0,
        graph_cost: float = 0.0,
        success: bool = True,
    ) -> None:
        self.documents_count += 1
        if success:
            self.documents_success += 1
        else:
            self.documents_failed += 1

        self.chunks_count += chunks_count
        self.entities_count += entities_count
        self.relationships_count += relationships_count
        self.embedding_tokens += embedding_tokens
        self.embedding_cost_usd += embedding_cost
        self.graph_tokens += graph_tokens
        self.graph_cost_usd += graph_cost

    def add_error(self, error: str) -> None:
        self.errors.append(error)
        logger.error(f"Ingestion error: {error}")

    def get_total_cost_usd(self) -> float:
        return self.embedding_cost_usd + self.graph_cost_usd

    def get_total_tokens(self) -> int:
        return self.embedding_tokens + self.graph_tokens

    def get_elapsed_time(self) -> float:
        return (datetime.now() - self.start_time).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        elapsed = self.get_elapsed_time()
        return {
            "session_id": self.session_id,
            "timestamp": self.start_time.isoformat(),
            "documents_processed": self.documents_count,
            "documents_success": self.documents_success,
            "documents_failed": self.documents_failed,
            "chunks_count": self.chunks_count,
            "entities_count": self.entities_count,
            "relationships_count": self.relationships_count,
            "embedding_tokens": self.embedding_tokens,
            "embedding_cost_usd": round(self.embedding_cost_usd, 6),
            "graph_tokens": self.graph_tokens,
            "graph_cost_usd": round(self.graph_cost_usd, 6),
            "total_tokens": self.get_total_tokens(),
            "total_cost_usd": round(self.get_total_cost_usd(), 6),
            "elapsed_seconds": round(elapsed, 2),
            "errors_count": len(self.errors),
            "errors": self.errors,
        }


# ============================================================================
# Helpers
# ============================================================================

def _jsonify(obj: Any) -> Any:
    """Converte objetos retornados pelo driver Neo4j para JSON-safe."""
    for m in ("iso_format", "isoformat"):
        if hasattr(obj, m):
            try:
                return getattr(obj, m)()
            except Exception:
                pass
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, (bytes, bytearray)):
        return obj.decode("utf-8", errors="replace")
    return obj


# ============================================================================
# Document ingestion pipeline
# ============================================================================

class DocumentIngestionPipeline:
    """Pipeline para ingerir documentos em vector DB e knowledge graph."""

    def __init__(
        self,
        config: IngestionConfig,
        documents_folder: str = "documents",
        clean_before_ingest: bool = False,
        resume_mode: bool = False,
        kg_sync_only: bool = False,
    ) -> None:
        self.config = config
        self.documents_folder = documents_folder
        self.clean_before_ingest = clean_before_ingest
        self.resume_mode = resume_mode
        self.kg_sync_only = kg_sync_only

        self.ingestion_session_id = str(uuid.uuid4())
        self.metrics = IngestionMetricsCollector(self.ingestion_session_id)

        self.chunker_config = ChunkingConfig()
        self.chunker = create_chunker(self.chunker_config)
        self.embedder = create_embedder()
        self.graph_builder = create_graph_builder()

        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return
        logger.info(
            f"Initializing ingestion pipeline (session: {self.ingestion_session_id[:8]}...)"
        )
        await initialize_database()
        await initialize_graph()
        await self.graph_builder.initialize()
        self._initialized = True
        logger.info("Ingestion pipeline initialized")

    async def close(self) -> None:
        if self._initialized:
            await self.graph_builder.close()
            await close_graph()
            await close_database()
            self._initialized = False

    async def ingest_documents(
        self, progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List["IngestionResult"]:
        """Ingest all documents from the folder, with summary support."""
        if not self._initialized:
            await self.initialize()

        if self.clean_before_ingest:
            await self._clean_databases()
            logger.info("Full clean requested - reprocessing all documents")

        # Create / ensure ingestion run exists
        async with db_pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    INSERT INTO ingestion_runs (ingestion_session_id, created_at, updated_at)
                    VALUES ($1::uuid, NOW(), NOW())
                    ON CONFLICT (ingestion_session_id) DO NOTHING;
                    """,
                    self.ingestion_session_id,
                )
        logger.info(f"Started ingestion run: {self.ingestion_session_id}")

        if self.kg_sync_only:
            logger.info("KG sync only: exporting Neo4j snapshot and syncing to Postgres...")
            await self._export_and_sync_kg()
            await self._save_ingestion_run()
            return []

        markdown_files = self._find_markdown_files()
        if not markdown_files:
            logger.warning(f"No markdown files found in {self.documents_folder}")
            # Mesmo sem arquivos, exporta/sincroniza o KG já existente
            await self._export_and_sync_kg()
            await self._save_ingestion_run()
            return []

        logger.info(f"Found {len(markdown_files)} markdown files to process")
        results: List[IngestionResult] = []

        for i, file_path in enumerate(markdown_files, start=1):
            doc_source = os.path.basename(file_path)
            logger.info(f"[{i}/{len(markdown_files)}] Processing: {doc_source}")
            try:
                result = await self._ingest_single_document(file_path, doc_source)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}", exc_info=True)
                self.metrics.add_error(str(e))
                continue
            if progress_callback:
                progress_callback(i, len(markdown_files))

        # After all documents: export Neo4j snapshot and persist to Postgres KG tables
        await self._export_and_sync_kg()

        await self._save_ingestion_run()
        return results

    async def _get_progress(self, document_source: str) -> Optional[Dict[str, Any]]:
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM ingestion_progress WHERE document_source=$1",
                document_source,
            )
            return dict(row) if row else None

    async def _update_progress(
        self,
        document_source: str,
        status: str,
        document_id: Optional[str] = None,
        checkpoint: Optional[Dict[str, Any]] = None,
    ) -> None:
        async with db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO ingestion_progress (document_source, status, document_id, checkpoint)
                VALUES ($1, $2, $3::uuid, $4::jsonb)
                ON CONFLICT (document_source) DO UPDATE SET
                    status = EXCLUDED.status,
                    document_id = COALESCE(EXCLUDED.document_id, ingestion_progress.document_id),
                    checkpoint = COALESCE(EXCLUDED.checkpoint, ingestion_progress.checkpoint),
                    updated_at = NOW()
                """,
                document_source,
                status,
                document_id,
                json.dumps(checkpoint) if checkpoint else None,
            )

    async def _ingest_single_document(
        self, file_path: str, document_source: str
    ) -> "IngestionResult":
        doc_start_time = datetime.now()
        document_id = str(uuid.uuid4())

        initial_entities_count = 0
        final_entities_count = 0
        relationships_count = 0
        graph_tokens = 0
        graph_cost = 0.0
        graph_errors: List[str] = []

        try:
            # 1) Read & prepare
            document_content = self._read_document(file_path)
            document_title = self._extract_title(document_content, file_path)
            document_metadata = self._extract_document_metadata(
                document_content, file_path
            )
            logger.info(f"Processing: {document_title}")

            # 2) Chunking
            chunks = self.chunker.chunk_document(
                content=document_content,
                title=document_title,
                source=document_source,
                metadata=document_metadata,
            )
            if not chunks:
                logger.warning(f"No chunks for {document_title}")
                await self._save_document_metrics(
                    document_id,
                    document_title,
                    document_source,
                    0, 0, 0, 0, 0, 0, 0,
                    (datetime.now() - doc_start_time).total_seconds() * 1000,
                    "no_chunks",
                )
                await self._update_progress(document_source, "failed")
                self.metrics.add_document_metrics(0, 0, 0, 0, 0, success=False)
                return IngestionResult(
                    document_id="",
                    title=document_title,
                    chunks_created=0,
                    entities_extracted=0,
                    relationships_created=0,
                    processing_time_ms=(datetime.now() - doc_start_time).total_seconds()
                    * 1000,
                    errors=["No chunks created"],
                )

            logger.info(f"Created {len(chunks)} chunks")

            # 3) Pre-extraction of entities
            if self.config.extract_entities:
                chunks = await self.graph_builder.extract_entities_from_chunks(chunks)
                initial_entities_count = sum(
                    len(entity_list)
                    for chunk in chunks
                    for _, entity_list in chunk.metadata.get("academic_entities", {}).items()
                )
                logger.info(
                    f"Extracted {initial_entities_count} initial entities (may be refined by Graph Builder)"
                )
            final_entities_count = initial_entities_count

            # 4) Embeddings
            embedded_chunks = await self.embedder.embed_chunks(chunks)
            embedding_metrics = self.embedder.get_metrics()
            logger.info(
                f"Embeddings: {embedding_metrics['total_tokens']} tokens, ${embedding_metrics['total_cost_usd']:.6f}"
            )

            # 5) Persist to Postgres (vector DB)
            document_id = await self._save_to_postgres(
                document_title,
                document_source,
                document_content,
                embedded_chunks,
                document_metadata,
            )
            logger.info(f"Saved to PostgreSQL: {document_id}")

            # 6) Checkpoints for vector DB
            await self._update_progress(document_source, "chunks_done", document_id)
            await self._update_progress(document_source, "embedded_done", document_id)

            # 7) Knowledge graph build (Neo4j)
            if not self.config.skip_graph_building:
                try:
                    logger.info("Building knowledge graph...")
                    graph_result = await self.graph_builder.add_document_to_graph(
                        chunks=embedded_chunks,
                        document_title=document_title,
                        document_source=document_source,
                        document_metadata=document_metadata,
                    )

                    # deltas do grafo
                    relationships_count = int(graph_result.get("relationships_created", 0))
                    final_entities_count = max(
                        final_entities_count, int(graph_result.get("nodes_created", 0))
                    )

                    # --- Consome tokens/custos reais retornados pelo builder (por documento) ---
                    graph_tokens = int(graph_result.get("total_llm_tokens", 0))
                    breakdown = graph_result.get("graph_cost_breakdown", {}) or {}
                    graph_cost = float(graph_result.get("total_cost_usd", 0.0))

                    prompt_t = int(graph_result.get("tokens_llm_prompt", 0))
                    completion_t = int(graph_result.get("tokens_llm_completion", 0))
                    emb_t = int(graph_result.get("tokens_embeddings", 0))
                    rerank_t = int(graph_result.get("tokens_reranker", 0))

                    logger.info(
                        (
                            "Graph: %d relationships, %d nodes. LLM tokens: %d | cost: $%.6f "
                            "(prompt=%d, completion=%d, emb=%d, rerank=%d)"
                        ),
                        relationships_count,
                        final_entities_count,
                        graph_tokens,
                        graph_cost,
                        prompt_t,
                        completion_t,
                        emb_t,
                        rerank_t,
                    )

                except Exception as e:
                    error_msg = f"Graph building failed: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    graph_errors.append(error_msg)
                    self.metrics.add_error(error_msg)

            # 8) Checkpoint (graph)
            await self._update_progress(document_source, "graph_done", document_id)

            # 9) Aggregate + persist per-document metrics
            processing_time = (datetime.now() - doc_start_time).total_seconds() * 1000
            self.metrics.add_document_metrics(
                chunks_count=len(chunks),
                entities_count=final_entities_count,
                relationships_count=relationships_count,
                embedding_tokens=embedding_metrics["total_tokens"],
                embedding_cost=embedding_metrics["total_cost_usd"],
                graph_tokens=graph_tokens,
                graph_cost=graph_cost,
                success=True,
            )
            await self._save_document_metrics(
                document_id,
                document_title,
                document_source,
                len(chunks),
                final_entities_count,
                relationships_count,
                embedding_metrics["total_tokens"],
                embedding_metrics["total_cost_usd"],
                graph_tokens,
                graph_cost,
                processing_time,
                "success",
            )
            await self._update_progress(
                document_source,
                "success",
                document_id,
                checkpoint={
                    "chunks": len(chunks),
                    "entities": final_entities_count,
                    "relationships": relationships_count,
                    "embedding_tokens": embedding_metrics["total_tokens"],
                    "graph_tokens": graph_tokens,
                    "processing_time_ms": processing_time,
                },
            )

            # 10) Export Neo4j snapshot and sync to Postgres KG tables
            await self._export_and_sync_kg()

            total_cost_doc = embedding_metrics["total_cost_usd"] + graph_cost
            logger.info(
                f"Document complete: {len(chunks)} chunks, Total Cost: ${total_cost_doc:.6f}, Time: {processing_time:.0f}ms"
            )

            return IngestionResult(
                document_id=document_id,
                title=document_title,
                chunks_created=len(chunks),
                entities_extracted=final_entities_count,
                relationships_created=relationships_count,
                processing_time_ms=processing_time,
                errors=graph_errors,
            )

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Document processing failed: {error_msg}", exc_info=True)

            processing_time = (datetime.now() - doc_start_time).total_seconds() * 1000
            await self._save_document_metrics(
                document_id,
                "",
                document_source,
                0, 0, 0, 0, 0, 0, 0,
                processing_time,
                "failed",
                error_message=error_msg,
            )
            await self._update_progress(document_source, "failed")
            raise

    # ----- KG snapshot → Postgres sync -----
    async def _export_and_sync_kg(self) -> None:
        """Call GraphitiClient.export_snapshot() and upsert into kg_entities/kg_edges."""
        try:
            graph_client = getattr(self.graph_builder, "graph_client", None)
            if not graph_client:
                logger.warning("Graph client not available on graph_builder; skipping KG sync.")
                return

            snapshot = await graph_client.export_snapshot()  # {entities: [...], edges: [...]}
            await self._sync_snapshot_to_pg(snapshot)
            logger.info(
                f"KG sync done: {len(snapshot.get('entities', []))} entities, {len(snapshot.get('edges', []))} edges"
            )
        except Exception as e:
            logger.error(f"Failed to export/sync KG snapshot: {e}", exc_info=True)
            self.metrics.add_error(f"KG sync failed: {e}")

    async def _sync_snapshot_to_pg(self, snapshot: Dict[str, Any]) -> None:
        """Persist entities/edges into kg_entities/kg_edges.

        Deduplication strategy:
        - Entities: look up by properties->>'neo4j_id'. If found, update; else insert.
        - Edges: check existence by (src_entity_id, dst_entity_id, relation, properties::jsonb).
        """
        entities: List[Dict[str, Any]] = snapshot.get("entities", []) or []
        edges: List[Dict[str, Any]] = snapshot.get("edges", []) or []

        id_map: Dict[str, str] = {}  # neo4j_id -> kg_entities.id

        async with db_pool.acquire() as conn:
            async with conn.transaction():
                # Upsert entities
                for ent in entities:
                    neo4j_id = str(ent.get("neo4j_id"))
                    labels = ent.get("labels") or []
                    name = ent.get("name") or neo4j_id
                    props = ent.get("properties") or {}
                    props["neo4j_id"] = neo4j_id

                    etype = labels[0] if labels else "Entity"
                    canonical_name = name.lower() if isinstance(name, str) else None
                    description = None
                    if isinstance(props, dict):
                        for key in ("description", "desc", "summary"):
                            if key in props and isinstance(props[key], str):
                                description = props[key]
                                break

                    props = _jsonify(props)

                    row = await conn.fetchrow(
                        "SELECT id::text FROM kg_entities WHERE properties ->> 'neo4j_id' = $1",
                        neo4j_id,
                    )
                    if row:
                        kg_id = row["id"]
                        await conn.execute(
                            """
                            UPDATE kg_entities
                               SET type=$2, name=$3, canonical_name=$4, description=$5, properties=$6::jsonb, updated_at=NOW()
                             WHERE id=$1::uuid
                            """,
                            kg_id, etype, name, canonical_name, description, json.dumps(props),
                        )
                    else:
                        kg_id = await conn.fetchval(
                            """
                            INSERT INTO kg_entities (type, name, canonical_name, description, properties)
                            VALUES ($1, $2, $3, $4, $5::jsonb)
                            RETURNING id::text
                            """,
                            etype, name, canonical_name, description, json.dumps(props),
                        )
                    id_map[neo4j_id] = kg_id

                # Upsert edges (best-effort dedupe)
                for rel in edges:
                    src_neo = str(rel.get("src_id"))
                    dst_neo = str(rel.get("dst_id"))
                    relation = rel.get("relation") or "RELATED_TO"
                    rprops = _jsonify(rel.get("properties") or {})

                    src_id = id_map.get(src_neo)
                    dst_id = id_map.get(dst_neo)
                    if not src_id or not dst_id:
                        logger.debug(
                            f"Skipping edge with missing endpoints: {src_neo} -> {dst_neo} ({relation})"
                        )
                        continue

                    exists = await conn.fetchval(
                        """
                        SELECT 1 FROM kg_edges
                         WHERE src_entity_id=$1::uuid AND dst_entity_id=$2::uuid
                           AND relation=$3 AND properties=$4::jsonb
                         LIMIT 1
                        """,
                        src_id, dst_id, relation, json.dumps(rprops),
                    )
                    if exists:
                        continue

                    await conn.execute(
                        """
                        INSERT INTO kg_edges (src_entity_id, dst_entity_id, relation, properties)
                        VALUES ($1::uuid, $2::uuid, $3, $4::jsonb)
                        """,
                        src_id, dst_id, relation, json.dumps(rprops),
                    )

    # ----- persistence helpers -----
    async def _save_document_metrics(
        self,
        document_id: str,
        title: str,
        document_source: str,
        chunks_count: int,
        entities_count: int,
        relationships_count: int,
        embedding_tokens: int,
        embedding_cost: float,
        graph_tokens: int,
        graph_cost: float,
        processing_time_ms: float,
        status: str,
        error_message: Optional[str] = None,
    ) -> None:
        async with db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO ingestion_metrics 
                (ingestion_session_id, document_id, document_title, document_source, 
                 chunks_count, entities_count, relationships_count, 
                 embedding_tokens, embedding_cost_usd, graph_tokens, graph_cost_usd, 
                 processing_time_ms, status, error_message)
                VALUES ($1::uuid, $2::uuid, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                """,
                self.ingestion_session_id,
                document_id,
                title,
                document_source,
                chunks_count,
                entities_count,
                relationships_count,
                embedding_tokens,
                embedding_cost,
                graph_tokens,
                graph_cost,
                processing_time_ms,
                status,
                error_message,
            )

    async def _save_ingestion_run(self) -> None:
        metrics_dict = self.metrics.to_dict()
        async with db_pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE ingestion_runs 
                SET 
                    documents_processed = $2,
                    documents_success = $3,
                    documents_failed = $4,
                    total_chunks_created = $5,
                    total_entities_extracted = $6,
                    total_relationships_created = $7,
                    total_embedding_tokens = $8,
                    total_embedding_cost_usd = $9,
                    total_graph_tokens = $10,
                    total_graph_cost_usd = $11,
                    total_time_seconds = $12,
                    updated_at = NOW()
                WHERE ingestion_session_id = $1::uuid
                """,
                self.ingestion_session_id,
                self.metrics.documents_count,
                self.metrics.documents_success,
                self.metrics.documents_failed,
                self.metrics.chunks_count,
                self.metrics.entities_count,
                self.metrics.relationships_count,
                self.metrics.embedding_tokens,
                self.metrics.embedding_cost_usd,
                self.metrics.graph_tokens,
                self.metrics.graph_cost_usd,
                self.metrics.get_elapsed_time(),
            )
        logger.info(f"Ingestion run updated: {json.dumps(metrics_dict, indent=2)}")

    def _find_markdown_files(self) -> List[str]:
        if not os.path.exists(self.documents_folder):
            logger.error(f"Documents folder not found: {self.documents_folder}")
            return []
        files: List[str] = []
        for pattern in ("*.md", "*.markdown", "*.txt"):
            files.extend(
                glob.glob(os.path.join(self.documents_folder, "**", pattern), recursive=True)
            )
        return sorted(set(files))

    def _read_document(self, file_path: str) -> str:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="latin-1") as f:
                return f.read()

    def _extract_title(self, content: str, file_path: str) -> str:
        for line in content.split("\n")[:10]:
            line = line.strip()
            if line.startswith("# "):
                return line[2:].strip()
        return os.path.splitext(os.path.basename(file_path))[0]

    def _extract_document_metadata(self, content: str, file_path: str) -> Dict[str, Any]:
        lines = content.split("\n")
        return {
            "file_path": file_path,
            "file_size": len(content),
            "line_count": len(lines),
            "word_count": len(content.split()),
            "ingestion_date": datetime.now().isoformat(),
        }

    async def _save_to_postgres(
        self,
        title: str,
        source: str,
        content: str,
        chunks: List[DocumentChunk],
        metadata: Dict[str, Any],
    ) -> str:
        async with db_pool.acquire() as conn:
            async with conn.transaction():
                doc_result = await conn.fetchrow(
                    "INSERT INTO documents (title, source, content, metadata) VALUES ($1, $2, $3, $4) RETURNING id::text",
                    title,
                    source,
                    content,
                    json.dumps(metadata),
                )
                document_id = doc_result["id"]

                for chunk in chunks:
                    embedding_data = None
                    if hasattr(chunk, "embedding") and getattr(chunk, "embedding"):
                        embedding_data = "[" + ",".join(map(str, chunk.embedding)) + "]"
                    await conn.execute(
                        """
                        INSERT INTO chunks
                            (document_id, content, embedding, chunk_index, metadata, token_count)
                        VALUES ($1::uuid, $2, $3::vector, $4, $5, $6)
                        """,
                        document_id,
                        chunk.content,
                        embedding_data,
                        chunk.index,
                        json.dumps(chunk.metadata or {}),
                        getattr(chunk, "token_count", None),
                    )
                return document_id

    async def _clean_databases(self) -> None:
        logger.warning("⚠️  Cleaning existing data...")
        async with db_pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("DELETE FROM messages")
                await conn.execute("DELETE FROM sessions")
                await conn.execute("DELETE FROM chunks")
                await conn.execute("DELETE FROM documents")
                await conn.execute("DELETE FROM ingestion_progress")
                await conn.execute("DELETE FROM ingestion_metrics")
                await conn.execute("DELETE FROM ingestion_runs")
                # Also clear KG mirrors in Postgres
                await conn.execute("DELETE FROM kg_edges")
                await conn.execute("DELETE FROM kg_entities")
        logger.info("Cleaned PostgreSQL")

        await self.graph_builder.clear_graph()
        logger.info("Cleaned knowledge graph")


# ============================================================================
# CLI
# ============================================================================

async def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Ingest documents into vector DB + knowledge graph with resumable checkpoints"
        )
    )
    parser.add_argument("--documents", "-d", default="documents", help="Documents folder")
    parser.add_argument("--clean", "-c", action="store_true", help="🚨 FULL CLEAN before ingest (reprocesses ALL)")
    parser.add_argument("--resume", "-r", action="store_true", help="✅ Resume from last checkpoint (default if no --clean)")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--no-entities", action="store_true")
    parser.add_argument("--fast", "-f", action="store_true", help="Skip graph building")
    parser.add_argument("--kg-sync-only", action="store_true", help="Export Neo4j to Postgres KG tables and exit")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    config = IngestionConfig(
        chunk_size=args.chunk_size,
        extract_entities=not args.no_entities,
        skip_graph_building=args.fast,
    )

    pipeline = DocumentIngestionPipeline(
        config=config,
        documents_folder=args.documents,
        clean_before_ingest=args.clean,
        resume_mode=args.resume or not args.clean,
        kg_sync_only=args.kg_sync_only,
    )

    def progress_callback(current: int, total: int) -> None:
        print(f"Progress: {current}/{total}")

    try:
        start = datetime.now()
        results = await pipeline.ingest_documents(progress_callback)
        elapsed = (datetime.now() - start).total_seconds()

        print("\n" + "=" * 70)
        print("INGESTION SUMMARY")
        print("=" * 70)
        print(f"Session ID: {pipeline.ingestion_session_id}")
        print(
            f"Documents: {len(results)} ({sum(1 for r in results if not r.errors)}/{len(results)} successful)"
        )
        print(f"Chunks: {sum(r.chunks_created for r in results)}")
        print(f"Entities: {sum(r.entities_extracted for r in results)}")
        print(f"Relationships: {sum(r.relationships_created for r in results)}")
        print(f"Cost: ${pipeline.metrics.get_total_cost_usd():.6f}")
        print(f"Tokens: {pipeline.metrics.get_total_tokens()}")
        print(f"Time: {elapsed:.2f}s")
        print("=" * 70)

        for r in results:
            status = "✓" if not r.errors else "✗"
            print(f"{status} {r.title}: {r.chunks_created} chunks, {r.processing_time_ms:.0f}ms")

    except KeyboardInterrupt:
        print("\n⏸️  Interrupted - progress saved. Run with --resume to continue.")
    except Exception as e:
        logger.error(f"Failed: {e}")
        raise
    finally:
        await pipeline.close()


if __name__ == "__main__":
    asyncio.run(main())
