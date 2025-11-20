"""
Database utilities for PostgreSQL with pgvector.

Provides connection pooling, session management, document operations,
vector/hybrid search, and ingestion metrics tracking.
"""

import os
import json
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, timezone
from contextlib import asynccontextmanager
from uuid import uuid4
import logging

import asyncpg
from asyncpg.pool import Pool
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


# ============================================================================
# DATABASE POOL
# ============================================================================

class DatabasePool:
    """Gerencia pool de conexões PostgreSQL com reconexão automática."""
    
    def __init__(self, database_url: Optional[str] = None):
        """Inicializa pool de database."""
        self.database_url = database_url or os.getenv("DATABASE_URL")
        if not self.database_url:
            self.database_url = self._construct_url()
        
        self.pool: Optional[Pool] = None
        self._initialized = False
    
    def _construct_url(self) -> str:
        """Constrói URL do database a partir de variáveis de ambiente."""
        host = os.getenv("DB_HOST", "localhost")
        port = os.getenv("DB_PORT", "5432")
        database = os.getenv("DB_NAME", "agentic_rag")
        user = os.getenv("DB_USER", "postgres")
        password = os.getenv("DB_PASSWORD", "")
        
        if not password:
            raise ValueError("Database password not set (DB_PASSWORD or DATABASE_URL required)")
        
        return f"postgresql://{user}:{password}@{host}:{port}/{database}"
    
    async def initialize(self):
        """Cria pool de conexões."""
        if self._initialized and self.pool:
            return
        
        try:
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=5,
                max_size=20,
                max_inactive_connection_lifetime=300,
                command_timeout=60,
                server_settings={'application_name': 'agentic_rag'}
            )
            self._initialized = True
            await self._create_functions()
            logger.info("Database pool initialized with SQL functions")
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            raise
    
    async def _create_functions(self):
        """Cria funções SQL customizadas."""
        async with self.pool.acquire() as conn:
            await conn.execute("DROP FUNCTION IF EXISTS match_chunks(vector, integer);")
            await conn.execute("DROP FUNCTION IF EXISTS hybrid_search(vector, text, integer, double precision);")
            # Função de busca por similaridade vetorial
            await conn.execute("""
                CREATE OR REPLACE FUNCTION match_chunks(
                    query_embedding VECTOR,
                    match_limit INT DEFAULT 10
                ) RETURNS TABLE (
                    chunk_id UUID,
                    document_id UUID,
                    content TEXT,
                    similarity FLOAT,
                    metadata JSONB,
                    document_title TEXT,    -- <--- FIX: Changed VARCHAR to TEXT
                    document_source TEXT    -- <--- FIX: Changed VARCHAR to TEXT
                ) AS $$
                BEGIN
                    RETURN QUERY
                    SELECT 
                        c.id,
                        c.document_id,
                        c.content,
                        1 - (c.embedding <=> query_embedding)::FLOAT AS similarity,
                        c.metadata,
                        d.title,
                        d.source
                    FROM chunks c
                    JOIN documents d ON c.document_id = d.id
                    WHERE c.embedding IS NOT NULL
                    ORDER BY c.embedding <=> query_embedding
                    LIMIT match_limit;
                END;
                $$ LANGUAGE plpgsql;
            """)
            
            # Função de busca híbrida (vetorial + full-text)
            await conn.execute("""
                CREATE OR REPLACE FUNCTION hybrid_search(
                    query_embedding VECTOR,
                    query_text TEXT,
                    match_limit INT DEFAULT 10,
                    text_weight FLOAT DEFAULT 0.3
                ) RETURNS TABLE (
                    chunk_id UUID,
                    document_id UUID,
                    content TEXT,
                    combined_score FLOAT,
                    vector_similarity FLOAT,
                    text_similarity FLOAT,
                    metadata JSONB,
                    document_title TEXT,    -- <--- FIX: Changed VARCHAR to TEXT
                    document_source TEXT    -- <--- FIX: Changed VARCHAR to TEXT
                ) AS $$
                BEGIN
                    RETURN QUERY
                    WITH vector_results AS (
                        SELECT 
                            c.id,
                            c.document_id,
                            c.content,
                            (1 - (c.embedding <=> query_embedding))::FLOAT AS vec_sim,
                            c.metadata,
                            d.title,
                            d.source
                        FROM chunks c
                        JOIN documents d ON c.document_id = d.id
                        WHERE c.embedding IS NOT NULL
                        ORDER BY c.embedding <=> query_embedding
                        LIMIT match_limit
                    ),
                    text_results AS (
                        SELECT 
                            c.id,
                            ts_rank(
                                to_tsvector('english', c.content),
                                plainto_tsquery('english', query_text)
                            )::FLOAT AS text_sim
                        FROM chunks c
                        WHERE to_tsvector('english', c.content) @@ 
                              plainto_tsquery('english', query_text)
                    )
                    SELECT 
                        v.id,
                        v.document_id,
                        v.content,
                        ((1 - text_weight) * v.vec_sim + text_weight * COALESCE(t.text_sim, 0))::FLOAT,
                        v.vec_sim,
                        COALESCE(t.text_sim, 0)::FLOAT,
                        v.metadata,
                        v.title,
                        v.source
                    FROM vector_results v
                    LEFT JOIN text_results t ON v.id = t.id
                    ORDER BY ((1 - text_weight) * v.vec_sim + text_weight * COALESCE(t.text_sim, 0)) DESC;
                END;
                $$ LANGUAGE plpgsql;
            """)
            
            # Função para recuperar chunks de um documento
            await conn.execute("""
                CREATE OR REPLACE FUNCTION get_document_chunks(doc_id UUID)
                RETURNS TABLE (
                    chunk_id UUID,
                    content TEXT,
                    chunk_index INT,
                    metadata JSONB
                ) AS $$
                BEGIN
                    RETURN QUERY
                    SELECT 
                        c.id,
                        c.content,
                        c.chunk_index,
                        c.metadata
                    FROM chunks c
                    WHERE c.document_id = doc_id
                    ORDER BY c.chunk_index ASC;
                END;
                $$ LANGUAGE plpgsql;
            """)
            
            logger.info("SQL functions created/updated")
    
    async def close(self):
        """Fecha pool de conexões."""
        if self.pool:
            await self.pool.close()
            self.pool = None
            self._initialized = False
            logger.info("Database pool closed")
    
    @asynccontextmanager
    async def acquire(self):
        """Adquire conexão do pool."""
        if not self._initialized:
            await self.initialize()
        
        async with self.pool.acquire() as connection:
            yield connection


# Global database pool instance
db_pool = DatabasePool()


# ============================================================================
# INITIALIZATION
# ============================================================================

async def initialize_database():
    """Inicializa pool de database."""
    await db_pool.initialize()


async def close_database():
    """Fecha pool de database."""
    await db_pool.close()


async def test_connection() -> bool:
    """Testa conexão com database."""
    try:
        async with db_pool.acquire() as conn:
            result = await conn.fetchval("SELECT 1")
            return result == 1
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False


# ============================================================================
# SESSION MANAGEMENT
# ============================================================================

async def create_session(
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    timeout_minutes: int = 60
) -> Dict[str, Any]:
    """Cria nova sessão."""
    session_id = session_id or str(uuid4())
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=timeout_minutes)
    
    async with db_pool.acquire() as conn:
        result = await conn.fetchrow(
            """
            INSERT INTO sessions (id, user_id, metadata, expires_at, created_at, updated_at)
            VALUES ($1::uuid, $2, $3::jsonb, $4, NOW(), NOW())
            RETURNING id::text, user_id, metadata, created_at, updated_at, expires_at
            """,
            session_id, user_id, json.dumps(metadata or {}), expires_at
        )
        
        return {
            "id": result["id"],
            "user_id": result["user_id"],
            "metadata": json.loads(result["metadata"]) if result["metadata"] else {},
            "created_at": result["created_at"].isoformat(),
            "updated_at": result["updated_at"].isoformat(),
            "expires_at": result["expires_at"].isoformat() if result["expires_at"] else None
        }


async def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Obtém sessão por ID."""
    async with db_pool.acquire() as conn:
        result = await conn.fetchrow(
            """
            SELECT id::text, user_id, metadata, created_at, updated_at, expires_at
            FROM sessions
            WHERE id = $1::uuid AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
            """,
            session_id
        )
        
        if result:
            return {
                "id": result["id"],
                "user_id": result["user_id"],
                "metadata": json.loads(result["metadata"]) if result["metadata"] else {},
                "created_at": result["created_at"].isoformat(),
                "updated_at": result["updated_at"].isoformat(),
                "expires_at": result["expires_at"].isoformat() if result["expires_at"] else None
            }
        return None


async def update_session(session_id: str, metadata: Dict[str, Any]) -> bool:
    """Atualiza metadados da sessão."""
    async with db_pool.acquire() as conn:
        result = await conn.execute(
            """
            UPDATE sessions
            SET metadata = metadata || $2::jsonb, updated_at = NOW()
            WHERE id = $1::uuid AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
            """,
            session_id, json.dumps(metadata)
        )
        return int(result.split()[-1]) > 0


# ============================================================================
# MESSAGE MANAGEMENT
# ============================================================================

async def save_message(
    session_id: str,
    role: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Salva mensagem em sessão."""
    async with db_pool.acquire() as conn:
        async with conn.transaction():
            result = await conn.fetchrow(
                """
                INSERT INTO messages (session_id, role, content, metadata, created_at)
                VALUES ($1::uuid, $2, $3, $4::jsonb, NOW())
                RETURNING id::text, session_id::text, role, content, metadata, created_at
                """,
                session_id, role, content, json.dumps(metadata or {})
            )
            
            # Atualiza session timestamp
            await conn.execute(
                "UPDATE sessions SET updated_at = NOW() WHERE id = $1::uuid",
                session_id
            )
        
        return {
            "id": result["id"],
            "session_id": result["session_id"],
            "role": result["role"],
            "content": result["content"],
            "metadata": json.loads(result["metadata"]) if result["metadata"] else {},
            "created_at": result["created_at"].isoformat()
        }


async def get_session_messages(
    session_id: str,
    limit: int = 50,
    offset: int = 0
) -> List[Dict[str, Any]]:
    """Obtém mensagens da sessão."""
    async with db_pool.acquire() as conn:
        results = await conn.fetch(
            """
            SELECT id::text, session_id::text, role, content, metadata, created_at
            FROM messages
            WHERE session_id = $1::uuid
            ORDER BY created_at ASC
            LIMIT $2 OFFSET $3
            """,
            session_id, limit, offset
        )
        
        return [
            {
                "id": row["id"],
                "session_id": row["session_id"],
                "role": row["role"],
                "content": row["content"],
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                "created_at": row["created_at"].isoformat()
            }
            for row in results
        ]


# ============================================================================
# DOCUMENT MANAGEMENT
# ============================================================================

async def get_document(document_id: str) -> Optional[Dict[str, Any]]:
    """Obtém documento por ID."""
    async with db_pool.acquire() as conn:
        result = await conn.fetchrow(
            """
            SELECT id::text, title, source, content, metadata, created_at, updated_at
            FROM documents WHERE id = $1::uuid
            """,
            document_id
        )
        
        if result:
            return {
                "id": result["id"],
                "title": result["title"],
                "source": result["source"],
                "content": result["content"],
                "metadata": json.loads(result["metadata"]) if result["metadata"] else {},
                "created_at": result["created_at"].isoformat(),
                "updated_at": result["updated_at"].isoformat()
            }
        return None


async def list_documents(
    limit: int = 100,
    offset: int = 0,
    metadata_filter: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """Lista documentos com filtro opcional."""
    async with db_pool.acquire() as conn:
        if metadata_filter:
            results = await conn.fetch(
                """
                SELECT 
                    d.id::text, d.title, d.source, d.metadata, 
                    d.created_at, d.updated_at, COUNT(c.id) AS chunk_count
                FROM documents d
                LEFT JOIN chunks c ON d.id = c.document_id
                WHERE d.metadata @> $1::jsonb
                GROUP BY d.id, d.title, d.source, d.metadata, d.created_at, d.updated_at
                ORDER BY d.created_at DESC
                LIMIT $2 OFFSET $3
                """,
                json.dumps(metadata_filter), limit, offset
            )
        else:
            results = await conn.fetch(
                """
                SELECT 
                    d.id::text, d.title, d.source, d.metadata,
                    d.created_at, d.updated_at, COUNT(c.id) AS chunk_count
                FROM documents d
                LEFT JOIN chunks c ON d.id = c.document_id
                GROUP BY d.id, d.title, d.source, d.metadata, d.created_at, d.updated_at
                ORDER BY d.created_at DESC
                LIMIT $1 OFFSET $2
                """,
                limit, offset
            )
        
        return [
            {
                "id": row["id"],
                "title": row["title"],
                "source": row["source"],
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                "created_at": row["created_at"].isoformat(),
                "updated_at": row["updated_at"].isoformat(),
                "chunk_count": row["chunk_count"]
            }
            for row in results
        ]


# ============================================================================
# INGESTION METRICS
# ============================================================================

async def save_ingestion_metric(
    document_id: str,
    session_id: Optional[str] = None,
    chunks_count: int = 0,
    entities_count: int = 0,
    relationships_count: int = 0,
    embedding_tokens: int = 0,
    embedding_cost_usd: float = 0.0,
    graph_tokens: int = 0,
    graph_cost_usd: float = 0.0,
    processing_time_ms: float = 0.0,
    status: str = "success",
    error_message: Optional[str] = None
) -> str:
    """Salva métrica de ingestão para análise."""
    async with db_pool.acquire() as conn:
        result = await conn.fetchval(
            """
            INSERT INTO ingestion_metrics 
            (document_id, session_id, chunks_count, entities_count, relationships_count,
             embedding_tokens, embedding_cost_usd, graph_tokens, graph_cost_usd,
             processing_time_ms, status, error_message, created_at)
            VALUES ($1::uuid, $2::uuid, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, NOW())
            RETURNING id::text
            """,
            document_id, session_id, chunks_count, entities_count, relationships_count,
            embedding_tokens, embedding_cost_usd, graph_tokens, graph_cost_usd,
            processing_time_ms, status, error_message
        )
        return result


async def get_ingestion_metrics(
    session_id: str,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """Obtém métricas de ingestão por sessão."""
    async with db_pool.acquire() as conn:
        results = await conn.fetch(
            """
            SELECT 
                id::text, document_id::text, session_id::text,
                chunks_count, entities_count, relationships_count,
                embedding_tokens, embedding_cost_usd, graph_tokens, graph_cost_usd,
                processing_time_ms, status, error_message, created_at
            FROM ingestion_metrics
            WHERE session_id = $1::uuid
            ORDER BY created_at DESC
            LIMIT $2
            """,
            session_id, limit
        )
        
        return [
            {
                "id": row["id"],
                "document_id": row["document_id"],
                "session_id": row["session_id"],
                "chunks_count": row["chunks_count"],
                "entities_count": row["entities_count"],
                "relationships_count": row["relationships_count"],
                "embedding_tokens": row["embedding_tokens"],
                "embedding_cost_usd": float(row["embedding_cost_usd"]),
                "graph_tokens": row["graph_tokens"],
                "graph_cost_usd": float(row["graph_cost_usd"]),
                "total_cost_usd": float(row["embedding_cost_usd"] + row["graph_cost_usd"]),
                "processing_time_ms": float(row["processing_time_ms"]),
                "status": row["status"],
                "error_message": row["error_message"],
                "created_at": row["created_at"].isoformat()
            }
            for row in results
        ]


async def get_ingestion_summary(session_id: str) -> Dict[str, Any]:
    """Obtém resumo de ingestão da sessão."""
    async with db_pool.acquire() as conn:
        result = await conn.fetchrow(
            """
            SELECT 
                COUNT(*) as total_documents,
                SUM(chunks_count) as total_chunks,
                SUM(entities_count) as total_entities,
                SUM(relationships_count) as total_relationships,
                SUM(embedding_tokens) as total_embedding_tokens,
                SUM(embedding_cost_usd) as total_embedding_cost,
                SUM(graph_tokens) as total_graph_tokens,
                SUM(graph_cost_usd) as total_graph_cost,
                AVG(processing_time_ms) as avg_processing_time
            FROM ingestion_metrics
            WHERE session_id = $1::uuid
            """,
            session_id
        )
        
        if result:
            return {
                "total_documents": result["total_documents"] or 0,
                "total_chunks": result["total_chunks"] or 0,
                "total_entities": result["total_entities"] or 0,
                "total_relationships": result["total_relationships"] or 0,
                "embedding_tokens": result["total_embedding_tokens"] or 0,
                "embedding_cost_usd": float(result["total_embedding_cost"] or 0),
                "graph_tokens": result["total_graph_tokens"] or 0,
                "graph_cost_usd": float(result["total_graph_cost"] or 0),
                "total_cost_usd": float((result["total_embedding_cost"] or 0) + (result["total_graph_cost"] or 0)),
                "avg_processing_time_ms": float(result["avg_processing_time"] or 0)
            }
        
        return {
            "total_documents": 0,
            "total_chunks": 0,
            "total_entities": 0,
            "total_relationships": 0,
            "embedding_tokens": 0,
            "embedding_cost_usd": 0,
            "graph_tokens": 0,
            "graph_cost_usd": 0,
            "total_cost_usd": 0,
            "avg_processing_time_ms": 0
        }


# ============================================================================
# VECTOR SEARCH
# ============================================================================

async def vector_search(
    embedding: List[float],
    limit: int = 10
) -> List[Dict[str, Any]]:
    """Busca por similaridade vetorial."""
    async with db_pool.acquire() as conn:
        embedding_str = '[' + ','.join(map(str, embedding)) + ']'
        
        results = await conn.fetch(
            "SELECT * FROM match_chunks($1::vector, $2)",
            embedding_str, limit
        )
        
        return [
            {
                "chunk_id": str(row["chunk_id"]),
                "document_id": str(row["document_id"]),
                "content": row["content"],
                "similarity": float(row["similarity"]),
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                "document_title": row["document_title"],
                "document_source": row["document_source"]
            }
            for row in results
        ]


async def hybrid_search(
    embedding: List[float],
    query_text: str,
    limit: int = 10,
    text_weight: float = 0.3
) -> List[Dict[str, Any]]:
    """Busca híbrida (vetorial + full-text)."""
    async with db_pool.acquire() as conn:
        embedding_str = '[' + ','.join(map(str, embedding)) + ']'
        
        results = await conn.fetch(
            "SELECT * FROM hybrid_search($1::vector, $2, $3, $4)",
            embedding_str, query_text, limit, text_weight
        )
        
        return [
            {
                "chunk_id": str(row["chunk_id"]),
                "document_id": str(row["document_id"]),
                "content": row["content"],
                "combined_score": float(row["combined_score"]),
                "vector_similarity": float(row["vector_similarity"]),
                "text_similarity": float(row["text_similarity"]),
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                "document_title": row["document_title"],
                "document_source": row["document_source"]
            }
            for row in results
        ]


# ============================================================================
# CHUNK MANAGEMENT
# ============================================================================

async def get_document_chunks(document_id: str) -> List[Dict[str, Any]]:
    """Obtém todos os chunks de um documento."""
    async with db_pool.acquire() as conn:
        results = await conn.fetch(
            "SELECT * FROM get_document_chunks($1::uuid)",
            document_id
        )
        
        return [
            {
                "chunk_id": str(row["chunk_id"]),
                "content": row["content"],
                "chunk_index": row["chunk_index"],
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {}
            }
            for row in results
        ]


# ============================================================================
# HEALTH & STATS
# ============================================================================

async def get_database_stats() -> Dict[str, Any]:
    """Obtém estatísticas do database."""
    async with db_pool.acquire() as conn:
        stats = {}
        
        stats["documents"] = await conn.fetchval("SELECT COUNT(*) FROM documents")
        stats["chunks"] = await conn.fetchval("SELECT COUNT(*) FROM chunks")
        stats["active_sessions"] = await conn.fetchval(
            "SELECT COUNT(*) FROM sessions WHERE expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP"
        )
        stats["messages"] = await conn.fetchval("SELECT COUNT(*) FROM messages")
        
        # Custos agregados
        ingestion_stats = await conn.fetchrow(
            """
            SELECT 
                SUM(embedding_cost_usd) as total_embedding,
                SUM(graph_cost_usd) as total_graph
            FROM ingestion_metrics
            """
        )
        
        stats["total_embedding_cost_usd"] = float(ingestion_stats["total_embedding"] or 0)
        stats["total_graph_cost_usd"] = float(ingestion_stats["total_graph"] or 0)
        stats["total_cost_usd"] = stats["total_embedding_cost_usd"] + stats["total_graph_cost_usd"]
        
        db_name = os.getenv("DB_NAME", "agentic_rag")
        stats["database_size"] = await conn.fetchval(
            "SELECT pg_size_pretty(pg_database_size($1))", db_name
        )
        
        return stats


async def health_check() -> Dict[str, Any]:
    """Verifica saúde do database."""
    health = {"database": "unhealthy", "pool": "unhealthy", "extensions": {}}
    
    try:
        async with db_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
            health["database"] = "healthy"
            
            if db_pool.pool:
                health["pool"] = "healthy"
                health["pool_size"] = db_pool.pool.get_size()
                health["pool_free"] = db_pool.pool.get_idle_size()
            
            extensions = await conn.fetch(
                """
                SELECT extname, extversion FROM pg_extension
                WHERE extname IN ('vector', 'uuid-ossp', 'pg_trgm')
                """
            )
            
            for ext in extensions:
                health["extensions"][ext["extname"]] = ext["extversion"]
            
            health["stats"] = await get_database_stats()
    
    except Exception as e:
        health["error"] = str(e)
        logger.error(f"Health check failed: {e}")
    
    return health