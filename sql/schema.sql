-- =============================================================================
-- Agentic RAG with Multi-Agent System - Complete Database Schema v3.2
-- =============================================================================
-- Features:
-- - Core document and chunk storage with pgvector
-- - Session and message management for conversations
-- - Knowledge graph (entities, edges, entity mentions)
-- - **INGESTION METRICS** - Rastreamento completo por documento
-- - Comprehensive evaluation and metrics infrastructure FOR SCIENTIFIC PAPERS
-- - Single-Agent vs Multi-Agent comparison metrics
-- - Answer validation tracking (hallucination detection)
-- - Optimized search functions (vector, hybrid, entity matching)
-- - Metric calculation functions (precision, recall, F1, MRR)
-- - Academic-ready aggregate views
-- - Automatic timestamp updates
-- =============================================================================

-- Required Extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- =============================================================================
-- CLEANUP (Development Only - Comment out for production)
-- =============================================================================
DROP TABLE IF EXISTS ingestion_metrics CASCADE;
DROP TABLE IF EXISTS ingestion_runs CASCADE;
DROP TABLE IF EXISTS ingestion_progress CASCADE;
DROP TABLE IF EXISTS tool_metrics CASCADE;
DROP TABLE IF EXISTS retrieval_metrics CASCADE;
DROP TABLE IF EXISTS query_metrics CASCADE;
DROP TABLE IF EXISTS benchmark_relevance CASCADE;
DROP TABLE IF EXISTS benchmark_queries CASCADE;
DROP TABLE IF EXISTS evaluation_runs CASCADE;
DROP TABLE IF EXISTS system_metrics CASCADE;
DROP TABLE IF EXISTS messages CASCADE;
DROP TABLE IF EXISTS sessions CASCADE;
DROP TABLE IF EXISTS chunk_entities CASCADE;
DROP TABLE IF EXISTS kg_edges CASCADE;
DROP TABLE IF EXISTS kg_entities CASCADE;
DROP TABLE IF EXISTS chunks CASCADE;
DROP TABLE IF EXISTS documents CASCADE;

-- =============================================================================
-- CORE TABLES - Documents and Chunks
-- =============================================================================

CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title TEXT NOT NULL,
    source TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_documents_metadata ON documents USING GIN (metadata);
CREATE INDEX idx_documents_created_at ON documents (created_at DESC);
CREATE INDEX idx_documents_source ON documents (source);

COMMENT ON TABLE documents IS 'Source documents ingested into the system';
COMMENT ON COLUMN documents.metadata IS 'Extensible metadata (author, date, tags, etc.)';

CREATE TABLE chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    embedding vector(1536),
    chunk_index INTEGER NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    token_count INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Vector search index (cosine distance)
CREATE INDEX idx_chunks_embedding ON chunks 
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 1000);

-- Standard indexes
CREATE INDEX idx_chunks_document_id ON chunks (document_id);
CREATE INDEX idx_chunks_chunk_index ON chunks (document_id, chunk_index);

-- Full-text search indexes
CREATE INDEX idx_chunks_content_trgm ON chunks USING GIN (content gin_trgm_ops);
CREATE INDEX idx_chunks_content_fts ON chunks USING GIN (to_tsvector('english'::regconfig, content));

COMMENT ON TABLE chunks IS 'Document chunks with embeddings for vector search';
COMMENT ON COLUMN chunks.embedding IS '1536-dimensional embedding vector (e.g., text-embedding-3-small)';

-- =============================================================================
-- SESSION MANAGEMENT - Conversational Context
-- =============================================================================

CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ
);

CREATE INDEX idx_sessions_user_id ON sessions (user_id);
CREATE INDEX idx_sessions_expires_at ON sessions (expires_at);
CREATE INDEX idx_sessions_created_at ON sessions (created_at DESC);

COMMENT ON TABLE sessions IS 'User conversation sessions';

CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_messages_session_id ON messages (session_id, created_at);
CREATE INDEX idx_messages_created_at ON messages (created_at DESC);

COMMENT ON TABLE messages IS 'Conversation messages within sessions';

-- =============================================================================
-- KNOWLEDGE GRAPH - Entities, Edges, and Mentions
-- =============================================================================

CREATE TABLE kg_entities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    type TEXT NOT NULL,
    name TEXT NOT NULL,
    canonical_name TEXT,
    description TEXT,
    properties JSONB DEFAULT '{}'::jsonb,
    embedding vector(1536),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_kg_entities_type ON kg_entities (type);
CREATE INDEX idx_kg_entities_name_trgm ON kg_entities USING GIN (name gin_trgm_ops);
CREATE INDEX idx_kg_entities_properties ON kg_entities USING GIN (properties);
CREATE INDEX idx_kg_entities_desc_fts ON kg_entities USING GIN (to_tsvector('english'::regconfig, COALESCE(description, '')));
CREATE INDEX idx_kg_entities_embedding ON kg_entities USING ivfflat (embedding vector_cosine_ops) WITH (lists = 500);

COMMENT ON TABLE kg_entities IS 'Knowledge graph entities with optional embeddings';

CREATE TABLE kg_edges (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    src_entity_id UUID NOT NULL REFERENCES kg_entities(id) ON DELETE CASCADE,
    dst_entity_id UUID NOT NULL REFERENCES kg_entities(id) ON DELETE CASCADE,
    relation TEXT NOT NULL,
    properties JSONB DEFAULT '{}'::jsonb,
    weight DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_kg_edges_endpoints ON kg_edges (src_entity_id, dst_entity_id);
CREATE INDEX idx_kg_edges_relation ON kg_edges (relation);
CREATE INDEX idx_kg_edges_properties ON kg_edges USING GIN (properties);

COMMENT ON TABLE kg_edges IS 'Directed edges (relations) between entities';

CREATE TABLE chunk_entities (
    chunk_id UUID NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    entity_id UUID NOT NULL REFERENCES kg_entities(id) ON DELETE CASCADE,
    mention_text TEXT,
    span_start INTEGER CHECK (span_start IS NULL OR span_start >= 0),
    span_end INTEGER CHECK (span_end IS NULL OR span_end >= 0),
    confidence DOUBLE PRECISION CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1)),
    PRIMARY KEY (chunk_id, entity_id)
);

CREATE INDEX idx_chunk_entities_entity ON chunk_entities (entity_id);

COMMENT ON TABLE chunk_entities IS 'Links entity mentions back to chunks for provenance';

-- =============================================================================
-- INGESTION METRICS - RASTREAMENTO COMPLETO POR DOCUMENTO
-- =============================================================================

-- Ative extensões
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Crie as novas tabelas (copie do schema.sql completo que enviei)
CREATE TABLE ingestion_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ingestion_session_id UUID NOT NULL UNIQUE,
    documents_processed INTEGER DEFAULT 0,
    documents_success INTEGER DEFAULT 0,
    documents_failed INTEGER DEFAULT 0,
    total_chunks_created INTEGER DEFAULT 0,
    total_entities_extracted INTEGER DEFAULT 0,
    total_relationships_created INTEGER DEFAULT 0,
    total_embedding_tokens INTEGER DEFAULT 0,
    total_embedding_cost_usd DECIMAL(12, 6) DEFAULT 0,
    total_graph_tokens INTEGER DEFAULT 0,
    total_graph_cost_usd DECIMAL(12, 6) DEFAULT 0,
    total_cost_usd DECIMAL(12, 6) DEFAULT 0, -- computed manually
    total_time_seconds FLOAT DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE OR REPLACE FUNCTION update_total_cost()
RETURNS TRIGGER AS $$
BEGIN
  NEW.total_cost_usd := COALESCE(NEW.total_embedding_cost_usd, 0)
                      + COALESCE(NEW.total_graph_cost_usd, 0);
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER set_total_cost
BEFORE INSERT OR UPDATE ON ingestion_runs
FOR EACH ROW
EXECUTE FUNCTION update_total_cost();

CREATE INDEX idx_ingestion_runs_session ON ingestion_runs(ingestion_session_id);
CREATE INDEX idx_ingestion_runs_created_at ON ingestion_runs(created_at DESC);

CREATE TABLE ingestion_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ingestion_session_id UUID NOT NULL REFERENCES ingestion_runs(ingestion_session_id) ON DELETE CASCADE,
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    document_title VARCHAR(500) NOT NULL,
    document_source VARCHAR(500) NOT NULL,
    chunks_count INTEGER DEFAULT 0,
    entities_count INTEGER DEFAULT 0,
    relationships_count INTEGER DEFAULT 0,
    embedding_tokens INTEGER DEFAULT 0,
    embedding_cost_usd DECIMAL(10, 6) DEFAULT 0,
    graph_tokens INTEGER DEFAULT 0,
    graph_cost_usd DECIMAL(10, 6) DEFAULT 0,
    total_cost_usd DECIMAL(10, 6) DEFAULT 0, -- will be computed manually
    processing_time_ms FLOAT DEFAULT 0,
    status VARCHAR(20) CHECK (status IN ('pending', 'chunks_done', 'embedded_done', 'graph_done', 'success', 'failed')),
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE OR REPLACE FUNCTION update_total_cost_metrics()
RETURNS TRIGGER AS $$
BEGIN
  NEW.total_cost_usd := COALESCE(NEW.embedding_cost_usd, 0)
                      + COALESCE(NEW.graph_cost_usd, 0);
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER set_total_cost_metrics
BEFORE INSERT OR UPDATE ON ingestion_metrics
FOR EACH ROW
EXECUTE FUNCTION update_total_cost_metrics();

CREATE INDEX idx_ingestion_metrics_session ON ingestion_metrics(ingestion_session_id);
CREATE INDEX idx_ingestion_metrics_document ON ingestion_metrics(document_id);

CREATE TABLE ingestion_progress (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_source VARCHAR(500) NOT NULL UNIQUE,
    status VARCHAR(50) NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'chunks_done', 'embedded_done', 'graph_done', 'success', 'failed')),
    document_id UUID REFERENCES documents(id) ON DELETE SET NULL,
    checkpoint JSONB DEFAULT '{}'::jsonb,
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_ingestion_progress_source ON ingestion_progress(document_source);

-- =============================================================================
-- EVALUATION INFRASTRUCTURE - FOR SCIENTIFIC EXPERIMENTS
-- =============================================================================

CREATE TABLE evaluation_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    description TEXT,
    configuration JSONB NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed', 'cancelled')),
    started_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    total_queries INTEGER DEFAULT 0,
    successful_queries INTEGER DEFAULT 0,
    failed_queries INTEGER DEFAULT 0,
    avg_latency_ms DOUBLE PRECISION,
    total_tokens INTEGER DEFAULT 0,
    total_cost_usd DOUBLE PRECISION DEFAULT 0.0,
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX idx_evaluation_runs_status ON evaluation_runs (status);
CREATE INDEX idx_evaluation_runs_started_at ON evaluation_runs (started_at DESC);
CREATE INDEX idx_evaluation_runs_name ON evaluation_runs (name);

COMMENT ON TABLE evaluation_runs IS 'Experiment runs for evaluation and comparison (for academic papers)';

-- =============================================================================
-- QUERY METRICS - ENHANCED FOR SINGLE-AGENT VS MULTI-AGENT COMPARISON
-- =============================================================================

CREATE TABLE query_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    evaluation_run_id UUID REFERENCES evaluation_runs(id) ON DELETE CASCADE,
    session_id UUID REFERENCES sessions(id) ON DELETE SET NULL,
    query_text TEXT NOT NULL,
    response_text TEXT,
    
    -- System identification (CRITICAL for comparison)
    system_type TEXT CHECK (system_type IN ('single-agent', 'multi-agent', 'hybrid')),
    
    -- Performance metrics
    total_latency_ms DOUBLE PRECISION NOT NULL CHECK (total_latency_ms >= 0),
    retrieval_latency_ms DOUBLE PRECISION CHECK (retrieval_latency_ms >= 0),
    generation_latency_ms DOUBLE PRECISION CHECK (generation_latency_ms >= 0),
    tool_execution_latency_ms DOUBLE PRECISION CHECK (tool_execution_latency_ms >= 0),
    
    -- Token and cost tracking
    prompt_tokens INTEGER DEFAULT 0 CHECK (prompt_tokens >= 0),
    completion_tokens INTEGER DEFAULT 0 CHECK (completion_tokens >= 0),
    total_tokens INTEGER DEFAULT 0 CHECK (total_tokens >= 0),
    estimated_cost_usd DOUBLE PRECISION DEFAULT 0.0 CHECK (estimated_cost_usd >= 0),
    
    -- Tool usage
    tools_used JSONB DEFAULT '[]'::jsonb,
    tool_count INTEGER DEFAULT 0 CHECK (tool_count >= 0),
    
    -- Multi-Agent System specific metrics
    mas_tasks_created INTEGER CHECK (mas_tasks_created >= 0),
    mas_tasks_completed INTEGER CHECK (mas_tasks_completed >= 0),
    mas_tasks_failed INTEGER CHECK (mas_tasks_failed >= 0),
    confidence_score DOUBLE PRECISION CHECK (confidence_score BETWEEN 0 AND 1),
    
    -- Validation metrics (MAS only)
    validation_is_valid BOOLEAN,
    validation_hallucination_score DOUBLE PRECISION CHECK (validation_hallucination_score BETWEEN 0 AND 1),
    validation_verified_claims INTEGER CHECK (validation_verified_claims >= 0),
    validation_total_claims INTEGER CHECK (validation_total_claims >= 0),
    
    -- Status
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT,
    
    -- Metadata
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_query_metrics_evaluation_run ON query_metrics (evaluation_run_id);
CREATE INDEX idx_query_metrics_session ON query_metrics (session_id);
CREATE INDEX idx_query_metrics_system_type ON query_metrics (system_type);
CREATE INDEX idx_query_metrics_created_at ON query_metrics (created_at DESC);
CREATE INDEX idx_query_metrics_success ON query_metrics (success);
CREATE INDEX idx_query_metrics_tools_used ON query_metrics USING GIN (tools_used);
CREATE INDEX idx_query_metrics_validation ON query_metrics (validation_is_valid);

COMMENT ON TABLE query_metrics IS 'Per-query performance and cost metrics with system type tracking';
COMMENT ON COLUMN query_metrics.system_type IS 'System used: single-agent (baseline) or multi-agent (MAS)';

-- =============================================================================
-- RETRIEVAL METRICS
-- =============================================================================

CREATE TABLE retrieval_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query_metric_id UUID NOT NULL REFERENCES query_metrics(id) ON DELETE CASCADE,
    evaluation_run_id UUID REFERENCES evaluation_runs(id) ON DELETE CASCADE,
    
    -- Retrieval configuration
    search_type TEXT NOT NULL CHECK (search_type IN ('vector', 'graph', 'hybrid')),
    k INTEGER NOT NULL CHECK (k > 0),
    
    -- Retrieved chunks
    retrieved_chunk_ids UUID[] DEFAULT '{}'::uuid[],
    retrieved_count INTEGER NOT NULL CHECK (retrieved_count >= 0),
    
    -- Relevance metrics (requires ground truth)
    relevant_retrieved INTEGER CHECK (relevant_retrieved >= 0),
    total_relevant INTEGER CHECK (total_relevant >= 0),
    precision_at_k DOUBLE PRECISION,
    recall_at_k DOUBLE PRECISION,
    f1_at_k DOUBLE PRECISION,
    average_precision DOUBLE PRECISION,
    
    -- Ranking metrics
    mrr DOUBLE PRECISION,
    ndcg_at_k DOUBLE PRECISION,
    
    -- Retrieval quality
    avg_similarity_score DOUBLE PRECISION,
    min_similarity_score DOUBLE PRECISION,
    max_similarity_score DOUBLE PRECISION,
    
    -- Context metrics
    total_context_length INTEGER,
    context_tokens INTEGER,
    
    -- Metadata
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_retrieval_metrics_query_metric ON retrieval_metrics (query_metric_id);
CREATE INDEX idx_retrieval_metrics_evaluation_run ON retrieval_metrics (evaluation_run_id);
CREATE INDEX idx_retrieval_metrics_search_type ON retrieval_metrics (search_type);
CREATE INDEX idx_retrieval_metrics_precision ON retrieval_metrics (precision_at_k);

COMMENT ON TABLE retrieval_metrics IS 'Retrieval quality metrics per query';

-- =============================================================================
-- TOOL METRICS
-- =============================================================================

CREATE TABLE tool_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query_metric_id UUID NOT NULL REFERENCES query_metrics(id) ON DELETE CASCADE,
    evaluation_run_id UUID REFERENCES evaluation_runs(id) ON DELETE CASCADE,
    
    tool_name TEXT NOT NULL,
    tool_args JSONB DEFAULT '{}'::jsonb,
    execution_order INTEGER NOT NULL CHECK (execution_order >= 0),
    
    -- Performance
    latency_ms DOUBLE PRECISION NOT NULL CHECK (latency_ms >= 0),
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT,
    
    -- Results
    results_count INTEGER DEFAULT 0 CHECK (results_count >= 0),
    results_sample JSONB,
    
    -- Metadata
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_tool_metrics_query_metric ON tool_metrics (query_metric_id);
CREATE INDEX idx_tool_metrics_evaluation_run ON tool_metrics (evaluation_run_id);
CREATE INDEX idx_tool_metrics_tool_name ON tool_metrics (tool_name);
CREATE INDEX idx_tool_metrics_success ON tool_metrics (success);
CREATE INDEX idx_tool_metrics_created_at ON tool_metrics (created_at DESC);

COMMENT ON TABLE tool_metrics IS 'Individual tool execution metrics';

-- =============================================================================
-- SYSTEM METRICS
-- =============================================================================

CREATE TABLE system_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_type TEXT NOT NULL,
    
    -- Time window
    window_start TIMESTAMPTZ NOT NULL,
    window_end TIMESTAMPTZ NOT NULL,
    window_size_minutes INTEGER NOT NULL CHECK (window_size_minutes > 0),
    CHECK (window_end >= window_start),
    
    -- Aggregate metrics
    total_queries INTEGER DEFAULT 0 CHECK (total_queries >= 0),
    successful_queries INTEGER DEFAULT 0 CHECK (successful_queries >= 0),
    failed_queries INTEGER DEFAULT 0 CHECK (failed_queries >= 0),
    
    avg_latency_ms DOUBLE PRECISION,
    p50_latency_ms DOUBLE PRECISION,
    p95_latency_ms DOUBLE PRECISION,
    p99_latency_ms DOUBLE PRECISION,
    
    total_tokens INTEGER DEFAULT 0 CHECK (total_tokens >= 0),
    total_cost_usd DOUBLE PRECISION DEFAULT 0.0 CHECK (total_cost_usd >= 0),
    
    -- Quality metrics
    avg_precision DOUBLE PRECISION,
    avg_recall DOUBLE PRECISION,
    avg_f1 DOUBLE PRECISION,
    avg_mrr DOUBLE PRECISION,
    
    -- Tool usage
    tool_usage_counts JSONB DEFAULT '{}'::jsonb,
    
    -- Metadata
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_system_metrics_type ON system_metrics (metric_type);
CREATE INDEX idx_system_metrics_window_start ON system_metrics (window_start DESC);
CREATE INDEX idx_system_metrics_created_at ON system_metrics (created_at DESC);

COMMENT ON TABLE system_metrics IS 'Aggregated system performance over time windows';

-- =============================================================================
-- BENCHMARK DATASETS
-- =============================================================================

CREATE TABLE benchmark_queries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    dataset_name TEXT NOT NULL,
    query_text TEXT NOT NULL,
    ground_truth_answer TEXT,
    query_type TEXT,
    difficulty TEXT CHECK (difficulty IN ('easy', 'medium', 'hard')),
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_benchmark_queries_dataset ON benchmark_queries (dataset_name);
CREATE INDEX idx_benchmark_queries_type ON benchmark_queries (query_type);
CREATE INDEX idx_benchmark_queries_difficulty ON benchmark_queries (difficulty);

COMMENT ON TABLE benchmark_queries IS 'Ground truth questions for evaluation';

CREATE TABLE benchmark_relevance (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    benchmark_query_id UUID NOT NULL REFERENCES benchmark_queries(id) ON DELETE CASCADE,
    chunk_id UUID NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    relevance_score INTEGER NOT NULL CHECK (relevance_score BETWEEN 0 AND 3),
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(benchmark_query_id, chunk_id)
);

CREATE INDEX idx_benchmark_relevance_query ON benchmark_relevance (benchmark_query_id);
CREATE INDEX idx_benchmark_relevance_chunk ON benchmark_relevance (chunk_id);
CREATE INDEX idx_benchmark_relevance_score ON benchmark_relevance (relevance_score);

COMMENT ON TABLE benchmark_relevance IS 'Ground truth relevance judgments for retrieval evaluation';

-- =============================================================================
-- SEARCH FUNCTIONS
-- =============================================================================

-- Vector search using cosine distance
CREATE OR REPLACE FUNCTION match_chunks(
    query_embedding vector(1536),
    match_count INT DEFAULT 10
)
RETURNS TABLE (
    chunk_id UUID,
    document_id UUID,
    content TEXT,
    similarity DOUBLE PRECISION,
    metadata JSONB,
    document_title TEXT,
    document_source TEXT
)
LANGUAGE plpgsql STABLE
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        c.id AS chunk_id,
        c.document_id,
        c.content,
        1 - (c.embedding <-> query_embedding) AS similarity,
        c.metadata,
        d.title AS document_title,
        d.source AS document_source
    FROM chunks c
    JOIN documents d ON c.document_id = d.id
    WHERE c.embedding IS NOT NULL
    ORDER BY c.embedding <-> query_embedding
    LIMIT match_count;
END;
$$;

-- Entity matching by embedding
CREATE OR REPLACE FUNCTION match_entities(
    query_embedding vector(1536),
    match_count INT DEFAULT 10
)
RETURNS TABLE (
    entity_id UUID,
    name TEXT,
    type TEXT,
    similarity DOUBLE PRECISION,
    properties JSONB
)
LANGUAGE sql STABLE
AS $$
    SELECT 
        id,
        name,
        type,
        1 - (embedding <-> query_embedding) AS similarity,
        properties
    FROM kg_entities
    WHERE embedding IS NOT NULL
    ORDER BY embedding <-> query_embedding
    LIMIT match_count
$$;

-- Hybrid search (vector + full-text)
CREATE OR REPLACE FUNCTION hybrid_search(
    query_embedding vector(1536),
    query_text TEXT,
    match_count INT DEFAULT 10,
    text_weight DOUBLE PRECISION DEFAULT 0.3
)
RETURNS TABLE (
    chunk_id UUID,
    document_id UUID,
    content TEXT,
    combined_score DOUBLE PRECISION,
    vector_similarity DOUBLE PRECISION,
    text_similarity DOUBLE PRECISION,
    metadata JSONB,
    document_title TEXT,
    document_source TEXT
)
LANGUAGE plpgsql STABLE
AS $$
DECLARE
    w DOUBLE PRECISION := LEAST(GREATEST(text_weight, 0), 1);
    vec_candidates INT := GREATEST(match_count * 5, 50);
    txt_candidates INT := GREATEST(match_count * 5, 50);
BEGIN
    RETURN QUERY
    WITH vector_results AS (
        SELECT 
            c.id AS chunk_id,
            c.document_id,
            c.content,
            1 - (c.embedding <-> query_embedding) AS vector_sim,
            c.metadata,
            d.title AS doc_title,
            d.source AS doc_source
        FROM chunks c
        JOIN documents d ON c.document_id = d.id
        WHERE c.embedding IS NOT NULL
        ORDER BY c.embedding <-> query_embedding
        LIMIT vec_candidates
    ),
    text_results AS (
        SELECT 
            c.id AS chunk_id,
            c.document_id,
            c.content,
            ts_rank_cd(to_tsvector('english'::regconfig, c.content), 
                      plainto_tsquery('english'::regconfig, query_text)) AS text_sim,
            c.metadata,
            d.title AS doc_title,
            d.source AS doc_source
        FROM chunks c
        JOIN documents d ON c.document_id = d.id
        WHERE to_tsvector('english'::regconfig, c.content) @@ 
              plainto_tsquery('english'::regconfig, query_text)
        ORDER BY text_sim DESC
        LIMIT txt_candidates
    )
    SELECT 
        COALESCE(v.chunk_id, t.chunk_id) AS chunk_id,
        COALESCE(v.document_id, t.document_id) AS document_id,
        COALESCE(v.content, t.content) AS content,
        (COALESCE(v.vector_sim, 0) * (1 - w) + COALESCE(t.text_sim, 0) * w) AS combined_score,
        COALESCE(v.vector_sim, 0) AS vector_similarity,
        COALESCE(t.text_sim, 0) AS text_similarity,
        COALESCE(v.metadata, t.metadata) AS metadata,
        COALESCE(v.doc_title, t.doc_title) AS document_title,
        COALESCE(v.doc_source, t.doc_source) AS document_source
    FROM vector_results v
    FULL OUTER JOIN text_results t USING (chunk_id)
    ORDER BY combined_score DESC
    LIMIT match_count;
END;
$$;

-- Get document chunks
CREATE OR REPLACE FUNCTION get_document_chunks(doc_id UUID)
RETURNS TABLE (
    chunk_id UUID,
    content TEXT,
    chunk_index INTEGER,
    metadata JSONB
)
LANGUAGE plpgsql STABLE
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        id AS chunk_id,
        chunks.content,
        chunks.chunk_index,
        chunks.metadata
    FROM chunks
    WHERE document_id = doc_id
    ORDER BY chunk_index;
END;
$$;

-- =============================================================================
-- METRIC CALCULATION FUNCTIONS
-- =============================================================================

-- Calculate precision, recall, F1
CREATE OR REPLACE FUNCTION calculate_retrieval_metrics(
    p_retrieved_chunk_ids UUID[],
    p_relevant_chunk_ids UUID[]
)
RETURNS TABLE (
    precision_score DOUBLE PRECISION,
    recall_score DOUBLE PRECISION,
    f1_score DOUBLE PRECISION,
    relevant_retrieved INTEGER,
    total_retrieved INTEGER,
    total_relevant INTEGER
)
LANGUAGE plpgsql IMMUTABLE
AS $$
DECLARE
    v_relevant_retrieved INTEGER := 0;
    v_total_retrieved INTEGER := COALESCE(array_length(p_retrieved_chunk_ids, 1), 0);
    v_total_relevant INTEGER := COALESCE(array_length(p_relevant_chunk_ids, 1), 0);
    v_precision DOUBLE PRECISION := 0.0;
    v_recall DOUBLE PRECISION := 0.0;
    v_f1 DOUBLE PRECISION := 0.0;
BEGIN
    SELECT COUNT(*) INTO v_relevant_retrieved
    FROM unnest(COALESCE(p_retrieved_chunk_ids, ARRAY[]::uuid[])) AS retrieved
    WHERE retrieved = ANY(COALESCE(p_relevant_chunk_ids, ARRAY[]::uuid[]));
    
    IF v_total_retrieved > 0 THEN
        v_precision := v_relevant_retrieved::DOUBLE PRECISION / v_total_retrieved::DOUBLE PRECISION;
    END IF;
    
    IF v_total_relevant > 0 THEN
        v_recall := v_relevant_retrieved::DOUBLE PRECISION / v_total_relevant::DOUBLE PRECISION;
    END IF;
    
    IF (v_precision + v_recall) > 0 THEN
        v_f1 := 2 * (v_precision * v_recall) / (v_precision + v_recall);
    END IF;
    
    RETURN QUERY SELECT v_precision, v_recall, v_f1, v_relevant_retrieved, v_total_retrieved, v_total_relevant;
END;
$$;

-- Calculate MRR (Mean Reciprocal Rank)
CREATE OR REPLACE FUNCTION calculate_mrr(
    p_retrieved_chunk_ids UUID[],
    p_relevant_chunk_ids UUID[]
)
RETURNS DOUBLE PRECISION
LANGUAGE plpgsql IMMUTABLE
AS $$
DECLARE
    v_rank INTEGER;
    v_len INTEGER := COALESCE(array_length(p_retrieved_chunk_ids, 1), 0);
BEGIN
    IF v_len = 0 OR COALESCE(array_length(p_relevant_chunk_ids, 1), 0) = 0 THEN
        RETURN 0.0;
    END IF;
    
    FOR v_rank IN 1..v_len LOOP
        IF p_retrieved_chunk_ids[v_rank] = ANY (COALESCE(p_relevant_chunk_ids, ARRAY[]::uuid[])) THEN
            RETURN 1.0 / v_rank::DOUBLE PRECISION;
        END IF;
    END LOOP;
    
    RETURN 0.0;
END;
$$;

-- Export evaluation run as JSON
CREATE OR REPLACE FUNCTION export_evaluation_run_json(p_run_id UUID)
RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    v_result JSONB;
BEGIN
    SELECT jsonb_build_object(
        'run_metadata', (
            SELECT row_to_json(er)
            FROM evaluation_runs er
            WHERE er.id = p_run_id
        ),
        'summary_statistics', (
            SELECT row_to_json(s)
            FROM evaluation_run_detailed_metrics s
            WHERE s.run_id = p_run_id
        ),
        'query_metrics', (
            SELECT jsonb_agg(row_to_json(qm))
            FROM query_metrics qm
            WHERE qm.evaluation_run_id = p_run_id
        ),
        'retrieval_metrics', (
            SELECT jsonb_agg(row_to_json(rm))
            FROM retrieval_metrics rm
            WHERE rm.evaluation_run_id = p_run_id
        )
    ) INTO v_result;
    
    RETURN v_result;
END;
$$;

COMMENT ON FUNCTION export_evaluation_run_json IS 'Export complete evaluation run as JSON for Python/R analysis';

-- =============================================================================
-- TRIGGERS
-- =============================================================================

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_documents_updated_at BEFORE UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_sessions_updated_at BEFORE UPDATE ON sessions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_kg_entities_updated_at BEFORE UPDATE ON kg_entities
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_ingestion_metrics_updated_at BEFORE UPDATE ON ingestion_metrics
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- VIEWS FOR ACADEMIC ANALYSIS
-- =============================================================================

-- Document summaries com ingestão
CREATE OR REPLACE VIEW document_summaries AS
SELECT 
    d.id,
    d.title,
    d.source,
    d.created_at,
    d.updated_at,
    d.metadata,
    COUNT(c.id) AS chunk_count,
    AVG(c.token_count) AS avg_tokens_per_chunk,
    COALESCE(SUM(c.token_count), 0) AS total_tokens,
    -- Ingestão
    im.status AS ingestion_status,
    im.embedding_tokens,
    im.embedding_cost_usd,
    im.graph_tokens,
    im.graph_cost_usd,
    im.total_cost_usd,
    im.processing_time_ms
FROM documents d
LEFT JOIN chunks c ON d.id = c.document_id
LEFT JOIN ingestion_metrics im ON d.id = im.document_id
GROUP BY d.id, d.title, d.source, d.created_at, d.updated_at, d.metadata, 
         im.status, im.embedding_tokens, im.embedding_cost_usd, im.graph_tokens, 
         im.graph_cost_usd, im.total_cost_usd, im.processing_time_ms;

COMMENT ON VIEW document_summaries IS 'Sumário de documentos com métricas de ingestão';

-- Ingestion summary (agregado de todos os documentos)
CREATE OR REPLACE VIEW ingestion_summary AS
SELECT 
    COUNT(*) AS total_documents_ingested,
    COUNT(*) FILTER (WHERE status = 'success') AS successful,
    COUNT(*) FILTER (WHERE status = 'partial') AS partial,
    COUNT(*) FILTER (WHERE status = 'failed') AS failed,
    SUM(chunks_count) AS total_chunks,
    SUM(entities_count) AS total_entities,
    SUM(relationships_count) AS total_relationships,
    SUM(embedding_tokens) AS total_embedding_tokens,
    SUM(embedding_cost_usd) AS total_embedding_cost,
    SUM(graph_tokens) AS total_graph_tokens,
    SUM(graph_cost_usd) AS total_graph_cost,
    SUM(total_cost_usd) AS total_cost,
    AVG(processing_time_ms) AS avg_processing_time_ms,
    MAX(created_at) AS last_ingestion
FROM ingestion_metrics;

COMMENT ON VIEW ingestion_summary IS 'Resumo agregado da ingestão de todos os documentos';

-- Ingestion by document
CREATE OR REPLACE VIEW ingestion_by_document AS
SELECT 
    document_title,
    document_source,
    status,
    chunks_count,
    entities_count,
    relationships_count,
    embedding_tokens,
    embedding_cost_usd,
    graph_tokens,
    graph_cost_usd,
    total_cost_usd,
    processing_time_ms,
    created_at,
    CASE 
        WHEN processing_time_ms > 0 THEN (chunks_count / (processing_time_ms / 1000.0))::INT
        ELSE 0
    END AS chunks_per_second
FROM ingestion_metrics
ORDER BY created_at DESC;

COMMENT ON VIEW ingestion_by_document IS 'Análise detalhe por documento: custo, tokens, throughput';

-- System comparison (Single-Agent vs Multi-Agent)
CREATE OR REPLACE VIEW system_comparison_metrics AS
SELECT 
    system_type,
    COUNT(*) AS total_queries,
    COUNT(*) FILTER (WHERE success = TRUE) AS successful_queries,
    (COUNT(*) FILTER (WHERE success = TRUE)::DOUBLE PRECISION / COUNT(*)::DOUBLE PRECISION) * 100 AS success_rate_pct,
    
    -- Latency statistics
    AVG(total_latency_ms) AS avg_latency_ms,
    STDDEV(total_latency_ms) AS stddev_latency_ms,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY total_latency_ms) AS median_latency_ms,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY total_latency_ms) AS p95_latency_ms,
    
    -- Token and cost
    AVG(total_tokens) AS avg_tokens,
    SUM(total_tokens) AS total_tokens,
    AVG(estimated_cost_usd) AS avg_cost_usd,
    SUM(estimated_cost_usd) AS total_cost_usd,
    
    -- MAS-specific metrics
    AVG(confidence_score) AS avg_confidence,
    AVG(validation_hallucination_score) AS avg_hallucination_score,
    COUNT(*) FILTER (WHERE validation_is_valid = TRUE) AS validated_answers,
    AVG(mas_tasks_created) AS avg_tasks_created,
    AVG(mas_tasks_completed) AS avg_tasks_completed
FROM query_metrics
WHERE system_type IS NOT NULL
GROUP BY system_type;

COMMENT ON VIEW system_comparison_metrics IS 'Single-Agent vs Multi-Agent comparison - ready for academic papers';

-- Validation analysis
CREATE OR REPLACE VIEW validation_analysis AS
SELECT 
    DATE(qm.created_at) AS date,
    qm.system_type,
    COUNT(*) AS total_queries,
    AVG(qm.validation_hallucination_score) AS avg_hallucination_score,
    STDDEV(qm.validation_hallucination_score) AS stddev_hallucination_score,
    AVG(qm.validation_verified_claims::DOUBLE PRECISION / 
        NULLIF(qm.validation_total_claims::DOUBLE PRECISION, 0)) AS avg_claim_verification_rate,
    COUNT(*) FILTER (WHERE qm.validation_is_valid = TRUE) AS valid_answers,
    COUNT(*) FILTER (WHERE qm.validation_is_valid = FALSE) AS invalid_answers,
    COUNT(*) FILTER (WHERE qm.validation_hallucination_score < 0.3) AS low_hallucination,
    COUNT(*) FILTER (WHERE qm.validation_hallucination_score >= 0.3 AND qm.validation_hallucination_score < 0.6) AS medium_hallucination,
    COUNT(*) FILTER (WHERE qm.validation_hallucination_score >= 0.6) AS high_hallucination
FROM query_metrics qm
WHERE qm.validation_hallucination_score IS NOT NULL
GROUP BY DATE(qm.created_at), qm.system_type
ORDER BY date DESC;

COMMENT ON VIEW validation_analysis IS 'Answer quality and hallucination tracking';

-- Detailed evaluation run metrics (ACADEMIC-READY)
CREATE OR REPLACE VIEW evaluation_run_detailed_metrics AS
SELECT 
    er.id AS run_id,
    er.name AS run_name,
    er.description,
    er.configuration,
    er.status,
    er.started_at,
    er.completed_at,
    EXTRACT(EPOCH FROM (er.completed_at - er.started_at)) AS duration_seconds,
    
    -- Query counts by system type
    COUNT(qm.id) AS total_queries,
    COUNT(qm.id) FILTER (WHERE qm.success = TRUE) AS successful_queries,
    COUNT(qm.id) FILTER (WHERE qm.success = FALSE) AS failed_queries,
    COUNT(qm.id) FILTER (WHERE qm.system_type = 'single-agent') AS single_agent_queries,
    COUNT(qm.id) FILTER (WHERE qm.system_type = 'multi-agent') AS multi_agent_queries,
    
    -- Latency statistics (for statistical tests)
    AVG(qm.total_latency_ms) AS avg_total_latency_ms,
    STDDEV(qm.total_latency_ms) AS stddev_total_latency_ms,
    MIN(qm.total_latency_ms) AS min_latency_ms,
    MAX(qm.total_latency_ms) AS max_latency_ms,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY qm.total_latency_ms) AS p25_latency_ms,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY qm.total_latency_ms) AS p50_latency_ms,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY qm.total_latency_ms) AS p75_latency_ms,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY qm.total_latency_ms) AS p95_latency_ms,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY qm.total_latency_ms) AS p99_latency_ms,
    
    -- Retrieval quality metrics
    AVG(rm.precision_at_k) AS avg_precision,
    STDDEV(rm.precision_at_k) AS stddev_precision,
    AVG(rm.recall_at_k) AS avg_recall,
    STDDEV(rm.recall_at_k) AS stddev_recall,
    AVG(rm.f1_at_k) AS avg_f1,
    STDDEV(rm.f1_at_k) AS stddev_f1,
    AVG(rm.mrr) AS avg_mrr,
    AVG(rm.ndcg_at_k) AS avg_ndcg,
    
    -- Validation metrics
    AVG(qm.validation_hallucination_score) AS avg_hallucination_score,
    STDDEV(qm.validation_hallucination_score) AS stddev_hallucination_score,
    AVG(qm.confidence_score) AS avg_confidence,
    COUNT(qm.id) FILTER (WHERE qm.validation_is_valid = TRUE) AS validated_answers,
    
    -- Cost analysis
    SUM(qm.total_tokens) AS total_tokens,
    AVG(qm.total_tokens) AS avg_tokens_per_query,
    SUM(qm.estimated_cost_usd) AS total_cost_usd,
    AVG(qm.estimated_cost_usd) AS avg_cost_per_query,
    
    -- MAS-specific
    AVG(qm.mas_tasks_created) AS avg_mas_tasks_created,
    AVG(qm.mas_tasks_completed) AS avg_mas_tasks_completed,
    AVG(qm.mas_tasks_failed) AS avg_mas_tasks_failed
FROM evaluation_runs er
LEFT JOIN query_metrics qm ON er.id = qm.evaluation_run_id
LEFT JOIN retrieval_metrics rm ON qm.id = rm.query_metric_id
GROUP BY er.id, er.name, er.description, er.configuration, er.status, er.started_at, er.completed_at;

COMMENT ON VIEW evaluation_run_detailed_metrics IS 'Comprehensive evaluation statistics - ready for academic papers with statistical tests';

-- Evaluation run summary (simplified)
CREATE OR REPLACE VIEW evaluation_run_summary AS
SELECT 
    er.id AS run_id,
    er.name AS run_name,
    er.status,
    er.started_at,
    er.completed_at,
    er.configuration,
    COUNT(qm.id) AS total_queries,
    COUNT(qm.id) FILTER (WHERE qm.success) AS successful_queries,
    COUNT(qm.id) FILTER (WHERE NOT qm.success) AS failed_queries,
    AVG(qm.total_latency_ms) AS avg_latency_ms,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY qm.total_latency_ms) AS p50_latency_ms,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY qm.total_latency_ms) AS p95_latency_ms,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY qm.total_latency_ms) AS p99_latency_ms,
    SUM(qm.total_tokens) AS total_tokens,
    SUM(qm.estimated_cost_usd) AS total_cost_usd,
    AVG(rm.precision_at_k) AS avg_precision,
    AVG(rm.recall_at_k) AS avg_recall,
    AVG(rm.f1_at_k) AS avg_f1,
    AVG(rm.mrr) AS avg_mrr,
    AVG(rm.ndcg_at_k) AS avg_ndcg
FROM evaluation_runs er
LEFT JOIN query_metrics qm ON er.id = qm.evaluation_run_id
LEFT JOIN retrieval_metrics rm ON qm.id = rm.query_metric_id
GROUP BY er.id, er.name, er.status, er.started_at, er.completed_at, er.configuration;

-- Tool usage statistics
CREATE OR REPLACE VIEW tool_usage_stats AS
SELECT 
    tm.tool_name,
    COUNT(*) AS usage_count,
    AVG(tm.latency_ms) AS avg_latency_ms,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY tm.latency_ms) AS p50_latency_ms,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY tm.latency_ms) AS p95_latency_ms,
    COUNT(*) FILTER (WHERE tm.success) AS success_count,
    COUNT(*) FILTER (WHERE NOT tm.success) AS failure_count,
    (COUNT(*) FILTER (WHERE tm.success)::DOUBLE PRECISION / 
     NULLIF(COUNT(*)::DOUBLE PRECISION, 0)) * 100 AS success_rate_pct
FROM tool_metrics tm
GROUP BY tm.tool_name
ORDER BY usage_count DESC;

-- Daily cost analysis
CREATE OR REPLACE VIEW daily_cost_analysis AS
SELECT 
    DATE(qm.created_at) AS date,
    COUNT(*) AS total_queries,
    SUM(qm.total_tokens) AS total_tokens,
    SUM(qm.estimated_cost_usd) AS total_cost_usd,
    AVG(qm.estimated_cost_usd) AS avg_cost_per_query,
    SUM(qm.prompt_tokens) AS total_prompt_tokens,
    SUM(qm.completion_tokens) AS total_completion_tokens
FROM query_metrics qm
GROUP BY DATE(qm.created_at)
ORDER BY date DESC;

-- Search type comparison
CREATE OR REPLACE VIEW search_type_comparison AS
SELECT 
    rm.search_type,
    COUNT(*) AS query_count,
    AVG(rm.precision_at_k) AS avg_precision,
    AVG(rm.recall_at_k) AS avg_recall,
    AVG(rm.f1_at_k) AS avg_f1,
    AVG(rm.mrr) AS avg_mrr,
    AVG(rm.ndcg_at_k) AS avg_ndcg,
    AVG(qm.retrieval_latency_ms) AS avg_retrieval_latency_ms
FROM retrieval_metrics rm
JOIN query_metrics qm ON rm.query_metric_id = qm.id
WHERE rm.precision_at_k IS NOT NULL
GROUP BY rm.search_type;

-- =============================================================================
-- COMPLETION
-- =============================================================================

COMMENT ON DATABASE CURRENT_DATABASE() IS 'Agentic RAG with Multi-Agent System and Knowledge Graph - v3.2 (Ingestion Metrics + Scientific Evaluation)';

-- Optional: Grant permissions (adjust for your environment)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO your_app_user;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO your_app_user;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO your_app_user;
