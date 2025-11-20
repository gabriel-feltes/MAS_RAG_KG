"""
Pydantic models for data validation and serialization.
"""

from typing import List, Dict, Any, Optional, Literal
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict, field_validator
from enum import Enum
import os

VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", 1536))


class MessageRole(str, Enum):
    """Message role enumeration."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class SearchType(str, Enum):
    """Search type enumeration."""
    VECTOR = "vector"
    HYBRID = "hybrid"
    GRAPH = "graph"


# ============================================================================
# REQUEST MODELS
# ============================================================================

class ChatRequest(BaseModel):
    """Chat request model."""
    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    user_id: Optional[str] = Field(None, description="User identifier")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    search_type: SearchType = Field(default=SearchType.HYBRID, description="Type of search to perform")
    
    model_config = ConfigDict(use_enum_values=True)


class SearchRequest(BaseModel):
    """Search request model."""
    query: str = Field(..., description="Search query")
    search_type: SearchType = Field(default=SearchType.HYBRID, description="Type of search")
    limit: int = Field(default=10, ge=1, le=50, description="Maximum results")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Search filters")
    
    model_config = ConfigDict(use_enum_values=True)


# ============================================================================
# RESPONSE MODELS
# ============================================================================

class DocumentMetadata(BaseModel):
    """Document metadata model."""
    id: str
    title: str
    source: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
    chunk_count: Optional[int] = None


class ChunkResult(BaseModel):
    """Chunk search result model."""
    chunk_id: str
    document_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    document_title: str
    document_source: str
    
    @field_validator('score')
    @classmethod
    def validate_score(cls, v: float) -> float:
        """Ensure score is between 0 and 1."""
        return max(0.0, min(1.0, v))


class GraphSearchResult(BaseModel):
    """Knowledge graph search result model."""
    fact: str
    uuid: str
    valid_at: Optional[str] = None
    invalid_at: Optional[str] = None
    source_node_uuid: Optional[str] = None


class EntityRelationship(BaseModel):
    """Entity relationship model."""
    from_entity: str
    to_entity: str
    relationship_type: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    """Search response model."""
    results: List[ChunkResult] = Field(default_factory=list)
    graph_results: List[GraphSearchResult] = Field(default_factory=list)
    total_results: int = 0
    search_type: SearchType
    query_time_ms: float


class ToolCall(BaseModel):
    """Tool call information model."""
    tool_name: str
    args: Dict[str, Any] = Field(default_factory=dict)
    tool_call_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Chat response model."""
    message: str
    session_id: str
    sources: List[DocumentMetadata] = Field(default_factory=list)
    tools_used: List[ToolCall] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StreamDelta(BaseModel):
    """Streaming response delta."""
    content: str
    delta_type: Literal["text", "tool_call", "end"] = "text"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    error_type: str
    details: Optional[Dict[str, Any]] = None
    request_id: Optional[str] = None


# ============================================================================
# DATABASE MODELS
# ============================================================================

class Document(BaseModel):
    """Document model."""
    id: Optional[str] = None
    title: str
    source: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class Chunk(BaseModel):
    """Document chunk model."""
    id: Optional[str] = None
    document_id: str
    content: str
    embedding: Optional[List[float]] = None
    chunk_index: int
    metadata: Dict[str, Any] = Field(default_factory=dict)
    token_count: Optional[int] = None
    created_at: Optional[datetime] = None
    
    @field_validator('embedding')
    @classmethod
    def validate_embedding(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        """Validate embedding dimensions."""
        if v is not None and len(v) != VECTOR_DIMENSION:
            raise ValueError(f"Embedding must have {VECTOR_DIMENSION} dimensions, got {len(v)}")
        return v


class Session(BaseModel):
    """Session model."""
    id: Optional[str] = None
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None


class Message(BaseModel):
    """Message model."""
    id: Optional[str] = None
    session_id: str
    role: MessageRole
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = None
    
    model_config = ConfigDict(use_enum_values=True)


# ============================================================================
# INGESTION MODELS
# ============================================================================

class IngestionConfig(BaseModel):
    """Configuration for document ingestion."""
    chunk_size: int = Field(default=800, ge=100, le=5000, description="Size of each chunk in characters")
    chunk_overlap: int = Field(default=100, ge=0, le=1000, description="Overlap between chunks")
    max_chunk_size: int = Field(default=1500, ge=500, le=10000, description="Maximum chunk size")
    use_semantic_chunking: bool = Field(default=False, description="Use semantic chunking (slower, more costly)")
    extract_entities: bool = Field(default=True, description="Extract entities from chunks")
    skip_graph_building: bool = Field(default=False, description="Skip knowledge graph building for faster ingestion")
    
    @field_validator('chunk_overlap')
    @classmethod
    def validate_overlap(cls, v: int, info) -> int:
        """Ensure overlap is less than chunk size."""
        chunk_size = info.data.get('chunk_size', 800)
        if v >= chunk_size:
            raise ValueError(f"Chunk overlap ({v}) must be less than chunk size ({chunk_size})")
        return v


class IngestionResult(BaseModel):
    """Result of document ingestion."""
    document_id: str
    title: str
    chunks_created: int
    entities_extracted: int
    relationships_created: int
    processing_time_ms: float
    errors: List[str] = Field(default_factory=list)


class IngestionMetric(BaseModel):
    """Ingestion metric for a single document."""
    id: Optional[str] = None
    session_id: str
    document_id: str
    document_title: str
    chunks_count: int
    entities_count: int
    relationships_count: int
    embedding_tokens: int
    embedding_cost_usd: float
    graph_tokens: int
    graph_cost_usd: float
    total_cost_usd: float
    processing_time_ms: float
    status: Literal["success", "failed", "partial"]
    error_message: Optional[str] = None
    created_at: Optional[datetime] = None


class IngestionSummary(BaseModel):
    """Summary of ingestion session."""
    session_id: str
    total_documents: int
    total_chunks: int
    total_entities: int
    total_relationships: int
    embedding_tokens: int
    embedding_cost_usd: float
    graph_tokens: int
    graph_cost_usd: float
    total_cost_usd: float
    avg_processing_time_ms: float
    errors_count: int


# ============================================================================
# AGENT MODELS
# ============================================================================

class AgentDependencies(BaseModel):
    """Dependencies for the agent."""
    session_id: str
    database_url: Optional[str] = None
    neo4j_uri: Optional[str] = None
    openai_api_key: Optional[str] = None
    
    model_config = ConfigDict(arbitrary_types_allowed=True)


class AgentContext(BaseModel):
    """Agent execution context."""
    session_id: str
    messages: List[Message] = Field(default_factory=list)
    tool_calls: List[ToolCall] = Field(default_factory=list)
    search_results: List[ChunkResult] = Field(default_factory=list)
    graph_results: List[GraphSearchResult] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# HEALTH CHECK MODELS
# ============================================================================

class HealthStatus(BaseModel):
    """Health check status."""
    status: Literal["healthy", "degraded", "unhealthy"]
    database: bool
    graph_database: bool
    llm_connection: bool
    version: str
    timestamp: datetime
    details: Dict[str, Any] = Field(default_factory=dict)
