"""
Tools for the Agentic RAG system.

This module defines the Pydantic data models for tool inputs and the asynchronous
functions that serve as tools for the specialist agents. These tools provide the
core capabilities for searching databases and the knowledge graph.
"""

import logging
from typing import List, Dict, Any, Optional # <--- Import Optional
from pydantic import BaseModel, Field
from pydantic_ai import RunContext

from .db_utils import vector_search, hybrid_search, get_document, list_documents
from .graph_utils import search_graph, get_entity_relationships, get_embedding, test_graph_connection
from .models import AgentDependencies
from .db_utils import test_connection

logger = logging.getLogger(__name__)

# ============================================================================
# INPUT MODELS
# ============================================================================

class VectorSearchInput(BaseModel):
    """Input model for the vector_search_tool."""
    query: str = Field(..., description="The semantic search query text.")
    limit: int = Field(default=5, description="The maximum number of results to return.")

class GraphSearchInput(BaseModel):
    """Input model for the graph_search_tool."""
    query: str = Field(..., description="The search query text for the knowledge graph.")
    limit: int = Field(default=5, description="The maximum number of results to return.")

class HybridSearchInput(BaseModel):
    """Input model for the hybrid_search_tool."""
    query: str = Field(..., description="The search query text.")
    limit: int = Field(default=5, description="The maximum number of results to return.")
    # ✅ FIXED: Added 'text_weight' to match its usage in the RetrievalAgent.
    text_weight: float = Field(default=0.3, description="Weight for full-text search (0.0 to 1.0).")

class DocumentInput(BaseModel):
    """Input model for retrieving a single document by its ID."""
    document_id: str = Field(..., description="The unique identifier for the document.")

class DocumentListInput(BaseModel):
    """Input model for listing available documents."""
    limit: int = Field(default=10, description="The maximum number of documents to return.")

class EntityRelationshipInput(BaseModel):
    """Input model for fetching entity relationships from the knowledge graph."""
    entity_name: str = Field(..., description="The name of the entity to query.")
    # ✅ FIXED: Changed 'max_depth' to 'depth' to match its usage in the KnowledgeAgent.
    depth: int = Field(default=2, description="The maximum traversal depth in the graph.")

class EntityTimelineInput(BaseModel):
    """Input model for fetching an entity's timeline."""
    entity_name: str = Field(..., description="The name of the entity for the timeline.")


# ============================================================================
# TOOL FUNCTIONS
# ============================================================================

async def vector_search_tool(
    ctx: RunContext[AgentDependencies],
    query: str,
    limit: int = 20
) -> List[Dict[str, Any]]:
    """
    Searches for relevant information using vector similarity (semantic search).
    """
    try:
        embedding = await get_embedding(query)
        results = await vector_search(embedding, limit=limit)
        return results
    except Exception as e:
        logger.error(f"Vector search tool failed for query '{query}': {e}", exc_info=True)
        return []

async def graph_search_tool(
    ctx: RunContext[AgentDependencies],
    query: str,
    limit: int = 20,
    center_node_distance: int = 2,
    use_hybrid_search: bool = True
) -> List[Dict[str, Any]]:
    """
    Searches the knowledge graph for entities and direct facts related to the query.
    """
    try:
        results = await search_graph(
            query,
            limit=limit,
            center_node_distance=center_node_distance,
            use_hybrid_search=use_hybrid_search
        )
        return results
    except Exception as e:
        logger.error(f"Graph search tool failed for query '{query}': {e}", exc_info=True)
        return []

async def hybrid_search_tool(
    ctx: RunContext[AgentDependencies],
    query: str,
    limit: int = 20,
    text_weight: float = 0.3  # ✅ FIXED: Added parameter to the function signature.
) -> List[Dict[str, Any]]:
    """
    Searches using a combination of vector similarity and full-text keyword matching.
    """
    try:
        embedding = await get_embedding(query)
        results = await hybrid_search(
            embedding=embedding,
            query_text=query,
            limit=limit,
            text_weight=text_weight  # Use the passed-in value.
        )
        return results
    except Exception as e:
        logger.error(f"Hybrid search tool failed for query '{query}': {e}", exc_info=True)
        return []

async def get_document_tool(
    ctx: RunContext[AgentDependencies],
    document_id: str
) -> Dict[str, Any]:
    """
    Retrieves a specific document by its unique ID.
    """
    try:
        doc = await get_document(document_id)
        if doc:
            return doc
        return {"error": f"Document with ID '{document_id}' not found."}
    except Exception as e:
        logger.error(f"Get document tool failed for ID '{document_id}': {e}", exc_info=True)
        return {"error": str(e)}

async def list_documents_tool(
    ctx: RunContext[AgentDependencies],
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Lists the documents available in the database.
    """
    try:
        return await list_documents(limit=limit)
    except Exception as e:
        logger.error(f"List documents tool failed: {e}", exc_info=True)
        return []

async def get_entity_relationships_tool(
    entity_name: str,
    depth: int = 2,
    # --- THIS IS THE FIX ---
    ctx: Optional[RunContext[AgentDependencies]] = None,
    # -----------------------
) -> Dict[str, Any]:
    """
    Gets relationships for a given entity from the knowledge graph up to a specified depth.
    """
    try:
        relationships = await get_entity_relationships(entity_name, depth=depth)
        return relationships
    except Exception as e:
        logger.error(f"Get entity relationships tool failed for '{entity_name}': {e}", exc_info=True)
        return {"error": str(e)}

async def get_entity_timeline_tool(
    entity_name: str,
    # --- THIS IS THE FIX ---
    ctx: Optional[RunContext[AgentDependencies]] = None,
    # -----------------------
) -> List[Dict[str, Any]]:
    """
    Gets a timeline of temporal events related to a specific entity.
    """
    try:
        # This is a placeholder; a real implementation would query the graph/DB for temporal facts.
        logger.info(f"Generating placeholder timeline for entity: {entity_name}")
        return [{
            "date": "2025-10-16",
            "event": f"A significant event related to {entity_name} occurred.",
            "source": "Knowledge Graph (Temporal)"
        }]
    except Exception as e:
        logger.error(f"Get entity timeline tool failed for '{entity_name}': {e}", exc_info=True)
        return []

# ============================================================================
# HEALTH CHECK FUNCTIONS
# ============================================================================

async def check_tools_health() -> Dict[str, bool]:
    """
    Runs a health check on the tool dependencies and returns simple boolean statuses.
    """
    from .db_utils import test_connection
    from .graph_utils import test_graph_connection
    
    db_ok = await test_connection()
    graph_ok = await test_graph_connection()
    
    # ✅ FIXED: Return simple boolean values to match the HealthResponse Pydantic model.
    return {
        "database_connection": db_ok,
        "graph_connection": graph_ok,
        "overall_status": db_ok and graph_ok
    }

def get_tool_performance_summary() -> Dict[str, Any]:
    """
    Gets a placeholder performance summary for the tools.
    In a real system, this would track actual metrics over time.
    """
    return {
        "vector_search": {"avg_latency_ms": 50, "success_rate": 0.98},
        "graph_search": {"avg_latency_ms": 30, "success_rate": 0.99},
        "hybrid_search": {"avg_latency_ms": 80, "success_rate": 0.97}
    }