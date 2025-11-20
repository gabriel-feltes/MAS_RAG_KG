"""
Main Pydantic AI agent for agentic RAG and knowledge graph: MAS + KG + RAG focus.
"""

import os
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext

from .prompts import SYSTEM_PROMPT
from .providers import get_model
from .tools import (
    vector_search_tool,
    graph_search_tool,
    hybrid_search_tool,
    get_document_tool,
    list_documents_tool,
    get_entity_relationships_tool,
    get_entity_timeline_tool,
    # Input Pydantic models
    VectorSearchInput,
    GraphSearchInput,
    HybridSearchInput,
    DocumentInput,
    DocumentListInput,
    EntityRelationshipInput,
    EntityTimelineInput
)

# Load environment variables
load_dotenv()
logger = logging.getLogger(__name__)

@dataclass
class AgentDependencies:
    """Dependencies for the agent."""
    session_id: str
    user_id: Optional[str] = None
    search_preferences: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.search_preferences is None:
            self.search_preferences = {
                "use_vector": True,
                "use_graph": True,
                "default_limit": 10
            }

# Initialize agent with explicit system prompt and selected model
rag_agent = Agent(
    get_model(),
    deps_type=AgentDependencies,
    system_prompt=SYSTEM_PROMPT,
)

# ========== Tool registrations ==========

@rag_agent.tool
async def vector_search(
    ctx: RunContext[AgentDependencies],
    query: str,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Semantic similarity vector search across all document chunks.
    """
    input_ = VectorSearchInput(query=query, limit=limit)
    results = await vector_search_tool(input_)
    return [r.model_dump() for r in results]

@rag_agent.tool
async def hybrid_search(
    ctx: RunContext[AgentDependencies],
    query: str,
    limit: int = 10,
    text_weight: float = 0.3
) -> List[Dict[str, Any]]:
    """
    Combines vector and keyword search for best recall (hybrid).
    """
    input_ = HybridSearchInput(query=query, limit=limit, text_weight=text_weight)
    results = await hybrid_search_tool(input_)
    return [r.model_dump() for r in results]

@rag_agent.tool
async def graph_search(
    ctx: RunContext[AgentDependencies],
    query: str
) -> List[Dict[str, Any]]:
    """
    Search knowledge graph for facts/relationships.
    """
    input_ = GraphSearchInput(query=query)
    results = await graph_search_tool(input_)
    return [r.model_dump() for r in results]

@rag_agent.tool
async def get_document(
    ctx: RunContext[AgentDependencies],
    document_id: str
) -> Optional[Dict[str, Any]]:
    """
    Fetch complete document (all metadata and content).
    """
    input_ = DocumentInput(document_id=document_id)
    doc = await get_document_tool(input_)
    return doc.model_dump() if doc else None

@rag_agent.tool
async def list_documents(
    ctx: RunContext[AgentDependencies],
    limit: int = 20,
    offset: int = 0
) -> List[Dict[str, Any]]:
    """
    List all documents in the knowledge base.
    """
    input_ = DocumentListInput(limit=limit, offset=offset)
    results = await list_documents_tool(input_)
    return [d.model_dump() for d in results]

@rag_agent.tool
async def get_entity_relationships(
    ctx: RunContext[AgentDependencies],
    entity_name: str,
    depth: int = 2
) -> Dict[str, Any]:
    """
    Get all relationships for an entity from the knowledge graph.
    """
    input_ = EntityRelationshipInput(entity_name=entity_name, depth=depth)
    return await get_entity_relationships_tool(input_)

@rag_agent.tool
async def get_entity_timeline(
    ctx: RunContext[AgentDependencies],
    entity_name: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Retrieve full temporal sequence of facts for an entity.
    """
    input_ = EntityTimelineInput(entity_name=entity_name, start_date=start_date, end_date=end_date)
    return await get_entity_timeline_tool(input_)
