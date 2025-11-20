"""
Retrieval Specialist Agent.

This agent is responsible for all search and retrieval operations:
- Vector search (semantic similarity)
- Graph search (knowledge graph)
- Hybrid search (combined vector + keyword)
- Document listing and full retrieval
"""

import logging
from typing import Dict, Any, List, Optional

from ..base_agent import BaseAgent
from ..state import (
    AgentState,
    AgentRole,
    SubTask,
    TaskType,
    RetrievalResult
)

# Import tools
try:
    from ...agent.tools import (
        vector_search_tool,
        graph_search_tool,
        hybrid_search_tool,
        get_document_tool,
        list_documents_tool,
        VectorSearchInput,
        GraphSearchInput,
        HybridSearchInput,
        DocumentInput,
        DocumentListInput
    )
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from agent.tools import (
        vector_search_tool,
        graph_search_tool,
        hybrid_search_tool,
        get_document_tool,
        list_documents_tool,
        VectorSearchInput,
        GraphSearchInput,
        HybridSearchInput,
        DocumentInput,
        DocumentListInput
    )

logger = logging.getLogger(__name__)


# ============================================================================
# RETRIEVAL SPECIALIST AGENT
# ============================================================================

class RetrievalAgent(BaseAgent):
    """
    Specialist agent for search and retrieval operations in scientific literature.
    
    Responsibilities:
    - Execute vector similarity search (semantic search over research papers)
    - Execute knowledge graph search (academic relationships and citations)
    - Execute hybrid search (vector + keyword for academic queries)
    - Execute document list and get operations (research article retrieval)
    - Re-rank and filter results (prioritize relevant papers)
    - Aggregate results from multiple searches
    
    Capabilities:
    - Semantic search using embeddings (find related research)
    - Graph traversal and pattern matching (citation networks, author collaborations)
    - Keyword-based full-text search (method names, concepts)
    - Full document retrieval (complete research articles)
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize retrieval agent for scientific literature.
        
        Args:
            model_name: Optional LLM model name
        """
        super().__init__(
            role=AgentRole.RETRIEVAL,
            name="Retrieval Specialist",
            description="Expert in scientific literature search and information retrieval across vector and graph databases",
            model_name=model_name
        )
        
        self.tools = [
            vector_search_tool,
            graph_search_tool,
            hybrid_search_tool,
            list_documents_tool,
            get_document_tool
        ]
        
        # Retrieval statistics
        self.total_chunks_retrieved = 0
        self.total_graph_facts_retrieved = 0
        self.total_papers_retrieved = 0
        self.avg_relevance_scores = []
        
        self.log_info("Initialized with vector, graph, hybrid, and document retrieval capabilities for scientific literature")
    
    # ========================================================================
    # ABSTRACT METHOD IMPLEMENTATIONS
    # ========================================================================
    
    async def process_task(self, state: Dict[str, Any], task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a retrieval task by dispatching to the correct search method.
        """
        task_type = task.get("task_type", "hybrid_search").lower()
        params = task.get("parameters", {})
        query = params.get("query") or state.get("query")
        
        self.log_info(f"Processing retrieval task of type: {task_type}")

        task_handlers = {
            "hybrid": self._execute_hybrid_search,
            "vector": self._execute_vector_search,
            "graph": self._execute_graph_search,
            "list_documents": self._execute_list_documents,
            "get_document": self._execute_get_document
        }

        # Default to hybrid search if no specific type is matched
        handler = task_handlers.get("hybrid")
        
        for keyword, method in task_handlers.items():
            if keyword in task_type:
                handler = method
                break
        
        # Query is not required for list_documents or get_document
        if not query and "list_documents" not in task_type and "get_document" not in task_type:
            raise ValueError("Query is missing from both task parameters and agent state.")
            
        try:
            self.log_info(f"Dispatching to handler: {handler.__name__}")
            results = await handler(params)
            return results
        except Exception as e:
            self.log_error(f"An unexpected error occurred in {handler.__name__}: {e}")
            raise

    def can_handle_task(self, task: Dict[str, Any]) -> bool:
        """Check if this agent can handle the task."""
        task_type = task.get("task_type", "")
        agent_role = task.get("agent_role", "")
        
        valid_types = [
            "retrieval", "vector_search", "graph_search", "hybrid_search",
            "list_documents", "get_document",
            "RETRIEVAL", "VECTOR_SEARCH", "GRAPH_SEARCH", "HYBRID_SEARCH",
            "LIST_DOCUMENTS", "GET_DOCUMENT"
        ]
        
        valid_roles = ["retrieval", "RETRIEVAL", AgentRole.RETRIEVAL.value]
        
        return task_type in valid_types or agent_role in valid_roles
    
    def get_system_prompt(self) -> str:
        """
        Get system prompt for LLM interactions.
        
        Returns:
            System prompt string
        """
        return """You are a Retrieval Specialist Agent in a multi-agent system focused on scientific literature.

Your responsibilities:
1. Execute vector similarity searches using semantic embeddings over research papers
2. Execute knowledge graph searches to find academic facts, citations, and relationships
3. Execute hybrid searches combining vector and keyword matching for academic queries
4. Execute requests to list all research articles or fetch a specific paper
5. Analyze search results and determine relevance to research questions
6. Re-rank results when multiple sources are used, prioritizing highly cited and relevant papers

Your expertise includes:
- Understanding semantic similarity and vector space models for academic text
- Knowledge graph traversal for citation networks and author collaborations
- Information retrieval metrics (precision, recall, relevance) in scholarly contexts
- Query reformulation and expansion for scientific literature
- Academic metadata extraction (authors, venues, publication dates)

Always prioritize:
- Retrieval precision (return only relevant research papers)
- Context preservation (maintain citation information, authorship)
- Efficiency (avoid redundant searches across paper databases)
- Result diversity (cover different methodologies, time periods, research groups)
- Citation quality (prioritize influential and well-cited papers when appropriate)

When providing results, include:
- Retrieved text chunks from papers with similarity scores
- Knowledge graph facts with source nodes (authors, papers, methods)
- Source document metadata (title, authors, venue, publication date)
- Aggregated relevance metrics
- Citation context when available"""

    # ============================================================================
    # RETRIEVAL METHODS - UNBIASED VERSION
    # ============================================================================

    async def _execute_vector_search(
        self,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute vector similarity search for semantic retrieval over research papers.
        
        ✅ FIXED: Now uses limit (not limit*2) to ensure fair comparison with graph search.
        """
        query = parameters.get("query", "")
        limit = parameters.get("limit", 10)
        
        self.log_info(f"Executing vector search for scientific literature: '{query[:50]}...' (limit={limit})")
        
        try:
            # ✅ BIAS FIX: Changed from limit*2 to limit
            chunks = await vector_search_tool(
                ctx=None,
                query=query,
                limit=limit  # ← FIXED: Was limit*2 before
            )
            
            if chunks is None: chunks = []
            if not isinstance(chunks, list):
                self.log_warning(f"Vector search tool returned non-list type: {type(chunks)}")
                chunks = []
                
            chunk_data = [
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "document_id": chunk.get("document_id"),
                    "content": chunk.get("content"),
                    "score": chunk.get("score") or chunk.get("similarity", 0.0),
                    "document_title": chunk.get("document_title", "Unknown Paper"),
                    "document_source": chunk.get("document_source", "Unknown Source"),
                    "metadata": chunk.get("metadata", {}),
                    # Scientific-specific metadata
                    "authors": chunk.get("metadata", {}).get("authors", []),
                    "publication_date": chunk.get("metadata", {}).get("publication_date"),
                    "venue": chunk.get("metadata", {}).get("venue"),
                    "citation_count": chunk.get("metadata", {}).get("citation_count", 0)
                }
                for chunk in chunks if isinstance(chunk, dict)
            ]
            
            # ✅ STABLE SORTING for determinism
            chunk_data = self._sort_chunks_stable(chunk_data)
            
            avg_score = sum(c["score"] for c in chunk_data) / len(chunk_data) if chunk_data else 0.0
            
            result = {
                "search_type": "vector",
                "chunks": chunk_data,
                "graph_facts": [],
                "total_results": len(chunk_data),
                "avg_score": avg_score,
                "sources": list(set(c.get("document_source") for c in chunk_data if c.get("document_source"))),
                "unique_papers": len(set(c.get("document_id") for c in chunk_data)),
                "query": query,
                "limit": limit,
                "retrieval_context": "scientific_literature"
            }
            
            self.total_papers_retrieved += result["unique_papers"]
            self.log_info(
                f"Vector search completed: {len(chunk_data)} chunks from "
                f"{result['unique_papers']} papers, avg_score={avg_score:.3f}"
            )
            return result
        except Exception as e:
            self.log_error(f"Vector search failed: {e}", exc_info=True)
            raise


    async def _execute_graph_search(
        self,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute knowledge graph search for academic relationships (citations, authorship).
        
        ✅ UNCHANGED: Already using limit correctly.
        """
        query = parameters.get("query", "")
        limit = parameters.get("limit", 10)
        center_node_distance = parameters.get("center_node_distance", 2)
        use_hybrid_search = parameters.get("use_hybrid_search", True)

        self.log_info(
            f"Executing graph search for academic relationships: '{query[:50]}...' "
            f"(limit={limit}, depth={center_node_distance})"
        )
        
        try:
            graph_results = await graph_search_tool(
                ctx=None,
                query=query,
                limit=limit,
                center_node_distance=center_node_distance,
                use_hybrid_search=use_hybrid_search
            )

            if graph_results is None: graph_results = []
            if not isinstance(graph_results, list):
                self.log_warning(f"Graph search tool returned non-list type: {type(graph_results)}")
                graph_results = []
                
            facts_data = [
                {
                    "fact": fact.get("fact"),
                    "uuid": fact.get("uuid"),
                    "valid_at": fact.get("valid_at"),
                    "invalid_at": fact.get("invalid_at"),
                    "source_node_uuid": fact.get("source_node_uuid"),
                    # Scientific context
                    "fact_type": self._classify_fact_type(fact.get("fact", ""))
                }
                for fact in graph_results if isinstance(fact, dict)
            ]

            result = {
                "search_type": "graph",
                "chunks": [],
                "graph_facts": facts_data,
                "total_results": len(facts_data),
                "avg_score": 1.0,
                "query": query,
                "limit": limit,
                "retrieval_context": "scientific_kg"
            }
            self.log_info(f"Graph search completed: {len(facts_data)} academic facts retrieved")
            return result
        except Exception as e:
            self.log_error(f"Graph search failed: {e}", exc_info=True)
            raise


    async def _execute_hybrid_search(
        self,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute hybrid search (vector + keyword + graph) for scientific literature.
        
        ✅ FIXED: Now retrieves equal amounts from both vector and graph (limit each).
        """
        query = parameters.get("query", "")
        limit = parameters.get("limit", 10)
        text_weight = parameters.get("text_weight", 0.3)
        center_node_distance = parameters.get("center_node_distance", 2)
        use_hybrid_search = parameters.get("use_hybrid_search", True)

        self.log_info(
            f"Executing hybrid search for scientific literature: '{query[:50]}...' "
            f"(limit={limit}, text_weight={text_weight})"
        )
        
        try:
            # Vector + keyword search
            chunks = await hybrid_search_tool(
                ctx=None,
                query=query, 
                limit=limit,  # ✅ OK: Already using limit
                text_weight=text_weight
            )
            
            if chunks is None: chunks = []
            if not isinstance(chunks, list):
                self.log_warning(f"Hybrid search tool (vector) returned non-list type: {type(chunks)}")
                chunks = []
                
            chunk_data = [
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "document_id": chunk.get("document_id"),
                    "content": chunk.get("content"),
                    "score": chunk.get("score") or chunk.get("combined_score", 0.0),
                    "document_title": chunk.get("document_title", "Unknown Paper"),
                    "document_source": chunk.get("document_source", "Unknown Source"),
                    "metadata": chunk.get("metadata", {}),
                    # Scientific-specific metadata
                    "authors": chunk.get("metadata", {}).get("authors", []),
                    "publication_date": chunk.get("metadata", {}).get("publication_date"),
                    "venue": chunk.get("metadata", {}).get("venue"),
                    "citation_count": chunk.get("metadata", {}).get("citation_count", 0)
                }
                for chunk in chunks if isinstance(chunk, dict)
            ]
            
            # ✅ STABLE SORTING for determinism
            chunk_data = self._sort_chunks_stable(chunk_data)
            
            avg_score = sum(c["score"] for c in chunk_data) / len(chunk_data) if chunk_data else 0.0
            
            # ✅ BIAS FIX: Graph search now uses full limit (not limit//2)
            facts_data = []
            try:
                graph_results = await graph_search_tool(
                    ctx=None,
                    query=query,
                    limit=limit,  # ← FIXED: Was limit//2 before
                    center_node_distance=center_node_distance,
                    use_hybrid_search=use_hybrid_search
                )
                
                if graph_results is None: graph_results = []
                if not isinstance(graph_results, list):
                    self.log_warning(f"Graph search tool returned non-list type: {type(graph_results)}")
                    graph_results = []

                facts_data = [
                    {
                        "fact": fact.get("fact"),
                        "uuid": fact.get("uuid"),
                        "valid_at": fact.get("valid_at"),
                        "invalid_at": fact.get("invalid_at"),
                        "fact_type": self._classify_fact_type(fact.get("fact", ""))
                    }
                    for fact in graph_results if isinstance(fact, dict)
                ]
                
                # ✅ STABLE SORTING for facts (by uuid)
                facts_data = sorted(facts_data, key=lambda f: f.get('uuid', ''))
                
            except Exception as e:
                self.log_warning(f"Graph search within hybrid search failed: {e}", exc_info=True)
                
            result = {
                "search_type": "hybrid",
                "chunks": chunk_data,
                "graph_facts": facts_data,
                "total_results": len(chunk_data) + len(facts_data),
                "avg_score": avg_score,
                "sources": list(set(c.get("document_source") for c in chunk_data if c.get("document_source"))),
                "unique_papers": len(set(c.get("document_id") for c in chunk_data)),
                "query": query,
                "limit": limit,
                "text_weight": text_weight,
                "retrieval_context": "scientific_literature_hybrid"
            }
            
            self.total_papers_retrieved += result["unique_papers"]
            self.log_info(
                f"Hybrid search completed: {len(chunk_data)} chunks from "
                f"{result['unique_papers']} papers, {len(facts_data)} facts, avg_score={avg_score:.3f}"
            )
            
            return result

        except Exception as e:
            self.log_error(f"Hybrid search execution failed: {e}", exc_info=True)
            raise


    async def _execute_list_documents(
        self,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute document listing to retrieve available research articles.
        
        ✅ UNCHANGED: Correctly uses limit.
        """
        limit = parameters.get("limit", 10)
        self.log_info(f"Executing list_documents tool for research articles (limit={limit})")
        
        try:
            documents = await list_documents_tool(ctx=None, limit=limit)
            
            if documents is None: documents = []
            if not isinstance(documents, list):
                self.log_warning(f"list_documents_tool returned non-list: {type(documents)}")
                documents = []

            # Format documents as "chunks" for synthesis
            chunk_data = [
                {
                    "chunk_id": doc.get("id"),
                    "document_id": doc.get("id"),
                    "content": self._format_paper_summary(doc),
                    "score": 1.0,
                    "document_title": doc.get("title", "Unknown Paper"),
                    "document_source": doc.get("source", "Unknown Source"),
                    "metadata": doc.get("metadata", {}),
                    # Scientific metadata
                    "authors": doc.get("metadata", {}).get("authors", []),
                    "publication_date": doc.get("metadata", {}).get("publication_date"),
                    "venue": doc.get("metadata", {}).get("venue")
                }
                for doc in documents if isinstance(doc, dict)
            ]

            result = {
                "search_type": "list_documents",
                "chunks": chunk_data,
                "graph_facts": [],
                "total_results": len(chunk_data),
                "avg_score": 1.0,
                "sources": list(set(c.get("document_source") for c in chunk_data)),
                "unique_papers": len(chunk_data),
                "query": "list_documents",
                "limit": limit,
                "retrieval_context": "document_listing"
            }
            
            self.total_papers_retrieved += result["unique_papers"]
            self.log_info(f"List documents completed: {len(chunk_data)} research articles found")
            return result
        except Exception as e:
            self.log_error(f"List documents failed: {e}", exc_info=True)
            raise


    async def _execute_get_document(
        self,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute single document retrieval to fetch a complete research article.
        
        ✅ UNCHANGED: Single document retrieval.
        """
        doc_id = parameters.get("document_id", "")
        if not doc_id:
            raise ValueError("document_id is required for get_document")
        
        self.log_info(f"Executing get_document tool for research article ID: {doc_id}")
        
        try:
            document = await get_document_tool(ctx=None, document_id=doc_id)
            
            chunk_data = []
            sources = []
            
            if document and isinstance(document, dict):
                source = document.get("source", "Unknown Source")
                title = document.get("title", "Unknown Paper")
                sources.append(source)
                
                chunk_data.append({
                    "chunk_id": document.get("id"),
                    "document_id": document.get("id"),
                    "content": document.get("content", ""),
                    "score": 1.0,
                    "document_title": title,
                    "document_source": source,
                    "metadata": document.get("metadata", {}),
                    # Scientific metadata
                    "authors": document.get("metadata", {}).get("authors", []),
                    "publication_date": document.get("metadata", {}).get("publication_date"),
                    "venue": document.get("metadata", {}).get("venue"),
                    "abstract": document.get("metadata", {}).get("abstract")
                })

            result = {
                "search_type": "get_document",
                "chunks": chunk_data,
                "graph_facts": [],
                "total_results": len(chunk_data),
                "avg_score": 1.0,
                "sources": sources,
                "unique_papers": len(chunk_data),
                "query": f"get_document:{doc_id}",
                "limit": 1,
                "retrieval_context": "full_document"
            }
            
            self.total_papers_retrieved += result["unique_papers"]
            self.log_info(f"Get document completed: {len(chunk_data)} research article retrieved")
            return result
        except Exception as e:
            self.log_error(f"Get document failed: {e}", exc_info=True)
            raise

    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _classify_fact_type(self, fact: str) -> str:
        """
        Classify a knowledge graph fact by type (academic context).
        
        Args:
            fact: Fact text
        
        Returns:
            Fact type
        """
        fact_lower = fact.lower()
        
        if any(kw in fact_lower for kw in ["authored", "author", "wrote"]):
            return "authorship"
        elif any(kw in fact_lower for kw in ["cites", "citation", "references"]):
            return "citation"
        elif any(kw in fact_lower for kw in ["published", "appeared in"]):
            return "publication"
        elif any(kw in fact_lower for kw in ["proposes", "introduces", "presents"]):
            return "contribution"
        elif any(kw in fact_lower for kw in ["affiliated", "institution", "university"]):
            return "affiliation"
        else:
            return "general"
    
    def _format_paper_summary(self, doc: Dict[str, Any]) -> str:
        """
        Format a paper document as a readable summary.
        
        Args:
            doc: Document dict
        
        Returns:
            Formatted summary
        """
        title = doc.get("title", "N/A")
        source = doc.get("source", "N/A")
        authors = doc.get("metadata", {}).get("authors", [])
        venue = doc.get("metadata", {}).get("venue", "N/A")
        date = doc.get("metadata", {}).get("publication_date", "N/A")
        
        author_str = ", ".join(authors[:3]) if authors else "N/A"
        if len(authors) > 3:
            author_str += " et al."
        
        return f"""Title: {title}
Authors: {author_str}
Venue: {venue}
Publication Date: {date}
Source: {source}"""
    
    def _sort_chunks_stable(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sort chunks with stable ordering for determinism.
        
        Primary sort: score (descending)
        Tie-breakers: chunk_id, document_id, document_title (ascending)
        
        This ensures that chunks with identical scores always appear
        in the same order, eliminating non-determinism from embedding APIs.
        
        Args:
            chunks: List of chunks to sort
        
        Returns:
            Sorted chunks with deterministic ordering
        """
        return sorted(
            chunks,
            key=lambda c: (
                -round(c.get('score', 0.0), 6),  # Score descending (round to 6 decimals)
                c.get('chunk_id', ''),           # Tie-breaker 1
                c.get('document_id', ''),        # Tie-breaker 2
                c.get('document_title', '')      # Tie-breaker 3
            )
        )
    
    def _update_statistics(self, result: Dict[str, Any]):
        """
        Update agent statistics with retrieval results.
        
        Args:
            result: Retrieval results
        """
        chunks = result.get("chunks", [])
        facts = result.get("graph_facts", [])
        
        self.total_chunks_retrieved += len(chunks)
        self.total_graph_facts_retrieved += len(facts)
        
        if result.get("avg_score") is not None:
            self.avg_relevance_scores.append(result["avg_score"])
    
    def _store_results_in_state(
        self,
        state: AgentState,
        result: Dict[str, Any],
        task: SubTask
    ):
        """
        Store retrieval results in state for other agents.
        
        Args:
            state: Current state
            result: Retrieval results
            task: Original task
        """
        retrieval_result = RetrievalResult(
            search_type=result["search_type"],
            chunks=result["chunks"],
            graph_facts=result["graph_facts"],
            total_results=result["total_results"],
            avg_score=result["avg_score"],
            sources=result["sources"]
        )
        
        state["retrieved_contexts"].append(retrieval_result)
        
        self.log_debug(f"Stored {result['total_results']} results in state")
    
    def get_retrieval_statistics(self) -> Dict[str, Any]:
        """
        Get retrieval-specific statistics.
        
        Returns:
            Dict with retrieval stats
        """
        avg_relevance = (
            sum(self.avg_relevance_scores) / len(self.avg_relevance_scores)
            if self.avg_relevance_scores
            else 0.0
        )
        
        base_stats = self.get_performance_stats()
        
        return {
            **base_stats,
            "total_chunks_retrieved": self.total_chunks_retrieved,
            "total_graph_facts_retrieved": self.total_graph_facts_retrieved,
            "total_papers_retrieved": self.total_papers_retrieved,
            "avg_relevance_score": avg_relevance,
            "chunks_per_execution": (
                self.total_chunks_retrieved / self.total_executions
                if self.total_executions > 0
                else 0.0
            ),
            "papers_per_execution": (
                self.total_papers_retrieved / self.total_executions
                if self.total_executions > 0
                else 0.0
            ),
            "domain": "scientific_literature"
        }
    
    # ========================================================================
    # ADVANCED RETRIEVAL METHODS
    # ========================================================================
    
    async def rerank_results(
        self,
        results: List[Dict[str, Any]],
        query: str
    ) -> List[Dict[str, Any]]:
        """
        Re-rank retrieval results prioritizing highly cited papers and relevance.
        
        Args:
            results: List of search results
            query: Original query
        
        Returns:
            Re-ranked results
        """
        self.log_info(f"Re-ranking {len(results)} results for scientific literature")
        
        # Sort by combination of relevance score and citation count
        def ranking_score(result):
            base_score = result.get("score", 0.0)
            citation_count = result.get("citation_count", 0)
            # Normalize citation count (log scale) and combine with relevance
            citation_boost = min(citation_count / 100.0, 1.0) * 0.2
            return base_score + citation_boost
        
        return sorted(results, key=ranking_score, reverse=True)
    
    async def deduplicate_results(
        self,
        results: List[Dict[str, Any]],
        threshold: float = 0.9
    ) -> List[Dict[str, Any]]:
        """
        Remove duplicate or highly similar results (same paper, different chunks).
        
        Args:
            results: List of search results
            threshold: Similarity threshold for deduplication
        
        Returns:
            Deduplicated results
        """
        self.log_info(f"Deduplicating {len(results)} results")
        
        seen_ids = set()
        deduplicated = []
        
        for result in results:
            chunk_id = result.get("chunk_id")
            if chunk_id and chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                deduplicated.append(result)
        
        self.log_info(f"Removed {len(results) - len(deduplicated)} duplicates")
        
        return deduplicated
    
    async def expand_query(self, query: str) -> List[str]:
        """
        Expand query with scientific synonyms and related terms.
        
        Args:
            query: Original query
        
        Returns:
            List of expanded queries
        """
        self.log_info(f"Expanding scientific query: '{query}'")
        
        prompt = f"""Generate 2-3 alternative scientific phrasings or related queries for:
"{query}"

Consider:
- Scientific terminology variations
- Method name synonyms
- Concept-related terms

Return as a JSON array of strings."""
        
        try:
            response = await self._call_llm(prompt, temperature=0.0)
            
            import json
            expanded = json.loads(response)
            
            self.log_info(f"Generated {len(expanded)} query variations")
            return expanded
        
        except Exception as e:
            self.log_warning(f"Query expansion failed: {e}")
            return [query]


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_retrieval_agent(model_name: Optional[str] = None) -> RetrievalAgent:
    """
    Create and initialize a retrieval agent for scientific literature.
    
    Args:
        model_name: Optional LLM model name
    
    Returns:
        Initialized RetrievalAgent
    """
    return RetrievalAgent(model_name=model_name)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

async def example_usage():
    """Example showing retrieval agent in action for scientific articles."""
    from ..state import create_initial_state, add_subtask
    
    # Create agent
    agent = create_retrieval_agent()
    
    # Create state for scientific query
    state = create_initial_state(
        query="What are the key innovations in Transformer architectures for NLP?",
        search_type="hybrid"
    )
    
    # Create retrieval task
    task = add_subtask(
        state=state,
        task_type=TaskType.HYBRID_SEARCH,
        description="Search for Transformer architecture research",
        parameters={
            "query": "Transformer architecture innovations NLP",
            "limit": 5,
            "text_weight": 0.3
        },
        assigned_agent=AgentRole.RETRIEVAL,
        priority=10
    )
    
    print(f"Created task: {task['task_id']}")
    print(f"Task type: {task['task_type']}")
    
    # Execute task
    print("\nExecuting retrieval task...")
    result = await agent.execute(state, task)
    
    print(f"\nTask result:")
    print(f"  Success: {result['success']}")
    print(f"  Latency: {result['latency_ms']:.1f}ms")
    print(f"  Chunks retrieved: {len(result['data'].get('chunks', []))}")
    print(f"  Unique papers: {result['data'].get('unique_papers', 0)}")
    print(f"  Graph facts retrieved: {len(result['data'].get('graph_facts', []))}")
    print(f"  Avg score: {result['data'].get('avg_score', 0.0):.3f}")
    
    # Show stored results in state
    print(f"\nResults stored in state:")
    print(f"  Retrieved contexts: {len(state['retrieved_contexts'])}")
    
    # Show agent statistics
    stats = agent.get_retrieval_statistics()
    print(f"\nAgent statistics:")
    print(f"  Total executions: {stats['total_executions']}")
    print(f"  Success rate: {stats['success_rate']:.1f}%")
    print(f"  Total chunks retrieved: {stats['total_chunks_retrieved']}")
    print(f"  Total papers retrieved: {stats['total_papers_retrieved']}")
    print(f"  Total graph facts: {stats['total_graph_facts_retrieved']}")
    print(f"  Avg relevance: {stats['avg_relevance_score']:.3f}")
    print(f"  Domain: {stats['domain']}")


if __name__ == "__main__":
    import asyncio
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Note: This example requires database and graph connections.")
    print("Run from main application context for full functionality.\n")
    
    # Uncomment to run example:
    # asyncio.run(example_usage())
