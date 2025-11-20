"""
Knowledge Specialist Agent.

This agent is responsible for knowledge graph operations:
- Entity extraction and enrichment (scientific entities)
- Relationship discovery (academic relationships)
- Temporal reasoning (publication timelines, research evolution)
- Graph pattern matching
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..base_agent import BaseAgent
from ..state import (
    AgentState,
    AgentRole,
    SubTask,
    TaskType,
    KnowledgeResult
)

# Import tools and utilities
try:
    from ...agent.tools import (
        get_entity_relationships_tool,
        get_entity_timeline_tool,
        EntityRelationshipInput,
        EntityTimelineInput
    )
    from ...agent.graph_utils import graph_client
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from agent.tools import (
        get_entity_relationships_tool,
        get_entity_timeline_tool,
        EntityRelationshipInput,
        EntityTimelineInput
    )
    from agent.graph_utils import graph_client

logger = logging.getLogger(__name__)


# ============================================================================
# KNOWLEDGE SPECIALIST AGENT
# ============================================================================

class KnowledgeAgent(BaseAgent):
    """
    Specialist agent for knowledge graph operations in scientific literature.
    
    Responsibilities:
    - Extract entities from research articles and queries
    - Find relationships between scientific entities
    - Discover temporal sequences and research timelines
    - Enrich entities with KG data
    - Graph traversal and pattern matching
    
    Capabilities:
    - Scientific entity recognition and disambiguation
    - Academic relationship extraction and validation
    - Temporal reasoning and research evolution tracking
    - Multi-hop graph traversal
    - Entity attribute enrichment from scholarly data
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize knowledge agent for scientific literature.
        
        Args:
            model_name: Optional LLM model name
        """
        super().__init__(
            role=AgentRole.KNOWLEDGE,
            name="Knowledge Specialist",
            description="Expert in scientific knowledge graph operations, academic entity extraction, and research relationship discovery",
            model_name=model_name
        )
        
        # Configure tools
        self.tools = [
            get_entity_relationships_tool,
            get_entity_timeline_tool
        ]
        
        # Knowledge statistics
        self.total_entities_processed = 0
        self.total_relationships_found = 0
        self.total_temporal_facts = 0
        self.entity_types_discovered = set()
        
        self.log_info("Initialized with scientific entity extraction and academic relationship discovery capabilities")

    # ========================================================================
    # ABSTRACT METHOD IMPLEMENTATIONS
    # ========================================================================

    async def process_task(self, state: Dict[str, Any], task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a knowledge graph task by dispatching to the correct handler.
        """
        task_type_str = task.get("task_type", "").lower()
        params = task.get("parameters", {})

        self.log_info(f"Processing knowledge task of type: {task_type_str}")
        
        try:
            if "entity_extraction" in task_type_str or "knowledge_extraction" in task_type_str:
                return await self._extract_entities(state, params)
            
            elif "relationship_finding" in task_type_str:
                return await self._find_relationships(params)
            
            elif "temporal_reasoning" in task_type_str:
                return await self._temporal_reasoning(params)
            
            else:
                self.log_warning(
                    f"Unknown task type '{task_type_str}', defaulting to entity extraction."
                )
                return await self._extract_entities(state, params)
                
        except Exception as e:
            self.log_error(f"An unexpected error occurred while processing task '{task_type_str}': {e}")
            raise

    def can_handle_task(self, task: Dict[str, Any]) -> bool:
        """Check if this agent can handle the task."""
        task_type = task.get("task_type", "")
        agent_role = task.get("agent_role", "")
        
        valid_types = [
            "knowledge_extraction", "entity_extraction", "relationship_finding", 
            "temporal_reasoning", "KNOWLEDGE_EXTRACTION", "ENTITY_EXTRACTION",
            "RELATIONSHIP_FINDING", "TEMPORAL_REASONING"
        ]
        
        valid_roles = ["knowledge", "KNOWLEDGE", AgentRole.KNOWLEDGE.value]
        
        return task_type in valid_types or agent_role in valid_roles

    def get_system_prompt(self) -> str:
        """
        Get system prompt for LLM interactions.
        
        Returns:
            System prompt string
        """
        return """You are a Knowledge Specialist Agent in a multi-agent system focused on scientific literature analysis.

Your responsibilities:
1. Extract scientific entities (authors, papers, methods, datasets, venues) from research articles
2. Find relationships between academic entities in the knowledge graph
3. Discover temporal patterns and research evolution timelines
4. Enrich entities with additional scholarly context from the graph
5. Disambiguate entities when multiple candidates exist (e.g., author name collisions)

Your expertise includes:
- Scientific Named Entity Recognition (NER)
- Academic relationship extraction and classification
- Research timeline construction and evolution tracking
- Graph pattern matching and traversal for citation networks
- Entity linking and disambiguation in scholarly contexts

Entity types you recognize:
- AUTHOR: Researchers and paper authors (e.g., "Geoffrey Hinton", "Yoshua Bengio")
- PAPER: Research articles, publications (e.g., "Attention Is All You Need")
- METHOD: Techniques, algorithms, models (e.g., "Transformer", "BERT", "Backpropagation")
- DATASET: Training/evaluation datasets (e.g., "ImageNet", "GLUE Benchmark")
- VENUE: Conferences, journals (e.g., "NeurIPS", "Nature")
- INSTITUTION: Universities, research labs (e.g., "MIT CSAIL", "DeepMind")
- CONCEPT: Research topics, fields (e.g., "Transfer Learning", "Attention Mechanism")
- METRIC: Evaluation metrics (e.g., "F1-Score", "Perplexity")

Relationship types you identify:
- AUTHORED: Author → Paper
- CITES: Paper → Paper (citation relationship)
- PUBLISHED_IN: Paper → Venue
- AFFILIATED_WITH: Author → Institution
- USES_METHOD: Paper → Method
- EVALUATED_ON: Paper → Dataset
- PROPOSES: Paper → Method/Concept
- EXTENDS: Method → Method (methodological evolution)
- RELATED_TO: Concept → Concept
- MEASURES: Metric → Evaluation task

Always prioritize:
- Accuracy over completeness (avoid false positives in citation/authorship)
- Entity disambiguation (resolve author homonyms, paper title variations)
- Temporal consistency (maintain correct publication chronology)
- Relationship validation (verify citations and authorship exist)
- Academic context preservation (maintain research domain context)"""
    
    # ========================================================================
    # KNOWLEDGE GRAPH METHODS
    # ========================================================================
    
    async def _extract_entities(
        self,
        state: AgentState,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract and enrich scientific entities from query and context.
        
        Args:
            state: Current state
            parameters: Extraction parameters
        
        Returns:
            Extracted entities with enrichment
        """
        entities = parameters.get("entities", {})
        query = state.get("query", "")
        
        self.log_info(f"Extracting scientific entities from query: '{query[:50]}...'")
        
        # Get entities provided in parameters (from query analysis)
        extracted_entities = []
        
        for entity_type, entity_list in entities.items():
            for entity_name in entity_list:
                # Enrich each entity with KG data
                enriched = await self._enrich_entity(entity_name, entity_type)
                extracted_entities.append(enriched)
                self.entity_types_discovered.add(entity_type)
        
        # Also extract entities from retrieved research contexts if available
        contexts = state.get("retrieved_contexts", [])
        if contexts:
            context_entities = await self._extract_from_contexts(contexts)
            extracted_entities.extend(context_entities)
        
        result = {
            "entities": extracted_entities,
            "entity_count": len(extracted_entities),
            "entity_types": list(set(e["type"] for e in extracted_entities)),
            "enrichment_sources": ["knowledge_graph", "query_analysis", "scholarly_data"]
        }
        
        self.log_info(f"Extracted {len(extracted_entities)} scientific entities")
        
        return result
    
    async def _find_relationships(
        self,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Find academic relationships for an entity in the knowledge graph.
        
        Args:
            parameters: Relationship search parameters
        
        Returns:
            Found relationships
        """
        entity_name = parameters.get("entity_name", "")
        depth = parameters.get("depth", 2)
        
        self.log_info(f"Finding academic relationships for: '{entity_name}' (depth={depth})")
        
        try:
            # Call relationship tool
            relationship_input = EntityRelationshipInput(
                entity_name=entity_name,
                depth=depth
            )
            
            relationship_data = await get_entity_relationships_tool(
                entity_name=relationship_input.entity_name,
                depth=relationship_input.depth
            )

            if relationship_data is None:
                raise ValueError(f"Tool call failed for entity '{entity_name}' relationships")
            
            # Parse relationships
            related_entities = relationship_data.get("related_entities", [])
            relationships = relationship_data.get("relationships", [])
            related_facts = relationship_data.get("related_facts", [])
            
            # Structure the results
            structured_relationships = self._structure_relationships(
                central_entity=entity_name,
                relationships=relationships,
                facts=related_facts
            )
            
            result = {
                "central_entity": entity_name,
                "related_entities": related_entities,
                "relationships": structured_relationships,
                "relationship_count": len(structured_relationships),
                "traversal_depth": depth,
                "facts": related_facts[:10]
            }
            
            self.log_info(
                f"Found {len(related_entities)} related entities, "
                f"{len(structured_relationships)} relationships"
            )
            
            return result
        
        except Exception as e:
            self.log_error(f"Relationship finding failed: {e}")
            raise
    
    async def _temporal_reasoning(
        self,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform temporal reasoning for a research entity (e.g., author's publication timeline).
        
        Args:
            parameters: Temporal reasoning parameters
        
        Returns:
            Timeline and temporal facts
        """
        entity_name = parameters.get("entity_name", "")
        start_date = parameters.get("start_date")
        end_date = parameters.get("end_date")
        
        self.log_info(f"Building research timeline for: '{entity_name}'")
        
        try:
            # Call timeline tool
            timeline_input = EntityTimelineInput(
                entity_name=entity_name,
                start_date=start_date,
                end_date=end_date
            )
            
            timeline_data = await get_entity_timeline_tool(
                entity_name=timeline_input.entity_name,
                start_date=timeline_input.start_date,
                end_date=timeline_input.end_date
            )

            if timeline_data is None:
                 raise ValueError(f"Tool call failed for entity '{entity_name}' timeline")
            
            if not isinstance(timeline_data, list):
                self.log_warning(f"Timeline tool returned non-list type: {type(timeline_data)}")
                timeline_data = []
            
            # Sort timeline by date (publication chronology)
            sorted_timeline = sorted(
                timeline_data,
                key=lambda x: x.get("valid_at", ""),
                reverse=False
            )
            
            # Extract key research milestones
            key_events = self._extract_key_events(sorted_timeline)
            
            result = {
                "entity_name": entity_name,
                "timeline": sorted_timeline,
                "key_events": key_events,
                "event_count": len(sorted_timeline),
                "time_range": {
                    "start": sorted_timeline[0].get("valid_at") if sorted_timeline else None,
                    "end": sorted_timeline[-1].get("valid_at") if sorted_timeline else None
                },
                "timeline_type": "research_evolution"
            }
            
            self.log_info(f"Built research timeline with {len(sorted_timeline)} events")
            
            return result
        
        except Exception as e:
            self.log_error(f"Temporal reasoning failed: {e}")
            raise
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    async def _enrich_entity(
        self,
        entity_name: str,
        entity_type: str
    ) -> Dict[str, Any]:
        """
        Enrich a scientific entity with knowledge graph data.
        
        Args:
            entity_name: Name of the entity
            entity_type: Type of entity (AUTHOR, PAPER, METHOD, etc.)
        
        Returns:
            Enriched entity data
        """
        try:
            # Get relationships for this entity (shallow search)
            relationship_input = EntityRelationshipInput(
                entity_name=entity_name,
                depth=1
            )
            
            kg_data = await get_entity_relationships_tool(
                entity_name=relationship_input.entity_name,
                depth=relationship_input.depth
            )

            if kg_data is None:
                 raise ValueError(f"Tool call failed during entity enrichment for '{entity_name}'")
            
            return {
                "name": entity_name,
                "type": entity_type,
                "related_entities": kg_data.get("related_entities", [])[:5],
                "facts": kg_data.get("related_facts", [])[:3],
                "enriched": True,
                "source": "scientific_kg"
            }
        
        except Exception as e:
            self.log_warning(f"Failed to enrich entity '{entity_name}': {e}")
            
            return {
                "name": entity_name,
                "type": entity_type,
                "related_entities": [],
                "facts": [],
                "enriched": False,
                "source": "query_only"
            }
    
    async def _extract_from_contexts(
        self,
        contexts: List[Any]
    ) -> List[Dict[str, Any]]:
        """
        Extract scientific entities from retrieved research article contexts using LLM.
        
        Args:
            contexts: Retrieved context chunks from research articles
        
        Returns:
            List of extracted entities
        """
        # Combine contexts (limit to avoid token overflow)
        combined_text = ""
        for context in contexts[:3]:
            if hasattr(context, 'chunks'):
                for chunk in context['chunks'][:2]:
                    combined_text += chunk.get('content', '') + "\n\n"
        
        if not combined_text.strip():
            return []
        
        # Use LLM to extract entities
        prompt = f"""Extract scientific entities from the following research article text.

Text:
{combined_text[:2000]}

Return a JSON array of entities with format:
[
  {{"name": "Entity Name", "type": "AUTHOR|PAPER|METHOD|DATASET|VENUE|INSTITUTION|CONCEPT|METRIC"}},
  ...
]

Only include prominent entities mentioned multiple times or in important research contexts.
You MUST respond with ONLY the JSON array. If no entities are found, return [].
"""
        
        try:
            response = await self._call_llm(prompt, temperature=0.0)
            
            # Parse response
            import json
            response_str = str(response).strip()
            # Remove surrounding Markdown code fences if present (e.g., ```json ... ```).
            if response_str.startswith("```"):
                # Strip the opening fence line (could be ``` or ```json)
                first_newline = response_str.find("\n")
                if first_newline != -1:
                    response_body = response_str[first_newline+1:]
                else:
                    response_body = response_str[3:]
                # Remove trailing closing fence if present
                if response_body.endswith("```"):
                    response_body = response_body[:-3]
                response_str = response_body.strip()

            entities_data = json.loads(response_str)
            
            # Enrich extracted entities
            enriched = []
            for entity_data in entities_data[:5]:
                enriched_entity = await self._enrich_entity(
                    entity_data["name"],
                    entity_data["type"]
                )
                enriched.append(enriched_entity)
            
            self.log_info(f"Extracted {len(enriched)} scientific entities from research contexts")
            return enriched
        
        except Exception as e:
            self.log_warning(f"Context entity extraction failed: {e}")
            return []
    
    def _structure_relationships(
        self,
        central_entity: str,
        relationships: List[Any],
        facts: List[Any]
    ) -> List[Dict[str, Any]]:
        """
        Structure academic relationships into a consistent format.
        
        Args:
            central_entity: Central entity name
            relationships: Raw relationship data
            facts: Related facts
        
        Returns:
            Structured relationships
        """
        structured = []
        
        # Process relationships
        for rel in relationships:
            structured.append({
                "source": central_entity,
                "target": rel.get("target", ""),
                "type": rel.get("type", "RELATED_TO"),
                "confidence": rel.get("confidence", 1.0)
            })
        
        # Extract relationships from facts (common in scholarly contexts)
        for fact in facts[:10]:
            fact_text = fact.get("fact", "")
            
            # Enhanced relationship extraction for academic context
            fact_lower = fact_text.lower()
            if "authored" in fact_lower or "wrote" in fact_lower:
                structured.append({
                    "source": central_entity,
                    "target": "",
                    "type": "AUTHORED",
                    "confidence": 0.8
                })
            elif "cites" in fact_lower or "references" in fact_lower:
                structured.append({
                    "source": central_entity,
                    "target": "",
                    "type": "CITES",
                    "confidence": 0.8
                })
            elif "published in" in fact_lower or "appeared in" in fact_lower:
                structured.append({
                    "source": central_entity,
                    "target": "",
                    "type": "PUBLISHED_IN",
                    "confidence": 0.85
                })
            elif "proposes" in fact_lower or "introduces" in fact_lower:
                structured.append({
                    "source": central_entity,
                    "target": "",
                    "type": "PROPOSES",
                    "confidence": 0.8
                })
        
        return structured
    
    def _extract_key_events(
        self,
        timeline: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extract key research milestones from timeline.
        
        Args:
            timeline: Full timeline data
        
        Returns:
            Key events (e.g., influential publications, major discoveries)
        """
        key_events = []
        
        for event in timeline:
            fact = event.get("fact", "")
            valid_at = event.get("valid_at", "")
            
            # Determine event importance (research-focused heuristic)
            importance = self._assess_event_importance(fact)
            
            if importance > 0.5:
                key_events.append({
                    "date": valid_at,
                    "event": fact,
                    "importance": importance,
                    "category": self._categorize_research_event(fact)
                })
        
        return key_events
    
    def _assess_event_importance(self, fact: str) -> float:
        """
        Assess the importance of a research event.
        
        Args:
            fact: Fact text
        
        Returns:
            Importance score (0-1)
        """
        # Research-focused importance keywords
        important_keywords = [
            "published", "authored", "proposed", "introduced",
            "breakthrough", "seminal", "pioneered", "discovered",
            "award", "citation", "influential", "novel"
        ]
        
        fact_lower = fact.lower()
        matches = sum(1 for keyword in important_keywords if keyword in fact_lower)
        
        # Normalize to 0-1
        return min(matches / 3.0, 1.0)
    
    def _categorize_research_event(self, fact: str) -> str:
        """
        Categorize a research event.
        
        Args:
            fact: Fact text
        
        Returns:
            Event category
        """
        fact_lower = fact.lower()
        
        if any(kw in fact_lower for kw in ["published", "paper", "article"]):
            return "publication"
        elif any(kw in fact_lower for kw in ["award", "prize", "honor"]):
            return "recognition"
        elif any(kw in fact_lower for kw in ["proposed", "introduced", "developed"]):
            return "contribution"
        elif any(kw in fact_lower for kw in ["collaboration", "partnership"]):
            return "collaboration"
        else:
            return "other"
    
    def _update_statistics(self, result: Dict[str, Any]):
        """
        Update agent statistics with KG results.
        
        Args:
            result: KG operation results
        """
        if "entities" in result:
            self.total_entities_processed += len(result["entities"])
        
        if "relationships" in result:
            self.total_relationships_found += len(result["relationships"])
        
        if "timeline" in result:
            self.total_temporal_facts += len(result["timeline"])
    
    def _store_results_in_state(
        self,
        state: AgentState,
        result: Dict[str, Any],
        task: SubTask
    ):
        """
        Store knowledge graph results in state.
        
        Args:
            state: Current state
            result: KG results
            task: Original task
        """
        # Create structured KnowledgeResult
        knowledge_result = KnowledgeResult(
            entities=result.get("entities", []),
            relationships=result.get("relationships", []),
            temporal_facts=result.get("timeline", []),
            graph_paths=[]
        )
        
        # Add to state's knowledge_graph_data
        if "knowledge_graph_data" not in state:
             state["knowledge_graph_data"] = []
        state["knowledge_graph_data"].append(knowledge_result)
        
        self.log_debug(f"Stored scientific KG results in state")
    
    def get_knowledge_statistics(self) -> Dict[str, Any]:
        """
        Get knowledge-specific statistics.
        
        Returns:
            Dict with knowledge stats
        """
        base_stats = self.get_performance_stats()
        
        return {
            **base_stats,
            "total_entities_processed": self.total_entities_processed,
            "total_relationships_found": self.total_relationships_found,
            "total_temporal_facts": self.total_temporal_facts,
            "entity_types_discovered": list(self.entity_types_discovered),
            "entities_per_execution": (
                self.total_entities_processed / self.total_executions
                if self.total_executions > 0
                else 0.0
            ),
            "domain": "scientific_literature"
        }
    
    # ========================================================================
    # ADVANCED KNOWLEDGE METHODS
    # ========================================================================
    
    async def disambiguate_entity(
        self,
        entity_name: str,
        context: str
    ) -> Dict[str, Any]:
        """
        Disambiguate scientific entity when multiple candidates exist (e.g., author homonyms).
        
        Args:
            entity_name: Ambiguous entity name
            context: Context to help disambiguation
        
        Returns:
            Disambiguated entity
        """
        self.log_info(f"Disambiguating scientific entity: '{entity_name}'")
        
        prompt = f"""Given the entity name "{entity_name}" in the following research context:

Context: {context[:500]}

Determine which specific entity is being referred to. Consider:
1. Research domain (machine learning, biology, physics, etc.)
2. Time period and publication dates
3. Related entities mentioned (co-authors, institutions, papers)
4. Disambiguation cues (middle initials, affiliations)

Return the most likely entity with brief explanation."""
        
        try:
            response = await self._call_llm(prompt, temperature=0.0)
            
            return {
                "original_name": entity_name,
                "disambiguated_name": response,
                "confidence": 0.8,
                "method": "llm_context",
                "disambiguation_type": "scientific_entity"
            }
        
        except Exception as e:
            self.log_warning(f"Entity disambiguation failed: {e}")
            return {
                "original_name": entity_name,
                "disambiguated_name": entity_name,
                "confidence": 0.5,
                "method": "fallback",
                "disambiguation_type": "scientific_entity"
            }
    
    async def find_citation_paths(
        self,
        source_paper: str,
        target_paper: str,
        max_hops: int = 3
    ) -> List[List[str]]:
        """
        Find citation paths between two papers in the graph.
        
        Args:
            source_paper: Source paper
            target_paper: Target paper
            max_hops: Maximum path length
        
        Returns:
            List of citation paths (each path is a list of papers)
        """
        self.log_info(f"Finding citation paths: {source_paper} → {target_paper}")
        
        # TODO: Implement citation path finding
        # This would require graph traversal in Graphiti
        
        return []


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_knowledge_agent(model_name: Optional[str] = None) -> KnowledgeAgent:
    """
    Create and initialize a knowledge agent for scientific literature.
    
    Args:
        model_name: Optional LLM model name
    
    Returns:
        Initialized KnowledgeAgent
    """
    return KnowledgeAgent(model_name=model_name)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

async def example_usage():
    """Example showing knowledge agent in action for scientific articles."""
    from ..state import create_initial_state, add_subtask
    
    # Create agent
    agent = create_knowledge_agent()
    
    # Create state for scientific query
    state = create_initial_state(
        query="What are the key contributions of the Transformer architecture in NLP?",
        search_type="hybrid"
    )
    
    # Create entity extraction task
    extraction_task = add_subtask(
        state=state,
        task_type=TaskType.ENTITY_EXTRACTION,
        description="Extract scientific entities from query",
        parameters={
            "entities": {
                "methods": ["Transformer"],
                "concepts": ["NLP", "attention mechanism"],
                "papers": ["Attention Is All You Need"]
            }
        },
        assigned_agent=AgentRole.KNOWLEDGE,
        priority=8
    )
    
    print(f"Created extraction task: {extraction_task['task_id']}")
    
    # Execute extraction task
    print("\nExecuting scientific entity extraction...")
    result1 = await agent.execute(state, extraction_task)
    
    print(f"\nExtraction result:")
    print(f"  Success: {result1['success']}")
    print(f"  Entities found: {len(result1['data'].get('entities', []))}")
    
    # Create relationship finding task
    relationship_task = add_subtask(
        state=state,
        task_type=TaskType.RELATIONSHIP_FINDING,
        description="Find relationships for Transformer architecture",
        parameters={
            "entity_name": "Transformer",
            "depth": 2
        },
        assigned_agent=AgentRole.KNOWLEDGE,
        priority=7
    )
    
    print(f"\nCreated relationship task: {relationship_task['task_id']}")
    
    # Execute relationship task
    print("\nExecuting academic relationship finding...")
    result2 = await agent.execute(state, relationship_task)
    
    print(f"\nRelationship result:")
    print(f"  Success: {result2['success']}")
    print(f"  Relationships found: {result2['data'].get('relationship_count', 0)}")
    
    # Show agent statistics
    stats = agent.get_knowledge_statistics()
    print(f"\nAgent statistics:")
    print(f"  Total executions: {stats['total_executions']}")
    print(f"  Entities processed: {stats['total_entities_processed']}")
    print(f"  Relationships found: {stats['total_relationships_found']}")
    print(f"  Domain: {stats['domain']}")


if __name__ == "__main__":
    import asyncio
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Note: This example requires graph database connection.")
    print("Run from main application context for full functionality.\n")
    
    # Uncomment to run example:
    # asyncio.run(example_usage())
