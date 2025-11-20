"""
Synthesis Specialist Agent.

This agent is responsible for synthesizing information:
- Combining results from retrieval and knowledge agents
- Generating comprehensive answers from scientific literature
- Handling multi-source information fusion
- Providing reasoning and explanations with proper citations
"""

import logging
from typing import Dict, Any, List, Optional
from collections import defaultdict
import json
import re

from ..base_agent import BaseAgent
from ..state import (
    AgentState,
    AgentRole,
    SubTask,
    TaskType,
    SynthesisResult,
    QueryIntent
)

logger = logging.getLogger(__name__)


# ============================================================================
# SYNTHESIS SPECIALIST AGENT
# ============================================================================

class SynthesisAgent(BaseAgent):
    """
    Specialist agent for information synthesis and answer generation from scientific literature.
    
    Responsibilities:
    - Combine results from multiple research sources
    - Generate comprehensive answers with academic rigor
    - Handle different query types (factual, comparison, temporal, methodological)
    - Provide reasoning and explanations with proper citations
    - Maintain source attribution (authors, papers, venues)
    - Resolve conflicting information across papers
    
    Capabilities:
    - Multi-source scientific literature fusion
    - Context-aware answer generation for research queries
    - Comparative analysis of methods and approaches
    - Timeline synthesis for research evolution
    - Confidence estimation based on paper quality
    - Citation tracking and attribution
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize synthesis agent for scientific literature.
        
        Args:
            model_name: Optional LLM model name
        """
        super().__init__(
            role=AgentRole.SYNTHESIS,
            name="Synthesis Specialist",
            description="Expert in combining information from multiple research sources and generating comprehensive academic answers",
            model_name=model_name
        )
        
        # Synthesis statistics
        self.total_sources_combined = 0
        self.total_papers_cited = 0
        self.total_answers_generated = 0
        self.avg_answer_length = 0
        self.avg_confidence_scores = []
        
        self.log_info("Initialized with multi-source synthesis capabilities for scientific literature")

    # ========================================================================
    # ABSTRACT METHOD IMPLEMENTATIONS
    # ========================================================================
    
    async def process_task(self, state: Dict[str, Any], task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a synthesis task by calling the main answer synthesis method.
        """
        task_type = task.get("task_type", "").lower()
        params = task.get("parameters", {})
        
        self.log_info(f"Processing {task_type} task for scientific literature")
        
        if "synthesis" not in task_type:
            self.log_warning(f"Unexpected task type '{task_type}', proceeding with synthesis.")

        try:
            return await self._synthesize_answer(state, params)
        except Exception as e:
            self.log_error(f"An unexpected error occurred during answer synthesis: {e}", exc_info=True)
            raise

    def can_handle_task(self, task: Dict[str, Any]) -> bool:
        """Check if this agent can handle the task."""
        task_type = task.get("task_type", "")
        agent_role = task.get("agent_role", "")
        
        valid_types = ["synthesis", "SYNTHESIS"]
        valid_roles = ["synthesis", "SYNTHESIS", AgentRole.SYNTHESIS.value]
        
        return task_type in valid_types or agent_role in valid_roles

    def get_system_prompt(self) -> str:
        """Get system prompt for LLM interactions."""
        return """You are a Synthesis Specialist Agent in a multi-agent system focused on scientific literature analysis.

    Your responsibilities:
    1. Combine information from multiple research sources (retrieval, knowledge graph)
    2. Generate comprehensive, well-structured answers with academic rigor
    3. Handle different research query types appropriately
    4. Maintain proper source attribution and citations (author names, paper titles, venues)
    5. Provide reasoning for your conclusions based on evidence from papers
    6. Resolve conflicting information across different research articles

    # ============================================================================
    # ⚠️ CRITICAL: HANDLING TWO TYPES OF INFORMATION SOURCES
    # ============================================================================

    You receive TWO distinct types of information:

    ## 1. RESEARCH PAPER EXCERPTS (Text Chunks from Vector/Keyword Search)
    - Direct excerpts from research papers
    - Semantic context and detailed explanations
    - Technical descriptions and methodologies
    - **USE FOR**: Content (what is said), definitions, explanations, methods

    ## 2. KNOWLEDGE GRAPH FACTS (Structured Relationships from Graph Database)
    - Structured relationships: citations, authorship, hierarchies
    - Factual statements: "Paper A cites Paper B", "Author X wrote Paper Y"
    - Metadata: publication dates, venues, citation counts
    - **USE FOR**: Structure (who said it), relationships, provenance, verification

    ## ⚠️ CRITICAL RULE: NEVER MIX ATTRIBUTIONS

    **CORRECT Attribution**:
    ✅ "GraphRAG combines knowledge graphs with retrieval [Paper: Hybrid Multi-Agent GraphRAG]"
    → Content from PAPER EXCERPT, attributed to PAPER

    ✅ "This paper cites Neo4j [Source: Knowledge Graph]"
    → Relationship from GRAPH FACT, attributed to GRAPH

    ✅ "GraphRAG enables multi-hop reasoning [Paper: Hybrid Multi-Agent GraphRAG][Source: Knowledge Graph]"
    → Content from BOTH, cite BOTH

    **WRONG Attribution**:
    ❌ "GraphRAG combines knowledge graphs [Source: Knowledge Graph]"
    → If this CONTENT came from a PAPER EXCERPT, you MUST cite the PAPER!

    ❌ "Author X proposes multi-hop reasoning [Paper: Some Paper]"
    → If "Author X wrote the paper" came from GRAPH FACT, cite that separately!

    ❌ "Neo4j is used for graph traversal [Paper: Hybrid Multi-Agent GraphRAG]"
    → If the paper MENTIONS Neo4j but the GRAPH FACT says "cites Neo4j", distinguish!

    # ============================================================================
    # QUERY TYPES AND HOW TO HANDLE THEM
    # ============================================================================

    **FACTUAL queries** ("What is X?", "Define X"):
    - Provide clear, concise definition from authoritative papers
    - Include key concepts and context
    - Cite seminal papers and recent reviews
    - Mention if definition has evolved over time

    **COMPARISON queries** ("Compare X and Y", "X vs Y"):
    - Structure as point-by-point comparison
    - Highlight methodological similarities and differences
    - Compare performance metrics when available
    - Use tables or structured format when appropriate
    - Cite papers that directly compare or use both approaches

    **TEMPORAL queries** ("When did X happen?", "Evolution of X"):
    - Present chronological timeline of research milestones
    - Include publication dates and venues
    - Show how methods/concepts evolved over time
    - Cite landmark papers that introduced key innovations

    **METHODOLOGICAL queries** ("How does X work?", "Explain X"):
    - Explain the technical approach step-by-step
    - Reference original papers that proposed the method
    - Include mathematical formulations if present in sources
    - Mention variations and improvements from later papers

    **RELATIONAL queries** ("How is X related to Y?"):
    - Explain connections and relationships based on citations
    - Use knowledge graph data (citations, authorship)
    - Show multi-hop relationships when relevant (e.g., paper A cites B which cites C)

    # ============================================================================
    # ANSWER FORMAT
    # ============================================================================

    1. **Direct answer** (1-2 sentences)
    2. **Detailed explanation** with supporting evidence from papers
    3. **Additional context** or related research
    4. **Source citations** with proper attribution
    5. **Confidence note** if sources conflict or data is limited

    # ============================================================================
    # CITATION FORMAT
    # ============================================================================

    - Papers: `[Paper: Full Title]` or `[Author et al., Venue Year]`
    - Graph facts: `[Source: Knowledge Graph]`
    - Both: `[Paper: Title][Source: Knowledge Graph]`

    # ============================================================================
    # ALWAYS
    # ============================================================================

    - Start with a direct answer to the question
    - Provide supporting details and context from papers
    - Be honest about uncertainty or conflicting findings across papers
    - Structure answers clearly with sections when needed
    - Include confidence assessment based on source quality and agreement
    - Mention citation counts or paper influence when relevant
    - Distinguish between established consensus and recent/controversial findings
    """
        
    # ========================================================================
    # SYNTHESIS METHODS
    # ========================================================================
    
    async def _synthesize_answer(
        self,
        state: AgentState,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Synthesize answer from retrieved contexts and knowledge graph data.
        
        Args:
            state: Current state
            params: Task parameters
        
        Returns:
            Synthesis result
        """
        query = params.get("query", state.get("query", ""))
        intent = params.get("intent", "factual")
        synthesis_type = params.get("synthesis_type", "comprehensive")
        
        # Convert intent to string if it's an Enum
        if hasattr(intent, 'value'):
            intent = intent.value
        elif hasattr(intent, 'name'):
            intent = intent.name.lower()
        
        # Ensure intent is a string
        intent = str(intent).lower()
        
        self.log_info(f"Synthesizing answer for: '{query[:50]}...' with intent: {intent}")
        
        try:
            # Get retrieval contexts and knowledge graph data
            retrieval_contexts = state.get("retrieved_contexts", [])
            knowledge_graph_data = state.get("knowledge_graph_data", [])
            
            # Prepare context for synthesis
            context = self._prepare_context(
                retrieval_contexts,
                knowledge_graph_data,
                query
            )
            
            # ✅ FIX: Call attribution to link KG facts to papers
            context = self._attribute_kg_facts_to_papers(context)
            
            self.log_info(
                f"Context prepared: {context['chunk_count']} chunks from "
                f"{context['paper_count']} papers, {context['fact_count']} facts, "
                f"{context['entity_count']} entities"
            )
            
            # Choose synthesis handler based on intent
            if intent in ['comparison', 'compare']:
                handler = self._generate_comparison_answer
            elif intent in ['temporal', 'timeline']:
                handler = self._generate_temporal_answer
            elif intent in ['survey', 'aggregation', 'overview']:
                handler = self._generate_survey_answer
            elif intent in ['methodological', 'procedural', 'how_to']:
                handler = self._generate_methodological_answer
            else:
                handler = self._generate_standard_answer
            
            self.log_info(f"Using synthesis handler: {handler.__name__}")
            
            # Generate answer
            result = await handler(query, context, state)
            
            # Log result
            answer_len = len(result.get('answer', ''))
            papers_cited = result.get('papers_cited', 0)
            confidence = result.get('confidence', 0.0)
            
            self.log_info(
                f"Answer generated: {answer_len} chars, {papers_cited} papers cited, "
                f"confidence={confidence:.2f}"
            )
            
            return result
        
        except Exception as e:
            self.log_error(f"An unexpected error occurred during answer synthesis: {e}", exc_info=True)
            raise
    
    # ========================================================================
    # CONTEXT PREPARATION
    # ========================================================================
    
    def _prepare_context(
        self,
        retrieval_contexts: List[Any],
        knowledge_graph_data: List[Any],
        query: str
    ) -> Dict[str, Any]:
        """
        Prepare unified context from retrieval and knowledge graph data.
        
        Args:
            retrieval_contexts: List of retrieval results
            knowledge_graph_data: List of knowledge graph results
            query: Original query
        
        Returns:
            Prepared context dict with all relevant data
        """
        # Collect chunks
        chunks = []
        graph_facts = []
        papers = set()
        
        for ctx in retrieval_contexts:
            if isinstance(ctx, dict):
                # Get chunks
                ctx_chunks = ctx.get('chunks', [])
                chunks.extend(ctx_chunks)
                
                # Track unique papers
                for chunk in ctx_chunks:
                    paper_title = chunk.get('document_title', '')
                    if paper_title:
                        papers.add(paper_title)
                
                # Get graph facts
                ctx_facts = ctx.get('graph_facts', [])
                graph_facts.extend(ctx_facts)
        
        # Collect entities and relationships
        entities = []
        relationships = []
        timeline = []
        
        for kg_data in knowledge_graph_data:
            if isinstance(kg_data, dict):
                entities.extend(kg_data.get('entities', []))
                relationships.extend(kg_data.get('relationships', []))
                timeline.extend(kg_data.get('timeline', []))
        
        # Sort chunks by score
        chunks.sort(key=lambda x: x.get('score', 0.0), reverse=True)
        
        # Prepare context summary
        context = {
            'query': query,
            'chunks': chunks,
            'graph_facts': graph_facts,
            'entities': entities,
            'relationships': relationships,
            'timeline': timeline,
            'papers': list(papers),
            
            # Counts
            'chunk_count': len(chunks),
            'fact_count': len(graph_facts),
            'entity_count': len(entities),
            'relationship_count': len(relationships),
            'paper_count': len(papers),
            
            # For synthesis
            'has_graph_data': len(graph_facts) > 0 or len(entities) > 0,
            'has_temporal_data': len(timeline) > 0
        }
        
        return context

    def _format_context_for_llm(self, context: Dict[str, Any]) -> str:
        """Format context with CLEAR SEPARATION between text chunks and graph facts."""
        formatted = []
        
        # ========================================================================
        # SECTION 1: RESEARCH PAPER EXCERPTS (TEXT CHUNKS)
        # ========================================================================
        if context.get("chunks"):
            formatted.append("\n" + "="*80)
            formatted.append("📄 RESEARCH PAPER EXCERPTS (Text Chunks from Vector/Keyword Search)")
            formatted.append("="*80)
            formatted.append("\n⚠️ **SOURCE TYPE**: Direct excerpts from research papers")
            formatted.append("📌 **USE FOR**: Content (what is said), definitions, explanations, methods")
            formatted.append("📌 **CITE AS**: [Paper: Full Title Here]\n")
            
            paper_map = defaultdict(list)
            for chunk in context["chunks"]:
                paper_title = chunk.get("document_title", "Unknown")
                paper_map[paper_title].append(chunk)
            
            for i, (paper_title, chunks) in enumerate(paper_map.items(), 1):
                first_chunk = chunks[0]
                authors = first_chunk.get("authors", [])
                venue = first_chunk.get("venue", "Unknown Venue")
                pub_date = first_chunk.get("publication_date", "")
                
                author_str = ", ".join(authors[:3]) if authors else ""
                if len(authors) > 3:
                    author_str += " et al."
                
                formatted.append(f"\n{'─'*80}")
                formatted.append(f"📄 Paper {i}: {paper_title}")
                formatted.append(f"   Authors: {author_str}")
                formatted.append(f"   Venue: {venue} ({pub_date})")
                formatted.append(f"   ✅ CITE AS: [Paper: {paper_title}]")
                formatted.append(f"{'─'*80}\n")
                
                for j, chunk in enumerate(chunks, 1):
                    content = chunk.get("content", "")
                    score = chunk.get("score", 0.0)
                    formatted.append(f"   Excerpt {j} (relevance: {score:.2f}):")
                    formatted.append(f"   {content.strip()}\n")
            
            formatted.append("\n" + "="*80 + "\n")
        
        # ========================================================================
        # SECTION 2: KNOWLEDGE GRAPH FACTS (STRUCTURED RELATIONSHIPS)
        # ========================================================================
        
        # ✅ FIX: Always show KG facts if they exist.
        if context.get("graph_facts"):
            formatted.append("\n" + "="*80)
            formatted.append("🔗 KNOWLEDGE GRAPH FACTS (Structured Relationships from Graph DB)")
            formatted.append("="*80)
            formatted.append("\n⚠️ **SOURCE TYPE**: Structured metadata and relationships (NOT paper content)")
            formatted.append("📌 **USE FOR**: Structure (who said it), citations, authorship, relationships")
            formatted.append("📌 **CITE AS**: [Source: Knowledge Graph]\n")
            formatted.append("⚠️ **CRITICAL**: If a PAPER EXCERPT already covers the same information,")
            formatted.append("                ALWAYS cite the PAPER first: [Paper: X][Source: Knowledge Graph]\n")
            
            for i, fact in enumerate(context["graph_facts"][:15], 1):
                fact_text = fact.get("fact", str(fact))
                
                # ✅ NEW: Show attribution note if available
                attributed_paper = fact.get("attributed_paper")
                attribution_conf = fact.get("attribution_confidence", 0)
                
                if attributed_paper and attribution_conf > 0.6: # Confidence threshold
                    formatted.append(
                        f"{i}. {fact_text} "
                        f"[NOTE: This fact appears to be supported by '{attributed_paper}']"
                    )
                else:
                    formatted.append(f"{i}. {fact_text}")

            formatted.append("\n" + "="*80 + "\n")
        
        # ========================================================================
        # SECTION 3: ENTITIES & RELATIONSHIPS (OPTIONAL)
        # ========================================================================
        if context.get("entities"):
            formatted.append("\n📍 **Entities mentioned**: " + ", ".join([
                e.get("name", str(e)) for e in context["entities"][:10]
            ]))
        
        if context.get("relationships"):
            formatted.append("\n🔗 **Key relationships**: " + str(len(context["relationships"])) + " found")
        
        return "\n".join(formatted)
    
    def _attribute_kg_facts_to_papers(
        self,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Match KG facts to papers when they're semantically similar.
        
        If a KG fact is essentially saying the same thing as a paper excerpt,
        annotate the fact with the paper source.
        """
        from difflib import SequenceMatcher
        
        chunks = context.get('chunks', [])
        facts = context.get('graph_facts', [])
        
        # Build content map
        paper_content_map = {}
        for chunk in chunks:
            paper_title = chunk.get('document_title')
            content = chunk.get('content', '').lower()
            if paper_title:
                if paper_title not in paper_content_map:
                    paper_content_map[paper_title] = []
                paper_content_map[paper_title].append(content)
        
        # Annotate facts with paper sources
        annotated_facts = []
        for fact in facts:
            fact_text = fact.get('fact', '').lower()
            best_paper = None
            best_score = 0.0
            
            for paper_title, contents in paper_content_map.items():
                for content in contents:
                    score = SequenceMatcher(None, fact_text, content).ratio()
                    if score > best_score:
                        best_score = score
                        best_paper = paper_title
            
            # If fact is >60% similar to a paper, attribute it
            if best_score > 0.6 and best_paper:
                fact['attributed_paper'] = best_paper
                fact['attribution_confidence'] = best_score
            
            annotated_facts.append(fact)
        
        context['graph_facts'] = annotated_facts
        return context
    
    def _is_claim_conservative(self, claim: str) -> bool:
        """
        Check if a claim is conservative (well-supported).
        
        Filters out speculative or extrapolated claims.
        """
        # Speculative language patterns
        speculative_patterns = [
            r'\bmight\b', r'\bmay\b', r'\bcould\b', r'\bpossibly\b',
            r'\bprobably\b', r'\blikely\b', r'\bsuggests?\b',
            r'\bimplies?\b', r'\bindicates?\b', r'\bseems?\b',
            r'\bappears?\b', r'\btends? to\b'
        ]
        
        import re
        for pattern in speculative_patterns:
            if re.search(pattern, claim, re.IGNORECASE):
                return False
        
        return True
    
    # ========================================================================
    # ANSWER GENERATION BY TYPE
    # ========================================================================

    async def _generate_standard_answer(
        self,
        query: str,
        context: Dict[str, Any],
        state: 'AgentState'
    ) -> Dict[str, Any]:
        """
        Generate factual answer based strictly on scientific literature.
        
        IMPROVED: Enforces strict separation between paper content and graph facts.
        """
        
        # Prepare context
        formatted_context = self._format_context_for_llm(context)
        
        paper_count = context.get("paper_count", 0)
        fact_count = context.get("fact_count", 0)
        
        # Build citation guidance
        if paper_count > 0 and fact_count > 0:
            citation_guidance = f"""
    # ============================================================================
    # 🎯 CRITICAL CITATION GUIDANCE
    # ============================================================================

    You have:
    - {paper_count} research papers (TEXT EXCERPTS)
    - {fact_count} knowledge graph facts (STRUCTURED RELATIONSHIPS)

    ## PRIORITY RULES:

    1️⃣ **If a claim comes from a PAPER EXCERPT**:
    ✅ Cite: `[Paper: Full Title]`
    
    2️⃣ **If a claim comes *ONLY* from a KNOWLEDGE GRAPH FACT**:
    ✅ Cite: `[Source: Knowledge Graph]`

    3️⃣ **If BOTH sources support a claim** (O KG Fact terá um `[NOTE: ...]`):
    ✅ **You MUST cite BOTH**: `[Paper: Full Title][Source: Knowledge Graph]`
    
    4️⃣ **NEVER mix attributions** (Exemplo permanece o mesmo)
    # ... (Resto do guidance permanece o mesmo)
    """
        elif paper_count > 0:
            citation_guidance = f"""
    # CITATION RULE:
    You have {paper_count} research papers.
    Cite as: `[Paper: Full Title]`
    Example: `[Paper: Attention Is All You Need]`
    """
        else:
            citation_guidance = """
    # CITATION RULE:
    You have knowledge graph facts only.
    Cite as: `[Source: Knowledge Graph]`
    """
        
        # Build prompt
        prompt = f"""You are a scientific research assistant answering a query based exclusively on the provided research literature.

    Do not use any external knowledge.

    **Research Query**: {query}

    **Available Research Literature**:
    {formatted_context}

    {citation_guidance}

    # ============================================================================
    # INSTRUCTIONS
    # ============================================================================

    1. Carefully analyze ALL sources (Research paper excerpts AND knowledge graph facts).
    2. Generate a comprehensive answer that directly responds to the query.
    3. ⚠️ **CRITICAL**: Base your answer ONLY on the provided literature.
    4. DO NOT include any external knowledge.
    5. For EVERY claim you make, add a citation immediately after it.
    
    6. ⚠️ **CITATION PRIORITY**: You MUST prioritize citing paper excerpts. You should ONLY cite [Source: Knowledge Graph] for a claim IF AND ONLY IF no paper excerpt supports that claim.
    
    7. ⚠️ **HANDLE OVERLAP:** If a KG Fact has a `[NOTE: ... This fact appears to be supported by 'Paper X']`, this means *both* sources support it. You **MUST** cite `[Paper: Paper X][Source: Knowledge Graph]`.
    
    8. If multiple papers agree on a claim, cite all: `[Paper: A][Paper: B]`.
    9. If papers conflict, acknowledge: "While [Paper: X] suggests Y, [Paper: Z] found W..."
    10. If literature is insufficient, state this clearly.
    11. ⚠️ **CONSERVATIVE GENERATION**: Only make claims that are DIRECTLY stated in the papers or facts.
    12. ⚠️ **AVOID SPECULATION**: Do not infer, extrapolate, or speculate beyond what sources explicitly say.

    **Answer**:
    """
        
        try:
            answer = await self._call_llm(prompt, temperature=0.0)
            
            sources_used = self._extract_sources(context)
            papers_cited = len(self._extract_unique_papers(context))
            confidence = self._assess_confidence(answer, context, state)
            reasoning = self._generate_reasoning(query, context, answer)
            
            result = {
                "answer": answer,
                "confidence": confidence,
                "papers_cited": papers_cited,
                "sources_used": sources_used,
                "reasoning": reasoning
            }
            
            self._update_statistics(result)
            return result
            
        except Exception as e:
            self.log.error(f"Standard answer generation failed: {e}", exc_info=True)
            return {
                "answer": f"Unable to generate answer: {str(e)}",
                "confidence": 0.0,
                "papers_cited": 0,
                "sources_used": [],
                "reasoning": "Answer generation failed"
            }

    async def generate_comparison_answer(
        self,
        query: str,
        context: Dict[str, Any],
        state: AgentState
    ) -> Dict[str, Any]:
        """
        Generate comparison answer for research methods or approaches.
        
        IMPROVED: Stricter prompt to prevent hallucination.
        """
        formatted_context = self._format_context_for_llm(context)
        paper_count = context.get("paper_count", 0)
        fact_count = context.get("fact_count", 0)

        self.log_info(f"Generating comparison answer for query: {query[:80]}...")
        
        # Build citation guidance
        citation_note = f"""
    CITATION INSTRUCTIONS:
    - You have {paper_count} research papers and {fact_count} KG facts.
    - **PRIORITY 1:** If info is in a paper, cite `[Paper: Full Title]`.
    - **PRIORITY 2:** If info is *only* in a KG fact, cite `[Source: Knowledge Graph]`.
    - **PRIORITY 3 (Overlap):** If a KG fact has a `[NOTE: ... supported by 'Paper X']`, you *MUST* cite both: `[Paper: Paper X][Source: Knowledge Graph]`.
    """
        
        # ✅ FIX: New, brutally strict prompt that leverages BOTH sources.
        prompt = f"""You are a research analyst. Your task is to answer the following query *ONLY* using the provided paper excerpts AND knowledge graph facts.

    **Research Question:** {query}

    **Available Research Literature:**
    {formatted_context}

    {citation_note}

    **CRITICAL INSTRUCTIONS - READ CAREFULLY:**
    1.  **DO NOT USE ANY EXTERNAL KNOWLEDGE.** Your entire answer must be derived *only* from the text in the "Available Research Excerpts" and "Knowledge Graph Facts" sections.
    2.  **USE BOTH SOURCES:** The KG facts provide high-level structure, and the Paper Excerpts provide detailed content. Use *both* to build your answer.
    3.  **DO NOT HALLUCINATE.** Do not invent facts, metrics, or methodological details. If the sources do not provide the information, you *MUST* state that the information is not present.
    4.  **CITE EVERYTHING:** Every factual claim *must* be followed by the correct citation (`[Paper: ...]` or `[Source: ...]` or both).
    5.  **NO GENERAL KNOWLEDGE:** Do not add general definitions (e.g., about "FAISS", "embeddings", "triples") unless those words are *explicitly* in the provided sources.

    **Answer Format:**
    1.  **Direct Answer:** Start with a brief comparison based on the sources.
    2.  **Detailed Explanation:** Provide a point-by-point comparison (a table is good). Cite every claim.
    3.  **If Not Possible:** If sources are insufficient, explicitly state *why* (e.g., "The papers describe GraphRAG [Paper: X] but the sources provide no details on vector-based RAG.").
    4.  **Confidence Note:** State your confidence.

    Comparison Answer:"""
        
        try:
            answer = await self._call_llm(prompt, temperature=0.0)
            
            # Fallback check (if answer is still empty)
            if not answer or len(answer.strip()) < 50:
                self.log_error(f"Synthesis returned empty/short answer, using generic fallback")
                answer = f"""Based on the {paper_count} research papers and {fact_count} knowledge graph facts, a direct comparison for "{query}" cannot be provided. The retrieved sources do not contain the necessary methodological or performance details to contrast both approaches."""
            
            # Extract metrics
            sources_used = self._extract_sources(context)
            papers_cited = len(self._extract_unique_papers(context))
            confidence = self._assess_confidence(answer, context, state)
            
            result = {
                "answer": answer,
                "confidence": confidence,
                "papers_cited": papers_cited,
                "sources_used": sources_used,
                "reasoning": f"Comparative synthesis from {paper_count} papers and {fact_count} facts"
            }
            
            self._update_statistics(result)
            
            self.log_info(
                f"Comparison answer generated: {len(answer)} chars, "
                f"{papers_cited} papers cited, confidence={result['confidence']:.2f}"
            )
            
            return result
        
        except Exception as e:
            self.log_error(f"Comparison answer generation failed with exception: {e}", exc_info=True)
            
            # Exception fallback
            return {
                "answer": f"Unable to generate a comparison for '{query}' due to a technical error: {str(e)}.",
                "confidence": 0.2,
                "papers_cited": 0,
                "sources_used": [],
                "reasoning": f"Exception during generation: {type(e).__name__}"
            }
        
    async def _generate_temporal_answer(
        self,
        query: str,
        context: Dict[str, Any],
        state: AgentState
    ) -> Dict[str, Any]:
        """
        Generate temporal/timeline answer for research evolution.
        """
        # This can reuse the standard answer logic
        return await self._generate_standard_answer(query, context, state)

    async def _generate_methodological_answer(
        self,
        query: str,
        context: Dict[str, Any],
        state: AgentState
    ) -> Dict[str, Any]:
        """
        Generate methodological answer explaining how methods work.
        """
        # This can reuse the standard answer logic
        return await self._generate_standard_answer(query, context, state)

    async def _generate_survey_answer(
        self,
        query: str,
        context: Dict[str, Any],
        state: AgentState
    ) -> Dict[str, Any]:
        """
        Generate a survey/overview answer synthesizing multiple papers.
        """
        # This can reuse the standard answer logic
        return await self._generate_standard_answer(query, context, state)
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _assess_confidence(
        self,
        answer: str,
        context: Dict[str, Any],
        state: AgentState
    ) -> float:
        """
        Assess confidence in the generated answer based on source quality.
        """
        confidence = 0.5  # Base confidence
        
        # More papers = higher confidence
        unique_papers = context.get("paper_count", 0)
        if unique_papers >= 5:
            confidence += 0.15
        elif unique_papers >= 3:
            confidence += 0.10
        
        # Knowledge graph facts add confidence
        if context["fact_count"] >= 5:
            confidence += 0.15
        elif context["fact_count"] >= 2:
            confidence += 0.10
        
        # Longer, more detailed answers suggest more information
        if len(answer) > 500:
            confidence += 0.10
        
        # Check for uncertainty markers
        uncertainty_markers = [
            "might", "possibly", "unclear", "uncertain",
            "insufficient", "limited evidence", "few studies"
        ]
        if any(marker in answer.lower() for marker in uncertainty_markers):
            confidence -= 0.15
        
        # Check for conflict markers
        conflict_markers = ["however", "conflicting", "disagree", "contradictory"]
        if any(marker in answer.lower() for marker in conflict_markers):
            confidence -= 0.10
        
        # Clamp to [0, 1]
        confidence = max(0.0, min(1.0, confidence))
        
        return confidence
    
    def _extract_sources(self, context: Dict[str, Any]) -> List[str]:
        """
        Extract unique paper sources with metadata from context.
        
        Args:
            context: Context with chunks
        
        Returns:
            List of source citations
        """
        sources = []
        seen_papers = set()
        
        for chunk in context.get("chunks", []):
            paper_title = chunk.get("document_title")
            if paper_title and paper_title not in seen_papers:
                seen_papers.add(paper_title)
                
                # Build citation string
                authors = chunk.get("authors", [])
                venue = chunk.get("venue", "")
                pub_date = chunk.get("publication_date", "")
                
                author_str = ", ".join(authors[:2]) if authors else ""
                if len(authors) > 2:
                    author_str += " et al."
                
                citation = paper_title
                if author_str:
                    citation = f"{author_str}: {citation}"
                if venue:
                    citation += f" ({venue}"
                    if pub_date:
                        citation += f", {pub_date}"
                    citation += ")"
                
                sources.append(citation)
        
        # Add knowledge graph as a source if facts were used
        if context.get("fact_count", 0) > 0:
            sources.append("Academic Knowledge Graph")
        
        return sources
    
    def _extract_unique_papers(self, context: Dict[str, Any]) -> List[str]:
        """
        Extract list of unique papers cited.
        
        Args:
            context: Context with chunks
        
        Returns:
            List of unique paper IDs
        """
        unique_papers = set()
        for chunk in context.get("chunks", []):
            paper_id = chunk.get("document_id") or chunk.get("document_title")
            if paper_id:
                unique_papers.add(paper_id)
        
        return list(unique_papers)
    
    def _generate_reasoning(
        self,
        query: str,
        context: Dict[str, Any],
        answer: str
    ) -> str:
        """
        Generate explanation of reasoning process.
        
        Args:
            query: Original query
            context: Context used
            answer: Generated answer
        
        Returns:
            Reasoning explanation
        """
        reasoning_parts = []
        
        reasoning_parts.append(
            f"To answer '{query}', I analyzed {context['chunk_count']} excerpts "
            f"from {context['paper_count']} research papers"
        )
        
        if context["fact_count"] > 0:
            reasoning_parts.append(
                f"and {context['fact_count']} facts from the academic knowledge graph"
            )
        
        if context["entity_count"] > 0:
            reasoning_parts.append(
                f"I identified {context['entity_count']} key entities "
                f"(authors, methods, concepts)"
            )
        
        if context["relationship_count"] > 0:
            reasoning_parts.append(
                f"and {context['relationship_count']} relationships "
                f"(citations, authorship, methodological connections)"
            )
        
        reasoning_parts.append(
            "I synthesized this information with proper attribution to generate "
            "a comprehensive answer based on the research literature"
        )
        
        return ". ".join(reasoning_parts) + "."
    
    def _update_statistics(self, result: Dict[str, Any]):
        """
        Update synthesis statistics.
        
        Args:
            result: Synthesis results
        """
        self.total_answers_generated += 1
        self.total_sources_combined += len(result.get("sources_used", []))
        
        answer_length = len(result.get("answer", ""))
        self.avg_answer_length = (
            (self.avg_answer_length * (self.total_answers_generated - 1) + answer_length)
            / self.total_answers_generated
        )
        
        confidence = result.get("confidence", 0.5)
        self.avg_confidence_scores.append(confidence)
    
    def _store_results_in_state(
        self,
        state: AgentState,
        result: Dict[str, Any],
        task: SubTask
    ):
        """
        Store synthesis results in state.
        
        Args:
            state: Current state
            result: Synthesis results
            task: Original task
        """
        synthesis_result = SynthesisResult(
            answer=result["answer"],
            confidence=result["confidence"],
            sources_used=result["sources_used"],
            reasoning=result["reasoning"]
        )
        
        state["synthesized_answer"] = synthesis_result
        state["final_answer"] = result["answer"]
        state["sources"] = result["sources_used"]
        state["confidence"] = result["confidence"]
        
        self.log_debug("Stored synthesis results in state")
    
    def get_synthesis_statistics(self) -> Dict[str, Any]:
        """
        Get synthesis-specific statistics.
        
        Returns:
            Dict with synthesis stats
        """
        avg_confidence = (
            sum(self.avg_confidence_scores) / len(self.avg_confidence_scores)
            if self.avg_confidence_scores
            else 0.0
        )
        
        base_stats = self.get_performance_stats()
        
        return {
            **base_stats,
            "total_answers_generated": self.total_answers_generated,
            "total_sources_combined": self.total_sources_combined,
            "total_papers_cited": self.total_papers_cited,
            "avg_answer_length": int(self.avg_answer_length),
            "avg_confidence": avg_confidence,
            "avg_sources_per_answer": (
                self.total_sources_combined / self.total_answers_generated
                if self.total_answers_generated > 0
                else 0.0
            ),
            "avg_papers_per_answer": (
                self.total_papers_cited / self.total_answers_generated
                if self.total_answers_generated > 0
                else 0.0
            ),
            "domain": "scientific_literature"
        }


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_synthesis_agent(model_name: Optional[str] = None) -> SynthesisAgent:
    """
    Create and initialize a synthesis agent for scientific literature.
    
    Args:
        model_name: Optional LLM model name
    
    Returns:
        Initialized SynthesisAgent
    """
    return SynthesisAgent(model_name=model_name)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

async def example_usage():
    """Example showing synthesis agent in action for scientific literature."""
    from ..state import create_initial_state, add_subtask, RetrievalResult, KnowledgeResult
    
    # Create agent
    agent = create_synthesis_agent()
    
    # Create state for scientific query
    state = create_initial_state(
        query="Compare Transformer and LSTM architectures for NLP",
        search_type="hybrid"
    )
    
    # Add mock retrieval results
    state["retrieved_contexts"].append(RetrievalResult(
        search_type="hybrid",
        chunks=[
            {
                "chunk_id": "1",
                "document_id": "attention_paper",
                "content": "The Transformer architecture relies entirely on self-attention mechanisms...",
                "score": 0.92,
                "document_title": "Attention Is All You Need",
                "document_source": "attention_paper.pdf",
                "authors": ["Vaswani", "Shazeer", "Parmar"],
                "venue": "NeurIPS",
                "publication_date": "2017",
                "citation_count": 85000
            },
            {
                "chunk_id": "2",
                "document_id": "lstm_paper",
                "content": "Long Short-Term Memory networks use gated recurrent units...",
                "score": 0.88,
                "document_title": "Long Short-Term Memory",
                "document_source": "lstm_paper.pdf",
                "authors": ["Hochreiter", "Schmidhuber"],
                "venue": "Neural Computation",
                "publication_date": "1997",
                "citation_count": 45000
            }
        ],
        graph_facts=[
            {"fact": "Transformer architecture proposed by Vaswani et al. in 2017", "fact_type": "contribution"},
            {"fact": "LSTM addresses vanishing gradient problem in RNNs", "fact_type": "contribution"}
        ],
        total_results=2,
        avg_score=0.90,
        sources=["attention_paper.pdf", "lstm_paper.pdf"]
    ))
    
    # Add mock knowledge results
    state["knowledge_graph_data"].append(KnowledgeResult(
        entities=[
            {"name": "Transformer", "type": "METHOD"},
            {"name": "LSTM", "type": "METHOD"},
            {"name": "Vaswani", "type": "AUTHOR"}
        ],
        relationships=[
            {"source": "Transformer", "target": "LSTM", "type": "EXTENDS"},
            {"source": "Vaswani", "target": "Attention Is All You Need", "type": "AUTHORED"}
        ],
        temporal_facts=[],
        graph_paths=[]
    ))
    
    # Create synthesis task
    task = add_subtask(
        state=state,
        task_type=TaskType.SYNTHESIS,
        description="Synthesize comparison answer for Transformer vs LSTM",
        parameters={
            "query": state["query"],
            "intent": QueryIntent.COMPARISON,
            "synthesis_type": "comprehensive"
        },
        assigned_agent=AgentRole.SYNTHESIS,
        priority=5
    )
    
    print(f"Created synthesis task: {task['task_id']}")
    
    # Execute task
    print("\nExecuting synthesis...")
    result = await agent.execute(state, task)
    
    print(f"\nSynthesis result:")
    print(f"  Success: {result['success']}")
    print(f"  Confidence: {result['data']['confidence']:.2f}")
    print(f"  Papers cited: {result['data']['papers_cited']}")
    print(f"  Sources used: {len(result['data']['sources_used'])}")
    print(f"  Answer length: {len(result['data']['answer'])} chars")
    print(f"\n  Answer preview:")
    print(f"  {result['data']['answer'][:300]}...")
    
    # Show agent statistics
    stats = agent.get_synthesis_statistics()
    print(f"\nAgent statistics:")
    print(f"  Total answers: {stats['total_answers_generated']}")
    print(f"  Total papers cited: {stats['total_papers_cited']}")
    print(f"  Avg confidence: {stats['avg_confidence']:.2f}")
    print(f"  Avg answer length: {stats['avg_answer_length']} chars")
    print(f"  Domain: {stats['domain']}")


if __name__ == "__main__":
    import asyncio
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    asyncio.run(example_usage())