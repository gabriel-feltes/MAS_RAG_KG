"""
Validation Specialist Agent.

This agent is responsible for validating and verifying answers:
- Fact-checking against research sources
- Hallucination detection
- Citation verification
- Consistency checking across papers
- Confidence validation
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import re
import json

from ..base_agent import BaseAgent
from ..state import (
    AgentState,
    AgentRole,
    SubTask,
    TaskType,
    ValidationResult
)

logger = logging.getLogger(__name__)

# ============================================================================
# VALIDATION SPECIALIST AGENT
# ============================================================================

class ValidationAgent(BaseAgent):
    """
    Specialist agent for answer validation and verification in scientific literature context.
    
    Responsibilities:
    - Fact-check synthesized answers against research papers
    - Detect hallucinations (unsupported claims)
    - Verify source citations (authors, papers, venues)
    - Check consistency across multiple papers
    - Add proper academic citations
    - Assess answer quality and research rigor
    
    Capabilities:
    - Claim extraction from scientific answers
    - Source verification against papers
    - Hallucination scoring with research context
    - Citation formatting (academic style)
    - Quality assessment for research queries
    - Detection of contradictions across studies
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize validation agent for scientific literature.
        
        Args:
            model_name: Optional LLM model name
        """
        super().__init__(
            role=AgentRole.VALIDATION,
            name="Validation Specialist",
            description="Expert in fact-checking, citation verification, and answer quality validation for scientific literature",
            model_name=model_name
        )
        
        # Validation statistics
        self.total_validations = 0
        self.total_claims_checked = 0
        self.total_hallucinations_detected = 0
        self.total_citations_verified = 0
        self.total_papers_referenced = 0
        self.avg_hallucination_scores = []
        
        self.log_info("Initialized with fact-checking and validation capabilities for scientific literature")
    
    # ========================================================================
    # ABSTRACT METHOD IMPLEMENTATIONS
    # ========================================================================
    
    async def process_task(self, state: AgentState, task: SubTask) -> Dict[str, Any]:
        """
        Process a validation task.
        
        Args:
            state: Current system state
            task: Validation subtask
        
        Returns:
            Validation results
        """
        task_type = str(task.get("task_type", "")).lower()
        parameters = task.get("parameters", {})
        
        self.log_info(f"Processing {task_type} task for scientific literature")
        
        if "validation" in task_type or task_type == TaskType.VALIDATION.value.lower():
            result = await self.validate_answer(state, parameters)
        else:
            raise ValueError(f"Unsupported task type: {task.get('task_type')}")
        
        # Update statistics
        self._update_statistics(result)
        
        # Store results in state
        self._store_results_in_state(state, result, task)
        
        return result
    
    def can_handle_task(self, task: Dict[str, Any]) -> bool:
        """Check if this agent can handle the task."""
        task_type = task.get("task_type", "")
        agent_role = task.get("agent_role", "")
        
        task_type_str = str(task_type).lower()
        agent_role_str = str(agent_role).lower()
        
        valid_types = ["validation", TaskType.VALIDATION.value.lower()]
        valid_roles = ["validation", AgentRole.VALIDATION.value.lower()]
        
        return task_type_str in valid_types or agent_role_str in valid_roles
    
    def get_system_prompt(self) -> str:
        """Get system prompt for LLM interactions."""
        return """You are a Validation Specialist Agent in a multi-agent system focused on scientific literature.

Your responsibilities:
1. Fact-check synthesized answers against research paper sources
2. Detect hallucinations (claims not supported by research literature)
3. Verify that citations are accurate (authors, paper titles, venues, dates)
4. Check consistency between claims and research findings
5. Add proper academic citations where missing
6. Assess overall answer quality and research rigor

Validation criteria for scientific literature:

**FACTUAL ACCURACY:**
- Every research claim should be supported by papers
- Numbers, dates, author names, and venues must be exact
- Methodological descriptions must match paper content
- Performance metrics must be verifiable
- Relationships between methods/concepts must be traceable

**HALLUCINATION DETECTION:**
- Flag claims with no paper support
- Identify speculative statements not from sources
- Detect contradictions with research findings
- Check for fabricated citations or non-existent papers
- Verify author attributions and paper titles
- Ensure venue/conference names are correct

**CITATION QUALITY:**
- All research facts should have source attribution
- Citations should be specific (paper title, authors, venue, year)
- Multiple papers strengthen claims
- Distinguish between established consensus and recent findings
- Note if claims come from highly-cited influential papers

**CONSISTENCY ACROSS STUDIES:**
- Check for internal contradictions
- Verify agreement across multiple papers
- Flag conflicting findings (e.g., "Paper A found X, but Paper B found Y")
- Note if results vary by methodology or dataset

**RESEARCH RIGOR:**
- Verify if claims represent current consensus
- Check if answer acknowledges limitations
- Ensure proper contextualization of findings
- Verify temporal accuracy (when methods were proposed)

Your output should include:
1. Validation status (valid/invalid/partially_valid)
2. List of verified research claims with paper sources
3. List of unsupported claims (potential hallucinations)
4. Citation verification results
5. Issues found (incorrect attributions, missing citations, contradictions)
6. Overall quality assessment for research context

Be thorough but fair - synthesis statements that combine multiple findings may not need individual citations if the underlying claims are cited."""
    
    # ========================================================================
    # VALIDATION METHODS
    # ========================================================================
    
    async def validate_answer(
        self,
        state: AgentState,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate the synthesized answer against research sources.
        
        Args:
            state: Current state with answer and sources
            parameters: Validation parameters
        
        Returns:
            Validation results with DETAILED claims
        """
        query = parameters.get("query", state["query"])
        validation_type = parameters.get("validation_type", "fact_check")
        
        # Get answer to validate
        answer = state.get("final_answer")
        if not answer:
            synthesized = state.get("synthesized_answer")
            if synthesized and isinstance(synthesized, dict):
                answer = synthesized.get("answer", "")
            else:
                answer = synthesized
        
        if not answer:
            self.log_warning("No answer to validate")
            return self._create_empty_validation()
        
        self.log_info(f"Validating scientific answer ({len(answer)} chars)")
        
        # Get source material
        retrieval_contexts = state.get("retrieved_contexts", [])
        knowledge_data = state.get("knowledge_graph_data", [])
        
        # Extract claims from answer
        claims = await self._extract_claims(answer)
        self.log_info(f"Extracted {len(claims)} research claims to verify")
        
        # Fact-check each claim against papers
        fact_checks = await self._fact_check_claims(
            claims=claims,
            retrieval_contexts=retrieval_contexts,
            knowledge_data=knowledge_data
        )
        
        # Calculate hallucination score
        hallucination_score = self._calculate_hallucination_score(fact_checks)
        
        # Verify citations (author names, paper titles, etc.)
        citation_verification = await self._verify_citations(answer, retrieval_contexts)
        
        # Add/improve citations
        answer_with_citations, citations = self._add_citations(
            answer=answer,
            fact_checks=fact_checks,
            retrieval_contexts=retrieval_contexts
        )
        
        # Identify issues
        issues = self._identify_issues(fact_checks, hallucination_score, citation_verification)
        
        # Determine overall validity
        is_valid = self._assess_validity(fact_checks, hallucination_score, citation_verification)
        
        # Count unique papers referenced
        unique_papers = len(set(
            source
            for fc in fact_checks
            if fc["supported"]
            for source in fc["sources"]
        ))
        
        # ✅ BUILD DETAILED CLAIMS ARRAY
        detailed_claims = []
        for i, fc in enumerate(fact_checks, 1):
            detailed_claims.append({
                "claim_id": i,
                "claim_text": fc["claim"],
                "status": "✓ VERIFIED" if fc["supported"] else "✗ NOT VERIFIED",
                "confidence": fc["confidence"],
                "supporting_sources": fc["sources"] if fc["supported"] else [],
                "source_count": fc["source_count"],
                "verification_method": "llm" if fc.get("llm_verified") else "fuzzy_match"
            })
        
        # ✅ BUILD HALLUCINATION EXAMPLES
        hallucination_examples = [
            {
                "claim_text": fc["claim"],
                "reason": "No supporting source found in research papers or knowledge graph"
            }
            for fc in fact_checks
            if not fc["supported"]
        ]
        
        result = {
            "is_valid": is_valid,
            "fact_checks": fact_checks,
            "hallucination_score": hallucination_score,
            "citation_verification": citation_verification,
            "citations": citations,
            "issues": issues,
            "original_answer": answer,
            "validated_answer": answer_with_citations,
            "total_claims": len(claims),
            "verified_claims": sum(1 for fc in fact_checks if fc["supported"]),
            "unsupported_claims": sum(1 for fc in fact_checks if not fc["supported"]),
            "papers_referenced": unique_papers,
            # ✅ NEW TRANSPARENCY FIELDS
            "detailed_claims": detailed_claims,
            "hallucination_examples": hallucination_examples,
            "verification_summary": {
                "total_claims_checked": len(claims),
                "verified_by_llm": sum(1 for fc in fact_checks if fc.get("llm_verified") and fc["supported"]),
                "verified_by_fuzzy": sum(1 for fc in fact_checks if not fc.get("llm_verified") and fc["supported"]),
                "unverified": len(hallucination_examples),
                "average_confidence": sum(fc["confidence"] for fc in fact_checks) / len(fact_checks) if fact_checks else 0.0
            }
        }
        
        self.total_papers_referenced += unique_papers
        
        self.log_info(
            f"Validation complete: valid={is_valid}, "
            f"hallucination={hallucination_score:.2f}, "
            f"verified={result['verified_claims']}/{result['total_claims']}, "
            f"papers={unique_papers}"
        )
        
        return result
    
    # ========================================================================
    # CLAIM EXTRACTION
    # ========================================================================
    
    async def _extract_claims(self, answer: str) -> List[str]:
        """Extract research claims from the scientific answer."""
        # ✅ FIX: Updated prompt to ignore meta-claims
        prompt = f"""Extract all factual, positive research claims from the following scientific answer.

A factual claim is a specific statement of fact asserted in the answer:
- "Transformers use self-attention mechanisms" (VERIFY)
- "BERT achieved 92% accuracy" (VERIFY)
- "ResNet was proposed by He et al." (VERIFY)

Do NOT extract the following:
- ⚠️ **META-CLAIMS (IMPORTANT):** Claims *about* the literature, such as "The provided literature does not contain...", "No paper mentions...", "The excerpts do not describe...".
- ⚠️ **CONCLUSIONS:** Summary statements like "Therefore, a comparison cannot be drawn...".
- ⚠️ **GENERAL KNOWLEDGE:** Statements like "Vector-based RAG typically uses..."
- Questions, transitional phrases, or non-factual sentences.

Answer:
{answer}

You MUST respond with a valid JSON array of strings. If no factual claims are found, you MUST return an empty list [].
Your response MUST be ONLY the JSON array, with no other text.

Example response:
["claim 1", "claim 2", ...]"""
        
        try:
            response = await self._call_llm(prompt, temperature=0.0)
            response_str = str(response).strip()
            
            # Handle fenced code blocks
            if response_str.startswith("```"):
                first_newline = response_str.find("\n")
                if first_newline != -1:
                    response_str = response_str[first_newline + 1:]
                if response_str.endswith("```"):
                    response_str = response_str[:-3].strip()
            
            response_str = response_str.strip()
            
            # Parse JSON
            claims = json.loads(response_str)
            return claims if isinstance(claims, list) else []
        
        except Exception as e:
            self.log_warning(f"Claim extraction failed: {e}")
            # Fallback: simple sentence splitting
            sentences = re.split(r'[.!?]', answer)
            return [s.strip() for s in sentences if len(s.strip()) > 20][:10]
        
    # ========================================================================
    # FACT CHECKING
    # ========================================================================
    
    async def _fact_check_claims(
        self,
        claims: List[str],
        retrieval_contexts: List[Any],
        knowledge_data: List[Any]
    ) -> List[Dict[str, Any]]:
        """Fact-check each research claim against papers."""
        source_texts = []
        graph_facts = []
        
        # Prepare source material
        for ctx in retrieval_contexts:
            if isinstance(ctx, dict):
                # Extract chunks from papers
                if "chunks" in ctx:
                    for chunk in ctx.get("chunks", []):
                        paper_title = chunk.get("document_title", "Unknown Paper")
                        authors = chunk.get("authors", [])
                        venue = chunk.get("venue", "")
                        pub_date = chunk.get("publication_date", "")
                        
                        # Add chunks with metadata
                        author_str = ", ".join(authors[:2]) if authors else ""
                        if len(authors) > 2:
                            author_str += " et al."
                        
                        source_citation = paper_title
                        if author_str:
                            source_citation = f"{author_str} - {paper_title}"
                        if venue:
                            source_citation = f"{source_citation} ({venue}"
                        if pub_date:
                            source_citation = f"{source_citation}, {pub_date})"
                        
                        source_texts.append({
                            "text": chunk.get("content", ""),
                            "source": source_citation,
                            "paper_title": paper_title
                        })
                
                # Add graph facts
                if "graph_facts" in ctx:
                    for fact in ctx.get("graph_facts", []):
                        if isinstance(fact, dict):
                            graph_facts.append(fact.get("fact", str(fact)))
                        else:
                            graph_facts.append(str(fact))
        
        # Extract graph facts from knowledge data
        for kg_data in knowledge_data:
            if isinstance(kg_data, dict):
                if "entities" in kg_data:
                    for entity in kg_data.get("entities", []):
                        if "facts" in entity:
                            graph_facts.extend(entity["facts"])
                
                if "temporal_facts" in kg_data:
                    for fact in kg_data.get("temporal_facts", []):
                        graph_facts.append(fact.get("fact", str(fact)))
        
        # Add graph facts to sources
        for fact in set(graph_facts):
            source_texts.append({
                "text": str(fact),
                "source": "Academic Knowledge Graph",
                "paper_title": "Knowledge Graph"
            })
        
        self.log_info(f"Checking against {len(source_texts)} source segments from research papers")
        
        # Verify each claim
        fact_checks = []
        for claim in claims:
            verification = await self._verify_single_claim_llm(claim, source_texts)
            fact_checks.append(verification)
        
        return fact_checks
    
    def _is_domain_knowledge(self, claim: str) -> bool:
        """
        Check if a claim represents common domain knowledge in RAG/NLP.
        """
        domain_patterns = [
            # Vector embeddings
            r'\b(vector|embedding)s?\b.*\b(cosine|similarity|distance)\b',
            r'\bANN\b.*\bindex',
            r'\bdense.*vectors?\b',
            
            # Graph concepts
            r'\bgraph.*\b(traversal|structure|nodes?|edges?)\b',
            r'\bknowledge graph\b',
            r'\bentity.*relations?',
            
            # RAG fundamentals
            r'\bretrieval.*augmented.*generation\b',
            r'\bsemantic search\b',
            r'\bdocument retrieval\b',
            
            # ML basics
            r'\btransformer.*attention\b',
            r'\bBERT\b.*\bpre-?trained\b',
            r'\bfine-?tun(e|ing)\b'
        ]
        
        import re
        claim_lower = claim.lower()
        
        for pattern in domain_patterns:
            if re.search(pattern, claim_lower, re.IGNORECASE):
                return True
        
        return False
    
    async def _verify_single_claim_llm(
        self,
        claim: str,
        source_texts: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Verify a single research claim against papers using LLM."""
        if not source_texts:
            return {
                "claim": claim,
                "supported": False,
                "confidence": 0.0,
                "sources": [],
                "source_count": 0,
                "llm_verified": False
            }
        
        # Combine all sources into context
        context_str = ""
        source_map = {}
        for i, source in enumerate(source_texts, 1):
            source_name = source["source"]
            source_text = source["text"]
            context_str += f"\n[Paper {i}] {source_name}:\n{source_text}\n"
            source_map[i] = source_name
        
        # LLM verification prompt
        prompt = f"""You are a scientific fact-checker. Your task is to verify a research claim against provided paper sources.

Here is the research claim:
CLAIM: {claim}

Here are the paper sources:
{context_str}

Analyze the sources and determine if the claim is supported by the research literature.

Consider:
- Direct support: paper explicitly states this
- Indirect support: can be inferred from paper content
- Paraphrasing: claim restates paper findings
- Methodological accuracy: descriptions match paper methods

You MUST respond with a valid JSON object in the following format:
{{
    "supported": true/false,
    "confidence": 0.0-1.0,
    "supporting_sources": [1, 2, ...]
}}

- supported: true if the papers contain information that supports the claim.
- confidence: Your confidence in this assessment (0.0 to 1.0)
- supporting_sources: List of paper numbers that support the claim. If not supported, return [].

You MUST respond with ONLY the JSON object, with no other text."""
        
        try:
            response_str = await self._call_llm(prompt, temperature=0.0)
            response_str = str(response_str).strip()
            
            # Handle fenced code blocks
            if response_str.startswith("```"):
                first_newline = response_str.find("\n")
                if first_newline != -1:
                    response_str = response_str[first_newline + 1:]
                if response_str.endswith("```"):
                    response_str = response_str[:-3].strip()
            
            result_json = json.loads(response_str)
            
            is_supported_by_llm = result_json.get("supported", False)
            confidence = float(result_json.get("confidence", 0.0))
            source_indices = result_json.get("supporting_sources", [])
            
            # Map indices back to source citations
            supporting_sources = list(set(
                source_map[i] for i in source_indices if i in source_map
            ))
            source_count = len(supporting_sources)
            
            # ✅ CRITICAL FIX: Revert to the simple, strict logic.
            # A claim is ONLY supported if the LLM says so AND it provides a valid source.
            is_supported = is_supported_by_llm and source_count > 0

            if is_supported_by_llm and source_count == 0:
                 self.log_warning(
                     f"LLM reported 'supported: true' but provided no valid "
                     f"sources for claim: {claim}"
                 )
            
            return {
                "claim": claim,
                "supported": is_supported, # This is now consistent
                "confidence": confidence,
                "sources": supporting_sources,
                "source_count": source_count, # Report the *actual* count
                "llm_verified": True
            }
        
        except Exception as e:
            self.log_warning(f"LLM-based claim verification failed: {e}. Claim: {claim}")
            return self._verify_single_claim_simple(claim, source_texts)
        
    def _verify_single_claim_simple(
        self,
        claim: str,
        source_texts: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Fallback: Simple string/word overlap verification."""
        supported = False
        supporting_sources = []
        confidence = 0.0
        
        claim_lower = claim.lower()
        
        for source in source_texts:
            source_text = source["text"].lower()
            
            # Exact substring match
            if claim_lower in source_text:
                supported = True
                supporting_sources.append(source["source"])
                confidence = max(confidence, 0.9)
            else:
                # Word overlap heuristic
                claim_words = set(re.findall(r'\w+', claim_lower))
                source_words = set(re.findall(r'\w+', source_text))
                
                # Remove common words
                common_words = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for", "of", "by"}
                claim_words -= common_words
                source_words -= common_words
                
                if len(claim_words) > 0:
                    overlap = len(claim_words & source_words) / len(claim_words)
                    if overlap > 0.5:
                        supported = True
                        supporting_sources.append(source["source"])
                        confidence = max(confidence, overlap * 0.9)
        
        supporting_sources = list(set(supporting_sources))
        
        return {
            "claim": claim,
            "supported": supported,
            "confidence": confidence,
            "sources": supporting_sources,
            "source_count": len(supporting_sources),
            "llm_verified": False
        }
    
    # ========================================================================
    # CITATION VERIFICATION
    # ========================================================================
    
    async def _verify_citations(
        self,
        answer: str,
        retrieval_contexts: List[Any]
    ) -> Dict[str, Any]:
        """
        Verify that papers cited in the answer actually exist in the retrieved contexts.
        
        Filters out false positives like section headers, bullet points, and metadata.
        """
        from difflib import SequenceMatcher
        
        def fuzzy_match(text1: str, text2: str) -> float:
            """Calculate similarity ratio between two strings."""
            return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
        
        def is_valid_citation(text: str) -> bool:
            """
            Check if extracted text is a valid citation (not a header or metadata).
            
            Returns False for:
            - Section headers (e.g., "1. Direct answer", "Additional context")
            - Single words or very short phrases
            - Numeric labels
            - Common metadata keywords
            """
            text_stripped = text.strip()
            
            # Too short to be a paper citation
            if len(text_stripped) < 10:
                return False
            
            # Pure numbers or simple numeric labels
            if text_stripped.isdigit() or re.match(r'^\d+\.?$', text_stripped):
                return False
            
            # Section header patterns
            header_patterns = [
                r'^\d+\.\s+',  # "1. ", "2. ", etc.
                r'^\*\*\d+\.\s+',  # "**1. "
                r'^#+\s+',  # Markdown headers
            ]
            if any(re.match(pattern, text_stripped) for pattern in header_patterns):
                return False
            
            # Common section keywords (not actual citations)
            non_citation_keywords = [
                'direct answer', 'detailed explanation', 'additional context',
                'source citations', 'confidence note', 'supporting evidence',
                'what', 'how', 'why', 'when', 'where',  # Question words
                'introduction', 'conclusion', 'summary', 'overview',
                'background', 'methodology', 'results', 'discussion',
                'table', 'figure', 'appendix', 'references',
                'agentic rag', 'graph databases', 'multi-hop reasoning',  # Common topics
                'modularity', 'scalability', 'explainability',
                'regulatory compliance', 'domain-specific',
                'structured knowledge', 'integration of'
            ]
            
            text_lower = text_stripped.lower()
            if any(keyword in text_lower for keyword in non_citation_keywords):
                # Exception: if it contains "Paper" or "et al.", it might be valid
                if 'paper' not in text_lower and 'et al' not in text_lower:
                    return False
            
            # Likely a topic/concept phrase if it starts with a common concept word
            concept_starters = [
                'the ', 'a ', 'an ', 'this ', 'these ', 'that ', 'those ',
                'our ', 'their ', 'its ', 'such '
            ]
            if any(text_lower.startswith(starter) for starter in concept_starters):
                # Exception: actual paper titles often start with "The" or "A"
                # Check if it looks like a paper title (has capitals, colons, etc.)
                if ':' not in text_stripped and not re.search(r'[A-Z][a-z]+\s+[A-Z][a-z]+', text_stripped):
                    return False
            
            return True
        
        # Extract paper titles/citations mentioned in the answer
        cited_patterns = [
            r'\[Paper \d*:\s*([^\]]+)\]',  # [Paper 1: Full Title] or [Paper: Full Title]
            r'\[Source:\s*([^\]]+)\]',      # [Source: Knowledge Graph]
            r'\(([A-Z][a-z]+ et al\.?,?\s+\d{4})\)',  # (Author et al., 2020)
        ]
        
        mentioned_papers = set()
        for pattern in cited_patterns:
            matches = re.findall(pattern, answer)
            for match in matches:
                cleaned = match.strip()
                # Apply validation filter
                if is_valid_citation(cleaned):
                    mentioned_papers.add(cleaned)
        
        self.log_debug(
            f"Extracted {len(mentioned_papers)} valid citation mentions from answer "
            f"(after filtering section headers)"
        )
        
        # Get actual papers from retrieval contexts
        actual_papers = []
        has_graph_facts = False
        
        for ctx in retrieval_contexts:
            if isinstance(ctx, dict):
                # Check for graph facts
                graph_facts = ctx.get("graph_facts", [])
                if graph_facts and len(graph_facts) > 0:
                    has_graph_facts = True
                
                # Collect paper info
                chunks = ctx.get("chunks", [])
                for chunk in chunks:
                    paper_info = {
                        "title": chunk.get("document_title", ""),
                        "source": chunk.get("document_source", ""),
                        "id": chunk.get("document_id", "")
                    }
                    if paper_info["title"]:
                        if not any(p["title"] == paper_info["title"] for p in actual_papers):
                            actual_papers.append(paper_info)
        
        self.log_debug(f"Found {len(actual_papers)} unique papers in retrieval contexts")
        if has_graph_facts:
            self.log_debug("Knowledge graph facts are available")
        
        # Check which mentioned papers are actually in the sources
        verified_citations = []
        unverified_citations = []
        
        for mentioned in mentioned_papers:
            # Special handling for generic "Knowledge Graph" citations
            if any(kg_term in mentioned.lower() for kg_term in ['knowledge graph', 'academic knowledge graph']):
                if has_graph_facts:
                    verified_citations.append({
                        'mentioned': mentioned,
                        'matched_paper': 'Knowledge Graph (graph facts)',
                        'similarity': 1.0
                    })
                    self.log_debug(f"✓ Verified Knowledge Graph citation: '{mentioned}'")
                    continue
                else:
                    unverified_citations.append(mentioned)
                    self.log_debug(f"✗ Unverified Knowledge Graph citation (no graph facts available): '{mentioned}'")
                    continue
            
            # Try to match against actual papers
            best_match = None
            best_score = 0.0
            
            for paper in actual_papers:
                # Calculate similarity scores
                title_score = fuzzy_match(mentioned, paper["title"])
                source_score = fuzzy_match(mentioned, paper["source"])
                
                # Check for exact substring matches
                if mentioned.lower() in paper["title"].lower():
                    title_score = max(title_score, 0.8)
                if paper["title"].lower() in mentioned.lower():
                    title_score = max(title_score, 0.8)
                
                if mentioned.lower() in paper["source"].lower():
                    source_score = max(source_score, 0.7)
                
                # Take the best score
                score = max(title_score, source_score)
                
                if score > best_score:
                    best_score = score
                    best_match = paper
            
            # Consider it verified if similarity > 40%
            if best_score >= 0.4:
                verified_citations.append({
                    'mentioned': mentioned,
                    'matched_paper': best_match['title'],
                    'similarity': best_score
                })
                self.log_debug(
                    f"✓ Verified citation '{mentioned[:40]}...' → "
                    f"'{best_match['title'][:40]}...' (similarity: {best_score:.2f})"
                )
            else:
                unverified_citations.append(mentioned)
                self.log_debug(f"✗ Unverified citation: '{mentioned[:60]}...'")
        
        # Calculate metrics
        total_citations = len(mentioned_papers)
        verified_count = len(verified_citations)
        unverified_count = len(unverified_citations)
        
        citation_accuracy = (
            verified_count / total_citations
            if total_citations > 0
            else 1.0
        )
        
        self.log_info(
            f"Citation verification: {verified_count}/{total_citations} verified "
            f"({citation_accuracy:.1%} accuracy)"
        )
        
        return {
            "total_citations_mentioned": total_citations,
            "verified_citations": verified_count,
            "unverified_citations": unverified_count,
            "verified_list": verified_citations,
            "unverified_list": unverified_citations,
            "papers_available": len(actual_papers),
            "has_graph_facts": has_graph_facts,
            "citation_accuracy": citation_accuracy
        }
    
    # ========================================================================
    # SCORING AND ASSESSMENT
    # ========================================================================
    
    def _calculate_hallucination_score(self, fact_checks: List[Dict[str, Any]]) -> float:
        """
        Calculate hallucination score with domain knowledge consideration.
        
        IMPROVED: Weights claims by confidence and domain knowledge.
        """
        if not fact_checks:
            return 0.0
        
        total_weight = 0.0
        unsupported_weight = 0.0
        
        for fc in fact_checks:
            # Check if this is domain knowledge
            is_domain = self._is_domain_knowledge(fc.get("claim", ""))
            
            # Weight: higher for explicit claims, lower for domain knowledge
            weight = 0.5 if is_domain else 1.0
            
            # If not supported but has high confidence, reduce penalty
            confidence = fc.get("confidence", 0.0)
            if not fc.get("supported") and confidence > 0.6:
                weight *= 0.5  # Reduce penalty for high-confidence unsupported
            
            total_weight += weight
            
            if not fc.get("supported"):
                unsupported_weight += weight
        
        hallucination_score = unsupported_weight / total_weight if total_weight > 0 else 0.0
        
        return hallucination_score
    
    def _identify_issues(
        self,
        fact_checks: List[Dict[str, Any]],
        hallucination_score: float,
        citation_verification: Dict[str, Any]
    ) -> List[str]:
        """Identify specific issues with the answer."""
        issues = []
        
        # Check hallucination score
        if hallucination_score > 0.4:
            issues.append(
                f"High hallucination risk (score: {hallucination_score:.2f}). "
                f"Answer may contain unsupported claims."
            )
        elif hallucination_score > 0.2:
            issues.append(
                f"Moderate hallucination risk (score: {hallucination_score:.2f}). "
                f"Some claims may lack strong support."
            )
        
        # Check failed fact checks
        failed_checks = [fc for fc in fact_checks if not fc.get("supported", False)]
        if failed_checks:
            issues.append(
                f"Unsupported claims detected: {len(failed_checks)} claims "
                f"could not be verified against sources."
            )
        
        # Check citation verification
        unverified_count = citation_verification.get("unverified_citations", 0)
        citation_accuracy = citation_verification.get("citation_accuracy", 1.0)
        papers_available = citation_verification.get("papers_available", 0)
        total_citations = citation_verification.get("total_citations_mentioned", 0)
        
        if isinstance(unverified_count, list):
            unverified_count = len(unverified_count)
        elif not isinstance(unverified_count, int):
            unverified_count = 0
        
        if unverified_count > 0:
            issues.append(
                f"Unverified citations: {unverified_count}/{total_citations} papers cited "
                f"may not be in the source material."
            )
        
        if citation_accuracy < 0.5:
            issues.append(
                f"Low citation accuracy ({citation_accuracy:.1%}). "
                f"Many cited papers do not match sources."
            )
        elif citation_accuracy < 0.7:
            issues.append(
                f"Moderate citation accuracy ({citation_accuracy:.1%}). "
                f"Some citations may be incomplete or inaccurate."
            )
        
        # Check if no sources were used
        if papers_available == 0:
            issues.append(
                "No source papers available for validation. "
                "Answer could not be fact-checked."
            )
        
        # Check verification rate
        verified_claims = sum(1 for fc in fact_checks if fc.get("supported", False))
        total_claims = len(fact_checks)
        if total_claims > 0:
            verification_rate = verified_claims / total_claims
            if verification_rate < 0.5:
                issues.append(
                    f"Low verification rate ({verification_rate:.1%}). "
                    f"Only {verified_claims}/{total_claims} claims verified."
                )
        
        return issues
    
    def _assess_validity(
        self,
        fact_checks: List[Dict[str, Any]],
        hallucination_score: float,
        citation_verification: Dict[str, Any]
    ) -> bool:
        """
        Determine if research answer is valid overall.
        
        IMPROVED: Stricter logic. Valid = high verification AND low hallucination.
        """
        if not fact_checks:
            return True  # No claims to check = valid
        
        # Check verification rate
        supported_count = sum(1 for fc in fact_checks if fc["supported"])
        verification_rate = supported_count / len(fact_checks) if len(fact_checks) > 0 else 1.0
        
        # ✅ FIX: Stricter validation logic.
        # Must have a low hallucination score AND a decent verification rate.
        
        # A high hallucination score is an automatic fail.
        if hallucination_score > 0.4:
             self.log_debug(f"Validity assessment: FAILED (Hallucination score {hallucination_score:.2f} > 0.4)")
             return False
        
        # A very low verification rate is also a fail.
        if verification_rate < 0.5:
             self.log_debug(f"Validity assessment: FAILED (Verification rate {verification_rate:.2f} < 0.5)")
             return False
        
        # Passed both checks
        self.log_debug(
            f"Validity assessment: PASSED (Verification={verification_rate:.2f}, "
            f"Hallucination={hallucination_score:.2f})"
        )
        return True

    # ========================================================================
    # CITATION ADDITION
    # ========================================================================
    
    def _add_citations(
        self,
        answer: str,
        fact_checks: List[Dict[str, Any]],
        retrieval_contexts: List[Any]
    ) -> Tuple[str, List[str]]:
        """Add proper academic citations to the answer."""
        # Collect unique sources
        all_sources = set()
        for fc in fact_checks:
            if fc["supported"]:
                all_sources.update(fc["sources"])
        
        citations = sorted(list(all_sources))
        
        # For now, just return the original answer and citation list
        # Future: Could add inline citations where missing
        return answer, citations
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _create_empty_validation(self) -> Dict[str, Any]:
        """Create empty validation result."""
        return {
            "is_valid": False,
            "fact_checks": [],
            "hallucination_score": 1.0,
            "citation_verification": {
                "total_citations": 0,
                "verified_citations": 0,
                "citation_accuracy": 0.0
            },
            "citations": [],
            "issues": ["No answer to validate"],
            "original_answer": "",
            "validated_answer": "",
            "total_claims": 0,
            "verified_claims": 0,
            "unsupported_claims": 0,
            "papers_referenced": 0,
            "detailed_claims": [],
            "hallucination_examples": [],
            "verification_summary": {
                "total_claims_checked": 0,
                "verified_by_llm": 0,
                "verified_by_fuzzy": 0,
                "unverified": 0,
                "average_confidence": 0.0
            }
        }
    
    def _update_statistics(self, result: Dict[str, Any]):
        """Update validation statistics."""
        self.total_validations += 1
        self.total_claims_checked += result.get("total_claims", 0)
        self.total_hallucinations_detected += result.get("unsupported_claims", 0)
        
        hallucination_score = result.get("hallucination_score", 0.0)
        self.avg_hallucination_scores.append(hallucination_score)
    
    def _store_results_in_state(
        self,
        state: AgentState,
        result: Any,
        task: SubTask
    ):
        """Store validation results in state."""
        from dataclasses import asdict, is_dataclass
        
        if is_dataclass(result):
            result_dict = asdict(result)
        elif isinstance(result, dict):
            result_dict = result
        else:
            try:
                result_dict = dict(result)
            except:
                result_dict = {"error": "Could not convert result to dict"}
        
        state["validation_report"] = result_dict
        
        logger.info(
            f"[Validation Specialist] Stored validation report with "
            f"{result_dict.get('total_claims', 0)} claims checked"
        )
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation-specific statistics."""
        avg_hallucination = (
            sum(self.avg_hallucination_scores) / len(self.avg_hallucination_scores)
            if self.avg_hallucination_scores
            else 0.0
        )
        
        base_stats = self.get_performance_stats()
        
        return {
            **base_stats,
            "total_validations": self.total_validations,
            "total_claims_checked": self.total_claims_checked,
            "total_hallucinations_detected": self.total_hallucinations_detected,
            "total_citations_verified": self.total_citations_verified,
            "total_papers_referenced": self.total_papers_referenced,
            "avg_hallucination_score": avg_hallucination,
            "avg_claims_per_validation": (
                self.total_claims_checked / self.total_validations
                if self.total_validations > 0
                else 0.0
            ),
            "hallucination_detection_rate": (
                self.total_hallucinations_detected / self.total_claims_checked
                if self.total_claims_checked > 0
                else 0.0
            ),
            "avg_papers_per_validation": (
                self.total_papers_referenced / self.total_validations
                if self.total_validations > 0
                else 0.0
            ),
            "domain": "scientific_literature"
        }


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_validation_agent(model_name: Optional[str] = None) -> ValidationAgent:
    """
    Create and initialize a validation agent for scientific literature.
    
    Args:
        model_name: Optional LLM model name
    
    Returns:
        Initialized ValidationAgent
    """
    return ValidationAgent(model_name=model_name)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

async def example_usage():
    """Example showing validation agent in action for scientific literature."""
    from ..state import create_initial_state, add_subtask, RetrievalResult, SynthesisResult
    
    # Create state
    state = create_initial_state(
        query="What are Transformers?",
        session_id="test-session"
    )
    
    # Simulate synthesized answer
    state["synthesized_answer"] = {
        "answer": """Transformers are neural network architectures introduced by Vaswani et al. in 2017.
They use self-attention mechanisms to process sequential data without recurrence.
BERT, proposed by Devlin et al., achieved 92% accuracy on GLUE benchmark.
The Transformer model has 100 million parameters."""  # Last claim is made up!
    }
    
    # Simulate retrieval contexts
    state["retrieved_contexts"] = [
        {
            "chunks": [
                {
                    "content": "The Transformer architecture uses self-attention mechanisms.",
                    "document_title": "Attention Is All You Need",
                    "authors": ["Vaswani", "Shazeer"],
                    "venue": "NeurIPS",
                    "publication_date": "2017"
                }
            ]
        }
    ]
    
    # Create validation task
    task = add_subtask(
        state=state,
        task_type=TaskType.VALIDATION,
        description="Validate synthesized research answer",
        parameters={
            "query": state["query"],
            "validation_type": "fact_check"
        },
        assigned_agent=AgentRole.VALIDATION,
        priority=3
    )
    
    print(f"Created validation task: {task['task_id']}")
    
    # Create agent
    agent = create_validation_agent()
    
    # Execute task
    print("\nExecuting validation...")
    result = await agent.execute(state, task)
    
    print(f"\nResult:")
    print(f"  Success: {result['success']}")
    print(f"  Valid: {result['data']['is_valid']}")
    print(f"  Total claims: {result['data']['total_claims']}")
    print(f"  Verified: {result['data']['verified_claims']}")
    print(f"  Unsupported: {result['data']['unsupported_claims']}")
    print(f"  Hallucination score: {result['data']['hallucination_score']:.2f}")
    print(f"  Papers referenced: {result['data']['papers_referenced']}")
    
    # ✅ SHOW DETAILED CLAIMS
    print(f"\n📋 DETAILED CLAIMS:")
    for claim_detail in result['data']['detailed_claims']:
        print(f"\n  {claim_detail['status']} (confidence: {claim_detail['confidence']:.2f})")
        print(f"    Claim: {claim_detail['claim_text'][:80]}...")
        if claim_detail['supporting_sources']:
            print(f"    Sources: {', '.join(claim_detail['supporting_sources'][:2])}")
    
    # ✅ SHOW HALLUCINATIONS
    if result['data']['hallucination_examples']:
        print(f"\n🚨 HALLUCINATIONS DETECTED:")
        for halluc in result['data']['hallucination_examples']:
            print(f"  ✗ {halluc['claim_text']}")
            print(f"    Reason: {halluc['reason']}")
    
    # Show issues
    print(f"\n⚠️ ISSUES:")
    for issue in result['data']['issues']:
        print(f"  - {issue}")
    
    # Show agent statistics
    stats = agent.get_validation_statistics()
    print(f"\nAgent statistics:")
    print(f"  Total validations: {stats['total_validations']}")
    print(f"  Claims checked: {stats['total_claims_checked']}")
    print(f"  Hallucinations detected: {stats['total_hallucinations_detected']}")
    print(f"  Domain: {stats['domain']}")


if __name__ == "__main__":
    import asyncio
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    asyncio.run(example_usage())