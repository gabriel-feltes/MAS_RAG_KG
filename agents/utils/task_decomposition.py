"""
Task decomposition utilities for Multi-Agent System.

This module analyzes research queries and breaks them down into subtasks
that can be distributed to specialist agents for scientific literature analysis.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import re
import json
import os

from ..state import (
    AgentState,
    SubTask,
    TaskType,
    AgentRole,
    add_subtask
)

# Import LLM for query analysis
try:
    from ...agent.providers import get_model
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from agent.providers import get_model

from pydantic_ai import Agent

logger = logging.getLogger(__name__)

# ============================================================================
# QUERY PREPROCESSING
# ============================================================================

def clean_query_for_retrieval(query: str) -> str:
    """
    Remove meta-instructions from query to improve semantic retrieval.
    
    Removes phrases like "based on the papers", "according to the documents",
    which don't help with semantic similarity search.
    
    Args:
        query: Raw user query
    
    Returns:
        Cleaned query optimized for semantic search
    
    Examples:
        >>> clean_query_for_retrieval("Based on the six papers, what do KGs offer to RAG?")
        "what do KGs offer to RAG?"
        
        >>> clean_query_for_retrieval("According to the documents, how does BERT work?")
        "how does BERT work?"
    """
    # Meta-instruction patterns to remove
    meta_patterns = [
        r'based on the (\w+\s+)?(?:six|five|four|three|two|\d+\s+)?papers?',
        r'according to the (\w+\s+)?papers?',
        r'from the (\w+\s+)?papers?',
        r'in the (\w+\s+)?papers?',
        r'using the (\w+\s+)?(?:papers?|documents?|articles?)',
        r'from the (?:papers?|documents?|articles?)',
        r'in the database',
        r'from the database',
        r'in this database',
        r'from this database',
        r'what papers',
        r'what documents',
        r'what articles',
        r'which papers',
        r'which documents',
    ]
    
    cleaned = query
    for pattern in meta_patterns:
        # Case-insensitive removal
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    
    # Clean up extra whitespace, commas, and punctuation at the start
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    cleaned = re.sub(r'^[,\s]+', '', cleaned)
    cleaned = cleaned.strip()
    
    # If we cleaned too much and query is empty, return original
    if not cleaned or len(cleaned) < 3:
        logger.warning(f"Query cleaning removed too much. Keeping original: {query}")
        return query
    
    if cleaned != query:
        logger.info(f"Cleaned query: '{query}' → '{cleaned}'")
    
    return cleaned

# ============================================================================
# QUERY TYPES
# ============================================================================

class QueryIntent(str, Enum):
    """Types of query intents for scientific literature."""
    FACTUAL = "factual"                    # "What is X?"
    COMPARISON = "comparison"              # "Compare X and Y"
    TEMPORAL = "temporal"                  # "When did X happen?"
    RELATIONAL = "relational"              # "How is X related to Y?"
    LIST_INVENTORY = "list_inventory"      # "List all papers"
    AGGREGATION = "aggregation"            # "Summarize X"
    CAUSAL = "causal"                      # "Why did X happen?"
    PROCEDURAL = "procedural"              # "How to do X?"
    EXPLORATORY = "exploratory"            # "Tell me about X"
    METHODOLOGICAL = "methodological"      # "How does method X work?"
    SURVEY = "survey"                      # "Survey of X"

class QueryComplexity(str, Enum):
    """Query complexity levels."""
    SIMPLE = "simple"          # Single fact lookup
    MODERATE = "moderate"      # Multiple facts, single domain
    COMPLEX = "complex"        # Multiple facts, multiple domains
    VERY_COMPLEX = "very_complex"  # Requires synthesis across many sources

# ============================================================================
# QUERY ANALYSIS
# ============================================================================

class QueryAnalyzer:
    """Analyzes research queries to determine intent, complexity, and required tasks."""
    
    def __init__(self, use_llm: bool = True):
        """
        Initialize query analyzer for scientific literature.
        
        Args:
            use_llm: Whether to use LLM for analysis (fallback to rules if False)
        """
        self.use_llm = use_llm
        
        # Keyword patterns for rule-based analysis
        self.intent_patterns = {
            QueryIntent.LIST_INVENTORY: [
                r'\blist all\b', r'\blist (?:the|my)\b', r'\bwhat papers\b',
                r'\bhow many papers\b', r'\bwhat articles\b', r'\bpapers are there\b',
                r'\barticles about\b', r'\bshow me papers\b', r'\bwhat documents\b'
            ],
            QueryIntent.FACTUAL: [
                r'\bwhat is\b', r'\bwhat are\b', r'\bdefine\b',
                r'\bexplain\b', r'\bdescribe\b'
            ],
            QueryIntent.COMPARISON: [
                r'\bcompare\b', r'\bvs\b', r'\bversus\b',
                r'\bdifference between\b', r'\bsimilarit(y|ies) between\b'
            ],
            QueryIntent.TEMPORAL: [
                r'\bwhen\b', r'\btimeline\b', r'\bhistory\b',
                r'\bover time\b', r'\bevolution\b', r'\bproposed in\b'
            ],
            QueryIntent.RELATIONAL: [
                r'\bhow .* related\b', r'\brelationship between\b',
                r'\bconnection between\b', r'\blink between\b',
                r'\bcites\b', r'\bcited by\b'
            ],
            QueryIntent.METHODOLOGICAL: [
                r'\bhow does .* work\b', r'\barchitecture of\b',
                r'\bmethod\b', r'\balgorithm\b', r'\bapproach\b'
            ],
            QueryIntent.SURVEY: [
                r'\bsurvey of\b', r'\breview of\b', r'\bstate of the art\b',
                r'\brecent work\b', r'\brecent advances\b'
            ],
            QueryIntent.AGGREGATION: [
                r'\bsummarize\b', r'\boverview\b', r'\bkey contributions\b',
                r'\bmain findings\b'
            ],
            QueryIntent.CAUSAL: [
                r'\bwhy\b', r'\bcause\b', r'\breason\b',
                r'\bexplain why\b', r'\bwhat motivated\b'
            ]
        }
        
        # Entity patterns for scientific literature
        self.entity_patterns = {
            'authors': r'\b(?:Vaswani|Devlin|Brown|Radford|He|LeCun|Hinton|Bengio)\b',
            'methods': r'\b(?:Transformer|BERT|GPT|ResNet|LSTM|CNN|GAN|VAE)\b',
            'venues': r'\b(?:NeurIPS|ICML|ICLR|ACL|CVPR|ICCV|AAAI)\b',
            'tasks': r'\b(?:translation|classification|generation|detection|segmentation)\b'
        }
    
    async def analyze(self, query: str) -> Dict[str, Any]:
        """
        Analyze a research query to determine intent, complexity, and entities.
        
        Args:
            query: Research query
        
        Returns:
            Analysis results
        """
        logger.info(f"Analyzing research query: {query[:100]}...")
        
        if self.use_llm:
            analysis = await self._analyze_with_llm(query)
        else:
            analysis = self._analyze_with_rules(query)
        
        logger.info(
            f"Query analysis: intent={analysis['intent']}, "
            f"complexity={analysis['complexity']}"
        )
        
        return analysis
    
    async def _analyze_with_llm(self, query: str) -> Dict[str, Any]:
        """Use LLM to analyze research query."""
        prompt = f"""Analyze the following scientific literature query and provide:
1. Intent (factual, comparison, temporal, relational, list_inventory, aggregation, causal, procedural, exploratory, methodological, or survey)
2. Complexity (simple, moderate, complex, or very_complex)
3. Key entities mentioned (authors, methods, venues, tasks)
4. Whether it requires multiple papers
5. Whether it requires synthesis/comparison

Query: "{query}"

Use 'list_inventory' for questions asking "what papers/articles are there?", "list all papers", or "how many papers".
Use 'methodological' for questions about how methods/algorithms work.
Use 'survey' for questions asking for overview of research area.
Use 'comparison' for comparing different methods/approaches.
Use 'temporal' for questions about when methods were proposed or research evolution.

You MUST respond with a valid JSON object. Your response must be ONLY the JSON,
with no other text before or after it.

Example JSON response format:
{{
    "intent": "...",
    "complexity": "...",
    "entities": {{"authors": [...], "methods": [...], "venues": [...], "tasks": [...]}},
    "requires_multiple_papers": true/false,
    "requires_synthesis": true/false,
    "key_concepts": [...]
}}"""
        
        try:
            # Configure for Groq
            import os
            model_name = os.getenv("LLM_CHOICE", "llama3-groq-70b-8192-tool-use-preview")
            
            # Add groq prefix if not present
            if ":" not in model_name:
                if "gpt-oss" in model_name or "compound" in model_name:
                    model_name = f"groq:{model_name}"
                elif model_name.startswith("llama"):
                    model_name = f"groq:{model_name}"
                elif model_name.startswith("gpt"):
                    model_name = f"openai:{model_name}"
            
            # Prepare client config for Groq
            client_config = {}
            if model_name.startswith("groq:"):
                client_config = {
                    "base_url": os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
                    "api_key": os.getenv("LLM_API_KEY")
                }
            
            # Create agent
            agent = Agent(model_name, **client_config)
            result = await agent.run(prompt)
            
            # Parse LLM response
            analysis_str = str(result.data).strip()
            
            # Handle potential markdown code blocks like ```json ... ```
            if "```" in analysis_str:
                # remove wrapping code fences if present (take content between first and last fence)
                start_idx = analysis_str.find("```")
                end_idx = analysis_str.rfind("```")
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    analysis_str = analysis_str[start_idx + 3:end_idx].strip()
            
            # As a safety net, strip a single pair of leading/trailing fences if still present
            if analysis_str.startswith("```") and analysis_str.endswith("```"):
                analysis_str = analysis_str[3:-3].strip()
            
            # Parse JSON
            analysis = json.loads(analysis_str)
            
            # Validate intent
            valid_intents = [e.value for e in QueryIntent]
            if "intent" in analysis and analysis["intent"] not in valid_intents:
                logger.warning(
                    f"LLM returned unknown intent: {analysis['intent']}. "
                    f"Defaulting to exploratory."
                )
                analysis["intent"] = QueryIntent.EXPLORATORY
            
            logger.info(f"LLM analysis successful: intent={analysis.get('intent')}, complexity={analysis.get('complexity')}")
            
            return analysis
        
        except json.JSONDecodeError as e:
            logger.warning(f"LLM returned invalid JSON: {e}. Response: {analysis_str[:200]}...")
            logger.warning("Falling back to rule-based analysis")
            return self._analyze_with_rules(query)
        
        except Exception as e:
            logger.warning(f"LLM analysis failed: {e}. Falling back to rules")
            return self._analyze_with_rules(query)

    def _analyze_with_rules(self, query: str) -> Dict[str, Any]:
        """Use rule-based analysis as fallback."""
        query_lower = query.lower()
        
        # Determine intent
        intent = QueryIntent.EXPLORATORY
        priority_intents = [
            QueryIntent.LIST_INVENTORY,
            QueryIntent.COMPARISON,
            QueryIntent.METHODOLOGICAL,
            QueryIntent.SURVEY,
            QueryIntent.TEMPORAL
        ]
        
        for intent_type in priority_intents:
            patterns = self.intent_patterns.get(intent_type, [])
            if any(re.search(pattern, query_lower) for pattern in patterns):
                intent = intent_type
                break
        
        if intent == QueryIntent.EXPLORATORY:
            for intent_type, patterns in self.intent_patterns.items():
                if intent_type in priority_intents:
                    continue
                if any(re.search(pattern, query_lower) for pattern in patterns):
                    intent = intent_type
                    break
        
        # Determine complexity
        word_count = len(query.split())
        has_multiple_entities = sum(
            len(re.findall(pattern, query))
            for pattern in self.entity_patterns.values()
        ) > 2
        
        if word_count < 10 and not has_multiple_entities:
            complexity = QueryComplexity.SIMPLE
        elif word_count < 20 and not has_multiple_entities:
            complexity = QueryComplexity.MODERATE
        elif word_count < 30 or has_multiple_entities:
            complexity = QueryComplexity.COMPLEX
        else:
            complexity = QueryComplexity.VERY_COMPLEX
        
        # Extract entities
        entities = {}
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, query, re.IGNORECASE)
            entities[entity_type] = list(set(matches))
        
        # Determine requirements
        requires_multiple_papers = (
            intent in [QueryIntent.COMPARISON, QueryIntent.SURVEY, QueryIntent.AGGREGATION] or
            has_multiple_entities
        )
        
        requires_synthesis = (
            intent in [
                QueryIntent.COMPARISON, QueryIntent.CAUSAL, 
                QueryIntent.EXPLORATORY, QueryIntent.SURVEY
            ] or
            complexity in [QueryComplexity.COMPLEX, QueryComplexity.VERY_COMPLEX]
        )
        
        return {
            "intent": intent,
            "complexity": complexity,
            "entities": entities,
            "requires_multiple_papers": requires_multiple_papers,
            "requires_synthesis": requires_synthesis,
            "key_concepts": list(set(sum(entities.values(), [])))
        }

# ============================================================================
# TASK DECOMPOSER
# ============================================================================

class TaskDecomposer:
    """Decomposes research queries into subtasks for specialist agents."""
    
    def __init__(self, analyzer: Optional[QueryAnalyzer] = None):
        """
        Initialize task decomposer for scientific literature.
        
        Args:
            analyzer: Query analyzer instance
        """
        self.analyzer = analyzer or QueryAnalyzer(use_llm=True)
    
    async def decompose(
        self,
        state: AgentState,
        search_type: str = "hybrid",
        retrieval_limit: int = 10
    ) -> List[SubTask]:
        """
        Decompose research query into subtasks.
        
        Args:
            state: Current state with query
            search_type: Preferred search type
            retrieval_limit: Number of papers to retrieve
        
        Returns:
            List of created subtasks
        """
        query = state["query"]
        
        logger.info(f"Decomposing research query: {query}")
        
        # Analyze query
        analysis = await self.analyzer.analyze(query)
        
        # Store analysis in state
        if "execution_plan" not in state or not state["execution_plan"]:
            state["execution_plan"] = {}
        state["execution_plan"]["query_analysis"] = analysis
        
        # Create subtasks based on analysis
        subtasks = await self._create_subtasks(
            state=state,
            query=query,
            analysis=analysis,
            search_type=search_type,
            retrieval_limit=retrieval_limit
        )
        
        logger.info(f"Created {len(subtasks)} subtasks for research query")
        
        return subtasks
    
    async def _create_subtasks(
        self,
        state: AgentState,
        query: str,
        analysis: Dict[str, Any],
        search_type: str,
        retrieval_limit: int
    ) -> List[SubTask]:
        """Create subtasks based on research query analysis."""
        subtasks = []
        
        intent = analysis.get("intent", QueryIntent.EXPLORATORY)
        complexity = analysis.get("complexity", QueryComplexity.MODERATE)
        entities = analysis.get("entities", {})
        
        retrieval_tasks = []
        kg_tasks = []

        # If inventory query, create only list documents task
        if intent == QueryIntent.LIST_INVENTORY:
            list_task = self._create_list_documents_task(state, retrieval_limit)
            subtasks.append(list_task)
            retrieval_tasks.append(list_task)
        
        else:
            # Task 1: Paper retrieval (always needed for other intents)
            retrieval_tasks = self._create_retrieval_tasks(
                state, query, search_type, entities, intent, retrieval_limit
            )
            subtasks.extend(retrieval_tasks)
            
            # Task 2: Knowledge graph tasks (if entities present AND not vector-only search)
            # ✅ FIX: Do not run knowledge tasks if the user requested a pure vector search
            if any(entities.values()) and search_type != "vector": 
                kg_tasks = self._create_knowledge_tasks(
                    state, entities, intent, complexity
                )
                subtasks.extend(kg_tasks)
        
        # Task 3: Synthesis (always needed)
        synthesis_task = self._create_synthesis_task(
            state, query, intent, retrieval_tasks, kg_tasks
        )
        subtasks.append(synthesis_task)
        
        # Task 4: Validation (for complex queries)
        if complexity in [QueryComplexity.COMPLEX, QueryComplexity.VERY_COMPLEX]:
            validation_task = self._create_validation_task(
                state, query, [synthesis_task]
            )
            subtasks.append(validation_task)
        
        return subtasks
    
    def _create_list_documents_task(
        self,
        state: AgentState,
        limit: int
    ) -> SubTask:
        """Create task to list papers/documents."""
        list_task = add_subtask(
            state=state,
            task_type=TaskType.LIST_DOCUMENTS,
            description="List all research papers in the database",
            parameters={"limit": limit},
            assigned_agent=AgentRole.RETRIEVAL,
            priority=10
        )
        return list_task

    def _create_retrieval_tasks(
        self,
        state: AgentState,
        query: str,
        search_type: str,
        entities: Dict[str, List[str]],
        intent: QueryIntent,
        limit: int
    ) -> List[SubTask]:
        """Create paper retrieval subtasks with query cleaning."""
        tasks = []
        search_config = state.get("search_config", {})

        # Get search parameters
        center_node_distance = search_config.get("center_node_distance", 2)
        use_hybrid_search = search_config.get("use_hybrid_search", True)
        text_weight = search_config.get("text_weight", 0.3)
        
        # ✅ CLEAN QUERY before retrieval
        cleaned_query = clean_query_for_retrieval(query)
        
        # Determine task type
        # ✅ FIX: Differentiate all three search types
        if search_type == "vector":
            task_type = TaskType.VECTOR_SEARCH
        elif search_type == "graph":
            task_type = TaskType.GRAPH_SEARCH  # <-- Was incorrectly set to HYBRID_SEARCH
        else:
            task_type = TaskType.HYBRID_SEARCH
        
        # Primary retrieval task with cleaned query
        main_retrieval = add_subtask(
            state=state,
            task_type=task_type,
            description=f"Search for research papers about: {cleaned_query}",
            parameters={
                "query": cleaned_query,  # ← Use cleaned query for retrieval
                "original_query": query,  # ← Keep original for reference
                "limit": limit,
                "search_type": search_type,
                "center_node_distance": center_node_distance,
                "use_hybrid_search": use_hybrid_search,
                "text_weight": text_weight
            },
            assigned_agent=AgentRole.RETRIEVAL,
            priority=10
        )
        tasks.append(main_retrieval)
        
        # Additional retrieval for comparison queries
        if intent == QueryIntent.COMPARISON:
            subjects = self._extract_comparison_subjects(query)
            
            for subject in subjects:
                task = add_subtask(
                    state=state,
                    task_type=task_type,
                    description=f"Search for papers about: {subject}",
                    parameters={
                        "query": subject,
                        "limit": limit // 2,
                        "search_type": search_type,
                        "center_node_distance": center_node_distance,
                        "use_hybrid_search": use_hybrid_search,
                        "text_weight": text_weight
                    },
                    assigned_agent=AgentRole.RETRIEVAL,
                    priority=9
                )
                tasks.append(task)
        
        return tasks
    
    def _create_knowledge_tasks(
        self,
        state: AgentState,
        entities: Dict[str, List[str]],
        intent: QueryIntent,
        complexity: QueryComplexity
    ) -> List[SubTask]:
        """Create knowledge graph subtasks for academic entities."""
        tasks = []
        search_config = state.get("search_config", {})
        depth = search_config.get("depth", 2)
        
        # Entity extraction task
        if entities:
            entity_task = add_subtask(
                state=state,
                task_type=TaskType.ENTITY_EXTRACTION,
                description="Extract and enrich academic entities from query",
                parameters={
                    "entities": entities,
                    "depth": depth
                },
                assigned_agent=AgentRole.KNOWLEDGE,
                priority=8
            )
            tasks.append(entity_task)
        
        # Relationship finding for relational queries
        if intent in [QueryIntent.RELATIONAL, QueryIntent.COMPARISON]:
            for entity_type, entity_list in entities.items():
                for entity in entity_list[:3]:
                    rel_task = add_subtask(
                        state=state,
                        task_type=TaskType.RELATIONSHIP_FINDING,
                        description=f"Find relationships for: {entity}",
                        parameters={
                            "entity_name": entity,
                            "depth": depth
                        },
                        assigned_agent=AgentRole.KNOWLEDGE,
                        priority=7
                    )
                    tasks.append(rel_task)
        
        # Temporal reasoning for temporal queries
        if intent == QueryIntent.TEMPORAL:
            for entity_type, entity_list in entities.items():
                for entity in entity_list[:2]:
                    temp_task = add_subtask(
                        state=state,
                        task_type=TaskType.TEMPORAL_REASONING,
                        description=f"Get timeline for: {entity}",
                        parameters={
                            "entity_name": entity
                        },
                        assigned_agent=AgentRole.KNOWLEDGE,
                        priority=7
                    )
                    tasks.append(temp_task)
        
        return tasks
    
    def _create_synthesis_task(
        self,
        state: AgentState,
        query: str,
        intent: QueryIntent,
        retrieval_tasks: List[SubTask],
        knowledge_tasks: List[SubTask]
    ) -> SubTask:
        """Create synthesis subtask."""
        dependencies = [task["task_id"] for task in retrieval_tasks]
        dependencies.extend([task["task_id"] for task in knowledge_tasks])
        
        synthesis_task = add_subtask(
            state=state,
            task_type=TaskType.SYNTHESIS,
            description=f"Synthesize answer with citations for: {query}",
            parameters={
                "query": query,
                "intent": intent,
                "synthesis_type": "comprehensive"
            },
            assigned_agent=AgentRole.SYNTHESIS,
            priority=5,
            dependencies=dependencies
        )
        
        return synthesis_task
    
    def _create_validation_task(
        self,
        state: AgentState,
        query: str,
        synthesis_tasks: List[SubTask]
    ) -> SubTask:
        """Create validation subtask."""
        dependencies = [task["task_id"] for task in synthesis_tasks]
        
        validation_task = add_subtask(
            state=state,
            task_type=TaskType.VALIDATION,
            description=f"Validate answer against research literature for: {query}",
            parameters={
                "query": query,
                "validation_type": "fact_check"
            },
            assigned_agent=AgentRole.VALIDATION,
            priority=3,
            dependencies=dependencies
        )
        
        return validation_task
    
    def _extract_comparison_subjects(self, query: str) -> List[str]:
        """
        Extract the subjects being compared from a comparison query.
        
        Args:
            query: Comparison query
        
        Returns:
            List of subjects (e.g., ["vector RAG", "graph RAG"])
        """
        # ✅ FIX: Properly escape regex patterns
        patterns = [
            r'compare\s+(.+?)\s+(?:and|vs\.?|versus)\s+(.+?)(?:\s|$)',
            r'difference\s+between\s+(.+?)\s+and\s+(.+?)(?:\s|$)',
            r'(.+?)\s+vs\.?\s+(.+?)(?:\s|$)',
            r'contrast\s+(.+?)\s+(?:and|with)\s+(.+?)(?:\s|$)'
        ]
        
        subjects = []
        
        for pattern in patterns:
            try:
                matches = re.findall(pattern, query, re.IGNORECASE)
                
                if matches:
                    # Extract subjects from first match
                    if isinstance(matches[0], tuple):
                        subjects.extend([s.strip() for s in matches[0] if s.strip()])
                    else:
                        subjects.append(matches[0].strip())
                    
                    # Stop after first successful match
                    break
            
            except re.error as e:
                logger.warning(f"Regex pattern failed: {pattern} - {e}")
                continue
        
        # ✅ Fallback: If no pattern matched, return generic subjects
        if not subjects:
            logger.warning(f"Could not extract comparison subjects from: {query}")
            
            # Try simple word extraction as fallback
            query_lower = query.lower()
            
            if 'rag' in query_lower:
                # Common RAG comparison
                if 'vector' in query_lower and 'graph' in query_lower:
                    subjects = ['vector RAG', 'graph RAG']
                elif 'single' in query_lower and 'multi' in query_lower:
                    subjects = ['single-agent', 'multi-agent']
            
            if not subjects:
                # Last resort: use query itself
                subjects = ['approach A', 'approach B']
                logger.warning(f"Using fallback subjects: {subjects}")
        
        logger.info(f"Extracted comparison subjects: {subjects}")
        return subjects

# ============================================================================
# EXECUTION PLANNER
# ============================================================================

class ExecutionPlanner:
    """Plans the execution order of subtasks."""
    
    @staticmethod
    def create_execution_plan(subtasks: List[SubTask]) -> Dict[str, Any]:
        """Create execution plan for subtasks."""
        no_deps = [task for task in subtasks if not task["dependencies"]]
        with_deps = [task for task in subtasks if task["dependencies"]]
        
        no_deps.sort(key=lambda t: t["priority"], reverse=True)
        with_deps.sort(key=lambda t: t["priority"], reverse=True)
        
        plan = {
            "total_tasks": len(subtasks),
            "parallel_phase": {
                "tasks": [task["task_id"] for task in no_deps],
                "count": len(no_deps)
            },
            "sequential_phases": [],
            "estimated_latency_ms": ExecutionPlanner._estimate_latency(subtasks)
        }
        
        remaining = with_deps.copy()
        completed_ids = set(task["task_id"] for task in no_deps)
        phase_num = 1
        
        while remaining:
            ready = [
                task for task in remaining
                if all(dep in completed_ids for dep in task["dependencies"])
            ]
            
            if not ready:
                logger.warning(f"Cannot resolve dependencies for {len(remaining)} tasks")
                break
            
            plan["sequential_phases"].append({
                "phase": phase_num,
                "tasks": [task["task_id"] for task in ready],
                "count": len(ready)
            })
            
            for task in ready:
                completed_ids.add(task["task_id"])
                remaining.remove(task)
            
            phase_num += 1
        
        return plan
    
    @staticmethod
    def _estimate_latency(subtasks: List[SubTask]) -> float:
        """Estimate total execution latency."""
        latency_estimates = {
            TaskType.VECTOR_SEARCH: 500,
            TaskType.GRAPH_SEARCH: 800,
            TaskType.HYBRID_SEARCH: 600,
            TaskType.LIST_DOCUMENTS: 300,
            TaskType.GET_DOCUMENT: 200,
            TaskType.ENTITY_EXTRACTION: 300,
            TaskType.RELATIONSHIP_FINDING: 1000,
            TaskType.TEMPORAL_REASONING: 800,
            TaskType.SYNTHESIS: 2000,
            TaskType.VALIDATION: 1500
        }
        
        total = sum(
            latency_estimates.get(TaskType(task["task_type"]), 500)
            for task in subtasks
            if task.get("task_type")
        )
        
        return total * 0.6

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

async def decompose_query(
    state: AgentState,
    search_type: str = "hybrid",
    retrieval_limit: int = 10,
    use_llm_analysis: bool = True
) -> Tuple[List[SubTask], Dict[str, Any]]:
    """Convenience function to decompose a research query."""
    analyzer = QueryAnalyzer(use_llm=use_llm_analysis)
    decomposer = TaskDecomposer(analyzer=analyzer)
    
    subtasks = await decomposer.decompose(state, search_type, retrieval_limit)
    
    planner = ExecutionPlanner()
    execution_plan = planner.create_execution_plan(subtasks)
    
    state["execution_plan"] = execution_plan
    
    return subtasks, execution_plan

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

async def example_usage():
    """Example showing task decomposition for scientific literature queries."""
    from ..state import create_initial_state
    
    state = create_initial_state(
        query="Based on the six papers, what do knowledge graphs offer to RAG?",
        session_id="test-session",
        search_type="hybrid"
    )
    
    print(f"Original Query: {state['query']}")
    print(f"Cleaned Query: {clean_query_for_retrieval(state['query'])}\n")
    
    subtasks, plan = await decompose_query(state, search_type="hybrid", retrieval_limit=5)
    
    print(f"Created {len(subtasks)} subtasks:")
    for task in subtasks:
        print(f"\n  Task: {task['task_id']}")
        print(f"    Type: {task['task_type']}")
        print(f"    Description: {task['description']}")
        if 'query' in task['parameters']:
            print(f"    Query: {task['parameters']['query']}")

if __name__ == "__main__":
    import asyncio
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    asyncio.run(example_usage())
