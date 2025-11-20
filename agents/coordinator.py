"""
Coordinator Agent - Orchestrator for Multi-Agent System.

This agent orchestrates the entire multi-agent workflow:
- Receives user queries about research literature
- Decomposes queries into subtasks
- Assigns tasks to specialist agents
- Monitors execution
- Aggregates results
- Returns final answer with citations
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio

from .base_agent import BaseAgent, AgentRegistry
from .state import (
    AgentState,
    AgentRole,
    SubTask,
    TaskType,
    SubTaskStatus,
    create_initial_state,
    get_pending_tasks,
    get_ready_tasks,
    mark_task_completed,
    mark_task_failed,
    is_execution_complete
)
from .utils.task_decomposition import (
    TaskDecomposer,
    ExecutionPlanner,
    QueryAnalyzer
)

# Import specialist agents
from .specialists.retrieval_agent import RetrievalAgent
from .specialists.knowledge_agent import KnowledgeAgent
from .specialists.synthesis_agent import SynthesisAgent
from .specialists.validation_agent import ValidationAgent

logger = logging.getLogger(__name__)


# ============================================================================
# COORDINATOR AGENT
# ============================================================================

class CoordinatorAgent(BaseAgent):
    """
    Coordinator agent that orchestrates all specialist agents for scientific literature queries.
    
    Responsibilities:
    - Research query analysis and decomposition
    - Task assignment to specialists
    - Execution monitoring with paper retrieval
    - Result aggregation from multiple sources
    - Error handling and recovery
    - Performance tracking for research queries
    
    The coordinator is the entry point for all research queries and manages
    the entire multi-agent workflow from start to finish.
    """
    
    def __init__(
        self,
        retrieval_agent: Optional[RetrievalAgent] = None,
        knowledge_agent: Optional[KnowledgeAgent] = None,
        synthesis_agent: Optional[SynthesisAgent] = None,
        validation_agent: Optional[ValidationAgent] = None,
        model_name: Optional[str] = None
    ):
        """
        Initialize coordinator agent for scientific literature.
        
        Args:
            retrieval_agent: Retrieval specialist (creates new if not provided)
            knowledge_agent: Knowledge specialist (creates new if not provided)
            synthesis_agent: Synthesis specialist (creates new if not provided)
            validation_agent: Validation specialist (creates new if not provided)
            model_name: Optional LLM model name
        """
        super().__init__(
            role=AgentRole.COORDINATOR,
            name="Research Coordinator",
            description="Orchestrator that manages all specialist agents for scientific literature queries",
            model_name=model_name
        )
        
        # Initialize specialist agents
        self.retrieval_agent = retrieval_agent or RetrievalAgent()
        self.knowledge_agent = knowledge_agent or KnowledgeAgent()
        self.synthesis_agent = synthesis_agent or SynthesisAgent()
        self.validation_agent = validation_agent or ValidationAgent()
        
        # Task decomposer and planner
        self.analyzer = QueryAnalyzer(use_llm=True)
        self.decomposer = TaskDecomposer(analyzer=self.analyzer)
        self.planner = ExecutionPlanner()
        
        # Agent registry
        self.registry = AgentRegistry()
        self.registry.register(self.retrieval_agent)
        self.registry.register(self.knowledge_agent)
        self.registry.register(self.synthesis_agent)
        self.registry.register(self.validation_agent)
        
        # Coordinator statistics
        self.total_queries_processed = 0
        self.total_research_queries = 0
        self.total_papers_retrieved = 0
        self.total_tasks_created = 0
        self.total_tasks_completed = 0
        self.total_tasks_failed = 0
        self.avg_execution_time_ms = 0.0
        
        self.log_info(
            "Initialized for scientific literature with 4 specialist agents: "
            "Retrieval, Knowledge, Synthesis, Validation"
        )
    
    # ========================================================================
    # ABSTRACT METHOD IMPLEMENTATIONS
    # ========================================================================
    
    async def process_task(
        self,
        state: AgentState,
        task: SubTask
    ) -> Dict[str, Any]:
        """
        Coordinators don't process individual tasks - they orchestrate.
        This method should not be called directly.
        """
        raise NotImplementedError(
            "Coordinator doesn't process individual tasks. "
            "Use process_query() instead."
        )
    
    def can_handle_task(self, task: SubTask) -> bool:
        """Coordinator handles all tasks by delegating to specialists."""
        return True
    
    def get_system_prompt(self) -> str:
        """Get system prompt for coordinator."""
        return """You are the Research Coordinator Agent in a multi-agent system for scientific literature analysis.

Your role is to orchestrate multiple specialist agents:
- Retrieval Agent: Searches research papers and academic knowledge graph
- Knowledge Agent: Extracts entities (authors, methods, papers) and relationships (citations, authorship)
- Synthesis Agent: Generates comprehensive answers with proper academic citations
- Validation Agent: Fact-checks against papers and verifies citations

Your responsibilities:
1. Analyze research queries to understand intent and complexity
2. Break down complex queries into subtasks (e.g., "Compare X and Y" → retrieve papers on X, retrieve papers on Y, extract methods, synthesize comparison)
3. Assign subtasks to appropriate specialist agents
4. Monitor execution and handle failures
5. Aggregate results from all agents
6. Ensure the final answer is comprehensive, properly cited, and validated

You make strategic decisions about:
- Which agents to involve (always retrieval first, then knowledge for entities/relationships)
- Task execution order (parallel retrieval when possible, sequential synthesis/validation)
- How to handle failures (retry strategies, fallback plans)
- When synthesis and validation are needed (always for final answer)

For research queries:
- Prioritize paper retrieval and citation extraction
- Ensure proper attribution (authors, venues, years)
- Track which papers support which claims
- Validate against research literature"""
    
    # ========================================================================
    # MAIN QUERY PROCESSING
    # ========================================================================
    
    async def process_query(
        self,
        query: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        search_type: str = "hybrid",
        enable_validation: bool = True,
        retrieval_limit: int = 10
    ) -> Dict[str, Any]:
        """
        Process a research query through the multi-agent system.
        
        This is the main entry point for research query processing.
        
        Args:
            query: Research query
            session_id: Optional session ID
            user_id: Optional user ID
            search_type: Search type (vector, graph, hybrid)
            enable_validation: Whether to validate the answer
            retrieval_limit: Number of papers/chunks to retrieve
        
        Returns:
            Dict with final answer, citations, and metadata
        """
        import time
        start_time = time.time()
        
        self.log_info(f"Processing research query: '{query[:100]}...'")
        
        # Create initial state
        state = create_initial_state(
            query=query,
            session_id=session_id,
            user_id=user_id,
            search_type=search_type
        )
        
        # Add research-specific metadata
        state["metadata"]["retrieval_limit"] = retrieval_limit
        state["metadata"]["domain"] = "scientific_literature"
        
        try:
            # Phase 1: Query analysis and task decomposition
            self.log_info("Phase 1: Research query analysis and task decomposition")
            await self._decompose_query(state, search_type, retrieval_limit)
            
            # Phase 2: Execute tasks
            self.log_info("Phase 2: Executing specialist agents for paper retrieval")
            await self._execute_tasks(state)
            
            # Phase 3: Synthesis (always needed for research queries)
            self.log_info("Phase 3: Synthesizing answer with citations")
            await self._ensure_synthesis(state)
            
            # Phase 4: Validation (if enabled)
            if enable_validation:
                self.log_info("Phase 4: Validating answer against research literature")
                await self._ensure_validation(state)
            
            # Calculate total execution time
            end_time = time.time()
            total_latency_ms = (end_time - start_time) * 1000
            
            # Update state with completion time
            state["metadata"]["completed_at"] = datetime.now().isoformat()
            state["metadata"]["total_latency_ms"] = total_latency_ms
            
            # Update statistics
            self.total_queries_processed += 1
            self.total_research_queries += 1
            self._update_avg_execution_time(total_latency_ms)
            
            # Count papers retrieved
            papers_retrieved = self._count_papers_retrieved(state)
            self.total_papers_retrieved += papers_retrieved
            
            # Prepare final response
            response = self._prepare_response(state, total_latency_ms, papers_retrieved)
            
            self.log_info(
                f"Research query completed in {total_latency_ms:.1f}ms: "
                f"{len(state['completed_tasks'])} tasks completed, "
                f"{papers_retrieved} papers retrieved, "
                f"{len(state['failed_tasks'])} failed"
            )
            
            return response
        
        except Exception as e:
            self.log_error(f"Research query processing failed: {e}")
            
            # Create error response
            return self._create_error_response(state, str(e))
    
    # ========================================================================
    # TASK DECOMPOSITION
    # ========================================================================
    
    async def _decompose_query(
        self,
        state: AgentState,
        search_type: str,
        retrieval_limit: int
    ):
        """
        Decompose research query into subtasks.
        
        Args:
            state: Current state
            search_type: Search type preference
            retrieval_limit: Number of results to retrieve
        """
        # Use task decomposer with research context
        subtasks = await self.decomposer.decompose(
            state, 
            search_type,
            retrieval_limit=retrieval_limit
        )
        
        # Create execution plan
        execution_plan = self.planner.create_execution_plan(subtasks)
        state["execution_plan"] = execution_plan
        
        # Update statistics
        self.total_tasks_created += len(subtasks)
        
        # Log decision
        decision = {
            "action": "research_query_decomposition",
            "subtasks_created": len(subtasks),
            "execution_plan": execution_plan,
            "retrieval_limit": retrieval_limit,
            "timestamp": datetime.now().isoformat()
        }
        state["coordinator_decisions"].append(decision)
        
        self.log_info(
            f"Created {len(subtasks)} subtasks for research query: "
            f"{execution_plan['parallel_phase']['count']} parallel, "
            f"{len(execution_plan['sequential_phases'])} sequential phases"
        )
    
    # ========================================================================
    # TASK EXECUTION
    # ========================================================================
    
    async def _execute_tasks(self, state: AgentState):
        """
        Execute all subtasks using specialist agents.
        
        Args:
            state: Current state
        """
        execution_plan = state.get("execution_plan", {})
        
        # Phase 1: Execute parallel tasks (typically retrieval tasks)
        parallel_tasks = execution_plan.get("parallel_phase", {}).get("tasks", [])
        if parallel_tasks:
            self.log_info(
                f"Phase 1: Parallel execution of {len(parallel_tasks)} tasks "
                f"(paper retrieval and knowledge extraction)"
            )
            await self._execute_parallel_phase(state, parallel_tasks)
        
        # Phase 2+: Execute sequential phases
        sequential_phases = execution_plan.get("sequential_phases", [])
        for phase in sequential_phases:
            phase_tasks = phase.get("tasks", [])
            if phase_tasks:
                self.log_info(
                    f"Executing phase {phase['phase']}: {len(phase_tasks)} tasks "
                    f"(dependent on previous results)"
                )
                await self._execute_parallel_phase(state, phase_tasks)
    
    # Arquivo: coordinator.py

    async def _execute_parallel_phase(
        self,
        state: AgentState,
        task_ids: List[str]
    ):
        """
        Execute multiple tasks in parallel, ensuring correct search_type is used.
        
        ✅ FIXED: Correctly passes the task parameters without overwriting search_type.
        """
        # Get task objects from state
        tasks = [
            task for task in state["subtasks"]
            if task["task_id"] in task_ids
        ]
        
        if not tasks:
            return
        
        self.log_info(f"Executing {len(tasks)} tasks in parallel")
        
        # Create execution coroutines for each task
        task_coroutines = []
        for task in tasks:
            # Get appropriate agent for the task's role
            agent = self.registry.get_by_role(task["assigned_agent"])
            
            if agent:
                coro = self._execute_single_task(state, task, agent)
                task_coroutines.append(coro)
                
            else:
                self.log_error(f"No agent found for role: {task['assigned_agent']}")
                mark_task_failed(state, task["task_id"])
        
        # Execute all tasks concurrently and wait for results
        if task_coroutines:
            results = await asyncio.gather(*task_coroutines, return_exceptions=True)
            
            # Log how many tasks succeeded
            successful = sum(1 for r in results if not isinstance(r, Exception))
            self.log_info(f"Parallel phase complete: {successful}/{len(results)} tasks succeeded")
        
    async def _execute_single_task(
        self,
        state: AgentState,
        task: SubTask,
        agent: BaseAgent
    ):
        """
        Execute a single task with an agent.
        
        Args:
            state: Current state
            task: Task to execute
            agent: Agent to execute with
        """
        task_id = task["task_id"]
        
        try:
            # Execute task
            result = await agent.execute(state, task)
            
            # Check if successful
            if result["success"]:
                mark_task_completed(state, task_id)
                self.total_tasks_completed += 1
                
                # Store result in state with proper categorization
                if agent.role == AgentRole.RETRIEVAL:
                    state["retrieval_results"].append(result)
                    # Log paper count
                    if "data" in result and "unique_papers" in result["data"]:
                        papers = result["data"]["unique_papers"]
                        self.log_info(f"Retrieved {papers} unique papers")
                elif agent.role == AgentRole.KNOWLEDGE:
                    state["knowledge_results"].append(result)
                elif agent.role == AgentRole.SYNTHESIS:
                    state["synthesis_results"].append(result)
                elif agent.role == AgentRole.VALIDATION:
                    state["validation_results"].append(result)
            else:
                mark_task_failed(state, task_id)
                self.total_tasks_failed += 1
                self.log_warning(f"Task {task_id} failed: {result.get('error')}")
        
        except Exception as e:
            mark_task_failed(state, task_id)
            self.total_tasks_failed += 1
            self.log_error(f"Task {task_id} execution error: {e}")
            raise
    
    # ========================================================================
    # SYNTHESIS AND VALIDATION
    # ========================================================================
    
    async def _ensure_synthesis(self, state: AgentState):
        """
        Ensure synthesis has been performed with proper citations.
        
        Args:
            state: Current state
        """
        # Check if synthesis was already done
        if state.get("synthesized_answer"):
            self.log_info("Synthesis already completed")
            return
        
        # Check if there's a pending/completed synthesis task
        synthesis_tasks = [
            task for task in state["subtasks"]
            if task["task_type"] == TaskType.SYNTHESIS
        ]
        
        if synthesis_tasks:
            # Synthesis task exists but may have failed
            completed = [t for t in synthesis_tasks if t["status"] == SubTaskStatus.COMPLETED]
            if completed:
                self.log_info("Synthesis task completed")
                return
        
        # No synthesis - create one now
        self.log_info("Creating synthesis task for research answer")
        
        from .state import add_subtask
        
        synthesis_task = add_subtask(
            state=state,
            task_type=TaskType.SYNTHESIS,
            description="Synthesize final answer with academic citations",
            parameters={
                "query": state["query"],
                "synthesis_type": "comprehensive"
            },
            assigned_agent=AgentRole.SYNTHESIS,
            priority=5
        )
        
        # Execute synthesis
        result = await self.synthesis_agent.execute(state, synthesis_task)
        
        if result["success"]:
            mark_task_completed(state, synthesis_task["task_id"])
            state["synthesis_results"].append(result)
            
            # Log citation info
            if "data" in result and "papers_cited" in result["data"]:
                papers_cited = result["data"]["papers_cited"]
                self.log_info(f"Synthesis completed with {papers_cited} papers cited")
        else:
            mark_task_failed(state, synthesis_task["task_id"])
            self.log_error(f"Synthesis failed: {result.get('error')}")
    
    async def _ensure_validation(self, state: AgentState):
        """
        Ensure validation has been performed against research literature.
        
        Args:
            state: Current state
        """
        # Check if validation already done
        if state.get("validation_report"):
            self.log_info("Validation already completed")
            return
        
        # Check if there's a validation task
        validation_tasks = [
            task for task in state["subtasks"]
            if task["task_type"] == TaskType.VALIDATION
        ]
        
        if validation_tasks:
            completed = [t for t in validation_tasks if t["status"] == SubTaskStatus.COMPLETED]
            if completed:
                self.log_info("Validation task completed")
                return
        
        # No validation - create one now
        self.log_info("Creating validation task to verify against research papers")
        
        from .state import add_subtask
        
        validation_task = add_subtask(
            state=state,
            task_type=TaskType.VALIDATION,
            description="Validate final answer against research literature",
            parameters={
                "query": state["query"],
                "validation_type": "fact_check"
            },
            assigned_agent=AgentRole.VALIDATION,
            priority=3
        )
        
        # Execute validation
        result = await self.validation_agent.execute(state, validation_task)
        
        if result["success"]:
            mark_task_completed(state, validation_task["task_id"])
            state["validation_results"].append(result)
            
            # Log validation results
            if "data" in result:
                is_valid = result["data"].get("is_valid", False)
                hallucination_score = result["data"].get("hallucination_score", 0.0)
                papers_ref = result["data"].get("papers_referenced", 0)
                self.log_info(
                    f"Validation completed: valid={is_valid}, "
                    f"hallucination={hallucination_score:.2f}, "
                    f"papers_referenced={papers_ref}"
                )
        else:
            mark_task_failed(state, validation_task["task_id"])
            self.log_warning(f"Validation failed: {result.get('error')}")
    
    # ========================================================================
    # RESPONSE PREPARATION
    # ========================================================================
    
    def _count_papers_retrieved(self, state: AgentState) -> int:
        """
        Count unique papers retrieved across all retrieval results.
        
        Args:
            state: Current state
        
        Returns:
            Number of unique papers
        """
        unique_papers = set()
        
        for result in state.get("retrieval_results", []):
            if "data" in result and "unique_papers" in result["data"]:
                unique_papers.add(result["data"]["unique_papers"])
            
            # Also count from chunks
            if "data" in result and "chunks" in result["data"]:
                for chunk in result["data"]["chunks"]:
                    paper_id = chunk.get("document_id") or chunk.get("document_title")
                    if paper_id:
                        unique_papers.add(paper_id)
        
        return len(unique_papers)
    
    def _prepare_response(
        self,
        state: AgentState,
        total_latency_ms: float,
        papers_retrieved: int
    ) -> Dict[str, Any]:
        """
        Prepare final response from state.
        
        Args:
            state: Final state
            total_latency_ms: Total execution time
            papers_retrieved: Number of papers retrieved
        
        Returns:
            Response dictionary with citations
        """
        # Get final answer
        final_answer = state.get("final_answer", "")
        
        # If no final answer, try to get from synthesis
        if not final_answer:
            synthesized = state.get("synthesized_answer")
            if synthesized and isinstance(synthesized, dict):
                final_answer = synthesized.get("answer", "")
        
        # Get validation report
        validation_report = state.get("validation_report", {})
        
        # Get agent results summary
        agent_results = {
            "retrieval": len(state.get("retrieval_results", [])),
            "knowledge": len(state.get("knowledge_results", [])),
            "synthesis": len(state.get("synthesis_results", [])),
            "validation": len(state.get("validation_results", []))
        }
        
        # Prepare response
        response = {
            "answer": final_answer,
            "query": state["query"],
            "session_id": state["metadata"]["session_id"],
            "confidence": state.get("confidence", 0.5),
            "sources": state.get("sources", []),
            
            # Research-specific metadata
            "research_metadata": {
                "papers_retrieved": papers_retrieved,
                "papers_cited": validation_report.get("papers_referenced", 0) if validation_report else 0,
                "domain": "scientific_literature"
            },
            
            # Execution metadata
            "metadata": {
                "query_id": state["metadata"]["query_id"],
                "total_latency_ms": total_latency_ms,
                "tasks_created": len(state["subtasks"]),
                "tasks_completed": len(state["completed_tasks"]),
                "tasks_failed": len(state["failed_tasks"]),
                "agent_results": agent_results,
                "search_type": state["metadata"]["search_type"]
            },
            
            # Validation info
            "validation": {
                "is_valid": validation_report.get("is_valid", False) if validation_report else None,
                "hallucination_score": validation_report.get("hallucination_score", 0.0) if validation_report else None,
                "verified_claims": validation_report.get("verified_claims", 0) if validation_report else 0,
                "unsupported_claims": validation_report.get("unsupported_claims", 0) if validation_report else 0,
                "issues": validation_report.get("issues", []) if validation_report else []
            },
            
            # Agent performance
            "agent_performance": self._get_agent_performance()
        }
        
        return response
    
    def _create_error_response(
        self,
        state: AgentState,
        error_message: str
    ) -> Dict[str, Any]:
        """
        Create error response.
        
        Args:
            state: Current state
            error_message: Error message
        
        Returns:
            Error response dictionary
        """
        return {
            "answer": "",
            "query": state["query"],
            "session_id": state["metadata"]["session_id"],
            "confidence": 0.0,
            "sources": [],
            "error": error_message,
            "research_metadata": {
                "papers_retrieved": 0,
                "papers_cited": 0,
                "domain": "scientific_literature"
            },
            "metadata": {
                "query_id": state["metadata"]["query_id"],
                "tasks_created": len(state.get("subtasks", [])),
                "tasks_completed": len(state.get("completed_tasks", [])),
                "tasks_failed": len(state.get("failed_tasks", []))
            }
        }
    
    # ========================================================================
    # STATISTICS AND MONITORING
    # ========================================================================
    
    def _update_avg_execution_time(self, latency_ms: float):
        """Update average execution time."""
        if self.total_queries_processed == 1:
            self.avg_execution_time_ms = latency_ms
        else:
            self.avg_execution_time_ms = (
                (self.avg_execution_time_ms * (self.total_queries_processed - 1) + latency_ms)
                / self.total_queries_processed
            )
    
    def _get_agent_performance(self) -> Dict[str, Any]:
        """Get performance summary for all agents."""
        return self.registry.get_performance_summary()
    
    def get_coordinator_statistics(self) -> Dict[str, Any]:
        """
        Get coordinator-specific statistics.
        
        Returns:
            Dict with coordinator stats
        """
        base_stats = self.get_performance_stats()
        
        return {
            **base_stats,
            "total_queries_processed": self.total_queries_processed,
            "total_research_queries": self.total_research_queries,
            "total_papers_retrieved": self.total_papers_retrieved,
            "total_tasks_created": self.total_tasks_created,
            "total_tasks_completed": self.total_tasks_completed,
            "total_tasks_failed": self.total_tasks_failed,
            "avg_execution_time_ms": self.avg_execution_time_ms,
            "avg_papers_per_query": (
                self.total_papers_retrieved / self.total_research_queries
                if self.total_research_queries > 0
                else 0.0
            ),
            "task_completion_rate": (
                self.total_tasks_completed / self.total_tasks_created * 100
                if self.total_tasks_created > 0
                else 0.0
            ),
            "domain": "scientific_literature",
            "agent_performance": self._get_agent_performance()
        }
    
    def print_statistics(self):
        """Print comprehensive statistics for all agents."""
        stats = self.get_coordinator_statistics()
        
        print("\n" + "="*70)
        print("SCIENTIFIC LITERATURE MULTI-AGENT SYSTEM STATISTICS")
        print("="*70)
        
        print(f"\n📊 Research Coordinator Statistics:")
        print(f"  Research Queries Processed: {stats['total_research_queries']}")
        print(f"  Papers Retrieved: {stats['total_papers_retrieved']}")
        print(f"  Avg Papers per Query: {stats['avg_papers_per_query']:.1f}")
        print(f"  Tasks Created: {stats['total_tasks_created']}")
        print(f"  Tasks Completed: {stats['total_tasks_completed']}")
        print(f"  Tasks Failed: {stats['total_tasks_failed']}")
        print(f"  Task Completion Rate: {stats['task_completion_rate']:.1f}%")
        print(f"  Avg Execution Time: {stats['avg_execution_time_ms']:.1f}ms")
        print(f"  Domain: {stats['domain']}")
        
        print(f"\n🤖 Agent Performance:")
        for role, agent_stats in stats['agent_performance'].items():
            print(f"\n  {role.upper()}:")
            print(f"    Executions: {agent_stats['total_executions']}")
            print(f"    Success Rate: {agent_stats['success_rate']:.1f}%")
            print(f"    Avg Latency: {agent_stats['avg_latency_ms']:.1f}ms")
        
        print("\n" + "="*70 + "\n")


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_coordinator(
    use_validation: bool = True,
    model_name: Optional[str] = None
) -> CoordinatorAgent:
    """
    Create and initialize a coordinator for scientific literature with all specialist agents.
    
    Args:
        use_validation: Whether to enable validation agent
        model_name: Optional LLM model name
    
    Returns:
        Initialized CoordinatorAgent
    """
    # Create specialist agents
    retrieval_agent = RetrievalAgent(model_name=model_name)
    knowledge_agent = KnowledgeAgent(model_name=model_name)
    synthesis_agent = SynthesisAgent(model_name=model_name)
    validation_agent = ValidationAgent(model_name=model_name) if use_validation else None
    
    # Create coordinator
    coordinator = CoordinatorAgent(
        retrieval_agent=retrieval_agent,
        knowledge_agent=knowledge_agent,
        synthesis_agent=synthesis_agent,
        validation_agent=validation_agent,
        model_name=model_name
    )
    
    return coordinator


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

async def example_usage():
    """Example showing coordinator in action for scientific literature queries."""
    
    # Create coordinator
    coordinator = create_coordinator()
    
    print("="*70)
    print("SCIENTIFIC LITERATURE MULTI-AGENT SYSTEM - COORDINATOR EXAMPLE")
    print("="*70)
    
    # Example research queries
    queries = [
        "What is the Transformer architecture and who proposed it?",
        "Compare BERT and GPT-2 in terms of architecture and training approach",
        "When was ResNet introduced and what problem did it solve?",
        "What are the key innovations in Vision Transformers?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'='*70}")
        print(f"Research Query {i}/{len(queries)}: {query}")
        print(f"{'='*70}\n")
        
        # Process query
        response = await coordinator.process_query(
            query=query,
            search_type="hybrid",
            enable_validation=True,
            retrieval_limit=5
        )
        
        # Display response
        print(f"Answer: {response['answer'][:300]}...")
        print(f"\nResearch Metadata:")
        print(f"  Papers Retrieved: {response['research_metadata']['papers_retrieved']}")
        print(f"  Papers Cited: {response['research_metadata']['papers_cited']}")
        print(f"  Latency: {response['metadata']['total_latency_ms']:.1f}ms")
        print(f"  Tasks: {response['metadata']['tasks_completed']}/{response['metadata']['tasks_created']}")
        print(f"  Confidence: {response['confidence']:.2f}")
        print(f"  Sources: {len(response['sources'])}")
        
        if response['validation']['is_valid'] is not None:
            print(f"\nValidation:")
            print(f"  Valid: {response['validation']['is_valid']}")
            print(f"  Hallucination Score: {response['validation']['hallucination_score']:.2f}")
            print(f"  Verified Claims: {response['validation']['verified_claims']}")
            print(f"  Unsupported Claims: {response['validation']['unsupported_claims']}")
    
    # Print overall statistics
    coordinator.print_statistics()


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
