"""
LangGraph workflow definition for Multi-Agent System.

This module defines the LangGraph StateGraph that orchestrates
all specialist agents for scientific literature analysis in a declarative workflow.
"""

import logging
from typing import Dict, Any, List, Optional, Literal
from datetime import datetime
from uuid import uuid4

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .state import (
    AgentState,
    AgentRole,
    SubTask,
    TaskType,
    SubTaskStatus,
    get_ready_tasks,
    mark_task_completed,
    mark_task_failed,
    is_execution_complete,
    create_initial_state
)
from .specialists.retrieval_agent import RetrievalAgent
from .specialists.knowledge_agent import KnowledgeAgent
from .specialists.synthesis_agent import SynthesisAgent
from .specialists.validation_agent import ValidationAgent
from .utils.task_decomposition import TaskDecomposer, ExecutionPlanner, QueryAnalyzer
from .utils.result_aggregation import aggregate_all_results

logger = logging.getLogger(__name__)


# ============================================================================
# GRAPH NODE FUNCTIONS
# ============================================================================

class MASGraphNodes:
    """Node functions for the Multi-Agent System graph for scientific literature."""
    
    def __init__(
        self,
        retrieval_agent: Optional[RetrievalAgent] = None,
        knowledge_agent: Optional[KnowledgeAgent] = None,
        synthesis_agent: Optional[SynthesisAgent] = None,
        validation_agent: Optional[ValidationAgent] = None
    ):
        """
        Initialize graph nodes with specialist agents for scientific literature.
        
        Args:
            retrieval_agent: Retrieval specialist for papers
            knowledge_agent: Knowledge specialist for academic entities
            synthesis_agent: Synthesis specialist for research answers
            validation_agent: Validation specialist for citation verification
        """
        self.retrieval_agent = retrieval_agent or RetrievalAgent()
        self.knowledge_agent = knowledge_agent or KnowledgeAgent()
        self.synthesis_agent = synthesis_agent or SynthesisAgent()
        self.validation_agent = validation_agent or ValidationAgent()
        
        # Task decomposer
        self.analyzer = QueryAnalyzer(use_llm=True)
        self.decomposer = TaskDecomposer(analyzer=self.analyzer)
        self.planner = ExecutionPlanner()
        
        logger.info("MASGraphNodes initialized for scientific literature analysis")
    
    # ========================================================================
    # COORDINATOR NODE
    # ========================================================================
    
    async def coordinator_node(self, state: AgentState) -> AgentState:
        """
        Coordinator node - analyzes query and creates subtasks for all specialist agents.
        """
        logger.info("🎯 COORDINATOR: Analyzing research query and creating subtasks")
        
        try:
            # ✅ FIX: Retrieve search_type and limit from state
            search_type = state.get("search_type", "hybrid")
            retrieval_limit = state.get("metadata", {}).get("retrieval_limit", 10)
            
            # Decompose query into subtasks
            subtasks = await self.decomposer.decompose(
                state,
                search_type=search_type,       # <-- PASS THE VALUE
                retrieval_limit=retrieval_limit  # <-- PASS THE VALUE
            )
            
            # ✅ ALWAYS CREATE SYNTHESIS TASK (unconditional)
            logger.info("⚠️ Force-creating synthesis task (unconditional)")
            
            # Create synthesis task
            from .state import add_subtask
            synthesis_task = add_subtask(
                state=state,
                task_type=TaskType.SYNTHESIS,
                description="Synthesize comprehensive answer from all retrieved research papers",
                parameters={
                    "query": state["query"],
                    "search_type": state.get("search_type", "hybrid")
                },
                assigned_agent=AgentRole.SYNTHESIS,
                priority=5,
                dependencies=[]  # ✅ NO DEPENDENCIES - will run immediately after retrieval completes
            )
            
            # ✅ FIX: Adiciona a nova tarefa de síntese à lista de subtarefas
            subtasks.append(synthesis_task)
            logger.info(f"✅ Force-created synthesis task: {synthesis_task['task_id']}")
            
            # ✅ FIX: Substitui (assigns) a lista de subtarefas em vez de estender (extend)
            # Isso evita que tarefas duplicadas sejam adicionadas se o nó for re-executado.
            state["subtasks"] = subtasks 
            
            # Count parallel vs sequential
            parallel_count = sum(1 for t in subtasks if not t.get("dependencies"))
            sequential_count = len(subtasks) - parallel_count
            
            logger.info(
                f"✓ Created {len(subtasks)} subtasks for research query: "
                f"{parallel_count} parallel (paper retrieval), "
                f"{sequential_count} sequential phases"
            )
            
            return state
        
        except Exception as e:
            logger.error(f"✗ Coordinator failed: {e}", exc_info=True)
            state["error"] = str(e)
            return state
    
    # ========================================================================
    # SPECIALIST AGENT NODES
    # ========================================================================

    async def retrieval_node(self, state: AgentState) -> AgentState:
        """Retrieval agent node - executes paper search tasks."""
        logger.info("🔍 RETRIEVAL AGENT: Executing paper search tasks")
        
        ready_tasks = get_ready_tasks(state)
        
        retrieval_tasks = [
            task for task in ready_tasks
            if (task.get("task_type") in ["vector_search", "graph_search", "hybrid_search", "retrieval"]
                or task.get("agent_role") in ["retrieval", AgentRole.RETRIEVAL.value])
        ]
        
        if not retrieval_tasks:
            logger.info("  No retrieval tasks ready")
            return state
        
        logger.info(f"  Processing {len(retrieval_tasks)} paper retrieval tasks")

        # Execute tasks
        for task in retrieval_tasks:
            try:
                result = await self.retrieval_agent.execute(state, task)
                
                if result.get("success"):
                    mark_task_completed(state, task["task_id"])
                    
                    # Ensure lists exist
                    if "retrieval_results" not in state:
                        state["retrieval_results"] = []
                    state["retrieval_results"].append(result)
                    
                    if "retrieved_contexts" not in state:
                        state["retrieved_contexts"] = []
                    state["retrieved_contexts"].append(result['data'])
                    
                    # Log paper count
                    unique_papers = result.get('data', {}).get('unique_papers', 0)
                    logger.info(
                        f"  ✓ Task {task['task_id']} completed: "
                        f"{unique_papers} unique papers retrieved"
                    )
                else:
                    mark_task_failed(state, task["task_id"], result.get("error"))
                    logger.warning(f"  ✗ Task {task['task_id']} failed: {result.get('error')}")
            
            except Exception as e:
                mark_task_failed(state, task["task_id"], str(e))
                logger.error(f"  ✗ Task {task['task_id']} error: {e}", exc_info=True)
        
        return state

    async def knowledge_node(self, state: AgentState) -> AgentState:
        """Knowledge agent node - executes academic entity extraction tasks."""
        logger.info("📊 KNOWLEDGE AGENT: Executing academic entity extraction tasks")
        
        ready_tasks = get_ready_tasks(state)
        
        knowledge_tasks = [
            task for task in ready_tasks
            if (task.get("task_type") in ["entity_extraction", "relationship_finding", "temporal_reasoning", "knowledge_extraction"]
                or task.get("agent_role") in ["knowledge", AgentRole.KNOWLEDGE.value])
        ]
        
        if not knowledge_tasks:
            logger.info("  No knowledge tasks ready")
            return state
        
        logger.info(f"  Processing {len(knowledge_tasks)} academic entity tasks")
        
        # Execute tasks
        for task in knowledge_tasks:
            try:
                result = await self.knowledge_agent.execute(state, task)
                
                if result.get("success"):
                    mark_task_completed(state, task["task_id"])

                    # Ensure lists exist
                    if "knowledge_results" not in state:
                        state["knowledge_results"] = []
                    state["knowledge_results"].append(result)
                    
                    if "knowledge_graph_data" not in state:
                        state["knowledge_graph_data"] = []
                    state["knowledge_graph_data"].append(result['data'])

                    # Log entity/relationship counts
                    entity_count = result.get('data', {}).get('entity_count', 0)
                    relationship_count = result.get('data', {}).get('relationship_count', 0)
                    logger.info(
                        f"  ✓ Task {task['task_id']} completed: "
                        f"{entity_count} entities, {relationship_count} relationships"
                    )
                else:
                    mark_task_failed(state, task["task_id"], result.get("error"))
                    logger.warning(f"  ✗ Task {task['task_id']} failed: {result.get('error')}")
            
            except Exception as e:
                mark_task_failed(state, task["task_id"], str(e))
                logger.error(f"  ✗ Task {task['task_id']} error: {e}", exc_info=True)
        
        return state

    async def synthesis_node(self, state: AgentState) -> AgentState:
        """
        Synthesis agent node - generates final answer with academic citations.
        
        IMPROVED: Better error handling, context validation, and debug logging.
        """
        logger.info("=" * 80)
        logger.info("🔬 SYNTHESIS AGENT: Generating answer with citations")
        logger.info("=" * 80)
        
        # ✅ CRITICAL DEBUG: Check retrieval context BEFORE synthesis
        retrieval_results = state.get("retrieval_results", [])
        retrieved_contexts = state.get("retrieved_contexts", [])
        
        logger.info(f"📊 Synthesis context check:")
        logger.info(f"  - retrieval_results: {len(retrieval_results)} items")
        logger.info(f"  - retrieved_contexts: {len(retrieved_contexts)} items")
        
        # Calculate total context available
        total_chunks = 0
        total_facts = 0
        for ctx in retrieved_contexts:
            if isinstance(ctx, dict):
                total_chunks += len(ctx.get("chunks", []))
                total_facts += len(ctx.get("graph_facts", []))
        
        logger.info(f"  - Total chunks: {total_chunks}")
        logger.info(f"  - Total facts: {total_facts}")
        logger.info(f"  - Total context items: {total_chunks + total_facts}")
        
        if total_chunks == 0 and total_facts == 0:
            logger.error("❌ CRITICAL: No retrieval context available for synthesis!")
            logger.error("This will likely result in empty synthesis output.")
        
        # ✅ FIX: Get ready tasks without agent_role parameter
        all_ready_tasks = get_ready_tasks(state)
        
        # Filter for synthesis tasks
        synthesis_tasks = [
            t for t in all_ready_tasks 
            if t.get("assigned_agent") == AgentRole.SYNTHESIS.value
            or t.get("task_type") in [TaskType.SYNTHESIS.value, "synthesis", "SYNTHESIS"]
        ]
        
        if not synthesis_tasks:
            logger.warning("No ready synthesis tasks found")
            return state
        
        synthesis_task = synthesis_tasks[0]
        logger.info(f"Processing synthesis task: {synthesis_task['task_id']}")
        
        try:
            # Execute synthesis with full context
            result = await self.synthesis_agent.execute(state, synthesis_task)
            
            # ✅ CRITICAL: Validate result structure
            if not result or not isinstance(result, dict):
                logger.error(f"❌ Synthesis returned invalid result type: {type(result)}")
                logger.error(f"Result content: {result}")
                
                mark_task_failed(
                    state,
                    synthesis_task["task_id"],
                    f"Invalid result type: {type(result)}"
                )
                
                state["final_answer"] = ""
                state["confidence"] = 0.0
                state["synthesized_answer"] = {
                    "answer": "",
                    "confidence": 0.0,
                    "papers_cited": 0,
                    "sources_used": [],
                    "reasoning": "Synthesis returned invalid result structure"
                }
                return state
            
            # ✅ Check if execution was successful
            if not result.get("success", False):
                logger.error(f"❌ Synthesis task failed: {result.get('error', 'Unknown error')}")
                
                mark_task_failed(
                    state,
                    synthesis_task["task_id"],
                    result.get("error", "Synthesis failed")
                )
                
                state["final_answer"] = ""
                state["confidence"] = 0.0
                return state
            
            # ✅ Extract and validate result data
            result_data = result.get("data")
            
            if not result_data or not isinstance(result_data, dict):
                logger.error(f"❌ Synthesis returned invalid data: {type(result_data)}")
                logger.error(f"Full result keys: {list(result.keys())}")
                logger.error(f"Result data: {result_data}")
                
                mark_task_failed(
                    state,
                    synthesis_task["task_id"],
                    f"Invalid data format: {type(result_data)}"
                )
                
                state["final_answer"] = ""
                state["confidence"] = 0.0
                state["synthesized_answer"] = {
                    "answer": "",
                    "confidence": 0.0,
                    "papers_cited": 0,
                    "sources_used": [],
                    "reasoning": "Synthesis data validation failed"
                }
                return state
            
            # ✅ Extract answer and validate length
            answer = result_data.get("answer", "")
            
            if not answer or not isinstance(answer, str):
                logger.error(f"❌ Synthesis returned invalid answer type: {type(answer)}")
                answer = ""
            
            answer_length = len(answer.strip())
            
            if answer_length < 10:
                logger.warning(f"⚠️ Synthesis returned very short answer: {answer_length} chars")
                logger.warning(f"Result data keys: {list(result_data.keys())}")
                logger.warning(f"Answer preview: '{answer[:200]}'")
            else:
                logger.info(f"✅ Synthesis completed successfully: {answer_length} chars")
            
            # ✅ Store results in state
            if hasattr(self.synthesis_agent, 'store_results_in_state'):
                self.synthesis_agent.store_results_in_state(
                    state, result_data, synthesis_task
                )
                logger.info("Results stored via store_results_in_state()")
            else:
                logger.warning("SynthesisAgent missing store_results_in_state, using fallback")
                
                state["final_answer"] = answer
                state["synthesized_answer"] = result_data
                state["confidence"] = result_data.get("confidence", 0.0)
            
            # ✅ Log success metrics
            papers_cited = result_data.get("papers_cited", 0)
            confidence = result_data.get("confidence", 0.0)
            sources_used = result_data.get("sources_used", [])
            
            logger.info(f"📄 Synthesis metrics:")
            logger.info(f"  - Answer length: {answer_length} chars")
            logger.info(f"  - Papers cited: {papers_cited}")
            logger.info(f"  - Confidence: {confidence:.2f}")
            logger.info(f"  - Sources used: {len(sources_used)}")
            
            # ✅ Mark task as completed
            mark_task_completed(state, synthesis_task["task_id"])
            logger.info(f"✅ Synthesis task {synthesis_task['task_id']} completed")
        
        except Exception as e:
            logger.error(f"❌ Synthesis execution failed with exception: {e}", exc_info=True)
            
            mark_task_failed(
                state,
                synthesis_task["task_id"],
                f"Exception: {str(e)}"
            )
            
            state["final_answer"] = ""
            state["confidence"] = 0.0
            state["error"] = str(e)
        
        logger.info("=" * 80)
        return state

    async def validation_node(self, state: AgentState) -> AgentState:
        """
        Validation agent node - validates answer against research literature.
        """
        logger.info("✅ VALIDATION AGENT: Validating answer against research papers")
        
        # Check if validation already done
        if state.get("validation_report"):
            logger.info("  Validation already completed")
            return state
        
        # Get or create validation task
        validation_tasks = [
            task for task in state.get("subtasks", [])
            if task.get("task_type") == TaskType.VALIDATION.value
        ]
        
        if not validation_tasks:
            from .state import add_subtask
            validation_task = add_subtask(
                state=state,
                task_type=TaskType.VALIDATION,
                description="Validate final answer against research literature",
                parameters={"query": state["query"], "validation_type": "fact_check"},
                assigned_agent=AgentRole.VALIDATION,
                priority=3
            )
            logger.info("  Created validation task for research verification")
        else:
            validation_task = validation_tasks[0]
        
        # Execute validation
        try:
            result = await self.validation_agent.execute(state, validation_task)
            
            if result.get("success"):
                mark_task_completed(state, validation_task["task_id"])
                
                if "validation_results" not in state:
                    state["validation_results"] = []
                state["validation_results"].append(result)
                
                # Convert ValidationResult dataclass to dict
                try:
                    validation_data = result.get('data')
                    
                    # Check if it's a dataclass
                    if hasattr(validation_data, '__dataclass_fields__'):
                        from dataclasses import asdict
                        validation_dict = asdict(validation_data)
                        logger.info("  ✅ Converted ValidationResult dataclass to dict")
                    elif isinstance(validation_data, dict):
                        validation_dict = validation_data
                        logger.info("  ✅ Validation data is already a dict")
                    else:
                        logger.warning(
                            f"  ⚠️ Unknown validation data format: {type(validation_data)}"
                        )
                        validation_dict = dict(validation_data) if validation_data else {}
                    
                    # Store in state
                    state["validation_report"] = validation_dict
                    
                    # Log validation results
                    is_valid = validation_dict.get('is_valid', False)
                    hallucination_score = validation_dict.get('hallucination_score', 0.0)
                    papers_ref = validation_dict.get('papers_referenced', 0)
                    verified_claims = validation_dict.get('verified_claims', 0)
                    total_claims = validation_dict.get('total_claims', 0)
                    
                    logger.info(
                        f"  ✓ Validation completed: valid={is_valid}, "
                        f"hallucination={hallucination_score:.2f}, "
                        f"claims={verified_claims}/{total_claims}, "
                        f"papers={papers_ref}"
                    )
                    
                except Exception as store_e:
                    logger.error(
                        f"  ✗ Validation completed but failed to store results: {store_e}",
                        exc_info=True
                    )
                    mark_task_failed(
                        state, validation_task["task_id"],
                        f"Failed to store results: {store_e}"
                    )
            else:
                mark_task_failed(
                    state, validation_task["task_id"],
                    result.get("error", "Unknown validation error")
                )
                logger.warning(f"  ✗ Validation failed: {result.get('error')}")
        
        except Exception as e:
            mark_task_failed(state, validation_task["task_id"], str(e))
            logger.error(f"  ✗ Validation error: {e}", exc_info=True)
        
        return state
    
    # ========================================================================
    # FINALIZATION NODE
    # ========================================================================

    async def finalization_node(self, state: AgentState) -> AgentState:
        """
        Finalization node - aggregates results for scientific literature response.
        
        IMPROVED: Better validation, fallback handling, and debug logging.
        """
        logger.info("=" * 80)
        logger.info("🏁 FINALIZATION: Preparing final research response")
        logger.info("=" * 80)
        
        try:
            # ✅ Debug: Check what we have before aggregation
            validation_report_before = state.get("validation_report")
            final_answer_before = state.get("final_answer", "")
            confidence_before = state.get("confidence", 0.0)
            
            logger.info(f"📊 Pre-finalization state:")
            logger.info(f"  - final_answer length: {len(final_answer_before)} chars")
            logger.info(f"  - confidence: {confidence_before}")
            logger.info(f"  - validation_report present: {validation_report_before is not None}")
            
            # ✅ Aggregate results from all agents
            retrieval_results = state.get("retrieval_results", [])
            knowledge_results = state.get("knowledge_results", [])
            synthesis_results = state.get("synthesis_results", [])
            validation_results = state.get("validation_results", [])
            
            logger.info(f"📦 Results to aggregate:")
            logger.info(f"  - retrieval: {len(retrieval_results)}")
            logger.info(f"  - knowledge: {len(knowledge_results)}")
            logger.info(f"  - synthesis: {len(synthesis_results)}")
            logger.info(f"  - validation: {len(validation_results)}")
            
            aggregated = aggregate_all_results(
                retrieval_results=retrieval_results,
                knowledge_results=knowledge_results,
                synthesis_results=synthesis_results,
                validation_results=validation_results
            )
            
            logger.info(f"✅ Aggregation complete. Keys: {list(aggregated.keys())}")
            
            # ✅ Update state with aggregated data
            synthesis_data = aggregated.get("synthesis", {})
            context_data = aggregated.get("context", {})
            
            # Use aggregated answer if current answer is empty
            aggregated_answer = synthesis_data.get("answer", "")
            
            if not final_answer_before or len(final_answer_before.strip()) < 10:
                if aggregated_answer and len(aggregated_answer.strip()) >= 10:
                    logger.warning(f"⚠️ Using aggregated answer ({len(aggregated_answer)} chars) - state answer was too short")
                    state["final_answer"] = aggregated_answer
                else:
                    logger.error("❌ Both state and aggregated answers are empty/short!")
                    state["final_answer"] = "No answer could be generated from research literature."
            else:
                state["final_answer"] = final_answer_before
                logger.info(f"✅ Using state answer ({len(final_answer_before)} chars)")
            
            # Update sources and confidence
            state["sources"] = context_data.get("sources", [])
            
            # Use max confidence between state and aggregated
            aggregated_confidence = synthesis_data.get("confidence", 0.0)
            state["confidence"] = max(confidence_before, aggregated_confidence)
            
            logger.info(f"📈 Final confidence: {state['confidence']:.2f} (state: {confidence_before:.2f}, agg: {aggregated_confidence:.2f})")
            
            # ✅ Preserve validation_report from validation_node
            if validation_report_before:
                logger.info("✅ Preserved validation_report from validation_node")
            else:
                # Fallback to aggregated validation
                aggregated_validation = aggregated.get("validation", {})
                if aggregated_validation:
                    state["validation_report"] = aggregated_validation
                    logger.info("✅ Set validation_report from aggregated results")
                else:
                    logger.warning("⚠️ No validation_report available (neither in state nor aggregated)")
            
            # ✅ Recalculate final task counts
            final_subtasks = state.get("subtasks", [])
            final_completed_count = sum(
                1 for t in final_subtasks 
                if t.get("status") == SubTaskStatus.COMPLETED.value
            )
            final_failed_count = sum(
                1 for t in final_subtasks 
                if t.get("status") == SubTaskStatus.FAILED.value
            )
            
            # Update state task counts
            state["tasks_completed"] = final_completed_count
            state["tasks_failed"] = final_failed_count
            state["completed_tasks"] = [
                t["task_id"] for t in final_subtasks 
                if t.get("status") == SubTaskStatus.COMPLETED.value
            ]
            state["failed_tasks"] = [
                t["task_id"] for t in final_subtasks 
                if t.get("status") == SubTaskStatus.FAILED.value
            ]
            
            # ✅ Final metadata
            if "metadata" not in state:
                state["metadata"] = {}
            
            state["metadata"]["completed_at"] = datetime.now().isoformat()
            state["metadata"]["domain"] = "scientific_literature"
            state["current_step"] = "completed"
            
            # ✅ Log final summary
            logger.info("=" * 80)
            logger.info("✅ FINALIZATION COMPLETE")
            logger.info(f"📊 Final metrics:")
            logger.info(f"  - Tasks completed: {final_completed_count}")
            logger.info(f"  - Tasks failed: {final_failed_count}")
            logger.info(f"  - Final answer length: {len(state.get('final_answer', ''))} chars")
            logger.info(f"  - Confidence: {state.get('confidence', 0.0):.2f}")
            logger.info(f"  - Sources: {len(state.get('sources', []))}")
            logger.info("=" * 80)
        
        except Exception as e:
            logger.error(f"❌ Finalization error: {e}", exc_info=True)
            
            state["error"] = str(e)
            state["tasks_completed"] = sum(
                1 for t in state.get("subtasks", []) 
                if t.get("status") == SubTaskStatus.COMPLETED.value
            )
            state["tasks_failed"] = sum(
                1 for t in state.get("subtasks", []) 
                if t.get("status") == SubTaskStatus.FAILED.value
            )
        
        return state

# ============================================================================
# ROUTING FUNCTIONS
# ============================================================================

def should_continue_execution(state: AgentState) -> Literal["parallel_execution", "synthesis", "end"]:
    """
    Decision function to determine if execution should continue or move to synthesis.
    
    IMPROVED: Don't skip synthesis just because coordinator had an error.
    """
    # ✅ Check if we have any retrieval results
    retrieval_results = state.get("retrieval_results", [])
    
    if retrieval_results and len(retrieval_results) > 0:
        logger.info("🔀 ROUTER: Retrieval complete, proceeding to synthesis")
        return "synthesis"
    
    # Check for pending tasks
    ready_tasks = get_ready_tasks(state)
    
    if ready_tasks and len(ready_tasks) > 0:
        logger.info(f"🔀 ROUTER: {len(ready_tasks)} tasks ready, continuing parallel execution")
        return "parallel_execution"
    
    # ✅ FIX: Even if there's an error, if we have ANY retrieved context, proceed to synthesis
    retrieved_contexts = state.get("retrieved_contexts", [])
    
    if retrieved_contexts and len(retrieved_contexts) > 0:
        logger.warning("🔀 ROUTER: Error detected but retrieved contexts available, proceeding to synthesis anyway")
        return "synthesis"
    
    # Check for error
    if state.get("error"):
        logger.info("🔀 ROUTER: Error detected and no contexts, moving to end")
        return "end"
    
    # Check if execution is complete
    if is_execution_complete(state):
        logger.info("🔀 ROUTER: Execution complete, proceeding to synthesis")
        return "synthesis"
    
    # Default: end
    logger.info("🔀 ROUTER: No more work, moving to end")
    return "end"


def should_validate(state: AgentState) -> Literal["validation", "finalization"]:
    """
    Determine if answer should be validated against research literature.
    
    Args:
        state: Current state
    
    Returns:
        Next node to execute
    """
    # Check if validation already done
    if state.get("validation_report"):
        logger.info("🔀 ROUTER: Research validation complete, moving to finalization")
        return "finalization"
    
    # Check if synthesis succeeded
    if not state.get("synthesized_answer") and not state.get("final_answer"):
        logger.info("🔀 ROUTER: No answer to validate, skipping to finalization")
        return "finalization"
    
    # Validate
    logger.info("🔀 ROUTER: Moving to research validation")
    return "validation"


async def parallel_execution_node(state: AgentState) -> Dict[str, Any]:
    """
    Executes paper retrieval and entity extraction tasks concurrently.
    """
    import asyncio
    
    logger.info("⚡ PARALLEL EXECUTION: Running paper retrieval and entity extraction")
    
    nodes = MASGraphNodes()
    ready_tasks = get_ready_tasks(state)
    
    tasks_to_run = []
    tasks_being_run_map = {}

    # Prepare coroutines
    for task in ready_tasks:
        if nodes.retrieval_agent.can_handle_task(task):
            tasks_to_run.append(nodes.retrieval_agent.execute(state, task))
            tasks_being_run_map[task["task_id"]] = task
        elif nodes.knowledge_agent.can_handle_task(task):
            tasks_to_run.append(nodes.knowledge_agent.execute(state, task))
            tasks_being_run_map[task["task_id"]] = task

    if not tasks_to_run:
        logger.info("  No parallel tasks ready in this cycle")
        return {}

    logger.info(f"  Executing {len(tasks_to_run)} tasks concurrently...")

    # Execute concurrently
    all_results = await asyncio.gather(*tasks_to_run, return_exceptions=True)

    logger.info("  Parallel execution finished. Processing results...")

    # Prepare state updates
    new_retrieval_results = []
    new_knowledge_results = []
    new_retrieved_contexts = []
    new_knowledge_graph_data = []
    updated_subtasks = state.get("subtasks", []).copy()
    processed_task_ids = set()

    for result in all_results:
        if isinstance(result, Exception):
            logger.error(
                f"  A concurrent task execution failed: {result}",
                exc_info=True
            )
            continue
            
        task_id = result.get("task_id")
        if not task_id:
            logger.warning("  Received result without task_id")
            continue
            
        processed_task_ids.add(task_id)
        original_task = tasks_being_run_map.get(task_id)

        task_updated = False
        for i, task in enumerate(updated_subtasks):
            if task["task_id"] == task_id:
                if result.get("success"):
                    updated_subtasks[i]["status"] = SubTaskStatus.COMPLETED.value
                    updated_subtasks[i]["completed_at"] = datetime.now().isoformat()
                    task_updated = True
                    task_data = result.get("data")
                    
                    if task_data is not None and original_task:
                        if nodes.retrieval_agent.can_handle_task(original_task):
                            new_retrieval_results.append(result)
                            new_retrieved_contexts.append(task_data)
                            
                            # Log paper count
                            unique_papers = task_data.get('unique_papers', 0)
                            logger.info(
                                f"  ✓ Retrieved {unique_papers} papers for task {task_id}"
                            )
                        elif nodes.knowledge_agent.can_handle_task(original_task):
                            new_knowledge_results.append(result)
                            new_knowledge_graph_data.append(task_data)
                            
                            # Log entity count
                            entity_count = task_data.get('entity_count', 0)
                            logger.info(
                                f"  ✓ Extracted {entity_count} entities for task {task_id}"
                            )
                    elif task_data is None:
                        logger.warning(f"  Task {task_id} succeeded but returned no data")
                        updated_subtasks[i]["status"] = SubTaskStatus.FAILED.value
                        updated_subtasks[i]["error"] = "Success but no data"
                else:
                    error_msg = result.get('error', 'Unknown error')
                    logger.warning(f"  Task {task_id} failed: {error_msg}")
                    updated_subtasks[i]["status"] = SubTaskStatus.FAILED.value
                    updated_subtasks[i]["error"] = error_msg
                    task_updated = True
                break
        
        if not task_updated:
            logger.warning(f"Could not find task {task_id} in subtasks list")

    # Mark tasks that never returned as failed
    for task_id, task in tasks_being_run_map.items():
        if task_id not in processed_task_ids:
            logger.error(f"  Task {task_id} never returned result")
            for i, st in enumerate(updated_subtasks):
                if st["task_id"] == task_id:
                    updated_subtasks[i]["status"] = SubTaskStatus.FAILED.value
                    updated_subtasks[i]["error"] = "Task execution timeout"
                    break

    # Construct update dictionary
    updates_to_state = {"subtasks": updated_subtasks}
    if new_retrieval_results:
        updates_to_state["retrieval_results"] = new_retrieval_results
        updates_to_state["retrieved_contexts"] = new_retrieved_contexts
    if new_knowledge_results:
        updates_to_state["knowledge_results"] = new_knowledge_results
        updates_to_state["knowledge_graph_data"] = new_knowledge_graph_data
        
    return updates_to_state


# ============================================================================
# GRAPH BUILDER
# ============================================================================

def create_mas_graph(
    enable_validation: bool = True,
    checkpointer: Optional[Any] = None
) -> StateGraph:
    """
    Create the Multi-Agent System LangGraph for scientific literature.
    
    Args:
        enable_validation: Whether to include validation node
        checkpointer: Optional checkpointer for state persistence
    
    Returns:
        Compiled StateGraph
    """
    logger.info("Building Multi-Agent System graph for scientific literature")
    
    # Create node functions
    nodes = MASGraphNodes()
    
    # Create graph
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("coordinator", nodes.coordinator_node)
    graph.add_node("parallel_execution", parallel_execution_node)
    graph.add_node("synthesis", nodes.synthesis_node)
    
    if enable_validation:
        graph.add_node("validation", nodes.validation_node)
    
    graph.add_node("finalization", nodes.finalization_node)
    
    # Set entry point
    graph.set_entry_point("coordinator")
    
    # Add edges
    graph.add_edge("coordinator", "parallel_execution")
    
    graph.add_conditional_edges(
        "parallel_execution",
        should_continue_execution,
        {
            "parallel_execution": "parallel_execution",
            "synthesis": "synthesis",
            "end": "finalization"
        }
    )
    
    if enable_validation:
        graph.add_conditional_edges(
            "synthesis",
            should_validate,
            {
                "validation": "validation",
                "finalization": "finalization"
            }
        )
        
        graph.add_edge("validation", "finalization")
    else:
        graph.add_edge("synthesis", "finalization")
    
    graph.add_edge("finalization", END)
    
    if checkpointer is None:
        checkpointer = MemorySaver()
    
    compiled_graph = graph.compile(checkpointer=checkpointer)
    
    logger.info("✓ Scientific literature MAS graph compiled successfully")
    
    return compiled_graph


# ============================================================================
# GRAPH EXECUTOR
# ============================================================================

class MASGraphExecutor:
    """Executor for the Multi-Agent System graph for scientific literature."""

    def __init__(
        self,
        enable_validation: bool = True,
        checkpointer: Optional[Any] = None
    ):
        """
        Initialize graph executor for scientific literature.

        Args:
            enable_validation: Whether to enable validation node
            checkpointer: Optional LangGraph checkpointer instance
        """
        self.graph = create_mas_graph(
            enable_validation=enable_validation,
            checkpointer=checkpointer
        )
        self.enable_validation = enable_validation
        logger.info("MASGraphExecutor initialized for scientific literature")

    async def execute(
        self,
        query: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        search_type: str = "hybrid",
        retrieval_limit: int = 10,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute the graph for a research query.
        
        Args:
            query: Research query
            session_id: Optional session ID
            user_id: Optional user ID
            search_type: Search type (vector, graph, hybrid)
            retrieval_limit: Number of papers to retrieve
            config: Optional additional config
        
        Returns:
            Response dict with answer, citations, and validation
        """
        effective_session_id = session_id or str(uuid4())

        logger.info(
            f"Executing MAS graph for research query: '{query[:100]}...' "
            f"(Session: {effective_session_id})"
        )

        initial_state = create_initial_state(
            query=query,
            session_id=effective_session_id,
            user_id=user_id,
            search_type=search_type,
            config=config or {}
        )
        
        # Add research-specific metadata
        if "metadata" not in initial_state:
            initial_state["metadata"] = {}
        initial_state["metadata"]["retrieval_limit"] = retrieval_limit
        initial_state["metadata"]["domain"] = "scientific_literature"

        final_state = {}
        try:
            graph_config = {
                "configurable": {"thread_id": effective_session_id},
                "recursion_limit": 100
            }

            final_state = await self.graph.ainvoke(initial_state, config=graph_config)
            
            metadata = final_state.get("metadata", {})

            final_subtasks = final_state.get("subtasks", [])
            final_completed_count = sum(
                1 for t in final_subtasks 
                if t.get("status") == SubTaskStatus.COMPLETED.value
            )
            final_failed_count = sum(
                1 for t in final_subtasks 
                if t.get("status") == SubTaskStatus.FAILED.value
            )

            # Calculate latency
            start_time_str = metadata.get("started_at")
            end_time_str = metadata.get("completed_at")
            latency_ms = None
            if start_time_str and end_time_str:
                try:
                    start_time_dt = datetime.fromisoformat(start_time_str)
                    end_time_dt = datetime.fromisoformat(end_time_str)
                    latency_ms = (end_time_dt - start_time_dt).total_seconds() * 1000
                except (ValueError, TypeError):
                    logger.warning("Could not parse timestamps for latency")

            search_config_used = initial_state.get("search_config", {})
            
            # Count papers retrieved
            papers_retrieved = 0
            for ctx in final_state.get("retrieved_contexts", []):
                if isinstance(ctx, dict):
                    papers_retrieved += ctx.get("unique_papers", 0)
            
            response = {
                "answer": final_state.get(
                    "final_answer",
                    "No answer could be generated from research literature."
                ),
                "query": query,
                "session_id": metadata.get("session_id"),
                "confidence": final_state.get("confidence", 0.0),
                "sources": final_state.get("sources", []),
                "research_metadata": {
                    "papers_retrieved": papers_retrieved,
                    "domain": "scientific_literature"
                },
                "metadata": {
                    "query_id": metadata.get("query_id"),
                    "tasks_created": len(final_subtasks),
                    "tasks_completed": final_completed_count,
                    "tasks_failed": final_failed_count,
                    "search_type": search_type,
                    "latency_ms": latency_ms,
                    "retrieval_limit": retrieval_limit
                },
                "validation": None
            }

            # ✅ Get validation report WITH ALL FIELDS
            validation_report = final_state.get("validation_report")
            if validation_report and isinstance(validation_report, dict):
                response["validation"] = {
                    # Basic fields
                    "is_valid": validation_report.get("is_valid", False),
                    "hallucination_score": validation_report.get("hallucination_score", 0.0),
                    "issues": validation_report.get("issues", []),
                    "verified_claims": validation_report.get("verified_claims", 0),
                    "total_claims": validation_report.get("total_claims", 0),
                    "papers_referenced": validation_report.get("papers_referenced", 0),
                    
                    # ✅ NEW: Detailed transparency fields
                    "detailed_claims": validation_report.get("detailed_claims", []),
                    "hallucination_examples": validation_report.get("hallucination_examples", []),
                    "verification_summary": validation_report.get("verification_summary", {}),
                    "citation_verification": validation_report.get("citation_verification", {})
                }
                
                logger.info(
                    f"Validation report included with {len(validation_report.get('detailed_claims', []))} detailed claims"
                )
            
            logger.info(
                f"✓ Graph execution complete: "
                f"{final_completed_count} tasks completed, "
                f"{papers_retrieved} papers retrieved"
            )
            return response

        except Exception as e:
            logger.error(
                f"✗ Graph execution failed: {e}",
                exc_info=True
            )
            last_known_state = final_state or initial_state
            final_subtasks_on_error = last_known_state.get("subtasks", [])
            completed_on_error = sum(
                1 for t in final_subtasks_on_error 
                if t.get("status") == SubTaskStatus.COMPLETED.value
            )
            failed_on_error = sum(
                1 for t in final_subtasks_on_error 
                if t.get("status") == SubTaskStatus.FAILED.value
            )
            
            return {
                "answer": "An error occurred during research processing.",
                "query": query,
                "session_id": effective_session_id,
                "confidence": 0.0,
                "sources": [],
                "error": str(e),
                "research_metadata": {
                    "papers_retrieved": 0,
                    "domain": "scientific_literature"
                },
                "metadata": {
                    "query_id": initial_state.get("metadata", {}).get("query_id"),
                    "session_id": effective_session_id,
                    "tasks_created": len(final_subtasks_on_error),
                    "tasks_completed": completed_on_error,
                    "tasks_failed": failed_on_error,
                }
            }

    def visualize(self, output_path: str = "mas_graph_scientific.png"):
        """Visualize the graph structure."""
        try:
            graph_image = self.graph.get_graph().draw_mermaid_png()
            with open(output_path, "wb") as f:
                f.write(graph_image)
            logger.info(f"Graph visualization saved to {output_path}")
        except Exception as e:
            logger.warning(f"Graph visualization failed: {e}")

    def get_graph_structure(self) -> str:
        """Get text representation of graph structure."""
        try:
            return self.graph.get_graph().draw_mermaid()
        except Exception as e:
            logger.error(f"Failed to get graph structure: {e}")
            return ""


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

async def example_usage():
    """Example showing LangGraph execution for scientific literature."""
    
    print("="*70)
    print("SCIENTIFIC LITERATURE MULTI-AGENT SYSTEM - LANGGRAPH EXECUTION")
    print("="*70)
    
    executor = MASGraphExecutor(enable_validation=True)
    
    queries = [
        "What is the Transformer architecture?",
        "Compare BERT and GPT-2 architectures"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'='*70}")
        print(f"Research Query {i}: {query}")
        print(f"{'='*70}\n")
        
        response = await executor.execute(
            query=query,
            search_type="hybrid",
            retrieval_limit=5
        )
        
        print(f"Answer: {response['answer'][:300]}...")
        print(f"\nResearch Metadata:")
        print(f"  Papers Retrieved: {response['research_metadata']['papers_retrieved']}")
        print(f"  Tasks: {response['metadata']['tasks_completed']}/{response['metadata']['tasks_created']}")
        print(f"  Confidence: {response['confidence']:.2f}")
        print(f"  Sources: {len(response['sources'])}")
        
        if response.get('validation'):
            print(f"\nValidation:")
            print(f"  Valid: {response['validation']['is_valid']}")
            print(f"  Hallucination: {response['validation']['hallucination_score']:.2f}")
            print(f"  Claims: {response['validation']['verified_claims']}/{response['validation']['total_claims']}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    import asyncio
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Note: This requires database and graph connections.\n")
