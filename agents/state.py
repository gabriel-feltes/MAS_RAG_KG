"""
Multi-Agent System state management - TypedDict version for LangGraph.

This module defines the state structure and management functions for
the multi-agent system focused on scientific literature analysis.
"""

from typing import Dict, Any, List, Optional, TypedDict
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


# ============================================================================
# ENUMS
# ============================================================================

class QueryIntent(str, Enum):
    """Types of query intents for scientific literature."""
    FACTUAL = "factual"                    # "What is X?"
    COMPARISON = "comparison"              # "Compare X and Y"
    TEMPORAL = "temporal"                  # "When did X happen?"
    RELATIONAL = "relational"              # "How is X related to Y?"
    LIST_INVENTORY = "list_inventory"      # "List all X"
    AGGREGATION = "aggregation"            # "Summarize X"
    CAUSAL = "causal"                      # "Why did X happen?"
    PROCEDURAL = "procedural"              # "How to do X?"
    EXPLORATORY = "exploratory"            # "Tell me about X"
    METHODOLOGICAL = "methodological"      # "How does method X work?"
    SURVEY = "survey"                      # "Survey of X"


class QueryComplexity(Enum):
    """Query complexity levels."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


class AgentRole(Enum):
    """Agent role types."""
    COORDINATOR = "coordinator"
    RETRIEVAL = "retrieval"
    KNOWLEDGE = "knowledge"
    SYNTHESIS = "synthesis"
    VALIDATION = "validation"


class TaskType(Enum):
    """Task types for agent execution."""
    RETRIEVAL = "retrieval"
    KNOWLEDGE_EXTRACTION = "knowledge_extraction"
    SYNTHESIS = "synthesis"
    VALIDATION = "validation"
    VECTOR_SEARCH = "vector_search"
    GRAPH_SEARCH = "graph_search"
    HYBRID_SEARCH = "hybrid_search"
    ENTITY_EXTRACTION = "entity_extraction"
    RELATIONSHIP_FINDING = "relationship_finding"
    TEMPORAL_REASONING = "temporal_reasoning"
    LIST_DOCUMENTS = "list_documents"
    GET_DOCUMENT = "get_document"


class SubTaskStatus(Enum):
    """Status of a subtask."""
    PENDING = "pending"
    READY = "ready"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


# ============================================================================
# DATACLASSES
# ============================================================================

@dataclass
class SubTask:
    """A subtask to be executed by an agent."""
    id: str
    task_type: TaskType
    description: str
    agent_role: AgentRole
    status: SubTaskStatus = SubTaskStatus.PENDING
    dependencies: List[str] = field(default_factory=list)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None


@dataclass
class AgentResult:
    """Result from an agent execution."""
    agent_role: AgentRole
    task_id: str
    success: bool
    result: Dict[str, Any]
    error: Optional[str] = None
    latency_ms: float = 0.0


@dataclass
class RetrievalResult:
    """Result from retrieval agent for scientific literature."""
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    total_chunks: int = 0
    search_type: str = "hybrid"
    avg_similarity: float = 0.0
    graph_facts: List[Dict[str, Any]] = field(default_factory=list)
    total_results: int = 0
    avg_score: float = 0.0
    sources: List[str] = field(default_factory=list)
    unique_papers: int = 0  # Number of unique papers retrieved


@dataclass
class KnowledgeResult:
    """Result from knowledge agent for academic entities."""
    entities: List[Dict[str, Any]] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    temporal_facts: List[Dict[str, Any]] = field(default_factory=list)
    graph_paths: List[List[Dict[str, Any]]] = field(default_factory=list)
    entity_count: int = 0
    relationship_count: int = 0


@dataclass
class SynthesisResult:
    """Result from synthesis agent for research answers."""
    answer: str = ""
    sources_used: List[str] = field(default_factory=list)
    confidence: float = 0.0
    reasoning: str = ""
    papers_cited: int = 0  # Number of papers cited in answer


@dataclass
class ValidationResult:
    """Result from validation agent for citation verification."""
    is_valid: bool = True
    hallucination_score: float = 0.0
    issues: List[str] = field(default_factory=list)
    fact_checks: List[Dict[str, Any]] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)
    original_answer: str = ""
    validated_answer: str = ""
    total_claims: int = 0
    verified_claims: int = 0
    unsupported_claims: int = 0
    papers_referenced: int = 0  # Number of papers referenced in validation
    citation_verification: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# TYPEDDICT FOR LANGGRAPH
# ============================================================================

class MASState(TypedDict, total=False):
    """State for Multi-Agent System - LangGraph compatible for scientific literature."""
    query: str
    session_id: str
    user_id: str
    search_type: str
    
    # Query analysis
    intent: str
    complexity: str
    
    # Tasks (both names for compatibility)
    tasks: List[Dict[str, Any]]
    subtasks: List[Dict[str, Any]]
    completed_tasks: List[str]
    failed_tasks: List[str]
    tasks_created: int
    tasks_completed: int
    tasks_failed: int
    
    # Execution plan
    execution_plan: Dict[str, Any]
    current_step: str
    
    # Results (Store full result packets for aggregation/logging)
    retrieval_results: List[Dict[str, Any]]
    knowledge_results: List[Dict[str, Any]]
    synthesis_results: List[Dict[str, Any]]
    validation_results: List[Dict[str, Any]]
    
    # Specific data payloads
    retrieved_contexts: List[Dict[str, Any]]  # List of RetrievalResult
    knowledge_graph_data: List[Dict[str, Any]]  # List of KnowledgeResult
    
    # Validation report
    validation_report: Dict[str, Any]  # Single ValidationResult
    
    # Synthesized answer details
    synthesized_answer: Dict[str, Any]  # Single SynthesisResult
    
    # Agent messages/history
    messages: List[Dict[str, Any]]
    agent_messages: List[Dict[str, Any]]  # For inter-agent communication
    coordinator_decisions: List[Dict[str, Any]]  # Coordinator decisions log
    
    # Final output (populated by finalization_node)
    final_answer: str
    confidence: float
    sources: List[str]
    
    # Metadata
    metadata: Dict[str, Any]
    error: str
    search_config: Dict[str, Any]


# Alias for compatibility
AgentState = MASState


# ============================================================================
# STATE MANAGEMENT FUNCTIONS
# ============================================================================

def create_initial_state(
    query: str,
    session_id: str,
    user_id: str = "",
    search_type: str = "hybrid",
    config: Optional[Dict[str, Any]] = None
) -> MASState:
    """
    Create initial MAS state for scientific literature query.
    
    Args:
        query: Research query
        session_id: Session ID
        user_id: Optional user ID
        search_type: Search type (vector, graph, hybrid)
        config: Optional configuration
    
    Returns:
        Initialized MASState
    """
    from uuid import uuid4
    
    return MASState(
        query=query,
        session_id=session_id,
        user_id=user_id,
        search_type=search_type,
        intent=QueryIntent.FACTUAL.value,
        complexity=QueryComplexity.MODERATE.value,
        tasks=[],
        subtasks=[],
        completed_tasks=[],
        failed_tasks=[],
        tasks_created=0,
        tasks_completed=0,
        tasks_failed=0,
        execution_plan={},
        current_step="coordinator",
        retrieval_results=[],
        knowledge_results=[],
        synthesis_results=[],
        validation_results=[],
        retrieved_contexts=[],
        knowledge_graph_data=[],
        synthesized_answer={},
        validation_report={},
        messages=[],
        agent_messages=[],
        coordinator_decisions=[],
        final_answer="",
        confidence=0.0,
        sources=[],
        metadata={
            "query_id": str(uuid4()),
            "session_id": session_id,
            "user_id": user_id,
            "search_type": search_type,
            "started_at": datetime.now().isoformat(),
            "domain": "scientific_literature"
        },
        error="",
        search_config=config or {}
    )


def add_agent_message(
    state: MASState,
    sender: AgentRole,
    receiver: AgentRole,
    content: Dict[str, Any]
) -> MASState:
    """
    Add an inter-agent message to the state.
    
    Args:
        state: Current state
        sender: Sending agent role
        receiver: Receiving agent role
        content: Message content
    
    Returns:
        Updated state
    """
    if "agent_messages" not in state:
        state["agent_messages"] = []
    
    state["agent_messages"].append({
        "sender": sender.value if isinstance(sender, AgentRole) else sender,
        "receiver": receiver.value if isinstance(receiver, AgentRole) else receiver,
        "content": content,
        "timestamp": datetime.now().isoformat()
    })
    
    return state


def add_message(
    state: MASState,
    agent_role: AgentRole,
    message: str
) -> MASState:
    """
    Add a message from an agent to the state.
    
    Args:
        state: Current state
        agent_role: Agent role
        message: Message text
    
    Returns:
        Updated state
    """
    if "messages" not in state:
        state["messages"] = []
    
    state["messages"].append({
        "agent_role": agent_role.value if isinstance(agent_role, AgentRole) else agent_role,
        "message": message,
        "timestamp": datetime.now().isoformat()
    })
    
    return state


def get_pending_tasks(state: MASState) -> List[Dict[str, Any]]:
    """
    Get all pending tasks.
    
    Args:
        state: Current state
    
    Returns:
        List of pending tasks
    """
    tasks = state.get("subtasks") or state.get("tasks", [])
    return [t for t in tasks if t.get("status") == SubTaskStatus.PENDING.value]


def get_ready_tasks(state: MASState) -> List[Dict[str, Any]]:
    """
    Get tasks that are ready to execute (dependencies met).
    
    Args:
        state: Current state
    
    Returns:
        List of ready tasks
    """
    ready = []
    completed = state.get("completed_tasks", [])
    tasks = state.get("subtasks") or state.get("tasks", [])
    
    for task in tasks:
        if task.get("status") == SubTaskStatus.PENDING.value:
            if all(dep_id in completed for dep_id in task.get("dependencies", [])):
                ready.append(task)
    
    return ready


def mark_task_completed(
    state: MASState,
    task_id: str,
    result: Dict[str, Any] = None
) -> MASState:
    """
    Mark a task as completed.
    
    Args:
        state: Current state
        task_id: Task ID
        result: Optional result data
    
    Returns:
        Updated state
    """
    tasks = state.get("subtasks") or state.get("tasks", [])
    
    for task in tasks:
        if task.get("id") == task_id or task.get("task_id") == task_id:
            task["status"] = SubTaskStatus.COMPLETED.value
            if result:
                task["result"] = result
            task["completed_at"] = datetime.now().isoformat()
            
            if "completed_tasks" not in state:
                state["completed_tasks"] = []
            state["completed_tasks"].append(task_id)
            
            state["tasks_completed"] = state.get("tasks_completed", 0) + 1
            break
    
    return state


def mark_task_failed(
    state: MASState,
    task_id: str,
    error: str = ""
) -> MASState:
    """
    Mark a task as failed.
    
    Args:
        state: Current state
        task_id: Task ID
        error: Error message
    
    Returns:
        Updated state
    """
    tasks = state.get("subtasks") or state.get("tasks", [])
    
    for task in tasks:
        if task.get("id") == task_id or task.get("task_id") == task_id:
            task["status"] = SubTaskStatus.FAILED.value
            task["error"] = error
            task["completed_at"] = datetime.now().isoformat()
            
            if "failed_tasks" not in state:
                state["failed_tasks"] = []
            state["failed_tasks"].append(task_id)
            
            state["tasks_failed"] = state.get("tasks_failed", 0) + 1
            break
    
    return state


def is_execution_complete(state: MASState) -> bool:
    """
    Check if all tasks are completed or failed.
    
    Args:
        state: Current state
    
    Returns:
        True if execution is complete
    """
    tasks = state.get("subtasks") or state.get("tasks", [])
    if not tasks:
        return False
    
    completed = len(state.get("completed_tasks", []))
    failed = len(state.get("failed_tasks", []))
    total_finished = completed + failed
    
    return total_finished >= len(tasks)


def add_subtask(
    state: MASState,
    task_type: TaskType,
    description: str,
    agent_role: Optional[AgentRole] = None,
    assigned_agent: Optional[AgentRole] = None,
    dependencies: Optional[List[str]] = None,
    parameters: Optional[Dict[str, Any]] = None,
    priority: int = 1,
    **kwargs
) -> Dict[str, Any]:
    """
    Add a new subtask to the state.
    
    Args:
        state: Current state
        task_type: Type of task
        description: Task description
        agent_role: Agent role (deprecated, use assigned_agent)
        assigned_agent: Agent to assign task to
        dependencies: Task dependencies
        parameters: Task parameters
        priority: Task priority
        **kwargs: Additional task fields
    
    Returns:
        Created task dict
    """
    from uuid import uuid4
    
    role = assigned_agent or agent_role or AgentRole.COORDINATOR
    
    task = {
        "id": str(uuid4()),
        "task_id": str(uuid4()),
        "task_type": task_type.value if isinstance(task_type, TaskType) else task_type,
        "description": description,
        "agent_role": role.value if isinstance(role, AgentRole) else role,
        "assigned_agent": role.value if isinstance(role, AgentRole) else role,
        "status": SubTaskStatus.PENDING.value,
        "dependencies": dependencies or [],
        "parameters": parameters or {},
        "priority": priority,
        "result": None,
        "error": None,
        "created_at": datetime.now().isoformat(),
        "completed_at": None
    }
    
    task.update(kwargs)
    
    if "tasks" not in state:
        state["tasks"] = []
    if "subtasks" not in state:
        state["subtasks"] = []
    
    state["tasks"].append(task)
    state["subtasks"].append(task)
    state["tasks_created"] = state.get("tasks_created", 0) + 1
    
    return task


def get_task_by_id(state: MASState, task_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a task by ID.
    
    Args:
        state: Current state
        task_id: Task ID
    
    Returns:
        Task dict or None
    """
    tasks = state.get("subtasks") or state.get("tasks", [])
    
    for task in tasks:
        if task.get("id") == task_id or task.get("task_id") == task_id:
            return task
    
    return None


def get_completed_task_count(state: MASState) -> int:
    """
    Get count of completed tasks.
    
    Args:
        state: Current state
    
    Returns:
        Number of completed tasks
    """
    return len(state.get("completed_tasks", []))


def get_failed_task_count(state: MASState) -> int:
    """
    Get count of failed tasks.
    
    Args:
        state: Current state
    
    Returns:
        Number of failed tasks
    """
    return len(state.get("failed_tasks", []))


def get_total_task_count(state: MASState) -> int:
    """
    Get total task count.
    
    Args:
        state: Current state
    
    Returns:
        Total number of tasks
    """
    tasks = state.get("subtasks") or state.get("tasks", [])
    return len(tasks)


def get_task_completion_rate(state: MASState) -> float:
    """
    Get task completion rate.
    
    Args:
        state: Current state
    
    Returns:
        Completion rate (0-1)
    """
    total = get_total_task_count(state)
    if total == 0:
        return 0.0
    
    completed = get_completed_task_count(state)
    return completed / total


# ============================================================================
# RESEARCH-SPECIFIC HELPER FUNCTIONS
# ============================================================================

def get_papers_retrieved_count(state: MASState) -> int:
    """
    Get count of unique papers retrieved.
    
    Args:
        state: Current state
    
    Returns:
        Number of unique papers
    """
    unique_papers = set()
    
    for ctx in state.get("retrieved_contexts", []):
        if isinstance(ctx, dict) and "chunks" in ctx:
            for chunk in ctx["chunks"]:
                paper_id = chunk.get("document_id") or chunk.get("document_title")
                if paper_id:
                    unique_papers.add(paper_id)
    
    return len(unique_papers)


def get_papers_cited_count(state: MASState) -> int:
    """
    Get count of papers cited in answer.
    
    Args:
        state: Current state
    
    Returns:
        Number of papers cited
    """
    synthesized = state.get("synthesized_answer", {})
    return synthesized.get("papers_cited", 0)


def get_validation_status(state: MASState) -> Dict[str, Any]:
    """
    Get validation status summary.
    
    Args:
        state: Current state
    
    Returns:
        Validation status dict
    """
    validation_report = state.get("validation_report", {})
    
    return {
        "is_valid": validation_report.get("is_valid", False),
        "hallucination_score": validation_report.get("hallucination_score", 0.0),
        "verified_claims": validation_report.get("verified_claims", 0),
        "total_claims": validation_report.get("total_claims", 0),
        "papers_referenced": validation_report.get("papers_referenced", 0)
    }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_usage():
    """Example showing state management for scientific literature queries."""
    
    # Create initial state
    state = create_initial_state(
        query="What is the Transformer architecture?",
        session_id="test-session-123",
        search_type="hybrid"
    )
    
    print("Initial state created:")
    print(f"  Query: {state['query']}")
    print(f"  Session ID: {state['session_id']}")
    print(f"  Domain: {state['metadata']['domain']}")
    
    # Add a retrieval task
    retrieval_task = add_subtask(
        state=state,
        task_type=TaskType.HYBRID_SEARCH,
        description="Search for Transformer papers",
        assigned_agent=AgentRole.RETRIEVAL,
        parameters={"query": "Transformer architecture", "limit": 5},
        priority=10
    )
    
    print(f"\nAdded retrieval task: {retrieval_task['task_id']}")
    print(f"  Total tasks: {get_total_task_count(state)}")
    print(f"  Pending tasks: {len(get_pending_tasks(state))}")
    
    # Mark task as completed
    mark_task_completed(state, retrieval_task['task_id'])
    
    print(f"\nTask completion:")
    print(f"  Completed: {get_completed_task_count(state)}")
    print(f"  Completion rate: {get_task_completion_rate(state):.1%}")
    
    # Check if execution is complete
    print(f"\nExecution complete: {is_execution_complete(state)}")


if __name__ == "__main__":
    example_usage()
