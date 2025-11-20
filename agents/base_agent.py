"""
Base agent class for Multi-Agent System.

This module defines the abstract base class that all specialist agents inherit from.
Provides common functionality for agent execution, state management, and communication
for scientific literature analysis.
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
import time
from uuid import uuid4

from pydantic_ai import Agent as PydanticAgent

from .state import (
    AgentState,
    AgentRole,
    SubTask,
    SubTaskStatus,
    add_agent_message
)

# Import providers and tools
try:
    from ..agent.providers import get_ingestion_model
    from ..agent.models import AgentDependencies
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from agent.providers import get_ingestion_model
    from agent.models import AgentDependencies

logger = logging.getLogger(__name__)


# ============================================================================
# BASE AGENT CLASS
# ============================================================================

class BaseAgent(ABC):
    """
    Abstract base class for all specialist agents in the MAS for scientific literature.
    Provides common execution interface, state management, error handling, etc.
    """
    
    def __init__(
        self,
        role: AgentRole,
        name: str,
        description: str,
        model_name: Optional[str] = None
    ):
        """
        Initialize base agent for scientific literature processing.
        
        Args:
            role: Agent role (COORDINATOR, RETRIEVAL, KNOWLEDGE, SYNTHESIS, VALIDATION)
            name: Human-readable agent name
            description: Agent description
            model_name: Optional LLM model name
        """
        self.role = role
        self.name = name
        self.description = description
        self.agent_id = f"{role.value}_{uuid4().hex[:8]}"

        # LLM Configuration
        default_model_name = os.getenv("LLM_CHOICE", "gpt-4o-mini")
        actual_model_name = model_name if model_name else default_model_name
        self.model_name_str = actual_model_name
        
        # Add provider prefix if needed
        self.model = actual_model_name
        if ":" not in self.model:
            if self.model.startswith("gpt"):
                self.model = f"openai:{self.model}"
            elif "gpt-oss" in self.model or "compound" in self.model:
                self.model = f"groq:{self.model}"
            elif self.model.startswith("claude"):
                self.model = f"anthropic:{self.model}"
        
        logger.info(f"Agent {self.name} configured to use model: '{self.model}'")
        
        if not self.model:
            logger.error(f"Agent {self.name} initialized WITHOUT a valid LLM model!")
        
        # Performance tracking
        self.total_executions = 0
        self.total_latency_ms = 0.0
        self.successful_executions = 0
        self.failed_executions = 0
        
        # Research-specific metrics
        self.total_papers_processed = 0
        self.total_citations_handled = 0
        
        # Agent-specific tools
        self.tools: List[Callable] = []
        
        logger.info(
            f"Initialized agent: {self.name} (ID: {self.agent_id}, "
            f"Role: {self.role.name}) for scientific literature"
        )
        
        if self.model is None:
            logger.error(
                f"CRITICAL: Agent {self.name} ({self.agent_id}) "
                f"failed to configure a model name."
            )
    
    # ========================================================================
    # ABSTRACT METHODS (Must be implemented by subclasses)
    # ========================================================================
    
    @abstractmethod
    async def process_task(
        self,
        state: AgentState,
        task: SubTask
    ) -> Dict[str, Any]:
        """
        Process a specific subtask.
        
        This is the core method that each specialist agent must implement.
        It should execute the task and return structured results.
        
        Args:
            state: Current system state
            task: Subtask to process
        
        Returns:
            Dict with task results
        
        Raises:
            Exception: If task processing fails
        """
        pass
    
    @abstractmethod
    def can_handle_task(self, task: SubTask) -> bool:
        """
        Check if this agent can handle a specific task.
        
        Args:
            task: Subtask to check
        
        Returns:
            True if agent can handle this task
        """
        pass
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Get the system prompt for this agent's LLM.
        
        Returns:
            System prompt string
        """
        pass
    
    # ========================================================================
    # PUBLIC METHODS
    # ========================================================================
    
    async def execute(
        self,
        state: AgentState,
        task: SubTask
    ) -> Dict[str, Any]:
        """
        Execute a subtask and return an AgentResult.
        
        This is the main entry point for task execution. It:
        1. Validates the task
        2. Updates task status
        3. Calls process_task()
        4. Handles errors
        5. Tracks performance
        6. Returns structured result
        
        Args:
            state: Current system state
            task: Subtask to execute
        
        Returns:
            Dict with execution outcome
        """
        start_time = time.time()
        task_id = task["task_id"]
        
        logger.info(
            f"[{self.name}] Starting task: {task_id} - {task['description']}"
        )
        
        # Validate task assignment
        if not self.can_handle_task(task):
            error_msg = f"Agent {self.name} cannot handle task type {task['task_type']}"
            logger.error(f"[{self.name}] {error_msg}")
            
            return self._create_error_result(
                task_id=task_id,
                error=error_msg,
                latency_ms=0.0
            )
        
        # Update task status
        task["status"] = SubTaskStatus.IN_PROGRESS
        task["started_at"] = datetime.now().isoformat()
        
        try:
            # Execute the task
            result_data = await self.process_task(state, task)
            
            # Calculate latency
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            
            # Update metrics
            self.total_executions += 1
            self.successful_executions += 1
            self.total_latency_ms += latency_ms
            
            # Update research-specific metrics
            self._update_research_metrics(result_data)
            
            # Create success result
            result = self._create_success_result(
                task_id=task_id,
                data=result_data,
                latency_ms=latency_ms
            )
            
            logger.info(
                f"[{self.name}] Completed task: {task_id} "
                f"in {latency_ms:.1f}ms"
            )
            
            return result
        
        except Exception as e:
            # Calculate latency even for errors
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            
            # Update metrics
            self.total_executions += 1
            self.failed_executions += 1
            
            # Log error
            logger.error(
                f"[{self.name}] Task failed: {task_id} - {str(e)}",
                exc_info=True
            )
            
            # Create error result
            result = self._create_error_result(
                task_id=task_id,
                error=str(e),
                latency_ms=latency_ms
            )
            
            return result
    
    def send_message(
        self,
        state: AgentState,
        receiver: AgentRole,
        content: Dict[str, Any]
    ):
        """
        Send a message to another agent.
        
        Args:
            state: Current state
            receiver: Receiving agent's role
            content: Message content
        """
        add_agent_message(
            state=state,
            sender=self.role,
            receiver=receiver,
            content=content
        )
        
        logger.debug(
            f"[{self.name}] Sent message to {receiver}: "
            f"{content.get('type', 'unknown')}"
        )
    
    def get_messages(
        self,
        state: AgentState,
        from_sender: Optional[AgentRole] = None
    ) -> List[Dict[str, Any]]:
        """
        Get messages addressed to this agent.
        
        Args:
            state: Current state
            from_sender: Optional filter by sender
        
        Returns:
            List of messages for this agent
        """
        messages = [
            msg for msg in state["agent_messages"]
            if msg["receiver"] == self.role
        ]
        
        if from_sender:
            messages = [
                msg for msg in messages
                if msg["sender"] == from_sender
            ]
        
        return messages
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for this agent.
        
        Returns:
            Dict with performance metrics including research-specific metrics
        """
        avg_latency = (
            self.total_latency_ms / self.total_executions
            if self.total_executions > 0
            else 0.0
        )
        
        success_rate = (
            (self.successful_executions / self.total_executions) * 100
            if self.total_executions > 0
            else 0.0
        )
        
        return {
            "agent_id": self.agent_id,
            "agent_name": self.name,
            "agent_role": self.role,
            "total_executions": self.total_executions,
            "successful_executions": self.successful_executions,
            "failed_executions": self.failed_executions,
            "success_rate": success_rate,
            "total_latency_ms": self.total_latency_ms,
            "avg_latency_ms": avg_latency,
            "total_papers_processed": self.total_papers_processed,
            "total_citations_handled": self.total_citations_handled,
            "domain": "scientific_literature"
        }
    
    # ========================================================================
    # PROTECTED HELPER METHODS
    # ========================================================================

    async def _call_llm(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,  # ← Novo parâmetro para determinismo
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Call LLM with a prompt using PydanticAgent.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt (uses default if None)
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            max_tokens: Maximum tokens to generate
        
        Returns:
            LLM response string
        """
        if system_prompt is None:
            system_prompt = self.get_system_prompt()
        
        if not self.model:
            self.log_error("LLM model name is not configured. Cannot call LLM.")
            return "Error: LLM not configured."

        try:
            # Prepare client config for different providers
            client_config = {}
            if self.model.startswith("groq:"):
                client_config = {
                    "base_url": os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
                    "api_key": os.getenv("LLM_API_KEY")
                }
            
            # Create agent with model string
            agent = PydanticAgent(
                model=self.model,
                system_prompt=system_prompt,
                **client_config
            )
            
            # Run with temperature control
            result = await agent.run(
                prompt,
                model_settings={
                    'temperature': temperature,
                    'max_tokens': max_tokens or 4096
                }
            )
            
            return result.data
            
        except Exception as e:
            self.log_error(f"LLM call failed: {e}", exc_info=True)
            return f"Error during LLM call: {e}"
        
    def _extract_from_state(
        self,
        state: AgentState,
        key: str,
        default: Any = None
    ) -> Any:
        """
        Safely extract data from state.
        
        Args:
            state: Current state
            key: Key to extract
            default: Default value if not found
        
        Returns:
            Extracted value or default
        """
        return state.get(key, default)
    
    def _update_research_metrics(self, result_data: Dict[str, Any]):
        """
        Update research-specific metrics from result data.
        
        Args:
            result_data: Result data from task execution
        """
        # Update papers processed
        if "unique_papers" in result_data:
            self.total_papers_processed += result_data["unique_papers"]
        elif "papers_cited" in result_data:
            self.total_papers_processed += result_data["papers_cited"]
        elif "papers_referenced" in result_data:
            self.total_papers_processed += result_data["papers_referenced"]
        
        # Update citations handled
        if "citations" in result_data:
            self.total_citations_handled += len(result_data["citations"])
        elif "sources_used" in result_data:
            self.total_citations_handled += len(result_data["sources_used"])

    # ========================================================================
    # RESULT FORMATTING METHODS
    # ========================================================================

    def _create_success_result(
        self,
        task_id: str,
        data: Dict[str, Any],
        latency_ms: float
    ) -> Dict[str, Any]:
        """
        Format a successful task result packet.
        
        Args:
            task_id: Task ID
            data: Result data
            latency_ms: Execution time
        
        Returns:
            Formatted success result
        """
        return {
            "success": True,
            "task_id": task_id,
            "latency_ms": latency_ms,
            "data": data
        }

    def _create_error_result(
        self,
        task_id: str,
        error: str,
        latency_ms: float
    ) -> Dict[str, Any]:
        """
        Format a failed task result packet.
        
        Args:
            task_id: Task ID
            error: Error message
            latency_ms: Execution time
        
        Returns:
            Formatted error result
        """
        return {
            "success": False,
            "task_id": task_id,
            "latency_ms": latency_ms,
            "error": error
        }
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def log_info(self, message: str, **kwargs):
        """Log info message with agent prefix."""
        logger.info(f"[{self.name}] {message}", **kwargs)

    def log_warning(self, message: str, **kwargs):
        """Log warning message with agent prefix."""
        logger.warning(f"[{self.name}] {message}", **kwargs)

    def log_error(self, message: str, exc_info: bool = False, **kwargs):
        """Log error message with agent prefix."""
        logger.error(f"[{self.name}] {message}", exc_info=exc_info, **kwargs)

    def log_debug(self, message: str, **kwargs):
        """Log debug message with agent prefix."""
        logger.debug(f"[{self.name}] {message}", **kwargs)
    
    def __repr__(self) -> str:
        """String representation of agent."""
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"role={self.role}, "
            f"id={self.agent_id})"
        )
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"{self.name} ({self.role})"


# ============================================================================
# AGENT REGISTRY
# ============================================================================

class AgentRegistry:
    """
    Registry for managing multiple agent instances in scientific literature system.
    
    Allows looking up agents by role and tracking all active agents.
    """
    
    def __init__(self):
        """Initialize empty registry."""
        self._agents: Dict[AgentRole, BaseAgent] = {}
        self._agents_by_id: Dict[str, BaseAgent] = {}
    
    def register(self, agent: BaseAgent):
        """
        Register an agent.
        
        Args:
            agent: Agent to register
        """
        self._agents[agent.role] = agent
        self._agents_by_id[agent.agent_id] = agent
        
        logger.info(f"Registered agent: {agent.name} (Role: {agent.role})")
    
    def get_by_role(self, role: AgentRole) -> Optional[BaseAgent]:
        """
        Get agent by role.
        
        Args:
            role: Agent role
        
        Returns:
            Agent instance or None
        """
        return self._agents.get(role)
    
    def get_by_id(self, agent_id: str) -> Optional[BaseAgent]:
        """
        Get agent by ID.
        
        Args:
            agent_id: Agent ID
        
        Returns:
            Agent instance or None
        """
        return self._agents_by_id.get(agent_id)
    
    def get_all(self) -> List[BaseAgent]:
        """Get all registered agents."""
        return list(self._agents.values())
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary for all agents.
        
        Returns:
            Dict with per-agent performance stats
        """
        return {
            agent.role: agent.get_performance_stats()
            for agent in self._agents.values()
        }
    
    def get_research_statistics(self) -> Dict[str, Any]:
        """
        Get research-specific statistics across all agents.
        
        Returns:
            Dict with aggregated research metrics
        """
        total_papers = sum(
            agent.total_papers_processed for agent in self._agents.values()
        )
        total_citations = sum(
            agent.total_citations_handled for agent in self._agents.values()
        )
        
        return {
            "total_papers_processed": total_papers,
            "total_citations_handled": total_citations,
            "domain": "scientific_literature"
        }
    
    def clear(self):
        """Clear all registered agents."""
        self._agents.clear()
        self._agents_by_id.clear()
        logger.info("Agent registry cleared")


# ============================================================================
# GLOBAL REGISTRY INSTANCE
# ============================================================================

_global_registry: Optional[AgentRegistry] = None

def get_registry() -> AgentRegistry:
    """
    Get the global agent registry.
    
    Returns:
        Global AgentRegistry instance
    """
    global _global_registry
    
    if _global_registry is None:
        _global_registry = AgentRegistry()
    
    return _global_registry


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

class ExampleAgent(BaseAgent):
    """Example agent implementation for testing scientific literature processing."""
    
    def __init__(self):
        """Initialize example agent."""
        super().__init__(
            role=AgentRole.RETRIEVAL,
            name="Example Research Agent",
            description="Example agent for scientific literature demonstration"
        )
    
    async def process_task(
        self,
        state: AgentState,
        task: SubTask
    ) -> Dict[str, Any]:
        """Process task (dummy implementation)."""
        self.log_info(f"Processing research task: {task['description']}")
        
        # Simulate paper retrieval
        import asyncio
        await asyncio.sleep(0.1)
        
        return {
            "status": "completed",
            "message": f"Task {task['task_id']} processed successfully",
            "unique_papers": 5,
            "citations": ["Paper 1", "Paper 2", "Paper 3"]
        }
    
    def can_handle_task(self, task: SubTask) -> bool:
        """Check if can handle task."""
        return task["assigned_agent"] == self.role
    
    def get_system_prompt(self) -> str:
        """Get system prompt."""
        return "You are an example agent for scientific literature processing demonstration."


async def example_usage():
    """Example showing how to use BaseAgent for scientific literature."""
    from .state import create_initial_state, add_subtask, TaskType
    
    # Create agent
    agent = ExampleAgent()
    
    # Create state
    state = create_initial_state(
        query="What are the key contributions of Transformers in NLP?",
        search_type="hybrid"
    )
    
    # Add task
    task = add_subtask(
        state,
        task_type=TaskType.VECTOR_SEARCH,
        description="Search for Transformer papers",
        parameters={"query": "Transformer NLP", "limit": 5},
        assigned_agent=AgentRole.RETRIEVAL
    )
    
    print(f"Created task: {task['task_id']}")
    
    # Execute task
    result = await agent.execute(state, task)
    
    print(f"\nTask result:")
    print(f"  Success: {result['success']}")
    print(f"  Latency: {result['latency_ms']:.1f}ms")
    print(f"  Papers Retrieved: {result['data'].get('unique_papers', 0)}")
    print(f"  Citations: {len(result['data'].get('citations', []))}")
    
    # Get performance stats
    stats = agent.get_performance_stats()
    print(f"\nAgent performance:")
    print(f"  Total executions: {stats['total_executions']}")
    print(f"  Success rate: {stats['success_rate']:.1f}%")
    print(f"  Avg latency: {stats['avg_latency_ms']:.1f}ms")
    print(f"  Papers processed: {stats['total_papers_processed']}")
    print(f"  Citations handled: {stats['total_citations_handled']}")
    print(f"  Domain: {stats['domain']}")
    
    # Test registry
    registry = get_registry()
    registry.register(agent)
    
    retrieved = registry.get_by_role(AgentRole.RETRIEVAL)
    print(f"\nRetrieved from registry: {retrieved}")
    
    # Get research statistics
    research_stats = registry.get_research_statistics()
    print(f"\nResearch statistics:")
    print(f"  Total papers: {research_stats['total_papers_processed']}")
    print(f"  Total citations: {research_stats['total_citations_handled']}")


if __name__ == "__main__":
    import asyncio
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run example
    asyncio.run(example_usage())
