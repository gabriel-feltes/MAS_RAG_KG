"""
Multi-Agent System package.
"""

from .graph import MASGraphExecutor
from .coordinator import CoordinatorAgent, create_coordinator
from .state import MASState, create_initial_state, QueryIntent, QueryComplexity

# Alias for compatibility
AgentState = MASState

__all__ = [
    "MASGraphExecutor",
    "CoordinatorAgent",
    "create_coordinator",
    "MASState",
    "AgentState",
    "create_initial_state",
    "QueryIntent",
    "QueryComplexity",
]
