"""
Agentic Planning System for MLE-Bench

A meta-agent system that analyzes ML engineering problems and dynamically
composes specialized agent workflows to solve them.

Key Components:
- Agent: Ai = {Li, Ri, Si, Ci, Hi}
- Plugin: Pj = {Fj, Cj, Uj}
- PlanningModule: Orchestrates workflow creation
- ExecutionEngine: Coordinates agent execution
"""

__version__ = "0.1.0"

from agents.planning_system.core.agent import Agent, AgentRole, AgentState, SpawnConfig
from agents.planning_system.core.plugin import Plugin, PluginFunctionality, PluginConfig, PluginConstraints
from agents.planning_system.core.llm import LLMConfig
from agents.planning_system.core.history import ExecutionHistory
from agents.planning_system.planning.planner import PlanningModule, WorkflowPlan
from agents.planning_system.execution.engine import WorkflowExecutionEngine

__all__ = [
    "Agent",
    "AgentRole",
    "AgentState",
    "SpawnConfig",
    "Plugin",
    "PluginFunctionality",
    "PluginConfig",
    "PluginConstraints",
    "LLMConfig",
    "ExecutionHistory",
    "PlanningModule",
    "WorkflowPlan",
    "WorkflowExecutionEngine",
]
