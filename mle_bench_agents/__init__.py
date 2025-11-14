"""
MLE-bench Multi-Agent System
Version: 1.0
Phase I: Core Infrastructure
"""

__version__ = "1.0.0"
__author__ = "Anthropic MLE-bench Team"

from mle_bench_agents.core.agent import Agent
from mle_bench_agents.core.message import Message, MessageType, MessagePriority
from mle_bench_agents.core.state import GlobalState, AgentState
from mle_bench_agents.core.queue import MessageQueue

__all__ = [
    "Agent",
    "Message",
    "MessageType",
    "MessagePriority",
    "GlobalState",
    "AgentState",
    "MessageQueue",
]
