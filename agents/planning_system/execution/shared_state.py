"""
Shared State

Shared state accessible to all agents during execution.
"""

import threading
from typing import Dict, Any, Optional
from datetime import datetime


class SharedState:
    """
    Shared state accessible to all agents

    Thread-safe key-value store for inter-agent coordination.
    """

    def __init__(self):
        self.data: Dict[str, Any] = {}
        self.lock = threading.RLock()
        self.update_history: Dict[str, list] = {}  # key → list of (timestamp, value) tuples

    def set(self, key: str, value: Any):
        """Set a value in shared state"""
        with self.lock:
            self.data[key] = value

            # Track update history
            if key not in self.update_history:
                self.update_history[key] = []
            self.update_history[key].append((datetime.now(), value))

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from shared state"""
        with self.lock:
            return self.data.get(key, default)

    def update(self, updates: Dict[str, Any]):
        """Update multiple keys at once"""
        with self.lock:
            for key, value in updates.items():
                self.set(key, value)

    def delete(self, key: str):
        """Delete a key from shared state"""
        with self.lock:
            if key in self.data:
                del self.data[key]

    def has_key(self, key: str) -> bool:
        """Check if key exists"""
        with self.lock:
            return key in self.data

    def keys(self) -> list:
        """Get all keys"""
        with self.lock:
            return list(self.data.keys())

    def get_all(self) -> Dict[str, Any]:
        """Get all data"""
        with self.lock:
            return self.data.copy()

    def get_history(self, key: str) -> list:
        """Get update history for a key"""
        with self.lock:
            return self.update_history.get(key, [])

    def clear(self):
        """Clear all data"""
        with self.lock:
            self.data.clear()
            self.update_history.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        with self.lock:
            return {
                "data": self.data.copy(),
                "num_keys": len(self.data),
            }

    def __repr__(self) -> str:
        with self.lock:
            return f"SharedState(keys={len(self.data)})"


class AgentRegistry:
    """
    Registry of active agents

    Tracks which agents are running and their status.
    """

    def __init__(self):
        self.agents: Dict[str, Dict[str, Any]] = {}  # agent_id → agent info
        self.lock = threading.RLock()

    def register(self, agent_id: str, agent_info: Dict[str, Any]):
        """Register an agent"""
        with self.lock:
            self.agents[agent_id] = {
                "info": agent_info,
                "registered_at": datetime.now(),
                "status": "registered",
            }

    def update_status(self, agent_id: str, status: str):
        """Update agent status"""
        with self.lock:
            if agent_id in self.agents:
                self.agents[agent_id]["status"] = status
                self.agents[agent_id]["last_updated"] = datetime.now()

    def get_agent(self, agent_id: str) -> Optional[Dict]:
        """Get agent info"""
        with self.lock:
            return self.agents.get(agent_id)

    def get_all_agents(self) -> Dict[str, Dict]:
        """Get all registered agents"""
        with self.lock:
            return self.agents.copy()

    def get_agents_by_status(self, status: str) -> Dict[str, Dict]:
        """Get agents with specific status"""
        with self.lock:
            return {
                agent_id: info
                for agent_id, info in self.agents.items()
                if info["status"] == status
            }

    def unregister(self, agent_id: str):
        """Unregister an agent"""
        with self.lock:
            if agent_id in self.agents:
                del self.agents[agent_id]

    def clear(self):
        """Clear all agents"""
        with self.lock:
            self.agents.clear()

    def __repr__(self) -> str:
        with self.lock:
            return f"AgentRegistry(agents={len(self.agents)})"
