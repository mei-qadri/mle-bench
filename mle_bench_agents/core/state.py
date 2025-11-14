"""
State management for multi-agent system.

Manages global state and agent-specific states with versioning and persistence.
"""

import json
import time
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
from threading import RLock


@dataclass
class AgentState:
    """State specific to an individual agent."""

    agent_id: str
    agent_type: str
    status: str = "idle"  # idle, active, paused, completed, failed
    current_task: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    execution_time: float = 0.0
    custom_state: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentState":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class GlobalState:
    """
    Global state managed by the Orchestrator.

    Tracks competition information, workflow progress, models, predictions,
    resources, and errors across all agents.
    """

    # Competition information
    competition_id: str = ""
    competition_type: str = ""  # CV, NLP, tabular, audio, multimodal
    complexity: str = ""  # low, medium, high
    metric_name: str = ""
    metric_direction: str = ""  # maximize, minimize

    # Time management
    start_time: float = field(default_factory=time.time)
    time_budget_total: int = 86400  # 24 hours default
    phase_time_allocations: Dict[str, int] = field(default_factory=dict)

    # Resource tracking
    resource_budget: Dict[str, Any] = field(default_factory=dict)
    resource_usage: Dict[str, Any] = field(default_factory=dict)

    # Workflow state
    current_phase: str = "initialized"
    phase_status: Dict[str, str] = field(default_factory=dict)

    # Agent states
    active_agents: Dict[str, AgentState] = field(default_factory=dict)
    agent_results: Dict[str, Any] = field(default_factory=dict)

    # Data & Artifacts
    data_summary: Dict[str, Any] = field(default_factory=dict)
    feature_pipeline: Optional[Any] = None
    cv_folds: Optional[Any] = None

    # Model tracking
    trained_models: List[Dict[str, Any]] = field(default_factory=list)
    predictions: Dict[str, Any] = field(default_factory=dict)
    ensemble_predictions: Optional[Any] = None
    best_cv_score: Optional[float] = None

    # Performance tracking
    leaderboard_target: Dict[str, float] = field(default_factory=dict)
    submission_history: List[Dict[str, Any]] = field(default_factory=list)

    # Error tracking
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    fallback_activated: bool = False

    # Custom state for extensions
    custom: Dict[str, Any] = field(default_factory=dict)

    @property
    def time_elapsed(self) -> int:
        """Calculate elapsed time in seconds."""
        return int(time.time() - self.start_time)

    @property
    def time_remaining(self) -> int:
        """Calculate remaining time in seconds."""
        return max(0, self.time_budget_total - self.time_elapsed)

    @property
    def time_used_percent(self) -> float:
        """Calculate percentage of time used."""
        return (self.time_elapsed / self.time_budget_total) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary (for serialization)."""
        state_dict = {}
        for key, value in asdict(self).items():
            if key == "active_agents":
                state_dict[key] = {
                    agent_id: agent_state.to_dict()
                    for agent_id, agent_state in value.items()
                }
            elif key in ["feature_pipeline", "cv_folds", "ensemble_predictions"]:
                # Skip objects that can't be easily serialized
                state_dict[key] = None
            else:
                state_dict[key] = value

        # Add computed properties
        state_dict["time_elapsed"] = self.time_elapsed
        state_dict["time_remaining"] = self.time_remaining
        state_dict["time_used_percent"] = self.time_used_percent

        return state_dict

    def update(self, updates: Dict[str, Any]) -> None:
        """Update state with new values."""
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)


class StateManager:
    """
    Manages state with versioning, persistence, and thread-safe access.
    """

    def __init__(
        self,
        state_file: Optional[Path] = None,
        enable_versioning: bool = True,
        max_versions: int = 100,
        auto_save_interval: int = 300  # 5 minutes
    ):
        """
        Initialize state manager.

        Args:
            state_file: Path to state persistence file
            enable_versioning: Whether to keep version history
            max_versions: Maximum number of versions to keep
            auto_save_interval: Auto-save interval in seconds
        """
        self.state = GlobalState()
        self.state_file = state_file
        self.enable_versioning = enable_versioning
        self.max_versions = max_versions
        self.auto_save_interval = auto_save_interval

        # Version history
        self.versions: List[Dict[str, Any]] = []

        # Thread safety
        self._lock = RLock()

        # Last save time
        self.last_save_time = time.time()

    def get_state(self) -> GlobalState:
        """Get current global state (thread-safe)."""
        with self._lock:
            return self.state

    def update_state(self, updates: Dict[str, Any]) -> None:
        """
        Update global state (thread-safe).

        Args:
            updates: Dictionary of state updates
        """
        with self._lock:
            # Save version before update
            if self.enable_versioning:
                self._save_version()

            # Update state
            self.state.update(updates)

            # Auto-save if needed
            if time.time() - self.last_save_time > self.auto_save_interval:
                self.save_state()

    def update_agent_state(
        self,
        agent_id: str,
        status: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """
        Update specific agent state.

        Args:
            agent_id: Agent ID
            status: Agent status
            result: Agent result
            **kwargs: Additional state updates
        """
        with self._lock:
            if agent_id not in self.state.active_agents:
                return

            agent_state = self.state.active_agents[agent_id]

            if status:
                agent_state.status = status

            if result:
                agent_state.result = result

            for key, value in kwargs.items():
                if hasattr(agent_state, key):
                    setattr(agent_state, key, value)

    def register_agent(self, agent_id: str, agent_type: str) -> None:
        """Register a new agent."""
        with self._lock:
            self.state.active_agents[agent_id] = AgentState(
                agent_id=agent_id,
                agent_type=agent_type,
                status="active",
                start_time=time.time()
            )

    def deregister_agent(self, agent_id: str) -> None:
        """Deregister an agent."""
        with self._lock:
            if agent_id in self.state.active_agents:
                agent_state = self.state.active_agents[agent_id]
                agent_state.status = "completed"
                agent_state.end_time = time.time()
                agent_state.execution_time = agent_state.end_time - agent_state.start_time

    def add_error(self, agent_id: str, error_type: str, error_message: str, **kwargs) -> None:
        """Add error to error log."""
        with self._lock:
            error_entry = {
                "timestamp": time.time(),
                "agent_id": agent_id,
                "error_type": error_type,
                "error_message": error_message,
                "phase": self.state.current_phase,
                **kwargs
            }
            self.state.errors.append(error_entry)

    def add_warning(self, message: str, **kwargs) -> None:
        """Add warning to warning log."""
        with self._lock:
            warning_entry = {
                "timestamp": time.time(),
                "message": message,
                "phase": self.state.current_phase,
                **kwargs
            }
            self.state.warnings.append(warning_entry)

    def _save_version(self) -> None:
        """Save current state as a version."""
        version = {
            "timestamp": time.time(),
            "state": deepcopy(self.state.to_dict())
        }
        self.versions.append(version)

        # Trim versions if exceeding max
        if len(self.versions) > self.max_versions:
            self.versions = self.versions[-self.max_versions:]

    def save_state(self, file_path: Optional[Path] = None) -> None:
        """
        Save state to disk.

        Args:
            file_path: Optional custom file path
        """
        save_path = file_path or self.state_file
        if not save_path:
            return

        with self._lock:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            state_dict = self.state.to_dict()

            with open(save_path, 'w') as f:
                json.dump(state_dict, f, indent=2)

            self.last_save_time = time.time()

    def load_state(self, file_path: Optional[Path] = None) -> None:
        """
        Load state from disk.

        Args:
            file_path: Optional custom file path
        """
        load_path = file_path or self.state_file
        if not load_path or not Path(load_path).exists():
            return

        with self._lock:
            with open(load_path, 'r') as f:
                state_dict = json.load(f)

            # Reconstruct GlobalState
            # Note: Some fields like feature_pipeline won't be restored
            for key, value in state_dict.items():
                if key == "active_agents":
                    self.state.active_agents = {
                        agent_id: AgentState.from_dict(agent_dict)
                        for agent_id, agent_dict in value.items()
                    }
                elif hasattr(self.state, key) and key not in ["time_elapsed", "time_remaining", "time_used_percent"]:
                    setattr(self.state, key, value)

    def get_version(self, index: int = -1) -> Optional[Dict[str, Any]]:
        """
        Get a specific version of state.

        Args:
            index: Version index (-1 for latest)

        Returns:
            State version or None if not found
        """
        with self._lock:
            if not self.versions:
                return None

            try:
                return self.versions[index]
            except IndexError:
                return None

    def reset(self) -> None:
        """Reset state to initial values."""
        with self._lock:
            self.state = GlobalState()
            self.versions = []
