"""
Agent State

Tracks the internal state of an agent during execution (Si component).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional


@dataclass
class AgentState:
    """
    Tracks the internal state of an agent during execution (Si).

    This includes current status, memory, observations, and outputs.
    """

    # Execution state
    status: str = "idle"  # idle, planning, acting, waiting, completed, failed
    current_step: int = 0  # Current step number

    # Memory and context
    working_memory: Dict[str, Any] = field(
        default_factory=dict
    )  # Short-term memory for current task
    observations: List[Dict] = field(
        default_factory=list
    )  # History of observations

    # Planning
    plan: List[Dict] = field(default_factory=list)  # Planned actions
    current_goal: str = ""  # Current objective

    # Results
    outputs: Dict[str, Any] = field(default_factory=dict)  # Final outputs
    artifacts: List[str] = field(default_factory=list)  # File paths to generated artifacts

    # Metrics
    actions_taken: int = 0
    errors_encountered: int = 0
    time_elapsed: float = 0.0

    # Inter-agent coordination
    messages_received: List[Dict] = field(default_factory=list)
    messages_sent: List[Dict] = field(default_factory=list)

    def update_status(self, new_status: str):
        """Update agent status"""
        valid_statuses = ["idle", "planning", "acting", "waiting", "completed", "failed"]
        if new_status not in valid_statuses:
            raise ValueError(
                f"Invalid status '{new_status}'. Must be one of {valid_statuses}"
            )
        self.status = new_status

    def add_observation(self, observation: Dict):
        """Add an observation to history"""
        self.observations.append(observation)

    def get_recent_observations(self, n: int = 5) -> List[Dict]:
        """Get the n most recent observations"""
        return self.observations[-n:]

    def set_goal(self, goal: str):
        """Set current goal"""
        self.current_goal = goal

    def add_to_plan(self, action: Dict):
        """Add an action to the plan"""
        self.plan.append(action)

    def get_next_planned_action(self) -> Optional[Dict]:
        """Get next action from plan (FIFO)"""
        if self.plan:
            return self.plan.pop(0)
        return None

    def add_output(self, key: str, value: Any):
        """Add to outputs"""
        self.outputs[key] = value

    def add_artifact(self, filepath: str):
        """Add artifact file path"""
        self.artifacts.append(filepath)

    def receive_message(self, from_agent: str, message: str):
        """Receive message from another agent"""
        self.messages_received.append(
            {"from": from_agent, "message": message, "timestamp": self.current_step}
        )

    def send_message(self, to_agent: str, message: str):
        """Record sent message"""
        self.messages_sent.append(
            {"to": to_agent, "message": message, "timestamp": self.current_step}
        )

    def increment_step(self):
        """Increment step counter"""
        self.current_step += 1
        self.actions_taken += 1

    def record_error(self):
        """Increment error counter"""
        self.errors_encountered += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "status": self.status,
            "current_step": self.current_step,
            "working_memory": self.working_memory,
            "current_goal": self.current_goal,
            "outputs": self.outputs,
            "artifacts": self.artifacts,
            "actions_taken": self.actions_taken,
            "errors_encountered": self.errors_encountered,
            "time_elapsed": self.time_elapsed,
        }

    def __repr__(self) -> str:
        return f"AgentState(status={self.status}, step={self.current_step}, actions={self.actions_taken})"
