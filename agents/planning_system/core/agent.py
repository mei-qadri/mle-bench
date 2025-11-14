"""
Agent System

Defines agents (Ai = {Li, Ri, Si, Ci, Hi}) that execute tasks.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from pathlib import Path
import json

from agents.planning_system.core.llm import LLMConfig
from agents.planning_system.core.state import AgentState
from agents.planning_system.core.history import ExecutionHistory
from agents.planning_system.core.plugin import Plugin


@dataclass
class AgentRole:
    """
    Defines the role and responsibilities of an agent (Ri).
    """

    # Identity
    role_name: str  # e.g., "DataExplorer", "ModelTrainer"
    description: str  # Human-readable role description

    # Capabilities
    allowed_actions: List[str] = field(default_factory=list)  # Action types
    available_plugins: List[str] = field(default_factory=list)  # Plugin IDs

    # Constraints
    max_steps: int = 100  # Maximum action steps
    max_time_seconds: int = 3600  # Time budget (1 hour default)
    resource_limits: Dict[str, Any] = field(default_factory=dict)

    # Communication
    can_delegate: bool = True  # Can request help from other agents
    can_broadcast: bool = False  # Can send messages to all agents

    # Success criteria
    success_condition: str = ""  # Description of when role is complete
    output_schema: Dict = field(default_factory=dict)  # Expected outputs

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "role_name": self.role_name,
            "description": self.description,
            "allowed_actions": self.allowed_actions,
            "available_plugins": self.available_plugins,
            "max_steps": self.max_steps,
            "max_time_seconds": self.max_time_seconds,
            "success_condition": self.success_condition,
        }


@dataclass
class SpawnConfig:
    """
    Configuration for agent spawning capabilities (Ci).
    """

    can_spawn: bool = False  # Whether agent can create children
    max_children: int = 0  # Maximum child agents
    allowed_child_roles: List[str] = field(default_factory=list)
    spawn_strategy: str = "sequential"  # sequential, parallel, adaptive

    # Resource allocation to children
    time_budget_per_child: int = 1800
    step_budget_per_child: int = 50

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "can_spawn": self.can_spawn,
            "max_children": self.max_children,
            "spawn_strategy": self.spawn_strategy,
        }


@dataclass
class Agent:
    """
    Complete agent specification: Ai = {Li, Ri, Si, Ci, Hi}
    """

    # Core components
    agent_id: str  # Unique identifier
    llm_config: LLMConfig  # Li: Language model configuration
    role: AgentRole  # Ri: Role specification
    state: AgentState  # Si: Internal state
    spawn_config: SpawnConfig  # Ci: Spawn capability
    history: ExecutionHistory  # Hi: Execution history

    # Runtime
    plugins: Dict[str, Plugin] = field(default_factory=dict)  # plugin_id → Plugin
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)

    # Environment
    workspace_dir: Optional[Path] = None
    data_dir: Optional[Path] = None

    # LLM client (initialized lazily)
    _llm_client: Optional[Any] = None

    def initialize(self, workspace_dir: Path, data_dir: Path):
        """Initialize agent with workspace"""
        self.workspace_dir = workspace_dir
        self.data_dir = data_dir

        # Initialize plugins
        for plugin in self.plugins.values():
            plugin.initialize()

        # Initialize LLM client
        self._initialize_llm_client()

        # Set start time
        from datetime import datetime

        self.history.start_time = datetime.now()
        self.state.update_status("idle")

    def _initialize_llm_client(self):
        """Initialize LLM client based on provider"""
        if self.llm_config.provider == "openai":
            try:
                from openai import OpenAI

                self._llm_client = OpenAI(api_key=self.llm_config.api_key)
            except ImportError:
                raise ImportError("OpenAI package not installed. Run: pip install openai")

        elif self.llm_config.provider == "anthropic":
            try:
                from anthropic import Anthropic

                self._llm_client = Anthropic(api_key=self.llm_config.api_key)
            except ImportError:
                raise ImportError(
                    "Anthropic package not installed. Run: pip install anthropic"
                )

        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_config.provider}")

    def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main agent loop: observe → reason → act

        Args:
            observation: Current observation from environment

        Returns:
            Result of action execution
        """
        # 1. Update state with observation
        self.state.add_observation(observation)
        self.history.add_observation(observation)

        # 2. Generate next action using LLM
        self.state.update_status("planning")
        action = self._generate_action(observation)

        # 3. Execute action via plugins
        self.state.update_status("acting")
        result = self._execute_action(action)

        # 4. Record in history
        self.history.add_action(
            action["action_name"], action.get("parameters", {}), result
        )

        # 5. Update state
        self.state.increment_step()

        return result

    def _generate_action(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate next action using LLM reasoning.

        This uses the LLM to decide what action to take based on:
        - Current goal
        - Recent observations
        - Available actions (from plugins)
        """
        # Build prompt
        prompt = self._build_action_generation_prompt(observation)

        # Call LLM
        if self.llm_config.provider == "openai":
            response = self._call_openai(prompt)
        elif self.llm_config.provider == "anthropic":
            response = self._call_anthropic(prompt)
        else:
            raise ValueError(f"Unsupported provider: {self.llm_config.provider}")

        # Parse response into action
        action = self._parse_action_from_response(response)

        # Record thought
        if "thought" in action:
            self.history.add_thought(action["thought"])

        return action

    def _build_action_generation_prompt(self, observation: Dict[str, Any]) -> str:
        """Build prompt for LLM to generate next action"""
        # Get available actions from plugins
        available_actions = []
        for plugin in self.plugins.values():
            for action_name, action in plugin.functionality.actions.items():
                available_actions.append(
                    {
                        "name": action_name,
                        "description": action.description,
                        "parameters": action.parameters,
                    }
                )

        prompt = f"""You are an AI agent with the role: {self.role.role_name}
Description: {self.role.description}

Current goal: {self.state.current_goal}
Current step: {self.state.current_step}/{self.role.max_steps}

Recent observations:
{json.dumps(self.state.get_recent_observations(3), indent=2)}

Latest observation:
{json.dumps(observation, indent=2)}

Available actions:
{json.dumps(available_actions, indent=2)}

Based on the above, decide on the next action to take.
Respond in JSON format:
{{
    "thought": "your reasoning about what to do next",
    "action_name": "name of the action to execute",
    "parameters": {{"param1": "value1", ...}}
}}

If the goal is complete, use action_name: "complete".
"""
        return prompt

    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API"""
        response = self._llm_client.chat.completions.create(
            model=self.llm_config.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.llm_config.temperature,
            max_tokens=self.llm_config.max_tokens,
        )

        self.history.api_calls += 1
        self.history.tokens_used += response.usage.total_tokens

        return response.choices[0].message.content

    def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic API"""
        response = self._llm_client.messages.create(
            model=self.llm_config.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.llm_config.temperature,
            max_tokens=self.llm_config.max_tokens,
        )

        self.history.api_calls += 1
        self.history.tokens_used += response.usage.input_tokens + response.usage.output_tokens

        return response.content[0].text

    def _parse_action_from_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into action dict"""
        try:
            # Try to parse as JSON
            action = json.loads(response)
            return action
        except json.JSONDecodeError:
            # Fallback: extract JSON from response
            import re

            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass

            # If all else fails, return error action
            return {
                "thought": "Failed to parse LLM response",
                "action_name": "error",
                "parameters": {"error": "Invalid response format"},
            }

    def _execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an action using appropriate plugin"""
        action_name = action.get("action_name")
        parameters = action.get("parameters", {})

        # Special actions
        if action_name == "complete":
            self.state.update_status("completed")
            return {"success": True, "message": "Goal completed"}

        if action_name == "error":
            self.state.record_error()
            return {"success": False, "error": parameters.get("error")}

        # Find plugin that has this action
        for plugin in self.plugins.values():
            if plugin.functionality.has_action(action_name):
                result = plugin.execute_action(action_name, parameters)

                # Track errors
                if not result.get("success", False):
                    self.state.record_error()
                    self.history.add_error(
                        Exception(result.get("error", "Unknown error")),
                        context=f"Action: {action_name}",
                    )

                return result

        # Action not found
        error_msg = f"No plugin found for action '{action_name}'"
        self.state.record_error()
        return {"success": False, "error": error_msg}

    def spawn_child(self, child_role: AgentRole) -> "Agent":
        """Spawn a child agent with specified role"""
        if not self.spawn_config.can_spawn:
            raise ValueError(f"Agent {self.agent_id} cannot spawn children")

        if len(self.children_ids) >= self.spawn_config.max_children:
            raise ValueError(
                f"Agent {self.agent_id} has reached max children limit"
            )

        # Create child agent
        child = Agent(
            agent_id=f"{self.agent_id}_child_{len(self.children_ids)}",
            llm_config=self.llm_config,  # Inherit LLM config
            role=child_role,
            state=AgentState(),
            spawn_config=SpawnConfig(can_spawn=False),  # Children cannot spawn
            history=ExecutionHistory(),
            parent_id=self.agent_id,
        )

        self.children_ids.append(child.agent_id)
        self.history.add_milestone(
            f"Spawned child agent: {child.agent_id}", {"role": child_role.role_name}
        )

        return child

    def send_message(self, to_agent_id: str, message: str):
        """Send message to another agent"""
        self.state.send_message(to_agent_id, message)
        self.history.add_message(self.agent_id, to_agent_id, message)

    def receive_message(self, from_agent_id: str, message: str):
        """Receive message from another agent"""
        self.state.receive_message(from_agent_id, message)
        self.history.add_message(from_agent_id, self.agent_id, message)

    def finalize(self):
        """Finalize agent execution"""
        from datetime import datetime

        self.history.end_time = datetime.now()

        # Save history
        if self.workspace_dir:
            history_file = self.workspace_dir / f"{self.agent_id}_history.json"
            self.history.to_json(str(history_file))

            # Also save as markdown
            md_file = self.workspace_dir / f"{self.agent_id}_history.md"
            with open(md_file, "w") as f:
                f.write(self.history.to_markdown())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "agent_id": self.agent_id,
            "role": self.role.to_dict(),
            "llm_config": self.llm_config.to_dict(),
            "state": self.state.to_dict(),
            "spawn_config": self.spawn_config.to_dict(),
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "plugins": [p.to_dict() for p in self.plugins.values()],
        }

    def __repr__(self) -> str:
        return f"Agent(id={self.agent_id}, role={self.role.role_name}, status={self.state.status})"
