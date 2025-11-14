"""
Plugin System

Defines plugins (Pj = {Fj, Cj, Uj}) that provide capabilities to agents.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Callable, Optional
import time
import signal
from contextlib import contextmanager


@dataclass
class Action:
    """
    A single action provided by a plugin.
    """

    action_name: str  # e.g., "read_csv", "train_model"
    description: str  # What this action does

    # Interface
    parameters: Dict[str, Dict] = field(
        default_factory=dict
    )  # param_name → schema
    returns: Dict = field(default_factory=dict)  # Return value schema

    # Implementation
    handler: Optional[Callable] = None  # Function that executes the action

    # Metadata
    estimated_time: float = 1.0  # Seconds (approximate)
    requires_gpu: bool = False


@dataclass
class PluginFunctionality:
    """
    Defines the actions/capabilities provided by a plugin (Fj).
    """

    # Identity
    plugin_id: str  # Unique identifier
    plugin_name: str  # Human-readable name
    description: str  # What the plugin does

    # Actions
    actions: Dict[str, Action] = field(default_factory=dict)  # action_name → Action

    # Dependencies
    required_packages: List[str] = field(default_factory=list)
    required_files: List[str] = field(default_factory=list)

    def register_action(self, action: Action):
        """Register a new action"""
        self.actions[action.action_name] = action

    def has_action(self, action_name: str) -> bool:
        """Check if action exists"""
        return action_name in self.actions


@dataclass
class PluginConfig:
    """
    Configuration parameters for plugin behavior (Cj).
    """

    # Runtime settings
    timeout: int = 300  # Max execution time per action (seconds)
    max_retries: int = 2  # Retry failed actions

    # Resource limits
    max_memory_mb: int = 4096  # Memory limit
    max_cpu_cores: int = 4  # CPU core limit
    allow_gpu: bool = False  # GPU access

    # Data access
    allowed_read_paths: List[str] = field(default_factory=list)
    allowed_write_paths: List[str] = field(default_factory=list)

    # Logging
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    save_intermediate_results: bool = True

    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PluginConstraints:
    """
    Defines when and how a plugin can be used (Uj).
    """

    # Availability constraints
    requires_internet: bool = False
    requires_gpu: bool = False
    min_memory_mb: int = 512
    min_disk_gb: int = 1

    # Usage limits
    max_calls_per_agent: int = -1  # -1 = unlimited
    max_concurrent_calls: int = 1  # Parallel execution limit

    # Temporal constraints
    cooldown_seconds: float = 0.0  # Minimum time between calls

    # Data constraints
    max_input_size_mb: float = 100.0  # Max input data size
    max_output_size_mb: float = 100.0  # Max output data size

    # Security
    sandbox_mode: bool = True  # Execute in sandbox
    allowed_network_hosts: List[str] = field(default_factory=list)

    # Dependencies
    prerequisite_plugins: List[str] = field(default_factory=list)
    incompatible_plugins: List[str] = field(default_factory=list)


class TimeoutError(Exception):
    """Raised when action execution exceeds timeout"""

    pass


@contextmanager
def timeout_handler(seconds: int):
    """Context manager for timeout"""

    def timeout_signal_handler(signum, frame):
        raise TimeoutError(f"Execution exceeded {seconds} seconds")

    # Set the signal handler and alarm
    signal.signal(signal.SIGALRM, timeout_signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)  # Disable the alarm


@dataclass
class Plugin:
    """
    Complete plugin specification: Pj = {Fj, Cj, Uj}
    """

    functionality: PluginFunctionality  # Fj: What it does
    config: PluginConfig  # Cj: How it's configured
    constraints: PluginConstraints  # Uj: Usage limits

    # Runtime state
    call_count: int = 0
    last_call_time: float = 0.0
    is_initialized: bool = False

    def initialize(self):
        """Initialize plugin (load models, etc.)"""
        if not self.is_initialized:
            # Verify dependencies
            self._check_dependencies()
            self.is_initialized = True

    def _check_dependencies(self):
        """Check if required packages are available"""
        for package in self.functionality.required_packages:
            try:
                __import__(package)
            except ImportError:
                raise ImportError(
                    f"Plugin '{self.functionality.plugin_id}' requires package '{package}'"
                )

    def execute_action(
        self, action_name: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute an action with constraint checking.

        Returns:
            Dict with 'success', 'result', and optionally 'error' keys
        """
        # 1. Initialize if needed
        if not self.is_initialized:
            self.initialize()

        # 2. Validate constraints
        try:
            self._check_constraints(action_name)
        except Exception as e:
            return {"success": False, "error": str(e), "result": None}

        # 3. Get action handler
        if action_name not in self.functionality.actions:
            return {
                "success": False,
                "error": f"Action '{action_name}' not found in plugin",
                "result": None,
            }

        action = self.functionality.actions[action_name]

        # 4. Validate parameters
        try:
            self._validate_params(action, params)
        except Exception as e:
            return {"success": False, "error": f"Invalid parameters: {e}", "result": None}

        # 5. Execute with timeout and error handling
        try:
            result = self._execute_with_timeout(action.handler, params, self.config.timeout)
            self.call_count += 1
            self.last_call_time = time.time()
            return {"success": True, "result": result, "error": None}

        except TimeoutError as e:
            return {"success": False, "error": str(e), "result": None}

        except Exception as e:
            # Retry logic
            if self.config.max_retries > 0:
                return self._retry_execution(action, params)
            else:
                return {
                    "success": False,
                    "error": f"{type(e).__name__}: {str(e)}",
                    "result": None,
                }

    def _check_constraints(self, action_name: str):
        """Verify usage constraints before execution"""
        # Check call limit
        if self.constraints.max_calls_per_agent != -1:
            if self.call_count >= self.constraints.max_calls_per_agent:
                raise RuntimeError("Plugin call limit exceeded")

        # Check cooldown
        if self.constraints.cooldown_seconds > 0:
            elapsed = time.time() - self.last_call_time
            if elapsed < self.constraints.cooldown_seconds:
                raise RuntimeError(
                    f"Cooldown period not met (waited {elapsed:.1f}s, need {self.constraints.cooldown_seconds}s)"
                )

        # Check GPU availability
        if self.constraints.requires_gpu:
            try:
                import torch

                if not torch.cuda.is_available():
                    raise RuntimeError("Plugin requires GPU but none available")
            except ImportError:
                raise RuntimeError("Plugin requires GPU but torch not available")

    def _validate_params(self, action: Action, params: Dict[str, Any]):
        """Validate parameters against action schema"""
        for param_name, param_schema in action.parameters.items():
            if param_schema.get("required", False) and param_name not in params:
                raise ValueError(f"Required parameter '{param_name}' missing")

    def _execute_with_timeout(
        self, handler: Callable, params: Dict[str, Any], timeout: int
    ) -> Any:
        """Execute handler with timeout protection"""
        if timeout > 0:
            try:
                with timeout_handler(timeout):
                    return handler(**params)
            except TimeoutError:
                raise
        else:
            return handler(**params)

    def _retry_execution(
        self, action: Action, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Retry execution with exponential backoff"""
        for attempt in range(self.config.max_retries):
            wait_time = 2**attempt  # Exponential backoff
            time.sleep(wait_time)

            try:
                result = self._execute_with_timeout(
                    action.handler, params, self.config.timeout
                )
                return {"success": True, "result": result, "error": None}
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    # Last attempt failed
                    return {
                        "success": False,
                        "error": f"Failed after {self.config.max_retries} retries: {str(e)}",
                        "result": None,
                    }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "plugin_id": self.functionality.plugin_id,
            "plugin_name": self.functionality.plugin_name,
            "description": self.functionality.description,
            "actions": list(self.functionality.actions.keys()),
            "config": {
                "timeout": self.config.timeout,
                "max_retries": self.config.max_retries,
            },
            "call_count": self.call_count,
        }

    def __repr__(self) -> str:
        return f"Plugin(id={self.functionality.plugin_id}, actions={len(self.functionality.actions)}, calls={self.call_count})"
