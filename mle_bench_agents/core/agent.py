"""
Base Agent class for multi-agent system.

All specialized agents inherit from this base class.
"""

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

from mle_bench_agents.core.message import (
    Message,
    MessageType,
    create_request_message,
)
from mle_bench_agents.core.queue import MessageQueue
from mle_bench_agents.core.state import StateManager, AgentState


class Agent(ABC):
    """
    Base class for all agents in the multi-agent system.

    Each agent has:
    - Unique ID and type
    - Access to message queue for communication
    - Access to state manager
    - System prompt defining behavior
    - Configuration parameters
    """

    def __init__(
        self,
        agent_id: str,
        agent_type: str,
        message_queue: MessageQueue,
        state_manager: StateManager,
        config: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize agent.

        Args:
            agent_id: Unique agent identifier
            agent_type: Type of agent (orchestrator, analysis, etc.)
            message_queue: Shared message queue
            state_manager: Shared state manager
            config: Agent configuration
            system_prompt: System prompt text
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.message_queue = message_queue
        self.state_manager = state_manager
        self.config = config or {}
        self.system_prompt = system_prompt

        # Agent-specific state
        self.local_state: Dict[str, Any] = {}

        # Subscribe to broadcasts by default
        self.message_queue.subscribe_to_broadcasts(self.agent_id)

        # Register with state manager
        self.state_manager.register_agent(self.agent_id, self.agent_type)

        # Logging
        self.logger = None  # Will be set by logging system

    @abstractmethod
    def process_message(self, message: Message) -> Optional[Message]:
        """
        Process incoming message.

        Args:
            message: Incoming message

        Returns:
            Response message or None
        """
        pass

    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute agent's main task.

        Args:
            context: Task context and parameters

        Returns:
            Execution result
        """
        pass

    def send_message(self, message: Message) -> bool:
        """
        Send a message via the message queue.

        Args:
            message: Message to send

        Returns:
            True if message was queued successfully
        """
        return self.message_queue.send(message)

    def send_request(
        self,
        receiver_id: str,
        action: str,
        priority: int = 5,
        timeout: float = 3600.0,
        **kwargs
    ) -> Message:
        """
        Send a request message.

        Args:
            receiver_id: Target agent ID
            action: Action to perform
            priority: Message priority
            timeout: Request timeout
            **kwargs: Additional payload parameters

        Returns:
            The sent message
        """
        message = create_request_message(
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            action=action,
            priority=priority,
            timeout=timeout,
            **kwargs
        )

        self.send_message(message)
        return message

    def send_response(
        self,
        original_message: Message,
        payload: Dict[str, Any],
        priority: Optional[int] = None
    ) -> bool:
        """
        Send a response to a message.

        Args:
            original_message: The original request message
            payload: Response payload
            priority: Optional priority override

        Returns:
            True if response was sent
        """
        response = original_message.create_response(
            sender_id=self.agent_id,
            payload=payload,
            priority=priority
        )

        return self.send_message(response)

    def send_error(
        self,
        original_message: Message,
        error_type: str,
        error_message: str,
        **kwargs
    ) -> bool:
        """
        Send an error message.

        Args:
            original_message: Original message that caused error
            error_type: Error classification
            error_message: Error description
            **kwargs: Additional error context

        Returns:
            True if error message was sent
        """
        error_msg = original_message.create_error(
            sender_id=self.agent_id,
            error_type=error_type,
            error_message=error_message,
            **kwargs
        )

        # Also log error in state
        self.state_manager.add_error(
            agent_id=self.agent_id,
            error_type=error_type,
            error_message=error_message,
            **kwargs
        )

        return self.send_message(error_msg)

    def receive_message(
        self,
        timeout: Optional[float] = None,
        block: bool = True
    ) -> Optional[Message]:
        """
        Receive next message from queue.

        Args:
            timeout: Timeout in seconds
            block: Whether to block until message arrives

        Returns:
            Next message or None
        """
        return self.message_queue.receive(
            agent_id=self.agent_id,
            timeout=timeout,
            block=block
        )

    def wait_for_response(
        self,
        request_message: Message,
        timeout: Optional[float] = None
    ) -> Optional[Message]:
        """
        Wait for response to a request.

        Args:
            request_message: The request message
            timeout: Timeout in seconds

        Returns:
            Response message or None if timeout
        """
        timeout = timeout or request_message.timeout

        return self.message_queue.receive_by_correlation(
            agent_id=self.agent_id,
            correlation_id=request_message.correlation_id,
            timeout=timeout
        )

    def invoke_agent(
        self,
        target_agent_id: str,
        action: str,
        timeout: float = 3600.0,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Invoke another agent and wait for response.

        Args:
            target_agent_id: Target agent ID
            action: Action to perform
            timeout: Request timeout
            **kwargs: Additional parameters

        Returns:
            Response payload or None if timeout/error
        """
        # Send request
        request = self.send_request(
            receiver_id=target_agent_id,
            action=action,
            timeout=timeout,
            **kwargs
        )

        # Wait for response
        response = self.wait_for_response(request, timeout=timeout)

        if response is None:
            self.log_warning(f"No response from {target_agent_id} for action {action}")
            return None

        if response.message_type == MessageType.ERROR:
            self.log_error(
                f"Error from {target_agent_id}: {response.payload.get('error_message')}"
            )
            return None

        return response.payload

    def spawn_child_agent(
        self,
        child_type: str,
        child_config: Dict[str, Any]
    ) -> Optional[str]:
        """
        Spawn a child agent (if agent can spawn).

        Args:
            child_type: Type of child agent
            child_config: Child agent configuration

        Returns:
            Child agent ID or None if failed
        """
        if not self.config.get("can_spawn", False):
            self.log_error(f"Agent {self.agent_id} cannot spawn children")
            return None

        # Generate child ID
        child_id = f"{self.agent_id}_child_{child_type}_{int(time.time())}"

        # Create child agent (to be implemented by orchestrator)
        # For now, just return the ID
        return child_id

    def update_state(self, **kwargs) -> None:
        """Update agent's state in state manager."""
        self.state_manager.update_agent_state(
            agent_id=self.agent_id,
            **kwargs
        )

    def get_global_state(self):
        """Get global state."""
        return self.state_manager.get_state()

    def update_global_state(self, updates: Dict[str, Any]) -> None:
        """Update global state."""
        self.state_manager.update_state(updates)

    def log_info(self, message: str) -> None:
        """Log info message."""
        if self.logger:
            self.logger.info(f"[{self.agent_id}] {message}")
        else:
            print(f"[INFO][{self.agent_id}] {message}")

    def log_warning(self, message: str) -> None:
        """Log warning message."""
        if self.logger:
            self.logger.warning(f"[{self.agent_id}] {message}")
        else:
            print(f"[WARNING][{self.agent_id}] {message}")

        self.state_manager.add_warning(message, agent_id=self.agent_id)

    def log_error(self, message: str, **kwargs) -> None:
        """Log error message."""
        if self.logger:
            self.logger.error(f"[{self.agent_id}] {message}")
        else:
            print(f"[ERROR][{self.agent_id}] {message}")

    def run(self) -> None:
        """
        Main run loop for agent.

        Continuously processes messages until stopped.
        """
        self.log_info(f"Agent {self.agent_id} started")
        self.update_state(status="active")

        try:
            while True:
                # Receive message (blocking with 1 second timeout)
                message = self.receive_message(timeout=1.0, block=True)

                if message is None:
                    continue

                # Check for stop signal
                if (message.message_type == MessageType.UPDATE and
                    message.payload.get("action") == "stop"):
                    self.log_info("Received stop signal")
                    break

                # Process message
                try:
                    response = self.process_message(message)

                    if response:
                        self.send_message(response)

                except Exception as e:
                    self.log_error(f"Error processing message: {e}")
                    self.send_error(
                        original_message=message,
                        error_type="processing_error",
                        error_message=str(e)
                    )

        except KeyboardInterrupt:
            self.log_info("Interrupted by user")

        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Shutdown agent gracefully."""
        self.log_info(f"Agent {self.agent_id} shutting down")
        self.update_state(status="completed")
        self.message_queue.unsubscribe_from_broadcasts(self.agent_id)
        self.state_manager.deregister_agent(self.agent_id)

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(id={self.agent_id}, type={self.agent_type})"
