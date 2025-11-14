"""
Message passing infrastructure for multi-agent system.

Defines message structure, types, and priorities for inter-agent communication.
"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class MessageType(Enum):
    """Message types for agent communication."""

    REQUEST = "REQUEST"
    RESPONSE = "RESPONSE"
    UPDATE = "UPDATE"
    ERROR = "ERROR"
    ALERT = "ALERT"


class MessagePriority(Enum):
    """Message priority levels (0-10)."""

    INFORMATIONAL = 0
    LOW = 5
    NORMAL = 6
    MEDIUM = 7
    ELEVATED = 8
    HIGH = 9
    CRITICAL = 10


@dataclass
class Message:
    """
    Base message class for inter-agent communication.

    Attributes:
        message_id: Unique identifier for message
        timestamp: Unix timestamp of message creation
        sender_id: Agent ID of sender
        receiver_id: Agent ID of receiver or "broadcast"
        message_type: Type of message
        priority: Priority level (0-10)
        payload: Message content/data
        correlation_id: ID for request-response pairing
        requires_response: Whether response is required
        timeout: Timeout in seconds
    """

    sender_id: str
    receiver_id: str
    message_type: MessageType
    payload: Dict[str, Any]
    priority: int = 5
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    correlation_id: Optional[str] = None
    requires_response: bool = False
    timeout: float = 3600.0

    def __post_init__(self):
        """Validate message fields."""
        # Convert string message_type to enum if needed
        if isinstance(self.message_type, str):
            self.message_type = MessageType(self.message_type)

        # Validate priority
        if not 0 <= self.priority <= 10:
            raise ValueError(f"Priority must be between 0 and 10, got {self.priority}")

        # Set correlation_id to message_id if not provided (for new requests)
        if self.correlation_id is None and self.message_type == MessageType.REQUEST:
            self.correlation_id = self.message_id

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "message_id": self.message_id,
            "timestamp": self.timestamp,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "message_type": self.message_type.value,
            "priority": self.priority,
            "correlation_id": self.correlation_id,
            "requires_response": self.requires_response,
            "timeout": self.timeout,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create message from dictionary."""
        return cls(
            message_id=data["message_id"],
            timestamp=data["timestamp"],
            sender_id=data["sender_id"],
            receiver_id=data["receiver_id"],
            message_type=MessageType(data["message_type"]),
            priority=data["priority"],
            correlation_id=data.get("correlation_id"),
            requires_response=data.get("requires_response", False),
            timeout=data.get("timeout", 3600.0),
            payload=data["payload"],
        )

    def create_response(
        self,
        sender_id: str,
        payload: Dict[str, Any],
        priority: Optional[int] = None
    ) -> "Message":
        """Create a response message to this message."""
        return Message(
            sender_id=sender_id,
            receiver_id=self.sender_id,
            message_type=MessageType.RESPONSE,
            payload=payload,
            priority=priority or self.priority,
            correlation_id=self.correlation_id or self.message_id,
            requires_response=False,
        )

    def create_error(
        self,
        sender_id: str,
        error_type: str,
        error_message: str,
        **kwargs
    ) -> "Message":
        """Create an error message in response to this message."""
        payload = {
            "error_type": error_type,
            "error_message": error_message,
            "original_message_id": self.message_id,
            **kwargs
        }

        return Message(
            sender_id=sender_id,
            receiver_id=self.sender_id,
            message_type=MessageType.ERROR,
            payload=payload,
            priority=MessagePriority.ELEVATED.value,
            correlation_id=self.correlation_id or self.message_id,
            requires_response=True,
        )

    def is_expired(self) -> bool:
        """Check if message has expired based on timeout."""
        return (time.time() - self.timestamp) > self.timeout

    def __repr__(self) -> str:
        """String representation of message."""
        return (
            f"Message(id={self.message_id[:8]}..., "
            f"type={self.message_type.value}, "
            f"from={self.sender_id}, "
            f"to={self.receiver_id}, "
            f"priority={self.priority})"
        )


def create_request_message(
    sender_id: str,
    receiver_id: str,
    action: str,
    priority: int = 5,
    timeout: float = 3600.0,
    **kwargs
) -> Message:
    """
    Helper function to create a REQUEST message.

    Args:
        sender_id: Sender agent ID
        receiver_id: Receiver agent ID
        action: Action to perform
        priority: Message priority (0-10)
        timeout: Timeout in seconds
        **kwargs: Additional payload parameters

    Returns:
        Message object
    """
    payload = {"action": action, **kwargs}

    return Message(
        sender_id=sender_id,
        receiver_id=receiver_id,
        message_type=MessageType.REQUEST,
        payload=payload,
        priority=priority,
        requires_response=True,
        timeout=timeout,
    )


def create_broadcast_message(
    sender_id: str,
    message_type: MessageType,
    payload: Dict[str, Any],
    priority: int = 6
) -> Message:
    """
    Helper function to create a broadcast message.

    Args:
        sender_id: Sender agent ID
        message_type: Type of message
        payload: Message payload
        priority: Message priority

    Returns:
        Message object
    """
    return Message(
        sender_id=sender_id,
        receiver_id="broadcast",
        message_type=message_type,
        payload=payload,
        priority=priority,
        requires_response=False,
    )
