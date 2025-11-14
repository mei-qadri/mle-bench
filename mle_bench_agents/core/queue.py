"""
Message queue implementation for inter-agent communication.

Provides priority-based message routing with broadcast support.
"""

import time
from collections import defaultdict, deque
from threading import Lock, Event
from typing import Dict, List, Optional, Set

from mle_bench_agents.core.message import Message, MessageType


class MessageQueue:
    """
    Thread-safe message queue with priority-based routing.

    Supports:
    - Agent-specific queues
    - Broadcast messaging
    - Priority-based message ordering
    - Message expiration handling
    """

    def __init__(self, max_queue_size: int = 1000):
        """
        Initialize message queue.

        Args:
            max_queue_size: Maximum messages per agent queue
        """
        self.max_queue_size = max_queue_size

        # Agent-specific message queues (sorted by priority)
        self._queues: Dict[str, List[Message]] = defaultdict(list)

        # Broadcast subscribers
        self._broadcast_subscribers: Set[str] = set()

        # Message history for correlation tracking
        self._message_history: Dict[str, Message] = {}

        # Locks for thread safety
        self._lock = Lock()
        self._queue_locks: Dict[str, Lock] = defaultdict(Lock)

        # Events for blocking receives
        self._events: Dict[str, Event] = defaultdict(Event)

        # Statistics
        self.stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "messages_expired": 0,
            "messages_dropped": 0,
        }

    def send(self, message: Message) -> bool:
        """
        Send a message to recipient(s).

        Args:
            message: Message to send

        Returns:
            True if message was queued successfully
        """
        with self._lock:
            # Store in message history for correlation
            self._message_history[message.message_id] = message

            # Check if expired
            if message.is_expired():
                self.stats["messages_expired"] += 1
                return False

            # Handle broadcast
            if message.receiver_id == "broadcast":
                success = self._broadcast(message)
            else:
                success = self._send_to_agent(message, message.receiver_id)

            if success:
                self.stats["messages_sent"] += 1

            return success

    def _send_to_agent(self, message: Message, agent_id: str) -> bool:
        """
        Send message to specific agent.

        Args:
            message: Message to send
            agent_id: Recipient agent ID

        Returns:
            True if successful
        """
        with self._queue_locks[agent_id]:
            queue = self._queues[agent_id]

            # Check queue size limit
            if len(queue) >= self.max_queue_size:
                self.stats["messages_dropped"] += 1
                return False

            # Add message and sort by priority (descending)
            queue.append(message)
            queue.sort(key=lambda m: m.priority, reverse=True)

            # Signal waiting receivers
            self._events[agent_id].set()

            return True

    def _broadcast(self, message: Message) -> bool:
        """
        Broadcast message to all subscribers.

        Args:
            message: Message to broadcast

        Returns:
            True if at least one delivery succeeded
        """
        if not self._broadcast_subscribers:
            return False

        success_count = 0
        for agent_id in self._broadcast_subscribers:
            if self._send_to_agent(message, agent_id):
                success_count += 1

        return success_count > 0

    def receive(
        self,
        agent_id: str,
        timeout: Optional[float] = None,
        block: bool = True
    ) -> Optional[Message]:
        """
        Receive next message for agent.

        Args:
            agent_id: Agent ID receiving the message
            timeout: Timeout in seconds (None = no timeout)
            block: Whether to block until message arrives

        Returns:
            Next message or None if timeout/no messages
        """
        start_time = time.time()

        while True:
            with self._queue_locks[agent_id]:
                queue = self._queues[agent_id]

                # Clean expired messages
                queue[:] = [m for m in queue if not m.is_expired()]

                if queue:
                    # Get highest priority message
                    message = queue.pop(0)
                    self.stats["messages_received"] += 1
                    self._events[agent_id].clear()
                    return message

            # Non-blocking mode
            if not block:
                return None

            # Check timeout
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    return None
                remaining = timeout - elapsed
            else:
                remaining = None

            # Wait for new message
            self._events[agent_id].wait(timeout=remaining)

    def receive_by_correlation(
        self,
        agent_id: str,
        correlation_id: str,
        timeout: Optional[float] = None
    ) -> Optional[Message]:
        """
        Receive message with specific correlation ID (for request-response).

        Args:
            agent_id: Agent ID
            correlation_id: Correlation ID to match
            timeout: Timeout in seconds

        Returns:
            Matching message or None
        """
        start_time = time.time()

        while True:
            with self._queue_locks[agent_id]:
                queue = self._queues[agent_id]

                # Look for message with matching correlation_id
                for i, message in enumerate(queue):
                    if message.correlation_id == correlation_id:
                        queue.pop(i)
                        self.stats["messages_received"] += 1
                        return message

            # Check timeout
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    return None
                remaining = timeout - elapsed
            else:
                remaining = 1.0  # Default 1 second wait

            # Wait for new messages
            self._events[agent_id].wait(timeout=min(remaining, 1.0))

    def peek(self, agent_id: str) -> Optional[Message]:
        """
        Peek at next message without removing it.

        Args:
            agent_id: Agent ID

        Returns:
            Next message or None
        """
        with self._queue_locks[agent_id]:
            queue = self._queues[agent_id]

            # Clean expired messages
            queue[:] = [m for m in queue if not m.is_expired()]

            if queue:
                return queue[0]

            return None

    def get_queue_size(self, agent_id: str) -> int:
        """Get number of messages in agent's queue."""
        with self._queue_locks[agent_id]:
            return len(self._queues[agent_id])

    def subscribe_to_broadcasts(self, agent_id: str) -> None:
        """Subscribe agent to broadcast messages."""
        with self._lock:
            self._broadcast_subscribers.add(agent_id)

    def unsubscribe_from_broadcasts(self, agent_id: str) -> None:
        """Unsubscribe agent from broadcast messages."""
        with self._lock:
            self._broadcast_subscribers.discard(agent_id)

    def clear_queue(self, agent_id: str) -> int:
        """
        Clear all messages for an agent.

        Args:
            agent_id: Agent ID

        Returns:
            Number of messages cleared
        """
        with self._queue_locks[agent_id]:
            count = len(self._queues[agent_id])
            self._queues[agent_id].clear()
            return count

    def get_message_by_id(self, message_id: str) -> Optional[Message]:
        """
        Get message from history by ID.

        Args:
            message_id: Message ID

        Returns:
            Message or None if not found
        """
        with self._lock:
            return self._message_history.get(message_id)

    def get_statistics(self) -> Dict[str, any]:
        """Get queue statistics."""
        with self._lock:
            stats = dict(self.stats)
            stats["total_queues"] = len(self._queues)
            stats["broadcast_subscribers"] = len(self._broadcast_subscribers)
            stats["messages_in_history"] = len(self._message_history)

            # Per-agent queue sizes
            queue_sizes = {}
            for agent_id, queue in self._queues.items():
                queue_sizes[agent_id] = len(queue)
            stats["queue_sizes"] = queue_sizes

            return stats

    def cleanup_expired_messages(self) -> int:
        """
        Clean up expired messages from all queues.

        Returns:
            Number of messages removed
        """
        removed_count = 0

        with self._lock:
            for agent_id in list(self._queues.keys()):
                with self._queue_locks[agent_id]:
                    queue = self._queues[agent_id]
                    original_size = len(queue)
                    queue[:] = [m for m in queue if not m.is_expired()]
                    removed_count += original_size - len(queue)

        self.stats["messages_expired"] += removed_count
        return removed_count

    def reset(self) -> None:
        """Reset queue to initial state."""
        with self._lock:
            self._queues.clear()
            self._broadcast_subscribers.clear()
            self._message_history.clear()
            self._queue_locks.clear()
            self._events.clear()

            self.stats = {
                "messages_sent": 0,
                "messages_received": 0,
                "messages_expired": 0,
                "messages_dropped": 0,
            }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"MessageQueue("
            f"queues={len(self._queues)}, "
            f"subscribers={len(self._broadcast_subscribers)}, "
            f"sent={self.stats['messages_sent']}, "
            f"received={self.stats['messages_received']})"
        )
