"""Tests for message queue."""

import pytest
from mle_bench_agents.core.queue import MessageQueue
from mle_bench_agents.core.message import Message, MessageType


def test_queue_creation():
    """Test queue initialization."""
    queue = MessageQueue()
    
    assert queue.max_queue_size == 1000
    assert queue.stats["messages_sent"] == 0


def test_send_receive():
    """Test basic send and receive."""
    queue = MessageQueue()
    
    msg = Message(
        sender_id="A1",
        receiver_id="A2",
        message_type=MessageType.REQUEST,
        payload={"action": "test"}
    )
    
    # Send message
    success = queue.send(msg)
    assert success == True
    
    # Receive message
    received = queue.receive("A2", timeout=1.0, block=False)
    assert received is not None
    assert received.message_id == msg.message_id


def test_priority_ordering():
    """Test priority-based message ordering."""
    queue = MessageQueue()
    
    # Send low priority message
    msg1 = Message(
        sender_id="A1",
        receiver_id="A2",
        message_type=MessageType.REQUEST,
        payload={"num": 1},
        priority=5
    )
    queue.send(msg1)
    
    # Send high priority message
    msg2 = Message(
        sender_id="A1",
        receiver_id="A2",
        message_type=MessageType.ALERT,
        payload={"num": 2},
        priority=10
    )
    queue.send(msg2)
    
    # High priority should be received first
    received = queue.receive("A2", timeout=1.0, block=False)
    assert received.payload["num"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
