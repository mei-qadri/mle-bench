"""Tests for message passing system."""

import pytest
from mle_bench_agents.core.message import Message, MessageType, create_request_message


def test_message_creation():
    """Test basic message creation."""
    msg = Message(
        sender_id="test_sender",
        receiver_id="test_receiver",
        message_type=MessageType.REQUEST,
        payload={"action": "test"}
    )
    
    assert msg.sender_id == "test_sender"
    assert msg.receiver_id == "test_receiver"
    assert msg.message_type == MessageType.REQUEST
    assert msg.payload["action"] == "test"


def test_request_message_helper():
    """Test request message helper function."""
    msg = create_request_message(
        sender_id="sender",
        receiver_id="receiver",
        action="test_action",
        param1="value1"
    )
    
    assert msg.message_type == MessageType.REQUEST
    assert msg.payload["action"] == "test_action"
    assert msg.payload["param1"] == "value1"
    assert msg.requires_response == True


def test_message_response():
    """Test creating response message."""
    request = create_request_message(
        sender_id="A1",
        receiver_id="A2",
        action="test"
    )
    
    response = request.create_response(
        sender_id="A2",
        payload={"status": "success"}
    )
    
    assert response.message_type == MessageType.RESPONSE
    assert response.correlation_id == request.correlation_id
    assert response.sender_id == "A2"
    assert response.receiver_id == "A1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
