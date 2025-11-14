"""Tests for state management."""

import pytest
from mle_bench_agents.core.state import GlobalState, StateManager, AgentState


def test_global_state_creation():
    """Test global state initialization."""
    state = GlobalState(competition_id="test-comp")
    
    assert state.competition_id == "test-comp"
    assert state.current_phase == "initialized"
    assert state.time_budget_total == 86400


def test_state_manager():
    """Test state manager operations."""
    manager = StateManager()
    
    # Register agent
    manager.register_agent("A1", "orchestrator")
    
    assert "A1" in manager.state.active_agents
    assert manager.state.active_agents["A1"].status == "active"
    
    # Update agent state
    manager.update_agent_state("A1", status="completed")
    
    assert manager.state.active_agents["A1"].status == "completed"


def test_state_time_properties():
    """Test time-related state properties."""
    state = GlobalState(time_budget_total=3600)
    
    assert state.time_remaining <= 3600
    assert state.time_elapsed >= 0
    assert 0 <= state.time_used_percent <= 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
