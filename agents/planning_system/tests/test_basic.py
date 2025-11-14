"""
Basic tests for the Agentic Planning System

Run with: python -m pytest agents/planning_system/tests/
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


def test_imports():
    """Test that all core modules can be imported"""
    from agents.planning_system.core.agent import Agent, AgentRole
    from agents.planning_system.core.plugin import Plugin
    from agents.planning_system.core.llm import LLMConfig
    from agents.planning_system.core.history import ExecutionHistory
    from agents.planning_system.planning.planner import PlanningModule
    from agents.planning_system.execution.engine import WorkflowExecutionEngine

    assert Agent is not None
    assert Plugin is not None
    assert LLMConfig is not None
    assert PlanningModule is not None


def test_llm_config():
    """Test LLM configuration"""
    from agents.planning_system.core.llm import LLMConfig, get_default_config

    config = get_default_config()
    assert config.model_name == "gpt-4o-2024-08-06"
    assert config.provider == "openai"
    assert config.temperature == 0.7

    # Test custom config
    custom = LLMConfig(model_name="gpt-4o-mini", temperature=0.3)
    assert custom.model_name == "gpt-4o-mini"
    assert custom.temperature == 0.3


def test_agent_state():
    """Test agent state management"""
    from agents.planning_system.core.state import AgentState

    state = AgentState()
    assert state.status == "idle"
    assert state.current_step == 0

    state.update_status("planning")
    assert state.status == "planning"

    state.increment_step()
    assert state.current_step == 1


def test_execution_history():
    """Test execution history tracking"""
    from agents.planning_system.core.history import ExecutionHistory

    history = ExecutionHistory()
    assert len(history.events) == 0

    history.add_thought("This is a test thought")
    assert len(history.events) == 1
    assert history.events[0]["type"] == "thought"

    history.add_action("test_action", {"param": "value"}, "result")
    assert len(history.events) == 2
    assert history.events[1]["type"] == "action"


def test_plugin_creation():
    """Test plugin creation"""
    from agents.planning_system.plugins import create_data_exploration_plugin

    plugin = create_data_exploration_plugin()
    assert plugin.functionality.plugin_id == "data_exploration"
    assert "read_csv" in plugin.functionality.actions
    assert "describe_data" in plugin.functionality.actions


def test_problem_specification():
    """Test problem specification"""
    from agents.planning_system.planning.planner import ProblemSpecification

    spec = ProblemSpecification(
        task_description="Test task",
        competition_id="test-competition",
        data_files=["train.csv", "test.csv"],
    )

    assert spec.competition_id == "test-competition"
    assert len(spec.data_files) == 2


def test_planning_module():
    """Test planning module basic functionality"""
    from agents.planning_system.planning.planner import (
        PlanningModule,
        ProblemSpecification,
    )

    spec = ProblemSpecification(
        task_description="Binary classification",
        competition_id="test-competition",
        evaluation_metric="accuracy",
        domain="tabular",
        difficulty="low",
    )

    planner = PlanningModule()
    plan = planner.plan_workflow(spec)

    assert len(plan.agents) > 0
    assert len(plan.subtasks) > 0
    assert plan.workflow_graph is not None


def test_event_bus():
    """Test event bus functionality"""
    from agents.planning_system.execution.event_bus import EventBus, Event

    bus = EventBus()
    assert len(bus.get_all_events()) == 0

    # Publish event
    event = Event(
        event_type="test_event",
        source_agent_id="agent1",
        data={"key": "value"},
    )
    bus.publish(event)

    assert len(bus.get_all_events()) == 1

    # Subscribe and get events
    bus.subscribe("agent2", ["test_event"])
    events = bus.get_events("agent2")
    assert len(events) == 1


def test_shared_state():
    """Test shared state functionality"""
    from agents.planning_system.execution.shared_state import SharedState

    state = SharedState()

    state.set("key1", "value1")
    assert state.get("key1") == "value1"

    state.update({"key2": "value2", "key3": "value3"})
    assert len(state.keys()) == 3

    assert state.has_key("key1")
    assert not state.has_key("nonexistent")


if __name__ == "__main__":
    """Run tests directly"""
    print("Running basic tests...\n")

    tests = [
        ("Imports", test_imports),
        ("LLM Config", test_llm_config),
        ("Agent State", test_agent_state),
        ("Execution History", test_execution_history),
        ("Plugin Creation", test_plugin_creation),
        ("Problem Specification", test_problem_specification),
        ("Planning Module", test_planning_module),
        ("Event Bus", test_event_bus),
        ("Shared State", test_shared_state),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            print(f"✓ {name}")
            passed += 1
        except Exception as e:
            print(f"✗ {name}: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
