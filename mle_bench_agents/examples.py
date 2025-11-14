"""
Example usage of MLE-bench Multi-Agent System.

Demonstrates core functionality for Phase I implementation.
"""

from mle_bench_agents.core.message import Message, MessageType, create_request_message
from mle_bench_agents.core.queue import MessageQueue
from mle_bench_agents.core.state import StateManager, GlobalState
from mle_bench_agents.agents.orchestrator import OrchestratorAgent
from mle_bench_agents.agents.resource_manager import ResourceManagerAgent
from mle_bench_agents.plugs.data_loading import DataLoadingPlug
from mle_bench_agents.plugs.metrics import MetricsPlug


def example_1_message_passing():
    """Example 1: Basic message passing."""
    print("=" * 60)
    print("Example 1: Message Passing")
    print("=" * 60)

    # Create message queue
    queue = MessageQueue()

    # Create request message
    request = create_request_message(
        sender_id="A1_orchestrator",
        receiver_id="A2_analysis",
        action="analyze_competition",
        priority=8,
        competition_id="spaceship-titanic",
        data_path="/home/data/"
    )

    print(f"Created request: {request}")
    print(f"  Message ID: {request.message_id}")
    print(f"  Type: {request.message_type.value}")
    print(f"  Priority: {request.priority}")
    print(f"  Payload: {request.payload}")

    # Send message
    success = queue.send(request)
    print(f"\nMessage sent: {success}")

    # Receive message
    received = queue.receive("A2_analysis", timeout=1.0, block=False)
    print(f"Message received: {received}")

    # Create response
    response = request.create_response(
        sender_id="A2_analysis",
        payload={
            "status": "success",
            "competition_type": "tabular",
            "complexity": "low"
        }
    )

    print(f"\nCreated response: {response}")
    print(f"  Correlation ID: {response.correlation_id}")

    # Send response
    queue.send(response)

    # Receive response
    received_response = queue.receive_by_correlation(
        agent_id="A1_orchestrator",
        correlation_id=request.correlation_id,
        timeout=1.0
    )

    print(f"Response received: {received_response.payload}")
    print()


def example_2_state_management():
    """Example 2: State management."""
    print("=" * 60)
    print("Example 2: State Management")
    print("=" * 60)

    # Create state manager
    state_mgr = StateManager()

    # Update global state
    state_mgr.update_state({
        "competition_id": "spaceship-titanic",
        "competition_type": "tabular",
        "complexity": "low",
        "metric_name": "accuracy",
        "time_budget_total": 3600  # 1 hour
    })

    # Access state
    state = state_mgr.get_state()
    print(f"Competition ID: {state.competition_id}")
    print(f"Competition Type: {state.competition_type}")
    print(f"Complexity: {state.complexity}")
    print(f"Time Budget: {state.time_budget_total}s")
    print(f"Time Elapsed: {state.time_elapsed}s")
    print(f"Time Remaining: {state.time_remaining}s")

    # Register agent
    state_mgr.register_agent("A1_orchestrator", "orchestrator")
    print(f"\nActive Agents: {list(state.active_agents.keys())}")

    # Update agent state
    state_mgr.update_agent_state(
        "A1_orchestrator",
        status="in_progress",
        current_task="Phase 1: Understanding"
    )

    agent_state = state.active_agents["A1_orchestrator"]
    print(f"Agent Status: {agent_state.status}")
    print(f"Current Task: {agent_state.current_task}")

    # Add warning
    state_mgr.add_warning("This is a test warning", agent_id="A1_orchestrator")
    print(f"\nWarnings: {len(state.warnings)}")
    print()


def example_3_resource_monitoring():
    """Example 3: Resource monitoring."""
    print("=" * 60)
    print("Example 3: Resource Monitoring")
    print("=" * 60)

    # Create infrastructure
    queue = MessageQueue()
    state_mgr = StateManager()

    # Create resource manager
    resource_mgr = ResourceManagerAgent(
        agent_id="A7_resource_manager",
        message_queue=queue,
        state_manager=state_mgr,
        config={"check_interval": 5}  # Check every 5 seconds
    )

    # Collect metrics once
    metrics = resource_mgr.collect_metrics()
    print("Current Resource Metrics:")
    print(f"  CPU: {metrics['cpu_percent']:.1f}%")
    print(f"  Memory: {metrics['memory_percent']:.1f}% ({metrics['memory_used_gb']:.1f} GB)")
    print(f"  GPU: {metrics['gpu_percent']:.1f}%")
    print(f"  Disk Free: {metrics['disk_free_gb']:.1f} GB")
    print(f"  Time Used: {metrics['time_used_percent']:.1f}%")

    # Check thresholds
    alerts = resource_mgr.check_thresholds(metrics)
    if alerts:
        print(f"\nAlerts Generated: {len(alerts)}")
        for alert in alerts:
            print(f"  [{alert['type']}] {alert['message']}")
    else:
        print("\nNo alerts - all resources within normal ranges")

    print()


def example_4_using_plugs():
    """Example 4: Using plug components."""
    print("=" * 60)
    print("Example 4: Using Plugs")
    print("=" * 60)

    # Example with metrics plug
    metrics_plug = MetricsPlug()

    # Simulate some predictions
    import numpy as np
    y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 0, 1, 1, 0, 1])

    # Compute accuracy
    accuracy = metrics_plug.execute(y_true, y_pred, metric="accuracy")
    print(f"Accuracy: {accuracy:.3f}")

    # Compute MAE
    y_true_reg = np.array([1.5, 2.3, 3.1, 4.0, 5.2])
    y_pred_reg = np.array([1.4, 2.5, 3.0, 3.9, 5.5])
    mae = metrics_plug.execute(y_true_reg, y_pred_reg, metric="mae")
    print(f"MAE: {mae:.3f}")

    print()


def example_5_complete_workflow():
    """Example 5: Complete workflow simulation."""
    print("=" * 60)
    print("Example 5: Complete Workflow (Simulation)")
    print("=" * 60)

    # Create infrastructure
    queue = MessageQueue()
    state_mgr = StateManager()

    # Create orchestrator
    orchestrator = OrchestratorAgent(
        agent_id="A1_orchestrator",
        message_queue=queue,
        state_manager=state_mgr,
        config={
            "time_allocations": {
                "understanding": 0.10,
                "preparation": 0.15,
                "modeling": 0.60,
                "ensembling": 0.10,
                "submission": 0.05
            }
        }
    )

    print("Orchestrator created")
    print(f"  Agent ID: {orchestrator.agent_id}")
    print(f"  Type: {orchestrator.agent_type}")
    print(f"  Can Spawn: {orchestrator.config.get('can_spawn', False)}")

    # Simulate competition context
    context = {
        "competition_id": "spaceship-titanic",
        "time_budget": 3600,  # 1 hour
        "data_path": "/home/data/",
        "description": "Binary classification - predict passenger transport"
    }

    print(f"\nSimulating competition: {context['competition_id']}")
    print(f"Time budget: {context['time_budget']}s")

    # Note: Full execution requires all agents to be implemented
    # For Phase I, we're just demonstrating the structure

    print("\nWorkflow phases:")
    print("  1. Understanding (10% = 360s)")
    print("  2. Preparation (15% = 540s)")
    print("  3. Modeling (60% = 2160s)")
    print("  4. Ensembling (10% = 360s)")
    print("  5. Submission (5% = 180s)")

    print("\n[Phase I] Core infrastructure ready for agent implementation")
    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("MLE-bench Multi-Agent System - Phase I Examples")
    print("=" * 60 + "\n")

    example_1_message_passing()
    example_2_state_management()
    example_3_resource_monitoring()
    example_4_using_plugs()
    example_5_complete_workflow()

    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
