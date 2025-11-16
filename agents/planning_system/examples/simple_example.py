"""
Simple Example: Using the Agentic Planning System

This demonstrates how to use the planning system on a simple competition.
"""

import sys
from pathlib import Path

# Add mle-bench to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from agents.planning_system.planning.planner import (
    PlanningModule,
    ProblemSpecification,
)
from agents.planning_system.execution.engine import WorkflowExecutionEngine
from agents.planning_system.plugins import (
    create_data_exploration_plugin,
    create_ml_training_plugin,
    create_submission_validation_plugin,
)


def main():
    """Run a simple example"""

    print("\n" + "="*60)
    print("Agentic Planning System - Simple Example")
    print("="*60 + "\n")

    # Step 1: Define the problem
    problem_spec = ProblemSpecification(
        task_description="Predict whether a passenger was transported to an alternate dimension",
        competition_id="spaceship-titanic",
        data_files=["train.csv", "test.csv"],
        evaluation_metric="accuracy",
        domain="tabular",
        difficulty="low",
    )

    # Step 2: Create a planning module
    print("Creating planning module...")
    planner = PlanningModule()

    # Step 3: Generate workflow plan
    print("\nGenerating workflow plan...")
    plan = planner.plan_workflow(problem_spec)

    # Step 4: Show the plan
    print("\n" + "="*60)
    print("Generated Workflow Plan")
    print("="*60 + "\n")
    print(plan.visualize())

    # Step 5: Add plugins to agents
    print("\nRegistering plugins with agents...")

    # Create plugin instances
    data_plugin = create_data_exploration_plugin()
    ml_plugin = create_ml_training_plugin()
    validation_plugin = create_submission_validation_plugin()

    # Assign plugins to agents based on their roles
    for agent in plan.agents:
        if "exploration" in agent.agent_id.lower():
            plan.plugins[agent.agent_id] = [data_plugin]
            print(f"  - {agent.agent_id}: data_exploration")

        elif "training" in agent.agent_id.lower():
            plan.plugins[agent.agent_id] = [ml_plugin, data_plugin]
            print(f"  - {agent.agent_id}: ml_training, data_exploration")

        elif "validation" in agent.agent_id.lower():
            plan.plugins[agent.agent_id] = [validation_plugin]
            print(f"  - {agent.agent_id}: submission_validation")

        else:
            plan.plugins[agent.agent_id] = []

    # Step 6: (Optional) Execute the plan
    # Uncomment to actually run the workflow
    """
    print("\nExecuting workflow...")
    workspace_dir = Path("./workspace/spaceship-titanic")
    data_dir = Path("/home/data")  # Adjust as needed

    engine = WorkflowExecutionEngine(plan, workspace_dir, data_dir)
    result = engine.execute()

    print("\n" + "="*60)
    print("Execution Result")
    print("="*60)
    print(f"Success: {result.success}")
    print(f"Execution Time: {result.execution_time:.1f}s")
    print(f"Errors: {len(result.errors)}")
    """

    print("\n" + "="*60)
    print("Example Complete!")
    print("="*60 + "\n")

    print("Next steps:")
    print("  1. Review the generated plan")
    print("  2. Uncomment the execution code to run the workflow")
    print("  3. Adjust plugins and agent roles as needed")
    print("  4. Use the CLI for full integration with MLE-Bench")


if __name__ == "__main__":
    main()
