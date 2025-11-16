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
    # Use commercial APIs (default)
    planner = PlanningModule()

    # Or use open-source models (zero cost, local execution):
    # planner = PlanningModule(model_provider="opensource")

    # Step 3: Generate workflow plan
    print("\nGenerating workflow plan...")
    plan = planner.plan_workflow(problem_spec)

    # Step 4: Show the plan
    print("\n" + "="*60)
    print("Generated Workflow Plan")
    print("="*60 + "\n")
    print(plan.visualize())

    # Step 5: Add plugins to agents (optional - requires dependencies)
    print("\nNote: Plugin registration requires ML dependencies (pandas, scikit-learn)")
    print("Skipping plugin registration in this example.")
    print("Plugins can be registered when executing with full dependencies.\n")

    # Initialize empty plugin list for all agents
    for agent in plan.agents:
        plan.plugins[agent.agent_id] = []

    # Uncomment below to register plugins (requires: pip install pandas numpy scikit-learn)
    """
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
    """

    # Step 6: (Optional) Execute the plan
    # Set to True to actually run the workflow (requires scikit-learn, pandas, etc.)
    EXECUTE_WORKFLOW = False

    if EXECUTE_WORKFLOW:
        print("\nExecuting workflow...")
        print("Note: This requires ML dependencies (pandas, scikit-learn, etc.)")

        # Check for required packages
        try:
            import pandas
            import sklearn
            import numpy
        except ImportError as e:
            print(f"\n⚠️  Missing dependency: {e}")
            print("Install with: pip install pandas numpy scikit-learn")
            print("Skipping execution...\n")
        else:
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
    else:
        print("\n" + "="*60)
        print("Note: Execution Skipped")
        print("="*60)
        print("\nTo execute the workflow, set EXECUTE_WORKFLOW = True in this script.")
        print("This requires: pip install pandas numpy scikit-learn\n")

    print("\n" + "="*60)
    print("Example Complete!")
    print("="*60 + "\n")

    print("Next steps:")
    print("  1. Review the generated plan above")
    print("  2. Install ML dependencies: pip install pandas numpy scikit-learn")
    print("  3. Set EXECUTE_WORKFLOW = True to run the workflow")
    print("  4. Or use the CLI for full integration: python agents/planning_system/run.py")


if __name__ == "__main__":
    main()
