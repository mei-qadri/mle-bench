"""
Command-Line Interface for Agentic Planning System

Usage:
    python -m agents.planning_system.cli plan <competition_id>
    python -m agents.planning_system.cli execute <plan_file>
    python -m agents.planning_system.cli run <competition_id>

Or use the helper script:
    ./agents/planning_system/run_planner.sh plan <competition_id>
"""

import argparse
from pathlib import Path
import sys

# Ensure mle-bench is in path
if __name__ == "__main__":
    mle_bench_dir = Path(__file__).parent.parent.parent.absolute()
    if str(mle_bench_dir) not in sys.path:
        sys.path.insert(0, str(mle_bench_dir))

from agents.planning_system.planning.planner import (
    PlanningModule,
    ProblemSpecification,
)
from agents.planning_system.execution.engine import WorkflowExecutionEngine
from agents.planning_system.collaboration.interface import HumanCollaborationInterface


def plan_command(args):
    """Create a workflow plan for a competition"""
    print(f"\nCreating workflow plan for: {args.competition_id}\n")

    # Create problem specification
    if args.mlebench_dir:
        mlebench_dir = Path(args.mlebench_dir)
        problem_spec = ProblemSpecification.from_competition(
            args.competition_id, mlebench_dir
        )
    else:
        problem_spec = ProblemSpecification(
            task_description=f"Solve {args.competition_id}",
            competition_id=args.competition_id,
        )

    # Create planning module with model provider
    planner = PlanningModule(model_provider=args.model_provider)

    # Generate plan
    plan = planner.plan_workflow(problem_spec)

    # Save plan
    output_dir = Path(args.output) / args.competition_id
    plan.to_config_files(output_dir)

    print(f"\nPlan saved to: {output_dir}")
    print("\nPlan Summary:")
    print(plan.visualize())

    return plan


def execute_command(args):
    """Execute a workflow plan"""
    import json

    print(f"\nExecuting workflow plan from: {args.plan_file}\n")

    # Load plan
    plan_file = Path(args.plan_file)
    if not plan_file.exists():
        print(f"Error: Plan file not found: {plan_file}")
        sys.exit(1)

    # For now, we need to reconstruct the plan from JSON
    # TODO: Implement proper plan serialization/deserialization
    print("Error: Plan execution from file not yet implemented")
    print("Please use 'run' command instead to plan and execute in one step")
    sys.exit(1)


def run_command(args):
    """Plan and execute a competition in one step"""
    print(f"\n{'='*60}")
    print(f"Running agentic planning system for: {args.competition_id}")
    print(f"{'='*60}\n")

    # Step 1: Create problem specification
    if args.mlebench_dir:
        mlebench_dir = Path(args.mlebench_dir)
        problem_spec = ProblemSpecification.from_competition(
            args.competition_id, mlebench_dir
        )
    else:
        problem_spec = ProblemSpecification(
            task_description=f"Solve {args.competition_id}",
            competition_id=args.competition_id,
        )

    # Step 2: Generate plan
    print("\n" + "="*60)
    print("STEP 1: PLANNING")
    print("="*60)

    planner = PlanningModule(model_provider=args.model_provider)
    plan = planner.plan_workflow(problem_spec)

    # Step 3: Human review (if not skipped)
    if not args.skip_approval:
        print("\n" + "="*60)
        print("STEP 2: HUMAN REVIEW")
        print("="*60 + "\n")

        interface = HumanCollaborationInterface()
        interface.present_plan(plan)

        # Get approval
        feedback = interface.gather_feedback()

        if feedback["status"] == "rejected":
            print("\nPlan rejected. Exiting.")
            sys.exit(0)

        elif feedback["status"] == "needs_modification":
            print("\nPlan modifications requested. Please re-run with updated requirements.")
            # TODO: Implement plan modification
            sys.exit(0)

    # Step 4: Execute plan
    print("\n" + "="*60)
    print("STEP 3: EXECUTION")
    print("="*60 + "\n")

    workspace_dir = Path(args.workspace) / args.competition_id
    data_dir = Path(args.data_dir) if args.data_dir else Path("/home/data")

    engine = WorkflowExecutionEngine(plan, workspace_dir, data_dir)
    result = engine.execute()

    # Step 5: Save results
    output_dir = Path(args.output) / args.competition_id
    output_dir.mkdir(parents=True, exist_ok=True)

    plan.to_config_files(output_dir)
    engine.save_execution_report(result, output_dir / "execution_report.json")

    print(f"\nResults saved to: {output_dir}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Agentic Planning System for MLE-Bench"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Plan command
    plan_parser = subparsers.add_parser("plan", help="Create a workflow plan")
    plan_parser.add_argument("competition_id", help="Competition ID")
    plan_parser.add_argument(
        "--mlebench-dir",
        default="/home/user/mle-bench",
        help="Path to MLE-Bench directory",
    )
    plan_parser.add_argument(
        "--output",
        default="./plans",
        help="Output directory for plan files",
    )
    plan_parser.add_argument(
        "--model-provider",
        default="openai",
        choices=["openai", "anthropic", "opensource"],
        help="Model provider to use (openai, anthropic, or opensource)",
    )

    # Execute command
    exec_parser = subparsers.add_parser("execute", help="Execute a workflow plan")
    exec_parser.add_argument("plan_file", help="Path to plan file")
    exec_parser.add_argument(
        "--workspace",
        default="./workspace",
        help="Workspace directory",
    )
    exec_parser.add_argument(
        "--data-dir",
        help="Data directory (defaults to /home/data)",
    )

    # Run command (plan + execute)
    run_parser = subparsers.add_parser(
        "run", help="Plan and execute in one step"
    )
    run_parser.add_argument("competition_id", help="Competition ID")
    run_parser.add_argument(
        "--mlebench-dir",
        default="/home/user/mle-bench",
        help="Path to MLE-Bench directory",
    )
    run_parser.add_argument(
        "--workspace",
        default="./workspace",
        help="Workspace directory",
    )
    run_parser.add_argument(
        "--data-dir",
        help="Data directory (defaults to /home/data)",
    )
    run_parser.add_argument(
        "--output",
        default="./output",
        help="Output directory for results",
    )
    run_parser.add_argument(
        "--skip-approval",
        action="store_true",
        help="Skip human approval step",
    )
    run_parser.add_argument(
        "--model-provider",
        default="openai",
        choices=["openai", "anthropic", "opensource"],
        help="Model provider to use (openai, anthropic, or opensource)",
    )

    args = parser.parse_args()

    if args.command == "plan":
        plan_command(args)
    elif args.command == "execute":
        execute_command(args)
    elif args.command == "run":
        run_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
