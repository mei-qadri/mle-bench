"""
Workflow Execution Engine

Executes approved workflow plans by coordinating agent execution.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
import networkx as nx
import time

from agents.planning_system.planning.planner import WorkflowPlan
from agents.planning_system.core.agent import Agent
from agents.planning_system.execution.event_bus import EventBus, Event
from agents.planning_system.execution.shared_state import SharedState, AgentRegistry


@dataclass
class ExecutionResult:
    """Result of workflow execution"""

    success: bool
    outputs: Dict[str, Any]  # Final outputs from agents
    execution_time: float  # Total execution time in seconds
    agent_results: Dict[str, Dict]  # Results from each agent
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class WorkflowExecutionEngine:
    """
    Executes approved workflow plans

    Coordinates agent execution according to the workflow graph.
    """

    def __init__(
        self,
        workflow_plan: WorkflowPlan,
        workspace_dir: Path,
        data_dir: Path,
    ):
        self.plan = workflow_plan
        self.workspace_dir = Path(workspace_dir)
        self.data_dir = Path(data_dir)

        # Create workspace if needed
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        # Coordination infrastructure
        self.agent_instances: Dict[str, Agent] = {}
        self.event_bus = EventBus()
        self.shared_state = SharedState()
        self.agent_registry = AgentRegistry()

        # Execution state
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

    def execute(self) -> ExecutionResult:
        """
        Execute the workflow plan

        Returns:
            ExecutionResult with outputs and metadata
        """
        print(f"\n{'='*60}")
        print("Starting Workflow Execution")
        print(f"{'='*60}\n")

        self.start_time = datetime.now()
        agent_results = {}
        errors = []

        try:
            # 1. Setup phase
            print("Phase 1: Setting up agents...")
            self._instantiate_agents()

            # 2. Execution phase
            print("\nPhase 2: Executing agents...")
            execution_order = self._get_topological_order()
            print(f"Execution order: {[aid.replace('agent_', '') for aid in execution_order]}\n")

            for i, agent_id in enumerate(execution_order, 1):
                print(f"\n[{i}/{len(execution_order)}] Executing: {agent_id}")
                print(f"{'-'*60}")

                agent = self.agent_instances[agent_id]

                # Execute agent
                result = self._execute_agent(agent)
                agent_results[agent_id] = result

                # Update shared state
                self.shared_state.set(f"{agent_id}_result", result)

                # Check for errors
                if result["status"] == "failed":
                    errors.append(f"Agent {agent_id} failed: {result.get('error', 'Unknown error')}")

                    # Decide whether to continue or stop
                    if self._should_terminate(result):
                        print(f"\nTerminating workflow due to critical failure in {agent_id}")
                        break

                print(f"✓ {agent_id} completed with status: {result['status']}")

            # 3. Finalization phase
            print(f"\n{'='*60}")
            print("Phase 3: Finalizing execution...")
            final_result = self._finalize_execution(agent_results, errors)

        except Exception as e:
            print(f"\nError during execution: {e}")
            import traceback
            traceback.print_exc()

            final_result = ExecutionResult(
                success=False,
                outputs={},
                execution_time=0.0,
                agent_results=agent_results,
                errors=[str(e)],
            )

        self.end_time = datetime.now()
        execution_time = (self.end_time - self.start_time).total_seconds()
        final_result.execution_time = execution_time

        print(f"\n{'='*60}")
        print(f"Workflow Execution Complete")
        print(f"Success: {final_result.success}")
        print(f"Execution Time: {execution_time:.1f}s")
        print(f"{'='*60}\n")

        return final_result

    def _instantiate_agents(self):
        """Create agent instances from specifications"""
        print("Instantiating agents...")

        for agent_spec in self.plan.agents:
            # Create agent workspace
            agent_workspace = self.workspace_dir / agent_spec.agent_id
            agent_workspace.mkdir(parents=True, exist_ok=True)

            # Assign plugins
            agent_spec.plugins = {
                p.functionality.plugin_id: p
                for p in self.plan.plugins.get(agent_spec.agent_id, [])
            }

            # Initialize agent
            agent_spec.initialize(agent_workspace, self.data_dir)

            # Register in infrastructure
            self.agent_instances[agent_spec.agent_id] = agent_spec
            self.agent_registry.register(
                agent_spec.agent_id,
                {"role": agent_spec.role.role_name},
            )

            print(f"  ✓ {agent_spec.agent_id} ({agent_spec.role.role_name})")

    def _execute_agent(self, agent: Agent) -> Dict[str, Any]:
        """
        Execute a single agent until completion

        Returns:
            Dict with status, outputs, and metadata
        """
        # Update registry
        self.agent_registry.update_status(agent.agent_id, "running")

        # Publish start event
        self.event_bus.publish(
            Event(
                event_type="agent_started",
                source_agent_id=agent.agent_id,
                data={"role": agent.role.role_name},
            )
        )

        # Set initial goal
        agent.state.set_goal(agent.role.success_condition)

        # Execution loop
        max_steps = agent.role.max_steps
        start_time = time.time()
        max_time = agent.role.max_time_seconds

        while agent.state.current_step < max_steps:
            # Check time limit
            elapsed = time.time() - start_time
            if elapsed > max_time:
                print(f"  ⚠ Time limit exceeded ({elapsed:.1f}s > {max_time}s)")
                agent.state.update_status("failed")
                break

            # Get observation
            observation = self._get_observation(agent)

            # Agent takes action
            try:
                result = agent.act(observation)

                # Check if completed
                if agent.state.status == "completed":
                    break

                # Print progress
                if agent.state.current_step % 10 == 0:
                    print(f"  Step {agent.state.current_step}/{max_steps}", end="\r")

            except Exception as e:
                print(f"  Error in step {agent.state.current_step}: {e}")
                agent.state.update_status("failed")
                agent.history.add_error(e, f"Step {agent.state.current_step}")
                break

        # Finalize agent
        agent.finalize()

        # Update registry
        self.agent_registry.update_status(agent.agent_id, agent.state.status)

        # Publish completion event
        self.event_bus.publish(
            Event(
                event_type="agent_completed",
                source_agent_id=agent.agent_id,
                data={
                    "status": agent.state.status,
                    "steps": agent.state.current_step,
                },
            )
        )

        return {
            "agent_id": agent.agent_id,
            "status": agent.state.status,
            "outputs": agent.state.outputs,
            "artifacts": agent.state.artifacts,
            "steps_taken": agent.state.current_step,
            "errors": agent.state.errors_encountered,
        }

    def _get_observation(self, agent: Agent) -> Dict[str, Any]:
        """
        Get current observation for agent

        Includes:
        - Shared state
        - Messages from other agents
        - Files in workspace
        """
        observation = {
            "step": agent.state.current_step,
            "goal": agent.state.current_goal,
            "shared_state": self.shared_state.get_all(),
            "workspace_files": [
                str(f.relative_to(agent.workspace_dir))
                for f in agent.workspace_dir.glob("**/*")
                if f.is_file()
            ],
        }

        # Get messages from event bus
        events = self.event_bus.get_events(agent.agent_id)
        observation["messages"] = [
            {"type": e.event_type, "source": e.source_agent_id, "data": e.data}
            for e in events
        ]

        return observation

    def _should_terminate(self, result: Dict) -> bool:
        """Decide if workflow should terminate early"""
        # For now, only terminate on critical failures
        # Could be enhanced with more sophisticated logic
        return result.get("status") == "failed" and result.get("critical", False)

    def _finalize_execution(
        self, agent_results: Dict[str, Dict], errors: List[str]
    ) -> ExecutionResult:
        """Finalize execution and collect results"""

        # Collect outputs from all agents
        all_outputs = {}
        for agent_id, result in agent_results.items():
            all_outputs[agent_id] = result.get("outputs", {})

        # Check for submission file
        submission_file = self.workspace_dir / "submission" / "submission.csv"
        if not submission_file.exists():
            # Try to find it in agent workspaces
            for agent_workspace in self.workspace_dir.glob("agent_*/"):
                potential_submission = agent_workspace / "submission.csv"
                if potential_submission.exists():
                    submission_file = potential_submission
                    break

        # Determine success
        success = len(errors) == 0 and all(
            r["status"] == "completed" for r in agent_results.values()
        )

        return ExecutionResult(
            success=success,
            outputs=all_outputs,
            execution_time=0.0,  # Will be set by caller
            agent_results=agent_results,
            errors=errors,
            metadata={
                "num_agents": len(agent_results),
                "submission_file": str(submission_file) if submission_file.exists() else None,
                "event_count": len(self.event_bus.get_all_events()),
            },
        )

    def _get_topological_order(self) -> List[str]:
        """Get execution order based on workflow DAG"""
        try:
            return list(nx.topological_sort(self.plan.workflow_graph))
        except nx.NetworkXError:
            # Fallback to simple order if graph has issues
            return [agent.agent_id for agent in self.plan.agents]

    def save_execution_report(self, result: ExecutionResult, output_file: Path):
        """Save execution report to file"""
        import json

        report = {
            "success": result.success,
            "execution_time": result.execution_time,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "agent_results": result.agent_results,
            "errors": result.errors,
            "metadata": result.metadata,
            "outputs": result.outputs,
        }

        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\nExecution report saved to: {output_file}")
