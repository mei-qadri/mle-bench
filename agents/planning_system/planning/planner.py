"""
Main Planning Module

Meta-agent that analyzes problems and creates agent workflows.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
import networkx as nx
import json

from agents.planning_system.core.agent import Agent, AgentRole, AgentState, SpawnConfig
from agents.planning_system.core.plugin import Plugin
from agents.planning_system.core.llm import LLMConfig, get_high_reasoning_config
from agents.planning_system.core.history import ExecutionHistory


@dataclass
class ProblemSpecification:
    """Input to the planning module"""

    # Problem description
    task_description: str  # Natural language description
    competition_id: str  # MLE-Bench competition ID

    # Data information
    data_files: List[str] = field(default_factory=list)  # Available data files
    data_description: str = ""  # Description of datasets

    # Success criteria
    evaluation_metric: str = ""  # e.g., "accuracy", "rmse"
    submission_format: str = ""  # Expected output format

    # Constraints
    time_limit_hours: int = 24
    compute_resources: Dict = field(default_factory=dict)

    # Additional context
    domain: str = ""  # e.g., "computer_vision", "nlp", "tabular"
    difficulty: str = ""  # "low", "medium", "high"

    @classmethod
    def from_competition(cls, competition_id: str, mlebench_dir: Path) -> "ProblemSpecification":
        """Create from MLE-Bench competition"""
        comp_dir = mlebench_dir / "mlebench" / "competitions" / competition_id

        # Read description
        desc_file = comp_dir / "description.md"
        if desc_file.exists():
            task_description = desc_file.read_text()
        else:
            task_description = f"Solve the {competition_id} competition"

        # Read config
        import yaml
        config_file = comp_dir / "config.yaml"
        if config_file.exists():
            with open(config_file) as f:
                config = yaml.safe_load(f)
        else:
            config = {}

        # Get data files
        public_dir = comp_dir / "prepared" / "public"
        if public_dir.exists():
            data_files = [str(f.relative_to(public_dir)) for f in public_dir.glob("*")]
        else:
            data_files = []

        return cls(
            task_description=task_description,
            competition_id=competition_id,
            data_files=data_files,
            evaluation_metric=config.get("metric", "unknown"),
            domain=config.get("domain", "unknown"),
        )


@dataclass
class Subtask:
    """A subtask in the workflow"""

    name: str  # Unique name
    description: str  # What needs to be done
    dependencies: List[str] = field(default_factory=list)  # Names of prerequisite subtasks
    estimated_time_minutes: int = 60
    requires_complex_reasoning: bool = False


@dataclass
class WorkflowPlan:
    """Output of the planning module"""

    # Analysis
    problem_analysis: Dict  # Problem understanding

    # Task breakdown
    subtasks: List[Subtask]  # Ordered list of subtasks

    # Agent composition
    agents: List[Agent]  # Agent specifications

    # Plugin assignments
    plugins: Dict[str, List[Plugin]]  # agent_id → plugins

    # Execution plan
    workflow_graph: nx.DiGraph  # DAG of agent dependencies

    # Resource planning
    resource_estimates: Dict  # Time, memory, cost estimates

    # Human review
    requires_approval: bool = True
    approval_checkpoints: List[str] = field(default_factory=list)

    def visualize(self) -> str:
        """Generate visual representation of workflow"""
        lines = ["# Workflow Plan\n"]

        lines.append("## Problem Analysis")
        lines.append(f"- Type: {self.problem_analysis.get('problem_type', 'Unknown')}")
        lines.append(f"- Complexity: {self.problem_analysis.get('complexity', 'Unknown')}")
        lines.append("")

        lines.append("## Subtasks")
        for i, subtask in enumerate(self.subtasks, 1):
            lines.append(f"{i}. **{subtask.name}**: {subtask.description}")
            if subtask.dependencies:
                lines.append(f"   - Dependencies: {', '.join(subtask.dependencies)}")
        lines.append("")

        lines.append("## Agents")
        for agent in self.agents:
            lines.append(f"- **{agent.role.role_name}**: {agent.role.description}")
            lines.append(f"  - LLM: {agent.llm_config.model_name}")
            lines.append(f"  - Max steps: {agent.role.max_steps}")

        lines.append("")
        lines.append("## Resource Estimates")
        lines.append(f"- Total time: {self.resource_estimates.get('total_time_hours', 'Unknown')} hours")
        lines.append(f"- Estimated cost: ${self.resource_estimates.get('total_cost_usd', 0):.2f}")

        return "\n".join(lines)

    def to_config_files(self, output_dir: Path):
        """Generate all configuration files for execution"""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save plan summary
        with open(output_dir / "workflow_plan.json", "w") as f:
            json.dump(
                {
                    "problem_analysis": self.problem_analysis,
                    "subtasks": [
                        {
                            "name": st.name,
                            "description": st.description,
                            "dependencies": st.dependencies,
                        }
                        for st in self.subtasks
                    ],
                    "agents": [agent.to_dict() for agent in self.agents],
                    "resource_estimates": self.resource_estimates,
                },
                f,
                indent=2,
            )

        # Save visualization
        with open(output_dir / "workflow_plan.md", "w") as f:
            f.write(self.visualize())


class PlanningModule:
    """
    Meta-agent that analyzes problems and creates agent workflows
    """

    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        agent_templates_file: Optional[Path] = None,
        plugin_catalog_file: Optional[Path] = None,
        model_provider: str = "openai",  # "openai", "anthropic", or "opensource"
    ):
        self.model_provider = model_provider
        self.llm_config = llm_config or self._get_planner_config()
        self.agent_templates = self._load_agent_templates(agent_templates_file)
        self.plugin_catalog = self._load_plugin_catalog(plugin_catalog_file)

    def _get_planner_config(self) -> LLMConfig:
        """Get LLM config for the planner based on provider"""
        if self.model_provider == "opensource":
            from agents.planning_system.core.llm import get_oss_high_reasoning_config
            return get_oss_high_reasoning_config()
        else:
            return get_high_reasoning_config()

    def _load_agent_templates(self, filepath: Optional[Path]) -> Dict:
        """Load agent templates"""
        # For now, return default templates
        # TODO: Load from YAML file
        return {
            "data_explorer": {
                "role_name": "DataExplorer",
                "description": "Explores and analyzes datasets",
                "max_steps": 50,
                "max_time_seconds": 1800,
            },
            "model_trainer": {
                "role_name": "ModelTrainer",
                "description": "Trains and evaluates ML models",
                "max_steps": 200,
                "max_time_seconds": 18000,
            },
            "submission_validator": {
                "role_name": "SubmissionValidator",
                "description": "Validates and prepares submissions",
                "max_steps": 20,
                "max_time_seconds": 600,
            },
        }

    def _load_plugin_catalog(self, filepath: Optional[Path]) -> Dict:
        """Load plugin catalog"""
        # For now, return empty
        # Plugins will be registered separately
        return {}

    def plan_workflow(self, problem_spec: ProblemSpecification) -> WorkflowPlan:
        """
        Main planning function

        Input: Problem specification
        Output: Complete workflow plan with agents and plugins
        """
        print(f"\n{'='*60}")
        print(f"Planning workflow for: {problem_spec.competition_id}")
        print(f"{'='*60}\n")

        # Phase 1: Problem Understanding
        print("Phase 1: Analyzing problem...")
        problem_analysis = self._analyze_problem(problem_spec)

        # Phase 2: Task Decomposition
        print("Phase 2: Decomposing into subtasks...")
        subtasks = self._decompose_into_subtasks(problem_analysis, problem_spec)

        # Phase 3: Agent Selection & Composition
        print("Phase 3: Designing agent team...")
        agent_specs = self._design_agents(subtasks, problem_analysis)

        # Phase 4: Plugin Selection
        print("Phase 4: Assigning plugins...")
        plugin_assignments = self._assign_plugins(agent_specs, subtasks)

        # Phase 5: Workflow Graph Construction
        print("Phase 5: Building workflow graph...")
        workflow_graph = self._build_workflow_graph(agent_specs, plugin_assignments, subtasks)

        # Phase 6: Resource Estimation
        print("Phase 6: Estimating resources...")
        resource_estimates = self._estimate_resources(workflow_graph, agent_specs)

        print("\nPlanning complete!\n")

        # Return complete plan
        return WorkflowPlan(
            problem_analysis=problem_analysis,
            subtasks=subtasks,
            agents=agent_specs,
            plugins=plugin_assignments,
            workflow_graph=workflow_graph,
            resource_estimates=resource_estimates,
        )

    def _analyze_problem(self, problem_spec: ProblemSpecification) -> Dict:
        """Analyze the problem to understand its characteristics"""
        # Simple rule-based analysis for now
        # TODO: Use LLM for deeper analysis

        analysis = {
            "problem_type": "classification",  # Default
            "data_modality": "tabular",  # Default
            "complexity": problem_spec.difficulty or "medium",
            "challenges": ["missing data", "feature engineering"],
            "recommended_approaches": ["tree-based models", "neural networks"],
        }

        # Heuristics based on competition ID and description
        desc_lower = problem_spec.task_description.lower()

        # Determine problem type
        if any(word in desc_lower for word in ["segment", "mask", "pixel"]):
            analysis["problem_type"] = "segmentation"
            analysis["data_modality"] = "image"
        elif any(word in desc_lower for word in ["image", "photo", "picture"]):
            analysis["problem_type"] = "classification"
            analysis["data_modality"] = "image"
        elif any(word in desc_lower for word in ["text", "nlp", "language"]):
            analysis["data_modality"] = "text"
        elif any(word in desc_lower for word in ["regression", "predict", "forecast"]):
            analysis["problem_type"] = "regression"

        return analysis

    def _decompose_into_subtasks(
        self, problem_analysis: Dict, problem_spec: ProblemSpecification
    ) -> List[Subtask]:
        """Break down the problem into subtasks"""
        problem_type = problem_analysis["problem_type"]
        data_modality = problem_analysis["data_modality"]

        # Template-based decomposition
        if data_modality == "tabular":
            subtasks = [
                Subtask(
                    "data_exploration",
                    "Explore and understand the dataset",
                    dependencies=[],
                    estimated_time_minutes=30,
                ),
                Subtask(
                    "data_preprocessing",
                    "Clean and preprocess data",
                    dependencies=["data_exploration"],
                    estimated_time_minutes=60,
                ),
                Subtask(
                    "feature_engineering",
                    "Create and select features",
                    dependencies=["data_preprocessing"],
                    estimated_time_minutes=90,
                    requires_complex_reasoning=True,
                ),
                Subtask(
                    "model_training",
                    "Train and evaluate models",
                    dependencies=["feature_engineering"],
                    estimated_time_minutes=180,
                ),
                Subtask(
                    "hyperparameter_tuning",
                    "Optimize model parameters",
                    dependencies=["model_training"],
                    estimated_time_minutes=120,
                ),
                Subtask(
                    "validation",
                    "Validate and prepare submission",
                    dependencies=["hyperparameter_tuning"],
                    estimated_time_minutes=30,
                ),
            ]
        elif data_modality == "image":
            subtasks = [
                Subtask(
                    "data_exploration",
                    "Analyze images and labels",
                    dependencies=[],
                    estimated_time_minutes=45,
                ),
                Subtask(
                    "data_augmentation",
                    "Create augmented training data",
                    dependencies=["data_exploration"],
                    estimated_time_minutes=60,
                ),
                Subtask(
                    "model_architecture",
                    "Design or select model architecture",
                    dependencies=["data_exploration"],
                    estimated_time_minutes=90,
                    requires_complex_reasoning=True,
                ),
                Subtask(
                    "model_training",
                    "Train deep learning model",
                    dependencies=["data_augmentation", "model_architecture"],
                    estimated_time_minutes=360,
                ),
                Subtask(
                    "validation",
                    "Validate and prepare submission",
                    dependencies=["model_training"],
                    estimated_time_minutes=30,
                ),
            ]
        else:
            # Generic workflow
            subtasks = [
                Subtask("data_exploration", "Explore data", dependencies=[]),
                Subtask("model_training", "Train model", dependencies=["data_exploration"]),
                Subtask("validation", "Validate submission", dependencies=["model_training"]),
            ]

        return subtasks

    def _design_agents(self, subtasks: List[Subtask], problem_analysis: Dict) -> List[Agent]:
        """Design specialized agents for subtasks"""
        agents = []

        # Create one agent per subtask (specialist approach)
        for subtask in subtasks:
            agent = self._create_agent_for_subtask(subtask, problem_analysis)
            agents.append(agent)

        return agents

    def _create_agent_for_subtask(
        self, subtask: Subtask, problem_analysis: Dict
    ) -> Agent:
        """Create a specialized agent for a specific subtask"""
        # Select LLM based on task complexity and model provider
        if self.model_provider == "opensource":
            from agents.planning_system.core.llm import (
                get_oss_default_config,
                get_oss_code_generation_config,
                get_oss_fast_execution_config,
            )

            if subtask.requires_complex_reasoning:
                llm_config = get_oss_code_generation_config()
            elif subtask.estimated_time_minutes < 60:
                llm_config = get_oss_fast_execution_config()
            else:
                llm_config = get_oss_default_config()
        else:
            from agents.planning_system.core.llm import (
                get_default_config,
                get_code_generation_config,
                get_fast_execution_config,
            )

            if subtask.requires_complex_reasoning:
                llm_config = get_code_generation_config()  # Use Claude for complex tasks
            elif subtask.estimated_time_minutes < 60:
                llm_config = get_fast_execution_config()  # Use mini for quick tasks
            else:
                llm_config = get_default_config()  # Use default for regular tasks

        # Create role
        role = AgentRole(
            role_name=subtask.name.replace("_", " ").title().replace(" ", ""),
            description=subtask.description,
            allowed_actions=self._determine_actions_for_subtask(subtask),
            max_steps=self._estimate_steps(subtask),
            max_time_seconds=subtask.estimated_time_minutes * 60,
            success_condition=f"Complete: {subtask.description}",
        )

        # Create agent
        return Agent(
            agent_id=f"agent_{subtask.name}",
            llm_config=llm_config,
            role=role,
            state=AgentState(),
            spawn_config=SpawnConfig(can_spawn=False),
            history=ExecutionHistory(),
        )

    def _determine_actions_for_subtask(self, subtask: Subtask) -> List[str]:
        """Determine which actions are needed for a subtask"""
        actions = ["execute_python", "read_file", "write_file"]

        if "exploration" in subtask.name.lower():
            actions.extend(["visualize_data", "describe_data"])
        elif "training" in subtask.name.lower():
            actions.extend(["train_model", "evaluate_model"])
        elif "validation" in subtask.name.lower():
            actions.extend(["validate_submission"])

        return actions

    def _estimate_steps(self, subtask: Subtask) -> int:
        """Estimate number of steps needed"""
        # Rough estimate: 1 step per 3 minutes
        return max(10, subtask.estimated_time_minutes // 3)

    def _assign_plugins(
        self, agents: List[Agent], subtasks: List[Subtask]
    ) -> Dict[str, List[Plugin]]:
        """Assign appropriate plugins to each agent"""
        assignments = {}

        for agent in agents:
            # For now, assign empty list
            # Plugins will be loaded separately
            assignments[agent.agent_id] = []

        return assignments

    def _build_workflow_graph(
        self, agents: List[Agent], plugin_assignments: Dict, subtasks: List[Subtask]
    ) -> nx.DiGraph:
        """Build workflow graph from subtask dependencies"""
        graph = nx.DiGraph()

        # Create mapping of subtask name to agent ID
        subtask_to_agent = {}
        for agent, subtask in zip(agents, subtasks):
            subtask_to_agent[subtask.name] = agent.agent_id

        # Add nodes
        for agent in agents:
            graph.add_node(agent.agent_id, agent=agent)

        # Add edges based on dependencies
        for subtask in subtasks:
            curr_agent_id = subtask_to_agent[subtask.name]

            for dep_name in subtask.dependencies:
                if dep_name in subtask_to_agent:
                    dep_agent_id = subtask_to_agent[dep_name]
                    graph.add_edge(dep_agent_id, curr_agent_id)

        # Validate DAG
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError("Workflow graph contains cycles!")

        return graph

    def _estimate_resources(
        self, workflow_graph: nx.DiGraph, agents: List[Agent]
    ) -> Dict:
        """Estimate resource requirements"""
        total_time_seconds = sum(agent.role.max_time_seconds for agent in agents)

        # Estimate cost (rough)
        # GPT-4o: ~$2.50 per million input tokens, ~$10 per million output tokens
        # Assume ~10k tokens per agent on average
        avg_cost_per_agent = 0.10  # $0.10 per agent
        total_cost = len(agents) * avg_cost_per_agent

        return {
            "total_time_hours": total_time_seconds / 3600,
            "total_cost_usd": total_cost,
            "num_agents": len(agents),
            "max_parallel_agents": self._compute_parallelism(workflow_graph),
        }

    def _compute_parallelism(self, graph: nx.DiGraph) -> int:
        """Compute maximum parallelism in workflow"""
        if graph.number_of_nodes() == 0:
            return 0

        # Find maximum width of any level in the DAG
        levels = list(nx.topological_generations(graph))
        return max(len(level) for level in levels) if levels else 1
