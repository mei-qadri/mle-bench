# Agentic Planning System for MLE-Bench

A meta-agent system that analyzes ML engineering problems and dynamically composes specialized agent workflows to solve them.

## Overview

This system acts as an **orchestrator** that:

1. **Understands** the problem domain and requirements
2. **Plans** the optimal workflow and agent composition
3. **Collaborates** with humans to validate the approach
4. **Instantiates** specialized agents with appropriate tools
5. **Coordinates** multi-agent collaboration toward solution

## Architecture

### Core Components

#### Agent: `Ai = {Li, Ri, Si, Ci, Hi}`

- **Li (LLM Config)**: Language model configuration (model, temperature, tokens, etc.)
- **Ri (Role)**: Agent's purpose, capabilities, constraints, and success criteria
- **Si (State)**: Current status, working memory, observations, and outputs
- **Ci (Spawn Config)**: Ability to create child agents
- **Hi (History)**: Complete execution trace with events and performance metrics

#### Plugin: `Pj = {Fj, Cj, Uj}`

- **Fj (Functionality)**: Actions provided with parameters and return types
- **Cj (Configuration)**: Timeouts, resource limits, logging settings
- **Uj (Constraints)**: Usage limits, resource requirements, security sandbox

### System Workflow

```
Problem Specification
    ↓
Planning Module
    ├─ Problem Understanding
    ├─ Task Decomposition
    ├─ Agent Design
    ├─ Plugin Assignment
    └─ Workflow Graph Construction
    ↓
Human Collaboration
    ├─ Plan Review
    └─ Approval/Modification
    ↓
Execution Engine
    ├─ Agent Instantiation
    ├─ Sequential/Parallel Execution
    └─ Result Collection
    ↓
Final Outputs & Reports
```

## Installation

### Prerequisites

```bash
# Core dependencies
pip install pandas numpy scikit-learn networkx pyyaml

# LLM providers - Choose one or more:

# Option 1: Commercial APIs (easiest, but costs money)
pip install openai anthropic

# Option 2: Open-Source Models (free, private, runs locally)
pip install transformers accelerate torch
pip install bitsandbytes  # For 4-bit quantization

# See OPENSOURCE_MODELS.md for complete guide
```

### Setup

The planning system is already integrated into the MLE-Bench repository:

```bash
cd /home/user/mle-bench/agents/planning_system
```

## Usage

### Command-Line Interface

**Note:** Use the standalone runner script for easiest usage:

#### 1. Plan Only

Generate a workflow plan without executing:

```bash
# From mle-bench directory
cd /path/to/mle-bench
python agents/planning_system/run.py plan spaceship-titanic --output ./plans

# Or use the shell script
./agents/planning_system/run_planner.sh plan spaceship-titanic --output ./plans
```

#### 2. Plan + Execute

Plan and execute in one step:

```bash
python agents/planning_system/run.py run spaceship-titanic \
    --workspace ./workspace \
    --output ./output

# Or with shell script
./agents/planning_system/run_planner.sh run spaceship-titanic \
    --workspace ./workspace \
    --output ./output
```

#### 3. Skip Human Approval

For automated workflows:

```bash
python agents/planning_system/run.py run spaceship-titanic \
    --skip-approval \
    --workspace ./workspace
```

### Python API

#### Basic Usage

```python
from pathlib import Path
from agents.planning_system.planning.planner import (
    PlanningModule,
    ProblemSpecification,
)
from agents.planning_system.execution.engine import WorkflowExecutionEngine

# Define problem
problem_spec = ProblemSpecification(
    task_description="Binary classification task",
    competition_id="spaceship-titanic",
    data_files=["train.csv", "test.csv"],
    evaluation_metric="accuracy",
)

# Generate plan
planner = PlanningModule()
plan = planner.plan_workflow(problem_spec)

# Execute plan
workspace = Path("./workspace/spaceship-titanic")
data_dir = Path("/home/data")

engine = WorkflowExecutionEngine(plan, workspace, data_dir)
result = engine.execute()

print(f"Success: {result.success}")
```

#### Using Plugins

```python
from agents.planning_system.plugins import (
    create_data_exploration_plugin,
    create_ml_training_plugin,
    create_submission_validation_plugin,
)

# Create plugins
data_plugin = create_data_exploration_plugin()
ml_plugin = create_ml_training_plugin()
validation_plugin = create_submission_validation_plugin()

# Assign to agents
for agent in plan.agents:
    if "exploration" in agent.agent_id:
        plan.plugins[agent.agent_id] = [data_plugin]
    elif "training" in agent.agent_id:
        plan.plugins[agent.agent_id] = [ml_plugin]
    # ... etc
```

## Examples

### Simple Example

See `examples/simple_example.py`:

```bash
python agents/planning_system/examples/simple_example.py
```

This demonstrates:
- Creating a problem specification
- Generating a workflow plan
- Assigning plugins to agents
- (Optional) Executing the workflow

### Using Open-Source Models

See `examples/opensource_models_example.py` and `OPENSOURCE_MODELS.md`:

```bash
python agents/planning_system/examples/opensource_models_example.py
```

Run the system with **zero API costs** using local open-source models:

```python
from agents.planning_system.core.llm import get_llama_3_8b_config
from agents.planning_system.planning.planner import PlanningModule

# Use Llama 3.1 8B (4-bit quantization, ~4GB RAM)
llm_config = get_llama_3_8b_config(load_in_4bit=True)
planner = PlanningModule(llm_config=llm_config)

# Generate plan using local model (no API costs!)
plan = planner.plan_workflow(problem_spec)
```

**Supported Models:**
- GPT OSS 20B/120B (reasoning with mxfp4)
- Llama 3.1 8B (general purpose, 4-bit)
- Mistral 7B (efficient, 4-bit)
- Qwen 2.5 7B (multilingual, 4-bit)

**Benefits:**
- ✓ No API costs
- ✓ Full privacy
- ✓ No rate limits
- ✓ Offline operation

See **[OPENSOURCE_MODELS.md](OPENSOURCE_MODELS.md)** for complete guide!

## Plugin Development

### Creating a Custom Plugin

```python
from agents.planning_system.core.plugin import (
    Plugin,
    PluginFunctionality,
    PluginConfig,
    PluginConstraints,
    Action,
)

def my_custom_action(param1: str, param2: int) -> dict:
    """Your custom action implementation"""
    result = # ... do something
    return {"success": True, "result": result}

def create_my_plugin() -> Plugin:
    actions = {
        "my_action": Action(
            action_name="my_action",
            description="What this action does",
            parameters={
                "param1": {"type": "string", "required": True},
                "param2": {"type": "integer", "required": False},
            },
            returns={"type": "dict"},
            handler=my_custom_action,
        ),
    }

    functionality = PluginFunctionality(
        plugin_id="my_plugin",
        plugin_name="My Custom Plugin",
        description="Plugin description",
        actions=actions,
    )

    config = PluginConfig(timeout=300)
    constraints = PluginConstraints()

    return Plugin(
        functionality=functionality,
        config=config,
        constraints=constraints,
    )
```

## Available Plugins

### 1. Data Exploration Plugin

**Actions:**
- `read_csv`: Read and summarize CSV files
- `describe_data`: Generate statistical summaries
- `detect_missing`: Analyze missing values
- `analyze_columns`: Identify column types
- `compute_correlations`: Calculate feature correlations

### 2. ML Training Plugin

**Actions:**
- `train_model`: Train scikit-learn models
- `make_predictions`: Generate predictions
- `evaluate_model`: Evaluate on validation data

**Supported Models:**
- Random Forest (Classifier/Regressor)
- Logistic Regression
- Linear Regression

### 3. Submission Validation Plugin

**Actions:**
- `validate_format`: Check submission format
- `check_with_grader`: Validate with grading server
- `prepare_submission`: Format predictions for submission

## Configuration

### LLM Providers

Set API keys via environment variables:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Agent Roles

Default agent roles for tabular classification:
1. **DataExploration**: Explore and understand dataset
2. **DataPreprocessing**: Clean and preprocess data
3. **FeatureEngineering**: Create and select features
4. **ModelTraining**: Train and evaluate models
5. **HyperparameterTuning**: Optimize model parameters
6. **Validation**: Validate and prepare submission

### Customizing Plans

Plans are automatically generated based on:
- Problem type (classification, regression, segmentation, etc.)
- Data modality (tabular, image, text, etc.)
- Complexity level (low, medium, high)

## Output Structure

After execution, outputs are organized as:

```
output/
└── <competition-id>/
    ├── workflow_plan.json       # Plan specification
    ├── workflow_plan.md          # Human-readable plan
    ├── execution_report.json     # Execution results
    └── workspace/                # Agent workspaces
        ├── agent_data_exploration/
        │   ├── *_history.json    # Execution history
        │   └── *_history.md      # Readable history
        ├── agent_model_training/
        └── ...
```

## Design Philosophy

Inspired by:
- **AIDE**: Dual-model architecture (separate planning and execution models)
- **MLAgentBench**: Structured tool/action spaces with clear semantics
- **OpenDevin**: Code-as-action unified interface

Core principles:
- **Composability**: Agents and plugins are building blocks
- **Specialization**: Each agent has a focused role
- **Adaptability**: System adapts to problem characteristics
- **Human-in-the-loop**: Critical decisions validated by humans
- **Observability**: Full visibility into agent reasoning

## Roadmap

### Current (v0.1.0)
- ✅ Core agent and plugin system
- ✅ Planning module with task decomposition
- ✅ Basic plugins (data, ML, validation)
- ✅ Execution engine with coordination
- ✅ CLI interface

### Next (v0.2.0)
- [ ] GPU-based deep learning plugins (PyTorch, TensorFlow)
- [ ] Hyperparameter optimization plugins
- [ ] Enhanced LLM-based plan modification
- [ ] Parallel agent execution
- [ ] Web UI for plan visualization

### Future (v0.3.0)
- [ ] Multi-competition transfer learning
- [ ] Self-improving agents (learn from past runs)
- [ ] Custom agent role definitions via YAML
- [ ] Integration with more LLM providers
- [ ] Advanced debugging and visualization tools

## Troubleshooting

### Common Issues

**1. Import errors**
```bash
# Make sure you're in the mle-bench directory
cd /home/user/mle-bench
python -m agents.planning_system.cli run <competition>
```

**2. Missing API keys**
```bash
# Set environment variables
export OPENAI_API_KEY="your-key"
```

**3. Plugin execution failures**
```bash
# Check plugin dependencies
pip install pandas numpy scikit-learn joblib
```

## Contributing

To extend the system:

1. **Add new plugins**: Create in `agents/planning_system/plugins/`
2. **Customize agent roles**: Modify `planning/planner.py`
3. **Add new problem types**: Extend `_analyze_problem()` and `_decompose_into_subtasks()`
4. **Improve execution**: Enhance `execution/engine.py`

## References

- [MLE-Bench Paper](https://arxiv.org/abs/2410.07095)
- [Design Document](../../docs/agentic_planning_system_design.md)
- AIDE: Agentic IDE for ML Engineering
- MLAgentBench: Agent Benchmark for ML Tasks
- OpenDevin/OpenHands: Code-driven AI Agent

## License

Same as MLE-Bench parent project.
