# Quick Start Guide

Get started with the Agentic Planning System in 5 minutes!

## Prerequisites

```bash
# Install core dependencies
pip install networkx pyyaml

# Optional: Install data/ML libraries for plugins
pip install pandas numpy scikit-learn

# Optional: Install LLM provider
pip install openai  # or anthropic
```

Set your API key:
```bash
export OPENAI_API_KEY="sk-..."
```

## Three Ways to Use the System

### 1. Run the Example (No Setup Required)

```bash
cd /path/to/mle-bench
python agents/planning_system/examples/simple_example.py
```

This will:
- Create a workflow plan for the "spaceship-titanic" competition
- Show you the agent team and subtasks
- Display the execution plan

**Output:**
```
============================================================
Agentic Planning System - Simple Example
============================================================

Creating planning module...
Generating workflow plan...

[Shows complete workflow plan with agents and plugins]
```

### 2. Plan a Workflow

Generate a plan without executing:

```bash
cd /path/to/mle-bench
python agents/planning_system/run.py plan spaceship-titanic --output ./my_plans
```

**Output:**
- `./my_plans/spaceship-titanic/workflow_plan.json` - Machine-readable plan
- `./my_plans/spaceship-titanic/workflow_plan.md` - Human-readable plan

### 3. Plan + Execute

**⚠️ Note:** This requires API credits and will make LLM calls!

```bash
python agents/planning_system/run.py run spaceship-titanic \
    --workspace ./workspace \
    --output ./output \
    --skip-approval
```

This will:
1. Analyze the problem
2. Create specialized agents
3. Execute the workflow
4. Generate predictions
5. Save results to `./output/spaceship-titanic/`

## What Gets Created

After running a workflow, you'll see:

```
output/spaceship-titanic/
├── workflow_plan.json          # The generated plan
├── workflow_plan.md             # Human-readable plan
├── execution_report.json        # Results and metrics
└── (workspace files from agents)
```

## Common Use Cases

### Generate a Plan for Any Competition

```bash
python agents/planning_system/run.py plan <competition-id> --output ./plans
```

Examples:
- `spaceship-titanic` (tabular, simple)
- `house-prices-advanced-regression-techniques` (tabular, medium)
- `digit-recognizer` (image, simple)

### Customize Agent Configuration

Edit `agents/planning_system/planning/planner.py` to:
- Add new problem types
- Customize subtask decomposition
- Adjust agent roles and LLM models
- Modify resource allocations

### Create Custom Plugins

See `agents/planning_system/plugins/` for examples:
1. Define actions (functions)
2. Create `PluginFunctionality`
3. Set `PluginConfig` and `PluginConstraints`
4. Return a `Plugin` instance

Example:
```python
from agents.planning_system.core.plugin import Plugin, Action, ...

def my_action(param: str) -> dict:
    return {"result": f"Processed {param}"}

def create_my_plugin() -> Plugin:
    actions = {
        "my_action": Action(
            action_name="my_action",
            description="Does something useful",
            parameters={"param": {"type": "string", "required": True}},
            handler=my_action,
        )
    }
    # ... create Plugin with functionality, config, constraints
    return Plugin(...)
```

## Troubleshooting

### ModuleNotFoundError: No module named 'agents'

**Solution:** Use the standalone runner:
```bash
python agents/planning_system/run.py <command>
```

NOT:
```bash
python -m agents.planning_system.cli  # ❌ Requires PYTHONPATH setup
```

### Missing pandas/numpy/scikit-learn

These are optional for planning. Install only if you want to use the built-in plugins:
```bash
pip install pandas numpy scikit-learn
```

### No OpenAI API key

Set it in your environment:
```bash
export OPENAI_API_KEY="sk-..."
```

Or use a different provider:
```python
from agents.planning_system.core.llm import LLMConfig

config = LLMConfig(
    model_name="claude-3-5-sonnet-20241022",
    provider="anthropic",
)
```

## Next Steps

1. **Read the README:** `agents/planning_system/README.md`
2. **Read the Design Doc:** `docs/agentic_planning_system_design.md`
3. **Run Tests:** `python agents/planning_system/tests/test_basic.py`
4. **Explore Plugins:** Check `agents/planning_system/plugins/`
5. **Build Your Own:** Extend the system with custom agents and plugins!

## Getting Help

- Check the README for detailed documentation
- Look at example code in `examples/simple_example.py`
- Review plugin implementations in `plugins/`
- Read the design document for architecture details

## Quick Reference

```bash
# Plan only
python agents/planning_system/run.py plan <competition>

# Plan + Execute (with approval)
python agents/planning_system/run.py run <competition>

# Plan + Execute (skip approval)
python agents/planning_system/run.py run <competition> --skip-approval

# Help
python agents/planning_system/run.py --help
python agents/planning_system/run.py plan --help
```

Happy planning! 🚀
