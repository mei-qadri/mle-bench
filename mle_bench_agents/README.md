# MLE-bench Multi-Agent System - Phase I Implementation

**Version:** 1.0.0  
**Status:** Core Infrastructure Complete

This package contains the Phase I implementation of the multi-agent system for solving MLE-bench competitions.

## Overview

The MLE-bench Multi-Agent System coordinates 7 specialized agents working together to solve machine learning competitions. This Phase I implementation provides the core infrastructure:

- **Message Passing System**: Priority-based message queue with broadcast support
- **State Management**: Thread-safe global and agent-specific state with versioning
- **Base Agent Framework**: Abstract base class for all agents
- **Agent Implementations**: Orchestrator (A1) and Resource Manager (A7)
- **12 Plug Components**: Reusable functionality modules
- **Testing Framework**: Unit tests for core components

## Installation

### Basic Installation

```bash
cd mle_bench_agents
pip install -e .
```

### With Deep Learning Support

```bash
pip install -e ".[torch,transformers,vision]"
```

### Development Installation

```bash
pip install -e ".[dev]"
```

## Project Structure

```
mle_bench_agents/
├── core/                          # Core infrastructure
│   ├── agent.py                  # Base agent class
│   ├── message.py                # Message system
│   ├── queue.py                  # Message queue
│   └── state.py                  # State management
├── agents/                       # Agent implementations
│   ├── orchestrator.py           # A1: Orchestrator
│   └── resource_manager.py       # A7: Resource Manager
├── plugs/                        # Plug components (P1-P12)
│   ├── base_plug.py             # Base plug class
│   ├── data_loading.py          # P1
│   ├── eda.py                   # P2
│   ├── preprocessing.py         # P3
│   ├── feature_engineering.py   # P4
│   ├── model_training.py        # P5
│   ├── hpo.py                   # P6
│   ├── cross_validation.py      # P7
│   ├── ensemble.py              # P8
│   ├── submission.py            # P9
│   ├── validation.py            # P10
│   ├── metrics.py               # P11
│   └── logging.py               # P12
├── utils/                       # Utilities
│   ├── config_loader.py        # Configuration loading
│   ├── logger.py               # Logging utilities
│   └── helpers.py              # Helper functions
└── tests/                      # Test suite
    ├── test_messages.py
    ├── test_state.py
    └── test_queue.py
```

## Quick Start

### 1. Basic Message Passing

```python
from mle_bench_agents.core.message import Message, MessageType, create_request_message
from mle_bench_agents.core.queue import MessageQueue

# Create message queue
queue = MessageQueue()

# Create and send message
msg = create_request_message(
    sender_id="A1",
    receiver_id="A2",
    action="analyze_competition",
    competition_id="spaceship-titanic"
)

queue.send(msg)

# Receive message
received = queue.receive("A2", timeout=5.0)
print(f"Received: {received}")
```

### 2. State Management

```python
from mle_bench_agents.core.state import StateManager

# Create state manager
state_mgr = StateManager()

# Update global state
state_mgr.update_state({
    "competition_id": "spaceship-titanic",
    "competition_type": "tabular",
    "time_budget_total": 86400
})

# Access state
state = state_mgr.get_state()
print(f"Competition: {state.competition_id}")
print(f"Time remaining: {state.time_remaining}s")
```

### 3. Using Plugs

```python
from mle_bench_agents.plugs.data_loading import DataLoadingPlug
from mle_bench_agents.plugs.metrics import MetricsPlug

# Load data
data_plug = DataLoadingPlug()
df = data_plug.execute("train.csv")

# Compute metrics
metrics_plug = MetricsPlug()
score = metrics_plug.execute(y_true, y_pred, metric="accuracy")
print(f"Accuracy: {score}")
```

### 4. Running the Orchestrator

```python
from mle_bench_agents.core.queue import MessageQueue
from mle_bench_agents.core.state import StateManager
from mle_bench_agents.agents.orchestrator import OrchestratorAgent

# Setup infrastructure
queue = MessageQueue()
state_mgr = StateManager()

# Create orchestrator
orchestrator = OrchestratorAgent(
    agent_id="A1_orchestrator",
    message_queue=queue,
    state_manager=state_mgr,
    config={}
)

# Run competition
result = orchestrator.execute({
    "competition_id": "spaceship-titanic",
    "time_budget": 3600  # 1 hour for testing
})

print(f"Result: {result}")
```

## Running Tests

```bash
# Run all tests
pytest mle_bench_agents/tests/ -v

# Run specific test
pytest mle_bench_agents/tests/test_messages.py -v

# Run with coverage
pytest mle_bench_agents/tests/ --cov=mle_bench_agents --cov-report=html
```

## Architecture

### Agent System

Each agent is defined as **Ai = {Li, Ri, Si, Ci, Hi}**:

- **Li**: Language model and configuration
- **Ri**: Role and responsibilities
- **Si**: State structure
- **Ci**: Can spawn children (boolean)
- **Hi**: History tracking configuration

### Plug System

Each plug is defined as **Pj = {Fj, Cj, Uj}**:

- **Fj**: Functionality (actions performed)
- **Cj**: Configuration parameters
- **Uj**: Constraints and usage rules

### Message Types

- **REQUEST**: Agent requests action from another agent
- **RESPONSE**: Reply to a request
- **UPDATE**: State update notification
- **ERROR**: Error notification
- **ALERT**: Urgent notification (resource limits, time)

## Configuration

Agent and plug configurations are loaded from YAML files:

- **Agent Config**: `docs/config/agent_configs.yaml`
- **Message Templates**: `docs/config/message_templates.yaml`

Example:

```python
from mle_bench_agents.utils.config_loader import load_agent_config

# Load agent configuration
config = load_agent_config("A1_orchestrator")
print(f"Model: {config['model']}")
print(f"Temperature: {config['temperature']}")
```

## Workflow Example

The orchestrator coordinates agents through 5 phases:

```
Phase 1: Understanding (10% time)
    └─ A2: Analysis Agent analyzes competition

Phase 2: Preparation (15% time)
    └─ A4: Engineering Agent prepares data

Phase 3: Modeling (60% time)
    └─ A3: Domain Specialist trains models

Phase 4: Ensembling (10% time)
    └─ A5: Ensemble Agent combines models

Phase 5: Submission (5% time)
    └─ A6: Validation Agent validates submission

Continuous:
    └─ A7: Resource Manager monitors resources
```

## Development

### Adding New Agents

1. Create agent class inheriting from `Agent`
2. Implement `process_message()` and `execute()` methods
3. Add to `agents/__init__.py`
4. Create system prompt in `docs/prompts/`

### Adding New Plugs

1. Create plug class inheriting from `BasePlug`
2. Implement `execute()` and `get_constraints()` methods
3. Add to `plugs/__init__.py`

### Code Style

```bash
# Format code
black mle_bench_agents/

# Lint code
flake8 mle_bench_agents/

# Type checking
mypy mle_bench_agents/
```

## Phase I Completion Status

✅ **Completed:**
- Message passing system
- State management with versioning
- Message queue with priority routing
- Base agent framework
- Orchestrator agent (A1)
- Resource manager agent (A7)
- All 12 plugs (basic implementations)
- Testing framework
- Configuration loading
- Logging utilities

📋 **Next Steps (Phase II):**
- Implement remaining agents (A2, A3a-d, A4, A5, A6)
- Enhance plug implementations
- Add LLM integration
- MLE-bench environment integration
- End-to-end workflow testing

## Resources

- **Design Document**: `../docs/multi_agent_system_design.md`
- **Workflow Specification**: `../docs/workflow_orchestration.md`
- **Agent Prompts**: `../docs/prompts/`
- **Architecture Diagrams**: `../docs/multi_agent_architecture_diagrams.md`

## License

MIT License

## Contact

For questions or contributions, please see the main MLE-bench repository.
