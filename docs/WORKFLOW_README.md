# Multi-Agent System Workflow Implementation

**Version:** 1.0
**Date:** 2025-11-14
**Status:** Implementation Ready

## Overview

This directory contains the complete workflow orchestration and agent prompt specifications for the MLE-bench Multi-Agent System. These files enable the coordination of 7 specialized agents working together to solve machine learning competitions.

## Directory Structure

```
docs/
├── WORKFLOW_README.md                      # This file
├── workflow_orchestration.md               # Orchestration logic and workflows
├── prompts/                                # Agent system prompts
│   ├── A1_orchestrator.md                 # Orchestrator agent prompt
│   ├── A2_analysis.md                     # Analysis agent prompt
│   ├── A3a_cv_specialist.md               # Computer vision specialist
│   ├── A3b_nlp_specialist.md              # NLP specialist
│   ├── A3c_tabular_specialist.md          # Tabular specialist
│   ├── A3d_audio_specialist.md            # Audio specialist
│   ├── A4_engineering.md                  # Feature engineering agent
│   ├── A5_ensemble.md                     # Ensemble agent
│   ├── A6_validation.md                   # Validation agent
│   └── A7_resource_manager.md             # Resource manager agent
├── config/                                 # Configuration files
│   ├── agent_configs.yaml                 # Agent and plug configurations
│   └── message_templates.yaml             # Message passing templates
└── multi_agent_architecture_diagrams.md   # Visual diagrams

Also see:
├── multi_agent_system_design.md           # Main design document
└── README.md                              # Documentation index
```

## Quick Start

### 1. Understanding the System

Start by reading these documents in order:

1. **[multi_agent_system_design.md](./multi_agent_system_design.md)** - Complete system design
2. **[workflow_orchestration.md](./workflow_orchestration.md)** - How agents work together
3. **[multi_agent_architecture_diagrams.md](./multi_agent_architecture_diagrams.md)** - Visual reference

### 2. Agent Prompts

Each agent has a detailed system prompt defining its role, responsibilities, and expected behavior:

| Agent | Prompt File | Purpose |
|-------|-------------|---------|
| **A1** | [A1_orchestrator.md](./prompts/A1_orchestrator.md) | Main coordinator |
| **A2** | [A2_analysis.md](./prompts/A2_analysis.md) | Data analysis & EDA |
| **A3a** | [A3a_cv_specialist.md](./prompts/A3a_cv_specialist.md) | Computer vision |
| **A3b** | [A3b_nlp_specialist.md](./prompts/A3b_nlp_specialist.md) | Natural language |
| **A3c** | [A3c_tabular_specialist.md](./prompts/A3c_tabular_specialist.md) | Tabular data |
| **A3d** | [A3d_audio_specialist.md](./prompts/A3d_audio_specialist.md) | Audio processing |
| **A4** | [A4_engineering.md](./prompts/A4_engineering.md) | Feature engineering |
| **A5** | [A5_ensemble.md](./prompts/A5_ensemble.md) | Model ensembling |
| **A6** | [A6_validation.md](./prompts/A6_validation.md) | Submission validation |
| **A7** | [A7_resource_manager.md](./prompts/A7_resource_manager.md) | Resource monitoring |

### 3. Configuration

**Agent Configuration** ([config/agent_configs.yaml](./config/agent_configs.yaml)):
- Agent LLM settings (model, temperature, tokens)
- Plug configurations
- Workflow time allocations
- Resource thresholds
- Environment paths

**Message Templates** ([config/message_templates.yaml](./config/message_templates.yaml)):
- Standardized message formats
- Request/response templates
- Error and alert messages
- Example communication flows

## Workflow Execution

### Phase-by-Phase Execution

The system orchestrates competition solving in 5 sequential phases:

#### Phase 1: Understanding (10% time)
- **Active Agents**: A1 (Orchestrator), A2 (Analysis)
- **Goal**: Understand competition, analyze data
- **Output**: Competition type, complexity, recommendations
- **File**: See [workflow_orchestration.md](./workflow_orchestration.md#phase-1-understanding)

#### Phase 2: Preparation (15% time)
- **Active Agents**: A1 (Orchestrator), A4 (Engineering)
- **Goal**: Engineer features, create CV folds
- **Output**: Feature pipeline, preprocessed data
- **File**: See [workflow_orchestration.md](./workflow_orchestration.md#phase-2-preparation)

#### Phase 3: Modeling (60% time)
- **Active Agents**: A1 (Orchestrator), A3 (Domain Specialists), A7 (Resource Manager)
- **Sub-phases**: Baseline (15%) → Advanced (70%) → Final (15%)
- **Goal**: Train models, optimize hyperparameters
- **Output**: Trained models, predictions, checkpoints
- **File**: See [workflow_orchestration.md](./workflow_orchestration.md#phase-3-modeling)

#### Phase 4: Ensembling (10% time)
- **Active Agents**: A1 (Orchestrator), A5 (Ensemble)
- **Goal**: Combine models optimally
- **Output**: Final predictions
- **File**: See [workflow_orchestration.md](./workflow_orchestration.md#phase-4-ensembling)

#### Phase 5: Submission (5% time)
- **Active Agents**: A1 (Orchestrator), A6 (Validation)
- **Goal**: Validate and submit
- **Output**: submission.csv
- **File**: See [workflow_orchestration.md](./workflow_orchestration.md#phase-5-submission)

#### Continuous Monitoring
- **Active Agents**: A7 (Resource Manager), A6 (Validation)
- **Frequency**: Every 30 seconds
- **Purpose**: Resource tracking, alerts
- **File**: See [workflow_orchestration.md](./workflow_orchestration.md#resource-monitoring-workflows)

### Agent Invocation Patterns

The orchestrator uses different patterns to coordinate agents:

1. **Sequential Chain**: A2 → A4 → A3 (dependencies)
2. **Parallel Execution**: Multiple A3 specialists simultaneously
3. **Conditional Selection**: Choose A3a/b/c/d based on competition type
4. **Dynamic Spawning**: A3 spawns HPO sub-agents
5. **Continuous Monitoring**: A7 runs in background

See [workflow_orchestration.md#agent-invocation-patterns](./workflow_orchestration.md#agent-invocation-patterns) for details.

## Message Passing

### Message Structure

All inter-agent communication follows this structure:

```python
{
  "message_id": str,           # Unique identifier
  "timestamp": float,          # Unix timestamp
  "sender_id": str,            # Agent ID
  "receiver_id": str,          # Agent ID or "broadcast"
  "message_type": str,         # REQUEST, RESPONSE, UPDATE, ERROR, ALERT
  "priority": int,             # 0-10
  "correlation_id": str,       # For request-response pairing
  "requires_response": bool,
  "timeout": float,            # Seconds
  "payload": dict              # Message content
}
```

### Message Types

- **REQUEST**: Agent requests action from another agent
- **RESPONSE**: Reply to a request
- **UPDATE**: State update notification
- **ERROR**: Error notification with recovery suggestions
- **ALERT**: Urgent notification (resource limits, time running out)

See [config/message_templates.yaml](./config/message_templates.yaml) for all templates.

### Priority Levels

- **10**: Critical (emergency stop, time critical)
- **9**: High (training results, validation)
- **8**: Elevated (analysis complete, resource alerts)
- **7**: Medium (phase transitions, warnings)
- **6**: Normal (resource updates)
- **5**: Low (progress updates)
- **0-4**: Informational

## State Management

### Global State

The Orchestrator (A1) maintains global state:

```python
{
  # Competition info
  "competition_id": str,
  "competition_type": str,  # CV, NLP, tabular, audio
  "complexity": str,        # low, medium, high
  "metric_name": str,
  "metric_direction": str,

  # Time tracking
  "time_budget_total": int,
  "time_elapsed": int,
  "time_remaining": int,
  "phase_time_allocations": dict,

  # Progress
  "current_phase": str,
  "phase_status": dict,

  # Results
  "trained_models": list,
  "best_cv_score": float,
  "predictions": dict,
  "ensemble_predictions": array,

  # Resource tracking
  "resource_usage": dict,

  # Error tracking
  "errors": list,
  "warnings": list
}
```

### Agent-Specific State

Each agent maintains its own state (see individual prompts):
- A2: Dataset summary, EDA results, recommendations
- A3: Model configs, CV scores, checkpoints
- A4: Feature pipeline, engineered features
- A5: Ensemble strategy, weights
- A6: Validation results
- A7: Resource metrics, alerts

## Error Handling

### Error Classification

Errors are classified into 5 types:

1. **TRANSIENT**: Temporary, retry possible (network timeout)
2. **RESOURCE**: Resource exhaustion (OOM, disk full)
3. **DATA**: Data quality issues (corrupted files)
4. **CONFIGURATION**: Invalid configuration
5. **FATAL**: Unrecoverable error

### Recovery Strategies

For each error type, the system has fallback strategies:

- **Transient**: Retry with exponential backoff (3 attempts)
- **Resource**: Degrade configuration (smaller batch, simpler model)
- **Data**: Skip problematic data, use defaults
- **Configuration**: Use default configuration
- **Fatal**: Activate phase-specific fallback, ensure submission

See [workflow_orchestration.md#error-handling-workflows](./workflow_orchestration.md#error-handling-workflows).

## Resource Management

### Monitored Resources

A7 (Resource Manager) continuously monitors:

- **CPU**: Usage percentage
- **Memory**: Usage percentage and absolute GB
- **GPU**: GPU utilization and memory
- **Disk**: Free space in GB
- **Time**: Elapsed and remaining seconds

### Thresholds

From [config/agent_configs.yaml](./config/agent_configs.yaml):

```yaml
resource_thresholds:
  cpu: {warning: 90, critical: 95}
  memory: {warning: 80, critical: 85}
  gpu: {warning: 90, critical: 95}
  disk: {warning: 85, critical: 90}
  time: {warning_remaining: 1800, critical_remaining: 600}
```

### Critical Actions

When thresholds exceeded:
- **Memory Critical**: Clear caches, reduce batch size
- **Time Critical**: Force move to submission phase
- **Disk Low**: Clean temporary files
- **Behind Schedule**: Skip optional optimizations

## Implementation Guide

### For Developers

#### 1. Setting Up Agent

```python
# Load agent configuration
import yaml
with open('docs/config/agent_configs.yaml') as f:
    config = yaml.safe_load(f)

agent_config = config['agents']['A1_orchestrator']

# Load system prompt
with open(agent_config['system_prompt_path']) as f:
    system_prompt = f.read()

# Initialize agent
agent = Agent(
    agent_id="A1_orchestrator",
    model=agent_config['model'],
    temperature=agent_config['temperature'],
    max_tokens=agent_config['max_tokens'],
    system_prompt=system_prompt
)
```

#### 2. Sending Messages

```python
from message_templates import load_template

# Create request message
message = create_message_from_template(
    template_name="analyze_competition",
    competition_id="spaceship-titanic",
    description=competition_desc,
    data_path="/home/data/",
    time_budget=3600
)

# Send message
message_queue.send(message)

# Wait for response
response = message_queue.receive(
    agent_id="A1_orchestrator",
    correlation_id=message.message_id,
    timeout=3600
)
```

#### 3. Handling Responses

```python
if response.message_type == "RESPONSE":
    if response.payload["status"] == "success":
        # Process results
        analysis_result = response.payload
        update_state(analysis_result)
    else:
        # Handle error
        handle_agent_error(response)

elif response.message_type == "ERROR":
    # Activate fallback
    activate_fallback(response.payload)
```

#### 4. Running Orchestrator

```python
from orchestrator import Orchestrator

# Initialize
orchestrator = Orchestrator(
    config_path="docs/config/agent_configs.yaml"
)

# Run competition
result = orchestrator.run_competition(
    competition_id="spaceship-titanic",
    time_budget=86400  # 24 hours
)

# Check result
print(f"Submission saved to: {result.submission_path}")
print(f"CV Score: {result.best_cv_score}")
```

### For Researchers

#### Modifying Agent Behavior

1. **Edit System Prompt**: Modify prompt file in `docs/prompts/`
2. **Adjust Configuration**: Update `docs/config/agent_configs.yaml`
3. **Change Time Allocation**: Modify `workflow.time_allocations` in config
4. **Add New Agent**: Create prompt file and add to config

#### Experimenting with Strategies

1. **Try Different Ensemble Methods**: Edit A5 prompt or config
2. **Change HPO Strategy**: Modify P6 plug configuration
3. **Adjust Resource Thresholds**: Update `resource_thresholds` in config
4. **Add New Plugs**: Define in config with Pj = {Fj, Cj, Uj} format

## Testing

### Unit Testing Agents

```python
# Test individual agent
def test_analysis_agent():
    agent = load_agent("A2_analysis")
    context = load_test_context("spaceship-titanic")

    result = agent.execute(context)

    assert result["status"] == "success"
    assert "competition_type" in result
    assert "complexity_assessment" in result
```

### Integration Testing

```python
# Test agent communication
def test_agent_communication():
    orchestrator = load_agent("A1_orchestrator")
    analysis = load_agent("A2_analysis")

    # Send request
    request = orchestrator.create_request("analyze_competition", {...})
    response = analysis.process_request(request)

    # Verify response
    assert response.message_type == "RESPONSE"
    assert response.correlation_id == request.message_id
```

### End-to-End Testing

```python
# Test full workflow
def test_full_workflow():
    orchestrator = Orchestrator()

    result = orchestrator.run_competition(
        competition_id="test-competition",
        time_budget=3600  # 1 hour for testing
    )

    assert os.path.exists(result.submission_path)
    assert result.status == "success"
```

## Troubleshooting

### Common Issues

**Issue**: Agent timeout
- **Cause**: Time budget too small
- **Solution**: Increase timeout in message or phase allocation

**Issue**: Memory error during training
- **Cause**: Batch size too large
- **Solution**: A7 will alert and reduce batch size automatically

**Issue**: Validation fails
- **Cause**: Submission format mismatch
- **Solution**: A6 will auto-fix common issues, check logs

**Issue**: Ensemble doesn't improve
- **Cause**: Models too similar
- **Solution**: Train more diverse models in Phase 3b

### Debugging

Enable detailed logging:

```python
# In agent_configs.yaml
P12_logging:
  log_level: "DEBUG"
  monitoring_frequency: 10  # More frequent logging
```

Check logs:
```bash
tail -f /home/logs/system.log
tail -f /home/logs/resource_metrics.jsonl
```

## Performance Optimization

### Time Optimization

1. **Skip HPO for low complexity**: Set `enable_hpo: false`
2. **Reduce CV folds**: Use 3 instead of 5 for faster validation
3. **Use smaller models**: Select faster architectures in Phase 3a/b
4. **Parallel training**: Train multiple models simultaneously

### Resource Optimization

1. **GPU utilization**: Ensure models use GPU via `device: "cuda"`
2. **Memory efficiency**: Use `mixed_precision: true`
3. **Batch processing**: Optimize batch size based on available memory
4. **Disk cleanup**: A7 automatically removes old checkpoints

### Quality Optimization

1. **Longer training**: Allocate more time to modeling phase
2. **More HPO trials**: Increase `hpo_trials` to 100+
3. **Ensemble diversity**: Train models with different architectures
4. **Better features**: Allocate more time to preparation phase

## Success Metrics

Track system performance:

```python
metrics = {
    "completion_rate": 0.98,      # 98% valid submissions
    "medal_rate": 0.36,            # 36% medal rate
    "avg_cv_score": 0.842,         # Average CV score
    "avg_time_used": 0.82,         # 82% time utilization
    "avg_gpu_util": 0.73,          # 73% GPU utilization
    "failure_rate": 0.02           # 2% failures
}
```

## Additional Resources

- **Design Document**: [multi_agent_system_design.md](./multi_agent_system_design.md)
- **Architecture Diagrams**: [multi_agent_architecture_diagrams.md](./multi_agent_architecture_diagrams.md)
- **Orchestration Logic**: [workflow_orchestration.md](./workflow_orchestration.md)
- **Main README**: [README.md](./README.md)

## Version History

- **1.0** (2025-11-14): Initial workflow implementation
  - 10 agent prompts created
  - Orchestration workflows defined
  - Configuration files completed
  - Message templates specified

## Contributing

To contribute improvements:

1. Modify relevant prompt or config file
2. Test changes with unit/integration tests
3. Update documentation
4. Benchmark performance impact
5. Submit changes with detailed notes

## License

Part of the MLE-bench Multi-Agent System project.

---

**Questions?** Refer to the design document or orchestration workflow for detailed explanations.

**Ready to implement?** Start with the Orchestrator agent (A1) and work through the phases sequentially.
