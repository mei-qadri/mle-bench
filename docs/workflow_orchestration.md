# Multi-Agent System Workflow Orchestration

**Version:** 1.0
**Date:** 2025-11-14
**Status:** Implementation Specification

## Table of Contents
1. [Orchestration Overview](#orchestration-overview)
2. [Workflow State Machine](#workflow-state-machine)
3. [Agent Invocation Patterns](#agent-invocation-patterns)
4. [Phase-by-Phase Execution](#phase-by-phase-execution)
5. [Message Passing Protocol](#message-passing-protocol)
6. [Error Handling Workflows](#error-handling-workflows)
7. [Resource Monitoring Workflows](#resource-monitoring-workflows)

---

## Orchestration Overview

The orchestration system coordinates the execution of 7 core agents through a centralized state machine managed by the Orchestrator Agent (A1). The workflow is deterministic in phase progression but adaptive in agent selection and resource allocation.

### Core Orchestration Principles

1. **Sequential Phases**: Phases execute in strict order (Understanding → Preparation → Modeling → Ensembling → Submission)
2. **Parallel Agent Execution**: Within phases, compatible agents execute in parallel
3. **Dynamic Agent Selection**: Domain specialists selected based on competition type
4. **Continuous Monitoring**: Resource Manager (A7) and Validation Agent (A6) run continuously
5. **Adaptive Time Allocation**: Time budgets adjust based on progress and results

### Orchestration Loop

```python
# Pseudo-code for main orchestration loop
def orchestrate_competition(competition_id):
    # Initialize
    state = initialize_state(competition_id)
    activate_continuous_agents()  # A7: Resource Manager, A6: Validation

    # Phase 1: Understanding
    phase1_results = execute_understanding_phase(state)
    state.update(phase1_results)

    # Phase 2: Preparation
    phase2_results = execute_preparation_phase(state)
    state.update(phase2_results)

    # Phase 3: Modeling
    phase3_results = execute_modeling_phase(state)
    state.update(phase3_results)

    # Phase 4: Ensembling
    phase4_results = execute_ensembling_phase(state)
    state.update(phase4_results)

    # Phase 5: Submission
    phase5_results = execute_submission_phase(state)
    state.update(phase5_results)

    # Finalize
    deactivate_continuous_agents()
    return state.submission_path
```

---

## Workflow State Machine

### Global State Structure

```python
class GlobalState:
    """Centralized state managed by Orchestrator (A1)"""

    # Competition Information
    competition_id: str
    competition_type: str  # CV, NLP, tabular, audio, multi-modal
    complexity: str  # low, medium, high
    metric_name: str
    metric_direction: str  # minimize, maximize

    # Time Management
    start_time: float
    time_budget_total: int  # seconds
    time_elapsed: int
    time_remaining: int
    phase_time_allocations: dict  # phase_name: seconds

    # Resource Tracking
    resource_budget: dict  # cpu, gpu, memory limits
    resource_usage: dict  # current usage

    # Workflow State
    current_phase: str
    phase_status: dict  # phase_name: status (pending, in_progress, completed, failed)

    # Agent States
    active_agents: dict  # agent_id: agent_state
    agent_results: dict  # agent_id: results

    # Data & Artifacts
    data_summary: dict
    feature_pipeline: object
    cv_folds: object
    trained_models: list[dict]
    predictions: dict  # model_id: predictions
    ensemble_predictions: array

    # Performance Tracking
    best_cv_score: float
    leaderboard_target: dict  # gold/silver/bronze thresholds
    submission_history: list[dict]

    # Error Tracking
    errors: list[dict]
    warnings: list[dict]
    fallback_activated: bool
```

### Phase State Transitions

```
[INITIALIZED]
     ↓
[PHASE_1_UNDERSTANDING] → (Analysis Agent)
     ↓ (success)
[PHASE_2_PREPARATION] → (Engineering Agent)
     ↓ (success)
[PHASE_3_MODELING] → (Domain Specialists)
     ↓ (success)
[PHASE_4_ENSEMBLING] → (Ensemble Agent)
     ↓ (success)
[PHASE_5_SUBMISSION] → (Validation Agent)
     ↓ (success)
[COMPLETED]

On error at any phase → [FALLBACK_STRATEGY] → Continue or [FAILED]
```

---

## Agent Invocation Patterns

### Pattern 1: Sequential Agent Chain

Used when agents have strict dependencies.

```python
def sequential_agent_chain(agents: list[Agent], state: State):
    """Execute agents in sequence, each dependent on previous"""
    results = []
    for agent in agents:
        # Prepare agent context from current state
        context = prepare_agent_context(agent, state)

        # Invoke agent
        try:
            result = invoke_agent(agent, context)
            results.append(result)
            state.update(result)
        except Exception as e:
            handle_agent_error(agent, e, state)
            break

    return results
```

**Example**: A2 (Analysis) → A4 (Engineering) → A3 (Domain Specialist)

### Pattern 2: Parallel Agent Execution

Used when agents can work independently.

```python
def parallel_agent_execution(agents: list[Agent], state: State):
    """Execute multiple agents in parallel"""
    contexts = [prepare_agent_context(agent, state) for agent in agents]

    # Launch all agents in parallel
    futures = []
    for agent, context in zip(agents, contexts):
        future = async_invoke_agent(agent, context)
        futures.append(future)

    # Collect results
    results = []
    for agent, future in zip(agents, futures):
        try:
            result = await_agent_result(future, timeout=agent.timeout)
            results.append(result)
            state.update(result)
        except TimeoutError:
            handle_timeout(agent, state)
        except Exception as e:
            handle_agent_error(agent, e, state)

    return results
```

**Example**: Multiple domain specialists training different models simultaneously

### Pattern 3: Conditional Agent Selection

Used when agent selection depends on state.

```python
def conditional_agent_selection(state: State):
    """Select appropriate agent(s) based on state"""

    competition_type = state.competition_type

    # Select domain specialist(s)
    if competition_type == "CV":
        return [AgentRegistry.get("A3a_cv")]
    elif competition_type == "NLP":
        return [AgentRegistry.get("A3b_nlp")]
    elif competition_type == "tabular":
        return [AgentRegistry.get("A3c_tabular")]
    elif competition_type == "audio":
        return [AgentRegistry.get("A3d_audio")]
    elif competition_type == "multi-modal":
        # Multi-modal: invoke multiple specialists
        return [
            AgentRegistry.get("A3a_cv"),
            AgentRegistry.get("A3b_nlp")
        ]
    else:
        # Unknown: try tabular as fallback
        return [AgentRegistry.get("A3c_tabular")]
```

### Pattern 4: Agent Spawning

Used when agents create sub-agents dynamically.

```python
def agent_spawn_pattern(parent_agent: Agent, spawn_config: dict, state: State):
    """Parent agent spawns child agent"""

    # Create child agent
    child_agent = create_agent(
        agent_type=spawn_config['type'],
        config=spawn_config['config'],
        parent_id=parent_agent.id
    )

    # Register child
    state.active_agents[child_agent.id] = child_agent

    # Prepare context
    context = prepare_agent_context(child_agent, state)

    # Invoke child
    result = invoke_agent(child_agent, context)

    # Report back to parent
    parent_agent.receive_child_result(child_agent.id, result)

    # Cleanup
    deactivate_agent(child_agent)

    return result
```

**Example**: A3 (Domain Specialist) spawns HPO sub-agent for hyperparameter optimization

### Pattern 5: Continuous Monitoring

Used for agents that run throughout execution.

```python
def continuous_monitoring_pattern(agent: Agent, state: State, check_interval: int):
    """Agent monitors continuously in background"""

    while state.current_phase != "COMPLETED":
        # Perform monitoring check
        result = agent.check(state)

        # Update state if needed
        if result.requires_update:
            state.update(result)

        # Send alerts if needed
        if result.has_alerts:
            send_alerts(result.alerts, state)

        # Sleep until next check
        time.sleep(check_interval)
```

**Example**: A7 (Resource Manager) monitoring every 30 seconds

---

## Phase-by-Phase Execution

### Phase 1: Understanding (10% of time budget)

**Objective**: Analyze competition and generate recommendations

**Agents**: A1 (Orchestrator), A2 (Analysis)

**Workflow**:

```python
def execute_understanding_phase(state: State) -> dict:
    """Phase 1: Understanding"""

    # Update state
    state.current_phase = "PHASE_1_UNDERSTANDING"
    state.phase_status["understanding"] = "in_progress"
    phase_start_time = time.time()

    # Allocate time budget (10% of total)
    phase_time_budget = state.time_budget_total * 0.10

    # Step 1: Load competition description
    competition_desc = load_competition_description(state.competition_id)
    sample_submission = load_sample_submission(state.competition_id)

    # Step 2: Invoke Analysis Agent (A2)
    analysis_context = {
        "competition_id": state.competition_id,
        "description": competition_desc,
        "sample_submission": sample_submission,
        "time_budget": phase_time_budget,
        "data_path": "/home/data/"
    }

    try:
        analysis_result = invoke_agent(
            agent=AgentRegistry.get("A2_analysis"),
            context=analysis_context,
            timeout=phase_time_budget
        )

        # Step 3: Update state with analysis results
        state.competition_type = analysis_result.competition_type
        state.complexity = analysis_result.complexity
        state.metric_name = analysis_result.metric_name
        state.metric_direction = analysis_result.metric_direction
        state.data_summary = analysis_result.data_summary
        state.leaderboard_target = analysis_result.leaderboard_target

        # Step 4: Adjust time allocations based on complexity
        adjust_time_allocations(state, analysis_result.recommendations)

        # Mark phase complete
        state.phase_status["understanding"] = "completed"
        phase_elapsed = time.time() - phase_start_time

        return {
            "status": "success",
            "phase": "understanding",
            "time_elapsed": phase_elapsed,
            "analysis_result": analysis_result
        }

    except Exception as e:
        # Handle failure
        state.phase_status["understanding"] = "failed"
        state.errors.append({
            "phase": "understanding",
            "agent": "A2",
            "error": str(e)
        })

        # Activate fallback: use conservative defaults
        fallback_understanding(state)

        return {
            "status": "fallback",
            "phase": "understanding",
            "error": str(e)
        }
```

**Key Decisions**:
- Competition type classification (CV/NLP/tabular/audio/multi-modal)
- Complexity assessment (low/medium/high)
- Time budget adjustments
- Modeling recommendations

### Phase 2: Preparation (15% of time budget)

**Objective**: Prepare data and engineer features

**Agents**: A1 (Orchestrator), A4 (Engineering)

**Workflow**:

```python
def execute_preparation_phase(state: State) -> dict:
    """Phase 2: Preparation"""

    state.current_phase = "PHASE_2_PREPARATION"
    state.phase_status["preparation"] = "in_progress"
    phase_start_time = time.time()

    # Allocate time budget (15% of total)
    phase_time_budget = state.time_budget_total * 0.15

    # Prepare context for Engineering Agent
    engineering_context = {
        "competition_id": state.competition_id,
        "competition_type": state.competition_type,
        "data_summary": state.data_summary,
        "metric_name": state.metric_name,
        "time_budget": phase_time_budget,
        "recommendations": state.agent_results.get("A2", {}).get("recommendations", [])
    }

    try:
        engineering_result = invoke_agent(
            agent=AgentRegistry.get("A4_engineering"),
            context=engineering_context,
            timeout=phase_time_budget
        )

        # Update state
        state.feature_pipeline = engineering_result.preprocessing_pipeline
        state.cv_folds = engineering_result.cv_folds
        state.data_summary.update(engineering_result.feature_summary)

        state.phase_status["preparation"] = "completed"
        phase_elapsed = time.time() - phase_start_time

        return {
            "status": "success",
            "phase": "preparation",
            "time_elapsed": phase_elapsed,
            "engineering_result": engineering_result
        }

    except Exception as e:
        state.phase_status["preparation"] = "failed"
        state.errors.append({
            "phase": "preparation",
            "agent": "A4",
            "error": str(e)
        })

        # Fallback: minimal preprocessing
        fallback_preparation(state)

        return {
            "status": "fallback",
            "phase": "preparation",
            "error": str(e)
        }
```

**Key Outputs**:
- Preprocessing pipeline
- Engineered features
- CV fold splits
- Data ready for training

### Phase 3: Modeling (60% of time budget)

**Objective**: Train models and generate predictions

**Agents**: A1 (Orchestrator), A3 (Domain Specialists), A7 (Resource Manager)

**Workflow**:

```python
def execute_modeling_phase(state: State) -> dict:
    """Phase 3: Modeling"""

    state.current_phase = "PHASE_3_MODELING"
    state.phase_status["modeling"] = "in_progress"
    phase_start_time = time.time()

    # Allocate time budget (60% of total)
    phase_time_budget = state.time_budget_total * 0.60

    # Sub-phase allocation
    baseline_time = phase_time_budget * 0.15
    advanced_time = phase_time_budget * 0.70
    final_time = phase_time_budget * 0.15

    # Step 1: Select domain specialist(s)
    specialists = conditional_agent_selection(state)

    # Step 2: Sub-phase 3a - Quick Baseline
    baseline_results = execute_baseline_modeling(
        specialists=specialists,
        state=state,
        time_budget=baseline_time
    )
    state.trained_models.extend(baseline_results.models)
    state.best_cv_score = baseline_results.best_cv_score

    # Step 3: Sub-phase 3b - Advanced Modeling
    advanced_results = execute_advanced_modeling(
        specialists=specialists,
        state=state,
        time_budget=advanced_time
    )
    state.trained_models.extend(advanced_results.models)
    if advanced_results.best_cv_score > state.best_cv_score:
        state.best_cv_score = advanced_results.best_cv_score

    # Step 4: Sub-phase 3c - Final Models
    final_results = execute_final_modeling(
        specialists=specialists,
        state=state,
        time_budget=final_time
    )
    state.trained_models.extend(final_results.models)

    # Step 5: Collect all predictions
    for model in state.trained_models:
        state.predictions[model.id] = model.test_predictions

    state.phase_status["modeling"] = "completed"
    phase_elapsed = time.time() - phase_start_time

    return {
        "status": "success",
        "phase": "modeling",
        "time_elapsed": phase_elapsed,
        "num_models": len(state.trained_models),
        "best_cv_score": state.best_cv_score
    }
```

**Sub-phase 3a: Baseline Modeling**

```python
def execute_baseline_modeling(specialists: list[Agent], state: State, time_budget: float):
    """Quick baseline models for validation"""

    # Configure for speed
    baseline_config = {
        "max_models": 2,
        "use_simple_models": True,
        "skip_hpo": True,
        "reduced_training_epochs": True
    }

    results = []
    for specialist in specialists:
        context = prepare_modeling_context(specialist, state, baseline_config, time_budget)
        result = invoke_agent(specialist, context, timeout=time_budget)
        results.append(result)

    return aggregate_modeling_results(results)
```

**Sub-phase 3b: Advanced Modeling**

```python
def execute_advanced_modeling(specialists: list[Agent], state: State, time_budget: float):
    """Train sophisticated models with HPO"""

    # Check if time permits HPO
    time_per_specialist = time_budget / len(specialists)
    enable_hpo = time_per_specialist > 3600  # At least 1 hour per specialist

    advanced_config = {
        "max_models": 5,
        "use_advanced_models": True,
        "enable_hpo": enable_hpo,
        "hpo_trials": 50 if enable_hpo else 0,
        "data_augmentation": state.competition_type in ["CV", "audio"],
        "transfer_learning": state.competition_type in ["CV", "NLP"],
        "ensemble_diversity": True  # Train diverse models for ensembling
    }

    # Execute specialists in parallel if multiple
    if len(specialists) > 1:
        results = parallel_agent_execution(specialists, state)
    else:
        # Single specialist: can train multiple models sequentially
        specialist = specialists[0]
        context = prepare_modeling_context(specialist, state, advanced_config, time_budget)
        result = invoke_agent(specialist, context, timeout=time_budget)
        results = [result]

    return aggregate_modeling_results(results)
```

**Sub-phase 3c: Final Modeling**

```python
def execute_final_modeling(specialists: list[Agent], state: State, time_budget: float):
    """Retrain best models on full train set"""

    # Select top models from previous phases
    top_models = select_top_models(state.trained_models, top_k=3)

    final_config = {
        "retrain_models": [m.config for m in top_models],
        "use_full_train_set": True,  # No validation split
        "generate_test_predictions": True,
        "save_checkpoints": True
    }

    results = []
    for specialist in specialists:
        context = prepare_modeling_context(specialist, state, final_config, time_budget)
        result = invoke_agent(specialist, context, timeout=time_budget)
        results.append(result)

    return aggregate_modeling_results(results)
```

**Key Outputs**:
- Trained models (checkpoints)
- Cross-validation scores
- Test predictions from each model

### Phase 4: Ensembling (10% of time budget)

**Objective**: Combine models and optimize ensemble

**Agents**: A1 (Orchestrator), A5 (Ensemble)

**Workflow**:

```python
def execute_ensembling_phase(state: State) -> dict:
    """Phase 4: Ensembling"""

    state.current_phase = "PHASE_4_ENSEMBLING"
    state.phase_status["ensembling"] = "in_progress"
    phase_start_time = time.time()

    # Allocate time budget (10% of total)
    phase_time_budget = state.time_budget_total * 0.10

    # Check if we have multiple models
    if len(state.trained_models) < 2:
        # Only one model: skip ensembling
        best_model = state.trained_models[0]
        state.ensemble_predictions = state.predictions[best_model.id]

        return {
            "status": "skipped",
            "phase": "ensembling",
            "reason": "Only one model trained",
            "using_model": best_model.id
        }

    # Prepare ensemble context
    ensemble_context = {
        "models": state.trained_models,
        "predictions": state.predictions,
        "cv_scores": {m.id: m.cv_score for m in state.trained_models},
        "metric_name": state.metric_name,
        "metric_direction": state.metric_direction,
        "time_budget": phase_time_budget
    }

    try:
        ensemble_result = invoke_agent(
            agent=AgentRegistry.get("A5_ensemble"),
            context=ensemble_context,
            timeout=phase_time_budget
        )

        # Check if ensemble improves over best model
        best_single_score = max(m.cv_score for m in state.trained_models)

        if ensemble_result.ensemble_cv_score > best_single_score:
            # Use ensemble
            state.ensemble_predictions = ensemble_result.predictions
            state.best_cv_score = ensemble_result.ensemble_cv_score
            ensemble_used = True
        else:
            # Ensemble didn't improve: use best single model
            best_model = max(state.trained_models, key=lambda m: m.cv_score)
            state.ensemble_predictions = state.predictions[best_model.id]
            ensemble_used = False

        state.phase_status["ensembling"] = "completed"
        phase_elapsed = time.time() - phase_start_time

        return {
            "status": "success",
            "phase": "ensembling",
            "time_elapsed": phase_elapsed,
            "ensemble_used": ensemble_used,
            "ensemble_cv_score": ensemble_result.ensemble_cv_score if ensemble_used else None,
            "best_cv_score": state.best_cv_score
        }

    except Exception as e:
        state.phase_status["ensembling"] = "failed"
        state.errors.append({
            "phase": "ensembling",
            "agent": "A5",
            "error": str(e)
        })

        # Fallback: use best single model
        best_model = max(state.trained_models, key=lambda m: m.cv_score)
        state.ensemble_predictions = state.predictions[best_model.id]

        return {
            "status": "fallback",
            "phase": "ensembling",
            "error": str(e),
            "using_best_model": best_model.id
        }
```

**Key Outputs**:
- Final predictions (ensemble or best single model)
- Ensemble strategy and weights (if used)

### Phase 5: Submission (5% of time budget)

**Objective**: Validate and submit predictions

**Agents**: A1 (Orchestrator), A6 (Validation)

**Workflow**:

```python
def execute_submission_phase(state: State) -> dict:
    """Phase 5: Submission"""

    state.current_phase = "PHASE_5_SUBMISSION"
    state.phase_status["submission"] = "in_progress"
    phase_start_time = time.time()

    # Allocate time budget (5% of total)
    phase_time_budget = state.time_budget_total * 0.05

    # Prepare validation context
    validation_context = {
        "competition_id": state.competition_id,
        "predictions": state.ensemble_predictions,
        "sample_submission_path": "/home/data/sample_submission.csv",
        "output_path": "/home/submission/submission.csv",
        "metric_name": state.metric_name,
        "cv_score": state.best_cv_score,
        "time_budget": phase_time_budget
    }

    max_attempts = 3
    attempt = 0

    while attempt < max_attempts:
        try:
            validation_result = invoke_agent(
                agent=AgentRegistry.get("A6_validation"),
                context=validation_context,
                timeout=phase_time_budget
            )

            if validation_result.is_valid:
                # Submission is valid
                state.phase_status["submission"] = "completed"
                state.submission_path = validation_result.submission_path

                phase_elapsed = time.time() - phase_start_time

                return {
                    "status": "success",
                    "phase": "submission",
                    "time_elapsed": phase_elapsed,
                    "submission_path": validation_result.submission_path,
                    "validation_result": validation_result
                }
            else:
                # Validation failed: attempt to fix
                if attempt < max_attempts - 1:
                    # Try to fix issues
                    validation_context["predictions"] = fix_validation_issues(
                        state.ensemble_predictions,
                        validation_result.errors
                    )
                    attempt += 1
                else:
                    # Max attempts reached: save best effort
                    save_best_effort_submission(state)

                    return {
                        "status": "warning",
                        "phase": "submission",
                        "message": "Validation issues detected, saved best effort",
                        "errors": validation_result.errors
                    }

        except Exception as e:
            attempt += 1
            if attempt >= max_attempts:
                state.phase_status["submission"] = "failed"
                state.errors.append({
                    "phase": "submission",
                    "agent": "A6",
                    "error": str(e)
                })

                # Emergency fallback: save raw predictions
                emergency_submission(state)

                return {
                    "status": "error",
                    "phase": "submission",
                    "error": str(e),
                    "fallback_submission": True
                }
```

**Key Outputs**:
- Valid submission file at `/home/submission/submission.csv`
- Validation report

---

## Message Passing Protocol

### Message Structure

```python
class Message:
    message_id: str  # UUID
    timestamp: float
    sender_id: str  # Agent ID
    receiver_id: str  # Agent ID or "broadcast"
    message_type: str  # REQUEST, RESPONSE, UPDATE, ERROR, ALERT
    priority: int  # 0 (low) to 10 (critical)
    correlation_id: str  # For request-response pairing
    requires_response: bool
    timeout: float  # seconds
    payload: dict  # Message content
```

### Message Types and Workflows

**REQUEST Message**

```python
# Orchestrator requesting Analysis Agent to analyze competition
request = Message(
    message_id=generate_uuid(),
    timestamp=time.time(),
    sender_id="A1_orchestrator",
    receiver_id="A2_analysis",
    message_type="REQUEST",
    priority=5,
    correlation_id=generate_uuid(),
    requires_response=True,
    timeout=3600,
    payload={
        "action": "analyze_competition",
        "competition_id": "spaceship-titanic",
        "data_path": "/home/data/",
        "time_budget": 3600
    }
)

# Send message
message_queue.send(request)
```

**RESPONSE Message**

```python
# Analysis Agent responding to Orchestrator
response = Message(
    message_id=generate_uuid(),
    timestamp=time.time(),
    sender_id="A2_analysis",
    receiver_id="A1_orchestrator",
    message_type="RESPONSE",
    priority=5,
    correlation_id=request.correlation_id,  # Same as request
    requires_response=False,
    payload={
        "status": "success",
        "competition_type": "tabular",
        "complexity": "low",
        "metric_name": "accuracy",
        "metric_direction": "maximize",
        "data_summary": {...},
        "recommendations": [...]
    }
)

message_queue.send(response)
```

**UPDATE Message**

```python
# Resource Manager broadcasting resource alert
update = Message(
    message_id=generate_uuid(),
    timestamp=time.time(),
    sender_id="A7_resource_manager",
    receiver_id="broadcast",
    message_type="ALERT",
    priority=8,
    requires_response=False,
    payload={
        "alert_type": "high_memory_usage",
        "memory_usage_percent": 85,
        "recommendation": "reduce_batch_size",
        "affected_agents": ["A3c_tabular"]
    }
)

message_queue.send(update)
```

### Message Routing

```python
class MessageQueue:
    def __init__(self):
        self.queues = {}  # agent_id: deque
        self.broadcast_subscribers = set()

    def send(self, message: Message):
        """Send message to recipient(s)"""
        if message.receiver_id == "broadcast":
            # Broadcast to all subscribers
            for agent_id in self.broadcast_subscribers:
                self.queues[agent_id].append(message)
        else:
            # Send to specific recipient
            self.queues[message.receiver_id].append(message)

        # Log message
        log_message(message)

    def receive(self, agent_id: str, timeout: float = None) -> Message:
        """Receive next message for agent"""
        queue = self.queues[agent_id]

        if timeout:
            start_time = time.time()
            while len(queue) == 0:
                if time.time() - start_time > timeout:
                    raise TimeoutError(f"No message received within {timeout}s")
                time.sleep(0.1)

        return queue.popleft() if queue else None
```

---

## Error Handling Workflows

### Error Classification

```python
class ErrorType(Enum):
    TRANSIENT = "transient"  # Temporary, retry possible
    RESOURCE = "resource"  # Resource exhaustion
    DATA = "data"  # Data quality issue
    CONFIGURATION = "configuration"  # Invalid config
    FATAL = "fatal"  # Unrecoverable
```

### Error Handling Workflow

```python
def handle_agent_error(agent: Agent, error: Exception, state: State):
    """Centralized error handling"""

    # Classify error
    error_type = classify_error(error)

    # Log error
    state.errors.append({
        "agent_id": agent.id,
        "error_type": error_type,
        "error_message": str(error),
        "timestamp": time.time(),
        "phase": state.current_phase
    })

    # Handle based on type
    if error_type == ErrorType.TRANSIENT:
        # Retry with exponential backoff
        return retry_agent_with_backoff(agent, state)

    elif error_type == ErrorType.RESOURCE:
        # Degrade configuration and retry
        degraded_config = degrade_resource_usage(agent.config)
        return retry_agent_with_config(agent, degraded_config, state)

    elif error_type == ErrorType.DATA:
        # Skip problematic data and continue
        return skip_and_continue(agent, state)

    elif error_type == ErrorType.CONFIGURATION:
        # Use default configuration
        default_config = get_default_config(agent.agent_type)
        return retry_agent_with_config(agent, default_config, state)

    elif error_type == ErrorType.FATAL:
        # Activate fallback strategy
        return activate_fallback(agent, state)
```

### Fallback Strategies

```python
# Fallback for Phase 1 (Understanding)
def fallback_understanding(state: State):
    # Use conservative defaults
    state.competition_type = "tabular"  # Safest default
    state.complexity = "medium"
    state.metric_name = "accuracy"  # Generic
    state.metric_direction = "maximize"

    # Minimal data summary
    state.data_summary = {
        "rows": "unknown",
        "columns": "unknown",
        "target": "unknown"
    }

# Fallback for Phase 2 (Preparation)
def fallback_preparation(state: State):
    # Minimal preprocessing: just load data as-is
    state.feature_pipeline = IdentityPipeline()
    state.cv_folds = create_simple_cv_folds(n_splits=5)

# Fallback for Phase 3 (Modeling)
def fallback_modeling(state: State):
    # Train simplest possible model
    if state.competition_type == "tabular":
        model = LogisticRegression()  # or RandomForest
    elif state.competition_type == "CV":
        model = SimpleResNet()
    # ... train and save

# Fallback for Phase 4 (Ensembling)
def fallback_ensembling(state: State):
    # Use best single model
    best_model = max(state.trained_models, key=lambda m: m.cv_score)
    state.ensemble_predictions = state.predictions[best_model.id]

# Fallback for Phase 5 (Submission)
def emergency_submission(state: State):
    # Save whatever predictions we have
    predictions = state.ensemble_predictions
    save_predictions_csv(predictions, "/home/submission/submission.csv")
```

---

## Resource Monitoring Workflows

### Resource Manager Continuous Loop

```python
def resource_manager_loop(state: State):
    """A7: Resource Manager continuous monitoring"""

    check_interval = 30  # seconds

    while state.current_phase != "COMPLETED":
        # Collect resource metrics
        metrics = {
            "cpu_usage": get_cpu_usage(),
            "gpu_usage": get_gpu_usage(),
            "memory_usage": get_memory_usage(),
            "disk_usage": get_disk_usage(),
            "time_elapsed": time.time() - state.start_time,
            "time_remaining": state.time_budget_total - (time.time() - state.start_time)
        }

        # Update state
        state.resource_usage.update(metrics)

        # Check thresholds
        alerts = check_resource_thresholds(metrics, state)

        # Send alerts if needed
        for alert in alerts:
            send_alert(alert, state)

        # Take action if critical
        if any(alert.priority >= 9 for alert in alerts):
            take_critical_action(alerts, state)

        # Sleep until next check
        time.sleep(check_interval)

def check_resource_thresholds(metrics: dict, state: State) -> list[Alert]:
    """Check if any resource thresholds exceeded"""
    alerts = []

    # CPU threshold
    if metrics["cpu_usage"] > 95:
        alerts.append(Alert(
            type="high_cpu_usage",
            priority=8,
            message=f"CPU usage at {metrics['cpu_usage']}%",
            recommendation="reduce_parallel_processes"
        ))

    # GPU threshold
    if metrics["gpu_usage"] > 95:
        alerts.append(Alert(
            type="high_gpu_usage",
            priority=8,
            message=f"GPU usage at {metrics['gpu_usage']}%",
            recommendation="reduce_batch_size"
        ))

    # Memory threshold
    if metrics["memory_usage"] > 85:
        alerts.append(Alert(
            type="high_memory_usage",
            priority=9,  # Critical
            message=f"Memory usage at {metrics['memory_usage']}%",
            recommendation="clear_cache_and_reduce_batch_size"
        ))

    # Time threshold
    time_usage_percent = (metrics["time_elapsed"] / state.time_budget_total) * 100
    phase_progress = get_phase_progress(state.current_phase, state)

    if time_usage_percent > phase_progress + 20:  # 20% behind schedule
        alerts.append(Alert(
            type="behind_schedule",
            priority=7,
            message=f"Time usage {time_usage_percent:.1f}% but phase progress {phase_progress:.1f}%",
            recommendation="skip_optional_optimizations"
        ))

    # Time running out
    if metrics["time_remaining"] < 600:  # Less than 10 minutes
        alerts.append(Alert(
            type="time_critical",
            priority=10,  # Maximum priority
            message=f"Only {metrics['time_remaining']}s remaining",
            recommendation="finalize_submission_immediately"
        ))

    return alerts

def take_critical_action(alerts: list[Alert], state: State):
    """Take immediate action for critical alerts"""
    for alert in alerts:
        if alert.type == "high_memory_usage":
            # Clear caches
            clear_model_caches()
            torch.cuda.empty_cache()

        elif alert.type == "time_critical":
            # Force move to submission phase
            force_phase_transition(state, "PHASE_5_SUBMISSION")
```

---

## Orchestration Configuration

### Time Budget Allocation

```yaml
time_allocations:
  default:
    understanding: 0.10  # 10%
    preparation: 0.15    # 15%
    modeling: 0.60       # 60%
    ensembling: 0.10     # 10%
    submission: 0.05     # 5%

  # Adjust based on complexity
  complexity_adjustments:
    low:
      understanding: 0.08
      preparation: 0.12
      modeling: 0.65
      ensembling: 0.10
      submission: 0.05

    medium:
      understanding: 0.10
      preparation: 0.15
      modeling: 0.60
      ensembling: 0.10
      submission: 0.05

    high:
      understanding: 0.12
      preparation: 0.18
      modeling: 0.55
      ensembling: 0.10
      submission: 0.05
```

### Resource Thresholds

```yaml
resource_thresholds:
  cpu:
    warning: 90
    critical: 95

  gpu:
    warning: 90
    critical: 95

  memory:
    warning: 80
    critical: 85

  disk:
    warning: 85
    critical: 90

  time:
    warning_remaining: 1800  # 30 minutes
    critical_remaining: 600  # 10 minutes
```

---

**End of Orchestration Workflow Document**
