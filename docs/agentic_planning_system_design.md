# Agentic Planning System Design

**Project**: MLE-Bench Multi-Agent Planning System
**Version**: 1.0
**Date**: 2025-11-14

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Research Insights from Existing Scaffolds](#research-insights)
3. [System Architecture](#system-architecture)
4. [Agent Specification](#agent-specification)
5. [Plugin Specification](#plugin-specification)
6. [Planning Module Design](#planning-module-design)
7. [Workflow Execution](#workflow-execution)
8. [Implementation Plan](#implementation-plan)
9. [Examples](#examples)

---

## 1. Executive Summary

This document outlines the design for a **meta-agent planning system** that can analyze ML engineering problems (like those in MLE-Bench) and dynamically compose specialized agent workflows to solve them.

### Key Innovation

Rather than creating a single monolithic agent, the system acts as an **orchestrator** that:
1. **Understands** the problem domain and requirements
2. **Plans** the optimal workflow and agent composition
3. **Collaborates** with humans to validate the approach
4. **Instantiates** specialized agents with appropriate tools
5. **Coordinates** multi-agent collaboration toward solution

### Design Philosophy

**Inspired by:**
- **AIDE**: Dual-model architecture (separate planning and execution models)
- **MLAgentBench**: Structured tool/action spaces with clear semantics
- **OpenDevin**: Code-as-action unified interface

**Core Principles:**
- **Composability**: Agents and plugins are building blocks
- **Specialization**: Each agent has a focused role and capability set
- **Adaptability**: System adapts to problem characteristics
- **Human-in-the-loop**: Critical decisions validated by humans
- **Observability**: Full visibility into agent reasoning and actions

---

## 2. Research Insights from Existing Scaffolds

### AIDE (Agentic IDE)

**Strengths:**
- Dual-model architecture (code generation + feedback/critique)
- Search-based debugging (20-step depth exploration)
- Iterative refinement with error recovery
- Multi-LLM support

**Limitations:**
- Monolithic agent structure
- Fixed action space
- Limited role specialization

**Key Takeaway**: Separation of concerns (planning vs execution) improves performance.

---

### MLAgentBench

**Strengths:**
- ReAct framework (Reasoning + Acting)
- Well-defined tool/action space
- Strong observability (log files, execution history)
- Realistic environment constraints

**Limitations:**
- Single-agent architecture
- Limited parallel execution
- No dynamic tool composition

**Key Takeaway**: Structured action spaces with clear semantics enable better reasoning.

---

### OpenDevin (OpenHands)

**Strengths:**
- CodeAct framework (code as unified action)
- Tight feedback loops
- Jupyter notebook output (excellent observability)
- Event-driven architecture

**Limitations:**
- Code-centric (may not suit all problems)
- Limited role specialization
- Single-agent workflow

**Key Takeaway**: Unified action spaces reduce complexity; event streams enable coordination.

---

## 3. System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Human User (Problem Input)                 │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PLANNING MODULE (Orchestrator)                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 1. Problem Understanding Agent                           │  │
│  │    - Parse task description                              │  │
│  │    - Analyze data characteristics                        │  │
│  │    - Identify success criteria                           │  │
│  └──────────────────────────────────────────────────────────┘  │
│                               ↓                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 2. Workflow Planning Agent                               │  │
│  │    - Decompose problem into subtasks                     │  │
│  │    - Design agent composition                            │  │
│  │    - Select plugins and tools                            │  │
│  │    - Estimate resource requirements                      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                               ↓                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 3. Human Collaboration Interface                         │  │
│  │    - Present workflow plan                               │  │
│  │    - Gather feedback and approval                        │  │
│  │    - Iterate on plan if needed                           │  │
│  └──────────────────────────────────────────────────────────┘  │
│                               ↓                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 4. Agent & Plugin Instantiation Engine                   │  │
│  │    - Generate agent configurations                       │  │
│  │    - Create plugin specifications                        │  │
│  │    - Set up execution environments                       │  │
│  └──────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│                      EXECUTION RUNTIME                          │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐   │
│  │  Agent A₁      │  │  Agent A₂      │  │  Agent A₃      │   │
│  │  (Explorer)    │  │  (Modeler)     │  │  (Validator)   │   │
│  │                │  │                │  │                │   │
│  │  Plugins:      │  │  Plugins:      │  │  Plugins:      │   │
│  │  - FileReader  │  │  - MLTrainer   │  │  - Grader      │   │
│  │  - DataExplore │  │  - HPTuner     │  │  - Submission  │   │
│  └────────────────┘  └────────────────┘  └────────────────┘   │
│           ↓                   ↓                   ↓             │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │               Shared State & Event Bus                   │  │
│  │  - Agent communication                                   │  │
│  │  - Resource coordination                                 │  │
│  │  - Progress tracking                                     │  │
│  └──────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│                    OUTPUT & EVALUATION                          │
│  - Final submission.csv                                         │
│  - Execution logs and metrics                                   │
│  - Agent interaction history                                    │
│  - Performance analytics                                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Agent Specification

### Formal Definition

An agent is defined as:

```
Ai = {Li, Ri, Si, Ci, Hi}
```

Where:
- **Li**: Language Model and Configuration
- **Ri**: Role specification
- **Si**: Internal state
- **Ci**: Boolean - can spawn child agents
- **Hi**: Execution history

### 4.1 Language Model Configuration (Li)

```python
@dataclass
class LLMConfig:
    """Language model configuration for an agent"""

    # Model selection
    model_name: str                    # e.g., "gpt-4o", "claude-3.5-sonnet"
    provider: str                      # e.g., "openai", "anthropic"

    # Generation parameters
    temperature: float = 0.7           # Sampling temperature
    max_tokens: int = 16384            # Max output tokens
    top_p: float = 1.0                 # Nucleus sampling

    # Request management
    timeout: int = 60                  # API timeout (seconds)
    max_retries: int = 3               # Retry on failures

    # Cost optimization
    use_cache: bool = True             # Enable prompt caching

    # Special configurations
    system_prompt_template: str = ""   # Custom system prompt
    few_shot_examples: List[Dict] = field(default_factory=list)
```

**Example Configurations:**

```python
# High-reasoning agent (for planning)
planner_llm = LLMConfig(
    model_name="o1-preview",
    provider="openai",
    temperature=1.0,
    max_tokens=32768
)

# Fast execution agent (for simple actions)
executor_llm = LLMConfig(
    model_name="gpt-4o-mini",
    provider="openai",
    temperature=0.3,
    max_tokens=4096
)

# Code generation specialist
coder_llm = LLMConfig(
    model_name="claude-3.5-sonnet",
    provider="anthropic",
    temperature=0.5,
    max_tokens=8192
)
```

---

### 4.2 Role Specification (Ri)

```python
@dataclass
class AgentRole:
    """Defines the role and responsibilities of an agent"""

    # Identity
    role_name: str                     # e.g., "DataExplorer", "ModelTrainer"
    description: str                   # Human-readable role description

    # Capabilities
    allowed_actions: List[str]         # Action types agent can perform
    available_plugins: List[str]       # Plugin IDs accessible to agent

    # Constraints
    max_steps: int = 100               # Maximum action steps
    max_time_seconds: int = 3600       # Time budget (1 hour default)
    resource_limits: Dict[str, Any] = field(default_factory=dict)

    # Communication
    can_delegate: bool = True          # Can request help from other agents
    can_broadcast: bool = False        # Can send messages to all agents

    # Success criteria
    success_condition: str = ""        # Description of when role is complete
    output_schema: Dict = field(default_factory=dict)  # Expected outputs
```

**Example Roles:**

```python
# Data exploration specialist
explorer_role = AgentRole(
    role_name="DataExplorer",
    description="Analyzes datasets to understand structure, distributions, and quality",
    allowed_actions=["read_file", "execute_python", "visualize_data"],
    available_plugins=["pandas_analyzer", "data_profiler"],
    max_steps=50,
    max_time_seconds=1800,
    success_condition="Generate data exploration report with key insights",
    output_schema={"type": "dict", "required": ["summary", "insights", "issues"]}
)

# Model training specialist
trainer_role = AgentRole(
    role_name="ModelTrainer",
    description="Trains and evaluates ML models",
    allowed_actions=["execute_python", "train_model", "evaluate_model"],
    available_plugins=["sklearn_trainer", "pytorch_trainer", "hyperparameter_tuner"],
    max_steps=200,
    max_time_seconds=18000,  # 5 hours
    can_delegate=True,
    success_condition="Produce validated model with performance metrics",
    output_schema={"type": "dict", "required": ["model_path", "metrics", "predictions"]}
)

# Validation and submission specialist
validator_role = AgentRole(
    role_name="SubmissionValidator",
    description="Validates submissions and ensures format compliance",
    allowed_actions=["read_file", "execute_bash", "validate_submission"],
    available_plugins=["format_checker", "grading_client"],
    max_steps=20,
    max_time_seconds=600,
    can_delegate=False,
    success_condition="Valid submission.csv that passes all checks",
    output_schema={"type": "dict", "required": ["valid", "submission_path"]}
)
```

---

### 4.3 Agent State (Si)

```python
@dataclass
class AgentState:
    """Tracks the internal state of an agent during execution"""

    # Execution state
    status: str = "idle"               # idle, planning, acting, waiting, completed, failed
    current_step: int = 0              # Current step number

    # Memory
    working_memory: Dict[str, Any] = field(default_factory=dict)
    observations: List[Dict] = field(default_factory=list)

    # Planning
    plan: List[Dict] = field(default_factory=list)  # Planned actions
    current_goal: str = ""             # Current objective

    # Results
    outputs: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)  # File paths

    # Metrics
    actions_taken: int = 0
    errors_encountered: int = 0
    time_elapsed: float = 0.0

    # Inter-agent coordination
    messages_received: List[Dict] = field(default_factory=list)
    messages_sent: List[Dict] = field(default_factory=list)
```

---

### 4.4 Spawn Capability (Ci)

```python
@dataclass
class SpawnConfig:
    """Configuration for agent spawning capabilities"""

    can_spawn: bool = False            # Whether agent can create children
    max_children: int = 0              # Maximum child agents
    allowed_child_roles: List[str] = field(default_factory=list)
    spawn_strategy: str = "sequential" # sequential, parallel, adaptive

    # Resource allocation to children
    time_budget_per_child: int = 1800
    step_budget_per_child: int = 50
```

**Example:**

```python
# Supervisor agent that can spawn workers
supervisor_spawn = SpawnConfig(
    can_spawn=True,
    max_children=3,
    allowed_child_roles=["DataCleaner", "FeatureEngineer", "ModelTuner"],
    spawn_strategy="parallel",
    time_budget_per_child=3600
)

# Worker agent that cannot spawn
worker_spawn = SpawnConfig(
    can_spawn=False,
    max_children=0
)
```

---

### 4.5 Execution History (Hi)

```python
@dataclass
class ExecutionHistory:
    """Records the complete execution trace of an agent"""

    # Timestamped events
    events: List[Dict] = field(default_factory=list)

    # Event types tracked:
    # - "thought": Agent reasoning/planning
    # - "action": Actions taken (with parameters)
    # - "observation": Results from actions
    # - "message": Inter-agent communication
    # - "error": Errors encountered
    # - "milestone": Key achievements

    # Structured logs
    start_time: datetime = None
    end_time: datetime = None

    # Performance tracking
    tokens_used: int = 0
    api_calls: int = 0
    cost_usd: float = 0.0

    def add_event(self, event_type: str, content: Dict):
        """Add a timestamped event to history"""
        self.events.append({
            "timestamp": datetime.now(),
            "type": event_type,
            "content": content
        })

    def to_notebook(self) -> str:
        """Export history as Jupyter notebook"""
        # Similar to OpenDevin's approach
        pass

    def to_markdown(self) -> str:
        """Export history as readable markdown"""
        pass
```

---

### 4.6 Complete Agent Class

```python
@dataclass
class Agent:
    """Complete agent specification: Ai = {Li, Ri, Si, Ci, Hi}"""

    # Core components
    agent_id: str                      # Unique identifier
    llm_config: LLMConfig              # Li: Language model configuration
    role: AgentRole                    # Ri: Role specification
    state: AgentState                  # Si: Internal state
    spawn_config: SpawnConfig          # Ci: Spawn capability
    history: ExecutionHistory          # Hi: Execution history

    # Runtime
    plugins: Dict[str, 'Plugin'] = field(default_factory=dict)
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)

    # Environment
    workspace_dir: Path = None
    data_dir: Path = None

    def act(self, observation: Dict) -> Dict:
        """Main agent loop: observe → reason → act"""
        # 1. Update state with observation
        self.state.observations.append(observation)

        # 2. Generate next action using LLM
        action = self._generate_action()

        # 3. Execute action via plugins
        result = self._execute_action(action)

        # 4. Record in history
        self.history.add_event("action", action)
        self.history.add_event("observation", result)

        return result

    def spawn_child(self, child_role: AgentRole) -> 'Agent':
        """Spawn a child agent with specified role"""
        if not self.spawn_config.can_spawn:
            raise ValueError(f"Agent {self.agent_id} cannot spawn children")

        # Create child agent with inherited configuration
        child = Agent(
            agent_id=f"{self.agent_id}_child_{len(self.children_ids)}",
            llm_config=self.llm_config,
            role=child_role,
            state=AgentState(),
            spawn_config=SpawnConfig(can_spawn=False),
            history=ExecutionHistory(),
            parent_id=self.agent_id
        )

        self.children_ids.append(child.agent_id)
        return child
```

---

## 5. Plugin Specification

### Formal Definition

A plugin is defined as:

```
Pj = {Fj, Cj, Uj}
```

Where:
- **Fj**: Functionality (actions the plugin provides)
- **Cj**: Configuration parameters
- **Uj**: Usage constraints

---

### 5.1 Plugin Functionality (Fj)

```python
@dataclass
class PluginFunctionality:
    """Defines the actions/capabilities provided by a plugin"""

    # Identity
    plugin_id: str                     # Unique identifier
    plugin_name: str                   # Human-readable name
    description: str                   # What the plugin does

    # Actions
    actions: Dict[str, 'Action']       # Available actions

    # Dependencies
    required_packages: List[str] = field(default_factory=list)
    required_files: List[str] = field(default_factory=list)

@dataclass
class Action:
    """A single action provided by a plugin"""

    action_name: str                   # e.g., "read_csv", "train_model"
    description: str                   # What this action does

    # Interface
    parameters: Dict[str, Dict] = field(default_factory=dict)  # param_name → schema
    returns: Dict = field(default_factory=dict)  # Return value schema

    # Implementation
    handler: Callable                  # Function that executes the action

    # Metadata
    estimated_time: float = 1.0        # Seconds (approximate)
    requires_gpu: bool = False
```

**Example Plugin:**

```python
# Data exploration plugin
data_explorer_functionality = PluginFunctionality(
    plugin_id="pandas_data_explorer",
    plugin_name="Pandas Data Explorer",
    description="Provides data exploration capabilities using pandas",
    actions={
        "read_csv": Action(
            action_name="read_csv",
            description="Read CSV file into DataFrame",
            parameters={
                "filepath": {"type": "string", "required": True},
                "nrows": {"type": "integer", "required": False}
            },
            returns={"type": "dataframe"},
            handler=pd.read_csv
        ),
        "describe_data": Action(
            action_name="describe_data",
            description="Generate statistical summary of DataFrame",
            parameters={
                "df": {"type": "dataframe", "required": True}
            },
            returns={"type": "dict"},
            handler=lambda df: df.describe().to_dict()
        ),
        "detect_missing": Action(
            action_name="detect_missing",
            description="Identify missing values in dataset",
            parameters={
                "df": {"type": "dataframe", "required": True}
            },
            returns={"type": "dict"},
            handler=lambda df: df.isnull().sum().to_dict()
        )
    },
    required_packages=["pandas", "numpy"]
)
```

---

### 5.2 Plugin Configuration (Cj)

```python
@dataclass
class PluginConfig:
    """Configuration parameters for plugin behavior"""

    # Runtime settings
    timeout: int = 300                 # Max execution time per action (seconds)
    max_retries: int = 2               # Retry failed actions

    # Resource limits
    max_memory_mb: int = 4096          # Memory limit
    max_cpu_cores: int = 4             # CPU core limit
    allow_gpu: bool = False            # GPU access

    # Data access
    allowed_read_paths: List[str] = field(default_factory=list)
    allowed_write_paths: List[str] = field(default_factory=list)

    # Logging
    log_level: str = "INFO"            # DEBUG, INFO, WARNING, ERROR
    save_intermediate_results: bool = True

    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)
```

**Example:**

```python
# Configuration for ML training plugin
trainer_config = PluginConfig(
    timeout=7200,  # 2 hours
    max_memory_mb=16384,  # 16GB
    max_cpu_cores=8,
    allow_gpu=True,
    allowed_read_paths=["/home/data", "/home/models"],
    allowed_write_paths=["/home/models", "/home/logs"],
    log_level="DEBUG",
    custom_params={
        "max_epochs": 100,
        "early_stopping_patience": 5,
        "save_checkpoints": True
    }
)
```

---

### 5.3 Usage Constraints (Uj)

```python
@dataclass
class PluginConstraints:
    """Defines when and how a plugin can be used"""

    # Availability constraints
    requires_internet: bool = False
    requires_gpu: bool = False
    min_memory_mb: int = 512
    min_disk_gb: int = 1

    # Usage limits
    max_calls_per_agent: int = -1      # -1 = unlimited
    max_concurrent_calls: int = 1      # Parallel execution limit

    # Temporal constraints
    cooldown_seconds: float = 0.0      # Minimum time between calls

    # Data constraints
    max_input_size_mb: float = 100.0   # Max input data size
    max_output_size_mb: float = 100.0  # Max output data size

    # Security
    sandbox_mode: bool = True          # Execute in sandbox
    allowed_network_hosts: List[str] = field(default_factory=list)

    # Dependencies
    prerequisite_plugins: List[str] = field(default_factory=list)
    incompatible_plugins: List[str] = field(default_factory=list)
```

**Example:**

```python
# Constraints for web scraping plugin
web_scraper_constraints = PluginConstraints(
    requires_internet=True,
    requires_gpu=False,
    max_calls_per_agent=50,
    max_concurrent_calls=3,
    cooldown_seconds=1.0,  # Rate limiting
    allowed_network_hosts=["api.example.com", "data.source.org"],
    sandbox_mode=True
)

# Constraints for GPU-intensive model training
gpu_trainer_constraints = PluginConstraints(
    requires_gpu=True,
    min_memory_mb=8192,
    max_concurrent_calls=1,  # Only one GPU job at a time
    max_input_size_mb=5000.0,
    sandbox_mode=False  # Needs direct GPU access
)
```

---

### 5.4 Complete Plugin Class

```python
@dataclass
class Plugin:
    """Complete plugin specification: Pj = {Fj, Cj, Uj}"""

    functionality: PluginFunctionality  # Fj: What it does
    config: PluginConfig                # Cj: How it's configured
    constraints: PluginConstraints      # Uj: Usage limits

    # Runtime state
    call_count: int = 0
    last_call_time: float = 0.0

    def execute_action(self, action_name: str, params: Dict) -> Any:
        """Execute an action with constraint checking"""

        # 1. Validate constraints
        self._check_constraints(action_name)

        # 2. Get action handler
        if action_name not in self.functionality.actions:
            raise ValueError(f"Action {action_name} not found in plugin")

        action = self.functionality.actions[action_name]

        # 3. Validate parameters
        self._validate_params(action, params)

        # 4. Execute with timeout and error handling
        try:
            result = self._execute_with_timeout(
                action.handler,
                params,
                timeout=self.config.timeout
            )
            self.call_count += 1
            self.last_call_time = time.time()
            return result

        except Exception as e:
            if self.config.max_retries > 0:
                # Retry logic
                return self._retry_execution(action, params)
            else:
                raise

    def _check_constraints(self, action_name: str):
        """Verify usage constraints before execution"""

        # Check call limit
        if self.constraints.max_calls_per_agent != -1:
            if self.call_count >= self.constraints.max_calls_per_agent:
                raise RuntimeError("Plugin call limit exceeded")

        # Check cooldown
        if self.constraints.cooldown_seconds > 0:
            elapsed = time.time() - self.last_call_time
            if elapsed < self.constraints.cooldown_seconds:
                raise RuntimeError(f"Cooldown period not met ({elapsed:.1f}s)")

        # Check resource availability
        if self.constraints.requires_gpu and not torch.cuda.is_available():
            raise RuntimeError("Plugin requires GPU but none available")
```

---

## 6. Planning Module Design

The planning module is the **meta-agent** that orchestrates the entire system.

### 6.1 Architecture

```python
class PlanningModule:
    """
    Meta-agent that analyzes problems and creates agent workflows
    """

    def __init__(self, llm_config: LLMConfig):
        self.llm_config = llm_config  # Use powerful model (e.g., o1-preview)
        self.agent_templates = self._load_agent_templates()
        self.plugin_catalog = self._load_plugin_catalog()

    def plan_workflow(self, problem_spec: ProblemSpecification) -> WorkflowPlan:
        """
        Main planning function

        Input: Problem specification (task description, data info, constraints)
        Output: Complete workflow plan with agents and plugins
        """

        # Phase 1: Problem Understanding
        problem_analysis = self._analyze_problem(problem_spec)

        # Phase 2: Task Decomposition
        subtasks = self._decompose_into_subtasks(problem_analysis)

        # Phase 3: Agent Selection & Composition
        agent_specs = self._design_agents(subtasks, problem_analysis)

        # Phase 4: Plugin Selection
        plugin_assignments = self._assign_plugins(agent_specs, subtasks)

        # Phase 5: Workflow Graph Construction
        workflow_graph = self._build_workflow_graph(
            agent_specs,
            plugin_assignments,
            subtasks
        )

        # Phase 6: Resource Estimation
        resource_estimates = self._estimate_resources(workflow_graph)

        # Return complete plan
        return WorkflowPlan(
            problem_analysis=problem_analysis,
            subtasks=subtasks,
            agents=agent_specs,
            plugins=plugin_assignments,
            workflow_graph=workflow_graph,
            resource_estimates=resource_estimates
        )
```

---

### 6.2 Problem Specification

```python
@dataclass
class ProblemSpecification:
    """Input to the planning module"""

    # Problem description
    task_description: str              # Natural language description
    competition_id: str                # MLE-Bench competition ID

    # Data information
    data_files: List[str]              # Available data files
    data_description: str              # Description of datasets

    # Success criteria
    evaluation_metric: str             # e.g., "accuracy", "rmse"
    submission_format: str             # Expected output format

    # Constraints
    time_limit_hours: int = 24
    compute_resources: Dict = field(default_factory=dict)

    # Additional context
    domain: str = ""                   # e.g., "computer_vision", "nlp", "tabular"
    difficulty: str = ""               # "low", "medium", "high"
```

---

### 6.3 Workflow Plan

```python
@dataclass
class WorkflowPlan:
    """Output of the planning module"""

    # Analysis
    problem_analysis: Dict             # Problem understanding

    # Task breakdown
    subtasks: List[Subtask]            # Ordered list of subtasks

    # Agent composition
    agents: List[Agent]                # Agent specifications

    # Plugin assignments
    plugins: Dict[str, List[Plugin]]   # agent_id → plugins

    # Execution plan
    workflow_graph: nx.DiGraph         # DAG of agent dependencies

    # Resource planning
    resource_estimates: Dict           # Time, memory, cost estimates

    # Human review
    requires_approval: bool = True
    approval_checkpoints: List[str] = field(default_factory=list)

    def visualize(self) -> str:
        """Generate visual representation of workflow"""
        pass

    def to_config_files(self, output_dir: Path):
        """Generate all configuration files for execution"""
        pass
```

---

### 6.4 Planning Process Details

#### Phase 1: Problem Understanding

```python
def _analyze_problem(self, problem_spec: ProblemSpecification) -> Dict:
    """
    Analyze the problem to understand:
    - Problem type (classification, regression, segmentation, etc.)
    - Data characteristics (size, modality, format)
    - Complexity level
    - Required techniques
    """

    analysis_prompt = f"""
    Analyze this machine learning problem:

    Task: {problem_spec.task_description}
    Data files: {problem_spec.data_files}
    Metric: {problem_spec.evaluation_metric}

    Provide:
    1. Problem type (classification/regression/clustering/etc.)
    2. Data modality (text/image/tabular/time-series/etc.)
    3. Estimated complexity (simple/moderate/complex)
    4. Key challenges
    5. Recommended approaches
    """

    # Use LLM to analyze
    response = self._call_llm(analysis_prompt)

    return {
        "problem_type": response["problem_type"],
        "data_modality": response["data_modality"],
        "complexity": response["complexity"],
        "challenges": response["challenges"],
        "recommended_approaches": response["approaches"]
    }
```

#### Phase 2: Task Decomposition

```python
def _decompose_into_subtasks(self, problem_analysis: Dict) -> List[Subtask]:
    """
    Break down the problem into subtasks

    Example for tabular classification:
    1. Data exploration and understanding
    2. Data cleaning and preprocessing
    3. Feature engineering
    4. Model selection and training
    5. Hyperparameter tuning
    6. Validation and submission
    """

    # Use problem analysis to determine subtasks
    subtask_templates = {
        "classification": [
            Subtask("data_exploration", "Explore and understand the dataset"),
            Subtask("data_preprocessing", "Clean and preprocess data"),
            Subtask("feature_engineering", "Create and select features"),
            Subtask("model_training", "Train classification models"),
            Subtask("hyperparameter_tuning", "Optimize model parameters"),
            Subtask("validation", "Validate and prepare submission")
        ],
        "image_segmentation": [
            Subtask("data_exploration", "Analyze images and masks"),
            Subtask("data_augmentation", "Create augmented training data"),
            Subtask("model_architecture", "Design segmentation model"),
            Subtask("model_training", "Train segmentation model"),
            Subtask("post_processing", "Refine predictions"),
            Subtask("validation", "Validate and prepare submission")
        ],
        # More templates...
    }

    # Select appropriate template based on problem type
    problem_type = problem_analysis["problem_type"]
    subtasks = subtask_templates.get(problem_type, self._generate_custom_subtasks(problem_analysis))

    return subtasks
```

#### Phase 3: Agent Design

```python
def _design_agents(self, subtasks: List[Subtask], problem_analysis: Dict) -> List[Agent]:
    """
    Design specialized agents for subtasks

    Strategy:
    - One agent per subtask (specialist approach)
    - OR group related subtasks (generalist approach)
    - Consider spawn hierarchies for complex tasks
    """

    agents = []

    # Option 1: Specialist agents (one per subtask)
    for subtask in subtasks:
        agent = self._create_agent_for_subtask(subtask, problem_analysis)
        agents.append(agent)

    # Option 2: Supervisor + workers (for complex problems)
    if problem_analysis["complexity"] == "complex":
        supervisor = self._create_supervisor_agent(subtasks)
        workers = [self._create_worker_agent(st) for st in subtasks]
        agents = [supervisor] + workers

    return agents

def _create_agent_for_subtask(self, subtask: Subtask, problem_analysis: Dict) -> Agent:
    """Create a specialized agent for a specific subtask"""

    # Select appropriate LLM based on task complexity
    if subtask.requires_complex_reasoning:
        llm_config = LLMConfig(model_name="gpt-4o", temperature=0.7)
    else:
        llm_config = LLMConfig(model_name="gpt-4o-mini", temperature=0.3)

    # Define role
    role = AgentRole(
        role_name=subtask.name,
        description=subtask.description,
        allowed_actions=self._determine_actions(subtask),
        max_steps=self._estimate_steps(subtask),
        max_time_seconds=self._estimate_time(subtask)
    )

    # Create agent
    return Agent(
        agent_id=f"agent_{subtask.name}",
        llm_config=llm_config,
        role=role,
        state=AgentState(),
        spawn_config=SpawnConfig(can_spawn=False),
        history=ExecutionHistory()
    )
```

#### Phase 4: Plugin Assignment

```python
def _assign_plugins(self, agents: List[Agent], subtasks: List[Subtask]) -> Dict[str, List[Plugin]]:
    """
    Assign appropriate plugins to each agent based on their role
    """

    assignments = {}

    for agent in agents:
        # Determine required capabilities
        required_capabilities = self._analyze_role_requirements(agent.role)

        # Search plugin catalog
        matching_plugins = []
        for plugin_id, plugin in self.plugin_catalog.items():
            if self._plugin_matches_requirements(plugin, required_capabilities):
                matching_plugins.append(plugin)

        assignments[agent.agent_id] = matching_plugins

    return assignments

def _plugin_matches_requirements(self, plugin: Plugin, requirements: List[str]) -> bool:
    """Check if plugin provides required capabilities"""

    plugin_capabilities = set(plugin.functionality.actions.keys())
    required_capabilities = set(requirements)

    # Plugin matches if it provides any required capability
    return len(plugin_capabilities & required_capabilities) > 0
```

#### Phase 5: Workflow Graph

```python
def _build_workflow_graph(
    self,
    agents: List[Agent],
    plugin_assignments: Dict,
    subtasks: List[Subtask]
) -> nx.DiGraph:
    """
    Build a directed acyclic graph (DAG) representing agent dependencies
    """

    graph = nx.DiGraph()

    # Add nodes (agents)
    for agent in agents:
        graph.add_node(
            agent.agent_id,
            agent=agent,
            plugins=plugin_assignments[agent.agent_id]
        )

    # Add edges (dependencies)
    for i, subtask in enumerate(subtasks):
        if i > 0:
            # Current subtask depends on previous subtask
            prev_agent_id = f"agent_{subtasks[i-1].name}"
            curr_agent_id = f"agent_{subtask.name}"

            graph.add_edge(prev_agent_id, curr_agent_id)

        # Add custom dependencies based on subtask requirements
        for dependency in subtask.dependencies:
            dep_agent_id = f"agent_{dependency}"
            curr_agent_id = f"agent_{subtask.name}"
            graph.add_edge(dep_agent_id, curr_agent_id)

    # Validate DAG
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Workflow graph contains cycles!")

    return graph
```

---

### 6.5 Human Collaboration Interface

```python
class HumanCollaborationInterface:
    """Interface for human-agent collaboration"""

    def present_plan(self, workflow_plan: WorkflowPlan) -> str:
        """
        Generate human-readable workflow plan presentation
        """

        presentation = []

        # 1. Problem summary
        presentation.append("## Problem Analysis")
        presentation.append(self._format_problem_analysis(workflow_plan.problem_analysis))

        # 2. Proposed workflow
        presentation.append("\n## Proposed Workflow")
        presentation.append(self._format_workflow_graph(workflow_plan.workflow_graph))

        # 3. Agent specifications
        presentation.append("\n## Agent Team")
        for agent in workflow_plan.agents:
            presentation.append(f"\n### Agent: {agent.role.role_name}")
            presentation.append(f"- **Role**: {agent.role.description}")
            presentation.append(f"- **LLM**: {agent.llm_config.model_name}")
            presentation.append(f"- **Time Budget**: {agent.role.max_time_seconds}s")
            presentation.append(f"- **Plugins**: {', '.join([p.functionality.plugin_name for p in workflow_plan.plugins[agent.agent_id]])}")

        # 4. Resource estimates
        presentation.append("\n## Resource Estimates")
        presentation.append(self._format_resource_estimates(workflow_plan.resource_estimates))

        # 5. Approval request
        presentation.append("\n## Approval Request")
        presentation.append("Please review the plan above and provide feedback:")
        presentation.append("- Type 'approve' to proceed with this plan")
        presentation.append("- Type 'modify: <description>' to request changes")
        presentation.append("- Type 'reject' to cancel")

        return "\n".join(presentation)

    def gather_feedback(self) -> Dict:
        """Collect human feedback on the plan"""

        response = input("\nYour response: ").strip().lower()

        if response == "approve":
            return {"status": "approved", "modifications": []}

        elif response.startswith("modify:"):
            modification = response.replace("modify:", "").strip()
            return {"status": "needs_modification", "modifications": [modification]}

        elif response == "reject":
            return {"status": "rejected", "modifications": []}

        else:
            print("Invalid response. Please try again.")
            return self.gather_feedback()

    def iterate_on_plan(self, workflow_plan: WorkflowPlan, modifications: List[str]) -> WorkflowPlan:
        """Revise the plan based on human feedback"""

        # Use LLM to interpret modifications and update plan
        revision_prompt = f"""
        The user requested the following modifications to the workflow plan:
        {modifications}

        Current plan:
        {workflow_plan}

        Generate a revised plan that incorporates the requested changes.
        """

        # Generate revised plan
        revised_plan = self._call_llm_for_plan_revision(revision_prompt)

        return revised_plan
```

---

## 7. Workflow Execution

Once the plan is approved, the execution engine instantiates agents and coordinates their execution.

### 7.1 Execution Engine

```python
class WorkflowExecutionEngine:
    """Executes approved workflow plans"""

    def __init__(self, workflow_plan: WorkflowPlan):
        self.plan = workflow_plan
        self.agent_instances = {}
        self.plugin_instances = {}
        self.event_bus = EventBus()
        self.shared_state = SharedState()

    def execute(self) -> ExecutionResult:
        """Execute the workflow plan"""

        # 1. Setup phase
        self._instantiate_agents()
        self._instantiate_plugins()
        self._setup_communication()

        # 2. Execution phase
        execution_order = self._get_topological_order()

        for agent_id in execution_order:
            agent = self.agent_instances[agent_id]

            # Execute agent
            result = self._execute_agent(agent)

            # Update shared state
            self.shared_state.update(agent_id, result)

            # Check for early termination
            if self._should_terminate(result):
                break

        # 3. Finalization phase
        final_result = self._finalize_execution()

        return final_result

    def _instantiate_agents(self):
        """Create agent instances from specifications"""

        for agent_spec in self.plan.agents:
            # Create agent instance
            agent = Agent(
                agent_id=agent_spec.agent_id,
                llm_config=agent_spec.llm_config,
                role=agent_spec.role,
                state=AgentState(),
                spawn_config=agent_spec.spawn_config,
                history=ExecutionHistory()
            )

            # Assign plugins
            agent.plugins = {
                p.functionality.plugin_id: p
                for p in self.plan.plugins[agent_spec.agent_id]
            }

            self.agent_instances[agent_spec.agent_id] = agent

    def _execute_agent(self, agent: Agent) -> Dict:
        """Execute a single agent until completion"""

        print(f"\n{'='*60}")
        print(f"Executing: {agent.role.role_name}")
        print(f"{'='*60}\n")

        agent.state.status = "planning"
        agent.history.start_time = datetime.now()

        while agent.state.current_step < agent.role.max_steps:
            # Get current observation
            observation = self._get_observation(agent)

            # Agent takes action
            result = agent.act(observation)

            # Check if goal achieved
            if self._goal_achieved(agent, result):
                agent.state.status = "completed"
                break

            agent.state.current_step += 1

        agent.history.end_time = datetime.now()

        return {
            "agent_id": agent.agent_id,
            "status": agent.state.status,
            "outputs": agent.state.outputs,
            "artifacts": agent.state.artifacts
        }

    def _get_topological_order(self) -> List[str]:
        """Get execution order based on workflow DAG"""
        return list(nx.topological_sort(self.plan.workflow_graph))
```

---

### 7.2 Shared State & Communication

```python
class SharedState:
    """Shared state accessible to all agents"""

    def __init__(self):
        self.data = {}
        self.lock = threading.Lock()

    def update(self, agent_id: str, result: Dict):
        """Update shared state with agent results"""
        with self.lock:
            self.data[agent_id] = result

    def get(self, key: str) -> Any:
        """Retrieve value from shared state"""
        with self.lock:
            return self.data.get(key)

class EventBus:
    """Event bus for inter-agent communication"""

    def __init__(self):
        self.subscribers = {}
        self.event_queue = queue.Queue()

    def publish(self, event: Dict):
        """Publish event to all subscribers"""
        self.event_queue.put(event)

    def subscribe(self, agent_id: str, event_types: List[str]):
        """Subscribe agent to specific event types"""
        for event_type in event_types:
            if event_type not in self.subscribers:
                self.subscribers[event_type] = []
            self.subscribers[event_type].append(agent_id)

    def get_events(self, agent_id: str) -> List[Dict]:
        """Get events for a specific agent"""
        events = []
        while not self.event_queue.empty():
            event = self.event_queue.get()
            if agent_id in self.subscribers.get(event["type"], []):
                events.append(event)
        return events
```

---

## 8. Implementation Plan

### Phase 1: Core Infrastructure (Week 1-2)

**Tasks:**
1. Implement base classes:
   - `Agent`, `Plugin`, `LLMConfig`, `AgentRole`, etc.
2. Create plugin system:
   - `PluginFunctionality`, `PluginConfig`, `PluginConstraints`
3. Build event bus and shared state
4. Setup basic execution engine

**Deliverables:**
- `agents/planning_system/core/` directory with base classes
- Unit tests for core components
- Example agent and plugin

---

### Phase 2: Planning Module (Week 3-4)

**Tasks:**
1. Implement `PlanningModule` class
2. Build problem analysis pipeline
3. Create task decomposition logic
4. Develop agent/plugin selection algorithms
5. Build workflow graph constructor

**Deliverables:**
- `agents/planning_system/planning/` directory
- Integration tests with sample problems
- Workflow visualization tools

---

### Phase 3: Human Collaboration Interface (Week 5)

**Tasks:**
1. Build plan presentation system
2. Create interactive feedback collection
3. Implement plan revision logic

**Deliverables:**
- `agents/planning_system/collaboration/` directory
- CLI interface for plan review
- Plan modification engine

---

### Phase 4: Plugin Library (Week 6-7)

**Tasks:**
1. Build standard plugins:
   - Data exploration (pandas)
   - ML training (sklearn, pytorch, tensorflow)
   - Hyperparameter tuning
   - Submission validation
2. Create plugin templates
3. Document plugin creation guide

**Deliverables:**
- `agents/planning_system/plugins/` directory
- 10+ standard plugins
- Plugin development guide

---

### Phase 5: Integration & Testing (Week 8)

**Tasks:**
1. Integrate with MLE-Bench infrastructure
2. End-to-end testing on sample competitions
3. Performance benchmarking
4. Bug fixes and optimization

**Deliverables:**
- Fully integrated system
- Test suite covering major competitions
- Performance report

---

## 9. Examples

### Example 1: Simple Tabular Classification

**Problem**: Spaceship Titanic (binary classification)

**Planning Module Output:**

```yaml
workflow_plan:
  problem_analysis:
    problem_type: "binary_classification"
    data_modality: "tabular"
    complexity: "simple"
    challenges: ["missing values", "categorical features"]

  agents:
    - agent_id: "explorer"
      role: "DataExplorer"
      llm_config:
        model_name: "gpt-4o-mini"
        temperature: 0.3
      plugins: ["pandas_explorer", "data_profiler"]
      max_steps: 30
      max_time: 1800

    - agent_id: "trainer"
      role: "ModelTrainer"
      llm_config:
        model_name: "gpt-4o"
        temperature: 0.5
      plugins: ["sklearn_trainer", "xgboost_trainer"]
      max_steps: 100
      max_time: 7200

    - agent_id: "validator"
      role: "SubmissionValidator"
      llm_config:
        model_name: "gpt-4o-mini"
        temperature: 0.1
      plugins: ["format_checker", "grading_client"]
      max_steps: 20
      max_time: 600

  workflow_graph:
    edges:
      - [explorer, trainer]
      - [trainer, validator]

  resource_estimates:
    total_time_hours: 3
    total_cost_usd: 2.50
    success_probability: 0.85
```

---

### Example 2: Complex Image Segmentation

**Problem**: HubMap Kidney Segmentation

**Planning Module Output:**

```yaml
workflow_plan:
  problem_analysis:
    problem_type: "instance_segmentation"
    data_modality: "image"
    complexity: "high"
    challenges: ["large images", "class imbalance", "GPU required"]

  agents:
    - agent_id: "supervisor"
      role: "WorkflowSupervisor"
      llm_config:
        model_name: "o1-preview"
        temperature: 1.0
      spawn_config:
        can_spawn: true
        max_children: 3
      max_steps: 500
      max_time: 86400

    - agent_id: "data_processor"
      role: "ImageDataProcessor"
      llm_config:
        model_name: "gpt-4o"
      plugins: ["image_loader", "augmentation_engine"]
      max_steps: 100
      max_time: 7200

    - agent_id: "model_architect"
      role: "SegmentationModelDesigner"
      llm_config:
        model_name: "claude-3.5-sonnet"
      plugins: ["unet_builder", "pytorch_modules"]
      max_steps: 50
      max_time: 3600

    - agent_id: "model_trainer"
      role: "DeepLearningTrainer"
      llm_config:
        model_name: "gpt-4o"
      plugins: ["pytorch_trainer", "tensorboard_logger"]
      max_steps: 300
      max_time: 72000
      constraints:
        requires_gpu: true

    - agent_id: "post_processor"
      role: "PredictionPostProcessor"
      llm_config:
        model_name: "gpt-4o-mini"
      plugins: ["rle_encoder", "mask_refiner"]
      max_steps: 50
      max_time: 3600

  workflow_graph:
    edges:
      - [supervisor, data_processor]
      - [supervisor, model_architect]
      - [data_processor, model_trainer]
      - [model_architect, model_trainer]
      - [model_trainer, post_processor]

  resource_estimates:
    total_time_hours: 22
    gpu_hours: 20
    total_cost_usd: 45.00
    success_probability: 0.65
```

---

## Conclusion

This design provides a flexible, composable system for tackling ML engineering problems through intelligent agent orchestration. The system learns from the strengths of AIDE, MLAgentBench, and OpenDevin while introducing:

1. **Dynamic composition**: Agents and plugins created based on problem analysis
2. **Human collaboration**: Plans validated before expensive execution
3. **Specialization**: Focused agents with clear roles and tools
4. **Observability**: Complete execution history and event tracking
5. **Scalability**: Support for simple to complex multi-agent workflows

The formal specifications (Agent Ai = {Li, Ri, Si, Ci, Hi} and Plugin Pj = {Fj, Cj, Uj}) provide a rigorous foundation while maintaining practical flexibility.
