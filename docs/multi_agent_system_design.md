# Multi-Agent System Design for MLE-bench

**Version:** 1.0
**Date:** 2025-11-14
**Status:** Design Specification

## Table of Contents
1. [System Overview](#system-overview)
2. [Agent Definitions](#agent-definitions)
3. [Plug Definitions](#plug-definitions)
4. [System Architecture](#system-architecture)
5. [Workflow and Coordination](#workflow-and-coordination)
6. [Communication Protocol](#communication-protocol)
7. [Implementation Roadmap](#implementation-roadmap)

---

## System Overview

The MLE-bench Multi-Agent System (MAS) is designed to solve diverse machine learning competitions through specialized agent collaboration. The system employs a hierarchical architecture with:

- **7 Core Agents**: Each with specific roles and capabilities
- **12 Plug Components**: Reusable action modules shared across agents
- **State Management**: Centralized state tracking and sharing
- **Dynamic Spawning**: Agents can create specialized sub-agents as needed

### Design Philosophy

1. **Specialization**: Agents focus on specific domains or pipeline stages
2. **Modularity**: Plugs provide reusable, composable functionality
3. **Adaptability**: System adapts to competition type and complexity
4. **Resource Awareness**: Explicit resource management and budgeting
5. **Failure Recovery**: Built-in error handling and fallback strategies

---

## Agent Definitions

Each agent is defined as **Ai = {Li, Ri, Si, Ci, Hi}** where:
- **Li**: Language model and configuration
- **Ri**: Role and responsibilities
- **Si**: State structure
- **Ci**: Can spawn children (boolean)
- **Hi**: History tracking configuration

### A1: Orchestrator Agent

**L1 (Language Model & Configuration):**
```yaml
model: claude-sonnet-4.5
temperature: 0.3
max_tokens: 4096
context_window: 200000
system_prompt: "You are the orchestrator agent coordinating a multi-agent system to solve ML competitions."
```

**R1 (Role):**
- Competition analysis and task classification
- Agent selection and spawning
- Workflow coordination and scheduling
- Resource allocation and time management
- Final submission assembly
- Performance monitoring and optimization

**S1 (State):**
```python
{
  "competition_id": str,
  "competition_type": str,  # CV, NLP, tabular, audio, multi-modal
  "complexity": str,  # low, medium, high
  "time_budget": int,  # seconds remaining
  "resource_budget": dict,  # CPU, GPU, memory limits
  "active_agents": list[str],  # Currently running agent IDs
  "workflow_stage": str,  # exploration, engineering, training, validation, submission
  "best_score": float,
  "submission_history": list[dict],
  "leaderboard_target": dict,  # gold/silver/bronze thresholds
  "error_log": list[dict]
}
```

**C1 (Can Spawn):** `True`

**H1 (History Tracking):**
```yaml
track_decisions: true
track_agent_communications: true
track_resource_usage: true
retention_policy: full  # Keep all history for post-analysis
max_history_tokens: 50000
```

---

### A2: Analysis Agent

**L2 (Language Model & Configuration):**
```yaml
model: claude-sonnet-4.5
temperature: 0.4
max_tokens: 8192
context_window: 200000
system_prompt: "You are an expert data scientist specializing in exploratory data analysis and competition understanding."
```

**R2 (Role):**
- Parse competition description and requirements
- Exploratory Data Analysis (EDA)
- Data quality assessment
- Metric understanding and interpretation
- Identify data challenges (imbalance, missing values, scale)
- Generate insights for downstream agents
- Recommend modeling approaches

**S2 (State):**
```python
{
  "dataset_summary": dict,  # columns, types, shapes, statistics
  "data_quality": dict,  # missing%, duplicates, anomalies
  "target_distribution": dict,  # class balance, range
  "feature_types": dict,  # numerical, categorical, text, image paths
  "correlation_matrix": array,
  "eda_visualizations": list[str],  # paths to generated plots
  "recommendations": list[str],  # modeling suggestions
  "identified_challenges": list[str],
  "metric_info": dict,  # metric name, optimization direction
  "sample_submission_format": dict
}
```

**C2 (Can Spawn):** `False`

**H2 (History Tracking):**
```yaml
track_decisions: true
track_agent_communications: true
track_resource_usage: false
retention_policy: summary  # Keep summarized findings
max_history_tokens: 20000
```

---

### A3: Domain Specialist Agents (A3a-d)

These agents are specialized for different data modalities. They share the same structure but different configurations.

#### A3a: Computer Vision Agent

**L3a (Language Model & Configuration):**
```yaml
model: claude-sonnet-4.5
temperature: 0.2
max_tokens: 4096
context_window: 200000
system_prompt: "You are a computer vision specialist expert in image classification, object detection, and segmentation."
tools_enabled: [vision_analysis, image_augmentation]
```

**R3a (Role):**
- Image preprocessing and augmentation strategy
- Architecture selection (CNN, ViT, hybrid)
- Transfer learning strategy
- Handle various image formats (TIFF, DICOM, PNG)
- Segmentation mask handling (RLE encoding)
- Multi-scale and ensemble strategies

#### A3b: NLP Agent

**L3b (Language Model & Configuration):**
```yaml
model: claude-sonnet-4.5
temperature: 0.2
max_tokens: 4096
context_window: 200000
system_prompt: "You are an NLP specialist expert in text classification, generation, and information extraction."
```

**R3b (Role):**
- Text preprocessing and tokenization
- Model selection (BERT, GPT, T5, etc.)
- Prompt engineering for few-shot learning
- Sequence length optimization
- Handle various NLP tasks (classification, QA, generation)

#### A3c: Tabular Agent

**L3c (Language Model & Configuration):**
```yaml
model: claude-sonnet-4.5
temperature: 0.2
max_tokens: 4096
context_window: 200000
system_prompt: "You are a tabular data specialist expert in feature engineering and gradient boosting methods."
```

**R3c (Role):**
- Feature engineering and selection
- Handling categorical and numerical features
- Missing value imputation strategies
- Model selection (XGBoost, LightGBM, CatBoost, TabNet)
- Cross-validation strategies

#### A3d: Audio Agent

**L3d (Language Model & Configuration):**
```yaml
model: claude-sonnet-4.5
temperature: 0.2
max_tokens: 4096
context_window: 200000
system_prompt: "You are an audio processing specialist expert in signal processing and audio classification."
```

**R3d (Role):**
- Audio preprocessing and feature extraction
- Spectrogram generation and augmentation
- Model selection for audio tasks
- Handle various audio formats

**S3 (Shared State Structure for Domain Agents):**
```python
{
  "model_architecture": str,  # Selected architecture
  "preprocessing_pipeline": dict,  # Steps and parameters
  "augmentation_strategy": dict,
  "training_config": dict,  # hyperparameters
  "validation_strategy": str,  # k-fold, stratified, time-based
  "training_logs": list[dict],  # epoch-wise metrics
  "model_checkpoints": list[str],  # paths to saved models
  "validation_scores": dict,  # CV scores
  "predictions": dict,  # validation and test predictions
  "feature_importance": dict  # if applicable
}
```

**C3 (Can Spawn):** `True` (can spawn hyperparameter tuning sub-agents)

**H3 (History Tracking):**
```yaml
track_decisions: true
track_agent_communications: true
track_resource_usage: true
retention_policy: checkpoints  # Keep model checkpoints and key decisions
max_history_tokens: 30000
```

---

### A4: Engineering Agent

**L4 (Language Model & Configuration):**
```yaml
model: claude-sonnet-4.5
temperature: 0.3
max_tokens: 4096
context_window: 200000
system_prompt: "You are a feature engineering expert specializing in creating powerful features from raw data."
```

**R4 (Role):**
- Automated feature engineering
- Feature interaction discovery
- Dimensionality reduction
- Feature selection and validation
- Create domain-specific features
- Feature preprocessing pipelines

**S4 (State):**
```python
{
  "original_features": list[str],
  "engineered_features": dict,  # feature_name: creation_logic
  "feature_importance_scores": dict,
  "selected_features": list[str],
  "preprocessing_pipeline": object,  # sklearn pipeline
  "feature_validation_scores": dict,  # contribution to CV score
  "feature_correlations": dict
}
```

**C4 (Can Spawn):** `False`

**H4 (History Tracking):**
```yaml
track_decisions: true
track_agent_communications: true
track_resource_usage: false
retention_policy: summary
max_history_tokens: 15000
```

---

### A5: Ensemble Agent

**L5 (Language Model & Configuration):**
```yaml
model: claude-sonnet-4.5
temperature: 0.2
max_tokens: 4096
context_window: 200000
system_prompt: "You are an ensemble learning expert combining multiple models for optimal performance."
```

**R5 (Role):**
- Collect predictions from multiple models
- Ensemble strategy selection (voting, averaging, stacking, blending)
- Weight optimization for ensemble
- Diversity analysis of base models
- Generate final predictions
- Calibration of ensemble outputs

**S5 (State):**
```python
{
  "base_models": list[str],  # model identifiers
  "model_predictions": dict,  # model_id: predictions
  "model_weights": dict,  # optimized weights
  "ensemble_strategy": str,  # voting, averaging, stacking, etc.
  "ensemble_cv_score": float,
  "diversity_metrics": dict,  # model diversity analysis
  "final_predictions": array
}
```

**C5 (Can Spawn):** `True` (can spawn optimization sub-agents)

**H5 (History Tracking):**
```yaml
track_decisions: true
track_agent_communications: true
track_resource_usage: false
retention_policy: summary
max_history_tokens: 10000
```

---

### A6: Validation Agent

**L6 (Language Model & Configuration):**
```yaml
model: claude-haiku-4
temperature: 0.1
max_tokens: 2048
context_window: 200000
system_prompt: "You are a validation specialist ensuring submission correctness and quality."
```

**R6 (Role):**
- Validate submission format
- Check for common errors (NaN, inf, wrong IDs)
- Verify submission against sample_submission.csv
- Cross-validation score tracking
- Leaderboard position estimation
- Quality assurance checks
- Communicate with grading server

**S6 (State):**
```python
{
  "submission_valid": bool,
  "validation_errors": list[str],
  "format_checks": dict,  # column names, dtypes, shape
  "cv_scores": list[float],  # cross-validation results
  "cv_mean": float,
  "cv_std": float,
  "estimated_leaderboard_score": float,
  "grading_server_response": dict,
  "submission_path": str
}
```

**C6 (Can Spawn):** `False`

**H6 (History Tracking):**
```yaml
track_decisions: false
track_agent_communications: true
track_resource_usage: false
retention_policy: minimal  # Only keep final validations
max_history_tokens: 5000
```

---

### A7: Resource Manager Agent

**L7 (Language Model & Configuration):**
```yaml
model: claude-haiku-4
temperature: 0.1
max_tokens: 2048
context_window: 50000
system_prompt: "You are a resource manager optimizing compute usage and time allocation."
```

**R7 (Role):**
- Monitor CPU, GPU, memory usage
- Time budget allocation across pipeline stages
- Detect resource bottlenecks
- Recommend resource optimization
- Kill runaway processes
- Data loading optimization
- Batch size recommendations

**S7 (State):**
```python
{
  "time_budget_total": int,  # seconds
  "time_elapsed": int,
  "time_remaining": int,
  "time_allocation": dict,  # stage: allocated_seconds
  "cpu_usage": float,  # percentage
  "gpu_usage": float,
  "memory_usage": float,
  "disk_usage": float,
  "bottlenecks": list[str],
  "optimization_recommendations": list[str],
  "active_processes": list[dict]
}
```

**C7 (Can Spawn):** `False`

**H7 (History Tracking):**
```yaml
track_decisions: false
track_agent_communications: false
track_resource_usage: true
retention_policy: metrics_only
max_history_tokens: 5000
```

---

## Plug Definitions

Each plug is defined as **Pj = {Fj, Cj, Uj}** where:
- **Fj**: Functionality (actions the plug performs)
- **Cj**: Configuration parameters
- **Uj**: Constraints and usage rules

### P1: Data Loading Plug

**F1 (Functionality):**
- Load CSV, Parquet, JSON files
- Load image datasets (with lazy loading)
- Load audio files
- Memory-efficient data streaming
- Data validation and type inference
- Handle large files (chunked reading)

**C1 (Configuration):**
```yaml
chunk_size: 10000  # rows per chunk for large files
lazy_loading: true  # for images
max_memory_percent: 40  # maximum memory to use
supported_formats: [csv, parquet, json, png, jpg, tiff, dicom, wav, mp3]
cache_enabled: true
cache_dir: /tmp/data_cache
```

**U1 (Constraints):**
- Must not exceed memory budget (40% of available RAM)
- Read-only access to /home/data/
- Must handle corrupted files gracefully
- Timeout: 300 seconds per file load operation
- Must release memory after agent completes

---

### P2: Exploratory Data Analysis (EDA) Plug

**F2 (Functionality):**
- Generate statistical summaries
- Detect missing values and outliers
- Compute correlations
- Create visualizations (histograms, scatter plots, correlation heatmaps)
- Data distribution analysis
- Target variable analysis

**C2 (Configuration):**
```yaml
max_visualizations: 10
visualization_format: png
statistical_tests: [normality, stationarity]
correlation_threshold: 0.95  # for detecting multicollinearity
outlier_method: IQR  # or z-score
missing_threshold: 0.5  # flag features with >50% missing
```

**U2 (Constraints):**
- Maximum 10 visualizations to save disk space
- Must complete within 600 seconds
- Cannot modify original data
- Visualizations saved to /home/logs/eda/
- Maximum 100MB total for visualization files

---

### P3: Preprocessing Plug

**F3 (Functionality):**
- Data cleaning (handle missing, duplicates)
- Normalization and scaling
- Encoding categorical variables
- Text preprocessing (tokenization, cleaning)
- Image preprocessing (resize, normalize)
- Audio preprocessing (resampling, normalization)
- Train-test splitting
- Cross-validation fold creation

**C3 (Configuration):**
```yaml
missing_strategy: auto  # auto, drop, impute_mean, impute_median, impute_mode
scaling_method: standard  # standard, minmax, robust
categorical_encoding: auto  # onehot, label, target, embedding
text_tokenizer: auto  # based on model selection
image_size: auto  # based on model requirements
cv_folds: 5
stratified: true
random_seed: 42
```

**U3 (Constraints):**
- Must preserve data types where possible
- Must maintain train-test separation
- Cannot introduce data leakage
- Must save preprocessing pipeline for inference
- Maximum processing time: 1800 seconds

---

### P4: Feature Engineering Plug

**F4 (Functionality):**
- Polynomial features
- Feature interactions
- Aggregation features
- Time-based features (from timestamps)
- Text features (TF-IDF, embeddings)
- Image features (edge detection, color histograms)
- Domain-specific features
- Feature selection

**C4 (Configuration):**
```yaml
polynomial_degree: 2
interaction_depth: 2
aggregation_functions: [mean, std, min, max, median]
text_vectorizer: tfidf  # or count, word2vec, bert
max_features: 1000
feature_selection_method: auto  # mutual_info, f_test, model_based
```

**U4 (Constraints):**
- Maximum 10000 features after engineering
- Must track feature provenance
- Feature creation must be reproducible
- Must complete within 3600 seconds
- Memory usage < 60% of available RAM

---

### P5: Model Training Plug

**F5 (Functionality):**
- Initialize model architectures
- Configure training loops
- Handle GPU/CPU training
- Implement early stopping
- Learning rate scheduling
- Gradient accumulation
- Mixed precision training
- Model checkpointing
- Training monitoring and logging

**C5 (Configuration):**
```yaml
device: auto  # auto, cpu, cuda
mixed_precision: true
gradient_accumulation_steps: 1
early_stopping_patience: 5
checkpoint_frequency: 1  # epochs
log_frequency: 100  # steps
max_epochs: 100
learning_rate: auto
batch_size: auto
optimizer: auto  # adam, adamw, sgd
scheduler: auto  # cosine, step, plateau
```

**U5 (Constraints):**
- Must respect time budget allocation
- Must save best checkpoint
- GPU memory < 95% to avoid OOM
- Must log training metrics every epoch
- Graceful degradation if resource limits hit
- Maximum training time per model: based on orchestrator allocation

---

### P6: Hyperparameter Optimization Plug

**F6 (Functionality):**
- Define search spaces
- Execute search strategies (grid, random, bayesian, optuna)
- Parallel trial execution
- Early stopping for poor trials
- Result tracking and analysis
- Best configuration selection

**C6 (Configuration):**
```yaml
strategy: bayesian  # grid, random, bayesian, optuna
max_trials: 50
parallel_trials: 2
timeout_per_trial: 3600  # seconds
optimization_metric: auto  # based on competition metric
search_space:
  learning_rate: [1e-5, 1e-2]
  batch_size: [16, 32, 64, 128]
  # additional parameters defined dynamically
```

**U6 (Constraints):**
- Total optimization time < 25% of overall time budget
- Must track all trial results
- Must save best configuration
- Cannot exceed GPU memory limits
- Must use cross-validation for trial evaluation

---

### P7: Cross-Validation Plug

**F7 (Functionality):**
- K-Fold cross-validation
- Stratified K-Fold
- Time-series cross-validation
- Group K-Fold
- Leave-One-Out (for small datasets)
- Custom split strategies
- CV score aggregation and statistics

**C7 (Configuration):**
```yaml
n_splits: 5
stratified: true
shuffle: true
random_state: 42
cv_strategy: auto  # auto, kfold, stratified, timeseries, group
group_column: null  # for group k-fold
```

**U7 (Constraints):**
- Must maintain data independence across folds
- Must use same folds for all models (reproducibility)
- CV execution time < 40% of total time budget
- Must save fold indices for reproducibility
- Minimum 2 folds, maximum 10 folds

---

### P8: Ensemble Plug

**F8 (Functionality):**
- Weighted averaging
- Voting (hard/soft)
- Stacking with meta-learner
- Blending
- Dynamic ensemble selection
- Ensemble weight optimization
- Prediction aggregation

**C8 (Configuration):**
```yaml
ensemble_method: auto  # voting, averaging, stacking, blending
weighting: optimized  # equal, optimized, manual
stacking_meta_learner: ridge  # for stacking
blending_holdout_size: 0.2  # for blending
optimization_metric: auto
n_jobs: -1
```

**U8 (Constraints):**
- Minimum 2 base models
- Maximum 20 base models
- Must validate ensemble on holdout set
- Ensemble creation time < 600 seconds
- Must improve upon best base model (or warn)

---

### P9: Submission Generation Plug

**F9 (Functionality):**
- Format predictions according to sample_submission.csv
- Handle various submission formats (single column, multi-class, RLE, etc.)
- ID matching and ordering
- Prediction post-processing
- File writing to /home/submission/
- Generate submission metadata

**C9 (Configuration):**
```yaml
submission_filename: submission.csv
float_precision: 6
rle_encoding: auto  # for segmentation tasks
id_column: auto  # detected from sample_submission
prediction_columns: auto
post_processing:
  clip: true  # clip to valid range
  round: false  # for classification
```

**U9 (Constraints):**
- Must match sample_submission.csv format exactly
- Must include all test IDs (no missing predictions)
- Must write to /home/submission/ only
- File size < 500MB
- Must validate before writing
- No NaN or inf values allowed

---

### P10: Validation Plug

**F10 (Functionality):**
- Format validation
- Schema validation
- Range validation
- ID validation
- Grading server communication
- Error reporting
- Fix common issues automatically

**C10 (Configuration):**
```yaml
grading_server_url: http://localhost:5000/validate
auto_fix: true  # automatically fix common issues
validation_checks:
  - format
  - schema
  - ids
  - values
  - dtypes
strict_mode: false  # error vs warning
```

**U10 (Constraints):**
- Must validate before final submission
- Maximum 3 grading server calls (to avoid abuse)
- Validation must complete in < 60 seconds
- Cannot modify predictions (only format)
- Must report all errors found

---

### P11: Metric Computation Plug

**F11 (Functionality):**
- Compute competition-specific metrics
- Support 15+ metric types (accuracy, AUC, RMSE, Dice, etc.)
- Handle various prediction formats
- Compute confidence intervals
- Metric-specific preprocessing (thresholding, rounding)

**C11 (Configuration):**
```yaml
metric: auto  # detected from competition
direction: auto  # minimize or maximize
averaging: auto  # for multi-class (micro, macro, weighted)
threshold: 0.5  # for binary classification
bootstrap_samples: 1000  # for confidence intervals
```

**U11 (Constraints):**
- Must support all MLE-bench metrics
- Must handle edge cases (all one class, etc.)
- Computation time < 30 seconds
- Must match official grading implementation
- Must work with various input formats

---

### P12: Logging and Monitoring Plug

**F12 (Functionality):**
- Structured logging
- Performance metrics tracking
- Resource usage monitoring
- Error logging and alerting
- Progress tracking
- Experiment tracking (MLflow-style)
- Dashboard generation

**C12 (Configuration):**
```yaml
log_level: INFO  # DEBUG, INFO, WARNING, ERROR
log_file: /home/logs/system.log
metrics_file: /home/logs/metrics.json
monitoring_frequency: 30  # seconds
track_experiments: true
dashboard_enabled: false  # resource-intensive
alert_on_errors: true
```

**U12 (Constraints):**
- Total log size < 1GB
- Monitoring overhead < 2% of CPU
- Must not interfere with training
- Logs must be parseable (JSON format)
- Must rotate logs if size limit exceeded

---

## System Architecture

### Hierarchical Structure

```
Orchestrator Agent (A1)
├── Resource Manager (A7) [Always Active]
├── Analysis Agent (A2) [Phase 1]
├── Engineering Agent (A4) [Phase 2]
├── Domain Specialist (A3a/b/c/d) [Phase 3]
│   ├── Sub-agent: HPO [Spawned if needed]
│   └── Sub-agent: HPO [Spawned if needed]
├── Ensemble Agent (A5) [Phase 4]
│   └── Sub-agent: Weight Optimizer [Spawned if needed]
└── Validation Agent (A6) [Continuous]
```

### Communication Patterns

**1. Hierarchical Communication:**
- Parent agents can communicate with direct children
- Children report results back to parents
- Orchestrator has global communication capability

**2. Shared State:**
- Centralized state store accessible by all agents
- State updates are logged and versioned
- Read: all agents; Write: owner agent only

**3. Message Queue:**
- Asynchronous message passing
- Priority queue for time-sensitive tasks
- Message types: REQUEST, RESPONSE, UPDATE, ERROR, ALERT

**4. Event System:**
- Agents can subscribe to events
- Event types: STATE_CHANGE, TASK_COMPLETE, ERROR, RESOURCE_ALERT
- Pub-sub pattern for loose coupling

### Data Flow

```
Input: Competition Data + Description
         ↓
[A1: Orchestrator] ← → [A7: Resource Manager]
         ↓
[A2: Analysis] → EDA Results → Shared State
         ↓
[A4: Engineering] → Feature Pipeline → Shared State
         ↓
[A3: Domain Specialist] → Model(s) → Checkpoints
         ↓
[A5: Ensemble] → Combined Predictions
         ↓
[A6: Validation] → Validated Submission
         ↓
Output: /home/submission/submission.csv
```

---

## Workflow and Coordination

### Phase 1: Understanding (10% of time budget)

**Responsible Agents:** A1 (Orchestrator), A2 (Analysis)

**Activities:**
1. Parse competition description (metric, task type, evaluation)
2. Load and inspect data (samples)
3. Perform EDA
4. Classify competition type (CV, NLP, tabular, audio, multi-modal)
5. Identify complexity and challenges
6. Generate recommendations

**Outputs:**
- Competition type classification
- Data summary and statistics
- Modeling recommendations
- Time and resource allocation plan

**Plugs Used:** P1 (Data Loading), P2 (EDA), P12 (Logging)

---

### Phase 2: Preparation (15% of time budget)

**Responsible Agents:** A1 (Orchestrator), A4 (Engineering)

**Activities:**
1. Data preprocessing pipeline creation
2. Feature engineering
3. Train-test split / CV fold creation
4. Feature selection
5. Prepare data for model training

**Outputs:**
- Preprocessing pipeline
- Engineered features
- CV folds
- Processed train/test data

**Plugs Used:** P3 (Preprocessing), P4 (Feature Engineering), P7 (Cross-Validation)

---

### Phase 3: Modeling (60% of time budget)

**Responsible Agents:** A1 (Orchestrator), A3 (Domain Specialists), A7 (Resource Manager)

**Activities:**
1. Select appropriate domain specialist(s)
2. Model architecture selection
3. Hyperparameter optimization (optional, time-permitting)
4. Model training with cross-validation
5. Multiple model training (diversity for ensemble)
6. Continuous validation and checkpointing

**Sub-Phase 3a: Quick Baseline (15% of phase 3)**
- Train simple, fast models
- Establish baseline performance
- Validate pipeline end-to-end

**Sub-Phase 3b: Advanced Modeling (70% of phase 3)**
- Train sophisticated models
- Hyperparameter tuning
- Transfer learning
- Data augmentation

**Sub-Phase 3c: Final Models (15% of phase 3)**
- Retrain best models on full train set
- Final checkpoints
- Generate test predictions

**Outputs:**
- Trained models (checkpoints)
- Cross-validation scores
- Test predictions from each model
- Model metadata and configs

**Plugs Used:** P5 (Model Training), P6 (HPO), P7 (CV), P11 (Metrics), P12 (Logging)

---

### Phase 4: Ensembling (10% of time budget)

**Responsible Agents:** A1 (Orchestrator), A5 (Ensemble), A6 (Validation)

**Activities:**
1. Collect predictions from all models
2. Analyze model diversity
3. Select ensemble strategy
4. Optimize ensemble weights
5. Generate final predictions
6. Validate submission format

**Outputs:**
- Final ensemble predictions
- Ensemble weights and strategy
- Submission file

**Plugs Used:** P8 (Ensemble), P9 (Submission Generation), P10 (Validation)

---

### Phase 5: Submission (5% of time budget)

**Responsible Agents:** A1 (Orchestrator), A6 (Validation)

**Activities:**
1. Final validation checks
2. Format submission file
3. Verify against grading server
4. Save submission
5. Generate submission report

**Outputs:**
- /home/submission/submission.csv
- Submission metadata
- Performance report

**Plugs Used:** P9 (Submission Generation), P10 (Validation), P12 (Logging)

---

### Continuous Activities (Throughout)

**Resource Monitoring (A7):**
- Monitor CPU, GPU, memory every 30 seconds
- Alert if thresholds exceeded
- Recommend optimizations

**Validation (A6):**
- Validate intermediate outputs
- Track CV scores
- Ensure reproducibility

**Orchestration (A1):**
- Monitor progress
- Adjust time allocation
- Handle errors and failures
- Coordinate agent communication

---

## Communication Protocol

### Message Format

```python
{
  "message_id": str,  # unique identifier
  "timestamp": float,
  "sender_id": str,  # agent ID
  "receiver_id": str,  # agent ID or "broadcast"
  "message_type": str,  # REQUEST, RESPONSE, UPDATE, ERROR, ALERT
  "priority": int,  # 0 (low) to 10 (critical)
  "payload": dict,  # message content
  "requires_response": bool,
  "correlation_id": str  # for request-response pairing
}
```

### Message Types

**REQUEST:**
- Agent requests action from another agent
- Must specify required response format
- Includes timeout

**RESPONSE:**
- Reply to a REQUEST
- Includes correlation_id
- Contains result or error

**UPDATE:**
- State update notification
- No response required
- Broadcast to interested parties

**ERROR:**
- Error notification
- Includes error details and stack trace
- May trigger fallback strategies

**ALERT:**
- Urgent notification (resource limits, time running out)
- High priority
- May require immediate action

### State Management

**Shared State Store:**
```python
{
  "global": {
    # Global state (A1)
  },
  "agents": {
    "A1": {/* Orchestrator state */},
    "A2": {/* Analysis state */},
    # ...
  },
  "artifacts": {
    "data": {/* data artifacts */},
    "models": {/* model checkpoints */},
    "predictions": {/* prediction files */},
    "visualizations": {/* plots */}
  },
  "history": [
    # State change history
  ]
}
```

**State Access Rules:**
- Read: All agents (via query API)
- Write: Owner agent only (via update API)
- Global state: Orchestrator only
- Artifacts: Append-only for non-owners

---

## Implementation Roadmap

### Phase I: Core Infrastructure (Weeks 1-2)

**Week 1:**
- [ ] Set up agent framework
- [ ] Implement message passing system
- [ ] Create shared state store
- [ ] Basic orchestrator agent (A1)
- [ ] Resource manager skeleton (A7)

**Week 2:**
- [ ] Implement all 12 plugs (basic versions)
- [ ] Logging and monitoring infrastructure
- [ ] Testing framework for agents
- [ ] Integration with MLE-bench environment

### Phase II: Essential Agents (Weeks 3-4)

**Week 3:**
- [ ] Analysis Agent (A2) implementation
- [ ] Validation Agent (A6) implementation
- [ ] Tabular Agent (A3c) implementation
- [ ] Basic end-to-end pipeline for tabular competitions

**Week 4:**
- [ ] CV Agent (A3a) implementation
- [ ] NLP Agent (A3b) implementation
- [ ] Engineering Agent (A4) implementation
- [ ] Test on low-complexity competitions

### Phase III: Advanced Features (Weeks 5-6)

**Week 5:**
- [ ] Ensemble Agent (A5) implementation
- [ ] Audio Agent (A3d) implementation
- [ ] Hyperparameter optimization
- [ ] Advanced plug features

**Week 6:**
- [ ] Agent spawning capability
- [ ] Dynamic resource allocation
- [ ] Failure recovery mechanisms
- [ ] Test on medium-complexity competitions

### Phase IV: Optimization (Weeks 7-8)

**Week 7:**
- [ ] Performance optimization
- [ ] Memory efficiency improvements
- [ ] Time allocation optimization
- [ ] Model selection improvements

**Week 8:**
- [ ] Multi-modal support
- [ ] Advanced ensemble strategies
- [ ] Domain-specific optimizations
- [ ] Test on high-complexity competitions

### Phase V: Evaluation and Refinement (Weeks 9-10)

**Week 9:**
- [ ] Full benchmark evaluation on all 75 competitions
- [ ] Analyze failure cases
- [ ] Performance profiling
- [ ] Identify bottlenecks

**Week 10:**
- [ ] Address failure modes
- [ ] Fine-tune time allocations
- [ ] Improve model selection heuristics
- [ ] Final validation and documentation

---

## Success Metrics

### System-Level Metrics

1. **Medal Rate:**
   - Gold: >10%
   - Silver: >20%
   - Bronze: >30%
   - **Target:** >35% overall medal rate

2. **Completion Rate:**
   - % of competitions with valid submissions
   - **Target:** >95%

3. **Resource Efficiency:**
   - Average GPU utilization: >70%
   - Average time utilization: >80%
   - Memory efficiency: <80% peak usage

4. **Reliability:**
   - Failure rate: <5%
   - Error recovery success: >90%

### Agent-Level Metrics

1. **Analysis Agent (A2):**
   - EDA quality score (manual evaluation)
   - Recommendation accuracy (do they lead to good models?)
   - Time to completion: <10% of budget

2. **Domain Specialists (A3):**
   - CV score vs baseline improvement
   - Model diversity (for ensembling)
   - Training efficiency (samples/second)

3. **Ensemble Agent (A5):**
   - Ensemble improvement over best base model
   - Ensemble creation time: <10% of budget

4. **Validation Agent (A6):**
   - False negative rate (missed errors): <1%
   - Validation time: <60 seconds

### Comparison Baselines

Current top MLE-bench agents (as of Nov 2024):
- **Operand:** 48% medal rate (16hr, Sonnet 3.5)
- **Neo multi-agent:** 30% medal rate (4hr, GPT-4o)

**Target:** Match or exceed Operand performance

---

## Risk Mitigation

### Identified Risks

1. **Time Overruns:**
   - **Mitigation:** Strict time budgets per phase, kill switch for long-running tasks

2. **Resource Exhaustion:**
   - **Mitigation:** A7 Resource Manager with hard limits, graceful degradation

3. **Agent Failures:**
   - **Mitigation:** Fallback strategies, error recovery, checkpointing

4. **Communication Overhead:**
   - **Mitigation:** Efficient message passing, minimize cross-agent communication

5. **State Inconsistency:**
   - **Mitigation:** Versioned state, atomic updates, validation

6. **Model Training Failures:**
   - **Mitigation:** Multiple model attempts, checkpointing, resume capability

7. **Ensemble Degrades Performance:**
   - **Mitigation:** Always keep best base model, validate ensemble improvement

### Fallback Strategies

1. **If Analysis fails:** Use minimal EDA, proceed with conservative preprocessing
2. **If Domain Specialist fails:** Fall back to simpler models (logistic regression, random forest)
3. **If Ensemble fails:** Submit best single model
4. **If time runs out:** Submit best available submission (even if incomplete)
5. **If resource exhausted:** Reduce batch size, use smaller models, disable augmentation

---

## Appendix

### A. Notation Reference

**Agent Definition:** Ai = {Li, Ri, Si, Ci, Hi}
- **Li:** Language model and configuration
- **Ri:** Role and responsibilities
- **Si:** State structure
- **Ci:** Can spawn children (boolean)
- **Hi:** History tracking configuration

**Plug Definition:** Pj = {Fj, Cj, Uj}
- **Fj:** Functionality (actions)
- **Cj:** Configuration parameters
- **Uj:** Constraints and usage rules

### B. Agent Summary Table

| Agent ID | Name | Model | Can Spawn | Primary Role |
|----------|------|-------|-----------|--------------|
| A1 | Orchestrator | Sonnet 4.5 | Yes | Coordination & planning |
| A2 | Analysis | Sonnet 4.5 | No | EDA & understanding |
| A3a | CV Specialist | Sonnet 4.5 | Yes | Computer vision tasks |
| A3b | NLP Specialist | Sonnet 4.5 | Yes | Natural language tasks |
| A3c | Tabular Specialist | Sonnet 4.5 | Yes | Tabular data tasks |
| A3d | Audio Specialist | Sonnet 4.5 | Yes | Audio processing tasks |
| A4 | Engineering | Sonnet 4.5 | No | Feature engineering |
| A5 | Ensemble | Sonnet 4.5 | Yes | Model ensembling |
| A6 | Validation | Haiku 4 | No | Submission validation |
| A7 | Resource Manager | Haiku 4 | No | Resource monitoring |

### C. Plug Summary Table

| Plug ID | Name | Used By | Primary Function |
|---------|------|---------|------------------|
| P1 | Data Loading | A2, A3*, A4 | Load various data formats |
| P2 | EDA | A2 | Exploratory analysis |
| P3 | Preprocessing | A4, A3* | Data cleaning & preparation |
| P4 | Feature Engineering | A4 | Create new features |
| P5 | Model Training | A3* | Train ML models |
| P6 | HPO | A3* | Hyperparameter tuning |
| P7 | Cross-Validation | A3*, A5 | CV strategies |
| P8 | Ensemble | A5 | Combine models |
| P9 | Submission | A5, A6 | Format predictions |
| P10 | Validation | A6 | Check submission |
| P11 | Metrics | A3*, A5, A6 | Compute scores |
| P12 | Logging | All | Track progress |

*A3* refers to all domain specialist agents

### D. Technology Stack

**Programming Language:** Python 3.11

**Core Libraries:**
- **Agent Framework:** Custom (or adapt existing: LangGraph, AutoGen)
- **State Management:** Redis or in-memory dict with persistence
- **Message Queue:** asyncio queue or RabbitMQ
- **LLM API:** Anthropic Claude API

**ML Libraries:**
- PyTorch, TensorFlow, scikit-learn
- XGBoost, LightGBM, CatBoost
- Transformers (Hugging Face)
- timm (PyTorch Image Models)
- Optuna (HPO)

**Utilities:**
- Pandas, NumPy, SciPy
- Matplotlib, Seaborn, Plotly
- tqdm (progress bars)

### E. Future Enhancements

1. **Meta-Learning:** Learn from past competitions to improve agent strategies
2. **Adaptive Time Allocation:** Dynamic reallocation based on progress
3. **Human-in-the-Loop:** Optional human feedback during exploration
4. **Transfer Learning:** Reuse models/features from similar competitions
5. **Automated Model Architecture Search:** NAS for custom architectures
6. **Multi-Agent Negotiation:** Agents negotiate resource allocation
7. **Continuous Learning:** Update agents based on MLE-bench results
8. **Explainability:** Generate explanations for model decisions
9. **Cost Optimization:** Minimize API calls while maintaining performance
10. **Distributed Execution:** Parallelize across multiple machines

---

**Document Control:**
- **Version:** 1.0
- **Status:** Design Specification - Ready for Review
- **Next Steps:** Implementation of Phase I (Core Infrastructure)
- **Review Required:** System architecture, resource allocations, time budgets
- **Approver:** Anthropic Team Lead

---

*End of Document*
