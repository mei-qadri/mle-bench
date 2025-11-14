# Multi-Agent System Architecture Diagrams

This document provides visual representations of the multi-agent system architecture for MLE-bench.

## 1. System Overview

```mermaid
graph TB
    subgraph "MLE-bench Environment"
        Data[Competition Data]
        Desc[Description]
        Sub[Submission]
    end

    subgraph "Multi-Agent System"
        A1[A1: Orchestrator<br/>Sonnet 4.5]
        A7[A7: Resource Manager<br/>Haiku 4]

        subgraph "Phase 1: Understanding"
            A2[A2: Analysis Agent<br/>Sonnet 4.5]
        end

        subgraph "Phase 2: Preparation"
            A4[A4: Engineering Agent<br/>Sonnet 4.5]
        end

        subgraph "Phase 3: Modeling"
            A3a[A3a: CV Agent<br/>Sonnet 4.5]
            A3b[A3b: NLP Agent<br/>Sonnet 4.5]
            A3c[A3c: Tabular Agent<br/>Sonnet 4.5]
            A3d[A3d: Audio Agent<br/>Sonnet 4.5]
        end

        subgraph "Phase 4: Ensembling"
            A5[A5: Ensemble Agent<br/>Sonnet 4.5]
        end

        subgraph "Phase 5: Validation"
            A6[A6: Validation Agent<br/>Haiku 4]
        end
    end

    Data --> A1
    Desc --> A1
    A1 --> A7
    A1 --> A2
    A2 --> A4
    A4 --> A3a
    A4 --> A3b
    A4 --> A3c
    A4 --> A3d
    A3a --> A5
    A3b --> A5
    A3c --> A5
    A3d --> A5
    A5 --> A6
    A6 --> Sub

    A7 -.Monitor.-> A2
    A7 -.Monitor.-> A4
    A7 -.Monitor.-> A3a
    A7 -.Monitor.-> A3b
    A7 -.Monitor.-> A3c
    A7 -.Monitor.-> A3d
    A7 -.Monitor.-> A5
```

## 2. Agent Hierarchy and Spawning

```mermaid
graph TD
    A1[A1: Orchestrator<br/>Can Spawn: Yes]

    A1 -->|Spawns| A2[A2: Analysis<br/>Can Spawn: No]
    A1 -->|Spawns| A4[A4: Engineering<br/>Can Spawn: No]
    A1 -->|Spawns| A3a[A3a: CV Specialist<br/>Can Spawn: Yes]
    A1 -->|Spawns| A3b[A3b: NLP Specialist<br/>Can Spawn: Yes]
    A1 -->|Spawns| A3c[A3c: Tabular Specialist<br/>Can Spawn: Yes]
    A1 -->|Spawns| A3d[A3d: Audio Specialist<br/>Can Spawn: Yes]
    A1 -->|Spawns| A5[A5: Ensemble<br/>Can Spawn: Yes]
    A1 -->|Spawns| A6[A6: Validation<br/>Can Spawn: No]
    A1 -->|Always Active| A7[A7: Resource Manager<br/>Can Spawn: No]

    A3a -->|Spawns if needed| HPO1[HPO Sub-Agent]
    A3b -->|Spawns if needed| HPO2[HPO Sub-Agent]
    A3c -->|Spawns if needed| HPO3[HPO Sub-Agent]
    A3d -->|Spawns if needed| HPO4[HPO Sub-Agent]

    A5 -->|Spawns if needed| WO[Weight Optimizer<br/>Sub-Agent]

    style A1 fill:#e1f5ff
    style A7 fill:#fff3cd
    style HPO1 fill:#d4edda
    style HPO2 fill:#d4edda
    style HPO3 fill:#d4edda
    style HPO4 fill:#d4edda
    style WO fill:#d4edda
```

## 3. Plug Architecture

```mermaid
graph LR
    subgraph "Plugs (Pj)"
        P1[P1: Data Loading]
        P2[P2: EDA]
        P3[P3: Preprocessing]
        P4[P4: Feature Eng]
        P5[P5: Model Training]
        P6[P6: HPO]
        P7[P7: Cross-Validation]
        P8[P8: Ensemble]
        P9[P9: Submission Gen]
        P10[P10: Validation]
        P11[P11: Metrics]
        P12[P12: Logging]
    end

    subgraph "Agents"
        A2[A2: Analysis]
        A3[A3: Domain<br/>Specialists]
        A4[A4: Engineering]
        A5[A5: Ensemble]
        A6[A6: Validation]
    end

    P1 --> A2
    P1 --> A3
    P1 --> A4

    P2 --> A2

    P3 --> A4
    P3 --> A3

    P4 --> A4

    P5 --> A3
    P6 --> A3
    P7 --> A3

    P8 --> A5
    P9 --> A5
    P9 --> A6

    P10 --> A6

    P11 --> A3
    P11 --> A5
    P11 --> A6

    P12 --> A2
    P12 --> A3
    P12 --> A4
    P12 --> A5
    P12 --> A6
```

## 4. Data Flow Pipeline

```mermaid
flowchart TD
    Start([Competition Start]) --> Load[Load Data & Description<br/>via P1]

    Load --> Analysis[A2: Analysis Agent<br/>EDA via P2]
    Analysis --> AnalysisOut{Competition<br/>Type?}

    AnalysisOut -->|CV| SelectCV[Select A3a]
    AnalysisOut -->|NLP| SelectNLP[Select A3b]
    AnalysisOut -->|Tabular| SelectTab[Select A3c]
    AnalysisOut -->|Audio| SelectAud[Select A3d]

    SelectCV --> Prep
    SelectNLP --> Prep
    SelectTab --> Prep
    SelectAud --> Prep

    Prep[A4: Feature Engineering<br/>via P3, P4] --> CV[A7: Create CV Folds<br/>via P7]

    CV --> Train[A3: Model Training<br/>via P5]
    Train --> HPO{Time for<br/>HPO?}
    HPO -->|Yes| RunHPO[Run HPO via P6]
    HPO -->|No| CVScore
    RunHPO --> CVScore[Compute CV Score<br/>via P11]

    CVScore --> Multi{Multiple<br/>Models?}
    Multi -->|Yes, train more| Train
    Multi -->|Done| Ensemble

    Ensemble[A5: Ensemble Models<br/>via P8] --> GenSub[Generate Submission<br/>via P9]

    GenSub --> Validate[A6: Validate<br/>via P10]
    Validate --> Valid{Valid?}

    Valid -->|No| Fix[Fix Issues]
    Fix --> GenSub
    Valid -->|Yes| Submit[Save to<br/>/home/submission/]

    Submit --> End([Competition End])

    style Start fill:#d4edda
    style End fill:#d4edda
    style Analysis fill:#e1f5ff
    style Prep fill:#e1f5ff
    style Train fill:#fff3cd
    style Ensemble fill:#ffe5e5
    style Validate fill:#f8d7da
```

## 5. Communication Pattern

```mermaid
sequenceDiagram
    participant O as A1: Orchestrator
    participant R as A7: Resource Manager
    participant A as A2: Analysis
    participant E as A4: Engineering
    participant D as A3: Domain Specialist
    participant En as A5: Ensemble
    participant V as A6: Validation

    O->>R: Start monitoring
    activate R

    O->>A: REQUEST: Analyze competition
    activate A
    A->>A: Use P1, P2
    A->>O: RESPONSE: Analysis results
    deactivate A

    O->>E: REQUEST: Engineer features
    activate E
    E->>E: Use P3, P4
    E->>O: RESPONSE: Feature pipeline
    deactivate E

    O->>D: REQUEST: Train models
    activate D
    D->>D: Use P5, P6, P7

    R->>D: ALERT: 60% time elapsed
    D->>O: UPDATE: CV scores
    D->>O: RESPONSE: Model checkpoints
    deactivate D

    O->>En: REQUEST: Create ensemble
    activate En
    En->>En: Use P8, P9
    En->>O: RESPONSE: Submission file
    deactivate En

    O->>V: REQUEST: Validate submission
    activate V
    V->>V: Use P10
    V->>O: RESPONSE: Validation result
    deactivate V

    R->>O: ALERT: 95% time elapsed
    O->>O: Finalize submission
    deactivate R
```

## 6. State Management

```mermaid
graph TD
    subgraph "Shared State Store"
        Global[Global State<br/>Owner: A1]

        subgraph "Agent States"
            S1[A1 State]
            S2[A2 State]
            S3[A3 State]
            S4[A4 State]
            S5[A5 State]
            S6[A6 State]
            S7[A7 State]
        end

        subgraph "Artifacts"
            Data[Data Artifacts]
            Models[Model Checkpoints]
            Preds[Predictions]
            Viz[Visualizations]
        end

        subgraph "History"
            StateHist[State Changes]
            MsgHist[Messages]
            ResHist[Resource Usage]
        end
    end

    A1Agent[A1: Orchestrator] -->|Write| S1
    A1Agent -->|Write| Global
    A1Agent -->|Read| S2
    A1Agent -->|Read| S3

    A2Agent[A2: Analysis] -->|Write| S2
    A2Agent -->|Write| Viz
    A2Agent -->|Read| Data

    A3Agent[A3: Domain] -->|Write| S3
    A3Agent -->|Write| Models
    A3Agent -->|Write| Preds

    A7Agent[A7: Resource Mgr] -->|Write| S7
    A7Agent -->|Write| ResHist
    A7Agent -->|Read All| Global

    style Global fill:#ffe5e5
    style StateHist fill:#d4edda
    style MsgHist fill:#d4edda
    style ResHist fill:#d4edda
```

## 7. Time Budget Allocation

```mermaid
gantt
    title Multi-Agent System Time Budget (24 hours = 100%)
    dateFormat X
    axisFormat %s

    section Phase 1
    Understanding (10%) :p1, 0, 10

    section Phase 2
    Preparation (15%) :p2, 10, 15

    section Phase 3
    Quick Baseline (9%) :p3a, 25, 9
    Advanced Modeling (42%) :p3b, 34, 42
    Final Models (9%) :p3c, 76, 9

    section Phase 4
    Ensembling (10%) :p4, 85, 10

    section Phase 5
    Submission (5%) :p5, 95, 5

    section Continuous
    Resource Monitoring :crit, 0, 100
    Validation :crit, 0, 100
```

## 8. Agent State Machines

### Orchestrator State Machine

```mermaid
stateDiagram-v2
    [*] --> Initialize
    Initialize --> Understanding
    Understanding --> Preparation
    Preparation --> Modeling
    Modeling --> Modeling: Multiple models
    Modeling --> Ensembling
    Ensembling --> Validation
    Validation --> Submission: Valid
    Validation --> Ensembling: Invalid
    Submission --> [*]

    Understanding --> Error: Analysis fails
    Preparation --> Error: Engineering fails
    Modeling --> Error: Training fails
    Ensembling --> Error: Ensemble fails
    Error --> Fallback
    Fallback --> Submission: Best effort
```

### Domain Specialist State Machine

```mermaid
stateDiagram-v2
    [*] --> Idle
    Idle --> LoadData: REQUEST received
    LoadData --> SelectModel
    SelectModel --> QuickBaseline
    QuickBaseline --> HPO: Time permits
    QuickBaseline --> FullTraining: Skip HPO
    HPO --> FullTraining
    FullTraining --> CrossValidation
    CrossValidation --> MoreModels: Train diverse models
    CrossValidation --> GeneratePredictions: Done
    MoreModels --> SelectModel
    GeneratePredictions --> Complete
    Complete --> [*]

    LoadData --> Error: Data load fails
    SelectModel --> Error: Config error
    FullTraining --> Error: OOM/timeout
    Error --> Retry: Attempt recovery
    Retry --> LoadData: Retry with smaller config
    Error --> Fallback: Max retries
    Fallback --> Complete: Return best effort
```

## 9. Plug Interaction Matrix

```mermaid
graph TD
    subgraph "Data Plugs"
        P1[P1: Loading]
        P2[P2: EDA]
        P3[P3: Preprocessing]
    end

    subgraph "Engineering Plugs"
        P4[P4: Feature Eng]
        P7[P7: CV]
    end

    subgraph "Training Plugs"
        P5[P5: Training]
        P6[P6: HPO]
        P11[P11: Metrics]
    end

    subgraph "Output Plugs"
        P8[P8: Ensemble]
        P9[P9: Submission]
        P10[P10: Validation]
    end

    subgraph "Support Plugs"
        P12[P12: Logging]
    end

    P1 --> P2
    P1 --> P3
    P2 --> P4
    P3 --> P4
    P4 --> P7
    P7 --> P5
    P5 --> P11
    P6 --> P5
    P11 --> P8
    P5 --> P8
    P8 --> P9
    P9 --> P10

    P12 -.Monitors.-> P1
    P12 -.Monitors.-> P2
    P12 -.Monitors.-> P5
    P12 -.Monitors.-> P8

    style P12 fill:#f8d7da
```

## 10. Error Handling and Recovery

```mermaid
flowchart TD
    Start[Task Execution] --> Monitor{Monitor<br/>Status}

    Monitor -->|Success| Complete[Task Complete]
    Monitor -->|Error| Classify{Error<br/>Type?}

    Classify -->|Transient| Retry{Retry<br/>Count?}
    Retry -->|< 3| Wait[Wait with<br/>Exponential Backoff]
    Wait --> Start
    Retry -->|>= 3| Fallback

    Classify -->|Resource| Degrade[Degrade Config<br/>Smaller batch, simpler model]
    Degrade --> CanContinue{Can<br/>Continue?}
    CanContinue -->|Yes| Start
    CanContinue -->|No| Fallback

    Classify -->|Fatal| Fallback[Activate Fallback<br/>Strategy]

    Fallback --> FallbackType{Fallback<br/>Available?}
    FallbackType -->|Yes| SimplerApproach[Use Simpler<br/>Approach]
    SimplerApproach --> Complete

    FallbackType -->|No| Report[Report Error<br/>Best Effort Submission]
    Report --> Complete

    Complete --> End([End])

    style Start fill:#d4edda
    style Fallback fill:#f8d7da
    style Complete fill:#d4edda
    style End fill:#d4edda
```

## 11. Resource Monitoring Flow

```mermaid
flowchart LR
    subgraph "A7: Resource Manager"
        Monitor[Monitor Every 30s]
        Check{Resource<br/>Status?}

        Monitor --> Check

        Check -->|Normal| Continue[Continue Monitoring]
        Continue --> Monitor

        Check -->|Warning| Warn[Send WARNING<br/>to Orchestrator]
        Warn --> Suggest[Suggest<br/>Optimizations]
        Suggest --> Monitor

        Check -->|Critical| Alert[Send ALERT<br/>to Orchestrator]
        Alert --> Action{Take<br/>Action?}

        Action -->|Kill Process| Kill[Terminate<br/>Runaway Process]
        Action -->|Reduce Usage| Reduce[Reduce Batch Size<br/>Clear Cache]

        Kill --> Monitor
        Reduce --> Monitor
    end

    subgraph "Monitored Resources"
        CPU[CPU Usage]
        GPU[GPU Usage]
        MEM[Memory Usage]
        DISK[Disk Usage]
        TIME[Time Remaining]
    end

    CPU --> Check
    GPU --> Check
    MEM --> Check
    DISK --> Check
    TIME --> Check

    style Alert fill:#f8d7da
    style Kill fill:#f8d7da
```

## 12. Ensemble Strategies Decision Tree

```mermaid
graph TD
    Start[Multiple Models<br/>Trained] --> Count{How many<br/>models?}

    Count -->|2-3| Diverse{Models<br/>Diverse?}
    Count -->|4-10| Analyze[Analyze Model<br/>Predictions]
    Count -->|>10| SelectTop[Select Top 10<br/>by CV Score]

    SelectTop --> Analyze

    Diverse -->|Yes| Average[Weighted Average]
    Diverse -->|No| Best[Use Best Model]

    Analyze --> Correlation{Prediction<br/>Correlation?}

    Correlation -->|Low < 0.7| Stack[Stacking<br/>with Meta-Learner]
    Correlation -->|Medium 0.7-0.9| Blend[Blending<br/>on Holdout]
    Correlation -->|High > 0.9| Vote[Voting<br/>or Average]

    Stack --> Optimize
    Blend --> Optimize
    Vote --> Optimize
    Average --> Optimize

    Optimize[Optimize Weights] --> Validate{Ensemble<br/>Better?}

    Validate -->|Yes| Use[Use Ensemble]
    Validate -->|No| Best

    Use --> Submit[Generate<br/>Submission]
    Best --> Submit

    style Start fill:#d4edda
    style Submit fill:#d4edda
    style Best fill:#fff3cd
```

---

## Diagram Conventions

- **Blue boxes**: Primary agents and components
- **Yellow boxes**: Resource management and monitoring
- **Green boxes**: Sub-agents and spawned processes
- **Red/Pink boxes**: Validation, errors, and critical paths
- **Dotted lines**: Monitoring or optional connections
- **Solid lines**: Data flow or command flow
- **Dashed arrows**: Bidirectional communication

---

*These diagrams complement the main design document: `multi_agent_system_design.md`*
