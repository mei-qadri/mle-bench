# MLE-bench Multi-Agent System Documentation

This directory contains the design documentation for the Multi-Agent System (MAS) for solving MLE-bench competitions.

## Documents

### 1. [Multi-Agent System Design](./multi_agent_system_design.md)
**Primary design specification document**

Comprehensive design specification including:
- System overview and design philosophy
- **Agent Definitions**: 7 core agents (A1-A7) with detailed specifications
  - Each agent defined as Ai = {Li, Ri, Si, Ci, Hi}
  - Language model configurations, roles, state structures, spawning capabilities
- **Plug Definitions**: 12 reusable plug components (P1-P12)
  - Each plug defined as Pj = {Fj, Cj, Uj}
  - Functionality, configuration, and constraints
- **Workflow**: 5-phase execution pipeline (Understanding → Preparation → Modeling → Ensembling → Submission)
- Communication protocols and state management
- Implementation roadmap (10-week plan)
- Success metrics and risk mitigation strategies

**Start here for understanding the complete system architecture.**

### 2. [Architecture Diagrams](./multi_agent_architecture_diagrams.md)
**Visual representation of the system**

Contains 12 Mermaid diagrams illustrating:
- System overview and component relationships
- Agent hierarchy and spawning capabilities
- Plug architecture and dependencies
- Data flow pipeline
- Communication patterns (sequence diagrams)
- State management structure
- Time budget allocation (Gantt chart)
- Agent state machines
- Error handling and recovery flows
- Resource monitoring workflows
- Ensemble strategy decision trees

**Use this for quick visual reference and understanding system interactions.**

## Quick Reference

### Agent Summary

| ID | Name | Model | Role |
|----|------|-------|------|
| A1 | Orchestrator | Sonnet 4.5 | Coordination & planning |
| A2 | Analysis | Sonnet 4.5 | EDA & task understanding |
| A3a-d | Domain Specialists | Sonnet 4.5 | CV/NLP/Tabular/Audio tasks |
| A4 | Engineering | Sonnet 4.5 | Feature engineering |
| A5 | Ensemble | Sonnet 4.5 | Model combination |
| A6 | Validation | Haiku 4 | Submission validation |
| A7 | Resource Manager | Haiku 4 | Resource monitoring |

### Plug Summary

| ID | Name | Primary Function |
|----|------|------------------|
| P1 | Data Loading | Load various data formats |
| P2 | EDA | Exploratory analysis |
| P3 | Preprocessing | Data cleaning & preparation |
| P4 | Feature Engineering | Create new features |
| P5 | Model Training | Train ML models |
| P6 | HPO | Hyperparameter tuning |
| P7 | Cross-Validation | CV strategies |
| P8 | Ensemble | Combine models |
| P9 | Submission | Format predictions |
| P10 | Validation | Check submission |
| P11 | Metrics | Compute scores |
| P12 | Logging | Track progress |

### Execution Phases

1. **Understanding (10%)**: Competition analysis, EDA
2. **Preparation (15%)**: Feature engineering, preprocessing
3. **Modeling (60%)**: Model training, hyperparameter tuning
4. **Ensembling (10%)**: Combine models, optimize weights
5. **Submission (5%)**: Validate and submit

## Design Principles

1. **Specialization**: Agents focused on specific domains or pipeline stages
2. **Modularity**: Plugs provide reusable, composable functionality
3. **Adaptability**: System adapts to competition type and complexity
4. **Resource Awareness**: Explicit resource management and budgeting
5. **Failure Recovery**: Built-in error handling and fallback strategies

## Target Performance

- **Medal Rate**: >35% (Gold: >10%, Silver: >20%, Bronze: >30%)
- **Completion Rate**: >95% valid submissions
- **Resource Efficiency**: >70% GPU utilization, <80% peak memory
- **Reliability**: <5% failure rate, >90% error recovery success

Comparison: Top current agent (Operand) achieves 48% medal rate with 16hr runtime.

## Implementation Status

**Current Status**: Design Phase Complete

**Next Steps**:
1. Phase I: Core Infrastructure (Weeks 1-2)
2. Phase II: Essential Agents (Weeks 3-4)
3. Phase III: Advanced Features (Weeks 5-6)
4. Phase IV: Optimization (Weeks 7-8)
5. Phase V: Evaluation (Weeks 9-10)

See [Implementation Roadmap](./multi_agent_system_design.md#implementation-roadmap) for details.

## For Developers

### Getting Started

1. **Read**: [System Design](./multi_agent_system_design.md) - Complete architecture
2. **Visualize**: [Architecture Diagrams](./multi_agent_architecture_diagrams.md) - System flows
3. **Implement**: Follow the roadmap in the design document

### Key Files for Implementation

- Agent definitions: Section 2 of design document
- Plug definitions: Section 3 of design document
- Communication protocol: Section 6 of design document
- State management: Diagram 6 in architecture diagrams

### Design Notation

**Agent**: Ai = {Li, Ri, Si, Ci, Hi}
- Li: Language model and configuration
- Ri: Role and responsibilities
- Si: State structure
- Ci: Can spawn children (boolean)
- Hi: History tracking configuration

**Plug**: Pj = {Fj, Cj, Uj}
- Fj: Functionality (actions)
- Cj: Configuration parameters
- Uj: Constraints and usage rules

## Contact & Feedback

For questions, suggestions, or contributions regarding this design:
- Review the design documents thoroughly
- Check the diagrams for visual clarification
- Refer to the implementation roadmap for phasing

---

**Version**: 1.0
**Date**: 2025-11-14
**Status**: Design Specification - Ready for Implementation
**Team**: Anthropic MLE-bench Multi-Agent System Team
