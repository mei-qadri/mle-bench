"""
Orchestrator Agent (A1) - Central Coordinator

Coordinates the entire multi-agent workflow for solving MLE-bench competitions.
"""

from typing import Any, Dict, Optional
from mle_bench_agents.core.agent import Agent
from mle_bench_agents.core.message import Message, MessageType


class OrchestratorAgent(Agent):
    """
    Orchestrator Agent responsible for:
    - Competition analysis and planning
    - Agent selection and spawning
    - Workflow coordination and scheduling
    - Resource allocation and time management
    - Final submission assembly
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, agent_type="orchestrator", **kwargs)

        # Workflow configuration
        self.time_allocations = self.config.get("time_allocations", {
            "understanding": 0.10,
            "preparation": 0.15,
            "modeling": 0.60,
            "ensembling": 0.10,
            "submission": 0.05
        })

    def process_message(self, message: Message) -> Optional[Message]:
        """Process incoming messages."""
        payload = message.payload
        action = payload.get("action")

        if action == "start_competition":
            # Start competition workflow
            result = self.orchestrate_competition(payload)
            return message.create_response(
                sender_id=self.agent_id,
                payload=result
            )

        elif message.message_type == MessageType.RESPONSE:
            # Handle agent responses
            return None  # Responses handled by workflow

        elif message.message_type == MessageType.ERROR:
            # Handle errors
            self.handle_agent_error(message)
            return None

        elif message.message_type == MessageType.ALERT:
            # Handle alerts from Resource Manager
            self.handle_alert(message)
            return None

        return None

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute orchestration for a competition."""
        return self.orchestrate_competition(context)

    def orchestrate_competition(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main orchestration method for competition workflow.

        Phases:
        1. Understanding (10%)
        2. Preparation (15%)
        3. Modeling (60%)
        4. Ensembling (10%)
        5. Submission (5%)
        """
        competition_id = context.get("competition_id")
        time_budget = context.get("time_budget", 86400)

        self.log_info(f"Starting orchestration for {competition_id}")

        # Initialize global state
        self.update_global_state({
            "competition_id": competition_id,
            "time_budget_total": time_budget,
            "current_phase": "understanding"
        })

        try:
            # Phase 1: Understanding
            self.log_info("=== PHASE 1: UNDERSTANDING ===")
            analysis_result = self.execute_understanding_phase(context)

            # Phase 2: Preparation
            self.log_info("=== PHASE 2: PREPARATION ===")
            engineering_result = self.execute_preparation_phase(analysis_result)

            # Phase 3: Modeling
            self.log_info("=== PHASE 3: MODELING ===")
            modeling_result = self.execute_modeling_phase(engineering_result)

            # Phase 4: Ensembling
            self.log_info("=== PHASE 4: ENSEMBLING ===")
            ensemble_result = self.execute_ensembling_phase(modeling_result)

            # Phase 5: Submission
            self.log_info("=== PHASE 5: SUBMISSION ===")
            submission_result = self.execute_submission_phase(ensemble_result)

            return {
                "status": "success",
                "competition_id": competition_id,
                "submission_path": submission_result.get("submission_path"),
                "best_cv_score": self.get_global_state().best_cv_score
            }

        except Exception as e:
            self.log_error(f"Orchestration failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    def execute_understanding_phase(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 1: Understanding - Invoke Analysis Agent."""
        phase_time = int(self.get_global_state().time_budget_total * self.time_allocations["understanding"])

        result = self.invoke_agent(
            target_agent_id="A2_analysis",
            action="analyze_competition",
            timeout=phase_time,
            **context
        )

        if result and result.get("status") == "success":
            # Update global state
            self.update_global_state({
                "competition_type": result.get("competition_analysis", {}).get("domain"),
                "complexity": result.get("complexity_assessment", {}).get("overall"),
                "metric_name": result.get("competition_analysis", {}).get("metric_name"),
                "data_summary": result.get("dataset_summary")
            })

        return result or {}

    def execute_preparation_phase(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 2: Preparation - Invoke Engineering Agent."""
        phase_time = int(self.get_global_state().time_budget_total * self.time_allocations["preparation"])

        result = self.invoke_agent(
            target_agent_id="A4_engineering",
            action="engineer_features",
            timeout=phase_time,
            analysis_result=analysis_result
        )

        return result or {}

    def execute_modeling_phase(self, engineering_result: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 3: Modeling - Invoke Domain Specialists."""
        phase_time = int(self.get_global_state().time_budget_total * self.time_allocations["modeling"])

        # Select domain specialist based on competition type
        competition_type = self.get_global_state().competition_type
        specialist_map = {
            "CV": "A3a_cv_specialist",
            "NLP": "A3b_nlp_specialist",
            "tabular": "A3c_tabular_specialist",
            "audio": "A3d_audio_specialist"
        }

        specialist_id = specialist_map.get(competition_type, "A3c_tabular_specialist")

        result = self.invoke_agent(
            target_agent_id=specialist_id,
            action="train_models",
            timeout=phase_time,
            phase="advanced",
            engineering_result=engineering_result
        )

        return result or {}

    def execute_ensembling_phase(self, modeling_result: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 4: Ensembling - Invoke Ensemble Agent."""
        phase_time = int(self.get_global_state().time_budget_total * self.time_allocations["ensembling"])

        result = self.invoke_agent(
            target_agent_id="A5_ensemble",
            action="create_ensemble",
            timeout=phase_time,
            modeling_result=modeling_result
        )

        return result or {}

    def execute_submission_phase(self, ensemble_result: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 5: Submission - Invoke Validation Agent."""
        phase_time = int(self.get_global_state().time_budget_total * self.time_allocations["submission"])

        result = self.invoke_agent(
            target_agent_id="A6_validation",
            action="validate_submission",
            timeout=phase_time,
            ensemble_result=ensemble_result
        )

        return result or {}

    def handle_agent_error(self, error_message: Message) -> None:
        """Handle error from an agent."""
        error_payload = error_message.payload
        agent_id = error_message.sender_id
        error_type = error_payload.get("error_type")

        self.log_error(f"Error from {agent_id}: {error_type} - {error_payload.get('error_message')}")

        # Activate fallback strategy based on error type
        # (To be implemented)

    def handle_alert(self, alert_message: Message) -> None:
        """Handle alert from Resource Manager."""
        alert_payload = alert_message.payload
        alert_type = alert_payload.get("alert_type")

        self.log_warning(f"Alert: {alert_type} - {alert_payload.get('message', '')}")

        # Take action based on alert type
        if alert_type == "time_critical":
            self.log_warning("Time critical - moving to submission phase")
            # Force phase transition

        elif alert_type == "high_memory":
            self.log_warning("High memory usage - consider reducing batch size")
