"""
Main runner for MLE-bench Multi-Agent System.

Coordinates all agents to solve a competition end-to-end.
"""

import time
from pathlib import Path
from typing import Dict, Any

from mle_bench_agents.core.queue import MessageQueue
from mle_bench_agents.core.state import StateManager
from mle_bench_agents.agents.orchestrator import OrchestratorAgent
from mle_bench_agents.agents.analysis import AnalysisAgent
from mle_bench_agents.agents.engineering import EngineeringAgent
from mle_bench_agents.agents.specialists.tabular import TabularSpecialistAgent
from mle_bench_agents.agents.ensemble import EnsembleAgent
from mle_bench_agents.agents.validation import ValidationAgent
from mle_bench_agents.agents.resource_manager import ResourceManagerAgent
from mle_bench_agents.utils.logger import setup_logger


class CompetitionRunner:
    """
    Main runner for competition solving.

    Initializes all agents and coordinates the workflow.
    """

    def __init__(
        self,
        log_dir: Path = Path("/home/logs"),
        log_level: str = "INFO"
    ):
        """
        Initialize runner.

        Args:
            log_dir: Directory for logs
            log_level: Logging level
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.logger = setup_logger(
            "CompetitionRunner",
            log_file=self.log_dir / "runner.log",
            level=log_level
        )

        # Create infrastructure
        self.message_queue = MessageQueue()
        self.state_manager = StateManager(
            state_file=self.log_dir / "state.json",
            enable_versioning=True
        )

        # Create agents
        self.agents = self._create_agents()

        self.logger.info("Competition Runner initialized")

    def _create_agents(self) -> Dict[str, Any]:
        """Create all agents."""
        agents = {}

        # A1: Orchestrator
        agents["A1_orchestrator"] = OrchestratorAgent(
            agent_id="A1_orchestrator",
            message_queue=self.message_queue,
            state_manager=self.state_manager,
            config={"can_spawn": True}
        )
        agents["A1_orchestrator"].logger = setup_logger("A1_Orchestrator", level="INFO")

        # A2: Analysis
        agents["A2_analysis"] = AnalysisAgent(
            agent_id="A2_analysis",
            message_queue=self.message_queue,
            state_manager=self.state_manager,
            config={}
        )
        agents["A2_analysis"].logger = setup_logger("A2_Analysis", level="INFO")

        # A4: Engineering
        agents["A4_engineering"] = EngineeringAgent(
            agent_id="A4_engineering",
            message_queue=self.message_queue,
            state_manager=self.state_manager,
            config={}
        )
        agents["A4_engineering"].logger = setup_logger("A4_Engineering", level="INFO")

        # A3c: Tabular Specialist
        agents["A3c_tabular_specialist"] = TabularSpecialistAgent(
            agent_id="A3c_tabular_specialist",
            message_queue=self.message_queue,
            state_manager=self.state_manager,
            config={"can_spawn": True}
        )
        agents["A3c_tabular_specialist"].logger = setup_logger("A3c_TabularSpecialist", level="INFO")

        # A5: Ensemble
        agents["A5_ensemble"] = EnsembleAgent(
            agent_id="A5_ensemble",
            message_queue=self.message_queue,
            state_manager=self.state_manager,
            config={}
        )
        agents["A5_ensemble"].logger = setup_logger("A5_Ensemble", level="INFO")

        # A6: Validation
        agents["A6_validation"] = ValidationAgent(
            agent_id="A6_validation",
            message_queue=self.message_queue,
            state_manager=self.state_manager,
            config={}
        )
        agents["A6_validation"].logger = setup_logger("A6_Validation", level="INFO")

        # A7: Resource Manager
        agents["A7_resource_manager"] = ResourceManagerAgent(
            agent_id="A7_resource_manager",
            message_queue=self.message_queue,
            state_manager=self.state_manager,
            config={"check_interval": 30}
        )
        agents["A7_resource_manager"].logger = setup_logger("A7_ResourceManager", level="INFO")

        self.logger.info(f"Created {len(agents)} agents")
        return agents

    def run_competition(
        self,
        competition_id: str,
        data_path: Path = Path("/home/data"),
        submission_path: Path = Path("/home/submission"),
        time_budget: int = 3600
    ) -> Dict[str, Any]:
        """
        Run competition end-to-end.

        Args:
            competition_id: Competition identifier
            data_path: Path to competition data
            submission_path: Path for submission output
            time_budget: Time budget in seconds

        Returns:
            Results dictionary
        """
        self.logger.info("="*80)
        self.logger.info(f"Starting Competition: {competition_id}")
        self.logger.info(f"Data Path: {data_path}")
        self.logger.info(f"Time Budget: {time_budget}s ({time_budget/3600:.1f}h)")
        self.logger.info("="*80)

        start_time = time.time()

        try:
            # Start resource monitoring
            self.logger.info("Starting resource monitoring...")
            self.agents["A7_resource_manager"].start_monitoring()

            # Prepare context
            context = {
                "competition_id": competition_id,
                "data_path": str(data_path),
                "submission_path": str(submission_path),
                "time_budget": time_budget
            }

            # Run orchestration
            self.logger.info("Starting orchestration...")
            orchestrator = self.agents["A1_orchestrator"]

            # Execute through manual orchestration
            result = self._run_orchestration(context)

            # Stop resource monitoring
            self.agents["A7_resource_manager"].stop_monitoring()

            elapsed_time = time.time() - start_time

            self.logger.info("="*80)
            self.logger.info(f"Competition Complete!")
            self.logger.info(f"Status: {result.get('status')}")
            self.logger.info(f"Time Elapsed: {elapsed_time:.1f}s ({elapsed_time/60:.1f}m)")
            if result.get("status") == "success":
                self.logger.info(f"Submission: {result.get('submission_path')}")
                self.logger.info(f"Best CV Score: {result.get('best_cv_score', 'N/A')}")
            self.logger.info("="*80)

            return result

        except Exception as e:
            self.logger.error(f"Competition failed: {e}")
            import traceback
            traceback.print_exc()

            return {
                "status": "error",
                "error": str(e),
                "elapsed_time": time.time() - start_time
            }

    def _run_orchestration(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run orchestration manually."""
        try:
            # Phase 1: Analysis
            self.logger.info("\n" + "="*80)
            self.logger.info("PHASE 1: UNDERSTANDING")
            self.logger.info("="*80)

            analysis_agent = self.agents["A2_analysis"]
            analysis_result = analysis_agent.execute(context)

            if analysis_result.get("status") != "success":
                self.logger.error("Analysis phase failed")
                return analysis_result

            self.logger.info(f"Analysis complete: {analysis_result.get('competition_analysis', {}).get('domain')} task")

            # Phase 2: Engineering
            self.logger.info("\n" + "="*80)
            self.logger.info("PHASE 2: PREPARATION")
            self.logger.info("="*80)

            engineering_agent = self.agents["A4_engineering"]
            engineering_context = {**context, "analysis_result": analysis_result}
            engineering_result = engineering_agent.execute(engineering_context)

            if engineering_result.get("status") != "success":
                self.logger.error("Engineering phase failed")
                return engineering_result

            self.logger.info(f"Engineering complete: {engineering_result.get('feature_summary', {}).get('total_features')} features")

            # Phase 3: Modeling
            self.logger.info("\n" + "="*80)
            self.logger.info("PHASE 3: MODELING")
            self.logger.info("="*80)

            modeling_agent = self.agents["A3c_tabular_specialist"]
            modeling_context = {
                **context,
                "analysis_result": analysis_result,
                "engineering_result": engineering_result,
                "phase": "advanced"
            }
            modeling_result = modeling_agent.execute(modeling_context)

            if modeling_result.get("status") != "success":
                self.logger.error("Modeling phase failed")
                return modeling_result

            self.logger.info(f"Modeling complete: {len(modeling_result.get('models_trained', []))} models trained")

            # Phase 4: Ensemble
            self.logger.info("\n" + "="*80)
            self.logger.info("PHASE 4: ENSEMBLING")
            self.logger.info("="*80)

            ensemble_agent = self.agents["A5_ensemble"]
            ensemble_context = {**context, "modeling_result": modeling_result}
            ensemble_result = ensemble_agent.execute(ensemble_context)

            if ensemble_result.get("status") != "success":
                self.logger.error("Ensemble phase failed")
                return ensemble_result

            self.logger.info(f"Ensemble complete: {ensemble_result.get('ensemble_strategy')}")

            # Phase 5: Validation
            self.logger.info("\n" + "="*80)
            self.logger.info("PHASE 5: SUBMISSION")
            self.logger.info("="*80)

            validation_agent = self.agents["A6_validation"]
            validation_context = {
                **context,
                "ensemble_result": ensemble_result,
                "output_path": Path(context["submission_path"]) / "submission.csv"
            }
            validation_result = validation_agent.execute(validation_context)

            if validation_result.get("status") != "success":
                self.logger.error("Validation phase failed")
                return validation_result

            self.logger.info(f"Validation complete: {validation_result.get('submission_path')}")

            # Return success
            return {
                "status": "success",
                "competition_id": context["competition_id"],
                "submission_path": validation_result.get("submission_path"),
                "best_cv_score": modeling_result.get("best_model", {}).get("cv_score"),
                "n_models": len(modeling_result.get("models_trained", [])),
                "ensemble_used": ensemble_result.get("ensemble_used", False)
            }

        except Exception as e:
            self.logger.error(f"Orchestration failed: {e}")
            import traceback
            traceback.print_exc()
            return {"status": "error", "error": str(e)}


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="MLE-bench Multi-Agent System Runner")
    parser.add_argument("--competition-id", type=str, default="test-competition", help="Competition ID")
    parser.add_argument("--data-path", type=str, default="/home/data", help="Path to competition data")
    parser.add_argument("--submission-path", type=str, default="/home/submission", help="Path for submission output")
    parser.add_argument("--time-budget", type=int, default=3600, help="Time budget in seconds")
    parser.add_argument("--log-dir", type=str, default="/home/logs", help="Log directory")
    parser.add_argument("--log-level", type=str, default="INFO", help="Log level")

    args = parser.parse_args()

    # Create runner
    runner = CompetitionRunner(
        log_dir=Path(args.log_dir),
        log_level=args.log_level
    )

    # Run competition
    result = runner.run_competition(
        competition_id=args.competition_id,
        data_path=Path(args.data_path),
        submission_path=Path(args.submission_path),
        time_budget=args.time_budget
    )

    # Print summary
    print("\n" + "="*80)
    print("COMPETITION SUMMARY")
    print("="*80)
    print(f"Status: {result.get('status')}")
    if result.get('status') == 'success':
        print(f"Submission: {result.get('submission_path')}")
        print(f"Best CV Score: {result.get('best_cv_score', 'N/A')}")
        print(f"Models Trained: {result.get('n_models', 0)}")
        print(f"Ensemble Used: {result.get('ensemble_used', False)}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
    print("="*80)

    return 0 if result.get("status") == "success" else 1


if __name__ == "__main__":
    exit(main())
