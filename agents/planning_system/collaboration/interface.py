"""
Human Collaboration Interface

Enables interactive plan review and modification.
"""

from typing import Dict, List
from agents.planning_system.planning.planner import WorkflowPlan


class HumanCollaborationInterface:
    """Interface for human-agent collaboration"""

    def present_plan(self, workflow_plan: WorkflowPlan):
        """
        Present the workflow plan to the user

        Args:
            workflow_plan: The workflow plan to present
        """
        print("\n" + "="*60)
        print("WORKFLOW PLAN FOR REVIEW")
        print("="*60 + "\n")

        # Show visualization
        print(workflow_plan.visualize())

        print("\n" + "="*60)

    def gather_feedback(self) -> Dict:
        """
        Collect human feedback on the plan

        Returns:
            Dict with 'status' and optional 'modifications'
        """
        print("\nPlease review the plan above.")
        print("\nOptions:")
        print("  [a] Approve - Proceed with this plan")
        print("  [m] Modify - Request changes to the plan")
        print("  [r] Reject - Cancel execution")

        while True:
            response = input("\nYour choice (a/m/r): ").strip().lower()

            if response == "a" or response == "approve":
                print("\n✓ Plan approved!")
                return {"status": "approved", "modifications": []}

            elif response == "m" or response == "modify":
                modification = input("\nDescribe the modifications you'd like: ").strip()
                if modification:
                    return {"status": "needs_modification", "modifications": [modification]}
                else:
                    print("Please provide a description of the modifications.")
                    continue

            elif response == "r" or response == "reject":
                confirm = input("\nAre you sure you want to reject? (yes/no): ").strip().lower()
                if confirm == "yes":
                    return {"status": "rejected", "modifications": []}
                else:
                    continue

            else:
                print("Invalid response. Please enter 'a', 'm', or 'r'.")

    def iterate_on_plan(self, workflow_plan: WorkflowPlan, modifications: List[str]) -> WorkflowPlan:
        """
        Revise the plan based on human feedback

        This is a placeholder for future LLM-based plan modification.

        Args:
            workflow_plan: Current plan
            modifications: List of requested modifications

        Returns:
            Revised workflow plan
        """
        # TODO: Implement LLM-based plan modification
        print("\nPlan modification not yet implemented.")
        print("Requested modifications:")
        for mod in modifications:
            print(f"  - {mod}")

        return workflow_plan
