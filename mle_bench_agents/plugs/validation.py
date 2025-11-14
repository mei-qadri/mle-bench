"""P10: Validation Plug - Submission validation."""

import pandas as pd
from pathlib import Path
from typing import Any, Dict
from mle_bench_agents.plugs.base_plug import BasePlug


class ValidationPlug(BasePlug):
    """Validate submission format and quality."""

    def get_constraints(self) -> Dict[str, Any]:
        return {
            "max_validation_time": 60,
            "auto_fix": self.config.get("auto_fix", True),
            "strict_mode": self.config.get("strict_mode", False)
        }

    def execute(self, submission_df: pd.DataFrame, sample_submission_path: Path, **kwargs) -> Dict[str, Any]:
        """Validate submission."""
        sample_sub = pd.read_csv(sample_submission_path)

        errors = []
        warnings = []

        # Check shape
        if submission_df.shape != sample_sub.shape:
            errors.append(f"Shape mismatch: got {submission_df.shape}, expected {sample_sub.shape}")

        # Check columns
        if list(submission_df.columns) != list(sample_sub.columns):
            errors.append(f"Column mismatch")

        # Check for NaN
        nan_count = submission_df.isna().sum().sum()
        if nan_count > 0:
            if self.constraints["auto_fix"]:
                submission_df = submission_df.fillna(0)
                warnings.append(f"Fixed {nan_count} NaN values")
            else:
                errors.append(f"Found {nan_count} NaN values")

        is_valid = len(errors) == 0

        return {
            "is_valid": is_valid,
            "errors": errors,
            "warnings": warnings,
            "submission_df": submission_df
        }
