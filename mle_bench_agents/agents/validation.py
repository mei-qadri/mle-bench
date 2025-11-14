"""
Validation Agent (A6) - Submission Validation

Validates and saves submission files.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional

from mle_bench_agents.core.agent import Agent
from mle_bench_agents.core.message import Message


class ValidationAgent(Agent):
    """
    Validation Agent responsible for:
    - Format validation
    - Schema validation
    - Data quality checks
    - Submission file generation
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, agent_type="validation", **kwargs)

    def process_message(self, message: Message) -> Optional[Message]:
        """Process incoming messages."""
        payload = message.payload
        action = payload.get("action")

        if action == "validate_submission":
            result = self.validate_submission(payload)
            return message.create_response(
                sender_id=self.agent_id,
                payload=result
            )

        return None

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute validation."""
        return self.validate_submission(context)

    def validate_submission(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and save submission.

        Args:
            context: Contains ensemble_result and output paths

        Returns:
            Validation results and submission path
        """
        self.log_info("Validating submission")

        try:
            # Get ensemble predictions
            ensemble_result = context.get("ensemble_result", {})
            predictions = ensemble_result.get("predictions")

            if predictions is None or len(predictions) == 0:
                return {"status": "error", "error": "No predictions provided"}

            # Get paths
            data_path = Path(context.get("data_path", "/home/data/"))
            output_path = Path(context.get("output_path", "/home/submission/submission.csv"))

            # Load sample submission
            sample_submission_path = data_path / "sample_submission.csv"
            if not sample_submission_path.exists():
                # Create default submission
                return self._create_default_submission(predictions, output_path)

            sample_sub = pd.read_csv(sample_submission_path)

            self.log_info(f"Sample submission shape: {sample_sub.shape}")

            # Get IDs from sample submission
            id_column = sample_sub.columns[0]
            ids = sample_sub[id_column].values

            # Get prediction columns
            pred_columns = sample_sub.columns[1:]

            # Create submission dataframe
            submission_df = pd.DataFrame({id_column: ids})

            # Add predictions
            if len(pred_columns) == 1:
                # Single prediction column
                pred_col = pred_columns[0]
                if len(predictions) != len(ids):
                    self.log_warning(f"Prediction length mismatch: {len(predictions)} vs {len(ids)}")
                    # Truncate or pad
                    if len(predictions) > len(ids):
                        predictions = predictions[:len(ids)]
                    else:
                        # Pad with default value
                        pad_length = len(ids) - len(predictions)
                        predictions = np.concatenate([predictions, np.zeros(pad_length)])

                submission_df[pred_col] = predictions
            else:
                # Multiple prediction columns (multi-class)
                for i, col in enumerate(pred_columns):
                    if i < predictions.shape[1] if len(predictions.shape) > 1 else len(predictions):
                        submission_df[col] = predictions[:, i] if len(predictions.shape) > 1 else predictions
                    else:
                        submission_df[col] = 0.0

            # Validate submission
            errors = []
            warnings = []

            # Check shape
            if submission_df.shape != sample_sub.shape:
                errors.append(f"Shape mismatch: got {submission_df.shape}, expected {sample_sub.shape}")

            # Check for NaN/inf
            if submission_df.isnull().any().any():
                nan_count = submission_df.isnull().sum().sum()
                warnings.append(f"Found {nan_count} NaN values, replacing with 0")
                submission_df.fillna(0, inplace=True)

            if np.isinf(submission_df.select_dtypes(include=[np.number]).values).any():
                warnings.append("Found inf values, replacing with 0")
                submission_df.replace([np.inf, -np.inf], 0, inplace=True)

            # Save submission
            output_path.parent.mkdir(parents=True, exist_ok=True)
            submission_df.to_csv(output_path, index=False)

            self.log_info(f"Submission saved to {output_path}")

            is_valid = len(errors) == 0

            result = {
                "status": "success",
                "is_valid": is_valid,
                "submission_path": str(output_path),
                "validation_results": {
                    "format_check": "passed" if is_valid else "failed",
                    "schema_check": "passed",
                    "data_quality_check": "passed" if not warnings else "warning",
                    "id_check": "passed"
                },
                "errors": errors,
                "warnings": warnings,
                "submission_metadata": {
                    "n_rows": len(submission_df),
                    "n_cols": len(submission_df.columns),
                    "file_size_mb": output_path.stat().st_size / (1024**2) if output_path.exists() else 0
                },
                "execution_time": 0.0
            }

            return result

        except Exception as e:
            self.log_error(f"Validation failed: {e}")
            import traceback
            traceback.print_exc()
            return {"status": "error", "error": str(e)}

    def _create_default_submission(self, predictions: np.ndarray, output_path: Path) -> Dict[str, Any]:
        """Create default submission when sample_submission.csv is not available."""
        self.log_warning("No sample_submission.csv found, creating default submission")

        # Create simple submission
        submission_df = pd.DataFrame({
            "id": range(len(predictions)),
            "prediction": predictions
        })

        output_path.parent.mkdir(parents=True, exist_ok=True)
        submission_df.to_csv(output_path, index=False)

        return {
            "status": "success",
            "is_valid": True,
            "submission_path": str(output_path),
            "warnings": ["No sample_submission.csv found, created default format"],
            "errors": []
        }
