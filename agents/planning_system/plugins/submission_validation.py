"""
Submission Validation Plugin

Provides capabilities for validating and preparing submissions.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Any
import requests

from agents.planning_system.core.plugin import (
    Plugin,
    PluginFunctionality,
    PluginConfig,
    PluginConstraints,
    Action,
)


def validate_submission_format(
    submission_file: str,
    sample_submission_file: str,
) -> Dict[str, Any]:
    """Validate that submission matches expected format"""

    # Load files
    submission = pd.read_csv(submission_file)
    sample = pd.read_csv(sample_submission_file)

    errors = []
    warnings = []

    # Check columns match
    if list(submission.columns) != list(sample.columns):
        errors.append(
            f"Column mismatch. Expected {list(sample.columns)}, got {list(submission.columns)}"
        )

    # Check number of rows
    if len(submission) != len(sample):
        warnings.append(
            f"Row count mismatch. Expected {len(sample)}, got {len(submission)}"
        )

    # Check for missing values
    if submission.isnull().any().any():
        missing_cols = submission.columns[submission.isnull().any()].tolist()
        errors.append(f"Missing values found in columns: {missing_cols}")

    # Check data types
    for col in submission.columns:
        if col in sample.columns:
            if submission[col].dtype != sample[col].dtype:
                warnings.append(
                    f"Data type mismatch for column '{col}': expected {sample[col].dtype}, got {submission[col].dtype}"
                )

    is_valid = len(errors) == 0

    return {
        "valid": is_valid,
        "errors": errors,
        "warnings": warnings,
        "num_rows": len(submission),
        "num_columns": len(submission.columns),
    }


def check_submission_with_grader(
    submission_file: str,
    grader_url: str = "http://localhost:5000/validate",
) -> Dict[str, Any]:
    """Validate submission using grading server"""

    try:
        # Read submission
        with open(submission_file, "rb") as f:
            files = {"submission": f}
            response = requests.post(grader_url, files=files, timeout=30)

        if response.status_code == 200:
            result = response.json()
            return {
                "success": True,
                "valid": result.get("valid", False),
                "message": result.get("message", ""),
                "errors": result.get("errors", []),
            }
        else:
            return {
                "success": False,
                "error": f"Grader returned status {response.status_code}",
            }

    except requests.RequestException as e:
        return {
            "success": False,
            "error": f"Failed to contact grader: {str(e)}",
        }


def prepare_submission(
    predictions_file: str,
    sample_submission_file: str,
    output_file: str = "submission.csv",
    id_column: str = "id",
    prediction_column: str = "target",
) -> Dict[str, Any]:
    """Prepare final submission file from predictions"""

    # Load sample submission to get format
    sample = pd.read_csv(sample_submission_file)

    # Load predictions
    predictions = pd.read_csv(predictions_file)

    # Create submission DataFrame
    submission = pd.DataFrame()

    # Copy ID column
    if id_column in predictions.columns:
        submission[id_column] = predictions[id_column]
    elif id_column in sample.columns:
        submission[id_column] = sample[id_column]
    else:
        return {
            "success": False,
            "error": f"ID column '{id_column}' not found",
        }

    # Copy prediction column
    if prediction_column in predictions.columns:
        # Get target column name from sample
        target_col = [c for c in sample.columns if c != id_column][0]
        submission[target_col] = predictions[prediction_column]
    else:
        return {
            "success": False,
            "error": f"Prediction column '{prediction_column}' not found",
        }

    # Ensure column order matches sample
    submission = submission[sample.columns]

    # Save
    submission.to_csv(output_file, index=False)

    return {
        "success": True,
        "output_file": output_file,
        "num_rows": len(submission),
    }


def create_submission_validation_plugin() -> Plugin:
    """Create a submission validation plugin instance"""

    actions = {
        "validate_format": Action(
            action_name="validate_format",
            description="Validate submission format against sample",
            parameters={
                "submission_file": {"type": "string", "required": True},
                "sample_submission_file": {"type": "string", "required": True},
            },
            returns={"type": "dict"},
            handler=validate_submission_format,
            estimated_time=5.0,
        ),
        "check_with_grader": Action(
            action_name="check_with_grader",
            description="Validate submission using grading server",
            parameters={
                "submission_file": {"type": "string", "required": True},
                "grader_url": {"type": "string", "required": False},
            },
            returns={"type": "dict"},
            handler=check_submission_with_grader,
            estimated_time=10.0,
        ),
        "prepare_submission": Action(
            action_name="prepare_submission",
            description="Prepare final submission file from predictions",
            parameters={
                "predictions_file": {"type": "string", "required": True},
                "sample_submission_file": {"type": "string", "required": True},
                "output_file": {"type": "string", "required": False},
                "id_column": {"type": "string", "required": False},
                "prediction_column": {"type": "string", "required": False},
            },
            returns={"type": "dict"},
            handler=prepare_submission,
            estimated_time=5.0,
        ),
    }

    functionality = PluginFunctionality(
        plugin_id="submission_validation",
        plugin_name="Submission Validation Plugin",
        description="Validates and prepares competition submissions",
        actions=actions,
        required_packages=["pandas", "requests"],
    )

    config = PluginConfig(
        timeout=60,
        max_retries=2,
        max_memory_mb=4096,
        allowed_read_paths=["/home/data", "/home/workspace"],
        allowed_write_paths=["/home/submission"],
        log_level="INFO",
    )

    constraints = PluginConstraints(
        requires_internet=False,  # Grading server is local
        requires_gpu=False,
        min_memory_mb=512,
        max_input_size_mb=500.0,
        sandbox_mode=True,
        allowed_network_hosts=["localhost", "127.0.0.1"],
    )

    return Plugin(
        functionality=functionality,
        config=config,
        constraints=constraints,
    )
