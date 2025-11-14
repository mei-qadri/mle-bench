"""
Data Exploration Plugin

Provides capabilities for exploring and analyzing datasets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any

from agents.planning_system.core.plugin import (
    Plugin,
    PluginFunctionality,
    PluginConfig,
    PluginConstraints,
    Action,
)


def read_csv_file(filepath: str, nrows: int = None) -> Dict[str, Any]:
    """Read CSV file and return summary"""
    df = pd.read_csv(filepath, nrows=nrows)
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "head": df.head().to_dict(),
    }


def describe_data(filepath: str) -> Dict[str, Any]:
    """Generate statistical summary of dataset"""
    df = pd.read_csv(filepath)

    summary = {
        "shape": df.shape,
        "columns": list(df.columns),
        "numeric_summary": df.describe().to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }

    return summary


def detect_missing_values(filepath: str) -> Dict[str, Any]:
    """Detect and analyze missing values"""
    df = pd.read_csv(filepath)

    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100

    return {
        "missing_counts": missing.to_dict(),
        "missing_percentages": missing_pct.to_dict(),
        "total_missing": int(missing.sum()),
        "columns_with_missing": list(missing[missing > 0].index),
    }


def analyze_column_types(filepath: str) -> Dict[str, Any]:
    """Analyze column data types and suggest transformations"""
    df = pd.read_csv(filepath)

    analysis = {}

    for col in df.columns:
        col_info = {
            "dtype": str(df[col].dtype),
            "nunique": int(df[col].nunique()),
            "missing": int(df[col].isnull().sum()),
        }

        # Determine if categorical
        if df[col].dtype == 'object':
            col_info["type"] = "categorical"
            col_info["unique_values"] = list(df[col].unique()[:10])  # First 10
        elif df[col].nunique() < 20 and df[col].dtype in ['int64', 'float64']:
            col_info["type"] = "categorical_numeric"
            col_info["unique_values"] = list(df[col].unique())
        else:
            col_info["type"] = "numeric"
            col_info["min"] = float(df[col].min()) if df[col].notna().any() else None
            col_info["max"] = float(df[col].max()) if df[col].notna().any() else None
            col_info["mean"] = float(df[col].mean()) if df[col].notna().any() else None

        analysis[col] = col_info

    return analysis


def compute_correlations(filepath: str) -> Dict[str, Any]:
    """Compute correlation matrix for numeric columns"""
    df = pd.read_csv(filepath)

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) == 0:
        return {"error": "No numeric columns found"}

    corr_matrix = df[numeric_cols].corr()

    return {
        "correlation_matrix": corr_matrix.to_dict(),
        "high_correlations": _find_high_correlations(corr_matrix),
    }


def _find_high_correlations(corr_matrix: pd.DataFrame, threshold: float = 0.8) -> list:
    """Find pairs of highly correlated features"""
    high_corr = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > threshold:
                high_corr.append({
                    "feature1": corr_matrix.columns[i],
                    "feature2": corr_matrix.columns[j],
                    "correlation": float(corr_val),
                })

    return high_corr


def create_data_exploration_plugin() -> Plugin:
    """Create a data exploration plugin instance"""

    # Define actions
    actions = {
        "read_csv": Action(
            action_name="read_csv",
            description="Read CSV file and get basic information",
            parameters={
                "filepath": {"type": "string", "required": True},
                "nrows": {"type": "integer", "required": False},
            },
            returns={"type": "dict"},
            handler=read_csv_file,
            estimated_time=5.0,
        ),
        "describe_data": Action(
            action_name="describe_data",
            description="Generate statistical summary of dataset",
            parameters={"filepath": {"type": "string", "required": True}},
            returns={"type": "dict"},
            handler=describe_data,
            estimated_time=10.0,
        ),
        "detect_missing": Action(
            action_name="detect_missing",
            description="Detect and analyze missing values",
            parameters={"filepath": {"type": "string", "required": True}},
            returns={"type": "dict"},
            handler=detect_missing_values,
            estimated_time=5.0,
        ),
        "analyze_columns": Action(
            action_name="analyze_columns",
            description="Analyze column types and characteristics",
            parameters={"filepath": {"type": "string", "required": True}},
            returns={"type": "dict"},
            handler=analyze_column_types,
            estimated_time=10.0,
        ),
        "compute_correlations": Action(
            action_name="compute_correlations",
            description="Compute correlation matrix for numeric features",
            parameters={"filepath": {"type": "string", "required": True}},
            returns={"type": "dict"},
            handler=compute_correlations,
            estimated_time=15.0,
        ),
    }

    # Create functionality
    functionality = PluginFunctionality(
        plugin_id="data_exploration",
        plugin_name="Data Exploration Plugin",
        description="Provides data exploration and analysis capabilities using pandas",
        actions=actions,
        required_packages=["pandas", "numpy"],
    )

    # Create config
    config = PluginConfig(
        timeout=300,  # 5 minutes
        max_retries=2,
        max_memory_mb=8192,  # 8GB for large datasets
        allowed_read_paths=["/home/data", "/home/workspace"],
        log_level="INFO",
    )

    # Create constraints
    constraints = PluginConstraints(
        requires_internet=False,
        requires_gpu=False,
        min_memory_mb=1024,
        max_input_size_mb=1000.0,  # 1GB max file size
        sandbox_mode=True,
    )

    return Plugin(
        functionality=functionality,
        config=config,
        constraints=constraints,
    )
