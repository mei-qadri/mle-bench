"""
ML Training Plugin

Provides capabilities for training and evaluating ML models.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import joblib
from pathlib import Path

from agents.planning_system.core.plugin import (
    Plugin,
    PluginFunctionality,
    PluginConfig,
    PluginConstraints,
    Action,
)


def train_sklearn_model(
    train_file: str,
    target_column: str,
    model_type: str = "random_forest",
    test_size: float = 0.2,
    output_path: str = "model.pkl",
) -> Dict[str, Any]:
    """Train a scikit-learn model"""
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

    # Load data
    df = pd.read_csv(train_file)

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Handle categorical variables (simple one-hot encoding)
    X = pd.get_dummies(X, drop_first=True)

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Select model
    if model_type == "random_forest_classifier":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        is_classification = True
    elif model_type == "random_forest_regressor":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        is_classification = False
    elif model_type == "logistic_regression":
        model = LogisticRegression(max_iter=1000, random_state=42)
        is_classification = True
    elif model_type == "linear_regression":
        model = LinearRegression()
        is_classification = False
    else:
        return {"success": False, "error": f"Unknown model type: {model_type}"}

    # Train model
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_val)

    if is_classification:
        metrics = {
            "accuracy": float(accuracy_score(y_val, y_pred)),
            "f1_score": float(f1_score(y_val, y_pred, average="weighted")),
        }
    else:
        metrics = {
            "mse": float(mean_squared_error(y_val, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_val, y_pred))),
            "r2": float(r2_score(y_val, y_pred)),
        }

    # Save model
    joblib.dump(model, output_path)

    return {
        "success": True,
        "model_path": output_path,
        "model_type": model_type,
        "metrics": metrics,
        "feature_columns": list(X.columns),
    }


def make_predictions(
    model_path: str,
    test_file: str,
    output_file: str = "predictions.csv",
) -> Dict[str, Any]:
    """Make predictions using a trained model"""

    # Load model
    model = joblib.load(model_path)

    # Load test data
    df = pd.read_csv(test_file)

    # Get feature columns (from training)
    # This is simplified - in practice, need to ensure same preprocessing
    X_test = df.select_dtypes(include=[np.number])

    # Make predictions
    predictions = model.predict(X_test)

    # Save predictions
    pred_df = pd.DataFrame({"predictions": predictions})
    if "id" in df.columns:
        pred_df.insert(0, "id", df["id"])

    pred_df.to_csv(output_file, index=False)

    return {
        "success": True,
        "output_file": output_file,
        "num_predictions": len(predictions),
    }


def evaluate_model(
    model_path: str,
    val_file: str,
    target_column: str,
) -> Dict[str, Any]:
    """Evaluate a trained model on validation data"""
    from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

    # Load model
    model = joblib.load(model_path)

    # Load validation data
    df = pd.read_csv(val_file)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Handle categorical (same as training)
    X = pd.get_dummies(X, drop_first=True)

    # Predict
    y_pred = model.predict(X)

    # Determine if classification or regression
    is_classification = hasattr(model, "predict_proba")

    if is_classification:
        metrics = {
            "accuracy": float(accuracy_score(y, y_pred)),
            "f1_score": float(f1_score(y, y_pred, average="weighted")),
        }
    else:
        metrics = {
            "mse": float(mean_squared_error(y, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
            "r2": float(r2_score(y, y_pred)),
        }

    return {
        "success": True,
        "metrics": metrics,
    }


def create_ml_training_plugin() -> Plugin:
    """Create an ML training plugin instance"""

    actions = {
        "train_model": Action(
            action_name="train_model",
            description="Train a scikit-learn model",
            parameters={
                "train_file": {"type": "string", "required": True},
                "target_column": {"type": "string", "required": True},
                "model_type": {"type": "string", "required": False},
                "test_size": {"type": "float", "required": False},
                "output_path": {"type": "string", "required": False},
            },
            returns={"type": "dict"},
            handler=train_sklearn_model,
            estimated_time=60.0,
        ),
        "make_predictions": Action(
            action_name="make_predictions",
            description="Make predictions using trained model",
            parameters={
                "model_path": {"type": "string", "required": True},
                "test_file": {"type": "string", "required": True},
                "output_file": {"type": "string", "required": False},
            },
            returns={"type": "dict"},
            handler=make_predictions,
            estimated_time=30.0,
        ),
        "evaluate_model": Action(
            action_name="evaluate_model",
            description="Evaluate model on validation data",
            parameters={
                "model_path": {"type": "string", "required": True},
                "val_file": {"type": "string", "required": True},
                "target_column": {"type": "string", "required": True},
            },
            returns={"type": "dict"},
            handler=evaluate_model,
            estimated_time=20.0,
        ),
    }

    functionality = PluginFunctionality(
        plugin_id="ml_training",
        plugin_name="ML Training Plugin",
        description="Provides ML model training and evaluation capabilities",
        actions=actions,
        required_packages=["pandas", "numpy", "scikit-learn", "joblib"],
    )

    config = PluginConfig(
        timeout=7200,  # 2 hours for training
        max_retries=1,
        max_memory_mb=16384,  # 16GB
        allow_gpu=False,
        allowed_read_paths=["/home/data", "/home/workspace"],
        allowed_write_paths=["/home/workspace", "/home/submission"],
        log_level="INFO",
    )

    constraints = PluginConstraints(
        requires_internet=False,
        requires_gpu=False,
        min_memory_mb=2048,
        max_input_size_mb=2000.0,
        sandbox_mode=True,
    )

    return Plugin(
        functionality=functionality,
        config=config,
        constraints=constraints,
    )
