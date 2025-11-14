"""P3: Preprocessing Plug - Data cleaning and transformation."""

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Any, Dict, Tuple
from mle_bench_agents.plugs.base_plug import BasePlug


class PreprocessingPlug(BasePlug):
    """Data cleaning, scaling, encoding, splitting."""

    def get_constraints(self) -> Dict[str, Any]:
        return {
            "max_processing_time": 1800,  # 30 minutes
            "no_data_leakage": True
        }

    def execute(self, data: pd.DataFrame, target_col: str = None, **kwargs) -> Dict[str, Any]:
        """Preprocess data."""
        result = {}

        # Handle missing values
        missing_strategy = self.config.get("missing_strategy", "drop")
        if missing_strategy == "drop":
            data = data.dropna()
        elif missing_strategy == "mean":
            data = data.fillna(data.mean())

        # Split features and target
        if target_col and target_col in data.columns:
            X = data.drop(columns=[target_col])
            y = data[target_col]

            # Train-test split
            test_size = kwargs.get("test_size", 0.2)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            result["X_train"] = X_train
            result["X_test"] = X_test
            result["y_train"] = y_train
            result["y_test"] = y_test
        else:
            result["data"] = data

        return result
