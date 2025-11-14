"""
Engineering Agent (A4) - Feature Engineering and Preprocessing

Handles data preprocessing, feature engineering, and CV fold creation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.impute import SimpleImputer

from mle_bench_agents.core.agent import Agent
from mle_bench_agents.core.message import Message
from mle_bench_agents.plugs.data_loading import DataLoadingPlug
from mle_bench_agents.plugs.preprocessing import PreprocessingPlug


class EngineeringAgent(Agent):
    """
    Engineering Agent responsible for:
    - Data preprocessing
    - Feature engineering
    - Feature selection
    - CV fold creation
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, agent_type="engineering", **kwargs)

        self.data_loader = DataLoadingPlug()
        self.preprocessing_plug = PreprocessingPlug()

    def process_message(self, message: Message) -> Optional[Message]:
        """Process incoming messages."""
        payload = message.payload
        action = payload.get("action")

        if action == "engineer_features":
            result = self.engineer_features(payload)
            return message.create_response(
                sender_id=self.agent_id,
                payload=result
            )

        return None

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute feature engineering."""
        return self.engineer_features(context)

    def engineer_features(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess data and engineer features.

        Args:
            context: Contains analysis_result and data paths

        Returns:
            Engineered features and CV folds
        """
        self.log_info("Starting feature engineering")

        try:
            # Get analysis results
            analysis_result = context.get("analysis_result", {})
            data_path = Path(context.get("data_path", "/home/data/"))

            # Load data
            train_file = data_path / "train.csv"
            if not train_file.exists():
                train_file = data_path / "train.parquet"

            train_df = self.data_loader.execute(train_file)

            # Load test data
            test_file = data_path / "test.csv"
            if not test_file.exists():
                test_file = data_path / "test.parquet"

            if test_file.exists():
                test_df = self.data_loader.execute(test_file)
            else:
                test_df = None

            # Identify target column
            feature_analysis = analysis_result.get("feature_analysis", {})
            target_col = feature_analysis.get("target_column")

            # Validate and find target column
            if not target_col or target_col not in train_df.columns:
                self.log_info(f"Target column '{target_col}' not found, attempting to identify...")
                target_col = self._identify_target(train_df)

            # Double check target column exists
            if target_col not in train_df.columns:
                # Try case-insensitive match
                col_lower_map = {col.lower(): col for col in train_df.columns}
                if target_col.lower() in col_lower_map:
                    target_col = col_lower_map[target_col.lower()]
                    self.log_info(f"Found target column with case mismatch: {target_col}")
                else:
                    # Last resort: use the last column
                    target_col = train_df.columns[-1]
                    self.log_warning(f"Could not find target column, using last column: {target_col}")

            # Separate features and target
            try:
                X_train = train_df.drop(columns=[target_col])
                y_train = train_df[target_col]
            except KeyError as e:
                self.log_error(f"Failed to separate target column '{target_col}': {e}")
                self.log_info(f"Available columns: {train_df.columns.tolist()}")
                raise

            if test_df is not None:
                # Align test columns with training columns
                # Drop columns that are in test but not in train (e.g., 'id', 'Id', 'ID')
                # Keep only columns that are in both
                common_cols = [col for col in X_train.columns if col in test_df.columns]

                # If no common columns, test might have different structure
                # Try dropping potential ID columns
                if len(common_cols) == 0:
                    id_cols = [col for col in test_df.columns if col.lower() in ['id', 'index']]
                    if id_cols:
                        test_df_no_id = test_df.drop(columns=id_cols)
                        common_cols = [col for col in X_train.columns if col in test_df_no_id.columns]
                        X_test = test_df_no_id[common_cols]
                    else:
                        X_test = test_df
                else:
                    X_test = test_df[common_cols]

                # Reorder to match X_train column order
                X_test = X_test[X_train.columns]
            else:
                X_test = X_train.head(0)  # Empty dataframe with same columns

            self.log_info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

            # Preprocess features
            X_train_processed, X_test_processed, preprocessing_info = self._preprocess_features(
                X_train, X_test, feature_analysis
            )

            # Encode target if needed
            y_train_encoded, target_encoder = self._encode_target(y_train, analysis_result)

            # Create CV folds
            cv_folds = self._create_cv_folds(X_train_processed, y_train_encoded, analysis_result)

            # Feature engineering
            engineered_features = self._engineer_features(X_train_processed, X_test_processed)

            result = {
                "status": "success",
                "preprocessing_pipeline": preprocessing_info,
                "cv_folds": cv_folds,
                "feature_summary": {
                    "original_features": len(X_train.columns),
                    "processed_features": X_train_processed.shape[1],
                    "engineered_features": 0,  # Placeholder
                    "total_features": X_train_processed.shape[1]
                },
                "data_ready": {
                    "X_train": X_train_processed,
                    "y_train": y_train_encoded,
                    "X_test": X_test_processed,
                    "target_encoder": target_encoder
                },
                "execution_time": 0.0
            }

            self.log_info(f"Feature engineering complete: {X_train_processed.shape[1]} features")
            return result

        except Exception as e:
            self.log_error(f"Feature engineering failed: {e}")
            return {"status": "error", "error": str(e)}

    def _identify_target(self, df: pd.DataFrame) -> str:
        """
        Identify target column using various heuristics.

        Tries multiple strategies:
        1. Common exact names
        2. Columns containing target-like keywords
        3. Falls back to last column
        """
        # Try exact matches first
        target_names = ['target', 'label', 'y', 'Transported', 'Survived', 'Target', 'Label']
        for name in target_names:
            if name in df.columns:
                self.log_info(f"Found target column by exact match: {name}")
                return name

        # Try partial matches (case-insensitive)
        target_keywords = ['target', 'label', 'prediction', 'score', 'class', 'outcome',
                          'energy', 'price', 'value', 'rating', 'survival']
        for col in df.columns:
            col_lower = col.lower()
            for keyword in target_keywords:
                if keyword in col_lower:
                    self.log_info(f"Found target column by keyword '{keyword}': {col}")
                    return col

        # Last resort: use last column (common convention in ML datasets)
        self.log_info(f"Using last column as target (default): {df.columns[-1]}")
        return df.columns[-1]

    def _preprocess_features(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        feature_analysis: Dict
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Preprocess features."""
        numerical_features = feature_analysis.get("numerical_features", [])
        categorical_features = feature_analysis.get("categorical_features", [])

        # Handle missing values
        if X_train.isnull().any().any():
            # Numerical: impute with median
            for col in numerical_features:
                if col in X_train.columns:
                    median_val = X_train[col].median()
                    X_train[col].fillna(median_val, inplace=True)
                    if col in X_test.columns:
                        X_test[col].fillna(median_val, inplace=True)

            # Categorical: impute with mode
            for col in categorical_features:
                if col in X_train.columns:
                    mode_val = X_train[col].mode()[0] if len(X_train[col].mode()) > 0 else "unknown"
                    X_train[col].fillna(mode_val, inplace=True)
                    if col in X_test.columns:
                        X_test[col].fillna(mode_val, inplace=True)

        # Encode categorical features
        label_encoders = {}
        for col in categorical_features:
            if col in X_train.columns:
                le = LabelEncoder()
                X_train[col] = le.fit_transform(X_train[col].astype(str))
                if col in X_test.columns:
                    # Handle unseen categories
                    X_test[col] = X_test[col].astype(str).map(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
                label_encoders[col] = le

        # Convert to numpy arrays
        X_train_array = X_train.values.astype(float)
        X_test_array = X_test.values.astype(float) if len(X_test) > 0 else np.array([])

        preprocessing_info = {
            "label_encoders": label_encoders,
            "feature_names": list(X_train.columns)
        }

        return X_train_array, X_test_array, preprocessing_info

    def _encode_target(self, y: pd.Series, analysis_result: Dict) -> Tuple[np.ndarray, Optional[LabelEncoder]]:
        """Encode target variable."""
        target_analysis = analysis_result.get("target_analysis", {})

        if target_analysis.get("type") == "classification":
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            return y_encoded, le
        else:
            return y.values, None

    def _create_cv_folds(
        self,
        X: np.ndarray,
        y: np.ndarray,
        analysis_result: Dict
    ) -> list:
        """Create cross-validation folds."""
        target_analysis = analysis_result.get("target_analysis", {})
        n_splits = 5

        if target_analysis.get("type") == "classification":
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        else:
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        folds = list(cv.split(X, y))
        self.log_info(f"Created {n_splits} CV folds")

        return folds

    def _engineer_features(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray
    ) -> Dict[str, Any]:
        """Engineer additional features."""
        # Placeholder for feature engineering
        # Could add polynomial features, interactions, etc.
        return {
            "engineered_count": 0,
            "method": "none"
        }
