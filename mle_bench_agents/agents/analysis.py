"""
Analysis Agent (A2) - EDA and Competition Understanding

Performs exploratory data analysis and provides recommendations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional

from mle_bench_agents.core.agent import Agent
from mle_bench_agents.core.message import Message, MessageType
from mle_bench_agents.plugs.data_loading import DataLoadingPlug
from mle_bench_agents.plugs.eda import EDAPlug


class AnalysisAgent(Agent):
    """
    Analysis Agent responsible for:
    - Competition understanding
    - Exploratory data analysis
    - Data quality assessment
    - Modeling recommendations
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, agent_type="analysis", **kwargs)

        # Initialize plugs
        self.data_loader = DataLoadingPlug()
        self.eda_plug = EDAPlug()

    def process_message(self, message: Message) -> Optional[Message]:
        """Process incoming messages."""
        payload = message.payload
        action = payload.get("action")

        if action == "analyze_competition":
            result = self.analyze_competition(payload)
            return message.create_response(
                sender_id=self.agent_id,
                payload=result
            )

        return None

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute analysis."""
        return self.analyze_competition(context)

    def analyze_competition(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze competition and generate recommendations.

        Args:
            context: Contains competition_id, data_path, description

        Returns:
            Analysis results with recommendations
        """
        self.log_info("Starting competition analysis")

        competition_id = context.get("competition_id")
        data_path = Path(context.get("data_path", "/home/data/"))

        try:
            # Load training data
            train_file = data_path / "train.csv"
            if not train_file.exists():
                train_file = data_path / "train.parquet"

            if not train_file.exists():
                return self._fallback_analysis(context)

            self.log_info(f"Loading data from {train_file}")
            train_df = self.data_loader.execute(train_file)

            # Basic dataset summary
            dataset_summary = {
                "n_rows": len(train_df),
                "n_cols": len(train_df.columns),
                "memory_mb": train_df.memory_usage(deep=True).sum() / (1024**2),
                "columns": list(train_df.columns)
            }

            # Identify target column (usually last column or 'target')
            target_col = self._identify_target_column(train_df)

            # Feature types
            numerical_features = train_df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_features = train_df.select_dtypes(include=['object']).columns.tolist()

            if target_col in numerical_features:
                numerical_features.remove(target_col)
            if target_col in categorical_features:
                categorical_features.remove(target_col)

            # Target analysis
            target_analysis = self._analyze_target(train_df, target_col)

            # Data quality
            data_quality = self._assess_data_quality(train_df)

            # Competition type classification
            competition_type = self._classify_competition_type(train_df, target_col, target_analysis)

            # Complexity assessment
            complexity = self._assess_complexity(dataset_summary, data_quality)

            # Generate recommendations
            recommendations = self._generate_recommendations(
                competition_type, target_analysis, data_quality, numerical_features, categorical_features
            )

            # Metric detection
            metric_info = self._detect_metric(competition_type, target_analysis)

            result = {
                "status": "success",
                "competition_analysis": {
                    "competition_id": competition_id,
                    "task_type": target_analysis.get("task_type"),
                    "domain": competition_type,
                    "metric_name": metric_info["name"],
                    "metric_direction": metric_info["direction"]
                },
                "dataset_summary": dataset_summary,
                "target_analysis": target_analysis,
                "feature_analysis": {
                    "numerical_features": numerical_features,
                    "categorical_features": categorical_features,
                    "n_numerical": len(numerical_features),
                    "n_categorical": len(categorical_features),
                    "target_column": target_col
                },
                "data_quality": data_quality,
                "complexity_assessment": {
                    "overall": complexity,
                    "factors": self._get_complexity_factors(dataset_summary, data_quality)
                },
                "recommendations": recommendations,
                "execution_time": 0.0
            }

            self.log_info(f"Analysis complete: {competition_type} task, {complexity} complexity")
            return result

        except Exception as e:
            self.log_error(f"Analysis failed: {e}")
            return self._fallback_analysis(context)

    def _identify_target_column(self, df: pd.DataFrame) -> str:
        """Identify the target column."""
        # Common target column names
        target_names = ['target', 'label', 'y', 'class', 'Transported', 'Survived']

        for name in target_names:
            if name in df.columns:
                return name

        # Default to last column
        return df.columns[-1]

    def _analyze_target(self, df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """Analyze target variable."""
        target = df[target_col]

        # Check if classification or regression
        n_unique = target.nunique()

        if n_unique <= 20 and target.dtype == 'object' or target.dtype == 'bool':
            # Classification
            value_counts = target.value_counts()
            class_balance = value_counts / len(target)

            task_type = "binary_classification" if n_unique == 2 else "multiclass_classification"

            # Assess imbalance
            min_ratio = class_balance.min()
            if min_ratio > 0.4:
                imbalance = "balanced"
            elif min_ratio > 0.1:
                imbalance = "moderate"
            else:
                imbalance = "severe"

            return {
                "type": "classification",
                "task_type": task_type,
                "n_classes": n_unique,
                "distribution": value_counts.to_dict(),
                "class_balance": class_balance.to_dict(),
                "imbalance_severity": imbalance
            }
        else:
            # Regression
            return {
                "type": "regression",
                "task_type": "regression",
                "mean": float(target.mean()),
                "std": float(target.std()),
                "min": float(target.min()),
                "max": float(target.max())
            }

    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data quality issues."""
        # Missing values
        missing = df.isnull().sum()
        missing_pct = missing / len(df)

        has_missing = (missing > 0).any()
        max_missing_pct = missing_pct.max() if has_missing else 0.0

        # Duplicates
        n_duplicates = df.duplicated().sum()

        # Constant features
        constant_features = [col for col in df.columns if df[col].nunique() == 1]

        return {
            "missing_values_present": bool(has_missing),
            "max_missing_percent": float(max_missing_pct),
            "missing_severity": "high" if max_missing_pct > 0.3 else "medium" if max_missing_pct > 0.1 else "low",
            "duplicates": int(n_duplicates),
            "constant_features": constant_features
        }

    def _classify_competition_type(self, df: pd.DataFrame, target_col: str, target_analysis: Dict) -> str:
        """Classify competition type based on data."""
        # Check for image/audio/text data
        # For now, assume tabular
        return "tabular"

    def _assess_complexity(self, dataset_summary: Dict, data_quality: Dict) -> str:
        """Assess overall complexity."""
        score = 0

        # Dataset size
        if dataset_summary["n_rows"] < 10000:
            score += 0
        elif dataset_summary["n_rows"] < 100000:
            score += 1
        else:
            score += 2

        # Number of features
        if dataset_summary["n_cols"] < 20:
            score += 0
        elif dataset_summary["n_cols"] < 100:
            score += 1
        else:
            score += 2

        # Data quality
        if data_quality["missing_severity"] == "high":
            score += 2
        elif data_quality["missing_severity"] == "medium":
            score += 1

        # Map score to complexity
        if score <= 2:
            return "low"
        elif score <= 4:
            return "medium"
        else:
            return "high"

    def _get_complexity_factors(self, dataset_summary: Dict, data_quality: Dict) -> Dict[str, str]:
        """Get complexity factors."""
        return {
            "dataset_size": "large" if dataset_summary["n_rows"] > 100000 else "medium" if dataset_summary["n_rows"] > 10000 else "small",
            "num_features": "high" if dataset_summary["n_cols"] > 100 else "medium" if dataset_summary["n_cols"] > 20 else "low",
            "missing_data": data_quality["missing_severity"],
            "data_type_diversity": "mixed"
        }

    def _detect_metric(self, competition_type: str, target_analysis: Dict) -> Dict[str, str]:
        """Detect appropriate metric."""
        if target_analysis.get("type") == "classification":
            if target_analysis.get("n_classes") == 2:
                return {"name": "accuracy", "direction": "maximize"}
            else:
                return {"name": "accuracy", "direction": "maximize"}
        else:
            return {"name": "rmse", "direction": "minimize"}

    def _generate_recommendations(
        self,
        competition_type: str,
        target_analysis: Dict,
        data_quality: Dict,
        numerical_features: list,
        categorical_features: list
    ) -> list:
        """Generate modeling recommendations."""
        recommendations = []

        # Preprocessing recommendations
        if data_quality["missing_values_present"]:
            recommendations.append({
                "type": "preprocessing",
                "action": "impute_missing_values",
                "priority": "high",
                "details": f"Handle missing values ({data_quality['missing_severity']} severity)"
            })

        # Feature engineering
        if len(numerical_features) > 0:
            recommendations.append({
                "type": "feature_engineering",
                "action": "create_interactions",
                "priority": "medium",
                "details": "Create polynomial and interaction features"
            })

        # Modeling recommendations
        if competition_type == "tabular":
            recommendations.append({
                "type": "modeling",
                "action": "gradient_boosting",
                "priority": "high",
                "details": "Use XGBoost/LightGBM for tabular data"
            })

        # Validation strategy
        if target_analysis.get("type") == "classification":
            recommendations.append({
                "type": "validation",
                "action": "stratified_cv",
                "priority": "high",
                "details": "Use stratified k-fold cross-validation"
            })

        return recommendations

    def _fallback_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback analysis when data loading fails."""
        self.log_warning("Using fallback analysis")

        return {
            "status": "fallback",
            "competition_analysis": {
                "competition_id": context.get("competition_id"),
                "task_type": "classification",
                "domain": "tabular",
                "metric_name": "accuracy",
                "metric_direction": "maximize"
            },
            "dataset_summary": {"n_rows": 0, "n_cols": 0},
            "target_analysis": {"type": "classification", "task_type": "binary_classification"},
            "feature_analysis": {"numerical_features": [], "categorical_features": []},
            "data_quality": {"missing_values_present": False},
            "complexity_assessment": {"overall": "medium"},
            "recommendations": []
        }
