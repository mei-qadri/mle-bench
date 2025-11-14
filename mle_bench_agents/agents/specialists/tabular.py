"""
Tabular Specialist Agent (A3c) - Machine Learning for Tabular Data

Trains gradient boosting models on tabular datasets.
"""

import numpy as np
from typing import Any, Dict, Optional
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error

from mle_bench_agents.core.agent import Agent
from mle_bench_agents.core.message import Message


class TabularSpecialistAgent(Agent):
    """
    Tabular Specialist responsible for:
    - Model selection for tabular data
    - Training with cross-validation
    - Hyperparameter optimization (optional)
    - Generating predictions
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, agent_type="tabular_specialist", **kwargs)

    def process_message(self, message: Message) -> Optional[Message]:
        """Process incoming messages."""
        payload = message.payload
        action = payload.get("action")

        if action == "train_models":
            result = self.train_models(payload)
            return message.create_response(
                sender_id=self.agent_id,
                payload=result
            )

        return None

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute model training."""
        return self.train_models(context)

    def train_models(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train models with cross-validation.

        Args:
            context: Contains engineering_result, phase, config

        Returns:
            Training results with models and predictions
        """
        self.log_info("Starting model training")

        try:
            # Get data
            engineering_result = context.get("engineering_result", {})
            data_ready = engineering_result.get("data_ready", {})

            X_train = data_ready.get("X_train")
            y_train = data_ready.get("y_train")
            X_test = data_ready.get("X_test")
            cv_folds = engineering_result.get("cv_folds", [])

            if X_train is None or y_train is None:
                return {"status": "error", "error": "No training data provided"}

            # Get analysis info
            analysis_result = context.get("analysis_result", {})
            target_analysis = analysis_result.get("target_analysis", {})
            task_type = target_analysis.get("type", "classification")

            # Get phase configuration
            phase = context.get("phase", "baseline")
            config = context.get("config", {})

            self.log_info(f"Training phase: {phase}, Task: {task_type}")
            self.log_info(f"Data shape: X={X_train.shape}, y={y_train.shape}")

            # Train models based on phase
            if phase == "baseline":
                trained_models = self._train_baseline(X_train, y_train, X_test, cv_folds, task_type)
            elif phase == "advanced":
                trained_models = self._train_advanced(X_train, y_train, X_test, cv_folds, task_type, config)
            else:
                trained_models = self._train_baseline(X_train, y_train, X_test, cv_folds, task_type)

            # Find best model
            if trained_models:
                best_model = max(trained_models, key=lambda m: m["cv_score_mean"])
            else:
                return {"status": "error", "error": "No models trained successfully"}

            result = {
                "status": "success",
                "phase": phase,
                "models_trained": trained_models,
                "best_model": {
                    "model_id": best_model["model_id"],
                    "cv_score": best_model["cv_score_mean"]
                },
                "execution_time": 0.0
            }

            self.log_info(f"Training complete: {len(trained_models)} models, best CV: {best_model['cv_score_mean']:.4f}")
            return result

        except Exception as e:
            self.log_error(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
            return {"status": "error", "error": str(e)}

    def _train_baseline(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        cv_folds: list,
        task_type: str
    ) -> list:
        """Train quick baseline models."""
        self.log_info("Training baseline models")

        models_trained = []

        # Model 1: Logistic Regression / Ridge
        if task_type == "classification":
            model1 = LogisticRegression(max_iter=1000, random_state=42)
            model_name = "LogisticRegression"
        else:
            model1 = Ridge(random_state=42)
            model_name = "Ridge"

        result1 = self._train_with_cv(model1, model_name, X_train, y_train, X_test, cv_folds, task_type)
        if result1:
            models_trained.append(result1)

        # Model 2: Random Forest
        if task_type == "classification":
            model2 = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        else:
            model2 = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)

        result2 = self._train_with_cv(model2, "RandomForest", X_train, y_train, X_test, cv_folds, task_type)
        if result2:
            models_trained.append(result2)

        return models_trained

    def _train_advanced(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        cv_folds: list,
        task_type: str,
        config: Dict
    ) -> list:
        """Train advanced models."""
        self.log_info("Training advanced models")

        models_trained = []

        # Try to import gradient boosting libraries
        try:
            import xgboost as xgb
            has_xgboost = True
        except ImportError:
            has_xgboost = False
            self.log_warning("XGBoost not available")

        try:
            import lightgbm as lgb
            has_lightgbm = True
        except ImportError:
            has_lightgbm = False
            self.log_warning("LightGBM not available")

        # XGBoost
        if has_xgboost:
            if task_type == "classification":
                # Check if binary or multiclass
                n_classes = len(np.unique(y_train))
                if n_classes == 2:
                    model = xgb.XGBClassifier(
                        n_estimators=300,
                        max_depth=6,
                        learning_rate=0.05,
                        random_state=42,
                        n_jobs=-1
                    )
                else:
                    model = xgb.XGBClassifier(
                        n_estimators=300,
                        max_depth=6,
                        learning_rate=0.05,
                        random_state=42,
                        n_jobs=-1,
                        objective='multi:softmax',
                        num_class=n_classes
                    )
            else:
                model = xgb.XGBRegressor(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    random_state=42,
                    n_jobs=-1
                )

            result = self._train_with_cv(model, "XGBoost", X_train, y_train, X_test, cv_folds, task_type)
            if result:
                models_trained.append(result)

        # LightGBM
        if has_lightgbm:
            if task_type == "classification":
                model = lgb.LGBMClassifier(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                )
            else:
                model = lgb.LGBMRegressor(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                )

            result = self._train_with_cv(model, "LightGBM", X_train, y_train, X_test, cv_folds, task_type)
            if result:
                models_trained.append(result)

        # If no advanced models available, fall back to baseline
        if not models_trained:
            self.log_warning("No advanced models available, using baseline")
            return self._train_baseline(X_train, y_train, X_test, cv_folds, task_type)

        return models_trained

    def _train_with_cv(
        self,
        model,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        cv_folds: list,
        task_type: str
    ) -> Optional[Dict]:
        """Train model with cross-validation."""
        try:
            self.log_info(f"Training {model_name}")

            cv_scores = []
            test_predictions_per_fold = []

            for fold_idx, (train_idx, val_idx) in enumerate(cv_folds):
                X_fold_train = X_train[train_idx]
                y_fold_train = y_train[train_idx]
                X_fold_val = X_train[val_idx]
                y_fold_val = y_train[val_idx]

                # Train model
                model.fit(X_fold_train, y_fold_train)

                # Validate
                val_pred = model.predict(X_fold_val)
                fold_score = self._compute_score(y_fold_val, val_pred, task_type)
                cv_scores.append(fold_score)

                # Predict on test
                if len(X_test) > 0:
                    test_pred = model.predict(X_test)
                    test_predictions_per_fold.append(test_pred)

            # Aggregate CV scores
            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)

            # Aggregate test predictions (average across folds)
            if test_predictions_per_fold:
                test_predictions = np.mean(test_predictions_per_fold, axis=0)
            else:
                test_predictions = np.array([])

            model_id = f"{model_name.lower()}_{np.random.randint(1000):03d}"

            return {
                "model_id": model_id,
                "model_name": model_name,
                "model_config": {},
                "cv_score_mean": float(cv_mean),
                "cv_score_std": float(cv_std),
                "cv_scores_per_fold": [float(s) for s in cv_scores],
                "test_predictions": test_predictions,
                "model_checkpoint": None,
                "training_time_seconds": 0.0,
                "hpo_used": False
            }

        except Exception as e:
            self.log_error(f"Failed to train {model_name}: {e}")
            return None

    def _compute_score(self, y_true: np.ndarray, y_pred: np.ndarray, task_type: str) -> float:
        """Compute appropriate score."""
        if task_type == "classification":
            return accuracy_score(y_true, y_pred)
        else:
            # For regression, return negative RMSE (to maximize)
            return -np.sqrt(mean_squared_error(y_true, y_pred))
