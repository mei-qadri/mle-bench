"""P11: Metrics Computation Plug - Calculate evaluation metrics."""

from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
from typing import Any, Dict
import numpy as np
from mle_bench_agents.plugs.base_plug import BasePlug


class MetricsPlug(BasePlug):
    """Compute various ML evaluation metrics."""

    def get_constraints(self) -> Dict[str, Any]:
        return {
            "max_computation_time": 30,
            "supported_metrics": [
                "accuracy", "auc", "rmse", "mae", "f1", "precision", "recall"
            ]
        }

    def execute(self, y_true, y_pred, metric: str = "accuracy", **kwargs) -> float:
        """Compute specified metric."""
        metric = metric.lower()

        if metric == "accuracy":
            return accuracy_score(y_true, y_pred)
        elif metric in ["auc", "roc_auc"]:
            return roc_auc_score(y_true, y_pred, **kwargs)
        elif metric == "rmse":
            return np.sqrt(mean_squared_error(y_true, y_pred))
        elif metric == "mae":
            return np.mean(np.abs(y_true - y_pred))
        else:
            raise ValueError(f"Unsupported metric: {metric}")
