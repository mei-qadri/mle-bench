"""P8: Ensemble Plug."""
from mle_bench_agents.plugs.base_plug import BasePlug
from typing import Any, Dict
import numpy as np

class EnsemblePlug(BasePlug):
    def get_constraints(self) -> Dict[str, Any]:
        return {"min_models": 2, "max_models": 20}
    
    def execute(self, predictions_list, method="average", **kwargs) -> Dict[str, Any]:
        """Create ensemble."""
        if method == "average":
            ensemble_pred = np.mean(predictions_list, axis=0)
        else:
            ensemble_pred = predictions_list[0]
        return {"ensemble_predictions": ensemble_pred, "method": method}
