"""P7: Cross-Validation Plug."""
from mle_bench_agents.plugs.base_plug import BasePlug
from sklearn.model_selection import StratifiedKFold
from typing import Any, Dict

class CrossValidationPlug(BasePlug):
    def get_constraints(self) -> Dict[str, Any]:
        return {"n_splits": 5, "stratified": True}
    
    def execute(self, X, y, **kwargs) -> Dict[str, Any]:
        """Create CV folds."""
        n_splits = self.config.get("n_splits", 5)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        folds = list(cv.split(X, y))
        return {"cv_folds": folds, "n_splits": n_splits}
