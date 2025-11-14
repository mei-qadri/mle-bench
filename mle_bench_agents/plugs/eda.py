"""P2: EDA Plug - Exploratory Data Analysis."""
from mle_bench_agents.plugs.base_plug import BasePlug
from typing import Any, Dict

class EDAPlug(BasePlug):
    def get_constraints(self) -> Dict[str, Any]:
        return {"max_time": 600, "max_visualizations": 10}
    
    def execute(self, data, **kwargs) -> Dict[str, Any]:
        """Perform EDA."""
        return {"summary": data.describe().to_dict() if hasattr(data, 'describe') else {}}
