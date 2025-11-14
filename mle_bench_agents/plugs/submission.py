"""P9: Submission Generation Plug."""
from mle_bench_agents.plugs.base_plug import BasePlug
from typing import Any, Dict
import pandas as pd

class SubmissionPlug(BasePlug):
    def get_constraints(self) -> Dict[str, Any]:
        return {"max_file_size": 500*1024*1024}  # 500MB
    
    def execute(self, predictions, ids, output_path, **kwargs) -> Dict[str, Any]:
        """Generate submission file."""
        submission = pd.DataFrame({"id": ids, "prediction": predictions})
        submission.to_csv(output_path, index=False)
        return {"submission_path": output_path, "n_rows": len(submission)}
