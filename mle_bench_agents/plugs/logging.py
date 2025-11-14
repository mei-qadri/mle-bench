"""P12: Logging and Monitoring Plug."""
from mle_bench_agents.plugs.base_plug import BasePlug
from typing import Any, Dict
import json
from pathlib import Path

class LoggingPlug(BasePlug):
    def get_constraints(self) -> Dict[str, Any]:
        return {"max_log_size": 1024*1024*1024}  # 1GB
    
    def execute(self, message, level="INFO", **kwargs) -> Dict[str, Any]:
        """Log message."""
        log_file = self.config.get("log_file", "/home/logs/system.log")
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, 'a') as f:
            f.write(f"[{level}] {message}\n")
        return {"logged": True}
