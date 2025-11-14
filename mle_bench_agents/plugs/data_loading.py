"""P1: Data Loading Plug - Load various data formats."""

import pandas as pd
from pathlib import Path
from typing import Any, Dict, Union
from mle_bench_agents.plugs.base_plug import BasePlug


class DataLoadingPlug(BasePlug):
    """Load CSV, Parquet, JSON, images, audio files."""

    def get_constraints(self) -> Dict[str, Any]:
        return {
            "max_memory_percent": self.config.get("max_memory_percent", 40),
            "supported_formats": ["csv", "parquet", "json", "png", "jpg", "tiff", "wav"],
            "read_only": True
        }

    def execute(self, file_path: Union[str, Path], **kwargs) -> Any:
        """Load data from file."""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Detect format
        suffix = file_path.suffix.lower()

        if suffix == ".csv":
            return pd.read_csv(file_path, **kwargs)
        elif suffix == ".parquet":
            return pd.read_parquet(file_path, **kwargs)
        elif suffix == ".json":
            return pd.read_json(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {suffix}")
