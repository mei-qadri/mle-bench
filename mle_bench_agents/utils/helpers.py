"""Helper utilities for the multi-agent system."""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional


def save_json(data: Dict[str, Any], file_path: Path) -> None:
    """Save data as JSON."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(file_path: Path) -> Dict[str, Any]:
    """Load data from JSON."""
    with open(file_path, 'r') as f:
        return json.load(f)


def save_pickle(obj: Any, file_path: Path) -> None:
    """Save object as pickle."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(file_path: Path) -> Any:
    """Load object from pickle."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string (e.g., "2h 30m 15s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs}s")

    return " ".join(parts)


def format_bytes(bytes_count: int) -> str:
    """
    Format bytes into human-readable size string.

    Args:
        bytes_count: Number of bytes

    Returns:
        Formatted size string (e.g., "1.5 GB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_count < 1024.0:
            return f"{bytes_count:.1f} {unit}"
        bytes_count /= 1024.0

    return f"{bytes_count:.1f} PB"


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
