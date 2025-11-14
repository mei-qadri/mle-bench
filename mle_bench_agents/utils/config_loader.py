"""Configuration loading utilities."""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def load_agent_config(
    agent_id: str,
    config_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Load configuration for specific agent.

    Args:
        agent_id: Agent ID
        config_path: Path to agent configs file

    Returns:
        Agent configuration
    """
    if config_path is None:
        # Default path
        config_path = Path(__file__).parent.parent.parent / "docs" / "config" / "agent_configs.yaml"

    config = load_config(config_path)

    agents_config = config.get("agents", {})

    if agent_id not in agents_config:
        raise ValueError(f"Agent {agent_id} not found in configuration")

    return agents_config[agent_id]


def load_prompt(prompt_path: Path) -> str:
    """
    Load system prompt from file.

    Args:
        prompt_path: Path to prompt file

    Returns:
        Prompt text
    """
    prompt_path = Path(prompt_path)

    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    with open(prompt_path, 'r') as f:
        prompt = f.read()

    return prompt


def get_prompt_path(agent_id: str, base_path: Optional[Path] = None) -> Path:
    """
    Get prompt file path for agent.

    Args:
        agent_id: Agent ID
        base_path: Base prompts directory

    Returns:
        Path to prompt file
    """
    if base_path is None:
        base_path = Path(__file__).parent.parent.parent / "docs" / "prompts"

    # Map agent IDs to prompt files
    prompt_map = {
        "A1_orchestrator": "A1_orchestrator.md",
        "A2_analysis": "A2_analysis.md",
        "A3a_cv_specialist": "A3a_cv_specialist.md",
        "A3b_nlp_specialist": "A3b_nlp_specialist.md",
        "A3c_tabular_specialist": "A3c_tabular_specialist.md",
        "A3d_audio_specialist": "A3d_audio_specialist.md",
        "A4_engineering": "A4_engineering.md",
        "A5_ensemble": "A5_ensemble.md",
        "A6_validation": "A6_validation.md",
        "A7_resource_manager": "A7_resource_manager.md",
    }

    filename = prompt_map.get(agent_id, f"{agent_id}.md")
    return base_path / filename
