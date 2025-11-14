"""
Base Plug class for all plug components.

Plugs are defined as Pj = {Fj, Cj, Uj} where:
- Fj: Functionality (actions the plug performs)
- Cj: Configuration parameters
- Uj: Constraints and usage rules
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BasePlug(ABC):
    """
    Base class for all plug components.

    Each plug provides specific functionality that can be used by multiple agents.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize plug.

        Args:
            config: Plug configuration (Cj)
        """
        self.config = config or {}
        self.constraints = self.get_constraints()  # Uj

    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """
        Execute plug functionality (Fj).

        Args:
            *args, **kwargs: Plug-specific arguments

        Returns:
            Plug-specific results
        """
        pass

    @abstractmethod
    def get_constraints(self) -> Dict[str, Any]:
        """
        Get plug constraints (Uj).

        Returns:
            Dictionary of constraints
        """
        pass

    def validate_inputs(self, **kwargs) -> bool:
        """
        Validate inputs against constraints.

        Args:
            **kwargs: Input parameters

        Returns:
            True if valid
        """
        # To be overridden by subclasses
        return True

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(config={self.config})"
