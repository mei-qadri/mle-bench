"""
Execution History

Tracks the complete execution trace of an agent (Hi component).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import json


@dataclass
class ExecutionHistory:
    """
    Records the complete execution trace of an agent (Hi).

    Tracks all events, actions, observations, and performance metrics.
    """

    # Timestamped events
    events: List[Dict] = field(default_factory=list)

    # Event types tracked:
    # - "thought": Agent reasoning/planning
    # - "action": Actions taken (with parameters)
    # - "observation": Results from actions
    # - "message": Inter-agent communication
    # - "error": Errors encountered
    # - "milestone": Key achievements

    # Structured logs
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    # Performance tracking
    tokens_used: int = 0
    api_calls: int = 0
    cost_usd: float = 0.0

    def add_event(
        self,
        event_type: str,
        content: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Add a timestamped event to history"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "content": content,
        }
        if metadata:
            event["metadata"] = metadata

        self.events.append(event)

    def add_thought(self, thought: str):
        """Add a reasoning/planning event"""
        self.add_event("thought", {"text": thought})

    def add_action(self, action_name: str, parameters: Dict[str, Any], result: Any):
        """Add an action execution event"""
        self.add_event(
            "action",
            {
                "name": action_name,
                "parameters": parameters,
                "result": str(result)[:1000],  # Truncate long results
            },
        )

    def add_observation(self, observation: Any):
        """Add an observation from the environment"""
        self.add_event("observation", {"content": str(observation)[:1000]})

    def add_message(self, from_agent: str, to_agent: str, message: str):
        """Add inter-agent communication event"""
        self.add_event(
            "message", {"from": from_agent, "to": to_agent, "message": message}
        )

    def add_error(self, error: Exception, context: Optional[str] = None):
        """Add error event"""
        self.add_event(
            "error",
            {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "context": context,
            },
        )

    def add_milestone(self, milestone: str, details: Optional[Dict] = None):
        """Add milestone achievement event"""
        content = {"milestone": milestone}
        if details:
            content.update(details)
        self.add_event("milestone", content)

    def get_events_by_type(self, event_type: str) -> List[Dict]:
        """Get all events of a specific type"""
        return [e for e in self.events if e["type"] == event_type]

    def get_recent_events(self, n: int = 10) -> List[Dict]:
        """Get the n most recent events"""
        return self.events[-n:]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "events": self.events,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "tokens_used": self.tokens_used,
            "api_calls": self.api_calls,
            "cost_usd": self.cost_usd,
        }

    def to_json(self, filepath: str):
        """Save history to JSON file"""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_markdown(self) -> str:
        """Export history as readable markdown"""
        lines = ["# Agent Execution History\n"]

        if self.start_time:
            lines.append(f"**Start Time**: {self.start_time.isoformat()}\n")
        if self.end_time:
            lines.append(f"**End Time**: {self.end_time.isoformat()}\n")

        lines.append(f"\n**Performance Metrics**:")
        lines.append(f"- API Calls: {self.api_calls}")
        lines.append(f"- Tokens Used: {self.tokens_used}")
        lines.append(f"- Cost: ${self.cost_usd:.4f}\n")

        lines.append("\n## Event Timeline\n")

        for i, event in enumerate(self.events, 1):
            timestamp = event["timestamp"]
            event_type = event["type"]
            content = event["content"]

            lines.append(f"### Event {i}: {event_type.upper()}")
            lines.append(f"*{timestamp}*\n")

            if event_type == "thought":
                lines.append(f"**Thought**: {content['text']}\n")
            elif event_type == "action":
                lines.append(f"**Action**: {content['name']}")
                lines.append(f"**Parameters**: {content['parameters']}")
                lines.append(f"**Result**: {content['result']}\n")
            elif event_type == "observation":
                lines.append(f"**Observation**: {content['content']}\n")
            elif event_type == "message":
                lines.append(
                    f"**Message**: {content['from']} → {content['to']}: {content['message']}\n"
                )
            elif event_type == "error":
                lines.append(f"**Error**: {content['error_type']}")
                lines.append(f"**Message**: {content['error_message']}")
                if content.get("context"):
                    lines.append(f"**Context**: {content['context']}\n")
            elif event_type == "milestone":
                lines.append(f"**Milestone**: {content['milestone']}\n")

        return "\n".join(lines)

    def to_notebook(self) -> str:
        """
        Export history as Jupyter notebook format (similar to OpenDevin).

        Returns JSON string in nbformat.
        """
        cells = []

        # Add header
        cells.append(
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# Agent Execution History\n"],
            }
        )

        # Add each event
        for event in self.events:
            event_type = event["type"]
            content = event["content"]

            if event_type == "thought":
                # Thoughts as markdown
                cells.append(
                    {
                        "cell_type": "markdown",
                        "metadata": {},
                        "source": [f"## Thought\n{content['text']}"],
                    }
                )
            elif event_type == "action":
                # Actions as code cells
                if "code" in content:
                    cells.append(
                        {
                            "cell_type": "code",
                            "metadata": {},
                            "source": [content["code"]],
                            "outputs": [
                                {
                                    "output_type": "stream",
                                    "name": "stdout",
                                    "text": [content.get("result", "")],
                                }
                            ],
                            "execution_count": len(
                                [c for c in cells if c["cell_type"] == "code"]
                            )
                            + 1,
                        }
                    )

        notebook = {
            "cells": cells,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4,
        }

        return json.dumps(notebook, indent=2)

    def save_notebook(self, filepath: str):
        """Save history as Jupyter notebook file"""
        with open(filepath, "w") as f:
            f.write(self.to_notebook())

    def __repr__(self) -> str:
        return f"ExecutionHistory(events={len(self.events)}, api_calls={self.api_calls}, cost=${self.cost_usd:.4f})"
