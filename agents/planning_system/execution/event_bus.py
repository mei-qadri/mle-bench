"""
Event Bus

Enables inter-agent communication through events.
"""

import queue
import threading
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Event:
    """An event in the system"""

    event_type: str  # Type of event
    source_agent_id: str  # Agent that generated the event
    data: Dict  # Event payload
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class EventBus:
    """
    Event bus for inter-agent communication

    Agents can publish events and subscribe to event types.
    """

    def __init__(self):
        self.subscribers: Dict[str, List[str]] = {}  # event_type → list of agent_ids
        self.event_queue = queue.Queue()
        self.event_history: List[Event] = []
        self.lock = threading.Lock()
        self.handlers: Dict[str, List[Callable]] = {}  # event_type → list of handlers

    def publish(self, event: Event):
        """Publish event to all subscribers"""
        with self.lock:
            self.event_queue.put(event)
            self.event_history.append(event)

            # Call registered handlers
            if event.event_type in self.handlers:
                for handler in self.handlers[event.event_type]:
                    try:
                        handler(event)
                    except Exception as e:
                        print(f"Error in event handler: {e}")

    def subscribe(self, agent_id: str, event_types: List[str]):
        """Subscribe agent to specific event types"""
        with self.lock:
            for event_type in event_types:
                if event_type not in self.subscribers:
                    self.subscribers[event_type] = []
                if agent_id not in self.subscribers[event_type]:
                    self.subscribers[event_type].append(agent_id)

    def unsubscribe(self, agent_id: str, event_types: List[str] = None):
        """Unsubscribe agent from event types"""
        with self.lock:
            if event_types is None:
                # Unsubscribe from all
                for subs_list in self.subscribers.values():
                    if agent_id in subs_list:
                        subs_list.remove(agent_id)
            else:
                for event_type in event_types:
                    if event_type in self.subscribers and agent_id in self.subscribers[event_type]:
                        self.subscribers[event_type].remove(agent_id)

    def get_events(self, agent_id: str, event_types: List[str] = None) -> List[Event]:
        """
        Get events for a specific agent

        Returns events that the agent is subscribed to.
        """
        with self.lock:
            events = []

            # Get subscribed event types for this agent
            subscribed_types = []
            for event_type, agent_list in self.subscribers.items():
                if agent_id in agent_list:
                    subscribed_types.append(event_type)

            # Filter by requested event types if specified
            if event_types:
                subscribed_types = [t for t in subscribed_types if t in event_types]

            # Get events from history
            for event in self.event_history:
                if event.event_type in subscribed_types:
                    events.append(event)

            return events

    def register_handler(self, event_type: str, handler: Callable):
        """Register a handler function for an event type"""
        with self.lock:
            if event_type not in self.handlers:
                self.handlers[event_type] = []
            self.handlers[event_type].append(handler)

    def get_all_events(self) -> List[Event]:
        """Get all events in history"""
        with self.lock:
            return self.event_history.copy()

    def clear(self):
        """Clear all events and subscriptions"""
        with self.lock:
            self.event_queue = queue.Queue()
            self.event_history.clear()
            self.subscribers.clear()
            self.handlers.clear()

    def __repr__(self) -> str:
        return f"EventBus(events={len(self.event_history)}, subscribers={len(self.subscribers)})"
