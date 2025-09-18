"""
Clean, object-oriented progress tracking system for parallel workers.

This module provides an intuitive API for workers to manage their progress
visualization without dealing with complex message passing or progress updates.
"""

import uuid
from typing import Optional, Callable, Any
from dataclasses import dataclass


@dataclass
class ProgressEvent:
    """Simple event-based progress message."""

    event_type: (
        str  # 'main_update', 'sub_create', 'sub_update', 'sub_complete', 'sub_remove'
    )
    worker_id: int
    device_id: int
    data: dict


class SubProgress:
    """Represents a sub-progress bar with automatic lifecycle management."""

    def __init__(self, name: str, total: int, color: str, parent: "MainProgress"):
        self.name = name
        self.total = total
        self.color = color
        self.parent = parent
        self.completed = 0
        self.id = str(uuid.uuid4())[:8]  # Short unique ID
        self._active = False

    def __enter__(self):
        """Context manager entry - creates the sub-progress bar."""
        self._active = True
        self.parent._emit_event(
            "sub_create",
            {
                "sub_id": self.id,
                "name": self.name,
                "total": self.total,
                "color": self.color,
                "completed": 0,
            },
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatically removes the sub-progress bar."""
        if self._active:
            self.parent._emit_event("sub_remove", {"sub_id": self.id})
            self._active = False

    def update(self, completed: Optional[int] = None, increment: int = 0):
        """Update the progress of this sub-progress bar."""
        if not self._active:
            return

        if completed is not None:
            self.completed = completed
        else:
            self.completed += increment

        # Clamp to valid range
        self.completed = max(0, min(self.completed, self.total))

        self.parent._emit_event(
            "sub_update",
            {"sub_id": self.id, "completed": self.completed, "total": self.total},
        )

    def complete(self):
        """Mark this sub-progress as complete."""
        self.update(completed=self.total)
        self.parent._emit_event("sub_complete", {"sub_id": self.id})

    def step(self, amount: int = 1):
        """Increment progress by a step amount."""
        self.update(increment=amount)


class MainProgress:
    """Represents the main worker progress bar."""

    def __init__(self, context: "ProgressContext"):
        self.context = context
        self.total = 100
        self.completed = 0
        self.current = ""
        self.active = True
        self.color = "blue"
        self._sub_progress_bars = {}

    def configure(
        self, total: int = 100, description: str = "Processing", color: str = "blue"
    ):
        """Configure the main progress bar."""
        self.total = total
        self.current = description
        self.color = color
        self._notify_update()
        return self

    def update(
        self,
        completed: Optional[int] = None,
        current: Optional[str] = None,
        total: Optional[str] = None,
        increment: int = 0,
    ):
        """Update the main progress bar."""
        if completed is not None:
            self.completed = completed
        else:
            self.completed += increment

        if current is not None:
            self.current = current
        if total is not None:
            self.total = total

        # Clamp to valid range
        self.completed = max(0, min(self.completed, self.total))
        self._notify_update()

    def step(self, amount: int = 1, current: Optional[str] = None):
        """Increment progress by a step amount."""
        self.update(increment=amount, current=current)

    def sub_progress(self, name: str, total: int, color: str = "cyan") -> SubProgress:
        """Create a new sub-progress bar."""
        return SubProgress(name, total, color, self)

    def finish(self, message: str = "Complete"):
        """Mark the main progress as finished."""
        self.completed = self.total
        self.current = message
        self.active = False
        self._notify_update()

    def _notify_update(self):
        """Send main progress update to the central manager."""
        self._emit_event(
            "main_update",
            {
                "completed": self.completed,
                "total": self.total,
                "current": self.current,
                "active": self.active,
                "color": self.color,
            },
        )

    def _emit_event(self, event_type: str, data: dict):
        """Emit an event through the progress context."""
        self.context._emit_event(event_type, data)

    def cleanup(self):
        """Clean up all sub-progress bars (called by context manager)."""
        for sub_id in list(self._sub_progress_bars.keys()):
            self._emit_event("sub_remove", {"sub_id": sub_id})
        self._sub_progress_bars.clear()


class ProgressContext:
    """
    Main entry point for worker progress management.

    Provides a clean, object-oriented interface for workers to manage
    their progress visualization without dealing with message queues directly.
    """

    def __init__(
        self, worker_id: int, device_id: int, callback: Callable[[ProgressEvent], None]
    ):
        self.worker_id = worker_id
        self.device_id = device_id
        self.callback = callback
        self.main_progress = MainProgress(self)

    def __enter__(self):
        """Context manager entry - returns the main progress bar."""
        return self.main_progress

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleans up all progress bars."""
        self.main_progress.cleanup()
        self.main_progress.finish()

    def _emit_event(self, event_type: str, data: dict):
        """Emit a progress event to the central progress manager."""
        event = ProgressEvent(
            event_type=event_type,
            worker_id=self.worker_id,
            device_id=self.device_id,
            data=data,
        )
        self.callback(event)


# Convenience functions for common progress patterns
def create_progress_context(
    worker_id: int, device_id: int, callback: Callable[[ProgressEvent], None]
) -> ProgressContext:
    """Factory function to create a progress context."""
    return ProgressContext(worker_id, device_id, callback)


def multi_stage_progress(progress: MainProgress, stages: list[tuple[str, int, str]]):
    """
    Helper function for common multi-stage progress patterns.

    Args:
        progress: The main progress bar
        stages: List of (name, total, color) tuples for each stage

    Returns:
        List of SubProgress context managers
    """
    return [progress.sub_progress(name, total, color) for name, total, color in stages]
