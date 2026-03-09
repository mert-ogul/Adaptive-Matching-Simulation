"""Core model objects for the Asymmetric(a, δ) expert system."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Literal

import numpy as np
import numpy.typing as npt

TaskType = Literal["c1", "c2"]


class EventType(Enum):
    """Discrete-event types used in the simulator."""

    ARRIVAL = auto()
    SERVICE_COMPLETION = auto()


@dataclass(slots=True)
class Task:
    """Represents a single task in the system."""

    task_id: int
    true_type: TaskType
    arrival_time: float
    belief: npt.NDArray[np.float64]
    attempts: int = 0
    attempt_history: list[tuple[str, bool, float]] = field(default_factory=list)
    departure_time: float | None = None


@dataclass(slots=True)
class Expert:
    """Represents an expert server."""

    expert_id: str
    index: int
    busy: bool = False
    current_task_id: int | None = None
    busy_time: float = 0.0
    last_start_busy: float | None = None


@dataclass(slots=True)
class Event:
    """Event stored in the priority queue."""

    time: float
    seq: int
    event_type: EventType
    expert_index: int | None = None
    task_id: int | None = None


@dataclass(slots=True)
class SystemState:
    """Mutable state of the simulation at a point in time."""

    time: float
    lam: float
    experts: list[Expert]
    tasks: dict[int, Task]
    task_pool: list[int]
    completed_task_ids: list[int]
    event_queue: list[tuple[float, int, Event]]
    rng: np.random.Generator
