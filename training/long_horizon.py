"""Long-Horizon Planning utilities for ATC multi-agent training.

Theme #2: Super Long-Horizon Planning & Instruction Following
─────────────────────────────────────────────────────────────
Standard LLM benchmarks test reasoning over a single context window (2–4K tokens).
Real-world ATC disruption recovery requires coordinating across *hours*, where:

  - Early sequencing decisions (minute 5) create wake-turbulence constraints
    that only bind at minute 120.
  - ATFM network slot deadlines cascade from a 09:00 departure to affect the
    11:00 wave if the first aircraft is delayed.
  - Airport configuration changes (runway inspection at 10:30) invalidate plans
    made at 08:00.
  - Agents must decompose "schedule the morning peak" into:
        Strategic:    Which aircraft need which runway? Who yields to emergencies?
        Tactical:     Which 15-min window gets each flight?
        Operational:  Exact slot assignment within the window?

This module provides three classes:

HierarchicalPlanDecomposer
    Takes a 6-hour TaskDefinition and decomposes it into planning epochs.
    Each epoch is a self-contained sub-task the agent can solve within its
    context window, with state forwarded between epochs.

EpisodeMemory
    Structured key-value store that persists across context turns.
    Agents write decisions and constraints; next-turn agents read them.
    Enables planning beyond single context window limits.

LongHorizonEpisodeBuilder
    Wraps MultiAgentATCEnvironment to run multi-epoch episodes:
    BID → NEGOTIATE → FINALIZE for each epoch, with carry-over constraints.
    Returns per-epoch and aggregate reward signals.

CascadeDetector
    Identifies whether a bad decision in epoch t caused a problem in epoch t+k.
    Used to compute the recovery bonus in reward functions.

Usage:
    builder = LongHorizonEpisodeBuilder(base_task, n_epochs=4)
    for epoch_task, epoch_context in builder.epochs():
        aman_obs, dman_obs = env.reset(mutated_task=epoch_task)
        # ... agents act ...
        result = env.finalize()
        builder.record_epoch(result, epoch_task)
    full_result = builder.aggregate()
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Tuple

try:
    from ..models import (
        FlightRecord,
        OperationType,
        PriorityClass,
        RunwaySpec,
        SlotAssignment,
        TaskDefinition,
        WakeClass,
    )
except ImportError:
    from models import (
        FlightRecord,
        OperationType,
        PriorityClass,
        RunwaySpec,
        SlotAssignment,
        TaskDefinition,
        WakeClass,
    )


# ── Episode Memory ────────────────────────────────────────────────────────────

@dataclass
class MemoryEntry:
    key:   str
    value: Any
    epoch: int
    ttl:   int = 999    # time-to-live in epochs (999 = permanent)


class EpisodeMemory:
    """Structured memory that persists agent decisions across planning epochs.

    Agents write to memory during epoch t; agents in epoch t+1 read relevant
    context. This enables planning beyond the LLM context window limit.

    Memory is organized into three namespaces:
        'strategic'    — high-level decisions (runway assignments, priority orders)
        'tactical'     — flight group → time window mappings
        'operational'  — specific slot assignments and carry-over constraints

    Constraints:
        Max 2048 characters of memory rendered per agent (stays in context window).
        Old entries (ttl expired) are pruned automatically.
    """

    MAX_RENDER_CHARS = 2048

    def __init__(self) -> None:
        self._entries: List[MemoryEntry] = []
        self._current_epoch: int = 0

    def advance_epoch(self) -> None:
        self._current_epoch += 1
        # Prune expired entries
        self._entries = [
            e for e in self._entries
            if (self._current_epoch - e.epoch) <= e.ttl
        ]

    def write(self, namespace: str, key: str, value: Any, ttl: int = 2) -> None:
        """Write a memory entry (overwrites same key in same namespace)."""
        full_key = f"{namespace}/{key}"
        self._entries = [e for e in self._entries if e.key != full_key]
        self._entries.append(MemoryEntry(
            key=full_key, value=value, epoch=self._current_epoch, ttl=ttl
        ))

    def read(self, namespace: str, key: str, default: Any = None) -> Any:
        """Read a memory entry. Returns default if not found."""
        full_key = f"{namespace}/{key}"
        for entry in reversed(self._entries):
            if entry.key == full_key:
                return entry.value
        return default

    def read_all(self, namespace: str) -> Dict[str, Any]:
        """Read all entries in a namespace."""
        prefix = f"{namespace}/"
        result = {}
        for entry in self._entries:
            if entry.key.startswith(prefix):
                result[entry.key[len(prefix):]] = entry.value
        return result

    def render_for_agent(self, role: str) -> str:
        """Render memory as text for LLM context injection.

        Only includes entries relevant to the agent's role. Truncated to
        MAX_RENDER_CHARS to stay within context budget.
        """
        lines = ["=== AGENT MEMORY (from previous planning epochs) ==="]
        role_namespaces = {
            "AMAN": ["strategic", "tactical", "operational"],
            "DMAN": ["strategic", "tactical", "operational"],
        }
        namespaces = role_namespaces.get(role, ["strategic"])

        for ns in namespaces:
            entries = self.read_all(ns)
            if entries:
                lines.append(f"\n[{ns.upper()}]")
                for k, v in entries.items():
                    lines.append(f"  {k}: {json.dumps(v, default=str)[:120]}")

        text = "\n".join(lines)
        if len(text) > self.MAX_RENDER_CHARS:
            text = text[:self.MAX_RENDER_CHARS] + "\n... [memory truncated]"
        return text

    def snapshot(self) -> Dict[str, Any]:
        return {
            "epoch": self._current_epoch,
            "entries": [
                {"key": e.key, "value": e.value, "epoch": e.epoch, "ttl": e.ttl}
                for e in self._entries
            ],
        }


# ── Hierarchical Plan Decomposer ──────────────────────────────────────────────

@dataclass
class PlanningEpoch:
    """One planning epoch — a 30-60 min slice of the full planning window."""
    epoch_id:     int
    start_minute: int
    end_minute:   int
    task:         TaskDefinition
    carry_over_constraints: List[str] = field(default_factory=list)


class HierarchicalPlanDecomposer:
    """Decomposes a long-horizon task into epoch-sized sub-tasks.

    Long tasks (planning_horizon_minutes > 60) are split into EPOCH_SIZE-minute
    windows. Each epoch task contains only the flights whose scheduling window
    falls within that epoch, plus any carry-over constraints from previous epochs.

    This forces agents to:
    1. Produce a strategic plan (which flights get which epoch?)
    2. Execute tactical decisions per-epoch
    3. Handle constraints that cascade from epoch to epoch

    Epoch overlap (OVERLAP_MINUTES) provides context: agents see flights from
    the boundary of the previous epoch so they can reason about wake constraints.
    """

    EPOCH_SIZE      = 45    # minutes per epoch (one context window = ~15 flights)
    OVERLAP_MINUTES = 10    # how many minutes of the previous epoch to include

    def __init__(self, task: TaskDefinition, n_epochs: Optional[int] = None) -> None:
        self._base_task = task
        self._horizon   = task.planning_horizon_minutes

        if n_epochs is not None:
            self._n_epochs = n_epochs
            self._epoch_size = max(10, self._horizon // n_epochs)
        else:
            self._n_epochs   = max(1, self._horizon // self.EPOCH_SIZE)
            self._epoch_size = self.EPOCH_SIZE

    def epochs(self) -> List[PlanningEpoch]:
        """Return the list of planning epochs for this task."""
        result = []
        for i in range(self._n_epochs):
            start = i * self._epoch_size
            end   = min(self._horizon, (i + 1) * self._epoch_size)
            # Include overlap from previous epoch
            obs_start = max(0, start - self.OVERLAP_MINUTES)

            epoch_flights = [
                f for f in self._base_task.flights
                if f.earliest_minute < end and f.latest_minute >= obs_start
            ]
            if not epoch_flights and i > 0:
                continue   # skip empty epochs

            epoch_task = self._base_task.model_copy(update={
                "task_id":                  f"{self._base_task.task_id}_epoch{i}",
                "title":                    f"{self._base_task.title} — Epoch {i+1}/{self._n_epochs}",
                "planning_horizon_minutes": self._epoch_size + self.OVERLAP_MINUTES,
                "flights":                  epoch_flights,
                "description":              (
                    f"{self._base_task.description}\n"
                    f"[EPOCH {i+1}/{self._n_epochs}: minute {start}–{end} of {self._horizon}]"
                ),
            })

            result.append(PlanningEpoch(
                epoch_id=i,
                start_minute=start,
                end_minute=end,
                task=epoch_task,
            ))
        return result

    def inject_carry_over(
        self,
        epoch: PlanningEpoch,
        memory: EpisodeMemory,
        previous_slots: List[SlotAssignment],
    ) -> PlanningEpoch:
        """Add carry-over constraints from previous epoch to current epoch.

        Carry-over slots from the previous epoch are passed as context so agents
        know which runway slots are already committed at the epoch boundary.
        This is the mechanism by which early decisions cascade into later epochs.
        """
        # Build wake-constraint descriptions for boundary slots
        boundary_slots = [
            s for s in previous_slots
            if s.assigned_minute >= (epoch.start_minute - self.OVERLAP_MINUTES)
        ]
        if boundary_slots:
            constraint_lines = []
            flight_map = {f.flight_id: f for f in epoch.task.flights}
            for slot in boundary_slots:
                f = flight_map.get(slot.flight_id)
                if f:
                    constraint_lines.append(
                        f"  {slot.flight_id} (wake={f.wake_class.value}) "
                        f"@ runway {slot.runway} T+{slot.assigned_minute} "
                        f"(carry-over from previous epoch)"
                    )
            epoch.carry_over_constraints = constraint_lines

            # Write to memory for agent injection
            memory.write("operational", "boundary_slots", constraint_lines, ttl=1)

        return epoch


# ── Cascade Detector ──────────────────────────────────────────────────────────

@dataclass
class CascadeEvent:
    """A cascade: an epoch-t decision that caused an epoch-(t+k) problem."""
    cause_epoch:    int
    effect_epoch:   int
    cause_flight:   str
    effect_flight:  str
    description:    str
    severity:       float    # 0.0 = minor, 1.0 = critical


class CascadeDetector:
    """Detects whether a decision in one epoch caused a problem in a later epoch.

    Detection heuristic: a cascade is suspected when:
    1. An epoch-t boundary slot (last 10 min) involves a Heavy aircraft.
    2. Epoch-(t+1) has a conflict involving the same runway.

    If detected, the conflict is flagged as cascade-triggered, enabling the
    recovery bonus to attribute the recovery to the correct root cause.
    """

    def detect(
        self,
        epoch_slots: Dict[int, List[SlotAssignment]],
        epoch_conflicts: Dict[int, int],
        flight_map: Dict[str, FlightRecord],
        epoch_size: int = 45,
    ) -> List[CascadeEvent]:
        """Detect cascade events across epochs.

        epoch_slots:     {epoch_id: [SlotAssignment]}
        epoch_conflicts: {epoch_id: conflict_count}
        flight_map:      {flight_id: FlightRecord}
        """
        events: List[CascadeEvent] = []

        for epoch_id, slots in epoch_slots.items():
            boundary_end = (epoch_id + 1) * epoch_size
            boundary_slots = [s for s in slots if s.assigned_minute >= boundary_end - 10]

            for bslot in boundary_slots:
                f = flight_map.get(bslot.flight_id)
                if f and f.wake_class == WakeClass.HEAVY:
                    # Heavy at boundary — likely to cascade wake constraint
                    next_epoch = epoch_id + 1
                    if epoch_conflicts.get(next_epoch, 0) > 0:
                        events.append(CascadeEvent(
                            cause_epoch=epoch_id,
                            effect_epoch=next_epoch,
                            cause_flight=bslot.flight_id,
                            effect_flight="unknown",
                            description=(
                                f"Heavy {bslot.flight_id} at boundary "
                                f"(T+{bslot.assigned_minute}) cascaded wake constraint "
                                f"into epoch {next_epoch}"
                            ),
                            severity=0.7,
                        ))

        return events

    def compute_recovery_signal(
        self,
        cascade_events: List[CascadeEvent],
        epoch_scores: Dict[int, float],
    ) -> float:
        """Return a recovery signal based on how well the agent resolved cascades.

        For each cascade: if the effect epoch scored higher than the cause epoch,
        the agent recovered from the cascade (positive signal). If lower, the
        cascade worsened (negative signal).
        """
        if not cascade_events:
            return 0.0

        total = 0.0
        for event in cascade_events:
            cause_score  = epoch_scores.get(event.cause_epoch, 0.5)
            effect_score = epoch_scores.get(event.effect_epoch, 0.5)
            delta = effect_score - cause_score
            total += delta * event.severity

        return round(max(-1.0, min(1.0, total / max(1, len(cascade_events)))), 4)


# ── Long-Horizon Episode Builder ──────────────────────────────────────────────

@dataclass
class EpochResult:
    epoch_id:        int
    composite_score: float
    aman_reward:     float
    dman_reward:     float
    conflict_count:  int
    slots:           List[SlotAssignment]


@dataclass
class LongHorizonResult:
    """Aggregate result across all epochs of a long-horizon episode."""
    n_epochs:        int
    epoch_results:   List[EpochResult]
    cascade_events:  List[CascadeEvent]
    aggregate_score: float    # mean composite across epochs
    worst_epoch:     int
    best_epoch:      int
    recovery_score:  float    # recovery gradient signal
    memory_snapshot: Dict[str, Any]


class LongHorizonEpisodeBuilder:
    """Orchestrates multi-epoch episodes for long-horizon planning evaluation.

    Usage:
        builder = LongHorizonEpisodeBuilder(task=bengaluru_hard, n_epochs=4)
        memory  = EpisodeMemory()
        cascade = CascadeDetector()

        all_slots: List[SlotAssignment] = []
        for epoch in builder.decomposer.epochs():
            epoch = builder.decomposer.inject_carry_over(epoch, memory, all_slots)
            # ... run MultiAgentATCEnvironment on epoch.task ...
            result = env.finalize()
            all_slots.extend(result.slots)
            builder.record_epoch(epoch.epoch_id, result, epoch.task)
            memory.advance_epoch()

        final = builder.aggregate(flight_map)
        print(f"Recovery score: {final.recovery_score:.3f}")
    """

    def __init__(self, task: TaskDefinition, n_epochs: Optional[int] = None) -> None:
        self.decomposer   = HierarchicalPlanDecomposer(task, n_epochs)
        self._results:    List[EpochResult]       = []
        self._all_slots:  Dict[int, List[SlotAssignment]] = {}

    def record_epoch(
        self,
        epoch_id: int,
        composite_score: float,
        aman_reward: float,
        dman_reward: float,
        conflict_count: int,
        slots: List[SlotAssignment],
    ) -> None:
        """Record the result of one epoch."""
        self._results.append(EpochResult(
            epoch_id=epoch_id,
            composite_score=composite_score,
            aman_reward=aman_reward,
            dman_reward=dman_reward,
            conflict_count=conflict_count,
            slots=slots,
        ))
        self._all_slots[epoch_id] = slots

    def aggregate(
        self,
        flight_map: Optional[Dict[str, FlightRecord]] = None,
        memory: Optional[EpisodeMemory] = None,
    ) -> LongHorizonResult:
        if not self._results:
            return LongHorizonResult(
                n_epochs=0, epoch_results=[], cascade_events=[],
                aggregate_score=0.0, worst_epoch=0, best_epoch=0,
                recovery_score=0.0, memory_snapshot={},
            )

        scores = {r.epoch_id: r.composite_score for r in self._results}
        conflicts = {r.epoch_id: r.conflict_count for r in self._results}

        cascade = CascadeDetector()
        cascade_events = cascade.detect(
            self._all_slots, conflicts, flight_map or {}
        ) if flight_map else []

        recovery_score = cascade.compute_recovery_signal(cascade_events, scores)
        composite_list = [r.composite_score for r in self._results]
        aggregate = sum(composite_list) / max(1, len(composite_list))

        best_epoch  = max(self._results, key=lambda r: r.composite_score).epoch_id
        worst_epoch = min(self._results, key=lambda r: r.composite_score).epoch_id

        return LongHorizonResult(
            n_epochs=len(self._results),
            epoch_results=self._results,
            cascade_events=cascade_events,
            aggregate_score=round(aggregate, 4),
            worst_epoch=worst_epoch,
            best_epoch=best_epoch,
            recovery_score=recovery_score,
            memory_snapshot=memory.snapshot() if memory else {},
        )


# ── Prompt augmentation for long-horizon agents ───────────────────────────────

def build_long_horizon_system_addendum(
    epoch_id: int,
    n_epochs: int,
    memory: EpisodeMemory,
    role: str,
    cascade_events: List[CascadeEvent] = (),
) -> str:
    """Build system prompt addendum for long-horizon episodes.

    Injected into the AMAN/DMAN system prompt when running in long-horizon mode.
    Provides epoch context, memory, and cascade warnings.
    """
    lines = [
        "",
        "═══ LONG-HORIZON PLANNING MODE ═══",
        f"You are in EPOCH {epoch_id + 1} of {n_epochs}.",
        f"Your decisions now will constrain epochs {epoch_id + 2}–{n_epochs}.",
        "Think ahead: wake turbulence gaps created now will persist into the next epoch.",
        "",
    ]

    if cascade_events:
        lines.append("⚠ ACTIVE CASCADE ALERTS (from previous epoch decisions):")
        for ev in cascade_events:
            lines.append(f"  {ev.description} (severity={ev.severity:.1f})")
        lines.append("")

    mem_text = memory.render_for_agent(role)
    if mem_text.strip():
        lines.append(mem_text)
        lines.append("")

    lines += [
        "HIERARCHICAL PLANNING INSTRUCTIONS:",
        "  1. STRATEGIC: Identify which flights must be prioritized across the full shift.",
        "  2. TACTICAL:  Assign each flight to a time window within THIS epoch.",
        "  3. OPERATIONAL: Assign exact slot minutes within each window.",
        "  Clearly articulate your strategic intent before specifying slots.",
        "═══════════════════════════════════",
    ]
    return "\n".join(lines)


# ── Reward augmentation for long-horizon episodes ─────────────────────────────

def compute_long_horizon_reward_bonus(
    long_horizon_result: LongHorizonResult,
    role: str,
) -> float:
    """Compute additive long-horizon reward bonus for AMAN or DMAN.

    Components:
        cascade_recovery:  did the agent resolve cascade problems? (+/- 0.15)
        consistency:       were epoch scores consistent (no wild swings)? (+ 0.10)
        planning_quality:  was the best epoch score high? (+ 0.10)
    """
    if long_horizon_result.n_epochs < 2:
        return 0.0

    # Cascade recovery signal
    cascade_bonus = long_horizon_result.recovery_score * 0.15

    # Consistency: penalize high variance across epochs
    scores = [r.composite_score for r in long_horizon_result.epoch_results]
    mean_s = sum(scores) / len(scores)
    variance = sum((s - mean_s) ** 2 for s in scores) / len(scores)
    consistency_bonus = max(0.0, 0.10 - 0.20 * variance)

    # Planning quality: reward for having at least one excellent epoch
    best_score = max(scores)
    planning_bonus = max(0.0, (best_score - 0.70) * 0.33)  # scales 0→0.10 from 0.70→1.0

    total = cascade_bonus + consistency_bonus + planning_bonus
    return round(max(-0.15, min(0.30, total)), 4)
