"""Backwards-compatibility shim for training/reward code that imports ChallengeGenerator.

The primary curriculum logic has moved to multi_agent/adapter.py which implements
ContextAdaptiveCurriculum — a *self-adapting* system that diagnoses agent weaknesses
and generates targeted scenarios, replacing the old adversarial self-play approach.

This module re-exports ChallengeGenerator as a thin subclass of
ContextAdaptiveCurriculum so all existing imports continue to work without changes.

Mutation catalogue (unchanged):
  TIGHTEN_WINDOW          → squeeze flight's [earliest, latest] by N minutes each side
  INJECT_EMERGENCY        → insert a new EMERGENCY/MEDICAL flight into the scenario
  INCREASE_WEATHER_PENALTY → degrade runway capacity (forces AMAN/DMAN onto fewer slots)
  ADD_ATFM_DEADLINE       → add hard network slot constraint to a departure
  CLOSE_RUNWAY_WINDOW     → runway unavailable for T minutes around peak hour
  ADD_CONFLICTING_FLIGHT  → inject Heavy arrival just before a Light departure (wake trap)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

try:
    from .adapter import (
        ContextAdaptiveCurriculum,
        AdaptationContext,
        extract_skill_scores,
        SKILL_DIMENSIONS,
    )
    from ..models import TaskDefinition
    from .models import GeneratorAction, GeneratorMutation, MutationType
except ImportError:
    from multi_agent.adapter import (
        ContextAdaptiveCurriculum,
        AdaptationContext,
        extract_skill_scores,
        SKILL_DIMENSIONS,
    )
    from models import TaskDefinition
    from multi_agent.models import GeneratorAction, GeneratorMutation, MutationType


# Constants preserved for callers that import them directly
ESCALATION_THRESHOLD = 0.65
FLOOR_THRESHOLD      = 0.30
MAX_MUTATIONS_PER_EPISODE = 3
MIN_WINDOW_WIDTH = 8
MASTERY_WINDOW = 10
MASTERY_THRESHOLD = 0.55


class ChallengeGenerator(ContextAdaptiveCurriculum):
    """Backwards-compatible interface over ContextAdaptiveCurriculum.

    All existing call sites (reward_functions.py, train_grpo.py, evaluate())
    work without modification. The underlying logic is now self-adapting:
    it identifies which skill dimension is weakest and generates targeted
    scenarios instead of blindly escalating composite difficulty.

    Old API surface preserved:
        .update(controller_score)
        .record(task_id, mutations_used, composite_score)
        .mutate(base_task, generator_action) → (task, is_solvable)
        .compute_reward(controller_score, is_solvable) → float
        .get_weak_mutations(threshold) → List[str]
        .mastery_report() → Dict
        .ema_score  (property)
        .difficulty_level  (property)
    """

    def __init__(self, seed: int = 0) -> None:
        super().__init__(seed=seed)
        # Legacy aliases so old attribute access doesn't break
        self._mutation_history: List[Dict] = []

    # ── Old API compatibility ─────────────────────────────────────────────────

    def update(self, controller_score: float) -> None:
        """Update rolling composite history (delegates to parent)."""
        self._composite_history.append(max(0.0, min(1.0, controller_score)))

    def record(
        self,
        task_id: str,
        mutations_used: Optional[List[str]] = None,
        composite_score: float = 0.5,
        *,
        skill_scores: Optional[Dict[str, float]] = None,
        composite: Optional[float] = None,
    ) -> None:
        """Record per-scenario result — accepts both old and new call signatures.

        Old: record(task_id, mutations_used, composite_score)
        New: record(task_id, skill_scores={...}, composite=0.6)
        """
        # New-style call via keyword args takes precedence
        final_composite = composite if composite is not None else composite_score
        if skill_scores is None:
            skill_scores = {dim: final_composite for dim in SKILL_DIMENSIONS}
        super().record(task_id=task_id, skill_scores=skill_scores, composite=final_composite)

    def mutate(
        self,
        base_task: TaskDefinition,
        generator_action: Optional[GeneratorAction] = None,
    ) -> Tuple[TaskDefinition, bool]:
        """Apply adaptive mutations. Returns (mutated_task, is_solvable).

        Wraps adapt() and discards the AdaptationContext (old callers don't use it).
        New callers should use adapt() directly to access the context for dynamic
        reward weighting.
        """
        mutated, solvable, _ctx = self.adapt(base_task, generator_action)
        return mutated, solvable

    def get_weak_mutations(self, threshold: float = MASTERY_THRESHOLD) -> List[str]:
        """Return mutation types that map to the currently weakest skill dimensions.

        Legacy shim: old code expected a list of MutationType strings. We translate
        the weakest skill dimensions into their associated mutations.
        """
        gaps = self._skill_profile.gap_vector()
        weak_dims = [d for d, g in gaps.items() if g > (1.0 - threshold)]
        weak_mutations: List[str] = []
        from .adapter import _SKILL_MUTATION_MAP
        for dim in weak_dims:
            for mtype in _SKILL_MUTATION_MAP.get(dim, []):
                if mtype.value not in weak_mutations:
                    weak_mutations.append(mtype.value)
        return weak_mutations

    def mastery_report(self) -> Dict:
        """Skill profile report (replaces old mastery_report)."""
        report = self.diagnostic_report()
        # Add legacy keys for callers that read them
        report["task_mastery"]     = {}
        report["mutation_mastery"] = {}
        report["weak_mutations"]   = self.get_weak_mutations()
        return report
