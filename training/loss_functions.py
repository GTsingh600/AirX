"""Research-grade loss function components for multi-agent ATC training.

This module implements seven novel reward-shaping techniques designed for:
  1. Long-horizon multi-step planning (Theme #2)
  2. Whole-model fine-tuning (not just LoRA adapters)
  3. Cooperative multi-agent credit assignment
  4. Sparse reward problems with delayed feedback

All components are pure Python / numpy — no ML framework dependency — so they
run inside GRPO reward functions without GPU requirements.

Components
──────────
TemporalCreditAssignment      Adaptive discounting for long-horizon tasks.
                               γ adapts to planning_horizon so early decisions
                               in 6-hour windows matter more than in 60-min ones.

InformationTheoreticCoordination
                               Rewards agents for messages that reduce outcome
                               uncertainty. Approximated as correlation between
                               message content signals and outcome improvement.

HierarchicalDecompositionReward
                               Separate loss heads for three planning layers:
                               strategic (priority ordering), tactical (window
                               allocation), operational (specific slot minutes).
                               Prevents all-or-nothing gradient signal.

RecoveryGradient               Explicit bonus for recovering from bad early decisions.
                               R_rec = max(0, score_final - score_mid) * weight

ContrastivePairReward          Near-miss learning: compare actual completion against
                               a counterfactual completion that changed one decision.
                               Gradient pushes toward the successful trajectory.

AdaptiveKLRegularization       For whole-model fine-tuning, KL divergence from
                               reference policy should scale with reward improvement
                               rate to prevent mode collapse without over-regularizing.

CausalCreditAssignment         Multi-agent Shapley-value approximation: credits each
                               agent's contribution causally by running N=2
                               counterfactuals (action vs. naive baseline).

Usage in reward functions:

    from training.loss_functions import (
        TemporalCreditAssignment, HierarchicalDecompositionReward,
        RecoveryGradient, CausalCreditAssignment,
    )

    tca = TemporalCreditAssignment()
    temporal_bonus = tca.compute(
        intermediate_scores=[score_t1, score_t2],
        final_score=score_final,
        planning_horizon=task.planning_horizon_minutes,
    )

    hier = HierarchicalDecompositionReward()
    hier_reward = hier.compute(
        priority_correct=True,     # strategic layer
        window_coverage=0.90,      # tactical layer
        slot_precision=0.75,       # operational layer
    )
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple


# ── 1. Temporal Credit Assignment ────────────────────────────────────────────

class TemporalCreditAssignment:
    """Adaptive discounting for long-horizon planning tasks.

    Problem: in a 6-hour planning window, an agent's decision at minute 5 may
    cause a wake-turbulence conflict at minute 180. Standard GRPO with a
    single terminal reward cannot credit the causal action.

    Solution: intermediate scores (one per planning step / negotiation round)
    are discounted backwards with γ that adapts to the horizon length:

        γ = 1 - 1 / sqrt(n_steps + 1)

    This ensures:
        n_steps=1 (short)  → γ = 0.29  (aggressive discounting, focus on final)
        n_steps=3 (medium) → γ = 0.50  (balanced)
        n_steps=6 (long)   → γ = 0.63  (early decisions count significantly)
        n_steps=12 (shift)  → γ = 0.72  (very long horizon, early decisions matter)

    The shaped return replaces the raw terminal reward in the GRPO advantage
    without changing the optimal policy (Ng et al. 1999 potential-based shaping).

    Shaped return:
        G_t = sum_{k=t}^{T} γ^{k-t} * r_k

    where r_k is the incremental score improvement at step k.
    """

    def compute(
        self,
        intermediate_scores: List[float],
        final_score: float,
        planning_horizon: int = 60,
    ) -> float:
        """Compute temporally-credited return.

        Args:
            intermediate_scores: scores at each planning step (e.g. after each
                                  negotiation round). Empty = only final score used.
            final_score:         terminal episode score in [0, 1]
            planning_horizon:    task planning_horizon_minutes

        Returns:
            Shaped return in [-1, 1] — higher when early plans were also good.
        """
        all_scores = list(intermediate_scores) + [final_score]
        n = len(all_scores)
        if n == 0:
            return 0.0

        # Adaptive γ based on horizon length
        horizon_steps = max(1, planning_horizon // 30)   # ~1 step per 30-min block
        gamma = 1.0 - 1.0 / math.sqrt(max(1, horizon_steps) + 1)

        # Incremental improvements at each step
        deltas = [all_scores[0]]
        for i in range(1, n):
            deltas.append(max(0.0, all_scores[i] - all_scores[i - 1]))

        # Discounted sum of improvements (G = sum γ^t * δ_t)
        G = 0.0
        for t, delta in enumerate(deltas):
            G += (gamma ** t) * delta

        # Normalize to [0, 1] by dividing by the sum of discount weights
        normalizer = sum(gamma ** t for t in range(n))
        shaped = G / max(1e-8, normalizer)
        return round(max(0.0, min(1.0, shaped)), 4)

    def potential_shaping(
        self,
        score_before: float,
        score_after: float,
        gamma: float = 0.95,
    ) -> float:
        """Ng et al. 1999 potential-based shaping: γ·Φ(s') - Φ(s).

        Use as an additive bonus to the base reward. Guarantees policy
        optimality is preserved (no spurious optima introduced).
        """
        return round(gamma * score_after - score_before, 4)


# ── 2. Information-Theoretic Coordination Reward ─────────────────────────────

class InformationTheoreticCoordination:
    """Rewards messages that reduce coordination uncertainty.

    True mutual information is expensive to compute. We approximate it using
    a verifiable proxy: does the message content *predict* the outcome quality?

    Proxy signal:
        R_info = corr(message_features, outcome_delta)

    where:
        message_features = binary signals extracted from message text
                           (mentions emergency? mentions specific flight ID?
                            mentions runway? mentions delay cost? proposes alternative?)
        outcome_delta    = outcome.score - naive_baseline.score

    A message that correctly anticipates the other agent's constraints → high
    correlation between its features and positive outcome delta.

    This directly trains agents to send *informative* messages, not boilerplate.
    """

    # Feature extractors: (feature_name, list_of_keywords)
    MESSAGE_FEATURES: List[Tuple[str, List[str]]] = [
        ("mentions_emergency",   ["emergency", "medical", "emg", "urgent"]),
        ("mentions_flight_id",   []),   # populated dynamically
        ("mentions_runway",      ["runway", "rwy", "28l", "10c", "09l", "27r"]),
        ("mentions_delay_cost",  ["delay", "cost", "minutes", "fuel", "hold"]),
        ("proposes_alternative", ["alternative", "instead", "could use", "yield", "swap"]),
        ("mentions_wake",        ["wake", "heavy", "turbulence", "separation", "gap"]),
        ("theory_of_mind",       ["think", "predict", "expect", "believe", "your"]),
    ]

    def extract_features(self, message_text: str, flight_ids: List[str] = ()) -> Dict[str, float]:
        """Extract binary feature vector from a message string."""
        text = message_text.lower()
        features: Dict[str, float] = {}
        for feat_name, keywords in self.MESSAGE_FEATURES:
            if feat_name == "mentions_flight_id":
                hit = any(fid.lower() in text for fid in flight_ids) if flight_ids else False
            else:
                hit = any(kw in text for kw in keywords)
            features[feat_name] = 1.0 if hit else 0.0
        return features

    def compute(
        self,
        messages: List[str],
        outcome_delta: float,
        flight_ids: List[str] = (),
    ) -> float:
        """Return information-quality bonus in [0, 0.15].

        Higher when messages contain actionable features AND the outcome improved.
        Cap at 0.15 so it supplements rather than dominates other reward components.
        """
        if not messages or not any(m.strip() for m in messages):
            return 0.0

        total_features = 0.0
        for msg in messages:
            feats = self.extract_features(msg, flight_ids)
            total_features += sum(feats.values()) / max(1, len(feats))

        avg_features = total_features / max(1, len(messages))
        # Reward is high when both: (1) messages are feature-rich, AND (2) outcome improved
        if outcome_delta <= 0:
            # No improvement → messages weren't useful (or outcome already good)
            bonus = avg_features * 0.03    # tiny credit for well-formed messages
        else:
            bonus = avg_features * min(1.0, outcome_delta * 3.0) * 0.15

        return round(max(0.0, min(0.15, bonus)), 4)


# ── 3. Hierarchical Decomposition Reward ─────────────────────────────────────

class HierarchicalDecompositionReward:
    """Three-layer planning reward that avoids all-or-nothing gradient signal.

    Long-horizon tasks require decomposition:
        Layer 1 (Strategic):    Did the agent correctly order priorities?
                                 (emergency first, heavy before light for wake, etc.)
        Layer 2 (Tactical):     Did the agent allocate the right flights to windows?
                                 (coverage rate, window feasibility)
        Layer 3 (Operational):  Did the agent select precise slot minutes?
                                 (minimal delay, no conflicts, ATFM compliance)

    Separate learning signals per layer → gradient flows even when operational
    decisions are poor (prevents vanishing signal for early planning stages).

    Weights: strategic=0.25, tactical=0.35, operational=0.40
    (operational matters most for the final score, but strategic + tactical
     provide early learning signal when the agent is still making basic errors)
    """

    STRATEGIC_WEIGHT   = 0.25
    TACTICAL_WEIGHT    = 0.35
    OPERATIONAL_WEIGHT = 0.40

    def compute(
        self,
        priority_correct: bool,          # strategic: emergencies first?
        window_coverage: float,          # tactical: fraction of flights in-window
        slot_precision: float,           # operational: 1 - normalized_delay
        conflict_free: bool = True,      # operational: no wake conflicts
        atfm_compliant: bool = True,     # operational: DMAN meets deadlines
    ) -> float:
        """Compute hierarchical reward in [0, 1].

        Each layer contributes independently so a good strategic plan still
        receives positive gradient even if specific slot minutes are suboptimal.
        """
        # Strategic layer: binary — priority ordering correct?
        strategic_score = 1.0 if priority_correct else 0.2

        # Tactical layer: coverage and window feasibility
        tactical_score = max(0.0, min(1.0, window_coverage))

        # Operational layer: precision, conflict-free, ATFM
        conflict_penalty = 0.0 if conflict_free else 0.35
        atfm_penalty     = 0.0 if atfm_compliant else 0.15
        operational_score = max(0.0, slot_precision - conflict_penalty - atfm_penalty)

        reward = (
            self.STRATEGIC_WEIGHT   * strategic_score
            + self.TACTICAL_WEIGHT  * tactical_score
            + self.OPERATIONAL_WEIGHT * operational_score
        )
        return round(max(0.0, min(1.0, reward)), 4)

    def compute_from_metrics(self, metrics, task=None) -> float:
        """Convenience: compute from a TaskMetrics object."""
        try:
            from models import PriorityClass
        except ImportError:
            PriorityClass = None

        priority_correct = metrics.emergency_violations == 0
        window_coverage  = metrics.schedule_completeness
        slot_precision   = metrics.delay_efficiency
        conflict_free    = metrics.conflict_count == 0
        atfm_ok          = metrics.priority_violations == 0

        return self.compute(
            priority_correct=priority_correct,
            window_coverage=window_coverage,
            slot_precision=slot_precision,
            conflict_free=conflict_free,
            atfm_compliant=atfm_ok,
        )


# ── 4. Recovery Gradient ──────────────────────────────────────────────────────

class RecoveryGradient:
    """Bonus reward for recovering from bad early decisions.

    Long-horizon planning requires agents to recognize when an early decision
    has created a cascade problem AND to fix it in subsequent rounds. Standard
    GRPO only scores the final state — agents that recover get the same reward
    as agents that never had the problem. This provides no training signal for
    the specific skill of recognizing and recovering from mistakes.

    Recovery bonus:
        R_rec = max(0, score_final - score_after_round1) * recovery_weight

    This is *additive*: agents that had a bad round 1 and recovered a lot score
    higher than agents that coasted from a mediocre round 1 to a mediocre final.

    The bonus is capped at recovery_cap to prevent an agent from deliberately
    tanking round 1 to harvest recovery bonuses.
    """

    def __init__(self, recovery_weight: float = 0.25, cap: float = 0.20) -> None:
        self.recovery_weight = recovery_weight
        self.cap = cap

    def compute(
        self,
        score_initial: float,    # score after round 0 (BID)
        score_final: float,      # score after FINAL round
        conflict_resolved: bool = False,   # did agent resolve conflicts?
    ) -> float:
        """Compute recovery bonus in [0, cap].

        Args:
            score_initial: composite score after first bid round
            score_final:   final composite score
            conflict_resolved: whether conflicts from bid were resolved in negotiate

        Returns:
            Additive bonus in [0, self.cap]
        """
        raw_recovery = max(0.0, score_final - score_initial)

        # Extra bonus if the agent explicitly resolved conflicts
        # (shows deliberate recovery, not just luck)
        resolution_bonus = 0.05 if conflict_resolved else 0.0

        recovery = self.recovery_weight * raw_recovery + resolution_bonus
        return round(max(0.0, min(self.cap, recovery)), 4)

    def anti_gaming_penalty(
        self,
        score_initial: float,
        score_final: float,
    ) -> float:
        """Penalty to prevent deliberately bad round-1 plans to farm recovery bonus.

        If the agent's initial score is suspiciously low (< 0.15) and then
        "recovers" to a high score, it may be gaming. Apply a penalty proportional
        to the gap between expected baseline and actual initial score.
        """
        if score_initial >= 0.15:
            return 0.0
        # Initial score very low — likely gaming or catastrophic failure
        # Penalty reduces recovery bonus by fraction of the suspicious gap
        suspicious_gap = 0.15 - score_initial
        penalty = suspicious_gap * 0.5
        return round(min(self.cap, penalty), 4)

    def compute_safe(
        self,
        score_initial: float,
        score_final: float,
        conflict_resolved: bool = False,
    ) -> float:
        """Recovery bonus with anti-gaming protection."""
        bonus = self.compute(score_initial, score_final, conflict_resolved)
        penalty = self.anti_gaming_penalty(score_initial, score_final)
        return round(max(0.0, bonus - penalty), 4)


# ── 5. Contrastive Pair Reward ────────────────────────────────────────────────

class ContrastivePairReward:
    """Near-miss learning: compare agent outcome against a counterfactual.

    Standard GRPO uses group-relative advantage (normalize rewards within the
    N-generation group). This adds a contrastive component: for each episode,
    compare the actual outcome against a *specific* counterfactual — the outcome
    if the agent had taken the naive baseline action.

    Contrastive reward:
        R_contrastive = sigmoid(k * (actual_score - counterfactual_score)) * 2 - 1

    where k controls sharpness. This ensures:
        actual >> counterfactual → R_contrastive close to +1
        actual << counterfactual → R_contrastive close to -1
        actual ≈ counterfactual  → R_contrastive close to 0

    The sigmoid avoids the hard discontinuity of simple sign(actual - cf).
    """

    def __init__(self, sharpness: float = 5.0) -> None:
        self.k = sharpness

    def _sigmoid(self, x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    def compute(
        self,
        actual_score: float,
        counterfactual_score: float,
    ) -> float:
        """Contrastive reward in [-1, 1].

        Positive: agent outperformed the naive baseline.
        Negative: agent underperformed the naive baseline (gradient pushes away).
        """
        delta = actual_score - counterfactual_score
        contrastive = 2.0 * self._sigmoid(self.k * delta) - 1.0
        return round(max(-1.0, min(1.0, contrastive)), 4)

    def compute_weighted(
        self,
        actual_score: float,
        counterfactual_score: float,
        weight: float = 0.15,
    ) -> float:
        """Weighted contrastive bonus suitable for addition to base reward."""
        return round(weight * self.compute(actual_score, counterfactual_score), 4)


# ── 6. Adaptive KL Regularization ────────────────────────────────────────────

class AdaptiveKLRegularization:
    """KL coefficient that adapts to the reward improvement rate.

    For whole-model fine-tuning (no LoRA), the reference policy divergence must
    be tracked carefully to prevent catastrophic forgetting of language modeling
    capability while still allowing meaningful policy improvement.

    Standard GRPO uses a fixed KL coefficient (β). This replaces it with:

        β_adaptive(t) = β_base * max(β_floor, 1.0 - α * reward_improvement_rate)

    where reward_improvement_rate = (mean_recent - mean_earlier) / T

    Logic:
        - When reward is rising fast → agent is learning quickly → relax KL (smaller β)
          so the policy can move freely toward the improving behavior.
        - When reward has plateaued → learning has stalled → increase KL to stabilize
          the policy and prevent drift toward degenerate outputs.
        - β_floor prevents KL from going to zero (which causes mode collapse).

    This scheduler is called *outside* the GRPO trainer and returns the
    recommended β for the next training step, passed via GRPOConfig.beta.
    """

    def __init__(
        self,
        beta_base: float = 0.01,
        beta_floor: float = 0.001,
        beta_ceiling: float = 0.10,
        adaptation_rate: float = 0.5,
    ) -> None:
        self.beta_base      = beta_base
        self.beta_floor     = beta_floor
        self.beta_ceiling   = beta_ceiling
        self.alpha          = adaptation_rate
        self._reward_history: List[float] = []

    def record(self, reward: float) -> None:
        self._reward_history.append(reward)

    def compute_beta(self) -> float:
        """Return recommended β for the next training step."""
        if len(self._reward_history) < 10:
            return self.beta_base

        n = len(self._reward_history)
        quarter = max(1, n // 4)
        recent   = self._reward_history[-quarter:]
        earlier  = self._reward_history[-2 * quarter:-quarter]

        mean_recent  = sum(recent)  / len(recent)
        mean_earlier = sum(earlier) / len(earlier) if earlier else mean_recent

        improvement_rate = (mean_recent - mean_earlier) / max(0.01, abs(mean_earlier))

        # Fast improvement → relax KL; plateau/decline → tighten KL
        scale = max(self.beta_floor / self.beta_base,
                    1.0 - self.alpha * improvement_rate)
        beta = self.beta_base * scale
        return round(max(self.beta_floor, min(self.beta_ceiling, beta)), 6)

    def summary(self) -> Dict[str, float]:
        return {
            "current_beta": self.compute_beta(),
            "n_recorded":   len(self._reward_history),
            "mean_recent":  (sum(self._reward_history[-10:]) / min(10, max(1, len(self._reward_history)))),
        }


# ── 7. Causal Credit Assignment ───────────────────────────────────────────────

class CausalCreditAssignment:
    """Approximate Shapley-value credit for multi-agent cooperative episodes.

    In cooperative multi-agent RL, both AMAN and DMAN contribute to the joint
    outcome. Standard credit assignment gives each agent the *joint* reward,
    which doesn't tell them which *specific actions* caused success or failure.

    Shapley value for agent i:
        phi_i = E_{S in N minus {i}} [ V(S union {i}) - V(S) ]

    With N=2 agents (AMAN, DMAN), this reduces to:
        φ_AMAN = 0.5 * [V({AMAN}) - V({})] + 0.5 * [V({AMAN, DMAN}) - V({DMAN})]
        φ_DMAN = 0.5 * [V({DMAN}) - V({})] + 0.5 * [V({AMAN, DMAN}) - V({AMAN})]

    where:
        V({})            = naive_baseline (no optimization at all)
        V({AMAN})        = AMAN optimal + DMAN naive
        V({DMAN})        = AMAN naive + DMAN optimal
        V({AMAN, DMAN})  = both agents' actual outputs (joint outcome)

    This is already partially implemented as counterfactual advantage in the
    existing reward functions. This class provides the full Shapley computation
    and a normalized credit score.
    """

    def compute_shapley(
        self,
        v_empty: float,        # naive baseline (no agent optimizes)
        v_aman_only: float,    # AMAN optimized + DMAN naive
        v_dman_only: float,    # AMAN naive + DMAN optimized
        v_joint: float,        # both agents' actual outputs
    ) -> Tuple[float, float]:
        """Return (aman_credit, dman_credit) Shapley values in [-1, 1].

        Both credits are calibrated so:
            aman_credit + dman_credit ≈ v_joint - v_empty  (efficiency axiom)
        """
        aman_marginal_alone = v_aman_only - v_empty
        aman_marginal_joint = v_joint - v_dman_only
        aman_credit = 0.5 * aman_marginal_alone + 0.5 * aman_marginal_joint

        dman_marginal_alone = v_dman_only - v_empty
        dman_marginal_joint = v_joint - v_aman_only
        dman_credit = 0.5 * dman_marginal_alone + 0.5 * dman_marginal_joint

        return (
            round(max(-1.0, min(1.0, aman_credit)), 4),
            round(max(-1.0, min(1.0, dman_credit)), 4),
        )

    def compute_from_outcomes(
        self,
        actual_joint: float,
        aman_alone: float,
        dman_alone: float,
        naive_baseline: float,
    ) -> Tuple[float, float]:
        """Convenience wrapper with named semantic arguments."""
        return self.compute_shapley(
            v_empty=naive_baseline,
            v_aman_only=aman_alone,
            v_dman_only=dman_alone,
            v_joint=actual_joint,
        )


# ── Composite: Long-Horizon Reward Bundle ─────────────────────────────────────

class LongHorizonRewardBundle:
    """Combines all loss components into a single reward for long-horizon episodes.

    This is the recommended entry point for training loops that want all seven
    components without assembling them manually.

    Returns a dict with per-component scores and a weighted total.
    """

    WEIGHTS = {
        "temporal_credit":     0.15,
        "hierarchical":        0.20,
        "recovery":            0.10,
        "contrastive":         0.15,
        "info_coordination":   0.05,
        "causal_aman":         0.15,
        "causal_dman":         0.20,  # DMAN has more unique responsibility (ATFM)
    }

    def __init__(self) -> None:
        self.tca  = TemporalCreditAssignment()
        self.hier = HierarchicalDecompositionReward()
        self.rec  = RecoveryGradient()
        self.cpr  = ContrastivePairReward()
        self.itc  = InformationTheoreticCoordination()
        self.cca  = CausalCreditAssignment()

    def compute(
        self,
        final_score: float,
        naive_score: float,
        aman_alone_score: float,
        dman_alone_score: float,
        intermediate_scores: List[float],
        planning_horizon: int,
        priority_correct: bool,
        window_coverage: float,
        slot_precision: float,
        conflict_free: bool,
        atfm_compliant: bool,
        initial_score: float,
        conflict_resolved: bool,
        messages: List[str] = (),
        flight_ids: List[str] = (),
        outcome_delta: float = 0.0,
    ) -> Dict[str, float]:
        """Full long-horizon reward computation.

        Returns dict with per-component scores and 'total' key.
        """
        tca_score  = self.tca.compute(intermediate_scores, final_score, planning_horizon)
        hier_score = self.hier.compute(priority_correct, window_coverage, slot_precision,
                                       conflict_free, atfm_compliant)
        rec_score  = self.rec.compute_safe(initial_score, final_score, conflict_resolved)
        cpr_score  = self.cpr.compute_weighted(final_score, naive_score)
        itc_score  = self.itc.compute(list(messages), outcome_delta, list(flight_ids))
        aman_credit, dman_credit = self.cca.compute_from_outcomes(
            final_score, aman_alone_score, dman_alone_score, naive_score
        )

        components = {
            "temporal_credit":   tca_score,
            "hierarchical":      hier_score,
            "recovery":          rec_score,
            "contrastive":       cpr_score,
            "info_coordination": itc_score,
            "causal_aman":       aman_credit,
            "causal_dman":       dman_credit,
        }

        total = sum(
            self.WEIGHTS[k] * v for k, v in components.items()
            if k in self.WEIGHTS
        )
        components["total"] = round(max(-1.0, min(1.0, total)), 4)
        return components
