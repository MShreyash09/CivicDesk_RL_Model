"""
Civic Desk Environment — Enhanced Multi-Axis Grading Engine.

Grading Axes (max total ≈ 2.5 per ticket):
  • Routing score   (0 – 1.0) : exact match on target_queue
  • Priority score  (0 – 0.5) : exact match on priority
  • Action-type     (0 – 0.5) : correct action_type (bonus for Request_Info on ambiguous)
  • Justification   (0 – 0.5) : keyword overlap with policy_keywords
"""

import uuid
import random
from typing import Any, Dict, Optional

# Graceful fallback: openenv may not be installed for local/benchmark use
try:
    from openenv.core.env_server import Environment
except ImportError:
    class Environment:
        """Minimal stub when openenv is not installed."""
        def __init__(self):
            pass

# Support both package-mode and direct-run imports
try:
    from civic_desk.models import (
        CivicDeskObservation,
        CivicDeskAction,
        StepResult,
        CivicDeskState,
    )
    from civic_desk.ticket_bank import TICKET_BANK, get_ticket_by_id
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models import (
        CivicDeskObservation,
        CivicDeskAction,
        StepResult,
        CivicDeskState,
    )
    from ticket_bank import TICKET_BANK, get_ticket_by_id


class CivicDeskEnvironment(Environment):
    """OpenEnv-compatible Civic Service Desk environment with multi-axis grading."""

    def __init__(self) -> None:
        super().__init__()
        self.turn_count: int = 0
        self.episode_id: str = str(uuid.uuid4())
        self.current_ticket: Optional[CivicDeskObservation] = None
        self._current_meta: Dict[str, Any] = {}  # expected answers + keywords

    # ── helpers ────────────────────────────────────────────────────────
    def _get_obs(self) -> CivicDeskObservation:
        """Return current ticket or a placeholder if none exists."""
        if self.current_ticket:
            return self.current_ticket
        return CivicDeskObservation(
            ticket_id="None",
            description="No active ticket",
            policy_snippet="",
            active_resources="",
            difficulty_level="",
        )

    def _load_ticket(self, ticket: Dict[str, Any]) -> None:
        """Populate internal state from a ticket-bank dict."""
        self.current_ticket = CivicDeskObservation(
            ticket_id=ticket["ticket_id"],
            description=ticket["description"],
            policy_snippet=ticket["policy_snippet"],
            active_resources=ticket["active_resources"],
            difficulty_level=ticket["difficulty_level"],
        )
        self._current_meta = {
            "expected_queue": ticket["expected_queue"],
            "expected_priority": ticket["expected_priority"],
            "expected_action_type": ticket["expected_action_type"],
            "policy_keywords": [kw.lower() for kw in ticket.get("policy_keywords", [])],
        }

    # ── reset ─────────────────────────────────────────────────────────
    def reset(self, *, ticket_id: Optional[str] = None, **kwargs: Any) -> StepResult:
        """
        Reset the environment and load a ticket.

        Args:
            ticket_id: If provided, load that exact ticket (deterministic mode).
                       Otherwise, select one at random.
        """
        self.turn_count = 0
        self.episode_id = str(uuid.uuid4())

        if ticket_id:
            ticket = get_ticket_by_id(ticket_id)
            if ticket is None:
                raise ValueError(f"Ticket '{ticket_id}' not found in the bank.")
        else:
            ticket = random.choice(TICKET_BANK)

        self._load_ticket(ticket)

        return StepResult(
            observation=self.current_ticket,
            reward=0.0,
            done=False,
        )

    # ── step (multi-axis grading) ─────────────────────────────────────
    def step(self, action_input: Any) -> StepResult:
        self.turn_count += 1

        # --- extract fields from action (dict or pydantic) ---
        if isinstance(action_input, dict):
            target_queue = action_input.get("target_queue", "")
            priority = action_input.get("priority", "")
            action_type = action_input.get("action_type", "")
            justification = action_input.get("justification", "") or ""
        else:
            target_queue = getattr(action_input, "target_queue", "")
            priority = getattr(action_input, "priority", "")
            action_type = getattr(action_input, "action_type", "")
            justification = getattr(action_input, "justification", "") or ""

        meta = self._current_meta

        # ── 1. Routing score (0 – 1.0) ───────────────────────────────
        routing_score = 1.0 if target_queue == meta["expected_queue"] else 0.0

        # ── 2. Priority score (0 – 0.5) ──────────────────────────────
        priority_score = 0.5 if priority == meta["expected_priority"] else 0.0

        # ── 3. Action-type score (0 – 0.5) ───────────────────────────
        action_type_score = 0.5 if action_type == meta["expected_action_type"] else 0.0

        # ── 4. Justification score (0 – 0.5) ─────────────────────────
        justification_lower = justification.lower()
        keywords = meta["policy_keywords"]
        if keywords:
            hits = sum(1 for kw in keywords if kw in justification_lower)
            keyword_ratio = hits / len(keywords)
        else:
            keyword_ratio = 1.0  # no keywords → full credit
        justification_score = round(0.5 * keyword_ratio, 3)

        # ── total ─────────────────────────────────────────────────────
        total_reward = round(
            routing_score + priority_score + action_type_score + justification_score, 3
        )

        # ── info dict for benchmark consumption ───────────────────────
        correct = routing_score == 1.0 and priority_score == 0.5

        info: Dict[str, Any] = {
            "grading_details": {
                "routing_score": routing_score,
                "priority_score": priority_score,
                "action_type_score": action_type_score,
                "justification_score": justification_score,
                "total_reward": total_reward,
                "correct_routing": routing_score == 1.0,
                "correct_priority": priority_score == 0.5,
                "correct_action_type": action_type_score == 0.5,
                "overall_correct": correct,
            },
            "expected": {
                "queue": meta["expected_queue"],
                "priority": meta["expected_priority"],
                "action_type": meta["expected_action_type"],
            },
            "agent_chose": {
                "queue": target_queue,
                "priority": priority,
                "action_type": action_type,
            },
        }

        return StepResult(
            observation=self._get_obs(),
            reward=total_reward,
            done=True,
            info=info,
        )

    # ── state property ────────────────────────────────────────────────
    @property
    def state(self) -> CivicDeskState:
        return CivicDeskState(
            episode_id=self.episode_id,
            turn_count=self.turn_count,
        )


# ── Quick self-test ───────────────────────────────────────────────────
if __name__ == "__main__":
    env = CivicDeskEnvironment()
    result = env.reset(ticket_id="TKT-101")
    print(f"Loaded: {result.observation.ticket_id} — {result.observation.description}")

    # Simulate a perfect answer
    step_result = env.step({
        "target_queue": "Public_Works",
        "priority": "High",
        "action_type": "Route",
        "justification": "Road blockage by tree branch — public works handles road obstructions.",
    })
    print(f"Reward: {step_result.reward} / 2.5")
    print(f"Details: {step_result.info}")