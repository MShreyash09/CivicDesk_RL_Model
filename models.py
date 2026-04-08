from pydantic import BaseModel
from typing import Literal, Optional


class CivicDeskObservation(BaseModel):
    """What the AI sees — the civic service ticket to triage.

    Fields:
        ticket_id:        Unique ticket identifier (e.g. "TKT-201").
        description:      The citizen's complaint or report text.
        policy_snippet:   Relevant municipal policy the AI should apply.
        active_resources: Currently available crews / assets for this area.
        difficulty_level: Difficulty tier — easy | medium | hard | ambiguous.
    """
    ticket_id: str
    description: str
    policy_snippet: str
    active_resources: str
    difficulty_level: str


class CivicDeskAction(BaseModel):
    """What the AI does — the triage decision.

    Fields:
        target_queue:  Department to route the ticket to.
        priority:      Urgency level assigned to the ticket.
        action_type:   Dispatch action — Route, Request_Info (for vague reports),
                       Escalate (for critical multi-dept situations), or Resolve.
        justification: Free-text reasoning citing relevant policy terms.
    """
    target_queue: Literal["Police", "Public_Works", "Sanitation", "Water"]
    priority: Literal["Low", "Medium", "High", "Critical"]
    action_type: Literal["Route", "Request_Info", "Escalate", "Resolve"]
    justification: Optional[str] = None


class StepResult(BaseModel):
    """Standard OpenEnv return type wrapping observation + reward."""
    observation: CivicDeskObservation
    reward: float
    done: bool
    info: dict = {}


class CivicDeskState(BaseModel):
    """Serialisable environment state snapshot."""
    episode_id: str
    turn_count: int