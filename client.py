# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Civic Desk Environment Client — updated to match the ticket-based schema."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import CivicDeskAction, CivicDeskObservation


class CivicDeskEnv(
    EnvClient[CivicDeskAction, CivicDeskObservation, State]
):
    """
    Client for the Civic Desk Environment.

    Uses the current ticket-based schema (CivicDeskAction / CivicDeskObservation).

    Example:
        >>> with CivicDeskEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.description)
        ...
        ...     action = CivicDeskAction(
        ...         target_queue="Public_Works",
        ...         priority="High",
        ...         action_type="Route",
        ...         justification="Road blockage — public works policy applies.",
        ...     )
        ...     result = client.step(action)
        ...     print(f"Reward: {result.reward}")
    """

    def _step_payload(self, action: CivicDeskAction) -> Dict:
        """
        Convert CivicDeskAction to JSON payload for step message.

        Maps the four action fields (target_queue, priority, action_type,
        justification) to the dict expected by the server.
        """
        return {
            "target_queue": action.target_queue,
            "priority": action.priority,
            "action_type": action.action_type,
            "justification": action.justification or "",
        }

    def _parse_result(self, payload: Dict) -> StepResult[CivicDeskObservation]:
        """
        Parse server response into StepResult[CivicDeskObservation].

        Handles potential nested 'observation' keys from HF Spaces proxy.
        """
        obs_data = payload.get("observation", {})
        # HF sometimes double-nests:  {"observation": {"observation": {...}}}
        if isinstance(obs_data, dict) and "observation" in obs_data:
            obs_data = obs_data["observation"]

        observation = CivicDeskObservation(
            ticket_id=obs_data.get("ticket_id", ""),
            description=obs_data.get("description", ""),
            policy_snippet=obs_data.get("policy_snippet", ""),
            active_resources=obs_data.get("active_resources", ""),
            difficulty_level=obs_data.get("difficulty_level", ""),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse server response into State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
