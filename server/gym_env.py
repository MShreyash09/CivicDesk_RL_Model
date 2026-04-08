"""
Civic Desk Gym Environment — Heterogeneous RL Architecture.

This module exposes a standard `gymnasium` interface for training an RL agent (PPO) 
to dispatch tickets correctly. 

State (Box array of float32, length=10):
  0: ticket_active (0.0 or 1.0)
  1: target_queue  (0=Police, 1=PW, 2=Sanitation, 3=Water)
  2: priority      (0=Low, 1=Medium, 2=High, 3=Critical)
  3: difficulty    (0=Easy, 1=Medium, 2=Hard, 3=Ambiguous)
  4: sla_timer     (turns waiting)
  5: res_police    (available units)
  6: res_pw
  7: res_san
  8: res_water
  9: turn_count    (current turn in the shift / 30.0)

Actions (Discrete 5):
  0: Wait
  1: Dispatch Police
  2: Dispatch Public Works
  3: Dispatch Sanitation
  4: Dispatch Water
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from typing import Optional, Dict, Any, Tuple
import sys
import os

# Add parent path so we can import ticket bank
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from civic_desk.ticket_bank import TICKET_BANK
except ImportError:
    from ticket_bank import TICKET_BANK

# Mappings for standardization
QUEUE_MAP = {"Police": 0, "Public_Works": 1, "Sanitation": 2, "Water": 3}
PRIO_MAP  = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
DIFF_MAP  = {"easy": 0, "medium": 1, "hard": 2, "ambiguous": 3}

# Constants
MAX_TURNS = 30
MAX_SLA = 5
MAX_RESOURCES = 3


class CivicDeskGymEnv(gym.Env):
    """
    A continuous multi-turn dispatch environment compatible with Stable-Baselines3.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, use_advanced_rules: bool = False):
        super().__init__()
        
        self.use_advanced_rules = use_advanced_rules
        
        # Action space: 0:Wait, 1:Police, 2:PW, 3:Sanitation, 4:Water
        self.action_space = spaces.Discrete(5)
        
        # State space: 10 dimensions, normalized ideally but raw floats work for PPO
        # [active_flag, queue_id, prio_id, diff_id, sla, res_pol, res_pw, res_san, res_wat, time]
        low = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([1.0, 3.0, 3.0, 3.0, 10.0, 5.0, 5.0, 5.0, 5.0, 30.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        self.turn_count = 0
        self.active_ticket = None
        self.sla_timer = 0
        
        # Resource pools (index matches Action - 1)
        # 0=Police, 1=PW, 2=San, 3=Wat
        self.resources = np.array([MAX_RESOURCES] * 4, dtype=np.float32)
        self.busy_timers = {0: [], 1: [], 2: [], 3: []}

    def _spawn_ticket(self):
        """Randomly selects a ticket and maps it to integers."""
        ticket = random.choice(TICKET_BANK)
        self.active_ticket = {
            "ticket_id": ticket["ticket_id"],
            "queue": QUEUE_MAP.get(ticket["expected_queue"], 1),
            "priority": PRIO_MAP.get(ticket["expected_priority"], 1),
            "difficulty": DIFF_MAP.get(ticket["difficulty_level"], 1)
        }
        self.sla_timer = 0

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(10, dtype=np.float32)
        if self.active_ticket is not None:
            obs[0] = 1.0
            obs[1] = float(self.active_ticket["queue"])
            obs[2] = float(self.active_ticket["priority"])
            obs[3] = float(self.active_ticket["difficulty"])
            obs[4] = float(self.sla_timer)
        
        obs[5] = self.resources[0]
        obs[6] = self.resources[1]
        obs[7] = self.resources[2]
        obs[8] = self.resources[3]
        obs[9] = float(self.turn_count)
        return obs
    
    def _update_resources(self):
        """Free up resources that have finished their job."""
        for dept_idx in range(4):
            new_timers = []
            for t in self.busy_timers[dept_idx]:
                if t - 1 <= 0:
                    self.resources[dept_idx] += 1
                else:
                    new_timers.append(t - 1)
            self.busy_timers[dept_idx] = new_timers

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.turn_count = 0
        self.resources = np.array([MAX_RESOURCES] * 4, dtype=np.float32)
        self.busy_timers = {0: [], 1: [], 2: [], 3: []}
        self._spawn_ticket()
        return self._get_obs(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self.turn_count += 1
        reward = 0.0
        
        # Tick active resources
        if self.use_advanced_rules:
            self._update_resources()
            
        info = {
            "action_taken": action,
            "was_correct": False,
            "msg": ""
        }

        # Handle Action
        if action == 0:  # WAIT
            if self.active_ticket is None:
                reward = 0.0  # Safe wait
                info["msg"] = "Waiting (No active ticket)"
            else:
                if self.use_advanced_rules:
                    reward = -0.1  # SLA penalty
                self.sla_timer += 1
                info["msg"] = f"Waiting (SLA={self.sla_timer})"
                
                if self.sla_timer > MAX_SLA:
                    reward = -2.0  # Ticket expired!
                    self.active_ticket = None
                    info["msg"] = "Ticket expired due to SLA breach!"

        else:  # DISPATCH (1..4)
            dept_idx = action - 1
            if self.active_ticket is None:
                reward = -1.0  # Wasted dispatch
                info["msg"] = "Invalid dispatch: No active ticket."
            else:
                # Check resource availability (if advanced rules on)
                if self.use_advanced_rules and self.resources[dept_idx] <= 0:
                    reward = -1.0
                    info["msg"] = f"Failed dispatch: Department {dept_idx} has no free units."
                    self.sla_timer += 1
                else:
                    # Valid dispatch attempt
                    if dept_idx == self.active_ticket["queue"]:
                        # CORRECT
                        reward = 1.0
                        info["was_correct"] = True
                        info["msg"] = f"Correct dispatch of dept {dept_idx}!"
                    else:
                        # INCORRECT
                        reward = -1.0
                        info["msg"] = f"Wrong dispatch! Sent {dept_idx}, needed {self.active_ticket['queue']}."
                    
                    # Lock the resource for a few turns
                    if self.use_advanced_rules:
                        self.resources[dept_idx] -= 1
                        lock_turns = 2 if self.active_ticket["priority"] < 2 else 4
                        self.busy_timers[dept_idx].append(lock_turns)
                    
                    # Ticket is resolved (pass or fail)
                    self.active_ticket = None

        # Spawn a new ticket randomly if we don't have one
        if self.active_ticket is None and random.random() < 0.6:
            self._spawn_ticket()

        terminated = self.turn_count >= MAX_TURNS
        truncated = False

        return self._get_obs(), reward, terminated, truncated, info


# Sanity check
if __name__ == "__main__":
    env = CivicDeskGymEnv(use_advanced_rules=False)
    obs, _ = env.reset()
    print("Initial Obs:", obs)
    obs, reward, done, _, info = env.step(1)
    print("Step 1:", reward, info)
