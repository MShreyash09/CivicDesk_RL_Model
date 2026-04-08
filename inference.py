"""
Civic Desk 
This script executes the Heterogeneous RL Policy (PPO + LLM Perception)
adhering strictly to the Pre-Submission Checklist formatting.
"""

import os
import json
import re
import time
import numpy as np
from openai import OpenAI
from stable_baselines3 import PPO

# Environment Map Imports
from server.gym_env import CivicDeskGymEnv
from server.gym_env import QUEUE_MAP, PRIO_MAP, DIFF_MAP

# Environment Variables 
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1/")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional: if using from_docker_image()
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

def state_normalizer(env_obs: np.ndarray, llm_data: dict) -> np.ndarray:
    """Merges physical environment state (resources/time) with LLM perceived state."""
    norm_state = env_obs.copy()
    norm_state[1] = float(QUEUE_MAP.get(llm_data.get("target_queue", "Public_Works"), 1))
    norm_state[2] = float(PRIO_MAP.get(llm_data.get("priority", "Medium"), 1))
    return norm_state

def main():
    print("[START]")
    print(f"[STEP] Initializing Hackathon Compliant PPO Agent connecting to: {API_BASE_URL}")

    #  OpenAI Client 
    # Initialize the client using exactly the specified env variables
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN or "sk-dummy-key"  # Fallback to prevent crash if running locally without key
    )
    
    use_mock = not HF_TOKEN
    
    # ── 1. Load PPO Model ──
    model_path = os.path.join(os.path.dirname(__file__), "ppo_civic_dispatcher.zip")
    if not os.path.exists(model_path):
        print("[STEP] ❌ PPO Model missing. Run `python train_rl.py` first.")
        print("[END]")
        return
        
    model = PPO.load(model_path)
    print(f"[STEP] RL Agent Loaded: stable_baselines3 MlpPolicy | Vector Length: 10")

    # ── 2. Run Continuous Environment ──
    env = CivicDeskGymEnv(use_advanced_rules=True)
    obs, _ = env.reset()

    for step in range(1, 11):
        if env.active_ticket:
            ticket_id = env.active_ticket['ticket_id']
            diff_str = list(DIFF_MAP.keys())[env.active_ticket['difficulty']]
            
            print(f"[STEP] [Turn {step}/30] Ticket Array Trigger: {ticket_id}")

            if use_mock:
                print(f"[STEP] [Turn {step}/30] Mocking LLM Perception Matrix (No HF_TOKEN)...")
                llm_parsed = {
                    "target_queue": list(QUEUE_MAP.keys())[env.active_ticket['queue']],
                    "priority": list(PRIO_MAP.keys())[env.active_ticket['priority']],
                    "difficulty": diff_str
                }
            else:
                try:
                    print(f"[STEP] [Turn {step}/30] Contacting OpenAI API for NLP State Compression...")
                    from ticket_bank import get_ticket_by_id
                    raw_ticket = get_ticket_by_id(ticket_id)
                    
                    prompt = f"""You are the perception module of an RL agent.
DESCRIPTION:  {raw_ticket['description']}
POLICY:       {raw_ticket['policy_snippet']}

Analyze the raw unstructured text.
Return ONLY a JSON object:
{{
  "target_queue": "Police" | "Public_Works" | "Sanitation" | "Water",
  "priority": "Low" | "Medium" | "High" | "Critical",
  "difficulty": "easy" | "medium" | "hard" | "ambiguous"
}}"""
                    # LLM implementation
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=150
                    )
                    
                    ai_output = response.choices[0].message.content
                    json_match = re.search(r"\{.*\}", ai_output, re.DOTALL)
                    if json_match:
                        llm_parsed = json.loads(json_match.group())
                    else:
                        print(f"[STEP] [Turn {step}/30] Hallucination fallback triggered.")
                        llm_parsed = {"target_queue": "Public_Works", "priority": "Medium", "difficulty": diff_str}
                except Exception as e:
                    print(f"[STEP] [Turn {step}/30] LLM API Error: {e}")
                    llm_parsed = {"target_queue": "Public_Works", "priority": "Medium", "difficulty": diff_str}

            # Normalise and Predict
            obs = state_normalizer(obs, llm_parsed)
        else:
            print(f"[STEP] [Turn {step}/30] Active capacity scan (No ticket)")

        # RL Decision
        action, _ = model.predict(obs, deterministic=True)
        act_map = {0: "Wait", 1: "Police", 2: "Pub_Works", 3: "Sanitation", 4: "Water"}
        
        # Env Sequence
        obs, reward, done, _, _ = env.step(action)
        res = [int(x) for x in obs[5:9]]
        
        print(f"[STEP] [Turn {step}/30] -> Agent Deployed: {act_map[int(action)]} | R: {reward:+.1f} | Res[POL:{res[0]}, PW:{res[1]}, SAN:{res[2]}, WAT:{res[3]}]")
        
        if done:
            break

    print("[STEP] Continuous Shift Completed Validly.")
    print("[END]")

if __name__ == "__main__":
    main()
