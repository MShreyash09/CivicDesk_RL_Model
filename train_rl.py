"""
Civic Desk — Fast RL Training Script

This script bypasses LLM inference entirely to synthesize tens of thousands of
episodes rapidly, allowing a PPO agent to learn the dispatch policy via StableBaselines3.

Usage:
    python train_rl.py
"""

import os
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

from server.gym_env import CivicDeskGymEnv

def main():
    print("🚀 Initializing Vectorized Gym Environment...")
    # Wrap in dummy vector env for stable baselines
    env_fn = lambda: CivicDeskGymEnv(use_advanced_rules=True)
    vec_env = make_vec_env(env_fn, n_envs=4)

    print("🧠 Creating PPO Agent (MlpPolicy)...")
    # For a state vector of length 10, a tiny MLP is more than enough
    model = PPO(
        "MlpPolicy", 
        vec_env, 
        verbose=1,
        learning_rate=0.003,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        # A tiny network trains instantly
        policy_kwargs=dict(net_arch=[64, 64])
    )

    print("\n⏳ Commencing Training Loop (15,000 steps)...")
    model.learn(total_timesteps=15_000, progress_bar=True)

    print("\n⚖️ Evaluating Trained Policy...")
    # Evaluate using a standard instance
    eval_env = CivicDeskGymEnv(use_advanced_rules=True)
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20)
    print(f"📊 Evaluation: Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Save the model
    save_path = os.path.join(os.path.dirname(__file__), "ppo_civic_dispatcher")
    model.save(save_path)
    print(f"✅ Model saved to {save_path}.zip!")

if __name__ == "__main__":
    main()
