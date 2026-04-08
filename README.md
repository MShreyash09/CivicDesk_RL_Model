---
title: Civic Desk Dispatcher
emoji: 🏛️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# 🏛️ Civic Desk: Heterogeneous Reinforcement Learning

Welcome to Civic Desk — an advanced, production-grade municipal dispatch simulator. This platform solves the unreliability of purely generative AI by employing a **Heterogeneous Reinforcement Learning Architecture**, blending Large Language Models (LLMs) with strict mathematical RL training.

## ✨ Why this stands out
LLMs are phenomenal at understanding unstructured, messy human text but terrible at dynamic capacity planning and tracking delayed mathematical constraints. By splitting our AI into two specialized modules, we overcome LLM hallucinations entirely:

1. **The Perception Module (Qwen-72B)**: Takes in unstructured civic support tickets and standardizes them into rigid JSON parameters (Severity, Department Target).
2. **The Decision Module (PPO RL)**: We train a Proximal Policy Optimization (PPO) agent within `gymnasium` to manage city resources across continuous, multi-turn shifts. It takes the standardized data from the LLM, views currently available resources, and dictates actions in milliseconds without hallucinating.

## 🏆 Hackathon Compliance Features
This repository has been entirely structured to comply with the auto-grading rules of the OpenEnv Platform:
- The entry point is perfectly bound to `inference.py`.
- We use the `openai` Python API client (pointed at an `API_BASE_URL` HuggingFace endpoint).
- It injects strict `[START]`, `[STEP]`, and `[END]` logging syntax around the live RL-Gymnasium loop natively.

---

## 🚀 Setup & Execution

### Installation
Ensure you install the custom RL requirements:
```bash
pip install -e "."
```

*(This automatically installs `gymnasium`, `stable-baselines3`, `openai`, `streamlit`, `pandas`, and `plotly`.)*

### 1. Train the Decision Agent Fast (The RL Curve)
To prevent the PPO agent from taking 10 hours to train due to LLM networking latency, we bypass the LLM entirely during training by using a mock script. This trains a 15,000-turn PPO agent locally in seconds.
```bash
python train_rl.py
```
*(This will generate the `ppo_civic_dispatcher.zip` model file).*

### 2. Run the Entry File (Inference)
The live loop integrates both the GenAI Perception and RL decision making in real-time.
```bash
python inference.py
```

### 3. Launch the Visual Streamlit Dashboard
We built a visually stunning dashboard to let judges map and monitor the active model capacities:
```bash
streamlit run dashboard.py
```
You can watch the RL agent parse active incoming tickets and drain/replenish physical city resources in real-time along a 30-turn shift!

---

## 🏗️ The 4-Axis Training Rubric
The Continuous Gym MDP issues delayed RL rewards based on 4 axes:
- **Routing**: Exact match on target department (`+1.0`).
- **Priority**: Exact match on priority level (`+0.5`).
- **Action Type**: Correct discrete action (`+0.5`).
- **Justification**: Valid policy constraint met.
- **Bonus mechanics**: Waiting invokes passive SLA decay (`-0.1`), and invalid dispatches trigger Resource Lock Penalties locally.
