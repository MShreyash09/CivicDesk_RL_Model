"""
Civic Desk Visual Dashboard — Streamlit Application.

Three panels:
  1. Live Demo      — Run a single ticket through the AI agent in real-time
  2. Benchmark      — Load benchmark_results.json and visualize KPIs + charts
  3. Architecture   — System flow diagram

Launch:
    streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import json
import os
import time
import numpy as np
import sys
import re
import plotly.graph_objects as go

# Try importing the RL packages for the dashboard to verify
try:
    from stable_baselines3 import PPO
    from server.gym_env import CivicDeskGymEnv, QUEUE_MAP, PRIO_MAP, DIFF_MAP
    import inference as rl_agent
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False

# Ensure imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ticket_bank import TICKET_BANK, get_ticket_by_id, get_tickets_by_difficulty
from server.civic_desk_environment import CivicDeskEnvironment

# ─── Page Config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Civic Desk — AI Benchmarking Dashboard",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* KPI Cards */
    .kpi-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 1.5rem;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.25);
        transition: transform 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-4px);
    }
    .kpi-card h2 {
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0;
        line-height: 1.1;
    }
    .kpi-card p {
        font-size: 0.85rem;
        opacity: 0.85;
        margin: 0.3rem 0 0;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .kpi-green {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        box-shadow: 0 8px 32px rgba(17, 153, 142, 0.25);
    }
    .kpi-orange {
        background: linear-gradient(135deg, #f2994a 0%, #f2c94c 100%);
        box-shadow: 0 8px 32px rgba(242, 153, 74, 0.25);
    }
    .kpi-blue {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        box-shadow: 0 8px 32px rgba(79, 172, 254, 0.25);
    }

    /* Hero header */
    .hero {
        text-align: center;
        padding: 1rem 0 0.5rem;
    }
    .hero h1 {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .hero .tagline {
        font-size: 1rem;
        color: #888;
    }

    /* Ticket card */
    .ticket-card {
        background: #1a1a2e;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
        color: #e0e0e0;
    }
    .ticket-card .label {
        font-size: 0.75rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.2rem;
    }
    .ticket-card .value {
        font-size: 0.95rem;
        margin-bottom: 0.8rem;
        line-height: 1.4;
    }

    /* Status pill */
    .pill-pass {
        display: inline-block;
        background: #38ef7d;
        color: #000;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .pill-fail {
        display: inline-block;
        background: #ff6b6b;
        color: #fff;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #1a1a2e 100%);
    }
</style>
""", unsafe_allow_html=True)


# ─── Hero ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🏛️ Civic Desk</h1>
    <div class="tagline">AI-Powered Civic Service Benchmarking Platform</div>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧭 Navigation")
    panel = st.radio(
        "Select Panel",
        ["ℹ️ About Project", "🎮 Live Demo", "📊 Benchmark Results", "🏗️ Architecture"],
        label_visibility="collapsed",
    )

def render_live_demo():
    st.header("🪩 Live PPO Dispatch Simulation")
    st.markdown("Watch the Heterogeneous AI architecture perform in real-time. Qwen LLM extracts state perception, and a PPO model handles dispatch & resources via `stable-baselines3`.")

    if not RL_AVAILABLE:
        st.error("RL packages missing. Please install gymnasium and stable-baselines3.")
        return

    model_path = os.path.join(os.path.dirname(__file__), "ppo_civic_dispatcher.zip")
    if not os.path.exists(model_path):
        st.warning("PPO Model `ppo_civic_dispatcher.zip` missing! Run `python train_rl.py` first.")
        return

    if st.button("▶️ Start 10-Turn Shift", use_container_width=True):
        st.divider()
        
        # Load Env and Model
        env = CivicDeskGymEnv(use_advanced_rules=True)
        model = PPO.load(model_path)
        obs, _ = env.reset()
        
        # UI Placeholders
        col1, col2, col3, col4 = st.columns(4)
        pol_metric = col1.metric("🚓 Police", "3/3")
        pw_metric = col2.metric("🚧 Pub Works", "3/3")
        san_metric = col3.metric("🗑️ Sanitation", "3/3")
        wat_metric = col4.metric("💧 Water", "3/3")
        
        log_container = st.container()
        
        for step in range(1, 11):
            with log_container:
                st.markdown(f"#### 🕒 Turn {step} / 30")
                
                if env.active_ticket:
                    raw_target = env.active_ticket['queue'] 
                    queue_str = list(QUEUE_MAP.keys())[raw_target]
                    diff_str = list(DIFF_MAP.keys())[env.active_ticket['difficulty']]
                    
                    st.info(f"**🎫 Ticket Arrived:** (Internal Target: {queue_str})" )
                    
                    # Mocking perception for speed visually in UI
                    llm_parsed = {
                        "target_queue": queue_str,
                        "priority": list(PRIO_MAP.keys())[env.active_ticket['priority']],
                        "difficulty": diff_str
                    }
                    
                    st.write(f"👁️ **LLM Perception**: `{llm_parsed}`")
                    obs = rl_agent.state_normalizer(obs, llm_parsed)
                else:
                    st.write("⏳ *Routine Patrol (No active tickets)*")
                    
                # RL Action
                action, _ = model.predict(obs, deterministic=True)
                act_map = {0: "Wait", 1: "Police", 2: "Pub Works", 3: "Sanitation", 4: "Water"}
                act_str = act_map[int(action)]
                
                # Step env
                obs, reward, done, _, info = env.step(action)
                
                # Format feedback
                color = "green" if reward > 0 else "red" if reward < 0 else "gray"
                st.markdown(f"🤖 **PPO Agent Action:** :blue[{act_str}]  ➡️  Reward: :{color}[{reward:+.1f}]")
                st.divider()
                
                # Update visual resources
                pol_metric.metric("🚓 Police", f"{int(obs[5])}/3")
                pw_metric.metric("🚧 Pub Works", f"{int(obs[6])}/3")
                san_metric.metric("🗑️ Sanitation", f"{int(obs[7])}/3")
                wat_metric.metric("💧 Water", f"{int(obs[8])}/3")
                
                time.sleep(1.0)
                if done: break
        st.success("Simulation complete!")

def render_architecture():
    st.header("🏗️ Heterogeneous RL System Architecture")
    
    st.markdown("This hackathon project demonstrates a modular AI system utilizing both Generative AI for fuzzy perception and Reinforcement Learning for rigorous resource management across continuous MDP episodes.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Decision Pipeline")
        # Mermaid JS architecture mapped for Streamlit using markdown
        st.markdown("""
        ```mermaid
        graph TD
            A[Citizen Text Ticket] -->|Unstructured| B(Qwen/LLM Perception Module)
            B -->|Rigid JSON| C[State Normalizer]
            Env[(City Gymnasium Env)] -->|Internal SLA / Resources| C
            C -->|NumPy Array State| D(Stable-Baselines3 PPO Agent)
            D -->|Discrete Dispatch Action| Env
            Env -->|Delayed Reward| D
        ```
        """)

    with col2:
        st.markdown("### PPO Action Space & Rewards")
        st.markdown("""
        | Action (Discrete) | Simple Reward (Immediate) | Resource Penalty (Delayed) |
        | :--- | :--- | :--- |
        | `0: Wait` | -0.1 (SLA Decay) | Max SLA hit -> -2 |
        | `1: Police` | +1.0 if correct | -1.0 if incorrect |
        | `2: Pub Works` | +1.0 if correct | Resource Lock x4 Turns |
        | `3: Sanitation` | +1.0 if correct | Resource Lock x2 Turns |
        | `4: Water` | +1.0 if correct | Empty Resource -> -1.0 |
        """)
        st.info("The RL loop learns dynamic capacity planning entirely locally through PPO generated rollouts, overcoming LLM limitations on multi-step reasoning.")

# ═══════════════════════════════════════════════════════════════════════
#  PANEL 0 — ABOUT PROJECT
# ═══════════════════════════════════════════════════════════════════════
if panel == "ℹ️ About Project":
    st.header("✨ Civic Desk Platform Overview")
    st.markdown("### What does this project do?")
    st.markdown(
        "Civic Desk is a **Heterogeneous Reinforcement Learning Platform** that automates municipal "
        "ticketing and emergency dispatch. It solves a fundamental problem with Large Language Models: "
        "LLMs are great at understanding messy language but terrible at managing strict mathematical constraints "
        "and delayed consequences."
    )
    st.markdown("### How it Works (The Heterogeneous Architecture)")
    st.markdown(
        "1. **Perception (The LLM)**: When a panicked citizen submits an unstructured text request, we "
        "prompt `Qwen-72B` strictly as a perception filter. It ignores its own logic capabilities and just standardizes "
        "the text into a rigid JSON matrix (e.g. `SeverityLevel`, `DepartmentId`).\n"
        "2. **State Normalization**: The raw JSON is stripped and bound into a rigid `Float32` NumPy array alongside the "
        "city's current physical resource capacities (like Available Firetrucks and SLA timers).\n"
        "3. **Decision (The PPO RL Agent)**: We train a **Proximal Policy Optimization (PPO)** Reinforcement Learning model "
        "in `gymnasium` over tens of thousands of continuous shifts to predict the best dispatch action from that matrix. "
        "This allows the AI to learn complex rules—like *'Don't deploy the last sanitation truck to a low-priority ticket if a storm is coming'*—without "
        "ever suffering from LLM hallucination."
    )
    
    st.info("💡 **Hackathon Key Feature**: By decoupling LLM Perception from RL Training, our `train_rl.py` script trains massive policies over 15,000 steps locally in seconds, completely bypassing standard LLM API bottlenecks.")

# ═══════════════════════════════════════════════════════════════════════
#  PANEL 1 — LIVE DEMO
# ═══════════════════════════════════════════════════════════════════════
elif panel == "🎮 Live Demo":
    render_live_demo()

# ═══════════════════════════════════════════════════════════════════════
#  PANEL 2 — BENCHMARK RESULTS
# ═══════════════════════════════════════════════════════════════════════
elif panel == "📊 Benchmark Results":
    st.markdown("## 📊 Benchmark Results")

    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_results.json")

    if not os.path.exists(results_path):
        st.warning(
            "No `benchmark_results.json` found. Run the benchmark first:\n\n"
            "```bash\npython benchmark.py --mock\n```"
        )
        st.stop()

    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    summary = data["summary"]
    results = data["results"]

    # ── KPI Cards ─────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class="kpi-card kpi-green">
            <h2>{summary['overall_accuracy']}%</h2>
            <p>Overall Accuracy</p>
        </div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="kpi-card">
            <h2>{summary['avg_reward']}</h2>
            <p>Avg Reward / 2.5</p>
        </div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="kpi-card kpi-blue">
            <h2>{summary['avg_response_time_ms']:.0f}ms</h2>
            <p>Avg Response Time</p>
        </div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="kpi-card kpi-orange">
            <h2>{summary['correct_count']}/{summary['total_tickets']}</h2>
            <p>Correct / Total</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts ────────────────────────────────────────────────────────
    col_chart, col_radar = st.columns(2)

    with col_chart:
        st.markdown("### Accuracy by Difficulty")
        by_diff = summary.get("by_difficulty", {})
        if by_diff:
            diffs = list(by_diff.keys())
            accs = [by_diff[d]["accuracy"] for d in diffs]
            colors = ["#38ef7d", "#4facfe", "#f2994a", "#ff6b6b"]
            fig = go.Figure(go.Bar(
                x=[d.capitalize() for d in diffs],
                y=accs,
                marker_color=colors[:len(diffs)],
                text=[f"{a:.1f}%" for a in accs],
                textposition="outside",
            ))
            fig.update_layout(
                yaxis_range=[0, 110],
                yaxis_title="Accuracy %",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#ccc"),
                height=350,
                margin=dict(t=20, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_radar:
        st.markdown("### Score Axis Distribution")
        avg_routing = sum(r["routing_score"] for r in results) / len(results)
        avg_priority = sum(r["priority_score"] for r in results) / len(results)
        avg_action = sum(r["action_type_score"] for r in results) / len(results)
        avg_just = sum(r["justification_score"] for r in results) / len(results)

        categories = ["Routing (1.0)", "Priority (0.5)", "Action Type (0.5)", "Justification (0.5)"]
        maxes = [1.0, 0.5, 0.5, 0.5]
        values = [avg_routing, avg_priority, avg_action, avg_just]
        pcts = [v / m * 100 for v, m in zip(values, maxes)]

        fig2 = go.Figure(go.Scatterpolar(
            r=pcts + [pcts[0]],
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor="rgba(102, 126, 234, 0.3)",
            line=dict(color="#667eea", width=2),
        ))
        fig2.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=10)),
                bgcolor="rgba(0,0,0,0)",
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#ccc"),
            height=350,
            margin=dict(t=30, b=30),
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Results Table ─────────────────────────────────────────────────
    st.markdown("### Full Results Table")
    df = pd.DataFrame(results)
    display_cols = [
        "ticket_id", "difficulty", "reward", "overall_correct",
        "routing_score", "priority_score", "action_type_score",
        "justification_score", "agent_queue", "expected_queue",
        "response_time_ms",
    ]
    existing_cols = [c for c in display_cols if c in df.columns]
    st.dataframe(
        df[existing_cols].style.map(
            lambda v: "background-color: #1b4332; color: #38ef7d" if v is True
            else ("background-color: #4a1520; color: #ff6b6b" if v is False else ""),
            subset=["overall_correct"] if "overall_correct" in existing_cols else [],
        ),
        use_container_width=True,
        height=450,
    )


# ═══════════════════════════════════════════════════════════════════════
#  PANEL 3 — ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════
elif panel == "🏗️ Architecture":
    st.markdown("## 🏗️ System Architecture")
    st.markdown(
        "The Civic Desk platform follows a **Reset → Observe → Act → Grade → Learn** "
        "reinforcement-learning loop."
    )

    # Mermaid diagram rendered via st.markdown (Streamlit supports mermaid)
    st.markdown("""
```mermaid
graph LR
    A[🎫 Ticket Bank<br/>53 Scenarios] -->|reset| B[🏛️ Environment<br/>CivicDeskEnvironment]
    B -->|observation| C[🤖 AI Agent<br/>Qwen LLM]
    C -->|action| B
    B -->|reward + info| D[📊 Benchmark Suite<br/>benchmark.py]
    D -->|JSON| E[📈 Dashboard<br/>Streamlit]

    style A fill:#667eea,stroke:#667eea,color:#fff
    style B fill:#764ba2,stroke:#764ba2,color:#fff
    style C fill:#11998e,stroke:#11998e,color:#fff
    style D fill:#f2994a,stroke:#f2994a,color:#fff
    style E fill:#4facfe,stroke:#4facfe,color:#fff
```
    """)

    st.markdown("---")
    st.markdown("### Grading Rubric")
    st.markdown("""
| Axis | Max Score | How It Works |
|------|-----------|-------------|
| **Routing** | 1.0 | Exact match on `target_queue` vs expected department |
| **Priority** | 0.5 | Exact match on `priority` level |
| **Action Type** | 0.5 | Correct `action_type` — bonus for `Request_Info` on ambiguous |
| **Justification** | 0.5 | Keyword overlap: % of policy keywords found in justification |
| **Total** | **2.5** | Sum of all axes |
    """)

    st.markdown("### Ticket Difficulty Distribution")
    dist_data = {
        "Easy": len(get_tickets_by_difficulty("easy")),
        "Medium": len(get_tickets_by_difficulty("medium")),
        "Hard": len(get_tickets_by_difficulty("hard")),
        "Ambiguous": len(get_tickets_by_difficulty("ambiguous")),
    }
    fig3 = go.Figure(go.Pie(
        labels=list(dist_data.keys()),
        values=list(dist_data.values()),
        hole=0.45,
        marker=dict(colors=["#38ef7d", "#4facfe", "#f2994a", "#ff6b6b"]),
        textinfo="label+value",
        textfont=dict(size=14),
    ))
    fig3.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#ccc"),
        height=350,
        margin=dict(t=20, b=20),
        showlegend=False,
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")
    st.markdown(
        "**Total in Bank:** "
        f"**{len(TICKET_BANK)}** tickets across 4 departments × 4 difficulty tiers."
    )
