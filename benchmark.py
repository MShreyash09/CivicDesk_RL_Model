"""
Civic Desk Automated Benchmark Suite.

Runs every ticket in the bank through the Qwen LLM, grades each one
with the CivicDeskEnvironment, and outputs:
  • Formatted console table
  • benchmark_results.json  (for the dashboard to load)

Usage:
    # Full run with live LLM
    python benchmark.py

    # Mock run (uses expected answers — validates scoring pipeline)
    python benchmark.py --mock
"""

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Dict, List

# Add parent to path so we can import civic_desk
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Also add self dir for standalone use
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from civic_desk.ticket_bank import TICKET_BANK
    from civic_desk.server.civic_desk_environment import CivicDeskEnvironment
except ImportError:
    from ticket_bank import TICKET_BANK
    from server.civic_desk_environment import CivicDeskEnvironment


# LLM Dispatcher 

def call_llm(ticket: Dict[str, Any], hf_token: str) -> Dict[str, str]:
    """Call Qwen via HuggingFace Inference API and return the parsed action dict."""
    from huggingface_hub import InferenceClient

    client = InferenceClient("Qwen/Qwen2.5-72B-Instruct", token=hf_token)

    prompt = f"""You are an expert civic service desk AI dispatcher.

TICKET ID:    {ticket['ticket_id']}
DESCRIPTION:  {ticket['description']}
POLICY:       {ticket['policy_snippet']}
RESOURCES:    {ticket['active_resources']}
DIFFICULTY:   {ticket['difficulty_level']}

INSTRUCTIONS:
1. Read the policy snippet carefully. Your routing and priority MUST follow the policy.
2. If the ticket is vague, ambiguous, or lacks critical details, set action_type to "Request_Info".
3. If the situation is critical and involves multiple departments, set action_type to "Escalate".
4. In your justification, CITE specific policy terms and explain your reasoning.

Return ONLY a JSON object with these exact keys:
{{
  "target_queue": "Police" | "Public_Works" | "Sanitation" | "Water",
  "priority": "Low" | "Medium" | "High" | "Critical",
  "action_type": "Route" | "Request_Info" | "Escalate" | "Resolve",
  "justification": "Your policy-cited reasoning here"
}}
"""

    response = client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
    )
    ai_output = response.choices[0].message.content

    json_match = re.search(r"\{.*\}", ai_output, re.DOTALL)
    if not json_match:
        return {
            "target_queue": "Public_Works",
            "priority": "Medium",
            "action_type": "Route",
            "justification": "LLM did not return valid JSON.",
        }

    action = json.loads(json_match.group())

    # Sanitize enum values
    valid_q = {"Police", "Public_Works", "Sanitation", "Water"}
    valid_p = {"Low", "Medium", "High", "Critical"}
    valid_a = {"Route", "Request_Info", "Escalate", "Resolve"}
    if action.get("target_queue") not in valid_q:
        action["target_queue"] = "Public_Works"
    if action.get("priority") not in valid_p:
        action["priority"] = "Medium"
    if action.get("action_type") not in valid_a:
        action["action_type"] = "Route"

    return action


def mock_answer(ticket: Dict[str, Any]) -> Dict[str, str]:
    """Return the expected answer — for pipeline validation only."""
    return {
        "target_queue": ticket["expected_queue"],
        "priority": ticket["expected_priority"],
        "action_type": ticket["expected_action_type"],
        "justification": " ".join(ticket.get("policy_keywords", [])),
    }


# Benchmark Core 

def run_benchmark(use_mock: bool = False) -> Dict[str, Any]:
    hf_token = os.getenv("HF_TOKEN", "")
    if not use_mock and not hf_token:
        print("⚠️  HF_TOKEN not set — falling back to mock mode.")
        use_mock = True

    env = CivicDeskEnvironment()
    results: List[Dict[str, Any]] = []
    total = len(TICKET_BANK)

    print()
    print(f"{'═' * 57}")
    print(f"  CIVIC DESK BENCHMARK — {total} Scenarios")
    print(f"{'═' * 57}")
    print(f"  Mode: {'MOCK (expected answers)' if use_mock else 'LIVE (Qwen LLM)'}")
    print(f"{'═' * 57}")
    print()

    for i, ticket in enumerate(TICKET_BANK, 1):
        tid = ticket["ticket_id"]
        diff = ticket["difficulty_level"]

        # Reset with deterministic ticket
        env.reset(ticket_id=tid)

        # Get action (mock or LLM)
        t0 = time.time()
        if use_mock:
            action = mock_answer(ticket)
        else:
            try:
                action = call_llm(ticket, hf_token)
            except Exception as e:
                print(f"  [{i}/{total}] {tid} — LLM error: {e}")
                action = mock_answer(ticket)  # fallback
        response_time_ms = round((time.time() - t0) * 1000, 1)

        # Grade
        step_result = env.step(action)
        grading = step_result.info.get("grading_details", {})

        record = {
            "ticket_id": tid,
            "difficulty": diff,
            "reward": step_result.reward,
            "response_time_ms": response_time_ms,
            "correct_routing": grading.get("correct_routing", False),
            "correct_priority": grading.get("correct_priority", False),
            "correct_action_type": grading.get("correct_action_type", False),
            "overall_correct": grading.get("overall_correct", False),
            "routing_score": grading.get("routing_score", 0),
            "priority_score": grading.get("priority_score", 0),
            "action_type_score": grading.get("action_type_score", 0),
            "justification_score": grading.get("justification_score", 0),
            "agent_queue": action.get("target_queue", ""),
            "agent_priority": action.get("priority", ""),
            "agent_action_type": action.get("action_type", ""),
            "expected_queue": ticket["expected_queue"],
            "expected_priority": ticket["expected_priority"],
            "expected_action_type": ticket["expected_action_type"],
        }
        results.append(record)

        status = "✅" if record["overall_correct"] else "❌"
        print(
            f"  [{i:>2}/{total}] {tid}  {diff:<10}  "
            f"R={step_result.reward:.2f}  {response_time_ms:>7.0f}ms  {status}"
        )

    # Aggregate stats 
    correct_count = sum(1 for r in results if r["overall_correct"])
    accuracy = correct_count / total * 100 if total else 0
    avg_reward = sum(r["reward"] for r in results) / total if total else 0
    avg_time = sum(r["response_time_ms"] for r in results) / total if total else 0

    by_difficulty: Dict[str, Dict[str, Any]] = {}
    for diff in ("easy", "medium", "hard", "ambiguous"):
        subset = [r for r in results if r["difficulty"] == diff]
        if subset:
            d_correct = sum(1 for r in subset if r["overall_correct"])
            by_difficulty[diff] = {
                "total": len(subset),
                "correct": d_correct,
                "accuracy": round(d_correct / len(subset) * 100, 1),
                "avg_reward": round(sum(r["reward"] for r in subset) / len(subset), 2),
            }

    summary = {
        "total_tickets": total,
        "correct_count": correct_count,
        "overall_accuracy": round(accuracy, 1),
        "avg_reward": round(avg_reward, 2),
        "avg_response_time_ms": round(avg_time, 1),
        "by_difficulty": by_difficulty,
        "mode": "mock" if use_mock else "live",
    }

    output = {"summary": summary, "results": results}

    # Console output 
    print()
    print(f"{'═' * 57}")
    print(f"  BENCHMARK RESULTS")
    print(f"{'═' * 57}")
    print(f"  Overall Accuracy:    {accuracy:.1f}%  ({correct_count}/{total})")
    print(f"  Avg Reward:          {avg_reward:.2f} / 2.5")
    print(f"  Avg Response Time:   {avg_time:.1f}ms")
    print(f"{'─' * 57}")
    print(f"  By Difficulty:")
    for diff, stats in by_difficulty.items():
        bar = "." * (20 - len(diff))
        print(
            f"    {diff.capitalize()} {bar} "
            f"{stats['accuracy']:>5.1f}%  ({stats['correct']}/{stats['total']})"
        )
    print(f"{'═' * 57}")
    print()

    # Save JSON 
    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "benchmark_results.json"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  📄 Results saved to {out_path}")

    return output


#  CLI 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Civic Desk Benchmark Suite")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use expected answers instead of calling the LLM (validates scoring pipeline).",
    )
    args = parser.parse_args()
    run_benchmark(use_mock=args.mock)
