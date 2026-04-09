"""


Environment variables required:
  API_BASE_URL  — API endpoint for the LLM
  MODEL_NAME    — Model identifier
  HF_TOKEN      — Hugging Face / API key

Usage:
  python inference.py
"""
import os
import sys
import json
import time
import requests
from openai import OpenAI

# ── Environment config ────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

TASKS = ["task_1_easy", "task_2_medium", "task_3_hard"]

# ── Validate required env vars before anything else ───────────────────────────
if not API_BASE_URL:
    print(json.dumps({
        "event": "CONFIG_ERROR",
        "error": "API_BASE_URL environment variable is not set",
        "timestamp": time.time(),
    }), flush=True)
    sys.exit(1)

if not HF_TOKEN:
    print(json.dumps({
        "event": "CONFIG_ERROR",
        "error": "HF_TOKEN environment variable is not set or empty",
        "timestamp": time.time(),
    }), flush=True)
    sys.exit(1)

# ── Initialise OpenAI client safely ──────────────────────────────────────────
try:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
except Exception as e:
    print(json.dumps({
        "event": "CLIENT_INIT_ERROR",
        "error": str(e),
        "timestamp": time.time(),
    }), flush=True)
    sys.exit(1)


# ── Structured logging helpers ────────────────────────────────────────────────
def log_start(task_id: str, episode: int):
    print(json.dumps({
        "event":     "START",
        "task_id":   task_id,
        "episode":   episode,
        "model":     MODEL_NAME,
        "timestamp": time.time(),
    }), flush=True)


def log_step(task_id: str, episode: int, step: int, action: int,
             reward: float, done: bool, info: dict):
    print(json.dumps({
        "event":         "STEP",
        "task_id":       task_id,
        "episode":       episode,
        "step":          step,
        "action":        action,
        "reward":        reward,
        "done":          done,
        "chosen_food":   info.get("chosen_food", ""),
        "food_category": info.get("food_category", ""),
        "health_score":  info.get("health_score_after", 0.0),
        "timestamp":     time.time(),
    }), flush=True)


def log_end(task_id: str, episode: int, total_reward: float,
            grader_score: float, steps: int, choices: list):
    print(json.dumps({
        "event":        "END",
        "task_id":      task_id,
        "episode":      episode,
        "total_reward": total_reward,
        "grader_score": grader_score,
        "steps":        steps,
        "choices":      choices,
        "timestamp":    time.time(),
    }), flush=True)


# ── Environment client helpers ────────────────────────────────────────────────
def env_reset(task_id: str) -> dict:
    try:
        r = requests.post(
            f"{ENV_BASE_URL}/reset",
            params={"task_id": task_id},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(
            f"Cannot connect to environment at {ENV_BASE_URL}. "
            f"Is the env container running? Error: {e}"
        )
    except requests.exceptions.Timeout:
        raise RuntimeError(
            f"Timeout connecting to environment at {ENV_BASE_URL}/reset for task {task_id}"
        )
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"HTTP error on /reset for {task_id}: {e}")


def env_step(task_id: str, action_index: int, reasoning: str = "") -> dict:
    payload = {"selected_item_index": action_index, "reasoning": reasoning}
    try:
        r = requests.post(
            f"{ENV_BASE_URL}/step",
            json=payload,
            params={"task_id": task_id},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(
            f"Cannot connect to environment at {ENV_BASE_URL}. "
            f"Is the env container running? Error: {e}"
        )
    except requests.exceptions.Timeout:
        raise RuntimeError(
            f"Timeout connecting to environment at {ENV_BASE_URL}/step for task {task_id}"
        )
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"HTTP error on /step for {task_id}: {e}")


def env_state(task_id: str) -> dict:
    try:
        r = requests.get(
            f"{ENV_BASE_URL}/state",
            params={"task_id": task_id},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(
            f"Cannot connect to environment at {ENV_BASE_URL}. "
            f"Is the env container running? Error: {e}"
        )
    except requests.exceptions.Timeout:
        raise RuntimeError(
            f"Timeout connecting to environment at {ENV_BASE_URL}/state for task {task_id}"
        )
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"HTTP error on /state for {task_id}: {e}")


# ── LLM agent ────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a health-conscious food advisor. 
You will be presented with a meal scenario including context (time of day, hunger level, budget, health goal)
and a list of food options numbered from 0.

Your task:
1. Analyze each food option's nutritional value, calories, and fit with the health goal.
2. Select the HEALTHIEST option that fits within the budget.
3. Respond ONLY with a JSON object in this exact format:
{"selected_item_index": <integer>, "reasoning": "<brief explanation>"}

Prioritize: nutrition_score > calories (lower is often better) > price.
Always pick healthy over junk food.
"""


def build_user_prompt(obs: dict) -> str:
    ctx = obs["context"]
    options = obs["food_options"]

    options_text = "\n".join([
        f"{i}. {f['name']} | Cal:{f['calories']} | NutriScore:{f['nutrition_score']}/10 "
        f"| Price:${f['price']} | {f['description']}"
        for i, f in enumerate(options)
    ])

    return f"""=== Meal {obs['meal_number']} — {ctx['time_of_day'].upper()} ===
Health Goal: {ctx['health_goal']}
Hunger Level: {ctx['hunger_level']}/10
Budget: ${ctx['budget']}
Current Health Score: {obs['current_health_score']}/100
Previous meals: {', '.join(ctx['previous_meals']) or 'None'}

Available options:
{options_text}

Choose the healthiest option. Reply only with JSON."""


def agent_choose(obs: dict) -> tuple[int, str]:
    """Call LLM to choose food item. Returns (index, reasoning)."""
    user_msg = build_user_prompt(obs)
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=200,
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown fences if present
        raw = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)
        idx = int(parsed.get("selected_item_index", 0))
        reasoning = parsed.get("reasoning", "")
        num_options = len(obs["food_options"])
        idx = max(0, min(idx, num_options - 1))
        return idx, reasoning
    except Exception as e:
        # Fallback: pick option 0 (usually healthy in easy task)
        print(json.dumps({"event": "AGENT_ERROR", "error": str(e)}), flush=True)
        return 0, f"fallback due to error: {e}"


# ── Episode runner ────────────────────────────────────────────────────────────
def run_episode(task_id: str, episode: int = 1) -> dict:
    log_start(task_id, episode)

    reset_result = env_reset(task_id)
    obs = reset_result["observation"]

    total_reward    = 0.0
    step_num        = 0
    all_rewards     = []
    all_categories  = []
    all_choices     = []
    all_nutrition   = []
    all_budget_ok   = []
    health_trajectory = []

    while True:
        action_idx, reasoning = agent_choose(obs)
        step_result = env_step(task_id, action_idx, reasoning)

        reward   = step_result["reward"]
        done     = step_result["done"]
        info     = step_result.get("info", {})
        next_obs = step_result["observation"]

        total_reward += reward
        step_num     += 1
        all_rewards.append(reward)
        all_categories.append(info.get("food_category", "unknown"))
        all_choices.append(info.get("chosen_food", "unknown"))
        all_nutrition.append(float(info.get("nutrition_score", 5.0)))
        all_budget_ok.append(True)
        health_trajectory.append(float(info.get("health_score_after", 50.0)))

        log_step(task_id, episode, step_num, action_idx, reward, done, info)

        if done:
            break
        obs = next_obs

    # Grade the episode
    grader_score = compute_grader_score(
        task_id, all_rewards, all_choices, health_trajectory,
        all_categories, all_nutrition, all_budget_ok
    )

    log_end(task_id, episode, round(total_reward, 4), grader_score, step_num, all_choices)

    return {
        "task_id":      task_id,
        "episode":      episode,
        "total_reward": round(total_reward, 4),
        "grader_score": grader_score,
        "steps":        step_num,
        "choices":      all_choices,
    }


def compute_grader_score(task_id, rewards, choices, trajectory, categories,
                         nutrition, budget_ok) -> float:
    """Compute grader score locally (mirrors server-side graders)."""
    if task_id == "task_1_easy":
        num_healthy   = sum(1 for c in categories if c == "healthy")
        healthy_ratio = num_healthy / len(categories) if categories else 0.0
        avg_reward    = sum(rewards) / len(rewards) if rewards else 0.0
        return round(min(1.0, 0.6 * healthy_ratio + 0.4 * avg_reward), 4)

    elif task_id == "task_2_medium":
        num_healthy   = sum(1 for c in categories if c == "healthy")
        healthy_ratio = num_healthy / len(categories) if categories else 0.0
        health_imp    = max(0, trajectory[-1] - trajectory[0]) / 50.0 if len(trajectory) >= 2 else 0.0
        health_imp    = min(1.0, health_imp)
        budget_ratio  = sum(budget_ok) / len(budget_ok) if budget_ok else 1.0
        consec = 0
        max_c  = 0
        for c in categories:
            if c == "junk":
                consec += 1
                max_c = max(max_c, consec)
            else:
                consec = 0
        consec_pen = 1.0 - min(1.0, max_c / len(categories))
        score = 0.40*healthy_ratio + 0.30*health_imp + 0.15*budget_ratio + 0.15*consec_pen
        return round(min(1.0, max(0.0, score)), 4)

    elif task_id == "task_3_hard":
        if len(trajectory) >= 2:
            improvements = sum(
                1 for i in range(1, len(trajectory)) if trajectory[i] >= trajectory[i-1]
            )
            traj_score = improvements / (len(trajectory) - 1)
        else:
            traj_score = 0.0
        avg_nut       = sum(nutrition) / len(nutrition) if nutrition else 0.0
        nut_comp      = avg_nut / 10.0
        num_healthy   = sum(1 for c in categories if c == "healthy")
        healthy_ratio = num_healthy / len(categories) if categories else 0.0
        reward_eff    = sum(rewards) / len(rewards) if rewards else 0.0
        score = 0.35*traj_score + 0.25*nut_comp + 0.25*healthy_ratio + 0.15*reward_eff
        return round(min(1.0, max(0.0, score)), 4)

    return 0.0


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(json.dumps({
        "event":     "INFERENCE_START",
        "model":     MODEL_NAME,
        "env_url":   ENV_BASE_URL,
        "tasks":     TASKS,
        "timestamp": time.time(),
    }), flush=True)

    all_results = []
    for task_id in TASKS:
        try:
            result = run_episode(task_id, episode=1)
            all_results.append(result)
        except Exception as e:
            print(json.dumps({
                "event":   "TASK_ERROR",
                "task_id": task_id,
                "error":   str(e),
            }), flush=True)

    # Summary
    print(json.dumps({
        "event":   "INFERENCE_COMPLETE",
        "results": all_results,
        "avg_grader_score": round(
            sum(r["grader_score"] for r in all_results) / len(all_results), 4
        ) if all_results else 0.0,
        "timestamp": time.time(),
    }), flush=True)


if __name__ == "__main__":
    main()
