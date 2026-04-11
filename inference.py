"""
inference.py — Self-contained inference script for HealthyFoodChoice RL environment.
Starts the FastAPI env server in a background thread, waits for it to be ready,
then runs the LLM agent against it.
"""
import os
import sys
import json
import time
import threading

# ── Safe imports ──────────────────────────────────────────────────────────────
try:
    import requests
except ImportError as e:
    print(json.dumps({"event": "IMPORT_ERROR", "error": f"requests: {e}"}), flush=True)
    sys.exit(1)

try:
    from openai import OpenAI
except ImportError as e:
    print(json.dumps({"event": "IMPORT_ERROR", "error": f"openai: {e}"}), flush=True)
    sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE_URL = (
    os.environ.get("API_BASE_URL")
    or os.environ.get("OPENAI_BASE_URL")
    or "https://api-inference.huggingface.co/v1"
)
MODEL_NAME = (
    os.environ.get("MODEL_NAME")
    or os.environ.get("MODEL")
    or "Qwen/Qwen2.5-72B-Instruct"
)
HF_TOKEN = (
    os.environ.get("HF_TOKEN")
    or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    or os.environ.get("OPENAI_API_KEY")
    or "dummy-token"
)
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "")
LOCAL_PORT = 7860
TASKS = ["task_1_easy", "task_2_medium", "task_3_hard"]

# ── Print config FIRST — validator must see output immediately ────────────────
print(json.dumps({
    "event":        "CONFIG",
    "api_base_url": API_BASE_URL,
    "model":        MODEL_NAME,
    "env_url":      ENV_BASE_URL or f"http://localhost:{LOCAL_PORT} (local)",
    "hf_token_set": HF_TOKEN != "dummy-token",
    "timestamp":    time.time(),
}), flush=True)


# ── Local server launcher ─────────────────────────────────────────────────────
def start_local_server(port: int):
    try:
        import uvicorn

        # Add repo root to sys.path
        repo_root = os.path.dirname(os.path.abspath(__file__))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        # Try server/app.py first (the real openenv-core based app)
        app_obj = None
        try:
            from server.app import app as server_app
            app_obj = server_app
            print(json.dumps({"event": "SERVER_IMPORT", "source": "server.app"}), flush=True)
        except Exception as e1:
            print(json.dumps({"event": "SERVER_IMPORT_WARN", "tried": "server.app", "error": str(e1)}), flush=True)
            # Fallback: try root app.py
            try:
                from app import app as root_app
                app_obj = root_app
                print(json.dumps({"event": "SERVER_IMPORT", "source": "app"}), flush=True)
            except Exception as e2:
                print(json.dumps({"event": "SERVER_IMPORT_ERROR", "tried": "app", "error": str(e2)}), flush=True)
                return

        config = uvicorn.Config(app_obj, host="0.0.0.0", port=port, log_level="error")
        server = uvicorn.Server(config)
        server.run()

    except Exception as e:
        print(json.dumps({"event": "SERVER_START_ERROR", "error": str(e)}), flush=True)


def wait_for_server(base_url: str, timeout: int = 60) -> bool:
    for i in range(timeout):
        try:
            r = requests.get(f"{base_url}/health", timeout=2)
            if r.status_code == 200:
                print(json.dumps({
                    "event": "SERVER_READY", "url": base_url,
                    "waited_seconds": i, "timestamp": time.time(),
                }), flush=True)
                return True
        except Exception:
            pass
        time.sleep(1)
    print(json.dumps({
        "event": "SERVER_TIMEOUT",
        "error": f"Server at {base_url} not ready after {timeout}s",
    }), flush=True)
    return False


# ── Determine active env URL, start server if needed ─────────────────────────
if ENV_BASE_URL:
    ACTIVE_ENV_URL = ENV_BASE_URL
    print(json.dumps({"event": "USING_EXTERNAL_ENV", "url": ACTIVE_ENV_URL}), flush=True)
else:
    ACTIVE_ENV_URL = f"http://localhost:{LOCAL_PORT}"
    print(json.dumps({"event": "STARTING_LOCAL_SERVER", "port": LOCAL_PORT}), flush=True)
    t = threading.Thread(target=start_local_server, args=(LOCAL_PORT,), daemon=True)
    t.start()
    if not wait_for_server(ACTIVE_ENV_URL, timeout=60):
        print(json.dumps({"event": "FATAL", "error": "Server failed to start"}), flush=True)
        sys.exit(1)


# ── Init OpenAI client ────────────────────────────────────────────────────────
try:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
except Exception as e:
    print(json.dumps({"event": "CLIENT_INIT_ERROR", "error": str(e)}), flush=True)
    sys.exit(1)


# ── Logging helpers ───────────────────────────────────────────────────────────
def log_start(task_id, episode):
    print(json.dumps({
        "event": "START", "task_id": task_id,
        "episode": episode, "model": MODEL_NAME, "timestamp": time.time(),
    }), flush=True)

def log_step(task_id, episode, step, action, reward, done, info):
    print(json.dumps({
        "event": "STEP", "task_id": task_id, "episode": episode,
        "step": step, "action": action, "reward": reward, "done": done,
        "chosen_food":   info.get("chosen_food", ""),
        "food_category": info.get("food_category", ""),
        "health_score":  info.get("health_score_after", 0.0),
        "timestamp": time.time(),
    }), flush=True)

def log_end(task_id, episode, total_reward, grader_score, steps, choices):
    print(json.dumps({
        "event": "END", "task_id": task_id, "episode": episode,
        "total_reward": total_reward, "grader_score": grader_score,
        "steps": steps, "choices": choices, "timestamp": time.time(),
    }), flush=True)


# ── Env HTTP helpers ──────────────────────────────────────────────────────────
def env_reset(task_id):
    try:
        r = requests.post(f"{ACTIVE_ENV_URL}/reset", params={"task_id": task_id}, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise RuntimeError(f"env_reset failed for {task_id}: {e}")

def env_step(task_id, action_index, reasoning=""):
    try:
        r = requests.post(
            f"{ACTIVE_ENV_URL}/step",
            json={"selected_item_index": action_index, "reasoning": reasoning},
            params={"task_id": task_id},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise RuntimeError(f"env_step failed for {task_id}: {e}")


# ── LLM agent ────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a health-conscious food advisor.
Given a meal scenario and food options numbered from 0, respond ONLY with JSON:
{"selected_item_index": <integer>, "reasoning": "<brief explanation>"}
Prioritize: nutrition_score > calories (lower is better) > price. Always pick healthy over junk."""

def build_user_prompt(obs):
    ctx = obs["context"]
    options_text = "\n".join([
        f"{i}. {f['name']} | Cal:{f['calories']} | NutriScore:{f['nutrition_score']}/10 "
        f"| Price:${f['price']} | {f['description']}"
        for i, f in enumerate(obs["food_options"])
    ])
    return (
        f"=== Meal {obs['meal_number']} — {ctx['time_of_day'].upper()} ===\n"
        f"Health Goal: {ctx['health_goal']}\n"
        f"Hunger Level: {ctx['hunger_level']}/10\n"
        f"Budget: ${ctx['budget']}\n"
        f"Current Health Score: {obs['current_health_score']}/100\n"
        f"Previous meals: {', '.join(ctx['previous_meals']) or 'None'}\n\n"
        f"Available options:\n{options_text}\n\nChoose the healthiest. Reply only with JSON."
    )

def agent_choose(obs):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_user_prompt(obs)},
            ],
            max_tokens=200,
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)
        idx = int(parsed.get("selected_item_index", 0))
        idx = max(0, min(idx, len(obs["food_options"]) - 1))
        return idx, parsed.get("reasoning", "")
    except Exception as e:
        print(json.dumps({"event": "AGENT_ERROR", "error": str(e)}), flush=True)
        return 0, f"fallback: {e}"


# ── Episode runner ────────────────────────────────────────────────────────────
def run_episode(task_id, episode=1):
    log_start(task_id, episode)
    obs = env_reset(task_id)["observation"]

    total_reward = 0.0
    step_num = 0
    rewards, categories, choices, nutrition, budget_ok, trajectory = [], [], [], [], [], []

    while True:
        action_idx, reasoning = agent_choose(obs)
        result = env_step(task_id, action_idx, reasoning)

        reward   = result["reward"]
        done     = result["done"]
        info     = result.get("info", {})
        next_obs = result["observation"]

        total_reward += reward
        step_num += 1
        rewards.append(reward)
        categories.append(info.get("food_category", "unknown"))
        choices.append(info.get("chosen_food", "unknown"))
        nutrition.append(float(info.get("nutrition_score", 5.0)))
        budget_ok.append(True)
        trajectory.append(float(info.get("health_score_after", 50.0)))

        log_step(task_id, episode, step_num, action_idx, reward, done, info)
        if done:
            break
        obs = next_obs

    grader_score = compute_grader_score(task_id, rewards, choices, trajectory,
                                        categories, nutrition, budget_ok)
    log_end(task_id, episode, round(total_reward, 4), grader_score, step_num, choices)
    return {
        "task_id": task_id, "episode": episode,
        "total_reward": round(total_reward, 4),
        "grader_score": grader_score, "steps": step_num, "choices": choices,
    }


def compute_grader_score(task_id, rewards, choices, trajectory,
                         categories, nutrition, budget_ok):
    if task_id == "task_1_easy":
        hr = sum(1 for c in categories if c == "healthy") / len(categories) if categories else 0.0
        ar = sum(rewards) / len(rewards) if rewards else 0.0
        return round(min(1.0, 0.6 * hr + 0.4 * ar), 4)

    elif task_id == "task_2_medium":
        hr  = sum(1 for c in categories if c == "healthy") / len(categories) if categories else 0.0
        hi  = min(1.0, max(0, trajectory[-1] - trajectory[0]) / 50.0) if len(trajectory) >= 2 else 0.0
        br  = sum(budget_ok) / len(budget_ok) if budget_ok else 1.0
        consec = max_c = 0
        for c in categories:
            consec = consec + 1 if c == "junk" else 0
            max_c = max(max_c, consec)
        cp = 1.0 - min(1.0, max_c / len(categories))
        return round(min(1.0, max(0.0, 0.40*hr + 0.30*hi + 0.15*br + 0.15*cp)), 4)

    elif task_id == "task_3_hard":
        ts  = sum(1 for i in range(1, len(trajectory)) if trajectory[i] >= trajectory[i-1]) / (len(trajectory)-1) if len(trajectory) >= 2 else 0.0
        nc  = (sum(nutrition) / len(nutrition) / 10.0) if nutrition else 0.0
        hr  = sum(1 for c in categories if c == "healthy") / len(categories) if categories else 0.0
        re  = sum(rewards) / len(rewards) if rewards else 0.0
        return round(min(1.0, max(0.0, 0.35*ts + 0.25*nc + 0.25*hr + 0.15*re)), 4)
    return 0.0


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(json.dumps({
        "event": "INFERENCE_START", "model": MODEL_NAME,
        "env_url": ACTIVE_ENV_URL, "tasks": TASKS, "timestamp": time.time(),
    }), flush=True)

    all_results = []
    for task_id in TASKS:
        try:
            result = run_episode(task_id, episode=1)
            all_results.append(result)
        except Exception as e:
            print(json.dumps({
                "event": "TASK_ERROR", "task_id": task_id, "error": str(e),
            }), flush=True)

    print(json.dumps({
        "event": "INFERENCE_COMPLETE",
        "results": all_results,
        "avg_grader_score": round(
            sum(r["grader_score"] for r in all_results) / len(all_results), 4
        ) if all_results else 0.0,
        "timestamp": time.time(),
    }), flush=True)


if __name__ == "__main__":
    main()
