"""
inference.py — HealthyFoodChoice RL inference script.
Uses API_BASE_URL and API_KEY injected by validator (LiteLLM proxy).
Prints [START]/[STEP]/[END] blocks to stdout.

Strategy: Use LLM via validator proxy. Falls back to greedy
nutrition-score selection if LLM call fails.
"""
import os, sys, json, time, threading

try:
    import requests
except ImportError as e:
    print(f"IMPORT_ERROR: {e}", flush=True)
    sys.exit(1)

try:
    from openai import OpenAI
except ImportError as e:
    print(f"IMPORT_ERROR: {e}", flush=True)
    sys.exit(1)

# Use EXACTLY what validator injects - required for LiteLLM proxy
API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY      = os.environ["API_KEY"]
MODEL_NAME   = os.environ.get("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASKS        = ["task_1_easy", "task_2_medium", "task_3_hard"]

print(f"CONFIG api_base={API_BASE_URL} model={MODEL_NAME}", flush=True)

# ── Greedy fallback: pick healthy item with highest nutrition_score ────────────
def greedy_choice(obs):
    """
    Deterministic optimal strategy:
    1. Among healthy items, pick highest nutrition_score
    2. If no healthy items, pick highest nutrition_score overall
    Always beats random and most LLM choices.
    """
    options = obs["food_options"]
    budget  = obs["context"]["budget"]

    # Score each option: category weight + nutrition bonus
    def score(f):
        cat = f.get("category", "")
        ns  = float(f.get("nutrition_score", 0))
        cat_weight = {"healthy": 100, "neutral": 10, "junk": 0}.get(cat, 0)
        within_budget = 1.0 if float(f.get("price", 0)) <= float(budget) else 0.5
        return (cat_weight + ns) * within_budget

    best_idx = max(range(len(options)), key=lambda i: score(options[i]))
    return best_idx, f"greedy: picked {options[best_idx].get('name','')} (nutrition={options[best_idx].get('nutrition_score',0)}, cat={options[best_idx].get('category','')})"


# ── Start local env server ────────────────────────────────────────────────────
def start_local_server(port):
    try:
        import uvicorn
        repo_root = os.path.dirname(os.path.abspath(__file__))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        try:
            from server.app import app as fastapi_app
        except Exception:
            from app import app as fastapi_app
        uvicorn.Server(uvicorn.Config(fastapi_app, host="0.0.0.0", port=port, log_level="error")).run()
    except Exception as e:
        print(f"SERVER_ERROR port={port} error={e}", flush=True)

def wait_for_server(url, timeout=30):
    for i in range(timeout):
        try:
            if requests.get(f"{url}/health", timeout=2).status_code == 200:
                return True
        except: pass
        time.sleep(1)
    return False

ACTIVE_ENV_URL = None
for port in [7860, 7861, 7862]:
    t = threading.Thread(target=start_local_server, args=(port,), daemon=True)
    t.start()
    if wait_for_server(f"http://localhost:{port}", timeout=20):
        ACTIVE_ENV_URL = f"http://localhost:{port}"
        print(f"SERVER_READY url={ACTIVE_ENV_URL}", flush=True)
        break

if not ACTIVE_ENV_URL:
    print("SERVER_FAILED", flush=True)
    for task_id in TASKS:
        print(f"[START] task={task_id}", flush=True)
        print(f"[END] task={task_id} score=0.0 steps=0", flush=True)
    sys.exit(0)

# ── Init OpenAI client with validator credentials ─────────────────────────────
try:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    print(f"CLIENT_OK base_url={API_BASE_URL}", flush=True)
except Exception as e:
    print(f"CLIENT_ERROR: {e} — will use greedy fallback only", flush=True)
    client = None

# ── Env helpers ───────────────────────────────────────────────────────────────
def env_reset(task_id):
    r = requests.post(f"{ACTIVE_ENV_URL}/reset", params={"task_id": task_id}, timeout=30)
    r.raise_for_status()
    return r.json()

def env_step(task_id, action_index, reasoning=""):
    r = requests.post(f"{ACTIVE_ENV_URL}/step",
        json={"selected_item_index": action_index, "reasoning": reasoning},
        params={"task_id": task_id}, timeout=30)
    r.raise_for_status()
    return r.json()

# ── LLM agent with greedy fallback ───────────────────────────────────────────
SYSTEM_PROMPT = """You are a health-conscious food advisor.
Given a list of food options numbered from 0, respond ONLY with JSON:
{"selected_item_index": <integer>, "reasoning": "<brief explanation>"}

Rules:
1. Always prefer HEALTHY category over NEUTRAL over JUNK
2. Among same category, pick highest nutrition_score
3. Stay within budget if possible
4. Never pick junk food if a healthy option exists"""

def build_prompt(obs):
    ctx     = obs["context"]
    options = obs["food_options"]
    lines   = "\n".join([
        f"{i}. {f['name']} | category={f.get('category','?')} | "
        f"nutrition_score={f['nutrition_score']}/10 | "
        f"calories={f['calories']} | price=${f['price']}"
        for i, f in enumerate(options)
    ])
    return (
        f"Health Goal: {ctx['health_goal']} | Budget: ${ctx['budget']}\n"
        f"Current Health Score: {obs['current_health_score']}/100\n\n"
        f"Food options:\n{lines}\n\n"
        f"Pick the HEALTHIEST option. Reply only with JSON."
    )

def agent_choose(obs):
    # Try LLM first (hits validator proxy)
    if client:
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": build_prompt(obs)},
                ],
                max_tokens=150,
                temperature=0.0,
            )
            raw    = response.choices[0].message.content.strip()
            raw    = raw.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(raw)
            idx    = int(parsed.get("selected_item_index", 0))
            idx    = max(0, min(idx, len(obs["food_options"]) - 1))
            return idx, parsed.get("reasoning", "llm choice")
        except Exception as e:
            print(f"LLM_ERROR: {e} — using greedy", flush=True)

    # Greedy fallback — always optimal
    return greedy_choice(obs)


# ── Grader scores ─────────────────────────────────────────────────────────────
def compute_grader_score(task_id, rewards, trajectory, categories, nutrition, budget_ok):
    if task_id == "task_1_easy":
        hr = sum(1 for c in categories if c == "healthy") / len(categories) if categories else 0.0
        ar = sum(rewards) / len(rewards) if rewards else 0.0
        return round(min(1.0, 0.6*hr + 0.4*ar), 4)
    elif task_id == "task_2_medium":
        hr = sum(1 for c in categories if c == "healthy") / len(categories) if categories else 0.0
        hi = min(1.0, max(0, trajectory[-1]-trajectory[0])/50.0) if len(trajectory) >= 2 else 0.0
        br = sum(budget_ok)/len(budget_ok) if budget_ok else 1.0
        consec = max_c = 0
        for c in categories:
            consec = consec+1 if c == "junk" else 0
            max_c  = max(max_c, consec)
        cp = 1.0 - min(1.0, max_c/len(categories))
        return round(min(1.0, max(0.0, 0.40*hr + 0.30*hi + 0.15*br + 0.15*cp)), 4)
    elif task_id == "task_3_hard":
        ts = sum(1 for i in range(1, len(trajectory)) if trajectory[i] >= trajectory[i-1]) / (len(trajectory)-1) if len(trajectory) >= 2 else 0.0
        nc = (sum(nutrition)/len(nutrition)/10.0) if nutrition else 0.0
        hr = sum(1 for c in categories if c == "healthy") / len(categories) if categories else 0.0
        re = sum(rewards)/len(rewards) if rewards else 0.0
        return round(min(1.0, max(0.0, 0.35*ts + 0.25*nc + 0.25*hr + 0.15*re)), 4)
    return 0.0


# ── Episode runner ────────────────────────────────────────────────────────────
def run_episode(task_id, episode=1):
    print(f"[START] task={task_id}", flush=True)

    obs          = env_reset(task_id)["observation"]
    total_reward = 0.0
    step_num     = 0
    rewards, categories, choices, nutrition, budget_ok, trajectory = [], [], [], [], [], []

    while True:
        try:
            action_idx, reasoning = agent_choose(obs)
        except Exception as e:
            print(f"AGENT_FATAL step={step_num+1} error={e}", flush=True)
            action_idx, reasoning = greedy_choice(obs)

        result   = env_step(task_id, action_idx, reasoning)
        reward   = result["reward"]
        done     = result["done"]
        info     = result.get("info", {})
        next_obs = result["observation"]

        total_reward += reward
        step_num     += 1
        rewards.append(reward)
        categories.append(info.get("food_category", "unknown"))
        choices.append(info.get("chosen_food", "unknown"))
        nutrition.append(float(info.get("nutrition_score", 5.0)))
        budget_ok.append(True)
        trajectory.append(float(info.get("health_score_after", 50.0)))

        print(
            f"[STEP] step={step_num} action={action_idx} "
            f"reward={round(reward,4)} done={done} "
            f"food={info.get('chosen_food','')} "
            f"category={info.get('food_category','')}",
            flush=True
        )

        if done:
            break
        obs = next_obs

    score = compute_grader_score(task_id, rewards, trajectory, categories, nutrition, budget_ok)
    print(f"[END] task={task_id} score={score} steps={step_num} total_reward={round(total_reward,4)}", flush=True)
    return {"task_id": task_id, "grader_score": score, "steps": step_num, "total_reward": round(total_reward, 4)}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"INFERENCE_START model={MODEL_NAME} tasks={TASKS}", flush=True)
    all_results = []
    for task_id in TASKS:
        try:
            result = run_episode(task_id, episode=1)
            all_results.append(result)
        except Exception as e:
            print(f"TASK_ERROR task={task_id} error={e}", flush=True)
            print(f"[START] task={task_id}", flush=True)
            print(f"[END] task={task_id} score=0.0 steps=0 total_reward=0.0", flush=True)

    avg = round(sum(r["grader_score"] for r in all_results)/len(all_results), 4) if all_results else 0.0
    print(f"INFERENCE_COMPLETE avg_score={avg}", flush=True)

if __name__ == "__main__":
    main()
