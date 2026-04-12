"""
inference.py — HealthyFoodChoice RL inference script.
Uses API_BASE_URL and API_KEY injected by validator.
Prints [START]/[STEP]/[END] blocks to stdout.
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

# Use EXACTLY what validator injects - no fallback to other providers
API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY      = os.environ["API_KEY"]
MODEL_NAME   = os.environ.get("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASKS = ["task_1_easy", "task_2_medium", "task_3_hard"]

print(f"CONFIG api_base={API_BASE_URL} model={MODEL_NAME}", flush=True)

# Start local env server
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
    print("SERVER_FAILED could not start on any port", flush=True)
    for task_id in TASKS:
        print(f"[START] task={task_id}", flush=True)
        print(f"[END] task={task_id} score=0.0 steps=0", flush=True)
    sys.exit(0)

# Init OpenAI client with validator-injected credentials
client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

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

SYSTEM_PROMPT = """You are a health-conscious food advisor.
Given food options numbered from 0, respond ONLY with JSON:
{"selected_item_index": <integer>, "reasoning": "<brief explanation>"}
Pick the option with the highest nutrition_score."""

def build_prompt(obs):
    ctx = obs["context"]
    options = "\n".join([
        f"{i}. {f['name']} | Cal:{f['calories']} | NutriScore:{f['nutrition_score']}/10 | Price:${f['price']}"
        for i, f in enumerate(obs["food_options"])
    ])
    return (f"Health Goal: {ctx['health_goal']}\nBudget: ${ctx['budget']}\n"
            f"Current Health Score: {obs['current_health_score']}/100\n\n"
            f"Options:\n{options}\n\nChoose the healthiest. Reply only with JSON.")

def agent_choose(obs):
    # Make API call through validator's proxy - no silent fallback
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_prompt(obs)}
        ],
        max_tokens=150,
        temperature=0.1,
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    parsed = json.loads(raw)
    idx = int(parsed.get("selected_item_index", 0))
    idx = max(0, min(idx, len(obs["food_options"]) - 1))
    return idx, parsed.get("reasoning", "")

def compute_grader_score(task_id, rewards, trajectory, categories, nutrition, budget_ok):
    if task_id == "task_1_easy":
        hr = sum(1 for c in categories if c == "healthy") / len(categories) if categories else 0.0
        return round(min(1.0, 0.6*hr + 0.4*(sum(rewards)/len(rewards) if rewards else 0)), 4)
    elif task_id == "task_2_medium":
        hr = sum(1 for c in categories if c == "healthy") / len(categories) if categories else 0.0
        hi = min(1.0, max(0, trajectory[-1]-trajectory[0])/50.0) if len(trajectory) >= 2 else 0.0
        br = sum(budget_ok)/len(budget_ok) if budget_ok else 1.0
        consec = max_c = 0
        for c in categories:
            consec = consec + 1 if c == "junk" else 0
            max_c = max(max_c, consec)
        cp = 1.0 - min(1.0, max_c/len(categories))
        return round(min(1.0, max(0.0, 0.40*hr + 0.30*hi + 0.15*br + 0.15*cp)), 4)
    elif task_id == "task_3_hard":
        ts = sum(1 for i in range(1, len(trajectory)) if trajectory[i] >= trajectory[i-1]) / (len(trajectory)-1) if len(trajectory) >= 2 else 0.0
        nc = (sum(nutrition)/len(nutrition)/10.0) if nutrition else 0.0
        hr = sum(1 for c in categories if c == "healthy") / len(categories) if categories else 0.0
        re = sum(rewards)/len(rewards) if rewards else 0.0
        return round(min(1.0, max(0.0, 0.35*ts + 0.25*nc + 0.25*hr + 0.15*re)), 4)
    return 0.0

def run_episode(task_id, episode=1):
    print(f"[START] task={task_id}", flush=True)

    obs = env_reset(task_id)["observation"]
    total_reward = 0.0
    step_num = 0
    rewards, categories, choices, nutrition, budget_ok, trajectory = [], [], [], [], [], []

    while True:
        try:
            action_idx, reasoning = agent_choose(obs)
        except Exception as e:
            print(f"AGENT_ERROR step={step_num+1} error={e}", flush=True)
            action_idx, reasoning = 0, "fallback"

        result   = env_step(task_id, action_idx, reasoning)
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

        print(f"[STEP] step={step_num} action={action_idx} reward={round(reward,4)} done={done} food={info.get('chosen_food','')} category={info.get('food_category','')}", flush=True)

        if done:
            break
        obs = next_obs

    score = compute_grader_score(task_id, rewards, trajectory, categories, nutrition, budget_ok)
    print(f"[END] task={task_id} score={score} steps={step_num} total_reward={round(total_reward,4)}", flush=True)
    return {"task_id": task_id, "grader_score": score, "steps": step_num, "total_reward": round(total_reward, 4)}

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
