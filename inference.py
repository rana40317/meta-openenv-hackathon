"""
inference.py — HealthyFoodChoice RL inference script.
Always starts the local server regardless of ENV_BASE_URL.
"""
import os, sys, json, time, threading

try:
    import requests
except ImportError as e:
    print(json.dumps({"event":"IMPORT_ERROR","error":str(e)}),flush=True); sys.exit(1)
try:
    from openai import OpenAI
except ImportError as e:
    print(json.dumps({"event":"IMPORT_ERROR","error":str(e)}),flush=True); sys.exit(1)

API_BASE_URL = os.environ.get("API_BASE_URL") or "https://api-inference.huggingface.co/v1"
MODEL_NAME   = os.environ.get("MODEL_NAME")   or "Qwen/Qwen2.5-72B-Instruct"
HF_TOKEN     = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "dummy-token"
LOCAL_PORT   = 7860
ACTIVE_ENV_URL = f"http://localhost:{LOCAL_PORT}"
TASKS = ["task_1_easy", "task_2_medium", "task_3_hard"]

print(json.dumps({"event":"CONFIG","api_base_url":API_BASE_URL,"model":MODEL_NAME,
    "env_url":ACTIVE_ENV_URL,"timestamp":time.time()}),flush=True)

# Always start local server — ignore ENV_BASE_URL, we control the environment
def start_local_server(port):
    try:
        import uvicorn
        repo_root = os.path.dirname(os.path.abspath(__file__))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        try:
            from server.app import app as fastapi_app
            print(json.dumps({"event":"SERVER_IMPORT","source":"server.app"}),flush=True)
        except Exception as e1:
            print(json.dumps({"event":"SERVER_IMPORT_WARN","error":str(e1)}),flush=True)
            from app import app as fastapi_app
            print(json.dumps({"event":"SERVER_IMPORT","source":"app"}),flush=True)
        uvicorn.Server(uvicorn.Config(fastapi_app,host="0.0.0.0",port=port,log_level="error")).run()
    except Exception as e:
        print(json.dumps({"event":"SERVER_ERROR","error":str(e)}),flush=True)

def wait_for_server(url, timeout=60):
    for i in range(timeout):
        try:
            if requests.get(f"{url}/health",timeout=2).status_code == 200:
                print(json.dumps({"event":"SERVER_READY","waited":i}),flush=True)
                return True
        except: pass
        time.sleep(1)
    return False

print(json.dumps({"event":"STARTING_LOCAL_SERVER","port":LOCAL_PORT}),flush=True)
t = threading.Thread(target=start_local_server,args=(LOCAL_PORT,),daemon=True)
t.start()
if not wait_for_server(ACTIVE_ENV_URL):
    print(json.dumps({"event":"FATAL","error":"Server failed to start"}),flush=True)
    sys.exit(1)

try:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
except Exception as e:
    print(json.dumps({"event":"CLIENT_INIT_ERROR","error":str(e)}),flush=True); sys.exit(1)

def log_start(task_id,episode):
    print(json.dumps({"event":"START","task_id":task_id,"episode":episode,"model":MODEL_NAME,"timestamp":time.time()}),flush=True)
def log_step(task_id,episode,step,action,reward,done,info):
    print(json.dumps({"event":"STEP","task_id":task_id,"episode":episode,"step":step,"action":action,
        "reward":reward,"done":done,"chosen_food":info.get("chosen_food",""),
        "food_category":info.get("food_category",""),"health_score":info.get("health_score_after",0.0),
        "timestamp":time.time()}),flush=True)
def log_end(task_id,episode,total_reward,grader_score,steps,choices):
    print(json.dumps({"event":"END","task_id":task_id,"episode":episode,"total_reward":total_reward,
        "grader_score":grader_score,"steps":steps,"choices":choices,"timestamp":time.time()}),flush=True)

def env_reset(task_id):
    r = requests.post(f"{ACTIVE_ENV_URL}/reset",params={"task_id":task_id},timeout=30)
    r.raise_for_status(); return r.json()
def env_step(task_id,action_index,reasoning=""):
    r = requests.post(f"{ACTIVE_ENV_URL}/step",json={"selected_item_index":action_index,"reasoning":reasoning},
        params={"task_id":task_id},timeout=30)
    r.raise_for_status(); return r.json()

SYSTEM_PROMPT = """You are a health-conscious food advisor.
Given food options numbered from 0, respond ONLY with JSON:
{"selected_item_index": <integer>, "reasoning": "<brief explanation>"}
Always pick the highest nutrition_score option."""

def agent_choose(obs):
    try:
        r = client.chat.completions.create(model=MODEL_NAME,
            messages=[{"role":"system","content":SYSTEM_PROMPT},
                      {"role":"user","content":str(obs)}],
            max_tokens=100,temperature=0.1)
        raw = r.choices[0].message.content.strip().replace("```json","").replace("```","")
        p = json.loads(raw)
        idx = max(0,min(int(p.get("selected_item_index",0)),len(obs["food_options"])-1))
        return idx, p.get("reasoning","")
    except Exception as e:
        print(json.dumps({"event":"AGENT_ERROR","error":str(e)}),flush=True)
        return 0, "fallback"

def run_episode(task_id,episode=1):
    log_start(task_id,episode)
    obs = env_reset(task_id)["observation"]
    total_reward=0.0; step_num=0
    rewards,categories,choices,nutrition,budget_ok,trajectory=[],[],[],[],[],[]
    while True:
        action_idx,reasoning = agent_choose(obs)
        result = env_step(task_id,action_idx,reasoning)
        reward,done,info,next_obs = result["reward"],result["done"],result.get("info",{}),result["observation"]
        total_reward+=reward; step_num+=1
        rewards.append(reward); categories.append(info.get("food_category","unknown"))
        choices.append(info.get("chosen_food","unknown"))
        nutrition.append(float(info.get("nutrition_score",5.0))); budget_ok.append(True)
        trajectory.append(float(info.get("health_score_after",50.0)))
        log_step(task_id,episode,step_num,action_idx,reward,done,info)
        if done: break
        obs = next_obs
    gs = compute_grader_score(task_id,rewards,choices,trajectory,categories,nutrition,budget_ok)
    log_end(task_id,episode,round(total_reward,4),gs,step_num,choices)
    return {"task_id":task_id,"episode":episode,"total_reward":round(total_reward,4),"grader_score":gs,"steps":step_num,"choices":choices}

def compute_grader_score(task_id,rewards,choices,trajectory,categories,nutrition,budget_ok):
    if task_id=="task_1_easy":
        hr=sum(1 for c in categories if c=="healthy")/len(categories) if categories else 0.0
        return round(min(1.0,0.6*hr+0.4*(sum(rewards)/len(rewards) if rewards else 0)),4)
    elif task_id=="task_2_medium":
        hr=sum(1 for c in categories if c=="healthy")/len(categories) if categories else 0.0
        hi=min(1.0,max(0,trajectory[-1]-trajectory[0])/50.0) if len(trajectory)>=2 else 0.0
        br=sum(budget_ok)/len(budget_ok) if budget_ok else 1.0
        consec=max_c=0
        for c in categories:
            consec=consec+1 if c=="junk" else 0; max_c=max(max_c,consec)
        cp=1.0-min(1.0,max_c/len(categories))
        return round(min(1.0,max(0.0,0.40*hr+0.30*hi+0.15*br+0.15*cp)),4)
    elif task_id=="task_3_hard":
        ts=sum(1 for i in range(1,len(trajectory)) if trajectory[i]>=trajectory[i-1])/(len(trajectory)-1) if len(trajectory)>=2 else 0.0
        nc=(sum(nutrition)/len(nutrition)/10.0) if nutrition else 0.0
        hr=sum(1 for c in categories if c=="healthy")/len(categories) if categories else 0.0
        re=sum(rewards)/len(rewards) if rewards else 0.0
        return round(min(1.0,max(0.0,0.35*ts+0.25*nc+0.25*hr+0.15*re)),4)
    return 0.0

def main():
    print(json.dumps({"event":"INFERENCE_START","model":MODEL_NAME,"env_url":ACTIVE_ENV_URL,"tasks":TASKS,"timestamp":time.time()}),flush=True)
    all_results=[]
    for task_id in TASKS:
        try:
            all_results.append(run_episode(task_id,episode=1))
        except Exception as e:
            print(json.dumps({"event":"TASK_ERROR","task_id":task_id,"error":str(e)}),flush=True)
    print(json.dumps({"event":"INFERENCE_COMPLETE","results":all_results,
        "avg_grader_score":round(sum(r["grader_score"] for r in all_results)/len(all_results),4) if all_results else 0.0,
        "timestamp":time.time()}),flush=True)

if __name__=="__main__":
    main()
