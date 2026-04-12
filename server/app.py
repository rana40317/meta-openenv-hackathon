import os
import sys
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from env.models import FoodChoiceAction, ResetResult, StepResult, StateResult
from env.environment import HealthyFoodEnvironment, TASKS

app = FastAPI(title="HealthyFoodChoice OpenEnv", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_envs: dict = {}

def get_env(task_id):
    if task_id not in TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task_id '{task_id}'")
    if task_id not in _envs:
        _envs[task_id] = HealthyFoodEnvironment(task_id=task_id)
    return _envs[task_id]

@app.get("/")
def root():
    return {"status": "running"}

@app.get("/health")
def health_check():
    return {"status": "ok", "environment": "HealthyFoodChoice", "tasks": list(TASKS.keys())}

@app.get("/tasks")
def list_tasks():
    return {"tasks": [{"id": t["id"], "difficulty": t["difficulty"], "num_meals": t["num_meals"]} for t in TASKS.values()]}

@app.post("/reset", response_model=ResetResult)
def reset(task_id: str = "task_1_easy"):
    return get_env(task_id).reset()

@app.post("/step", response_model=StepResult)
def step(action: FoodChoiceAction, task_id: str = "task_1_easy"):
    try:
        return get_env(task_id).step(action)
    except (RuntimeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/state", response_model=StateResult)
def state(task_id: str = "task_1_easy"):
    try:
        return get_env(task_id).state()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
