
import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from env.models import FoodChoiceAction, ResetResult, StepResult, StateResult
from env.environment import HealthyFoodEnvironment, TASKS

app = FastAPI(
    title="HealthyFoodChoice OpenEnv",
    description="RL environment for healthy food decision-making",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Per-task environment instances
_envs: dict = {}


def get_env(task_id: str) -> HealthyFoodEnvironment:
    if task_id not in TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task_id '{task_id}'. Valid: {list(TASKS.keys())}")
    if task_id not in _envs:
        _envs[task_id] = HealthyFoodEnvironment(task_id=task_id)
    return _envs[task_id]


@app.get("/")
def root():
    return {
        "name": "HealthyFoodChoice OpenEnv",
        "version": "1.0.0",
        "status": "running",
        "docs": "http://localhost:7860/docs",
        "endpoints": {
            "health": "GET  /health",
            "tasks":  "GET  /tasks",
            "reset":  "POST /reset?task_id=task_1_easy",
            "step":   "POST /step?task_id=task_1_easy",
            "state":  "GET  /state?task_id=task_1_easy",
        }
    }


@app.get("/health")
def health_check():
    return {"status": "ok", "environment": "HealthyFoodChoice", "tasks": list(TASKS.keys())}


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "id": t["id"],
                "difficulty": t["difficulty"],
                "num_meals": t["num_meals"],
                "description": t["description"][:120] + "..."
            }
            for t in TASKS.values()
        ]
    }


@app.post("/reset", response_model=ResetResult)
def reset(task_id: str = "task_1_easy"):
    env = get_env(task_id)
    return env.reset()


@app.post("/step", response_model=StepResult)
def step(action: FoodChoiceAction, task_id: str = "task_1_easy"):
    env = get_env(task_id)
    try:
        return env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/state", response_model=StateResult)
def state(task_id: str = "task_1_easy"):
    env = get_env(task_id)
    try:
        return env.state()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    print(f"""
╔══════════════════════════════════════════════╗
║   HealthyFoodChoice RL Environment           ║
║   Running on http://localhost:{port}           ║
║                                              ║
║   API Docs → http://localhost:{port}/docs      ║
║   Health   → http://localhost:{port}/health    ║
║   Tasks    → http://localhost:{port}/tasks     ║
╚══════════════════════════════════════════════╝
    """)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
