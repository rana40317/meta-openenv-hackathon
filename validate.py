"""
validate.py — Pre-submission validation script.
Checks OpenEnv spec compliance, endpoints, graders, and inference script.

Usage:
  python validate.py [--base-url http://localhost:7860]
"""
import sys
import os
import json
import argparse
import importlib
import yaml
import requests

BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

PASS = "✅ PASS"
FAIL = "❌ FAIL"
WARN = "⚠️  WARN"

results = []


def check(name: str, passed: bool, detail: str = ""):
    status = PASS if passed else FAIL
    msg = f"{status}  {name}"
    if detail:
        msg += f"\n       {detail}"
    print(msg)
    results.append((name, passed))
    return passed


# ── 1. openenv.yaml validation ────────────────────────────────────────────────
def validate_yaml():
    print("\n── openenv.yaml ──────────────────────────────────────")
    try:
        with open("openenv.yaml") as f:
            spec = yaml.safe_load(f)
        check("openenv.yaml parseable", True)

        required_keys = ["name", "version", "tasks", "observation_space", "action_space", "reward"]
        for k in required_keys:
            check(f"  field '{k}' present", k in spec)

        tasks = spec.get("tasks", [])
        check("  3+ tasks defined", len(tasks) >= 3, f"found {len(tasks)}")

        for t in tasks:
            check(f"  task '{t.get('id')}' has grader", "grader" in t)

    except FileNotFoundError:
        check("openenv.yaml exists", False, "File not found")
    except Exception as e:
        check("openenv.yaml valid", False, str(e))


# ── 2. Typed models validation ────────────────────────────────────────────────
def validate_models():
    print("\n── Typed Models ──────────────────────────────────────")
    try:
        from env.models import (
            EnvironmentObservation, FoodChoiceAction,
            StepResult, ResetResult, StateResult
        )
        check("EnvironmentObservation importable", True)
        check("FoodChoiceAction importable", True)
        check("StepResult importable", True)
        check("ResetResult importable", True)
        check("StateResult importable", True)
    except ImportError as e:
        check("env.models importable", False, str(e))


# ── 3. Environment step/reset/state ──────────────────────────────────────────
def validate_environment():
    print("\n── Environment Logic ─────────────────────────────────")
    try:
        from env.environment import HealthyFoodEnvironment, TASKS
        from env.models import FoodChoiceAction

        check("HealthyFoodEnvironment importable", True)
        check("3+ tasks defined", len(TASKS) >= 3, f"found: {list(TASKS.keys())}")

        env = HealthyFoodEnvironment("task_1_easy", seed=0)
        reset_result = env.reset()
        check("reset() returns ResetResult", reset_result is not None)
        check("reset() has observation", hasattr(reset_result, "observation"))

        state = env.state()
        check("state() returns StateResult", state is not None)

        obs = reset_result.observation
        action = FoodChoiceAction(selected_item_index=0, reasoning="test")
        step_result = env.step(action)
        check("step() returns StepResult", step_result is not None)
        check("reward in [0,1]", 0.0 <= step_result.reward <= 1.0,
              f"reward={step_result.reward}")
        check("done is bool", isinstance(step_result.done, bool))

    except Exception as e:
        check("environment logic", False, str(e))


# ── 4. Graders validation ─────────────────────────────────────────────────────
def validate_graders():
    print("\n── Graders ───────────────────────────────────────────")
    try:
        from graders.task_graders import grade_task_1_easy, grade_task_2_medium, grade_task_3_hard
        from env.models import FoodCategory

        r1 = grade_task_1_easy(
            episode_rewards=[0.85, 0.9, 0.7, 0.88, 0.92],
            choices_made=["Quinoa Bowl", "Grilled Chicken", "Oatmeal", "Salmon", "Greek Yogurt"],
            health_trajectory=[55.0, 63.0, 71.0, 78.0, 87.0],
            food_categories=["healthy", "healthy", "healthy", "healthy", "healthy"]
        )
        check("grade_task_1_easy runs", True)
        check("  score in [0,1]", 0.0 <= r1["score"] <= 1.0, f"score={r1['score']}")

        r2 = grade_task_2_medium(
            episode_rewards=[0.8, 0.5, 0.85, 0.4, 0.9, 0.82, 0.78],
            choices_made=["A", "B", "C", "D", "E", "F", "G"],
            health_trajectory=[50, 57, 52, 60, 55, 63, 70],
            food_categories=["healthy", "neutral", "healthy", "junk", "healthy", "healthy", "healthy"],
            budget_respected=[True]*7
        )
        check("grade_task_2_medium runs", True)
        check("  score in [0,1]", 0.0 <= r2["score"] <= 1.0, f"score={r2['score']}")

        r3 = grade_task_3_hard(
            episode_rewards=[0.75]*10,
            choices_made=["X"]*10,
            health_trajectory=[50+i*3 for i in range(10)],
            food_categories=["healthy"]*8 + ["neutral", "healthy"],
            nutrition_scores=[8.0]*10,
            budget_respected=[True]*10
        )
        check("grade_task_3_hard runs", True)
        check("  score in [0,1]", 0.0 <= r3["score"] <= 1.0, f"score={r3['score']}")

    except Exception as e:
        check("graders importable", False, str(e))


# ── 5. HTTP endpoints ─────────────────────────────────────────────────────────
def validate_endpoints(base_url: str):
    print(f"\n── HTTP Endpoints ({base_url}) ────────────────────────")

    # Health check
    try:
        r = requests.get(f"{base_url}/health", timeout=10)
        check("/health returns 200", r.status_code == 200, f"status={r.status_code}")
    except Exception as e:
        check("/health reachable", False, str(e))
        return

    # Reset
    for task_id in ["task_1_easy", "task_2_medium", "task_3_hard"]:
        try:
            r = requests.post(f"{base_url}/reset", params={"task_id": task_id}, timeout=10)
            check(f"/reset task={task_id}", r.status_code == 200, f"status={r.status_code}")
        except Exception as e:
            check(f"/reset task={task_id}", False, str(e))

    # Step
    try:
        requests.post(f"{base_url}/reset", params={"task_id": "task_1_easy"}, timeout=10)
        r = requests.post(
            f"{base_url}/step",
            json={"selected_item_index": 0, "reasoning": "test"},
            params={"task_id": "task_1_easy"},
            timeout=10
        )
        check("/step returns 200", r.status_code == 200)
        data = r.json()
        reward = data.get("reward", -1)
        check("  reward in [0,1]", 0.0 <= reward <= 1.0, f"reward={reward}")
        check("  done field present", "done" in data)
    except Exception as e:
        check("/step works", False, str(e))

    # State
    try:
        r = requests.get(f"{base_url}/state", params={"task_id": "task_1_easy"}, timeout=10)
        check("/state returns 200", r.status_code == 200)
    except Exception as e:
        check("/state works", False, str(e))


# ── 6. Inference script ───────────────────────────────────────────────────────
def validate_inference_script():
    print("\n── Inference Script ──────────────────────────────────")
    check("inference.py exists", os.path.exists("inference.py"))

    with open("inference.py") as f:
        code = f.read()

    check("  imports OpenAI client", "from openai import OpenAI" in code or "import openai" in code)
    check("  uses API_BASE_URL env var", "API_BASE_URL" in code)
    check("  uses MODEL_NAME env var", "MODEL_NAME" in code)
    check("  uses HF_TOKEN env var", "HF_TOKEN" in code)
    check("  emits [START] event", '"START"' in code or "'START'" in code)
    check("  emits [STEP] event", '"STEP"' in code or "'STEP'" in code)
    check("  emits [END] event", '"END"' in code or "'END'" in code)


# ── 7. Dockerfile ─────────────────────────────────────────────────────────────
def validate_dockerfile():
    print("\n── Dockerfile ────────────────────────────────────────")
    check("Dockerfile exists", os.path.exists("Dockerfile"))
    if os.path.exists("Dockerfile"):
        with open("Dockerfile") as f:
            df = f.read()
        check("  EXPOSE 7860", "7860" in df)
        check("  CMD or ENTRYPOINT present", "CMD" in df or "ENTRYPOINT" in df)


# ── Summary ───────────────────────────────────────────────────────────────────
def print_summary():
    print("\n══════════════════════════════════════════════════════")
    passed = sum(1 for _, p in results if p)
    total  = len(results)
    print(f"  Validation complete: {passed}/{total} checks passed")
    if passed == total:
        print("  🎉 All checks passed! Ready to submit.")
    else:
        failed = [n for n, p in results if not p]
        print(f"  Failed checks: {', '.join(failed)}")
    print("══════════════════════════════════════════════════════")
    return passed == total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=BASE_URL,
                        help="Base URL of running environment server")
    parser.add_argument("--skip-endpoints", action="store_true",
                        help="Skip HTTP endpoint checks (if server not running)")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════╗")
    print("║   HealthyFoodChoice — Pre-Submission Validator        ║")
    print("╚══════════════════════════════════════════════════════╝")

    validate_yaml()
    validate_models()
    validate_environment()
    validate_graders()
    validate_inference_script()
    validate_dockerfile()

    if not args.skip_endpoints:
        validate_endpoints(args.base_url)
    else:
        print("\n── HTTP Endpoints ─────────────────── [SKIPPED] ──────")

    ok = print_summary()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
