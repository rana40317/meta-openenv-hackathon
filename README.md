# 🥗 HealthyFoodChoice — OpenEnv RL Environment

An OpenEnv-compliant reinforcement learning environment where an LLM agent must make **healthy food choices** across multiple meals. The agent observes meal context and food options, then selects the healthiest item to maximize its health score.

---

## 🌍 Environment Description

The agent simulates a person navigating daily food choices — breakfast, lunch, dinner, and snacks — across varying contexts:

- **Hunger levels** (1–10 scale)
- **Budget constraints** (per-meal spending limit)
- **Health goals** (weight loss, muscle gain, maintenance, general health)
- **Previous meal history** (to track patterns)

Each food item has a `nutrition_score` (0–10), calorie count, price, and category (healthy / neutral / junk).

### Real-World Relevance
Poor dietary choices contribute to obesity, diabetes, and cardiovascular disease. This environment trains agents to provide evidence-based meal recommendations that maximize nutritional quality within real-world constraints.

---

## 📦 Project Structure

```
food-rl-env/
├── app.py               # FastAPI OpenEnv server
├── inference.py         # Baseline LLM agent inference script
├── validate.py          # Pre-submission validation script
├── openenv.yaml         # OpenEnv specification
├── Dockerfile           # HuggingFace Spaces deployment
├── requirements.txt
├── env/
│   ├── models.py        # Typed Pydantic models
│   ├── environment.py   # RL environment logic (step/reset/state)
│   └── food_data.py     # Food item database
└── graders/
    └── task_graders.py  # Per-task grading functions
```

---

## 🎮 Tasks

| Task | Difficulty | Steps | Description |
|------|-----------|-------|-------------|
| `task_1_easy` | Easy | 5 | Binary choice: 1 healthy vs 1 junk per meal |
| `task_2_medium` | Medium | 7 | 4 options (healthy + neutral + junk), budget + weight-loss goal |
| `task_3_hard` | Hard | 10 | 5 options, tight budget, muscle-gain goal, health trajectory optimization |

---

## 📐 Observation Space

```python
EnvironmentObservation:
  context: MealContext          # hunger, budget, health_goal, time_of_day
  food_options: List[FoodItem]  # 2–5 options per meal
  current_health_score: float   # 0.0–100.0
  day: int                      # current day
  meal_number: int              # step index
  task_id: str
  task_description: str
```

## 🕹️ Action Space

```python
FoodChoiceAction:
  selected_item_index: int   # 0-based index into food_options
  reasoning: str (optional)  # LLM reasoning for the choice
```

## 🏆 Reward Function

| Food Category | Reward Range | Formula |
|--------------|-------------|---------|
| Healthy | 0.70–1.00 | `0.7 + (nutrition_score/10) × 0.3` |
| Neutral | 0.30–0.50 | `0.3 + (nutrition_score/10) × 0.2` |
| Junk | 0.00–0.20 | `(nutrition_score/10) × 0.2` |
| Budget bonus | +0.02 | if `price ≤ budget` |

The reward provides **partial progress signals** — even suboptimal choices yield non-zero rewards — avoiding gradient collapse in policy learning.

---

## 🚀 Setup & Running

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Start the environment server
python app.py
# → Server running at http://localhost:7860

# Run validation (with server running)
python validate.py

# Run baseline inference (set env vars first)
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="sk-..."
export ENV_BASE_URL="http://localhost:7860"
python inference.py
```

### Docker

```bash
docker build -t food-rl-env .
docker run -p 7860:7860 food-rl-env
```

---

## 🔌 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check → `{"status": "ok"}` |
| `GET` | `/tasks` | List all tasks |
| `POST` | `/reset?task_id=task_1_easy` | Reset environment, get first observation |
| `POST` | `/step?task_id=task_1_easy` | Submit action, get next obs + reward |
| `GET` | `/state?task_id=task_1_easy` | Current episode state |

### Example: Reset

```bash
curl -X POST "http://localhost:7860/reset?task_id=task_1_easy"
```

### Example: Step

```bash
curl -X POST "http://localhost:7860/step?task_id=task_1_easy" \
  -H "Content-Type: application/json" \
  -d '{"selected_item_index": 0, "reasoning": "Highest nutrition score"}'
```

---

## 📊 Grader Scoring

### Task 1 (Easy)
```
score = 0.6 × healthy_ratio + 0.4 × avg_reward
```

### Task 2 (Medium)
```
score = 0.40 × healthy_ratio
      + 0.30 × health_improvement_normalized
      + 0.15 × budget_adherence
      + 0.15 × (1 - consecutive_junk_penalty)
```

### Task 3 (Hard)
```
score = 0.35 × health_trajectory_upward_trend
      + 0.25 × avg_nutrition_score_normalized
      + 0.25 × healthy_ratio (+ consistency streak bonus)
      + 0.15 × reward_efficiency
```

All scores are in **[0.0, 1.0]**.

---

## 📋 Inference Script Log Format

The `inference.py` script emits structured JSON logs to stdout:

```json
{"event": "START",  "task_id": "task_1_easy", "episode": 1, "model": "...", "timestamp": ...}
{"event": "STEP",   "task_id": "task_1_easy", "episode": 1, "step": 1, "action": 0, "reward": 0.85, "done": false, "chosen_food": "Grilled Chicken Salad", "food_category": "healthy", "health_score": 62.75, "timestamp": ...}
{"event": "END",    "task_id": "task_1_easy", "episode": 1, "total_reward": 4.25, "grader_score": 0.91, "steps": 5, "choices": [...], "timestamp": ...}
```

---

## 🌐 HuggingFace Spaces Deployment

1. Create a new HF Space (Docker SDK)
2. Push this repository
3. Set Space secrets: `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`
4. Space will auto-build from `Dockerfile` and expose on port 7860

---

## ✅ Pre-Submission Checklist

```bash
# Run full validator
python validate.py --base-url http://localhost:7860

# Expected output: All checks passed ✅
```

Checks performed:
- ✅ `openenv.yaml` valid with 3+ tasks
- ✅ Typed models (Pydantic)
- ✅ `reset()` / `step()` / `state()` endpoints
- ✅ Rewards in [0.0, 1.0]
- ✅ All 3 graders runnable
- ✅ `inference.py` uses OpenAI client + env vars
- ✅ Structured [START]/[STEP]/[END] logs
- ✅ Dockerfile with port 7860

---

## 🔧 Environment Variables

| Variable | Description |
|----------|-------------|
| `API_BASE_URL` | LLM API endpoint (OpenAI-compatible) |
| `MODEL_NAME` | Model identifier (e.g., `gpt-4o-mini`) |
| `HF_TOKEN` | HuggingFace / API key |
| `ENV_BASE_URL` | Environment server URL (default: `http://localhost:7860`) |
