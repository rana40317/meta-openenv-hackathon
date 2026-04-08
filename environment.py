"""
HealthyFoodChoice RL Environment - OpenEnv compliant implementation.
"""
import random
from typing import List, Optional, Dict, Any
from env.models import (
    FoodItem, FoodCategory, MealContext, FoodChoiceAction,
    EnvironmentObservation, StepResult, ResetResult, StateResult
)
from env.food_data import FOOD_DATABASE, MEAL_CONTEXTS


TASKS = {
    "task_1_easy": {
        "id": "task_1_easy",
        "description": (
            "Basic Healthy Choice: You are given a clear choice between one healthy "
            "and one junk food item. Choose the healthier option to maximize your health score. "
            "You have 5 meals to make the best choices."
        ),
        "num_meals": 5,
        "num_options": 2,
        "difficulty": "easy",
        "options_config": {"healthy_ratio": 0.5, "include_neutral": False},
        "budget": 20.0,
        "health_goal": "general_health",
    },
    "task_2_medium": {
        "id": "task_2_medium",
        "description": (
            "Nutritional Navigation: Choose from 4 food options that include healthy, neutral, "
            "and junk items. Consider your hunger level, budget, and health goal across 7 meals. "
            "Partial credit for neutral choices; full credit for healthy choices."
        ),
        "num_meals": 7,
        "num_options": 4,
        "difficulty": "medium",
        "options_config": {"healthy_ratio": 0.5, "include_neutral": True},
        "budget": 15.0,
        "health_goal": "weight_loss",
    },
    "task_3_hard": {
        "id": "task_3_hard",
        "description": (
            "Strategic Meal Planning: Navigate 10 meals across a full day with varying hunger "
            "levels, a tight budget of $12/meal, and a muscle gain goal. Choose from 5 options "
            "including deceptively labeled items. Optimize for protein, calories, and nutrition "
            "score simultaneously. Reward is based on cumulative health trajectory."
        ),
        "num_meals": 10,
        "num_options": 5,
        "difficulty": "hard",
        "options_config": {"healthy_ratio": 0.4, "include_neutral": True},
        "budget": 12.0,
        "health_goal": "muscle_gain",
    },
}


class HealthyFoodEnvironment:
    """
    OpenEnv-compliant environment for healthy food choice RL tasks.
    The agent must maximize healthy food choices across multiple meals.
    """

    def __init__(self, task_id: str = "task_1_easy", seed: Optional[int] = 42):
        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id: {task_id}. Choose from {list(TASKS.keys())}")
        self.task_config = TASKS[task_id]
        self.seed = seed
        self.rng = random.Random(seed)

        # Episode state
        self._current_obs: Optional[EnvironmentObservation] = None
        self._total_reward: float = 0.0
        self._steps: int = 0
        self._done: bool = False
        self._health_trajectory: List[float] = []
        self._choices_made: List[str] = []
        self._current_health_score: float = 50.0  # Start at 50/100

    def reset(self) -> ResetResult:
        """Reset the environment to initial state."""
        self.rng = random.Random(self.seed)
        self._total_reward = 0.0
        self._steps = 0
        self._done = False
        self._health_trajectory = []
        self._choices_made = []
        self._current_health_score = 50.0

        obs = self._generate_observation()
        self._current_obs = obs

        return ResetResult(
            observation=obs,
            info={"task_id": self.task_config["id"], "difficulty": self.task_config["difficulty"]}
        )

    def step(self, action: FoodChoiceAction) -> StepResult:
        """Take a step in the environment."""
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        if self._current_obs is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        options = self._current_obs.food_options
        if action.selected_item_index < 0 or action.selected_item_index >= len(options):
            raise ValueError(
                f"Invalid action index {action.selected_item_index}. "
                f"Must be 0-{len(options)-1}"
            )

        chosen = options[action.selected_item_index]
        reward = self._compute_reward(chosen)

        # Update health score
        delta = self._health_delta(chosen)
        self._current_health_score = max(0.0, min(100.0, self._current_health_score + delta))
        self._health_trajectory.append(self._current_health_score)
        self._choices_made.append(chosen.name)
        self._total_reward += reward
        self._steps += 1

        done = self._steps >= self.task_config["num_meals"]
        self._done = done

        if not done:
            next_obs = self._generate_observation()
            self._current_obs = next_obs
        else:
            next_obs = self._current_obs

        return StepResult(
            observation=next_obs,
            reward=round(reward, 4),
            done=done,
            info={
                "chosen_food": chosen.name,
                "food_category": chosen.category.value,
                "nutrition_score": chosen.nutrition_score,
                "health_score_after": self._current_health_score,
                "step": self._steps,
                "reasoning": action.reasoning,
            }
        )

    def state(self) -> StateResult:
        """Return current environment state."""
        if self._current_obs is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return StateResult(
            current_observation=self._current_obs,
            total_reward=round(self._total_reward, 4),
            steps_taken=self._steps,
            episode_done=self._done,
            health_trajectory=self._health_trajectory,
            choices_made=self._choices_made,
        )

    def _generate_observation(self) -> EnvironmentObservation:
        """Generate a new observation (meal scenario)."""
        cfg = self.task_config
        meal_types = ["breakfast", "lunch", "dinner", "snack"]
        meal_type = meal_types[self._steps % 4]

        context = MealContext(
            time_of_day=meal_type,
            hunger_level=self.rng.randint(3, 9),
            budget=cfg["budget"],
            health_goal=cfg["health_goal"],
            previous_meals=list(self._choices_made[-3:]),
        )

        food_options = self._sample_food_options()

        return EnvironmentObservation(
            context=context,
            food_options=food_options,
            current_health_score=self._current_health_score,
            day=(self._steps // 4) + 1,
            meal_number=self._steps + 1,
            task_id=cfg["id"],
            task_description=cfg["description"],
        )

    def _sample_food_options(self) -> List[FoodItem]:
        """Sample food options based on task configuration."""
        cfg = self.task_config["options_config"]
        num = self.task_config["num_options"]
        include_neutral = cfg["include_neutral"]
        healthy_ratio = cfg["healthy_ratio"]

        num_healthy = max(1, round(num * healthy_ratio))
        num_junk = num - num_healthy if not include_neutral else max(1, (num - num_healthy) // 2)
        num_neutral = num - num_healthy - num_junk if include_neutral else 0

        options = []
        options += self.rng.sample(FOOD_DATABASE["healthy"], min(num_healthy, len(FOOD_DATABASE["healthy"])))
        options += self.rng.sample(FOOD_DATABASE["junk"], min(num_junk, len(FOOD_DATABASE["junk"])))
        if num_neutral > 0:
            options += self.rng.sample(FOOD_DATABASE["neutral"], min(num_neutral, len(FOOD_DATABASE["neutral"])))

        self.rng.shuffle(options)
        return options[:num]

    def _compute_reward(self, chosen: FoodItem) -> float:
        """
        Compute reward with partial progress signals.
        - Healthy choice: 0.7 base + up to 0.3 bonus from nutrition_score
        - Neutral choice: 0.3-0.5 depending on nutrition_score
        - Junk choice: 0.0-0.2 (small reward even for junk to avoid zero-reward collapse)
        """
        if chosen.category == FoodCategory.HEALTHY:
            base = 0.7
            bonus = (chosen.nutrition_score / 10.0) * 0.3
            reward = base + bonus
        elif chosen.category == FoodCategory.NEUTRAL:
            reward = 0.3 + (chosen.nutrition_score / 10.0) * 0.2
        else:  # JUNK
            # Tiny reward so gradient doesn't collapse; penalizes heavily
            reward = max(0.0, (chosen.nutrition_score / 10.0) * 0.2)

        # Budget adherence bonus (small)
        if chosen.price <= self.task_config["budget"]:
            reward = min(1.0, reward + 0.02)

        return round(min(1.0, max(0.0, reward)), 4)

    def _health_delta(self, chosen: FoodItem) -> float:
        """Change in health score based on food choice."""
        if chosen.category == FoodCategory.HEALTHY:
            return chosen.nutrition_score * 1.5
        elif chosen.category == FoodCategory.NEUTRAL:
            return 0.0
        else:
            return -(10.0 - chosen.nutrition_score) * 1.2
