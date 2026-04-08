"""
Typed models for the HealthyFoodChoice OpenEnv environment.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class FoodCategory(str, Enum):
    HEALTHY = "healthy"
    JUNK = "junk"
    NEUTRAL = "neutral"


class FoodItem(BaseModel):
    name: str
    category: FoodCategory
    calories: int
    nutrition_score: float = Field(..., ge=0.0, le=10.0, description="Nutritional quality 0-10")
    price: float
    description: str


class MealContext(BaseModel):
    time_of_day: str  # breakfast, lunch, dinner, snack
    hunger_level: int = Field(..., ge=1, le=10)
    budget: float
    health_goal: str  # weight_loss, muscle_gain, maintenance, general_health
    previous_meals: List[str] = Field(default_factory=list)


class FoodChoiceAction(BaseModel):
    selected_item_index: int = Field(..., ge=0, description="Index of chosen food item from options")
    reasoning: Optional[str] = Field(None, description="Agent's reasoning for the choice")


class EnvironmentObservation(BaseModel):
    context: MealContext
    food_options: List[FoodItem]
    current_health_score: float = Field(..., ge=0.0, le=100.0)
    day: int
    meal_number: int
    task_id: str
    task_description: str


class StepResult(BaseModel):
    observation: EnvironmentObservation
    reward: float = Field(..., ge=0.0, le=1.0)
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class ResetResult(BaseModel):
    observation: EnvironmentObservation
    info: Dict[str, Any] = Field(default_factory=dict)


class StateResult(BaseModel):
    current_observation: EnvironmentObservation
    total_reward: float
    steps_taken: int
    episode_done: bool
    health_trajectory: List[float]
    choices_made: List[str]
