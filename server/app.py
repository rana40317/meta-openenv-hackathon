"""
server/app.py — FastAPI server for HealthyFoodChoice RL environment.
Uses openenv.core.env_server to create the app.
"""
from openenv.core.env_server import create_fastapi_app
from .food_environment import FoodEnvironment
from ..models import FoodAction, FoodObservation

env = FoodEnvironment()
app = create_fastapi_app(env, FoodAction, FoodObservation)
