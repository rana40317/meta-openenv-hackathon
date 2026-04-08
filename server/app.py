"""
server/app.py — FastAPI server for HealthyFoodChoice RL environment.
"""
from openenv.core.env_server import create_fastapi_app
from .food_environment import FoodEnvironment
from ..models import FoodAction, FoodObservation
import uvicorn

env = FoodEnvironment()
app = create_fastapi_app(env, FoodAction, FoodObservation)

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
