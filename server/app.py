import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.env_server import create_fastapi_app
from server.food_environment import FoodEnvironment
from server.models import FoodAction, FoodObservation

env = FoodEnvironment()
app = create_fastapi_app(env, FoodAction, FoodObservation)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
