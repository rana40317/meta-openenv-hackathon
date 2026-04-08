"""
Food database for the HealthyFoodChoice environment.
"""
from env.models import FoodItem, FoodCategory

FOOD_DATABASE = {
    "healthy": [
        FoodItem(name="Grilled Chicken Salad", category=FoodCategory.HEALTHY, calories=320,
                 nutrition_score=8.5, price=12.0,
                 description="Fresh greens with grilled chicken, tomatoes, cucumbers"),
        FoodItem(name="Quinoa Buddha Bowl", category=FoodCategory.HEALTHY, calories=410,
                 nutrition_score=9.0, price=13.5,
                 description="Quinoa, roasted veggies, chickpeas, tahini dressing"),
        FoodItem(name="Oatmeal with Berries", category=FoodCategory.HEALTHY, calories=280,
                 nutrition_score=8.0, price=5.0,
                 description="Rolled oats topped with fresh berries and honey"),
        FoodItem(name="Grilled Salmon", category=FoodCategory.HEALTHY, calories=450,
                 nutrition_score=9.2, price=18.0,
                 description="Omega-3 rich salmon with steamed broccoli"),
        FoodItem(name="Lentil Soup", category=FoodCategory.HEALTHY, calories=230,
                 nutrition_score=8.7, price=8.0,
                 description="Protein-packed lentils with carrots and spices"),
        FoodItem(name="Greek Yogurt Parfait", category=FoodCategory.HEALTHY, calories=190,
                 nutrition_score=7.8, price=6.0,
                 description="Greek yogurt with granola and fresh fruit"),
        FoodItem(name="Avocado Toast", category=FoodCategory.HEALTHY, calories=340,
                 nutrition_score=7.5, price=9.0,
                 description="Whole grain toast with avocado, eggs, and seeds"),
        FoodItem(name="Mixed Nuts", category=FoodCategory.HEALTHY, calories=170,
                 nutrition_score=7.0, price=4.0,
                 description="Almonds, walnuts, cashews - heart-healthy fats"),
        FoodItem(name="Veggie Stir Fry", category=FoodCategory.HEALTHY, calories=290,
                 nutrition_score=8.3, price=10.0,
                 description="Colorful vegetables with tofu in light sauce"),
        FoodItem(name="Brown Rice Bowl", category=FoodCategory.HEALTHY, calories=380,
                 nutrition_score=7.9, price=9.5,
                 description="Brown rice with black beans, peppers, salsa"),
    ],
    "junk": [
        FoodItem(name="Double Cheeseburger", category=FoodCategory.JUNK, calories=720,
                 nutrition_score=2.0, price=8.5,
                 description="Two beef patties, cheese, special sauce, fries"),
        FoodItem(name="Large Pepperoni Pizza", category=FoodCategory.JUNK, calories=800,
                 nutrition_score=2.5, price=14.0,
                 description="Three slices of greasy pepperoni pizza"),
        FoodItem(name="Fried Chicken Bucket", category=FoodCategory.JUNK, calories=950,
                 nutrition_score=1.8, price=12.0,
                 description="Deep-fried chicken pieces with coleslaw"),
        FoodItem(name="Chocolate Milkshake", category=FoodCategory.JUNK, calories=580,
                 nutrition_score=1.5, price=5.5,
                 description="Extra thick chocolate milkshake with whipped cream"),
        FoodItem(name="Nachos Supreme", category=FoodCategory.JUNK, calories=660,
                 nutrition_score=2.2, price=9.0,
                 description="Loaded nachos with cheese, sour cream, jalapeños"),
        FoodItem(name="Hot Dog with Fries", category=FoodCategory.JUNK, calories=740,
                 nutrition_score=1.9, price=7.0,
                 description="Beef hot dog in bun with large fries"),
        FoodItem(name="Candy Bar Pack", category=FoodCategory.JUNK, calories=420,
                 nutrition_score=1.2, price=3.5,
                 description="Assorted chocolate candy bars"),
        FoodItem(name="Deep Dish Mac & Cheese", category=FoodCategory.JUNK, calories=680,
                 nutrition_score=2.8, price=11.0,
                 description="Creamy loaded macaroni and cheese"),
        FoodItem(name="Onion Rings", category=FoodCategory.JUNK, calories=480,
                 nutrition_score=2.0, price=5.0,
                 description="Crispy battered deep-fried onion rings"),
        FoodItem(name="Soda + Chips Combo", category=FoodCategory.JUNK, calories=390,
                 nutrition_score=1.0, price=4.0,
                 description="Large soda with a bag of potato chips"),
    ],
    "neutral": [
        FoodItem(name="Cheese Sandwich", category=FoodCategory.NEUTRAL, calories=420,
                 nutrition_score=5.0, price=6.0,
                 description="Whole wheat bread with cheese and lettuce"),
        FoodItem(name="Vegetable Soup", category=FoodCategory.NEUTRAL, calories=180,
                 nutrition_score=6.0, price=7.0,
                 description="Mixed vegetable broth-based soup"),
        FoodItem(name="Pasta Marinara", category=FoodCategory.NEUTRAL, calories=460,
                 nutrition_score=5.5, price=10.0,
                 description="Spaghetti with tomato sauce and herbs"),
        FoodItem(name="Egg Fried Rice", category=FoodCategory.NEUTRAL, calories=410,
                 nutrition_score=5.2, price=8.0,
                 description="Fried rice with scrambled eggs and vegetables"),
    ]
}

MEAL_CONTEXTS = {
    "breakfast": {
        "time_of_day": "breakfast",
        "typical_hunger": (4, 8),
        "health_goals": ["weight_loss", "general_health", "muscle_gain", "maintenance"]
    },
    "lunch": {
        "time_of_day": "lunch",
        "typical_hunger": (5, 9),
        "health_goals": ["weight_loss", "general_health", "muscle_gain", "maintenance"]
    },
    "dinner": {
        "time_of_day": "dinner",
        "typical_hunger": (6, 10),
        "health_goals": ["weight_loss", "general_health", "muscle_gain", "maintenance"]
    },
    "snack": {
        "time_of_day": "snack",
        "typical_hunger": (2, 6),
        "health_goals": ["weight_loss", "general_health", "maintenance"]
    }
}
