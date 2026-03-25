import random

nutrients = ["calories", "protein", "carbs", "fiber", "fat", "saturated fat", "sugar", "sodium", "iron", "calcium"]
units = {"calories": "kcal", "protein": "g", "carbs": "g", "fiber": "g", "fat": "g", "sugar": "g", "sodium": "mg", "iron": "mg"}
cuisines = ["Italian", "Mexican", "Indian", "Chinese", "Mediterranean", "Japanese", "French"]
meals = ["breakfast", "lunch", "dinner", "snack"]

new_queries = []

for _ in range(100):
    type = random.choice(["range", "min", "max", "combo"])
    nutrient = random.choice(nutrients)
    unit = units.get(nutrient, "g")
    
    if type == "range":
        low, high = sorted([random.randint(5, 20), random.randint(25, 50)])
        q = f"I need a {random.choice(meals)} with between {low}{unit} and {high}{unit} of {nutrient}."
    elif type == "min":
        val = random.randint(20, 60)
        q = f"Give me a {random.choice(cuisines)} dish that has at least {val}{unit} of {nutrient}."
    elif type == "max":
        val = random.randint(5, 15)
        q = f"I want a {random.choice(meals)} with a maximum of {val}{unit} of {nutrient}."
    else: # combo
        q = f"Show me a {random.choice(meals)} under {random.randint(500, 800)} calories but with more than {random.randint(20, 40)}g of protein."
    
    new_queries.append(q)

# This list was then cleaned and saved to the .txt file provided above.