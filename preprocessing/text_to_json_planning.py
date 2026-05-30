import json
import argparse
from typing import List, Optional
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import os

# =========================
# Models
# =========================

class RecipeConstraints(BaseModel):
    category: Optional[str]
    cuisine: Optional[str]
    ingredients_incl: Optional[List[Optional[str]]]
    ingredients_excl: Optional[List[Optional[str]]]
    utensils_incl: Optional[List[Optional[str]]]
    utensils_excl: Optional[List[Optional[str]]]
    time_class: Optional[str] #"Duration (very short, short, average, long, very long)")
    
    # Nutrients as [min, max]
    calories_kcal: Optional[List[Optional[float]]]
    carbohydrates_g: Optional[List[Optional[float]]]
    cholesterol_mg: Optional[List[Optional[float]]]
    fiber_g: Optional[List[Optional[float]]]
    protein_g: Optional[List[Optional[float]]]
    saturated_fat_g: Optional[List[Optional[float]]]
    sodium_mg: Optional[List[Optional[float]]]
    sugar_g: Optional[List[Optional[float]]]
    fat_g: Optional[List[Optional[float]]]
    unsaturated_fat_g: Optional[List[Optional[float]]]

    extra: Optional[str]


class GlobalConstraints(BaseModel):
    # Aggregate nutrients across all meals
    total_calories_kcal: Optional[List[Optional[float]]] = None
    total_carbohydrates_g: Optional[List[Optional[float]]] = None
    total_cholesterol_mg: Optional[List[Optional[float]]] = None
    total_fiber_g: Optional[List[Optional[float]]] = None
    total_protein_g: Optional[List[Optional[float]]] = None
    total_saturated_fat_g: Optional[List[Optional[float]]] = None
    total_sodium_mg: Optional[List[Optional[float]]] = None
    total_sugar_g: Optional[List[Optional[float]]] = None
    total_fat_g: Optional[List[Optional[float]]] = None
    total_unsaturated_fat_g: Optional[List[Optional[float]]] = None

    # Structure constraints
    required_categories: Optional[List[Optional[str]]] = None
    min_meals: Optional[int] = None
    max_meals: Optional[int] = None
    unique_categories: Optional[bool] = None  # no duplicate meal types

    extra: Optional[str] = None


class MealPlanConstraints(BaseModel):
    meals: List[RecipeConstraints]
    global_constraints: GlobalConstraints


# =========================
# Sanitizers
# =========================

def sanitize_list(value):
    if not value:
        return []
    return [str(v) for v in value if v is not None]


def sanitize_nutrient(value):
    FRACTION = 0.2

    if value is None:
        return [None, None]

    # If model returns a single number
    if isinstance(value, (int, float)):
        v = float(value)
        return [v * (1 - FRACTION), v * (1 + FRACTION)]

    if isinstance(value, list):
        cleaned = []
        for v in value:
            try:
                cleaned.append(float(v) if v is not None else None)
            except (ValueError, TypeError):  # FIX 5: no bare except
                cleaned.append(None)

        # Case: single value like [300]
        if len(cleaned) == 1 and cleaned[0] is not None:
            v = cleaned[0]
            return [v * (1 - FRACTION), v * (1 + FRACTION)]

        # FIX 1: Case: equal values like [300, 300] — expand with ±20%
        if len(cleaned) >= 2 and cleaned[0] == cleaned[1] and cleaned[0] is not None:
            v = cleaned[0]
            return [v * (1 - FRACTION), v * (1 + FRACTION)]

        # Ensure exactly two values
        while len(cleaned) < 2:
            cleaned.append(None)

        return cleaned[:2]

    return [None, None]


def normalize_keys(data):
    key_map = {
        "utensils_in": "utensils_incl",
    }
    return {key_map.get(k, k): v for k, v in data.items()}


# =========================
# LLM Extraction
# =========================

def text_to_mealplan(text: str, seed=None) -> MealPlanConstraints:
    load_dotenv(dotenv_path="env.env")
    llm = ChatOllama(
        model="qwen3:14b",
        temperature=0,
        format="json",
        reasoning=False,
        base_url=os.getenv("RUNPOD_URL"),
        options={
            "seed": seed
        }
    )

    prompt = f"""
Extract the constraints from the question and output ONLY valid JSON. Response should only contain data from the question.

There are TWO levels:

1. Meal-level constraints (each meal separately)
2. Global constraints (apply across ALL meals)

Return JSON in this format:
{{
  "meals": [
    {{
      "category": ...,
      "cuisine": ...,
      "ingredients_incl": [...],
      "ingredients_excl": [...],
      "utensils_incl": [...],
      "utensils_excl": [...],
      "time_class": ...,
      "calories_kcal": [min, max],
      "carbohydrates_g": [min, max],
      "cholesterol_mg": [min, max],
      "fiber_g": [min, max],
      "protein_g": [min, max],
      "saturated_fat_g": [min, max],
      "sodium_mg": [min, max],
      "sugar_g": [min, max],
      "fat_g": [min, max],
      "unsaturated_fat_g": [min, max],
      "extra": ...
    }}
  ],
  "global_constraints": {{
    "total_calories_kcal": [min, max],
    "total_carbohydrates_g": [min, max],
    "total_cholesterol_mg": [min, max],
    "total_fiber_g": [min, max],
    "total_protein_g": [min, max],
    "total_saturated_fat_g": [min, max],
    "total_sodium_mg": [min, max],
    "total_sugar_g": [min, max],
    "total_fat_g": [min, max],
    "total_unsaturated_fat_g": [min, max],
    "required_categories": [...],
    "min_meals": int,
    "max_meals": int,
    "unique_categories": bool,
    "extra": ...
  }}
}}

Rules:
- Each meal = separate object
- DO NOT merge meals
- Do NOT assume missing information such as ingredients.
- If information is not present, use null.
- Lists must always be lists
- Nutritional values must be [min, max]. Min and max must be different numbers. If only a single value is given, return [value, value]. If user mentions "at least" or "more than", return [value, null]. If user mentions "at most" or "less than", return [0, value]. 
- Possible time_class values: very short, short, average, long, very long, null.
- Possible category values: Breakfast, Lunch, Dinner, Snack, Appetizer, Dessert, null
- Do not add any comments with # or //
- The extra field can ONLY contain short, structured constraints that do not fit any other key or null.

time_class mapping:
<=10 minutes → very short
11-25 minutes → short
26-45 minutes → average
46-90 minutes → long
>90 minutes → very long

Global rules:
- "total_*" applies to sum across meals
- Example: "breakfast + dinner" → required_categories
- Example: "3 meals" → min_meals = max_meals = 3

STRICT:
- ONLY JSON
- NO explanation

Query:
{text}
"""

    response = llm.invoke(prompt)
    print(response.content)

    try:
        data = json.loads(response.content)
    except json.JSONDecodeError:
        print("JSON parsing failed, using fallback")
        data = {"meals": [], "global_constraints": {}}

    meals_raw = data.get("meals", [])
    global_raw = data.get("global_constraints", {})

    # =========================
    # Process meals
    # =========================

    meals = []

    for item in meals_raw:
        item = normalize_keys(item)

        for key in ["ingredients_incl", "ingredients_excl", "utensils_incl", "utensils_excl"]:
            item[key] = sanitize_list(item.get(key))

        for key in [
            "calories_kcal", "carbohydrates_g", "cholesterol_mg", "fiber_g",
            "protein_g", "saturated_fat_g", "sodium_mg",
            "sugar_g", "fat_g", "unsaturated_fat_g"
        ]:
            item[key] = sanitize_nutrient(item.get(key))

        for key in ["category", "cuisine", "time_class"]:
            item.setdefault(key, None)

        item.setdefault("extra", None)

        meals.append(RecipeConstraints(**item))

    # =========================
    # Process global constraints
    # =========================

    # FIX 4: normalize_keys on global_raw too
    global_raw = normalize_keys(global_raw)

    for key in [
        "total_calories_kcal", "total_carbohydrates_g", "total_cholesterol_mg",
        "total_fiber_g", "total_protein_g", "total_saturated_fat_g",
        "total_sodium_mg", "total_sugar_g", "total_fat_g", "total_unsaturated_fat_g"
    ]:
        global_raw[key] = sanitize_nutrient(global_raw.get(key))

    global_raw["required_categories"] = sanitize_list(global_raw.get("required_categories"))
    global_raw.setdefault("min_meals", None)
    global_raw.setdefault("max_meals", None)
    global_raw.setdefault("unique_categories", None)
    global_raw.setdefault("extra", None)  # FIX 3: prevent ValidationError on missing extra

    global_constraints = GlobalConstraints(**global_raw)

    return MealPlanConstraints(
        meals=meals,
        global_constraints=global_constraints
    )


# =========================
# CLI
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert text to MealPlanConstraints")
    parser.add_argument("prompt", type=str)
    args = parser.parse_args()

    result = text_to_mealplan(args.prompt)

    print(json.dumps(result.model_dump(), indent=4))