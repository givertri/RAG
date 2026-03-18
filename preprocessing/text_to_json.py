import json
from langchain_ollama import ChatOllama
from typing import List, Optional
from pydantic import BaseModel
import argparse

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

def sanitize_list_field(value):
    if value is None:
        return []
    return value

def sanitize_nutrient_field(value):

    fraction = 0.2

    if value is None:
        return [None, None]

    # If model returns a single number
    if isinstance(value, (int, float)):
        v = float(value)
        delta = v * fraction
        return [v - delta, v + delta]

    if isinstance(value, list):
        sanitized = []
        for v in value:
            try:
                sanitized.append(float(v) if v is not None else None)
            except (ValueError, TypeError):
                sanitized.append(None)

        # Case: single value like [300]
        if len(sanitized) == 1 and sanitized[0] is not None:
            v = sanitized[0]
            delta = v * fraction
            return [v - delta, v + delta]

        # Case: [300, 300]
        if len(sanitized) >= 2 and sanitized[0] == sanitized[1] and sanitized[0] is not None:
            v = sanitized[0]
            delta = v * fraction
            return [v - delta, v + delta]

        # Ensure exactly two values
        while len(sanitized) < 2:
            sanitized.append(None)

        return sanitized[:2]

    return [None, None]

def normalize_keys(data):
    key_map = {
        "utensils_in": "utensils_incl"
    }

    normalized = {}

    for key, value in data.items():
        corrected_key = key_map.get(key, key)
        normalized[corrected_key] = value

    return normalized

def text_to_json(text):
    llm = ChatOllama(
        model="qwen3:4b",
        temperature=0,
        format='json',
        reasoning=False
    )

    prompt = f"""
    Extract the constraints from the question and output ONLY valid JSON. Response should only contain data from the question.

    Allowed keys (all should be present):
    category, cuisine, ingredients_incl, ingredients_excl,
    utensils_incl, utensils_excl, time_class,
    calories_kcal, carbohydrates_g, cholesterol_mg,
    fiber_g, protein_g, saturated_fat_g, sodium_mg,
    sugar_g, fat_g, unsaturated_fat_g.

    Rules:
    - Do NOT assume missing information such as ingredients.
    - If information is not present, use null.
    - Nutritional values must be [min, max]. Min and max must be different numbers. If only a single value is given, return [value, value]. If user mentions "at least", return [value, null]. If user mentions "at most", return [0, value]. 
    - Possible time_class values: very short, short, average, long, very long, null
    - Possible category values: Breakfast, Lunch, Dinner, Snack, Appetizer, Dessert, null

    Question: {text}

    JSON output:
    """

    response = llm.invoke(prompt)
    print(response.content)
    try:
        data = json.loads(response.content)
        data = normalize_keys(data)
        #print(data)
    except json.JSONDecodeError:
        data = {} # fallback to empty constraints

    for key in ["ingredients_incl", "ingredients_excl", "utensils_incl", "utensils_excl"]:
        data[key] = sanitize_list_field(data.get(key))

    nutrient_keys = [
        "calories_kcal", "carbohydrates_g", "cholesterol_mg", "fiber_g",
        "protein_g", "saturated_fat_g", "sodium_mg", "sugar_g", "fat_g", "unsaturated_fat_g"
    ]
    for key in nutrient_keys:
        data[key] = sanitize_nutrient_field(data.get(key))
    
    for key in ["category", "cuisine", "time_class"]:
        if key not in data:
            data[key] = None

    return RecipeConstraints(**data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert text prompt to RecipeConstraints JSON")
    parser.add_argument("prompt", type=str, help="Text prompt describing recipe constraints")
    args = parser.parse_args()

    result = text_to_json(args.prompt)
    print(json.dumps(vars(result), indent=4))