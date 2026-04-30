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


def text_to_json(text) -> List[RecipeConstraints]:
    llm = ChatOllama(
        model="qwen3:4b",
        temperature=0,
        format="json",
        reasoning=False,
        base_url="http://203.57.40.79:10203"
    )

    prompt = f"""
Extract recipe constraints from the user query.

The query may contain MULTIPLE meals. Return a JSON LIST where each item represents ONE meal.

Allowed keys (ALL must be present in each object):
category, cuisine, ingredients_incl, ingredients_excl,
utensils_incl, utensils_excl, time_class,
calories_kcal, carbohydrates_g, cholesterol_mg,
fiber_g, protein_g, saturated_fat_g, sodium_mg,
sugar_g, fat_g, unsaturated_fat_g, extra.

Rules:
- Each object = ONE meal
- Do NOT merge multiple meals into one object
- If only ONE meal → return a list with ONE object
- Do NOT assume missing info
- Missing fields → null

Formatting rules:
- Ingredients & utensils → lists
- Nutrients → [min, max]
- If single value → [value, value]
- "at least" → [value, null]
- "at most" → [0, value]

Time mapping:
<=10 → very short
11-25 → short
26-45 → average
46-90 → long
>90 → very long

Category values:
Breakfast, Lunch, Dinner, Snack, Appetizer, Dessert, null

STRICT OUTPUT:
- ONLY valid JSON
- MUST be a list
- NO explanations

Query:
{text}
"""

    response = llm.invoke(prompt)
    print(response.content)

    try:
        raw_data = json.loads(response.content)

        # Ensure list
        if isinstance(raw_data, dict):
            raw_data = [raw_data]

    except json.JSONDecodeError:
        print("JSON parse failed, fallback to single empty constraint")
        raw_data = [{}]

    results = []

    for item in raw_data:
        item = normalize_keys(item)

        # ---- sanitize lists ----
        for key in ["ingredients_incl", "ingredients_excl", "utensils_incl", "utensils_excl"]:
            item[key] = sanitize_list_field(item.get(key))

        # ---- sanitize nutrients ----
        nutrient_keys = [
            "calories_kcal", "carbohydrates_g", "cholesterol_mg", "fiber_g",
            "protein_g", "saturated_fat_g", "sodium_mg", "sugar_g",
            "fat_g", "unsaturated_fat_g"
        ]
        for key in nutrient_keys:
            item[key] = sanitize_nutrient_field(item.get(key))

        # ---- ensure scalar fields ----
        for key in ["category", "cuisine", "time_class"]:
            if key not in item:
                item[key] = None

        results.append(RecipeConstraints(**item))

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert text prompt to RecipeConstraints JSON")
    parser.add_argument("prompt", type=str, help="Text prompt describing recipe constraints")
    args = parser.parse_args()

    result = text_to_json(args.prompt)
    print(json.dumps(vars(result), indent=4))