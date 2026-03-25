from pymilvus import connections, Collection

def test_filter(collection_name: str, filter_expr: str, limit: int = 10):
    # Connect to Milvus
    connections.connect(alias="default", host="localhost", port="19530")

    # Load collection
    collection = Collection(collection_name)
    collection.load()

    print("Running filter:")
    print(filter_expr)

    # Run pure scalar query (NO vector search)
    results = collection.query(
        expr=filter_expr or "",
        limit=limit,
        output_fields=["*"]  # or specify fields
    )

    print(f"\nReturned {len(results)} results:\n")
    for i, r in enumerate(results):
        print(f"{i+1}. {r}")

    return results

def build_milvus_filter(metadata_json):
    if hasattr(metadata_json, "dict"):
        metadata_json = metadata_json.dict()

    conditions = []

     # Array inclusion fields (cuisine, category)
    for key in ["cuisine", "category"]:
        values = metadata_json.get(key)
        if values:
            if isinstance(values, str):
                values = [values]
            for v in values:
                conditions.append(f'ARRAY_CONTAINS({key}, "{v.lower()}")')

    # Plain scalar field
    time_class = metadata_json.get("time_class")
    if time_class:
        conditions.append(f'time_class == "{time_class}"')

    # Exclusion array fields
    for ingredient in metadata_json.get("ingredients_excl", []):
        conditions.append(f'not ARRAY_CONTAINS(ingredients, "{ingredient}")')
    for utensil in metadata_json.get("utensils_excl", []):
        conditions.append(f'not ARRAY_CONTAINS(utensils, "{utensil}")')

    # Numeric range fields
    numeric_fields = [
        "calories_kcal", "carbohydrates_g", "cholesterol_mg", "fiber_g",
        "protein_g", "saturated_fat_g", "sodium_mg", "sugar_g",
        "fat_g", "unsaturated_fat_g"
    ]

    for field in numeric_fields:
        value_range = metadata_json.get(field)
        if value_range:
            start, end = value_range

            if start is not None:
                conditions.append(f"{field} >= {start}")
            if end is not None:
                conditions.append(f"{field} <= {end}")

    return " and ".join(conditions) if conditions else None

if __name__ == "__main__":
    example_json = {
        "category": "Dinner",
        "cuisine": "Italian",
        "time_class": "average",
        "ingredients_excl": ["red meat"],
        "utensils_excl": ["grill"],
        "calories_kcal": [700, None],
        "protein_g": [10, None],
        "fiber_g": [5, None]
    }

    example_json = {
        "cuisine": "Belgian",
        "calories_kcal": [700, None]
    }

    filter_expr = build_milvus_filter(example_json)

    test_filter(
        collection_name="recipes",
        filter_expr=filter_expr,
        limit=5
    )