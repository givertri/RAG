from core.base_retriever import BaseRetriever

def build_milvus_filter(metadata_json):
    metadata_json = metadata_json.dict()
    conditions = []

    # categorical equality
    for key in ["category", "cuisine", "time_class"]:
        value = metadata_json.get(key)
        if value:
            conditions.append(f'{key} == "{value}"')

    # include ingredients (handled by semantic search)
    #for ingredient in metadata_json.get("ingredients_incl", []):
        #conditions.append(f'ARRAY_CONTAINS(ingredients, "{ingredient}")')

    # exclude ingredients
    for ingredient in metadata_json.get("ingredients_excl", []):
        conditions.append(f'not ARRAY_CONTAINS(ingredients, "{ingredient}")')

    # include utensils (handled by semantic search)
    #for utensil in metadata_json.get("utensils_incl", []):
        #conditions.append(f'ARRAY_CONTAINS(utensils, "{utensil}")')

    # exclude utensils
    for utensil in metadata_json.get("utensils_excl", []):
        conditions.append(f'not ARRAY_CONTAINS(utensils, "{utensil}")')

    # numeric ranges
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

    if conditions:
        return " and ".join(conditions)

    return None

class LangchainRetrieverHybrid(BaseRetriever):
    def __init__(self, vectorstore, k=5, json_query={}):
        self.vectorstore = vectorstore
        self.k = k
        self.json_query = json_query

    def retrieve(self, query: str):
        filter_expr = build_milvus_filter(self.json_query)
        print("MILVUS FILTER:", filter_expr)

        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": self.k,
                "filter": filter_expr,
                "ranker_type": "rrf",       # Hybrid search
                "ranker_params": {"k": 60}  # Hybrid search
            }
        )
        return retriever.invoke(query)

if __name__ == "__main__":
    example_json = {
        "category": "Dinner",
        "cuisine": "Italian",
        "time_class": "medium",

        "ingredients_incl": ["tomato", "basil", "garlic"],
        "ingredients_excl": ["nuts", "anchovy"],

        "utensils_incl": [],
        "utensils_excl": ["oven"],

        "calories_kcal": [200, 600],
        "carbohydrates_g": [20, 80],
        "cholesterol_mg": [None, 100],
        "fiber_g": [5, None],
        "protein_g": [10, 40],
        "saturated_fat_g": [None, 10],
        "sodium_mg": [None, 800],
        "sugar_g": [None, 20],
        "fat_g": [5, 25],
        "unsaturated_fat_g": [None, None]
    }

    filter_expr = build_milvus_filter(example_json)
    print(filter_expr)