from core.base_retriever import BaseRetriever

def build_milvus_filter(metadata_json):
    if hasattr(metadata_json, "dict"):
        metadata_json = metadata_json.dict()

    conditions = []

    def is_valid(val):
        """Checks if a value is actually usable and not a 'null' string."""
        if val is None:
            return False
        if isinstance(val, str) and val.lower() == "null":
            return False
        return True

    # 1. Array inclusion fields (cuisine, category)
    for key in ["cuisine", "category"]:
        values = metadata_json.get(key)
        if values:
            if isinstance(values, str):
                values = [values]
            # Filter out "null" strings from the list
            valid_values = [v for v in values if is_valid(v)]
            for v in valid_values:
                conditions.append(f'ARRAY_CONTAINS({key}, "{v.lower()}")')

    # 2. Plain scalar field (time_class)
    time_class = metadata_json.get("time_class")
    if is_valid(time_class):
        conditions.append(f'time_class == "{time_class}"')

    # 3. Exclusion array fields
    for key, field_name in [("ingredients_excl", "ingredients"), ("utensils_excl", "utensils")]:
        items = metadata_json.get(key, [])
        if isinstance(items, list):
            for item in items:
                if is_valid(item):
                    conditions.append(f'not ARRAY_CONTAINS({field_name}, "{item}")')

    # 4. Numeric range fields
    numeric_fields = [
        "calories_kcal", "carbohydrates_g", "cholesterol_mg", "fiber_g",
        "protein_g", "saturated_fat_g", "sodium_mg", "sugar_g",
        "fat_g", "unsaturated_fat_g"
    ]

    for field in numeric_fields:
        value_range = metadata_json.get(field)
        # Ensure it's a list/tuple of length 2 and not a "null" string
        if isinstance(value_range, (list, tuple)) and len(value_range) == 2:
            start, end = value_range
            if is_valid(start):
                conditions.append(f"{field} >= {start}")
            if is_valid(end):
                conditions.append(f"{field} <= {end}")

    return " and ".join(conditions) if conditions else None

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
                "fetch_k": 50,
                "filter": filter_expr,
                "ranker_type": "rrf",       # Hybrid search
                "ranker_params": {"k": 20}  # Hybrid search
            }
        )
        results = retriever.invoke(query)
        
        if not results:
            print("No documents found with filter. Retrying without filter...")
            retriever_no_filter = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": self.k,
                    "fetch_k": 50,
                    "ranker_type": "rrf",
                    "ranker_params": {"k": 20}
                }
            )
            results = retriever_no_filter.invoke(query)

        return results

if __name__ == "__main__":
    example_json = {
        "category": "Dinner",
        "cuisine": "Italian",
        "time_class": "average",

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