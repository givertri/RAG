from core.base_retriever import BaseRetriever
import re
from preprocessing.split_query import split_query

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
    def __init__(self, vectorstore, k=5, json_query=None):
        self.vectorstore = vectorstore
        self.k = k
        self.json_query = json_query or []

    def _retrieve_single(self, query: str, filter_expr: str):
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": self.k,
                "fetch_k": 50,
                "filter": filter_expr,
                "ranker_type": "rrf",
                "ranker_params": {"k": 20}
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

    def retrieve(self, query: str):
        # Ensure json_query is a list
        if isinstance(self.json_query, dict):
            json_queries = [self.json_query]
        else:
            json_queries = self.json_query

        n_meals = len(json_queries)

        # Split query into subqueries
        subqueries = split_query(query, n_meals)

        all_results = []

        for i, (meal_json, subquery) in enumerate(zip(json_queries, subqueries)):
            print(f"\n--- Retrieving meal {i+1} ---")
            print("Subquery:", subquery)

            filter_expr = build_milvus_filter(meal_json)
            print("MILVUS FILTER:", filter_expr)

            results = self._retrieve_single(subquery, filter_expr)

            all_results.append({
                "meal_index": i,
                "query": subquery,
                "filter": filter_expr,
                "results": results
            })

        return all_results

if __name__ == "__main__":
    json_query = [
        {
            "category": "Dinner",
            "cuisine": "Italian",
            "ingredients_excl": ["nuts"]
        },
        {
            "category": "Breakfast",
            "protein_g": [20, None]
        }
        ]

    query = "Italian dinner with tomato and basil and a high protein breakfast"

    #retriever = LangchainRetrieverHybrid(vectorstore, k=5, json_query=json_query)
    #results = retriever.retrieve(query)
    #print(results)