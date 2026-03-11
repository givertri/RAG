from core.base_retriever import BaseRetriever

def build_milvus_filter(metadata_json):
    filter_expr = []

    # Categorical equality filters
    for key in ["category", "cuisine", "time_class"]:
        value = metadata_json.get(key)
        if value:
            filter_expr.append({key: value})

    # Include ingredients
    ingredients_incl = metadata_json.get("ingredients_incl", [])
    for ingredient in ingredients_incl:
        filter_expr.append({"ingredients": {"$contains": ingredient}})

    # Exclude ingredients
    ingredients_excl = metadata_json.get("ingredients_excl", [])
    for ingredient in ingredients_excl:
        filter_expr.append({"ingredients": {"$not_contains": ingredient}})

    # Numeric ranges
    numeric_fields = [
        "calories_kcal", "carbohydrates_g", "cholesterol_mg", "fiber_g",
        "protein_g", "saturated_fat_g", "sodium_mg", "sugar_g",
        "fat_g", "unsaturated_fat_g"
    ]
    for field in numeric_fields:
        range = metadata_json.get(field)
        if range and any(range):
            start, end = range
            expr = {}
            if start is not None:
                expr["$gte"] = start
            if end is not None:
                expr["$lte"] = end
            filter_expr.append({field: expr})

    # Combine all with $and
    if len(filter_expr) == 1:
        return filter_expr[0]
    elif filter_expr:
        return {"$and": filter_expr}
    else:
        return {}

class LangchainRetrieverHybrid(BaseRetriever):
    def __init__(self, vectorstore, k=5):
        self.vectorstore = vectorstore
        self.k = k

    def retrieve(self, query: str, metadata_json: dict = None):
        filter_expr = build_milvus_filter(metadata_json or {})

        retriever = self.vectorstore.as_retriever(
            search_type="hybrid",
            search_kwargs={
                "k": self.k,
                "filter": filter_expr
            }
        )
        return retriever.invoke(query)