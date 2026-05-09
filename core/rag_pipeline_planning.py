import time
from .base_retriever import BaseRetriever
from .base_generator import BaseGenerator
from evaluation.rag_logger import log_rag_result


class RAGPipeline:
    def __init__(self, retriever: BaseRetriever, generator: BaseGenerator):
        self.retriever = retriever
        self.generator = generator

    def filter_doc_metadata(self, doc):
        excluded_fields = {
            "pk",
            "name",
            "cuisine",
            "ingredients",
            "instructions",
            "dense",
            "sparse",
        }
        doc.metadata = {k: v for k, v in doc.metadata.items() if k not in excluded_fields}
        return doc

    def run(self, query: str, global_constraints=None):
        # --- Retrieval ---
        retrieval_start = time.perf_counter()

        meal_results = self.retriever.retrieve(query)
        for meal in meal_results:
            print(f"\n=== Meal {meal['meal_index'] + 1} | query: {meal['query']} ===")
            for doc in meal["results"]:
                print(" -", doc.metadata.get("name"), "|", doc.metadata.get("ingredients"))
            print("\n")

        filtered_meal_results = []
        for meal in meal_results:
            filtered_docs = [self.filter_doc_metadata(doc) for doc in meal["results"]]
            filtered_meal_results.append({**meal, "results": filtered_docs})

        retrieval_time = time.perf_counter() - retrieval_start

        # --- Generation ---
        generation_start = time.perf_counter()
        response = self.generator.generate(
            query,
            filtered_meal_results,
            with_retrieval=True,
            global_constraints=global_constraints,
        )
        generation_time = time.perf_counter() - generation_start

        all_docs = [doc for meal in filtered_meal_results for doc in meal["results"]]
        log_rag_result(
            question=query,
            answer=response,
            contexts=all_docs,
            ret_time=retrieval_time,
            gen_time=generation_time,
        )

        return response, filtered_meal_results