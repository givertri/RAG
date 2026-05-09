from components.indexer_ollama import OllamaIndexer
from components.retriever_langchain_hybrid_planning import LangchainRetrieverHybrid
from components.generator_llama_planning import LlamaGenerator
from core.rag_pipeline_planning import RAGPipeline
from preprocessing.text_to_json_planning import text_to_mealplan
import argparse

def main(prompt=None):
    if prompt is None:
        parser = argparse.ArgumentParser(description="Run RAG pipeline with a custom prompt.")
        parser.add_argument(
            "prompt",
            type=str,
            nargs="?",
            default="Give me a tasty dish.",
            help="The query/prompt to send to the RAG pipeline"
        )
        args = parser.parse_args()
        prompt = args.prompt

    # Indexer
    indexer = OllamaIndexer(collection_name="recipes")
    vectorstore = indexer.get_vectorstore()

    # Create retriever based on CLI argument
    json_query = text_to_mealplan(prompt)

    num_meals = len(json_query.meals)
    print("Number of meals:", num_meals)

    retriever = LangchainRetrieverHybrid(vectorstore, json_query=json_query.meals)

    # Generator
    generator = LlamaGenerator()

    # RAG pipeline
    rag = RAGPipeline(retriever, generator)

    # Run prompt
    response, filtered_meal_results = rag.run(
        prompt,
        global_constraints=json_query.global_constraints,
    )
    return response, filtered_meal_results


# For batch runs
def main_override(prompt):
    main(prompt)

if __name__ == "__main__":
    main()