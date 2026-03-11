from components.loader_simple import SimpleLoader
from components.loader_fromcsv import CsvLoader
from components.indexer_ollama import OllamaIndexer
from components.retriever_langchain import LangchainRetriever
from components.retriever_langchain_hybrid import LangchainRetrieverHybrid
from components.generator_llama import LlamaGenerator
from core.rag_pipeline import RAGPipeline
import argparse

def main(prompt=None, use_hybrid=False):
    if prompt is None:
        parser = argparse.ArgumentParser(description="Run RAG pipeline with a custom prompt.")
        parser.add_argument(
            "prompt",
            type=str,
            nargs="?",
            default="Give me a tasty dish.",
            help="The query/prompt to send to the RAG pipeline"
        )
        parser.add_argument(
            "--retriever",
            type=str,
            choices=["regular", "hybrid"],
            default="regular",
            help="Which retriever to use: 'regular' or 'hybrid'"
        )
        args = parser.parse_args()
        prompt = args.prompt
        use_hybrid = args.retriever == "hybrid"

    # Indexer
    indexer = OllamaIndexer(collection_name="recipes")
    vectorstore = indexer.get_vectorstore()

    metadata_json = {
        "category": "Appetizer",
        "ingredients_incl": ["cheese"],
        "time_class": "short",
        "cuisine": "Mexican",
        "calories_kcal": [None, None],
        "protein_g": [None, None]
    }

    # Create retriever based on CLI argument
    if use_hybrid:
        retriever = LangchainRetrieverHybrid(vectorstore, metadata_json)
    else:
        retriever = LangchainRetriever(vectorstore)

    # Generator
    generator = LlamaGenerator()

    # RAG pipeline
    rag = RAGPipeline(retriever, generator)

    # Run prompt
    rag.run(prompt)

# For batch runs
def main_override(prompt, use_hybrid=False):
    main(prompt, use_hybrid)

if __name__ == "__main__":
    main()