import pandas as pd
import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics.collections import Faithfulness, AnswerRelevancy
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory
from openai import OpenAI  # Required for the factory client

def run_eval():
    # --- Load and Prep Data ---
    df = pd.read_csv("../rag_outputs.csv")
    df = df.rename(columns={"question": "user_input", "answer": "response"})
    df["contexts"] = df["contexts"].apply(json.loads)
    df["contexts"] = df["contexts"].apply(lambda ctx_list: [c["page_content"] for c in ctx_list])
    dataset = Dataset.from_pandas(df)

    # --- Initialize the OpenAI Client pointing to Ollama ---
    # Ollama's local server acts as an OpenAI-compatible proxy
    openai_client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama"  # Required string, though ignored by Ollama
    )

    # --- Initialize Models via Factory with the client ---
    llm = llm_factory(
        model='llama3.2', 
        client=openai_client
    )
    
    # Embedding factory often works better with the same pattern
    embeddings = embedding_factory(
        model='nomic-embed-text', 
        client=openai_client
    )

    # --- Initialize Metrics ---
    faithfulness = Faithfulness(llm=llm)
    answer_relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)

    # --- Run Evaluation ---
    # is_async=False is still necessary to prevent the Windows RLock error
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy],
        is_async=False 
    )
    return result

if __name__ == "__main__":
    results = run_eval()
    if results:
        print("\n--- Evaluation Results ---")
        print(results)