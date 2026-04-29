import json
import pandas as pd
import csv
import time
from datetime import datetime
from pathlib import Path
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

from components.indexer_ollama import OllamaIndexer
from components.retriever_langchain_hybrid import LangchainRetrieverHybrid
from components.generator_llama import LlamaGenerator
from core.rag_pipeline import RAGPipeline
from preprocessing.text_to_json import text_to_json


# ---------------------------------------------------------------------------
# 1. Evaluator LLM
# ---------------------------------------------------------------------------
eval_llm = ChatOllama(
    model="qwen3:4b",
    temperature=0,
    format="json",
    reasoning=False,
    base_url="http://203.57.40.79:10203"
)


def call_llm(prompt: str) -> dict:
    response = eval_llm.invoke([HumanMessage(content=prompt)])
    return json.loads(response.content)


# ---------------------------------------------------------------------------
# 2a. Pure-Python score computation (no LLM arithmetic)
# ---------------------------------------------------------------------------
def compute_scores(data: dict) -> dict:
    def ratio(items: list, key: str) -> float:
        if not items:
            return 1.0
        return sum(
            1 for x in items 
            if x.get(key) is True or x.get(key) == "true"
        ) / len(items)

    return {
        "faithfulness":            ratio(data.get("faithfulness", {}).get("claims", []), "supported"),
        "context_precision":       ratio(data.get("context_precision", {}).get("contexts", []), "relevant"),
        "constraint_satisfaction": ratio(data.get("constraint_satisfaction", {}).get("constraints", []), "satisfied"),
    }


# ---------------------------------------------------------------------------
# 2. Single-call evaluation — all 3 metrics in one prompt
# ---------------------------------------------------------------------------
def evaluate_row(
    question: str,
    answer: str,
    contexts: list[str],
    constraints: dict,
) -> dict:
    """
    Returns a dict with two levels:
      - 'scores': flat {metric: float} for CSV
      - 'detail': full breakdown {metric: {claims/contexts/constraints}} for JSON
    """
    context_block   = "\n\n".join(f"[Context {i+1}]: {c}" for i, c in enumerate(contexts))
    constraints_str = json.dumps(constraints, indent=2)

    prompt = f"""You are an evaluation assistant scoring a RAG system. Evaluate the following in one pass.

--- QUESTION ---
{question}

--- RETRIEVED CONTEXTS ---
{context_block}

--- ANSWER ---
{answer}

--- EXTRACTED CONSTRAINTS (from the question) ---
{constraints_str}

Perform these three evaluations:

1. FAITHFULNESS: List every factual claim in the answer. For each claim state whether it is
   supported by the retrieved contexts (true) or not (false).

2. CONTEXT_PRECISION: For each context, state whether it is relevant to answering the
   question (true/false).

3. CONSTRAINT_SATISFACTION: Using the extracted constraints, state whether the answer
   respects each constraint (true/false). Null values mean no constraint.

Respond ONLY with this JSON and nothing else:
{{
  "faithfulness": {{
    "claims": [{{"claim": "<text>", "supported": <true|false>}}]
  }},
  "context_precision": {{
    "contexts": [{{"relevant": <true|false>}}]
  }},
  "constraint_satisfaction": {{
    "constraints": [{{"satisfied": <true|false>}}]
  }}
}}"""

    try:
        data   = call_llm(prompt)
        scores = compute_scores(data)   # Python does the maths, not the LLM
        return {"scores": scores, "detail": data}
    except Exception as e:
        print(f"  [WARN] Score parsing failed: {e}")
        return {
            "scores": {
                "faithfulness":            None,
                "context_precision":       None,
                "constraint_satisfaction": None,
            },
            "detail": {},
        }

# ---------------------------------------------------------------------------
# CSV writer line-by-line
# ---------------------------------------------------------------------------
def write_csv_line(csv_path: Path, row: dict, header: list[str] = None):
    """Append a single row to CSV. Create file with header if not exists."""
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header or row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# JSON writer line-by-line
# ---------------------------------------------------------------------------
def write_json_line(json_path: Path, row: dict):
    """Append a single row to JSON file (newline-delimited) to avoid losing progress."""
    with open(json_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Evaluation function with timing, live aggregates, and line-by-line output
# ---------------------------------------------------------------------------
def run_evaluation(vectorstore, test_questions: list[dict], output_dir: str = "eval_results") -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # CSV path
    csv_path = output_dir / f"eval_scores_{timestamp}.csv"
    # JSON path (newline-delimited)
    json_path = output_dir / f"eval_detail_{timestamp}.jsonl"

    # Keep running aggregates
    aggregate = {
        "faithfulness": [],
        "context_precision": [],
        "constraint_satisfaction": [],
        "rag_response_time": [],
    }

    for idx, item in enumerate(test_questions, 1):
        prompt = item["question"]
        print(f"\nEvaluating [{idx}/{len(test_questions)}]: {prompt}")

        # text_to_json
        json_query = text_to_json(prompt)
        constraints = json_query.model_dump()

        # RAG pipeline
        retriever = LangchainRetrieverHybrid(vectorstore, json_query=json_query)

        start_time = time.time()
        answer, retrieved_docs = RAGPipeline(retriever, LlamaGenerator(stream = False)).run(prompt)
        rag_time = time.time() - start_time

        contexts = [doc.page_content for doc in retrieved_docs]

        # Evaluate
        result = evaluate_row(prompt, answer, contexts, constraints)

        # CSV row
        csv_row = {
            "question":                prompt,
            "answer":                  answer,
            "faithfulness":            result["scores"]["faithfulness"],
            "context_precision":       result["scores"]["context_precision"],
            "constraint_satisfaction": result["scores"]["constraint_satisfaction"],
            "rag_response_time":       rag_time,
        }

        # JSON row
        json_row = {
            "question":    prompt,
            "answer":      answer,
            "contexts":    contexts,
            "constraints": constraints,
            "scores":      result["scores"],
            "rag_response_time": rag_time,
            "detail":      result["detail"],
        }

        # Update aggregates
        for key in aggregate:
            val = csv_row.get(key)
            if val is not None:
                aggregate[key].append(val)

        # Compute running means
        running_mean = {k: sum(v)/len(v) if v else None for k, v in aggregate.items()}
        running_mean_prefixed = {
            f"{k}_mean": v for k, v in running_mean.items()
        }
        print(f"  Running mean metrics: {running_mean}")

        # Write immediately
        write_csv_line(csv_path, {**csv_row, **running_mean_prefixed}, header=list(csv_row.keys()) + list(running_mean_prefixed.keys()))
        write_json_line(json_path, {**json_row, "running_mean": running_mean})

    print(f"\nCSV saved line-by-line: {csv_path}")
    print(f"JSON (newline-delimited) saved: {json_path}")

    # Return final DataFrame
    df = pd.read_csv(csv_path)
    return df

def load_questions_from_txt(filepath: str) -> list[dict]:
    questions = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                questions.append({"question": line})
    return questions

# ---------------------------------------------------------------------------
# 5. Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_set = load_questions_from_txt("evaluation/testset.txt")

    indexer = OllamaIndexer(collection_name="recipes")
    vectorstore = indexer.get_vectorstore()

    results = run_evaluation(vectorstore, test_set, output_dir="eval_results")

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(results.to_string(index=False))

    print("\nMean scores:")
    print(results[["faithfulness", "context_precision", "constraint_satisfaction", "rag_response_time"]].mean())