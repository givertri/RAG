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
from components.retriever_langchain import LangchainRetriever
from components.generator_llama import LlamaGenerator
from core.rag_pipeline import RAGPipeline
from preprocessing.text_to_json import text_to_json
from dotenv import load_dotenv
import os

SEEDS = [42, 43, 44]
SEED = SEEDS[2]

# ---------------------------------------------------------------------------
# 1. Evaluator LLM
# ---------------------------------------------------------------------------
load_dotenv(dotenv_path="env.env")

eval_llm = ChatOllama(
    model="qwen3:14b",
    temperature=0,
    format="json",
    reasoning=False,
    base_url=os.getenv("RUNPOD_URL"),
    options={
        "seed": SEED
    }
)


def call_llm(prompt: str) -> dict:
    response = eval_llm.invoke([HumanMessage(content=prompt)])
    print(response.content)
    return json.loads(response.content)


# ---------------------------------------------------------------------------
# 2. Score computation
# ---------------------------------------------------------------------------
def compute_scores(data: dict) -> dict:
    def ratio(items: list, key: str) -> float:
        if not items:
            return None
        return sum(1 for x in items if x.get(key) is True) / len(items)

    return {
        "faithfulness": ratio(data.get("faithfulness", {}).get("claims", []), "supported"),
        "context_precision": ratio(data.get("context_precision", {}).get("contexts", []), "relevant"),
        "constraint_satisfaction": ratio(data.get("constraint_satisfaction", {}).get("constraints", []), "satisfied"),
        "hallucination_rate": ratio(data.get("hallucination", {}).get("claims", []), "hallucinated"),
    }


# ---------------------------------------------------------------------------
# 3. Evaluation prompt (single unified schema)
# ---------------------------------------------------------------------------
def evaluate_row(question, answer, contexts, constraints, system):
    context_block = "\n\n".join(f"[Context {i+1}]: {c}" for i, c in enumerate(contexts))
    constraints_str = json.dumps(constraints, indent=2)

    eval_tasks = []

    if system != "llm":
        eval_tasks.append("""
1. FAITHFULNESS: For each claim in the answer, is it supported by the retrieved contexts? Numerical/unit conversions are considered faithful if and only if they are mathematically correct given a value in the retrieved context. (true/false)
2. CONTEXT_PRECISION: For each context document, state whether it is relevant to the question. (true/false)
""")

    if system == "llm":
        eval_tasks.append("""
3. CONSTRAINT_SATISFACTION: For each extracted constraint, state whether the answer respects the constraint. (true/false)
4. HALLUCINATION: For each claim in the answer, state whether the claim is unsupported, unverifiable, speculative, internally inconsistent, or likely fabricated. (true/false)
A claim should be marked hallucinated=true if:
- it cannot be verified from reliable culinary/nutritional knowledge,
- it invents nutritional values, cooking properties, cuisines, or ingredients,
- it asserts unsupported health claims,
- it contradicts known ingredient properties,
- it overstates certainty,
- or the statement is plausible-sounding but unverifiable.

When uncertain, prefer hallucinated=true.
""")
    else:
        eval_tasks.append("""
3. CONSTRAINT_SATISFACTION: For each extracted constraint, state whether the answer respects the constraint. (true/false)
""")

    prompt = f"""
You are an evaluation assistant scoring a QA / RAG system.

--- QUESTION ---
{question}

--- RETRIEVED CONTEXTS ---
{context_block}

--- ANSWER ---
{answer}

--- EXTRACTED CONSTRAINTS ---
{constraints_str}

Evaluate ONLY the requested metrics:

{''.join(eval_tasks)}

Respond ONLY JSON with relevant fields (empty list if not relevant):
{{
    "faithfulness": {{"claims": [{{"claim": "", "supported": true}}]}}, 
    "context_precision": {{"contexts": [{{"relevant": true}}]}}, 
    "constraint_satisfaction": {{"constraints": [{{"satisfied": true}}]}}, 
    "hallucination": {{"claims": [{{"claim": "", "hallucinated": false}}]}}
}}
"""

    try:
        data = call_llm(prompt)
        scores = compute_scores(data)
        return {"scores": scores, "detail": data}
    except Exception as e:
        print(f"[WARN] Eval failed: {e}")
        return {"scores": {}, "detail": {}}


# ---------------------------------------------------------------------------
# 4. System runners
# ---------------------------------------------------------------------------
def run_llm_only(prompt: str):
    llm = LlamaGenerator(stream=False, seed=SEED)
    answer = llm.generate(prompt, with_retrieval=False, docs=[])
    return answer, []


def run_rag_hybrid(prompt, vectorstore, json_query):
    retriever = LangchainRetrieverHybrid(vectorstore, json_query=json_query)
    answer, docs = RAGPipeline(retriever, LlamaGenerator(stream=False)).run(prompt)
    return answer, docs


def run_rag_standard(prompt, vectorstore):
    retriever = LangchainRetriever(vectorstore)
    answer, docs = RAGPipeline(retriever, LlamaGenerator(stream=False)).run(prompt)
    return answer, docs


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------
def sanitize_metadata(meta):
    """Recursively convert objects into JSON-serializable structures."""

    # primitives
    if meta is None or isinstance(meta, (str, int, float, bool)):
        return meta

    # dicts
    if isinstance(meta, dict):
        return {str(k): sanitize_metadata(v) for k, v in meta.items()}

    # lists / tuples / protobuf repeated containers
    if isinstance(meta, (list, tuple)) or (
        hasattr(meta, "__iter__") and not isinstance(meta, (str, bytes))
    ):
        return [sanitize_metadata(v) for v in meta]

    # fallback
    return str(meta)

CSV_FIELDS = [
    "system",
    "question",
    "answer",
    "response_time",
    "faithfulness",
    "context_precision",
    "constraint_satisfaction",
    "hallucination_rate",
    "faithfulness_mean",
    "context_precision_mean",
    "constraint_satisfaction_mean",
    "hallucination_rate_mean",
    "response_time_mean",
]

def write_csv_line(path, row, header=None):
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, quoting=csv.QUOTE_ALL)
        if not exists:
            writer.writeheader()
        writer.writerow(row)

CSV_FIELDS_REDUCED = [
    "system",
    "response_time",
    "faithfulness",
    "context_precision",
    "constraint_satisfaction",
    "hallucination_rate",
]

def write_csv_line_reduced(path, row):
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS_REDUCED, quoting=csv.QUOTE_ALL)
        if not exists:
            writer.writeheader()

        # filter only required keys
        filtered_row = {k: row.get(k) for k in CSV_FIELDS_REDUCED}
        writer.writerow(filtered_row)

def write_json_line(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# 5. MAIN evaluation
# ---------------------------------------------------------------------------
def run_evaluation(vectorstore, test_questions, output_dir="eval_results", start_index=0):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"eval_{timestamp}.csv"
    csv_reduced_path = output_dir / f"eval_{timestamp}_reduced.csv"
    json_path = output_dir / f"eval_{timestamp}.jsonl"

    systems = ["llm", "rag_hybrid", "rag_standard"]

    # aggregates per system
    aggregate = {
        s: {
            "faithfulness": [],
            "context_precision": [],
            "constraint_satisfaction": [],
            "hallucination_rate": [],
            "response_time": []
        } for s in systems
    }

    for idx, item in enumerate(test_questions[start_index:], start=start_index + 1):
        question = item["question"]
        print(f"\n[{idx}/{len(test_questions)}] {question}")

        json_query = text_to_json(question, seed=SEED)
        constraints = json_query.model_dump()

        for system in systems:
            print(f"  -> {system}")

            start = time.time()

            if system == "llm":
                answer, docs = run_llm_only(question)

            elif system == "rag_hybrid":
                answer, docs = run_rag_hybrid(question, vectorstore, json_query)

            else:
                answer, docs = run_rag_standard(question, vectorstore)

            resp_time = time.time() - start
            contexts = [{"content": d.page_content, "metadata": d.metadata} for d in docs]

            result = evaluate_row(question, answer, contexts, constraints, system)
            scores = result["scores"]

            # build row
            row = {
                "system": system,
                "question": question,
                "answer": answer,
                "response_time": resp_time,
            }

            row["faithfulness"] = None
            row["context_precision"] = None
            row["constraint_satisfaction"] = None
            row["hallucination_rate"] = None

            # apply metric filtering
            if system == "llm":
                row["constraint_satisfaction"] = scores.get("constraint_satisfaction")
                row["hallucination_rate"] = scores.get("hallucination_rate")
            else:
                row["faithfulness"] = scores.get("faithfulness")
                row["context_precision"] = scores.get("context_precision")
                row["constraint_satisfaction"] = scores.get("constraint_satisfaction")

            # update aggregates
            for k in aggregate[system]:
                if k in row and row[k] is not None:
                    aggregate[system][k].append(row[k])

            # compute running means
            means = {}
            for k, values in aggregate[system].items():
                valid = [v for v in values if v is not None]
                if len(valid) == 0:
                    means[f"{k}_mean"] = None
                else:
                    means[f"{k}_mean"] = sum(valid) / len(valid)

            print(f"     running means: {means}")

            # write outputs
            write_csv_line(csv_path, {**row, **means})
            write_csv_line_reduced(csv_reduced_path, row)
            write_json_line(json_path, sanitize_metadata({
                **row,
                "contexts": contexts,
                "constraints": constraints,
                "scores": scores,
                "running_mean": means,
                "detail": result["detail"]
            }))

    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved JSONL: {json_path}")

    return pd.read_csv(csv_path, engine="python")


# ---------------------------------------------------------------------------
# 6. Load questions
# ---------------------------------------------------------------------------
def load_questions_from_txt(filepath):
    questions = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                questions.append({"question": line})
    return questions


# ---------------------------------------------------------------------------
# 7. Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_set = load_questions_from_txt("evaluation/test_sets/testset.txt")

    indexer = OllamaIndexer(collection_name="recipes")
    vectorstore = indexer.get_vectorstore()

    df = run_evaluation(vectorstore, test_set)

    print("\nFINAL RESULTS:")
    print(df.groupby("system").mean(numeric_only=True))