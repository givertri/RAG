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
        "faithfulness": ratio(
            data.get("faithfulness", {}).get("claims", []),
            "supported"
        ),
        "context_precision": ratio(
            data.get("context_precision", {}).get("contexts", []),
            "relevant"
        ),
        "constraint_satisfaction": ratio(
            data.get("constraint_satisfaction", {}).get("constraints", []),
            "satisfied"
        ),
    }


# ---------------------------------------------------------------------------
# 3. Evaluation prompt
# ---------------------------------------------------------------------------
def evaluate_row(question, answer, contexts, constraints):
    context_block = "\n\n".join(
        f"[Context {i+1}]: {c['content']}"
        for i, c in enumerate(contexts)
    )

    constraints_str = json.dumps(constraints, indent=2)

    prompt = f"""
You are an evaluation assistant scoring a constraint-aware RAG system.

--- QUESTION ---
{question}

--- RETRIEVED CONTEXTS ---
{context_block}

--- ANSWER ---
{answer}

--- EXTRACTED CONSTRAINTS ---
{constraints_str}

Evaluate the following metrics:

1. FAITHFULNESS:
For each claim in the answer, determine whether it is fully supported by the retrieved contexts. (true/false)

2. CONTEXT_PRECISION:
For each retrieved context, determine whether it is relevant to the user question. (true/false)

3. CONSTRAINT_SATISFACTION:
For each extracted constraint, determine whether the answer respects the constraint. (true/false)

Return ONLY valid JSON in this schema:

{{
    "faithfulness": {{
        "claims": [
            {{
                "claim": "",
                "supported": true
            }}
        ]
    }},
    "context_precision": {{
        "contexts": [
            {{
                "relevant": true
            }}
        ]
    }},
    "constraint_satisfaction": {{
        "constraints": [
            {{
                "satisfied": true
            }}
        ]
    }}
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
# 4. Constraint-aware Hybrid RAG runner
# ---------------------------------------------------------------------------
def run_constraint_aware_rag(prompt, vectorstore, json_query):
    retriever = LangchainRetrieverHybrid(
        vectorstore,
        json_query=json_query
    )

    answer, docs = RAGPipeline(
        retriever,
        LlamaGenerator(stream=False, seed=SEED)
    ).run(prompt)

    return answer, docs


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------
def sanitize_metadata(meta):
    """Recursively convert objects into JSON-serializable structures."""

    if meta is None or isinstance(meta, (str, int, float, bool)):
        return meta

    if isinstance(meta, dict):
        return {
            str(k): sanitize_metadata(v)
            for k, v in meta.items()
        }

    if isinstance(meta, (list, tuple)) or (
        hasattr(meta, "__iter__")
        and not isinstance(meta, (str, bytes))
    ):
        return [sanitize_metadata(v) for v in meta]

    return str(meta)


CSV_FIELDS = [
    "question",
    "answer",
    "response_time",
    "faithfulness",
    "context_precision",
    "constraint_satisfaction",
    "faithfulness_mean",
    "context_precision_mean",
    "constraint_satisfaction_mean",
    "response_time_mean",
]


def write_csv_line(path, row):
    exists = path.exists()

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=CSV_FIELDS,
            quoting=csv.QUOTE_ALL
        )

        if not exists:
            writer.writeheader()

        writer.writerow(row)


CSV_FIELDS_REDUCED = [
    "response_time",
    "faithfulness",
    "context_precision",
    "constraint_satisfaction",
]


def write_csv_line_reduced(path, row):
    exists = path.exists()

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=CSV_FIELDS_REDUCED,
            quoting=csv.QUOTE_ALL
        )

        if not exists:
            writer.writeheader()

        filtered_row = {
            k: row.get(k)
            for k in CSV_FIELDS_REDUCED
        }

        writer.writerow(filtered_row)


def write_json_line(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# 5. MAIN evaluation
# ---------------------------------------------------------------------------
def run_evaluation(
    vectorstore,
    test_questions,
    output_dir="eval_results",
    start_index=0
):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = output_dir / f"eval_{timestamp}.csv"
    csv_reduced_path = output_dir / f"eval_{timestamp}_reduced.csv"
    json_path = output_dir / f"eval_{timestamp}.jsonl"

    aggregate = {
        "faithfulness": [],
        "context_precision": [],
        "constraint_satisfaction": [],
        "response_time": []
    }

    for idx, item in enumerate(
        test_questions[start_index:],
        start=start_index + 1
    ):
        question = item["question"]

        print(f"\n[{idx}/{len(test_questions)}] {question}")

        json_query = text_to_json(question, seed=SEED)
        constraints = json_query.model_dump()

        start = time.time()

        answer, docs = run_constraint_aware_rag(
            question,
            vectorstore,
            json_query
        )

        resp_time = time.time() - start

        contexts = [
            {
                "content": d.page_content,
                "metadata": d.metadata
            }
            for d in docs
        ]

        result = evaluate_row(
            question,
            answer,
            contexts,
            constraints
        )

        scores = result["scores"]

        row = {
            "question": question,
            "answer": answer,
            "response_time": resp_time,
            "faithfulness": scores.get("faithfulness"),
            "context_precision": scores.get("context_precision"),
            "constraint_satisfaction": scores.get(
                "constraint_satisfaction"
            ),
        }

        # update aggregates
        for k in aggregate:
            if row.get(k) is not None:
                aggregate[k].append(row[k])

        # running means
        means = {}

        for k, values in aggregate.items():
            valid = [v for v in values if v is not None]

            means[f"{k}_mean"] = (
                sum(valid) / len(valid)
                if valid else None
            )

        print(f"     running means: {means}")

        # write outputs
        write_csv_line(csv_path, {**row, **means})

        write_csv_line_reduced(
            csv_reduced_path,
            row
        )

        write_json_line(
            json_path,
            sanitize_metadata({
                **row,
                "contexts": contexts,
                "constraints": constraints,
                "scores": scores,
                "running_mean": means,
                "detail": result["detail"]
            })
        )

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
                questions.append({
                    "question": line
                })

    return questions


# ---------------------------------------------------------------------------
# 7. Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    test_set = load_questions_from_txt("evaluation/test_sets/testset_negative.txt")
    #test_set = load_questions_from_txt("evaluation/test_sets/testset_numeric.txt")
    #test_set = load_questions_from_txt("evaluation/test_sets/testset_highly_constrained.txt")

    indexer = OllamaIndexer(
        collection_name="recipes"
    )

    vectorstore = indexer.get_vectorstore()

    df = run_evaluation(
        vectorstore,
        test_set
    )

    print("\nFINAL RESULTS:")
    print(df.mean(numeric_only=True))