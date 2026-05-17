import json
import pandas as pd
import csv
import time
from datetime import datetime
from pathlib import Path

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

from components.indexer_ollama import OllamaIndexer
from components.retriever_langchain_hybrid_planning import LangchainRetrieverHybrid
from components.generator_llama_planning import LlamaGenerator
from core.rag_pipeline_planning import RAGPipeline
from preprocessing.text_to_json_planning import text_to_mealplan, MealPlanConstraints
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
    options={"seed": SEED}
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
        "faithfulness":            ratio(data.get("faithfulness",            {}).get("claims",      []), "supported"),
        "context_precision":       ratio(data.get("context_precision",       {}).get("contexts",    []), "relevant"),
        "constraint_satisfaction": ratio(data.get("constraint_satisfaction", {}).get("constraints", []), "satisfied")
    }


# ---------------------------------------------------------------------------
# 3. Evaluation prompt — multi-meal aware
# ---------------------------------------------------------------------------
def evaluate_row(question: str, answer: str, meal_contexts: list[dict],
                 constraints: dict, system: str) -> dict:
    """
    Parameters
    ----------
    meal_contexts : list of dicts with keys "meal_index", "query", "results"
                    where results is a list of {content, metadata} dicts.
    constraints   : MealPlanConstraints.model_dump()
    """
    # Build a labelled context block per meal
    context_sections = []
    for meal in meal_contexts:
        idx = meal["meal_index"] + 1
        sub_q = meal.get("query", "")
        docs = meal.get("results", [])
        docs_text = "\n\n".join(
            f"  [{j+1}] {d['content']}\n  Metadata: {d['metadata']}"
            for j, d in enumerate(docs)
        ) if docs else "  (no documents retrieved)"
        context_sections.append(f"[Meal {idx} – sub-query: {sub_q}]\n{docs_text}")

    context_block = "\n\n".join(context_sections)
    constraints_str = json.dumps(constraints, indent=2)

    prompt = f"""
You are an evaluation assistant scoring a multi-meal RAG system.

--- QUESTION ---
{question}

--- RETRIEVED CONTEXTS (grouped by meal) ---
{context_block}

--- ANSWER ---
{answer}

--- EXTRACTED CONSTRAINTS (MealPlanConstraints) ---
{constraints_str}

Evaluate the following metrics:

1. FAITHFULNESS: For each factual claim in the answer, is it supported by the
   retrieved contexts? Numerical/unit conversions are faithful only if
   mathematically correct given a retrieved value. (true/false per claim)

2. CONTEXT_PRECISION: For each context document (across all meals), is it
   relevant to the sub-query it was retrieved for? (true/false per document)

3. CONSTRAINT_SATISFACTION: For each constraint (meal-level AND global), does
   the answer satisfy it? Include per-meal constraints and global constraints
   (e.g. total calories, required categories). (true/false per constraint)

Respond ONLY with JSON matching this exact schema (empty list if not applicable):
{{
    "faithfulness":            {{"claims":      [{{"claim": "", "supported": true}}]}},
    "context_precision":       {{"contexts":    [{{"doc_ref": "Meal N [j]", "relevant": true}}]}},
    "constraint_satisfaction": {{"constraints": [{{"constraint": "", "satisfied": true}}]}}
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
# 4. System runner — returns (answer, filtered_meal_results)
# ---------------------------------------------------------------------------
def run_rag_hybrid(prompt: str, vectorstore, meal_plan: MealPlanConstraints):
    """
    meal_plan : MealPlanConstraints (from text_to_mealplan)
    Returns   : (answer: str, filtered_meal_results: list[dict])
    """
    retriever = LangchainRetrieverHybrid(
        vectorstore,
        json_query=meal_plan.meals   # list[RecipeConstraints]
    )
    generator = LlamaGenerator(stream=False)
    pipeline  = RAGPipeline(retriever, generator)

    answer, filtered_meal_results = pipeline.run(
        prompt,
        global_constraints=meal_plan.global_constraints
    )
    return answer, filtered_meal_results


# ---------------------------------------------------------------------------
# 5. Serialisation helpers
# ---------------------------------------------------------------------------
def sanitize_metadata(meta):
    if meta is None or isinstance(meta, (str, int, float, bool)):
        return meta
    if isinstance(meta, dict):
        return {str(k): sanitize_metadata(v) for k, v in meta.items()}
    if isinstance(meta, (list, tuple)) or (
        hasattr(meta, "__iter__") and not isinstance(meta, (str, bytes))
    ):
        return [sanitize_metadata(v) for v in meta]
    return str(meta)


CSV_FIELDS = [
    "system", "question", "answer", "n_meals", "response_time",
    "faithfulness", "context_precision", "constraint_satisfaction",
    "faithfulness_mean", "context_precision_mean",
    "constraint_satisfaction_mean", "response_time_mean",
]

CSV_FIELDS_REDUCED = [
    "system", "n_meals", "response_time",
    "faithfulness", "context_precision", "constraint_satisfaction",
]


def write_csv_line(path: Path, row: dict):
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, quoting=csv.QUOTE_ALL)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in CSV_FIELDS})


def write_csv_line_reduced(path: Path, row: dict):
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS_REDUCED, quoting=csv.QUOTE_ALL)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in CSV_FIELDS_REDUCED})


def write_json_line(path: Path, row: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# 6. MAIN evaluation loop
# ---------------------------------------------------------------------------
def run_evaluation(vectorstore, test_questions: list[dict], output_dir: str = "eval_results_planning", start_index: int = 0) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    timestamp     = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path      = output_dir / f"eval_{timestamp}.csv"
    csv_reduced   = output_dir / f"eval_{timestamp}_reduced.csv"
    json_path     = output_dir / f"eval_{timestamp}.jsonl"

    systems = ["rag_hybrid"]

    aggregate = {
        s: {k: [] for k in [
            "faithfulness", "context_precision",
            "constraint_satisfaction", "response_time"
        ]} for s in systems
    }

    for idx, item in enumerate(test_questions[start_index:], start=start_index + 1):
        question = item["question"]
        print(f"\n[{idx}/{len(test_questions)}] {question}")

        # --- Parse constraints (multi-meal aware) ---
        meal_plan   = text_to_mealplan(question, seed=SEED)
        constraints = meal_plan.model_dump()
        n_meals     = len(meal_plan.meals)

        for system in systems:
            print(f"  -> {system}  ({n_meals} meal(s))")

            start  = time.time()
            answer, filtered_meal_results = run_rag_hybrid(question, vectorstore, meal_plan)
            resp_time = time.time() - start

            # Build meal_contexts for the evaluator
            meal_contexts = [
                {
                    "meal_index": meal["meal_index"],
                    "query":      meal.get("query", ""),
                    "results": [
                        {"content": d.page_content, "metadata": d.metadata}
                        for d in meal["results"]
                    ],
                }
                for meal in filtered_meal_results
            ]

            result = evaluate_row(question, answer, meal_contexts, constraints, system)
            scores = result["scores"]

            row = {
                "system":       system,
                "question":     question,
                "answer":       answer,
                "n_meals":      n_meals,
                "response_time": resp_time,
                "faithfulness":            scores.get("faithfulness"),
                "context_precision":       scores.get("context_precision"),
                "constraint_satisfaction": scores.get("constraint_satisfaction"),
            }

            for k in aggregate[system]:
                val = row.get(k) if k in row else scores.get(k)
                if val is not None:
                    aggregate[system][k].append(val)

            means = {
                f"{k}_mean": (
                    sum(vs) / len(vs) if (vs := [v for v in vals if v is not None]) else None
                )
                for k, vals in aggregate[system].items()
            }
            print(f"     running means: {means}")

            write_csv_line(csv_path, {**row, **means})
            write_csv_line_reduced(csv_reduced, row)
            write_json_line(json_path, sanitize_metadata({
                **row,
                "meal_contexts": meal_contexts,
                "constraints":   constraints,
                "scores":        scores,
                "running_mean":  means,
                "detail":        result["detail"],
            }))

    print(f"\nSaved CSV : {csv_path}")
    print(f"Saved JSONL: {json_path}")
    return pd.read_csv(csv_path, engine="python")


# ---------------------------------------------------------------------------
# 7. Load questions
# ---------------------------------------------------------------------------
def load_questions_from_txt(filepath: str) -> list[dict]:
    questions = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                questions.append({"question": line})
    return questions


# ---------------------------------------------------------------------------
# 8. Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_set    = load_questions_from_txt("evaluation/test_sets/testset_planning.txt")
    indexer     = OllamaIndexer(collection_name="recipes")
    vectorstore = indexer.get_vectorstore()

    df = run_evaluation(vectorstore, test_set)

    print("\nFINAL RESULTS:")
    print(df.groupby("system").mean(numeric_only=True))