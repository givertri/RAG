import csv
import json
from pathlib import Path

CSV_PATH = "rag_outputs.csv"

def make_json_serializable(obj):
    # Basic types (already safe)
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj

    # Dict
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}

    # List / tuple / set
    if isinstance(obj, (list, tuple, set)):
        return [make_json_serializable(v) for v in obj]

    # Protobuf repeated containers (like RepeatedScalarContainer)
    if type(obj).__name__ == "RepeatedScalarContainer":
        return [make_json_serializable(v) for v in obj]

    # Protobuf messages (if any)
    if hasattr(obj, "ListFields"):
        return {k.name: make_json_serializable(v) for k, v in obj.ListFields()}

    # Anything iterable (fallback)
    if hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes)):
        try:
            return [make_json_serializable(v) for v in obj]
        except TypeError:
            pass

    # Final fallback: string
    return str(obj)

def log_rag_result(question, answer, contexts, ret_time, gen_time):
    file_exists = Path(CSV_PATH).exists()

    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["question", "answer", "contexts", "ret_time", "gen_time"]
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "question": question,
            "answer": answer,
            "contexts": json.dumps([{
                "page_content": doc.page_content,
                "metadata": make_json_serializable(doc.metadata)
            } for doc in contexts]),
            "ret_time": ret_time,
            "gen_time": gen_time
        })