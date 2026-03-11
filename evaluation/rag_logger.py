import csv
import json
from pathlib import Path

CSV_PATH = "rag_outputs.csv"

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
                "metadata": doc.metadata
            } for doc in contexts]),
            "ret_time": ret_time,
            "gen_time": gen_time
        })