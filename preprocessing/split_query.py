import json
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage


def split_query(text, n_meals):
    llm = ChatOllama(
        model="qwen3:4b",
        temperature=0,
        format="json",
        reasoning=False,
        base_url="http://203.57.40.79:10203"
    )

    prompt = f"""
You are a query parser.

Split the following user request into exactly {n_meals} meal-specific search queries.

Rules:
- Each query should correspond to ONE meal
- Keep them concise and optimized for recipe retrieval
- Preserve important constraints (cuisine, diet, ingredients, etc.)
- Do NOT add explanations

Output format (STRICT JSON):
{{
  "queries": ["query 1", "query 2", "..."]
}}

User query:
"{text}"
"""

    response = llm.invoke([HumanMessage(content=prompt)])

    # ---- Safe parsing ----
    try:
        data = json.loads(response.content)
        queries = data.get("queries", [])
    except Exception:
        print("Failed to parse LLM output.")
        return [text] * n_meals

    # ---- Validation ----
    if not isinstance(queries, list) or len(queries) != n_meals:
        print("Wrong number of queries.")
        return [text] * n_meals

    return queries