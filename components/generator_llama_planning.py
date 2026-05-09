from core.base_generator import BaseGenerator
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import os


class LlamaGenerator(BaseGenerator):
    def __init__(self, stream: bool = True):
        load_dotenv(dotenv_path="env.env")
        self.stream = stream
        self.llm = ChatOllama(
            model="llama3.2:latest",
            temperature=0.2,
            base_url=os.getenv("RUNPOD_URL"),
        )

    # ------------------------------------------------------------------
    # Context builder
    # ------------------------------------------------------------------

    def _build_context(self, meal_results: list) -> str:
        sections = []
        for meal in meal_results:
            idx = meal["meal_index"] + 1
            query = meal.get("query", "")
            docs = meal.get("results", [])
            doc_text = "\n\n".join(f"{d.page_content}\nMetadata:\n{d.metadata}" for d in docs) if docs else "No recipes retrieved."
            sections.append(f"--- Meal {idx} (query: {query}) ---\n{doc_text}")
        return "\n\n".join(sections)

    # ------------------------------------------------------------------
    # Global constraints renderer
    # ------------------------------------------------------------------

    def _build_global_constraints_text(self, global_constraints) -> str:
        """
        Renders a GlobalConstraints object (or dict) into a plain-English
        block that the LLM can reason over when selecting/adjusting recipes.
        Only non-null fields are emitted to keep the prompt clean.
        """
        if global_constraints is None:
            return ""

        # Accept both Pydantic models and plain dicts
        if hasattr(global_constraints, "dict"):
            gc = global_constraints.dict()
        else:
            gc = dict(global_constraints)

        lines = []

        # --- Structural constraints ---
        required = [c for c in (gc.get("required_categories") or []) if c]
        if required:
            lines.append(f"- Required meal categories (one recipe per category): {', '.join(required)}")

        min_m = gc.get("min_meals")
        max_m = gc.get("max_meals")
        if min_m is not None and max_m is not None:
            if min_m == max_m:
                lines.append(f"- Exactly {min_m} meal(s) must be provided.")
            else:
                lines.append(f"- Between {min_m} and {max_m} meals must be provided.")
        elif min_m is not None:
            lines.append(f"- At least {min_m} meal(s) must be provided.")
        elif max_m is not None:
            lines.append(f"- At most {max_m} meal(s) must be provided.")

        if gc.get("unique_categories"):
            lines.append("- No two meals may share the same category.")

        # --- Aggregate nutrient targets ---
        nutrient_labels = {
            "total_calories_kcal":      ("Total calories across all meals",  "kcal"),
            "total_carbohydrates_g":    ("Total carbohydrates across all meals", "g"),
            "total_cholesterol_mg":     ("Total cholesterol across all meals",   "mg"),
            "total_fiber_g":            ("Total fiber across all meals",          "g"),
            "total_protein_g":          ("Total protein across all meals",        "g"),
            "total_saturated_fat_g":    ("Total saturated fat across all meals",  "g"),
            "total_sodium_mg":          ("Total sodium across all meals",         "mg"),
            "total_sugar_g":            ("Total sugar across all meals",          "g"),
            "total_fat_g":              ("Total fat across all meals",            "g"),
            "total_unsaturated_fat_g":  ("Total unsaturated fat across all meals","g"),
        }

        for key, (label, unit) in nutrient_labels.items():
            value_range = gc.get(key)
            if not isinstance(value_range, (list, tuple)) or len(value_range) != 2:
                continue
            lo, hi = value_range
            if lo is not None and hi is not None:
                lines.append(f"- {label}: {lo:.1f}–{hi:.1f} {unit}")
            elif lo is not None:
                lines.append(f"- {label}: at least {lo:.1f} {unit}")
            elif hi is not None:
                lines.append(f"- {label}: at most {hi:.1f} {unit}")

        if not lines:
            return ""

        return "Global constraints that must hold across ALL meals combined:\n" + "\n".join(lines)

    # ------------------------------------------------------------------
    # Prompt builder
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        query: str,
        meal_results: list,
        with_retrieval: bool,
        global_constraints=None,
    ) -> str:
        n = len(meal_results)
        meal_word = "recipe" if n == 1 else f"{n} recipes (one per meal)"

        per_meal_instructions = (
            f"Provide {meal_word}. For each recipe:\n"
            "- Give a clear title.\n"
            "- Write a short description of the dish.\n"
            "- List all ingredients with quantities in European metric units (g, ml, etc.).\n"
            "- Provide step-by-step cooking instructions in logical order.\n"
            "- Keep each recipe concise but complete.\n"
        )

        global_block = self._build_global_constraints_text(global_constraints)
        if global_block:
            global_section = (
                f"\n{global_block}\n"
                "When selecting and presenting recipes, make sure the combination satisfies "
                "all global constraints listed above. If a retrieved recipe does not fit, "
                "adjust portion sizes or substitute it with a suitable alternative.\n"
            )
        else:
            global_section = ""

        if with_retrieval:
            context = self._build_context(meal_results)
            return (
                "You are a retrieval-augmented assistant specialized in suggesting meals and recipes. "
                "Use the retrieved information below as your primary source of truth. "
                "If the information is incomplete you may supplement with general cooking knowledge, "
                "but do not contradict the retrieved content.\n\n"
                f"{per_meal_instructions}"
                f"{global_section}\n"
                f"Retrieved information:\n{context}\n\n"
                f"Question: {query}\n"
                "Answer:"
            )
        else:
            return (
                "You are an assistant specialized in suggesting meals and recipes.\n\n"
                f"{per_meal_instructions}"
                f"{global_section}\n"
                f"Question: {query}\n"
                "Answer:"
            )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate(
        self,
        query: str,
        meal_results: list,
        with_retrieval: bool = True,
        global_constraints=None,
        stream: bool = None,
    ) -> str:
        if stream is None:
            stream = self.stream

        prompt = self._build_prompt(query, meal_results, with_retrieval, global_constraints)
        print(prompt)

        if stream:
            full_response = ""
            for chunk in self.llm.stream(prompt):
                print(chunk.content, end="", flush=True)
                full_response += chunk.content
            print()
            return full_response
        else:
            response = self.llm.invoke(prompt)
            return response.content