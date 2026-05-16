from core.base_generator import BaseGenerator
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import os


class LlamaGenerator(BaseGenerator):
    def __init__(self, stream: bool = True, seed = None):
        load_dotenv(dotenv_path="env.env")

        self.stream = stream

        # Connect to locally running Ollama instance
        self.llm = ChatOllama(
            model="llama3.2:3b",
            temperature=0.2,
            base_url=os.getenv("RUNPOD_URL"),
            options={
                "seed": seed
            }
        )

    def generate(self, query, docs, with_retrieval, stream: bool = None):
        # Allow override per call; fallback to default
        if stream is None:
            stream = self.stream

        context = "\n\n".join([
            f"{d.page_content}\nMetadata:\n{d.metadata}"
            for d in docs
        ])

        sysprompt = (
            f"You are an assistant specialized in suggesting meals and recipes.\n\n"
            
            "When responding:\n"
            "- Provide a clear recipe with a title.\n"
            "- Include a short description of the dish.\n"
            "- List all ingredients with quantities in European metric units (grams, milliliters, etc.).\n"
            "- Provide step-by-step cooking instructions in a clear, logical order.\n"
            "- Keep the answer concise but complete.\n\n"

            f"Question: {query}\n"
            f"Answer:"
        )

        syspromptRAG = (
            f"You are a retrieval-augmented assistant specialized in suggesting meals and recipes. Use the retrieved information below as your primary source of truth. If the information is incomplete, "
            "you may supplement it with general cooking knowledge, but do not contradict the retrieved content.\n\n"

            "When responding:\n"
            "- Provide a clear recipe with a title.\n"
            "- Include a short description of the dish.\n"
            "- List all ingredients with quantities in European metric units (grams, milliliters, etc.).\n"
            "- Provide step-by-step cooking instructions in a clear, logical order.\n"
            "- Keep the answer concise but complete.\n\n"

            "Retrieved information:\n"
            f"{context}\n\n"
            f"Question: {query}\n"
            f"Answer:"
        )

        prompt = syspromptRAG if with_retrieval else sysprompt
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