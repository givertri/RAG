from core.base_generator import BaseGenerator
from langchain_ollama import ChatOllama
from dotenv import load_dotenv


class LlamaGenerator(BaseGenerator):

    def __init__(self):
        load_dotenv(dotenv_path="env.env")

        # Connect to locally running Ollama instance
        self.llm = ChatOllama(
            model="llama3.2:latest",   # Your local Ollama model
            temperature=0.2,
        )

    def generate(self, query, docs, with_retrieval):
        context = "\n\n".join([d.page_content for d in docs])

        sysprompt = (
            f"You are a helpful assistant.\n"
            f"Question: {query}\n"
            f"Answer:"
        )

        syspromptRAG = (
            f"You are a meal-suggesting RAG. Use the retrieved information (which is correct) in your answer.\n\n"
            f"Give the recipe instructions in European metrics.\n\n"
            f"Retrieved information:\n{context}\n\n"
            f"Question: {query}\n"
            f"Answer:"
        )

        prompt = syspromptRAG if with_retrieval else sysprompt
        
        full_response = ""

        for chunk in self.llm.stream(prompt):
            print(chunk.content, end="", flush=True)  # stream to console
            full_response += chunk.content

        print()  # newline after completion
        return full_response
