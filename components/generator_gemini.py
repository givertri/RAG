from core.base_generator import BaseGenerator
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

class GeminiGenerator(BaseGenerator):

    def __init__(self):
        load_dotenv(dotenv_path="env.env")
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            temperature=0.2
        )

    def generate(self, query, docs, with_retrieval):
        context = "\n\n".join([d.page_content for d in docs])

        prompt = (
            f"You are a helpful assistant.\nQuestion: {query}\nAnswer:"
        )

        promptRAG = (
            f"You are a RAG. Use the retrieved information (which is correct) in your answer.\nRetrieved information: {context}\nQuestion: {query}\nAnswer:"
        )

        if with_retrieval:
            return self.llm.invoke(promptRAG).content
        else:
            return self.llm.invoke(prompt).content
