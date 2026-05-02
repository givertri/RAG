from core.base_retriever import BaseRetriever

class LangchainRetriever(BaseRetriever):
    def __init__(self, vectorstore, k=5):
        self.vectorstore = vectorstore
        self.k = k

    def retrieve(self, query: str):
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": self.k
            }
        )

        results = retriever.invoke(query)
        return results