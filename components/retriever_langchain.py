from core.base_retriever import BaseRetriever

class LangchainRetriever(BaseRetriever):

    def __init__(self, vectorstore, k=5):
        self.retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )

    def retrieve(self, query: str):
        return self.retriever.invoke(query)
