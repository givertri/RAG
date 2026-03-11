from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document

class BaseGenerator(ABC):

    @abstractmethod
    def generate(self, query: str, docs: List[Document], with_retrieval: bool) -> str:
        """Generate an answer from query + context."""
        pass
