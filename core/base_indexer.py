from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document

class BaseIndexer(ABC):

    @abstractmethod
    def index(self, documents: List[Document]):
        """Index all documents into the vector store."""
        pass

    @abstractmethod
    def get_vectorstore(self):
        """Return the vector store instance (for retrievers)."""
        pass
