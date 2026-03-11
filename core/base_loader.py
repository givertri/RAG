from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document

class BaseLoader(ABC):

    @abstractmethod
    def load(self) -> List[Document]:
        """Load and optionally preprocess documents."""
        pass
