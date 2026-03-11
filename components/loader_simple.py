from langchain_core.documents import Document
from core.base_loader import BaseLoader

class SimpleLoader(BaseLoader):

    def __init__(self, texts: list[str]):
        self.texts = texts

    def load(self):
        return [Document(page_content=t) for t in self.texts]
