from core.base_indexer import BaseIndexer
from langchain_milvus import Milvus
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from uuid import uuid4

class GoogleIndexer(BaseIndexer):

    def __init__(self, collection_name="recipes"):
        self.collection_name = collection_name
        load_dotenv(dotenv_path="env.env")
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        self.URI = "http://localhost:19530"

    def index(self, documents=None):
        self.vector_store = Milvus(
            embedding_function=self.embeddings,
            connection_args={"uri": self.URI},
            collection_name=self.collection_name,
            index_params = {"index_type": "AUTOINDEX", "metric_type": "COSINE"}
        )
        #self.vector_store.drop()
        if documents:
            uuids = [str(uuid4()) for _ in range(len(documents))]
            self.vector_store.add_documents(documents=documents, ids=uuids)

    def get_vectorstore(self):
        return self.vector_store
