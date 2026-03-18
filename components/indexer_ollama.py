from core.base_indexer import BaseIndexer
from langchain_milvus import Milvus, BM25BuiltInFunction
from langchain_ollama import OllamaEmbeddings
from uuid import uuid4


class OllamaIndexer(BaseIndexer):

    def __init__(self, collection_name="recipes"):
        self.collection_name = collection_name

        # Use Ollama local embedding model
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text:latest",
            base_url="http://localhost:11434"  # Ollama URL
        )

        self.URI = "http://localhost:19530" # Milvus

        bm25_func = BM25BuiltInFunction(output_field_names="sparse")

        self.vector_store = Milvus(
            embedding_function=self.embeddings,
            builtin_function=bm25_func,
            vector_field=["dense", "sparse"],    # Hybrid search
            connection_args={"uri": self.URI},
            collection_name=self.collection_name,
            drop_old=False,
            index_params=[
                {
                    #"field_name": "dense",
                    "index_type": "AUTOINDEX",
                    "metric_type": "COSINE",
                },
                {
                    #"field_name": "sparse",
                    "index_type": "SPARSE_INVERTED_INDEX",
                    "metric_type": "BM25",
                }
            ]
        )

    def index(self, documents=None):
        if not documents:
            return

        # Get current number of rows in the collection
        total_docs = len(documents)
        print(f"Starting indexing {total_docs} documents.")
    
        all_ids = [str(doc.metadata['row_id']) for doc in documents]

        batch_size = 100
        for i in range(0, total_docs, batch_size):
            batch_docs = documents[i : i + batch_size]
            batch_ids = all_ids[i : i + batch_size]

            # Pass the batch_ids to add_documents
            self.vector_store.add_documents(documents=batch_docs, ids=batch_ids)

            print(f"Indexed {i + len(batch_docs)}/{total_docs} documents...")

        print(f"Indexing completed. Total rows in vector store: {self.get_num_rows()}")

    def get_vectorstore(self):
        return self.vector_store
    
    def drop(self):
        self.vector_store.drop()
    
    def get_rows(self, n):
        collection = self.vector_store.col
        collection.load()

        results = collection.query(
            expr="",
            output_fields=["*"],
            limit=n
        )

        return results
    
    def get_num_rows(self):
        collection = self.vector_store.col
        collection.load()
        return collection.num_entities
