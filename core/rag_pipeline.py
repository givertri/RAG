import time
from .base_retriever import BaseRetriever
from .base_generator import BaseGenerator
from evaluation.rag_logger import log_rag_result

class RAGPipeline:

    def __init__(self, retriever: BaseRetriever, generator: BaseGenerator):
        self.retriever = retriever
        self.generator = generator

    def run(self, query: str) -> str:
        # Measure retrieval time
        retrieval_start = time.perf_counter()
        docs = self.retriever.retrieve(query)
        retrieval_end = time.perf_counter()
        retrieval_time = retrieval_end - retrieval_start

        #print(f"Retrieval time: {retrieval_time:.4f} seconds")

        # Measure generation time
        generation_start = time.perf_counter()
        response = self.generator.generate(query, docs, with_retrieval=True)
        generation_end = time.perf_counter()
        generation_time = generation_end - generation_start

        #print(f"Generation time: {generation_time:.4f} seconds")

        log_rag_result(question=query, answer=response, contexts=docs, ret_time=retrieval_time, gen_time=generation_time)

        return response, docs
