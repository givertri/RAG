from components.loader_fromcsv import CsvLoader
from components.indexer_ollama import OllamaIndexer

def main():
    #drop()
    #print("dropped")
    populate()
    print("populated")
    #count()
    #view(1)

def populate():
    docs = None
    def load_data():
        loader = CsvLoader("dataset/cleaned.csv")
        return loader.load()

    docs = load_data()
    # Index documents
    indexer = OllamaIndexer(collection_name="recipes")
    indexer.index(docs)

def drop():
    indexer = OllamaIndexer(collection_name="recipes")
    indexer.drop()

def view(n):
    indexer = OllamaIndexer(collection_name="recipes")
    print(indexer.get_rows(n))

def count():
    indexer = OllamaIndexer(collection_name="recipes")
    print("Row count db: ", indexer.get_num_rows())

if __name__ == "__main__":
    main()