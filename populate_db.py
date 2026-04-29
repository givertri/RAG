from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
    FunctionType, Function
)
from components.loader_fromcsv import CsvLoader
from components.indexer_ollama import OllamaIndexer

COLLECTION_NAME = "recipes"
MILVUS_URI = "http://localhost:19530"

def main():
    #drop()
    create_collection_if_not_exists()
    populate()
    count()
    print("Populated collection successfully.")

def create_collection_if_not_exists():
    # Connect to Milvus
    connections.connect(uri=MILVUS_URI)
    
    # Check if collection already exists
    existing_collections = utility.list_collections()
    if COLLECTION_NAME in existing_collections:
        print(f"Collection '{COLLECTION_NAME}' already exists, skipping creation.")
        return

    # Define schema with all CSV fields
    fields = [
        FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535, enable_analyzer=True),
        FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(
            name="cuisine",
            dtype=DataType.ARRAY,
            element_type=DataType.VARCHAR,
            max_length=100,
            max_capacity=10
        ),
        FieldSchema(
            name="category",
            dtype=DataType.ARRAY,
            element_type=DataType.VARCHAR,
            max_length=100,
            max_capacity=10
        ),
        FieldSchema(name="time_class", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(
            name="ingredients",
            dtype=DataType.ARRAY,
            element_type=DataType.VARCHAR,
            max_length=200,
            max_capacity=70
        ),
        FieldSchema(
            name="utensils",
            dtype=DataType.ARRAY,
            element_type=DataType.VARCHAR,
            max_length=100,
            max_capacity=20
        ),
        # numeric fields
        FieldSchema(name="rating", dtype=DataType.FLOAT),
        FieldSchema(name="calories_kcal", dtype=DataType.FLOAT),
        FieldSchema(name="protein_g", dtype=DataType.FLOAT),
        FieldSchema(name="fat_g", dtype=DataType.FLOAT),
        FieldSchema(name="carbohydrates_g", dtype=DataType.FLOAT),
        FieldSchema(name="cholesterol_mg", dtype=DataType.FLOAT),
        FieldSchema(name="fiber_g", dtype=DataType.FLOAT),
        FieldSchema(name="saturated_fat_g", dtype=DataType.FLOAT),
        FieldSchema(name="sodium_mg", dtype=DataType.FLOAT),
        FieldSchema(name="sugar_g", dtype=DataType.FLOAT),
        FieldSchema(name="unsaturated_fat_g", dtype=DataType.FLOAT),
        # vector fields for hybrid search
        FieldSchema(name="dense", dtype=DataType.FLOAT_VECTOR, dim=768),
        FieldSchema(name="sparse", dtype=DataType.SPARSE_FLOAT_VECTOR)
    ]

    bm25_function = Function(
        name="bm25",
        function_type=FunctionType.BM25,
        input_field_names=["text"],
        output_field_names=["sparse"],
    )

    schema = CollectionSchema(fields, description="Recipes dataset with hybrid search")
    schema.add_function(bm25_function)
    Collection(name=COLLECTION_NAME, schema=schema)
    print(f"Created collection '{COLLECTION_NAME}' with all CSV fields and ARRAY fields for ingredients and utensils.")

def populate():
    # Load CSV documents
    loader = CsvLoader("dataset/cleaned.csv")
    docs = loader.load()
    
    # Index documents
    indexer = OllamaIndexer(collection_name=COLLECTION_NAME)
    indexer.index(docs)

def drop():
    indexer = OllamaIndexer(collection_name=COLLECTION_NAME)
    indexer.drop()

def view(n):
    indexer = OllamaIndexer(collection_name=COLLECTION_NAME)
    print(indexer.get_rows(n))

def count():
    indexer = OllamaIndexer(collection_name=COLLECTION_NAME)
    print("Row count db: ", indexer.get_num_rows())

if __name__ == "__main__":
    main()
