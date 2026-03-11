import csv
from langchain_core.documents import Document
from core.base_loader import BaseLoader

class CsvLoader(BaseLoader):
    def __init__(self, file_path: str, max_rows: int = None):
        """
        Initialize the loader with the CSV file path.
        """
        self.file_path = file_path
        self.max_rows = max_rows

    def load(self):
        """
        Load CSV rows and convert them to LangChain Documents.
        Each row should have 'name' and 'instructions' columns.
        Metadata includes all other columns.
        """
        documents = []
        with open(self.file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for i, row in enumerate(reader, start=1):

                if self.max_rows is not None and i > self.max_rows:
                    break

                # Combine features
                name = row.get('name', '').strip()
                cuisine = row.get('cuisine', '').strip()
                ingredients = row.get('ingredients', '').strip()
                instructions = row.get('instructions', '').strip()
                page_content = f"{name}, {cuisine}, {ingredients}, {instructions}"

                # Metadata
                metadata = {k: v for k, v in row.items()}

                # Create a LangChain Document
                doc = Document(page_content=page_content, metadata=metadata)
                documents.append(doc)

                # Print status every 10000 rows
                if i % 10000 == 0:
                    print(f"Processed {i} rows...")

        print(f"Finished processing {len(documents)} rows.")
        return documents
