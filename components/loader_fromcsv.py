import csv
import ast # To safely handle string-represented lists
from langchain_core.documents import Document
from core.base_loader import BaseLoader

class CsvLoader(BaseLoader):
    def __init__(self, file_path: str, max_rows: int = None):
        self.file_path = file_path
        self.max_rows = max_rows

    def _clean_list_string(self, text):
        """Converts "['a', 'b']" to "a, b" for better keyword indexing."""
        try:
            # Safely evaluate the string as a list
            data = ast.literal_eval(text)
            if isinstance(data, list):
                return ", ".join(data)
        except (ValueError, SyntaxError):
            pass
        return text

    def load(self):
        documents = []
        float_cols = {'rating', 'calories_kcal', 'protein_g', 'fat_g', 'carbohydrates_g'} # TODO add all columns
    
        with open(self.file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            # i starts at 1 for the first data row (skipping the header)
            for i, row in enumerate(reader, start=1):
                if self.max_rows is not None and i > self.max_rows:
                    break

                name = row.get('name', '').strip()
                cuisine = self._clean_list_string(row.get('cuisine', ''))
                ingredients = self._clean_list_string(row.get('ingredients', ''))
                instructions = self._clean_list_string(row.get('instructions', ''))
            
                page_content = (
                    f"Recipe: {name}\n"
                    f"Cuisine: {cuisine}\n"
                    f"Ingredients: {ingredients}\n"
                    f"Instructions: {instructions}"
                )

                metadata = {}
                for k, v in row.items():
                    if not v or v.strip() == "":
                        metadata[k] = 0.0 if k in float_cols else ""
                        continue
                
                    if k in float_cols:
                        try:
                            metadata[k] = float(v)
                        except ValueError:
                            metadata[k] = 0.0
                    else:
                        metadata[k] = v
            
                metadata['row_id'] = i # store unique row number in metadata

                documents.append(Document(page_content=page_content, metadata=metadata))

                if i % 10000 == 0:
                    print(f"Processed {i} rows...")

        print(f"Finished processing {len(documents)} rows.")
        return documents