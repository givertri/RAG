import csv
import ast
from langchain_core.documents import Document
from core.base_loader import BaseLoader
import nltk
from nltk.stem import WordNetLemmatizer
import re

nltk.download("wordnet")
nltk.download("omw-1.4")

class CsvLoader(BaseLoader):
    def __init__(self, file_path: str, max_rows: int = None):
        self.file_path = file_path
        self.max_rows = max_rows
        self.lemmatizer = WordNetLemmatizer()
        self.INGREDIENT_STOPWORDS = {
            'cup', 'cups', 'tsp', 'teaspoon', 'tablespoon', 'tbsp', 't', 'oz', 'ounce',
            'ounces', 'pound', 'lb', 'lbs', 'can', 'jar', 'slice', 'slices', 
            'clove', 'cloves', 'bunch', 'pinch', 'piece', 'pieces', 'stalk', 'stalks',
            'medium', 'large', 'small', 'fresh', 'drained', 'chopped', 'diced', 
            'beaten', 'thinly', 'peeled', 'cooked', 'optional', 'minced', 'skinless',
            'skinon', 'ground', 'allpurpose', 'white', 'red', 'green', 'plain', 'black',
            'refrigerated', 'round', 'for', 'and', 'or', 'to', 'only', 'half', 'sprig',
            'leaf', 'juiced', 'zested', 'cored', 'cut', 'into', 'chunk', 'sliced', 'shredded',
            'reserved', 'taste', 'chunk', 'chunks', 'half', 'halves', 'style', 'greek', 'low',
            'sodium', 'boneless', 'dried', 'frying', 'cold', 'baked', 'spray', 'cooking',
            'all', 'purpose', 'with', 'on', 'skin', 'grain', 'style', 'fluid', 
            'bulk', 'half', 'bottle', 'box', 'package', 'container', 'bag', 'freshly',
            'uncooked'
        }
        self.UNITS = {'tablespoon', 'tbsp', 'tsp', 'teaspoon', 't', 'cup', 'cups', 'oz', 'ounce', 'ounces', 'pound', 'lb', 'lbs'}
        self.EXTRA_JUNK = {
            'divided', 'needed', 'thawed', 'rinsed', 'pitted', 'shucked', 'seeded', 
            'quartered', 'halved', 'pounded', 'cubed', 'melted', 'softened', 'grated', 
            'crushed', 'shredded', 'shaved', 'prepared', 'separated', 'more', 'plus', 
            'extra', 'roughly', 'finely', 'lightly', 'thinly', 'vertically', 'thick', 
            'inch', 'quart', 'pint', 'dash', 'bite', 'sized', 'size', 'room', 
            'temperature', 'pulp', 'peeling', 'weight', 'volume', 'at', 'in', 'six',
            'lengthwise', 'cover', 'dusting', 'garnish', 'packed', 'neck', 'giblet',
            'removed', 'mccormick', 'hidden', 'cove', 'pure', 'extract', 'liquid'
        }
        self.KITCHEN_SUPPLIES = {
            'foil', 'wrap', 'toothpick', 'stick', 'skewer', 'bag', 'parchment', 
            'paper', 'plastic', 'container', 'slow', 'cooker', 'reynolds'
        }
        self.full_stopword_set = self.INGREDIENT_STOPWORDS | self.UNITS | self.EXTRA_JUNK | self.KITCHEN_SUPPLIES

    def _clean_list_string(self, text):
        """Converts "['a', 'b']" to "a, b" for embedding."""
        try:
            data = ast.literal_eval(text)
            if isinstance(data, list):
                return ", ".join(data)
        except (ValueError, SyntaxError):
            pass
        return text

    def _parse_list(self, text):
        """Convert string-represented list to normalized Python list."""
        if not text or text.strip() == "":
            return []

        try:
            data = ast.literal_eval(text)
            if isinstance(data, list):
                return [
                    str(item).strip().lower()
                    for item in data
                    if str(item).strip()
                ]
        except (ValueError, SyntaxError):
            pass

        # fallback (if not proper list string)
        return [
            item.strip().lower()
            for item in text.split(",")
            if item.strip()
        ]

    def _normalize_ingredient(self, ingredient: str) -> str:
        ingredient = ingredient.lower()
        ingredient = re.sub(r"\(.*?\)", "", ingredient)
        ingredient = re.sub(r"[®™*]", "", ingredient)
        ingredient = re.sub(r"[,.\-/_/]", " ", ingredient)
        ingredient = re.sub(r"\d+/?\d*", "", ingredient)

        words = ingredient.split()
        normalized_words = []
        
        for w in words:
            lemma = self.lemmatizer.lemmatize(w)
            if lemma not in self.full_stopword_set and len(lemma) > 2:
                normalized_words.append(lemma)
            
        return " ".join(normalized_words).strip()

    def load(self):
        documents = []

        float_cols = {
            'rating', 'calories_kcal', 'protein_g', 'fat_g',
            'carbohydrates_g', 'cholesterol_mg', 'fiber_g',
            'saturated_fat_g', 'sodium_mg', 'sugar_g',
            'unsaturated_fat_g'
        }

        list_cols = {'ingredients', 'utensils', 'cuisine', 'category'}

        with open(self.file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)

            for i, row in enumerate(reader, start=1):
                if self.max_rows is not None and i > self.max_rows:
                    break

                name = row.get('name', '').strip()

                # STRING versions (for embedding)
                cuisine_str = self._clean_list_string(row.get('cuisine', ''))
                ingredients_str = self._clean_list_string(row.get('ingredients', ''))
                instructions = self._clean_list_string(row.get('instructions', ''))

                # LIST versions (for filtering)
                cuisine_list = self._parse_list(row.get('cuisine', ''))
                category_list = self._parse_list(row.get('category', ''))
                ingredients_list_raw = self._parse_list(row.get('ingredients', ''))
                raw_normalized = [self._normalize_ingredient(ing) for ing in ingredients_list_raw]
                ingredients_list = sorted(list(set(ing for ing in raw_normalized if ing)))
                utensils_list = self._parse_list(row.get('utensils', ''))

                page_content = (
                    f"Recipe: {name}\n"
                    f"Cuisine: {cuisine_str}\n"
                    f"Ingredients: {ingredients_str}\n"
                    f"Instructions: {instructions}"
                )

                metadata = {}
                for k, v in row.items():
                    if k in list_cols:
                        metadata[k] = locals().get(f"{k}_list", [])
                        continue
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

                metadata['pk'] = i

                documents.append(
                    Document(page_content=page_content, metadata=metadata)
                )

                if i % 10000 == 0:
                    print(f"Processed {i} rows...")

        print(f"Finished processing {len(documents)} rows.")
        return documents