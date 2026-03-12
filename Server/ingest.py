import ollama
import pandas as pd
import ast
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm
import sys

CSV_PATH = 'full_dataset.csv'
DB_PATH = "./recipe_db"
COLLECTION_NAME = "recipes"
EMBEDDING_MODEL_NAME = 'nomic-embed-text'
DB_BATCH_SIZE = 100

print("Initializing Database connection for INGESTION...")
client = chromadb.PersistentClient(path=DB_PATH)

class OllamaEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __call__(self, input: list[str]) -> list[list[float]]:
        try:
            response = ollama.embed(model=EMBEDDING_MODEL_NAME, input=input)
            return response['embeddings']
        except Exception as e:
            print(f"Error embedding batch: {e}")
            return []

embedding_func = OllamaEmbeddingFunction()

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_func,
    metadata={"hnsw:space": "cosine"}
)

def format_list_with_bullets(items):
    return "\n".join([f"- {item}" for item in items])

def format_list_numbered(items):
    return "\n".join([f"{i+1}. {item}" for i, item in enumerate(items)])

def safe_eval(val):
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except:
            return []
    return val if isinstance(val, list) else []

def ingest_data():
    try:
        existing_count = collection.count()
        print(f"Database currently holds {existing_count} recipes.")
        
        print(f"Reading {CSV_PATH}...")
        df = pd.read_csv(CSV_PATH)
        total_rows = len(df)
        
        if existing_count >= total_rows:
            print("All recipes are already in the database! Nothing to do.")
            return

        print(f"Resuming ingestion from row {existing_count}...")

        documents = []
        metadatas = []
        ids = []

        for index, row in tqdm(df.iterrows(), total=total_rows, initial=existing_count, desc="Embedding Recipes"):
            
            if index < existing_count:
                continue

            try:
                title = row['title']
                link = row['link']
                source = row['source']
                
                ing_list = safe_eval(row['ingredients'])
                dir_list = safe_eval(row['directions'])
                ner_list = safe_eval(row['NER'])

                if not ing_list or not dir_list:
                    continue

                formatted_ingredients = format_list_with_bullets(ing_list)
                formatted_directions = format_list_numbered(dir_list)
                formatted_keywords = ", ".join(ner_list)

                text_chunk = (
                    f"Recipe Title: {title}\n\n"
                    f"Ingredients:\n{formatted_ingredients}\n\n"
                    f"Instructions:\n{formatted_directions}\n\n"
                    f"Search Keywords: {formatted_keywords}"
                )

                meta = {
                    "title": title,
                    "link": link if isinstance(link, str) else "",
                    "source": source if isinstance(source, str) else "Unknown",
                    "ingredients_list": str(ner_list)
                }

                documents.append(text_chunk)
                metadatas.append(meta)
                ids.append(f"recipe_{index}")

                if len(documents) >= DB_BATCH_SIZE:
                    collection.add(documents=documents, metadatas=metadatas, ids=ids)
                    documents, metadatas, ids = [], [], []

            except Exception as e:
                print(f"Skipping row {index} due to error: {e}")
                continue

        if documents:
            collection.add(documents=documents, metadatas=metadatas, ids=ids)

        print("Ingestion Complete!")

    except KeyboardInterrupt:
        print("\n\n[PAUSED] Ingestion interrupted by user.")
        print("Progress has been saved. Run the script again to resume.")
        sys.exit(0)
    except FileNotFoundError:
        print(f"Error: {CSV_PATH} not found.")

if __name__ == "__main__":
    ingest_data()