import ollama
import chromadb
from chromadb.utils import embedding_functions
import json

DB_PATH = "./recipe_db"
COLLECTION_NAME = "recipes"
EMBEDDING_MODEL_NAME = 'nomic-embed-text'
GENERATION_MODEL_NAME = 'llama3.1'

print("Initializing ChefBot...")
client = chromadb.PersistentClient(path=DB_PATH)

class OllamaEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __call__(self, input: list[str]) -> list[list[float]]:
        response = ollama.embed(model=EMBEDDING_MODEL_NAME, input=input)
        return response['embeddings']

embedding_func = OllamaEmbeddingFunction()
collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_func)

class SessionState:
    def __init__(self):
        self.mode = "IDLE" 
        self.search_results = [] 
        self.current_recipe = None 
        self.chat_history = []

state = SessionState()


def determine_intent(user_input, state):
    """
    decides what the user wants to do based on input and current state.
    returns: 'SEARCH', 'SELECT', 'CHAT', or 'EXIT'
    """
    if user_input.lower() in ['exit', 'quit', 'q']:
        return 'EXIT'

    if state.mode == "IDLE":
        return "SEARCH"

    prompt = f"""
    Current Mode: {state.mode}
    Available Search Results: {[r['title'] for r in state.search_results] if state.search_results else "None"}
    Active Recipe: {state.current_recipe['title'] if state.current_recipe else "None"}
    
    User Input: "{user_input}"
    
    Task: Classify the User Input into exactly one of these categories:
    1. SEARCH (User wants to find NEW recipes, e.g., "I have tomatoes", "show me dessert", "list 10 recipes")
    2. SELECT (User is picking a recipe from the Available Search Results, e.g., "number 1", "the chicken one", "show me that")
    3. CHAT (User is asking a question about the Active Recipe or general talk, e.g., "how long to cook?", "is it healthy?", "thank you")
    
    Reply ONLY with the category word.
    """
    try:
        response = ollama.chat(model=GENERATION_MODEL_NAME, messages=[{'role': 'user', 'content': prompt}])
        intent = response['message']['content'].strip().upper()
        
        if "SEARCH" in intent: return "SEARCH"
        if "SELECT" in intent: return "SELECT"
        return "CHAT"
    except:
        return "CHAT"

def search_database(query):
    print(f"(Searching for: {query}...)")
    results = collection.query(
        query_texts=[query], 
        n_results=10 
    )
    
    hits = []
    if results['documents']:
        for i in range(len(results['documents'][0])):
            hits.append({
                "id": results['ids'][0][i],
                "title": results['metadatas'][0][i]['title'],
                "source": results['metadatas'][0][i]['source'],
                "link": results['metadatas'][0][i]['link'],
                "full_text": results['documents'][0][i]
            })
    return hits

def chat_loop():
    print("\n" + "="*50)
    print(" ChefBot is Online")
    print(" Try: 'I have chicken and rice' or 'List 5 desserts'")
    print("="*50 + "\n")

    while True:
        try:
            user_input = input('You > ').strip()
            if not user_input: continue
            
            intent = determine_intent(user_input, state)
            
            system_instruction = ""
            context_text = ""


            if intent == 'EXIT':
                print("ChefBot > Bon Appétit!")
                break

            elif intent == 'SEARCH':
                hits = search_database(user_input)
                
                if not hits:
                    print("ChefBot > I couldn't find any recipes matching that. Try different ingredients.")
                    continue
                
                state.search_results = hits
                state.mode = "BROWSING"
                state.current_recipe = None
                
                context_text = "FOUND RECIPES:\n" + "\n".join([f"{i+1}. {h['title']} (Source: {h['source']})" for i, h in enumerate(hits)])
                
                system_instruction = (
                    "You are a helpful chef. The user just searched for recipes. "
                    "Briefly list the recipes found in the context below. "
                    "Ask the user which one they would like to cook."
                )

            elif intent == 'SELECT':

                selection_prompt = f"""
                User Input: "{user_input}"
                Options:
                {[f"{i+1}. {r['title']}" for i, r in enumerate(state.search_results)]}
                
                Which index (1-10) is the user trying to select? 
                Reply ONLY with the number (e.g. "1"). If unsure, reply "0".
                """
                resp = ollama.chat(model=GENERATION_MODEL_NAME, messages=[{'role': 'user', 'content': selection_prompt}])
                try:
                    selection_index = int(''.join(filter(str.isdigit, resp['message']['content']))) - 1
                except:
                    selection_index = -1

                if 0 <= selection_index < len(state.search_results):
                    state.current_recipe = state.search_results[selection_index]
                    state.mode = "COOKING"
                    
                    recipe = state.current_recipe
                    context_text = recipe['full_text']
                    
                    system_instruction = (
                        "You are a helpful chef. The user has selected a specific recipe. "
                        "Present the recipe details nicely. "
                        "Include the Source and Link if available. "
                        f"Link: {recipe['link']}\n"
                        f"Source: {recipe['source']}"
                    )
                else:
                    system_instruction = "The user tried to select a recipe but it was unclear. Ask them to clarify which number or name they want."

            elif intent == 'CHAT':

                if state.mode == "COOKING" and state.current_recipe:
                    context_text = state.current_recipe['full_text']
                    system_instruction = (
                        "You are a chef guiding the user through the recipe below. "
                        "Answer their specific questions (e.g. substitutions, times, temps) based ONLY on the text."
                    )
                elif state.mode == "BROWSING":
                     context_text = "PREVIOUSLY FOUND:\n" + "\n".join([f"{i+1}. {h['title']}" for i, h in enumerate(state.search_results)])
                     system_instruction = "You are discussing the list of recipes found. Answer questions about which one sounds best."
                else:
                    system_instruction = "You are a chef. Ask the user what ingredients they have."

            
            full_prompt = [
                {'role': 'system', 'content': f"{system_instruction}\n\nCONTEXT DATA:\n{context_text}"}
            ]
            full_prompt.extend(state.chat_history[-4:]) 
            full_prompt.append({'role': 'user', 'content': user_input})

            print("ChefBot > ", end="", flush=True)
            stream = ollama.chat(model=GENERATION_MODEL_NAME, messages=full_prompt, stream=True)
            
            bot_reply = ""
            for chunk in stream:
                content = chunk['message']['content']
                print(content, end='', flush=True)
                bot_reply += content
            print("\n")

            state.chat_history.append({'role': 'user', 'content': user_input})
            state.chat_history.append({'role': 'assistant', 'content': bot_reply})

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    chat_loop()
