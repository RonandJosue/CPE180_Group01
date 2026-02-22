import streamlit as st
import ollama
import chromadb
from chromadb.utils import embedding_functions

DB_PATH = "C:/Users/JML/OneDrive/Documents/CPE AI Spec Proj/recipe_db"
COLLECTION_NAME = "recipes"
EMBEDDING_MODEL_NAME = 'nomic-embed-text'
GENERATION_MODEL_NAME = 'llama3.1'

@st.cache_resource
def get_collection():
    try:
        client = chromadb.PersistentClient(path=DB_PATH)
        embedding_func = embedding_functions.OllamaEmbeddingFunction(
            url="http://localhost:11434/api/embeddings",
            model_name=EMBEDDING_MODEL_NAME
        )
        return client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_func)
    except Exception as e:
        return None

collection = get_collection()

st.set_page_config(page_title="ChefBot", page_icon=None)
st.title("ChefBot")

st.markdown("""
    <style>
    .fixed-bottom {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: var(--background-color);
        padding: 20px 3rem;
        z-index: 9999;
        border-top: 1px solid var(--secondary-background-color);
    }
    .block-container {
        padding-bottom: 120px;
    }
    </style>
""", unsafe_allow_html=True)

if not collection:
    st.error("Failed to connect to Database. Make sure ingest.py ran successfully.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm ChefBot. Tell me what ingredients you have, or ask for a specific dish!"}
    ]

if "mode" not in st.session_state: st.session_state.mode = "IDLE" 
if "search_results" not in st.session_state: st.session_state.search_results = []
if "current_recipe" not in st.session_state: st.session_state.current_recipe = None
if "edit_mode" not in st.session_state: st.session_state.edit_mode = False
if "last_user_text" not in st.session_state: st.session_state.last_user_text = ""
if "is_generating" not in st.session_state: st.session_state.is_generating = False

def determine_intent(user_input):
    if user_input.lower() in ['exit', 'quit', 'q']: return 'EXIT'
    if st.session_state.mode == "IDLE": return "SEARCH"

    prompt = f"""
    Current Mode: {st.session_state.mode}
    Available Search Results: {[r['title'] for r in st.session_state.search_results] if st.session_state.search_results else "None"}
    Active Recipe: {st.session_state.current_recipe['title'] if st.session_state.current_recipe else "None"}
    
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
    except: return "CHAT"

def search_db(query):
    try:
        results = collection.query(query_texts=[query], n_results=10)
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
    except: return []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

with st.container():
    if not st.session_state.is_generating and not st.session_state.edit_mode and len(st.session_state.messages) > 1:
        col_spacer, col_btn = st.columns([0.85, 0.15])
        with col_btn:
            if st.button("Edit", help="Edit your last message"):
                if st.session_state.messages[-1]["role"] == "assistant":
                    st.session_state.messages.pop()
                if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
                    last_msg = st.session_state.messages.pop()
                    st.session_state.last_user_text = last_msg["content"]
                    st.session_state.edit_mode = True
                    st.rerun()

if st.session_state.edit_mode:
    st.markdown('<div class="fixed-bottom">', unsafe_allow_html=True)
    with st.form("edit_form"):
        st.write("Edit your message:")
        new_prompt = st.text_area("Prompt", value=st.session_state.last_user_text, label_visibility="collapsed")
        col_a, col_b = st.columns([0.2, 0.8])
        with col_a:
            submit_edit = st.form_submit_button("Resend", type="primary")
        with col_b:
            cancel_edit = st.form_submit_button("Cancel")
        
        if submit_edit:
            st.session_state.messages.append({"role": "user", "content": new_prompt})
            st.session_state.edit_mode = False
            st.session_state.is_generating = True
            st.rerun()
        if cancel_edit:
            st.session_state.messages.append({"role": "user", "content": st.session_state.last_user_text})
            st.session_state.edit_mode = False
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.is_generating:
    st.markdown('<div class="fixed-bottom">', unsafe_allow_html=True)
    if st.button("⏹ STOP GENERATING", use_container_width=True, type="primary"):
        st.session_state.is_generating = False
        st.warning("Generation stopped.")
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

else:
    if prompt := st.chat_input("What are we cooking?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.is_generating = True
        st.rerun()

if st.session_state.is_generating:
    
    with st.chat_message("assistant"):
        last_prompt = st.session_state.messages[-1]["content"]
        intent = determine_intent(last_prompt)
        
        system_msg = ""
        context_data = ""

        if intent == "SEARCH":
            hits = search_db(last_prompt)
            if hits:
                st.session_state.search_results = hits
                st.session_state.mode = "BROWSING"
                st.session_state.current_recipe = None
                
                context_data = "FOUND RECIPES:\n" + "\n".join([f"{i+1}. {h['title']} (Source: {h['source']})" for i, h in enumerate(hits)])
                
                system_msg = (
                    "You are a helpful chef. The user just searched for recipes. "
                    "Briefly list the recipes found in the context below. "
                    "Ask the user which one they would like to cook."
                )
            else:
                system_msg = "I couldn't find any recipes matching that. Try different ingredients."

        elif intent == "SELECT":
            match_prompt = f"""
            User Input: "{last_prompt}"
            Options:
            {[f"{i+1}. {r['title']}" for i, r in enumerate(st.session_state.search_results)]}
            
            Which index (1-10) is the user trying to select? 
            Reply ONLY with the number (e.g. "1"). If unsure, reply "0".
            """
            resp = ollama.chat(model=GENERATION_MODEL_NAME, messages=[{'role': 'user', 'content': match_prompt}])
            try: idx = int(''.join(filter(lambda x: x.isdigit() or x == '-', resp['message']['content']))) - 1
            except: idx = -1

            if 0 <= idx < len(st.session_state.search_results):
                st.session_state.current_recipe = st.session_state.search_results[idx]
                st.session_state.mode = "COOKING"
                
                recipe = st.session_state.current_recipe
                context_data = recipe['full_text']
                
                system_msg = (
                    "You are a helpful chef. The user has selected a specific recipe. "
                    "Present the recipe details nicely. "
                    "Include the Source and Link if available. "
                    f"Link: {recipe['link']}\n"
                    f"Source: {recipe['source']}"
                )
            else:
                system_msg = "The user tried to select a recipe but it was unclear. Ask them to clarify which number or name they want."
                context_data = "Previous Options:\n" + "\n".join([f"{i+1}. {h['title']}" for i, h in enumerate(st.session_state.search_results)])

        elif intent == "CHAT":
            if st.session_state.mode == "COOKING" and st.session_state.current_recipe:
                context_data = st.session_state.current_recipe['full_text']
                system_msg = (
                    "You are a chef guiding the user through the recipe below. "
                    "Answer their specific questions (e.g. substitutions, times, temps) based ONLY on the text."
                )
            elif st.session_state.mode == "BROWSING":
                context_data = "PREVIOUSLY FOUND:\n" + "\n".join([f"{i+1}. {h['title']}" for i, h in enumerate(st.session_state.search_results)])
                system_msg = "You are discussing the list of recipes found. Answer questions about which one sounds best."
            else:
                system_msg = "You are a chef. Ask the user what ingredients they have."

        full_prompt = [{'role': 'system', 'content': f"{system_msg}\n\nCONTEXT DATA:\n{context_data}"}]
        full_prompt.extend([m for m in st.session_state.messages[-5:] if m['role'] != 'system'])
        
        stream = ollama.chat(model=GENERATION_MODEL_NAME, messages=full_prompt, stream=True)
        response = st.write_stream(chunk['message']['content'] for chunk in stream)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.is_generating = False
    st.rerun()
