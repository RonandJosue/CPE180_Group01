import streamlit as st
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
import json
import os
import uuid
from datetime import datetime

# --- CONFIGURATION ---
DB_PATH = "C:/Users/JML/OneDrive/Documents/CPE AI Spec Proj/recipe_db"
COLLECTION_NAME = "recipes"
EMBEDDING_MODEL_NAME = 'nomic-embed-text'      
GENERATION_MODEL_NAME = 'llama-3.1-8b-instant' 
HISTORY_FILE = "chat_history.json"

api_key = os.environ.get("GROQ_API_KEY")

if not api_key:
    st.error("Groq API Key not found. Please set GROQ_API_KEY in the environment variables.")
    st.stop()

# We use the OpenAI library, but point it to Groq!
groq_client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)

# --- DB SETUP ---
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
        st.error(f"DB Error: {e}")
        return None

collection = get_collection()

st.set_page_config(page_title="ChefBot", page_icon=None, layout="wide")

# --- CHAT HISTORY HELPERS ---
def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f)

def save_current_session():
    history = load_history()
    
    title = "New Chat"
    if len(st.session_state.messages) > 1:
        first_user_msg = next((m["content"] for m in st.session_state.messages if m["role"] == "user"), "New Chat")
        title = first_user_msg[:30] + ("..." if len(first_user_msg) > 30 else "")
        
    history[st.session_state.session_id] = {
        "title": title,
        "messages": st.session_state.messages,
        "mode": st.session_state.mode,
        "search_results": st.session_state.search_results,
        "current_recipe": st.session_state.current_recipe,
        "updated_at": datetime.now().isoformat()
    }
    save_history(history)

def delete_history_item(session_to_delete):
    history = load_history()
    if session_to_delete in history:
        del history[session_to_delete]
        save_history(history)
    
    if st.session_state.session_id == session_to_delete:
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am ChefBot. Tell me what ingredients you have, or ask for a specific dish!"}
        ]
        st.session_state.mode = "IDLE"
        st.session_state.search_results = []
        st.session_state.current_recipe = None
        st.session_state.is_generating = False
        st.session_state.edit_mode = False

# --- SESSION STATE INITIALIZATION ---
if "session_id" not in st.session_state: st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am ChefBot. Tell me what ingredients you have, or ask for a specific dish!"}
    ]
if "mode" not in st.session_state: st.session_state.mode = "IDLE" 
if "search_results" not in st.session_state: st.session_state.search_results = []
if "current_recipe" not in st.session_state: st.session_state.current_recipe = None
if "edit_mode" not in st.session_state: st.session_state.edit_mode = False
if "last_user_text" not in st.session_state: st.session_state.last_user_text = ""
if "is_generating" not in st.session_state: st.session_state.is_generating = False

# --- SIDEBAR: CHAT HISTORY UI ---
with st.sidebar:
    st.title("ChefBot History")
    if st.button("New Chat", use_container_width=True, type="primary"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I am ChefBot. Tell me what ingredients you have, or ask for a specific dish!"}]
        st.session_state.mode = "IDLE"
        st.session_state.search_results = []
        st.session_state.current_recipe = None
        st.session_state.is_generating = False
        st.session_state.edit_mode = False
        st.rerun()
        
    st.divider()
    
    history = load_history()
    if history:
        sorted_sessions = sorted(history.items(), key=lambda x: x[1].get('updated_at', ''), reverse=True)
        
        for sess_id, sess_data in sorted_sessions:
            col_title, col_del = st.columns([0.85, 0.15])
            
            is_active = (sess_id == st.session_state.session_id)
            button_label = f"> {sess_data['title']}" if is_active else sess_data['title']
            
            with col_title:
                if st.button(button_label, key=f"btn_{sess_id}", use_container_width=True):
                    st.session_state.session_id = sess_id
                    st.session_state.messages = sess_data.get("messages", [])
                    st.session_state.mode = sess_data.get("mode", "IDLE")
                    st.session_state.search_results = sess_data.get("search_results", [])
                    st.session_state.current_recipe = sess_data.get("current_recipe", None)
                    st.session_state.is_generating = False
                    st.session_state.edit_mode = False
                    st.rerun()
            
            with col_del:
                if st.button("X", key=f"del_{sess_id}", help="Delete this chat"):
                    delete_history_item(sess_id)
                    st.rerun()

# --- MAIN UI STYLES ---
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
    
    /* Remove background from the overall chat row */
    [data-testid="stChatMessage"] {
        background-color: transparent !important;
    }

    /* Stop the invisible container from taking up 100% width */
    [data-testid="stChatMessageContent"] {
        flex-grow: 0 !important;
        width: fit-content !important;
    }

    /* --- USER MESSAGE STYLING --- */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        flex-direction: row-reverse;
    }
    
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
        background-color: #6c757d !important;
        border-radius: 18px 18px 0px 18px !important;
        padding: 10px 15px !important;
        margin-right: 10px;
        margin-left: auto;
        box-shadow: 0px 2px 4px rgba(0,0,0,0.1);
    }
    
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] * {
        color: white !important;
    }

    /* --- ASSISTANT MESSAGE STYLING --- */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] {
        background-color: var(--secondary-background-color) !important;
        border-radius: 18px 18px 18px 0px !important;
        padding: 10px 15px !important;
        margin-left: 10px;
        box-shadow: 0px 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

if not collection:
    st.error("Failed to connect to Database. Make sure Ollama is running.")
    st.stop()

# --- LOGIC HELPERS ---
def determine_intent(user_input):
    if user_input.lower() in ['exit', 'quit', 'q']: return 'EXIT'
    if st.session_state.mode == "IDLE": return "SEARCH"

    prompt = f"""
    You are an internal router for a recipe application.
    Current App Mode: {st.session_state.mode}
    Search Results Currently Displayed: {[r['title'] for r in st.session_state.search_results] if st.session_state.search_results else "None"}
    Currently Active Recipe: {st.session_state.current_recipe['title'] if st.session_state.current_recipe else "None"}
    
    User Input: "{user_input}"
    
    Your goal is to classify the User Input into one of three distinct actions:
    1. SEARCH: The user is stating ingredients they have, asking for dish ideas, or looking for new recipes (e.g., "I have chicken", "Find me a pasta dish", "What can I make with eggs and rice?").
    2. SELECT: The user is choosing one of the 'Search Results Currently Displayed' (e.g., "Number 1", "The beef stew", "Show me the second one").
    3. CHAT: The user is asking a specific question about the 'Currently Active Recipe', or just making general conversation (e.g., "How long do I bake it?", "Can I use butter instead of oil?", "Thank you!").
    
    Reply ONLY with the exact word SEARCH, SELECT, or CHAT. Do not include any other text or punctuation.
    """
    try:
        response = groq_client.chat.completions.create(
            model=GENERATION_MODEL_NAME, 
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.0
        )
        intent = response.choices[0].message.content.strip().upper()
        if "SEARCH" in intent: return "SEARCH"
        if "SELECT" in intent: return "SELECT"
        return "CHAT"
    except Exception as e: 
        print(f"Intent Error: {e}")
        return "CHAT"

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
    except Exception as e:
        print(f"Search Error: {e}") 
        return []

# --- RENDER MESSAGES ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- EDIT BUTTON LOGIC ---
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

# --- INPUT AND GENERATION ---
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
            save_current_session()
            st.rerun()
        if cancel_edit:
            st.session_state.messages.append({"role": "user", "content": st.session_state.last_user_text})
            st.session_state.edit_mode = False
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.is_generating:
    st.markdown('<div class="fixed-bottom">', unsafe_allow_html=True)
    if st.button("STOP GENERATING", use_container_width=True, type="primary"):
        st.session_state.is_generating = False
        st.warning("Generation stopped.")
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

else:
    if prompt := st.chat_input("What are we cooking?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.is_generating = True
        save_current_session()
        st.rerun()

# --- ASSISTANT RESPONSE ---
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
                    "You are ChefBot, a helpful AI chef. The user just searched for recipes based on their ingredients. "
                    "Look at the FOUND RECIPES in the context data below. "
                    "Present a warm, brief summary of these options, and ask the user which number they would like to cook. "
                    "Do not list out the full recipes yet, just the names or options."
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
            resp = groq_client.chat.completions.create(
                model=GENERATION_MODEL_NAME, 
                messages=[{'role': 'user', 'content': match_prompt}],
                temperature=0.0
            )
            try: idx = int(''.join(filter(lambda x: x.isdigit() or x == '-', resp.choices[0].message.content))) - 1
            except: idx = -1

            if 0 <= idx < len(st.session_state.search_results):
                st.session_state.current_recipe = st.session_state.search_results[idx]
                st.session_state.mode = "COOKING"
                
                recipe = st.session_state.current_recipe
                context_data = recipe['full_text']
                
                system_msg = (
                    "You are ChefBot, a helpful AI chef. The user has selected a recipe from the list. "
                    "Using ONLY the context data below, present the recipe cleanly. "
                    "Structure it well with ingredients and step-by-step instructions. "
                    "Make sure to include the Source and Link if provided in the context data. "
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
                    "You are ChefBot, guiding the user through the recipe provided in the context data. "
                    "Answer their specific questions about ingredients, substitutions, cooking times, or methods based ONLY on the provided recipe text. "
                    "If the answer is not in the recipe, use your general culinary knowledge but clarify that it is an addition to the recipe."
                )
            elif st.session_state.mode == "BROWSING":
                context_data = "PREVIOUSLY FOUND:\n" + "\n".join([f"{i+1}. {h['title']}" for i, h in enumerate(st.session_state.search_results)])
                system_msg = "You are discussing the list of recipes found. Answer questions about which one sounds best."
            else:
                system_msg = "You are ChefBot. Ask the user what ingredients they have or what they want to cook."

        # Assemble the full prompt history cleanly
        full_prompt = [{'role': 'system', 'content': f"{system_msg}\n\nCONTEXT DATA:\n{context_data}"}]
        
        # Append the last 4 messages plus the current prompt to give Groq proper context
        full_prompt.extend([m for m in st.session_state.messages[-5:] if m['role'] != 'system'])
        
        def stream_parser():
            stream = groq_client.chat.completions.create(
                model=GENERATION_MODEL_NAME, 
                messages=full_prompt, 
                stream=True
            )
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content

        response = st.write_stream(stream_parser())
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.is_generating = False
    save_current_session()
    st.rerun()
