# ChefBot

ChefBot is an AI-powered recipe assistant that helps users discover recipes based on the ingredients they currently have available. The system uses a retrieval-augmented generation (RAG) approach that combines semantic search with a large language model to provide relevant recipe recommendations and cooking guidance through a conversational interface.

The application retrieves relevant recipes from a vector database using embedding similarity and then uses a language model to generate natural language responses for recipe suggestions, cooking steps, and ingredient substitutions.

---

## Course Information

Course: Cloud AI and MLOps with Huawei Cloud ModelArts, MindSpore, and DevOps  
Instructor: Engr. Dionis Padilla

---

## Project Members

- John Mark Lopez  
- Ronand Alaric Josue

---

## Dataset

This project uses the **RecipeNLG dataset**, which contains a large collection of structured recipe data including titles, ingredients, and cooking instructions.

Dataset source:  
https://www.kaggle.com/datasets/saldenisov/recipenlg

---

## Features

- Ingredient-based recipe search  
- Semantic search using vector embeddings  
- Conversational AI cooking assistant  
- Step-by-step recipe presentation  
- Chat history management  
- Interactive web interface using Streamlit  

---

## Technology Stack

Programming Language:
- Python

Libraries and Tools:
- Streamlit
- ChromaDB
- Ollama
- Pandas
- tqdm
- OpenAI Python SDK

Models:
- nomic-embed-text (embedding model via Ollama)  
- llama-3.1-8b-instant (generation model via Groq API)

---

## Project Structure

```
ChefBot/
├── ingest_recipes.py
├── app.py
├── full_dataset.csv
├── recipe_db/
├── chat_history.json
├── requirements.txt
└── README.md
```

---

## Requirements

- Python 3.9 or newer  
- Ollama installed locally  
- Groq API key  

Python packages:
- streamlit
- chromadb
- pandas
- tqdm
- ollama
- openai

---

## Installation

Clone the repository:

```
git clone https://github.com/YOUR_USERNAME/chefbot.git
cd chefbot
```

Create a virtual environment:

```
python -m venv venv
```

Activate the environment:

Windows
```
venv\Scripts\activate
```

Linux / macOS
```
source venv/bin/activate
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## Setup

Install Ollama and download the embedding model:

```
ollama pull nomic-embed-text
```

Set your Groq API key.

Windows:
```
set GROQ_API_KEY=your_api_key
```

Linux / macOS:
```
export GROQ_API_KEY=your_api_key
```

---

## Database Ingestion

Before running the application, ingest the recipe dataset into the vector database:

```
python ingest_recipes.py
```

---

## Running the Application

Start the Streamlit interface:

```
streamlit run app.py
```

The application will open in your browser where you can interact with ChefBot.

---

## Hugging Face Link

```
https://mycocacc4th-chefbot.hf.space/
```
