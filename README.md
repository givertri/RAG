# 🍽️ RAG Recipe Assistant

A Retrieval-Augmented Generation (RAG) application for generating recipe suggestions and meal plans. The application uses a Milvus vector database, a hybrid retriever and LLM generation via Ollama.

This code is part of the master's thesis by Gilles Vertriest for the advanced master of Artificial Intelligence at KU Leuven: "Constraint-Aware RAG for Recipe Suggestion and Meal Planning".

---

## Directory Structure

```text
RAG/
├── analysis/              # Data analysis notebooks
├── components/            # Application components: generators, indexers, data loader & retrievers
├── core/                  # Component base classes & RAG pipelines
├── dataset/               # Original dataset, cleaned dataset & preprocessing notebooks
├── evaluation/            # Evaluation scripts
│   └── test_sets/         # Test queries for evaluation
├── preprocessing/         # Qwen preprocessors: query splitter & JSON extractors
├── docker-compose.yml     # Milvus Docker Compose file
├── main.py                # Single-recipe application entry point
├── main_planning.py       # Multi-recipe application entry point
├── populate_db.py         # Database population script
├── requirements.txt       # Python dependencies
└── setup_ollama.sh        # Ollama setup script
```
---

## Installation & Setup

### 1. Clone the Repository

Get the latest version of the project:

```bash
git clone https://github.com/givertri/RAG
cd RAG
```

#### Install dependencies

```bash
pip install -r requirements.txt
```

---

### 2. Start and Populate Milvus

This project uses **Milvus** as vector database. 

#### Start Milvus

```bash
docker compose up -d
```

This creates a directory called ``volumes`` for data persistance when run for the first time.

#### Populate the Database

```bash
python -m populate_db
```

---

### 3. Install Ollama (Linux)

#### Windows
Follow the official installation guide: [Ollama download](https://ollama.com/download/windows)

Download Required Models:

```bash
ollama pull qwen3:4b
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

#### Linux

Use the setup script:

```bash
bash setup_ollama.sh
```

---

### 4. Set Environment Variable

Create a file in the root directory called ``env.env`` and set the environment variable for the URL where Ollama is running.

For example:
```python
RUNPOD_URL=http://localhost:11434
```

This file can also be used to store an API key for an external LLM if needed.

---

## Running the RAG Pipeline

### Single-Recipe RAG

Run the application with a custom prompt/query:

```bash
python main.py "Give me a dinner dish with tomatoes."
```

#### Optional: Choose Retriever Type

* **Hybrid** → combines structured + semantic retrieval and filtering
* **Regular** → semantic-only retrieval

```bash
python main.py "Give me a quick pasta recipe." --retriever regular
```

#### Default Behavior

Running the application without arguments is equivalent to:

```bash
python main.py "Give me a tasty dish." --retriever regular
```

### Multi-Recipe RAG

Run the application with a custom prompt/query:

```bash
python main_planning.py "Give me a breakfast recipe with eggs and a dinner dish with tomatoes."
```

#### Default Behavior

Running the application without arguments is equivalent to:

```bash
python main_planning.py "Give me a tasty dish."
```

---

## Components

* **Indexer**: Stores and indexes Nomic & BM25 embeddings
* **Preprocessors:**:
  * `split_query` → splits one query in multiple subqueries (for multi-recipe RAG)
  * `text_to_json` → extracts constraints in query to structured JSON constraints
* **Retrievers**:
  * `LangchainRetriever` → semantic search
  * `LangchainRetrieverHybrid` → keyword search + semantic search + filtering
* **Generator**: Uses Llama 3.2 for generation with retrieved context

---

## Batch Evaluation

### Constraint-Aware RAG only
```bash
python -m evaluation.evaluation_short
```
### All systems: LLM-only, Standard RAG, Constraint-Aware RAG
```bash
python -m evaluation.multi-evaluation
```
### Multi-Recipe RAG only
```bash
python -m evaluation.multi-evaluation_planning
```
### Edge case evaluation
```bash
python -m evaluation.multi-evaluation_edge_cases
```
