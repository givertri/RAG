# 🍽️ RAG Recipe Assistant

A Retrieval-Augmented Generation (RAG) pipeline for generating recipe suggestions using local embeddings, hybrid retrieval, and LLM generation via Ollama.

---

## 📦 Installation & Setup

### 1. Clone the Repository

Get the latest version of the project:

```bash
git clone https://github.com/givertri/RAG
cd RAG
```

---

### 2. Start and Populate Milvus

This project uses **Milvus** as the vector database.

#### Start Milvus with Docker Compose

```bash
docker compose up -d
```

#### Populate the Database

```bash
python populate_db.py
```

---

### 3. Install Ollama (Linux)

Follow the official installation:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

#### Download Required Models

```bash
ollama pull qwen3:4b
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

---

## 🚀 Running the RAG Pipeline

Run the main script with a custom prompt:

```bash
python main.py "Give me a spicy vegan dish"
```

### Optional: Choose Retriever Type

* **Hybrid (default)** → combines structured + semantic retrieval
* **Regular** → semantic-only retrieval

```bash
python main.py "Give me a quick pasta recipe" --retriever regular
```

---

## 🧠 How It Works

### Pipeline Components

* **Indexer**: Uses Nomic & BM25 embeddings to connect to Milvus
* **Retriever**:
  * `LangchainRetriever` → basic vector similarity
  * `LangchainRetrieverHybrid` → keyword search + semantic search + filtering
* **Generator**: Uses Llama 3.2 for generation with retrieved context
* **Pipeline**: Combines retrieval + generation

---

## 🛠️ Code Entry Point

```python
python main.py "Your prompt here"
```

### Default Behavior

If no prompt is provided:

```bash
python main.py
```

It will run with:

```
"Give me a tasty dish."
```

---

## ⚙️ Arguments

| Argument      | Description                    | Default  |
| ------------- | ------------------------------ | -------- |
| `prompt`      | Input query for the RAG system | Optional |
| `--retriever` | `regular` or `hybrid`          | hybrid   |

---

## 🔄 Example Output Flow

1. Prompt is parsed
2. (Hybrid only) Prompt is converted to structured JSON constraints
3. Relevant documents retrieved from Milvus
4. Generator produces final answer
