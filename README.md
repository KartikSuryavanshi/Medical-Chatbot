# Medical Chatbot — Fully Local RAG Application

An intelligent medical chatbot powered by **Llama 3.2 (via Ollama)**, **LangChain**, and a **local FAISS vector database**. No cloud accounts, no API keys, no model downloads — everything runs on your own machine.

The chatbot answers medical questions by retrieving relevant context from your own PDF documents (e.g. a medical textbook) and passing it to a local LLM to generate accurate, grounded answers.


## Tech Stack

| Layer | Technology |
|---|---|
| **LLM (inference)** | Llama 3.2 3B via [Ollama](https://ollama.com) — runs fully locally |
| **Vector Database** | FAISS (local, saved to disk — no cloud required) |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace |
| **RAG Framework** | LangChain (LCEL — LangChain Expression Language) |
| **Web Framework** | Flask |
| **Frontend** | HTML + CSS + jQuery (custom interactive dark-theme UI) |
| **PDF Loading** | PyPDF via LangChain Community loaders |


## Project Structure

```
Medical-Chatbot/
├── app.py                 # Flask server, RAG chain, API routes
├── store_index.py         # One-time script: reads PDFs → builds FAISS index
├── requirements.txt       # Python dependencies
├── src/
│   ├── helper.py          # PDF loader, text splitter, embeddings model
│   └── prompt.py          # Prompt template for the LLM
├── data/                  # Place your medical PDF(s) here
├── faiss_index/           # Auto-generated local vector index
│   ├── index.faiss
│   └── index.pkl
├── templates/
│   └── chat.html          # Chat UI
├── static/
│   └── style.css          # Dark theme UI styles
└── research/
    └── trials.ipynb       # Experimentation notebook
```


## How to Run

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com) installed

### STEP 01 — Clone the Repository

```bash
git clone https://github.com/KartikSuryavanshi/Medical-Chatbot.git
cd Medical-Chatbot
```

### STEP 02 — Start Ollama and Pull the Model

```bash
# Start the Ollama server (keep this running in a separate terminal)
ollama serve

# Pull the Llama 3.2 3B model (one-time download, ~2 GB)
ollama pull llama3.2:3b
```

### STEP 03 — Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### STEP 04 — Install Requirements

```bash
pip install -r requirements.txt
```

### STEP 05 — Add Your PDF Documents

Place your medical PDF files inside the `data/` directory:

```
data/
└── Medical_book.pdf
```

### STEP 06 — Build the Local FAISS Index

This reads your PDFs, generates embeddings, and saves the vector index locally.
Run this **once** (or again whenever you add/change PDFs):

```bash
# Index first 100 pages (faster for testing)
MAX_PAGES_PER_PDF=100 python store_index.py

# Or index all pages
python store_index.py
```

What this does:
- Loads PDFs from `data/` using `PyPDFLoader`
- Splits text into 500-token chunks (20-token overlap)
- Embeds chunks using `all-MiniLM-L6-v2` (~90 MB, auto-downloaded on first run)
- Saves the FAISS index to `faiss_index/` on disk

### STEP 07 — Run the Application

```bash
python app.py
```

Open your browser at:
```
http://127.0.0.1:8080
```

---

## How It Works (RAG Pipeline)

```
User Question
     │
     ▼
FAISS Retriever  ──→  Top-2 relevant chunks from your PDF
     │
     ▼
Prompt Template  ──→  "Use this context to answer: {context} ... {question}"
     │
     ▼
Ollama LLM (Llama 3.2 3B)  ──→  Generates grounded, context-aware answer
     │
     ▼
Flask API  ──→  Rendered in chat UI with typewriter effect
```

## UI Features

- Quick-question chips for instant demo questions (diabetes, blood pressure, cholesterol, etc.)
- Typing indicator while the bot is thinking
- Typewriter animation on bot responses
- Copy-to-clipboard button on every response
- Clear chat button
- Character counter in the input box
- Scroll-to-bottom floating button


## Configuration

The application works out of the box with no `.env` file needed.
You can optionally adjust these defaults:

| Variable | Default | Description |
|---|---|---|
| `FAISS_INDEX_PATH` | `faiss_index` | Path to the saved FAISS index directory |
| `OLLAMA_MODEL` | `llama3.2:3b` | Ollama model name to use for inference |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `MAX_PAGES_PER_PDF` | unlimited | Cap pages loaded per PDF during indexing |
