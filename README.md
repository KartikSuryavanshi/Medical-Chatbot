# 🏥 End-to-End Medical Chatbot using Llama 2

An intelligent medical chatbot powered by Meta's Llama 2 language model, LangChain, and Pinecone vector database. The chatbot provides medical information based on your PDF documents, with a modern, responsive web interface.

## 🚀 How to Run

### Prerequisites
- Python 3.9+
- Pinecone account with Serverless index
- Llama 2 model file

### STEP 01 - Clone the Repository

```bash
git clone https://github.com/KartikSuryavanshi/Medical-Chatbot.git
cd Medical-Chatbot
```

### STEP 02 - Create a Conda Environment

```bash
conda create -n mchatbot python=3.9 -y
conda activate mchatbot
```

### STEP 03 - Install Requirements

```bash
pip install -r requirements.txt
```

### STEP 04 - Configure Environment Variables

Create a `.env` file in the root directory and add your Pinecone credentials:

```ini
PINECONE_API_KEY="your-pinecone-api-key-here"
PINECONE_INDEX_HOST="https://your-index-name.svc.aped-xxxx-xxxx.pinecone.io"
```

**Important Notes:**
- Get your `PINECONE_API_KEY` from [Pinecone Dashboard → API Keys](https://app.pinecone.io)
- Get your `PINECONE_INDEX_HOST` from your Pinecone index details page
- This project uses **Pinecone Serverless** (new version)
- `PINECONE_API_ENV` is **deprecated** and not needed

### STEP 05 - Download the Llama 2 Model

Download the quantized Llama 2 model and place it in the `model/` directory:

**Model:** `llama-2-7b-chat.ggmlv3.q4_0.bin`

**Download Link:** [Llama-2-7B-Chat-GGML on Hugging Face](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main)

```bash
# Ensure the model is placed at:
model/llama-2-7b-chat.ggmlv3.q4_0.bin
```

### STEP 06 - Create Pinecone Index

Create a Pinecone Serverless index with these specifications:
- **Index Name:** `chat`
- **Dimension:** `1024`
- **Metric:** `cosine`
- **Cloud:** AWS
- **Region:** us-east-1

### STEP 07 - Store PDF Documents in Vector Database

Place your medical PDF documents in the `data/` directory, then run:

```bash
python store_index.py
```

This will:
- Load PDF documents from `data/`
- Split them into chunks
- Generate embeddings using `BAAI/bge-large-en-v1.5` (1024 dimensions)
- Store vectors in your Pinecone index

### STEP 08 - Run the Application

```bash
python app.py
```

The Flask server will start on `http://localhost:8080`

Open your browser and navigate to:
```
http://localhost:8080
```

## 🛠️ Tech Stack

- **Python 3.9**
- **LangChain** - Framework for LLM applications
- **LangChain Community** - Community integrations
- **Flask** - Web framework
- **Meta Llama 2** - Large Language Model
- **Pinecone Serverless** - Vector database
- **Sentence Transformers** - Embeddings (BAAI/bge-large-en-v1.5)
- **CTransformers** - Efficient inference for quantized models
- **Bootstrap 5** - Frontend UI framework

## 📁 Project Structure

```
Medical-Chatbot-using-Llama/
├── app.py                 # Flask application
├── store_index.py         # Script to index PDF documents
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (create this)
├── model/
│   └── llama-2-7b-chat.ggmlv3.q4_0.bin  # Llama 2 model (download)
├── data/                  # Place your PDF documents here
├── src/
│   ├── helper.py         # Helper functions for PDF loading & embeddings
│   └── prompt.py         # Prompt templates
├── templates/
│   └── chat.html         # Frontend chat interface
└── static/
    └── style.css         # Custom styles
```

## 🔧 Configuration Notes

### Embedding Model
The project uses **BAAI/bge-large-en-v1.5** which produces 1024-dimensional embeddings to match the Pinecone index dimension.

### Pinecone Serverless
This project uses the **new Pinecone Serverless** architecture. Key differences:
- No `pinecone.init()` - use `Pinecone(api_key=...)` instead
- Connect via index host URL, not environment name
- More cost-effective and scalable

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## ⭐ Show Your Support

Give a ⭐️ if this project helped you!
