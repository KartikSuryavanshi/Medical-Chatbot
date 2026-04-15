from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
from dotenv import load_dotenv
from src.prompt import *
import os
from pathlib import Path

app = Flask(__name__)

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
FAISS_INDEX_PATH = os.environ.get('FAISS_INDEX_PATH', str(BASE_DIR / 'faiss_index'))
OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL', 'llama3.2:3b')
OLLAMA_BASE_URL = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')

# Download embeddings model
embeddings = download_hugging_face_embeddings()

# Load local FAISS index as vector store
if not os.path.exists(FAISS_INDEX_PATH):
    raise FileNotFoundError(
        f"FAISS index not found at '{FAISS_INDEX_PATH}'. Run 'python store_index.py' first."
    )

docsearch = FAISS.load_local(
    FAISS_INDEX_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = docsearch.as_retriever(search_kwargs={'k': 2})

# Create prompt template
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Load the LLM via Ollama (local inference, no model file download required)
llm = OllamaLLM(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0.8,
    num_predict=512,
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Build RAG chain using LCEL
qa_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | PROMPT
    | llm
    | StrOutputParser()
)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print(f"User question: {msg}")
    result = qa_chain.invoke(msg)
    print(f"Response: {result}")
    return str(result)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)


# Create the QA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(f"User question: {input}")
    result = qa({"query": input})
    print(f"Response: {result['result']}")
    return str(result["result"])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
