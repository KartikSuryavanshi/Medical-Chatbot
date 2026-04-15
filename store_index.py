from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
FAISS_INDEX_PATH = os.environ.get('FAISS_INDEX_PATH', str(BASE_DIR / 'faiss_index'))
MAX_PAGES_PER_PDF = int(os.environ.get('MAX_PAGES_PER_PDF', '250'))

print("Loading PDF documents...")
extracted_data = load_pdf(str(DATA_DIR), max_pages_per_pdf=MAX_PAGES_PER_PDF)

if not extracted_data:
	raise ValueError("No PDF pages were loaded from the data directory. Add valid PDF files and retry.")

print("Creating text chunks...")
text_chunks = text_split(extracted_data)

print("Downloading embeddings model...")
embeddings = download_hugging_face_embeddings()

print("Creating local FAISS index...")
docsearch = FAISS.from_documents(text_chunks, embeddings)

print(f"Saving FAISS index to '{FAISS_INDEX_PATH}'...")
docsearch.save_local(FAISS_INDEX_PATH)

print("Documents successfully stored in local FAISS index!")
