from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_INDEX_HOST = os.environ.get('PINECONE_INDEX_HOST')

print("Loading PDF documents...")
extracted_data = load_pdf("data/")

print("Creating text chunks...")
text_chunks = text_split(extracted_data)

print("Downloading embeddings model...")
embeddings = download_hugging_face_embeddings()

# Initialize Pinecone with the new Serverless pattern
print("Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)

# Connect to existing index using host URL
index_name = "chat"

print(f"Connecting to index '{index_name}'...")
index = pc.Index(
    name=index_name,
    host=PINECONE_INDEX_HOST
)

print("Storing documents in Pinecone...")
# Create vector store and add documents
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name,
    pinecone_api_key=PINECONE_API_KEY
)

print("Documents successfully stored in Pinecone index!")
