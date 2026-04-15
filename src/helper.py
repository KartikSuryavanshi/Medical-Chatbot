from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from pathlib import Path


#Extract data from the PDF
def load_pdf(data, max_pages_per_pdf=250):
    data_path = Path(data)
    documents = []

    # Parse PDF files lazily and cap pages to avoid long indexing times on very large books.
    for pdf_file in sorted(data_path.glob("*.pdf")):
        loader = PyPDFLoader(str(pdf_file))
        for i, page in enumerate(loader.lazy_load()):
            if max_pages_per_pdf and i >= max_pages_per_pdf:
                break
            documents.append(page)

    return documents



#Create text chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 20)
    text_chunks = text_splitter.split_documents(extracted_data)

    return text_chunks



#download embedding model
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings