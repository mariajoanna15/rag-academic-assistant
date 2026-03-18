import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

DATA_PATH = "data"
DB_PATH = "vectorstore"

def load_documents():
    documents = []
    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DATA_PATH, file))
            docs = loader.load()
            
            # ✅ add metadata (IMPORTANT for resume)
            for doc in docs:
                doc.metadata["source"] = file
            
            documents.extend(docs)
    return documents

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_documents(documents)

def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(DB_PATH)

if __name__ == "__main__":
    print("Loading documents...")
    docs = load_documents()

    print(f"Loaded {len(docs)} documents")

    print("Splitting documents...")
    chunks = split_documents(docs)

    print(f"Created {len(chunks)} chunks")

    print("Creating vector store...")
    create_vectorstore(chunks)

    print("✅ Done! Vector DB saved.")