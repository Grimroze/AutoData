import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# Using FastEmbed for local embeddings without API keys
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from src.config import CHUNK_SIZE, CHUNK_OVERLAP, CHROMA_DB_DIR, DATA_DIR

def get_vector_store():
    embeddings = FastEmbedEmbeddings()
    
    if os.path.exists(CHROMA_DB_DIR) and os.listdir(CHROMA_DB_DIR):
        print("Loading existing ChromaDB...")
        return Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
        
    print(f"Creating new ChromaDB in {CHROMA_DB_DIR}...")
    documents = []
    
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created {DATA_DIR}. Please add text/pdf files and restart.")
        return None
        
    for file in os.listdir(DATA_DIR):
        file_path = os.path.join(DATA_DIR, file)
        if file.endswith(".txt"):
            loader = TextLoader(file_path, encoding = 'utf-8')
            documents.extend(loader.load())
        elif file.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
            
    if not documents:
        print(f"No documents found in {DATA_DIR}.")
        return None
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs = text_splitter.split_documents(documents)
    
    db = Chroma.from_documents(docs, embeddings, persist_directory=CHROMA_DB_DIR)
    print(f"Successfully chunked and saved {len(docs)} documents to ChromaDB.")
    return db
