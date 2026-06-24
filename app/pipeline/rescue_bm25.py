import pickle
import os
from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from core import config

print("Connecting to ChromaDB...")

# 1. Connect to your existing ChromaDB
device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
embeddings = HuggingFaceEmbeddings(
    model_name=config.MODEL_NAME,
    model_kwargs={"trust_remote_code": True, "device": device}
)

vector_store = Chroma(
    persist_directory=config.CHROMA_DB_DIR,
    embedding_function=embeddings
)

# 2. Fetch EVERYTHING currently inside ChromaDB
print("Fetching existing chunks from ChromaDB...")
db_contents = vector_store.get() # .get() with no arguments returns everything

documents_text = db_contents.get("documents", [])
metadatas = db_contents.get("metadatas", [])

if not documents_text:
    print("ChromaDB is empty! You will need to run the PDF processor again.")
    exit()

# 3. Reconstruct the LangChain Document objects that BM25 expects
print(f"Found {len(documents_text)} chunks. Reconstructing for BM25...")
bmdoc = []
for text, meta in zip(documents_text, metadatas):
    bmdoc.append(Document(page_content=text, metadata=meta))

# 4. Save to the pickle file
bm25_store_path = Path(config.CHROMA_DB_DIR) / "bm25_chunks.pkl"

with open(bm25_store_path, "wb") as f:
    pickle.dump(bmdoc, f)

print(f"\n🎉 Success! Extracted {len(bmdoc)} chunks from Chroma and saved to {bm25_store_path}.")