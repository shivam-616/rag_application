import pickle
from pathlib import Path

from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from core import config


def load_bm25_retriever(k=20):
    bm25_store_path = Path(config.CHROMA_DB_DIR) / "bm25_chunks.pkl"

    if not bm25_store_path.exists():
        raise FileNotFoundError(
            f"No BM25 chunk store found at {bm25_store_path}. "
            f"Run ingestion.py first."
        )

    with open(bm25_store_path, "rb") as f:
        chunks = pickle.load(f)

    print(f"Loaded {len(chunks)} chunks for BM25 index")

    bm25_retriever = BM25Retriever.from_documents(chunks, k=k)
    return bm25_retriever


def load_vector_retriever(k: int = 20):
    embeddings = HuggingFaceEmbeddings(
        model_name=config.MODEL_NAME,
        model_kwargs={"trust_remote_code": True}
    )

    vector_store = Chroma(
        persist_directory=config.CHROMA_DB_DIR,
        embedding_function=embeddings
    )

    vector_retriever = vector_store.as_retriever(
        search_kwargs={"k": k}
    )
    return vector_retriever


if __name__ == "__main__":
    bm25 = load_bm25_retriever(k=5)
    vector = load_vector_retriever(k=5)

    query = "  sex in life"

    print("=== BM25 results ===")
    for r in bm25.invoke(query):
        print(r.metadata.get("paper_id"), "-", r.page_content[:80])

    print("\n=== Vector results ===")
    for r in vector.invoke(query):
        print(r.metadata.get("paper_id"), "-", r.page_content[:80])
