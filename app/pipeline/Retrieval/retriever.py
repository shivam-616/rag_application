import pickle
from itertools import islice
from pathlib import Path
from pydoc import Doc
from xml.dom.minidom import Document

from joblib.externals.loky.backend import context
from langchain_community.retrievers import BM25Retriever
from promt.rag_promt import rag_prompt
from rank_bm25 import BM25Okapi
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from pandas.core.window import doc
from sympy.physics.units import temperature

from core import config
from promt import rag_promt


def load_bm25_retriever(m=20):
    bm25_store_path = Path(config.CHROMA_DB_DIR) / "bm25_chunks.pkl"
    print(bm25_store_path)
    if not bm25_store_path.exists():
        raise FileNotFoundError(
            f"No BM25 chunk store found at {bm25_store_path}. "
            f"Run ingestion.py first."
        )

    with open(bm25_store_path, "rb") as f:
        chunks = pickle.load(f)

    print(f"Loaded {len(chunks)} chunks for BM25 index")

    bm25_retriever = BM25Retriever.from_documents(documents=chunks, k=m)
    return bm25_retriever


def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)


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


def ranked_doc(retriever, query, k=20):
    raw_doc = retriever.invoke(query)
    ranked_doc = []
    for i, doc in enumerate(raw_doc, start=1):
        ranked_doc.append([i, doc])

    doc_scores = sorted(ranked_doc, key=lambda x: x[0], reverse=False)
    return doc_scores[:k]


def printing_result(docs):
    # Header
    print(f"{'-' * 83}")
    print(f"| {'Document Content (Snippet)':<55} | {'Rank/Score':<20}       |")
    print(f"{'=' * 83}")

    # Rows
    for r in docs:
        # Clean newlines from content to avoid breaking the table rows
        clean_content = r[1].page_content.replace("\n", " ")[:52] + "..."
        # If using actual numerical scores, use r.scores. If using ranks, use the index.
        score_val = (str(r[0]))

        paper_id = r[1].metadata["paper_id"]
        print(f"| {paper_id}| {clean_content:<55} | {score_val:<20} |")
        print(f"{'-' * 83}")

def printing_result1(docs):
    print(f"{'-' * 83}")
    print(f"| {'Paper ID':<20} | {'Document Snippet':<40} | {'RRF Score':<10} |")
    print(f"{'=' * 83}")
    for paper_id, data in docs:
        snippet = data["doc"].page_content.replace("\n", " ")[:60] + "..."
        score = f"{data['score']:.4f}"
        print(f"| {paper_id:<20} | {snippet:<40} | {score:<10} |")
        print(f"{'-' * 83}")
def rrf_calculation(retrivers, k=60, num=10):
    rrm_rank = {}
    for r in retrivers:
        for i in r:
            key = i[1].metadata["paper_id"]
            current_rank = 1 / (k + i[0])
            if key in rrm_rank:
                rrm_rank[key]["score"] += current_rank
            else:
                rrm_rank[key] = {"score": current_rank, "doc": i[1]}
    rrm_rank = sorted(rrm_rank.items(), key=lambda x: x[1]["score"], reverse=True)
    return rrm_rank[:num]


if __name__ == "__main__":
    llm = ChatGoogleGenerativeAI(
        model=config.Gemini_MODEL_NAME,
        temperature=0.3
    )
    query = "what is prompt"

    bm_load = load_bm25_retriever(20)
    bm_result = ranked_doc(bm_load, query, k=20)
    print("=== BM25 results ===")
    printing_result(bm_result)

    vector_retriever = load_vector_retriever(20)
    vector_result = ranked_doc(vector_retriever, query, k=20)
    print("=== Vector results ===")
    printing_result(vector_result)

    rrf_result = rrf_calculation([bm_result, vector_result], 60, 10)
    print("=== RRf results ===")
    print(printing_result1(rrf_result))
