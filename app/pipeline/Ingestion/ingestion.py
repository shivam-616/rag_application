import os
import gc
import pickle
import re
import json
import torch
import concurrent.futures
from pathlib import Path

from langchain_community.retrievers import BM25Retriever
from langchain_docling import DoclingLoader
from docling.chunking import HybridChunker
from langchain_docling.loader import ExportType
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from core import config

# --- OPTIMIZATION: Un-throttle CPU Cores ---
# Removed restrictive "1" thread limits to allow full hardware utilization.
# Docling and PyTorch will now scale to your available CPU cores automatically.
if "OMP_NUM_THREADS" in os.environ: del os.environ["OMP_NUM_THREADS"]
if "MKL_NUM_THREADS" in os.environ: del os.environ["MKL_NUM_THREADS"]
if "DOCLING_NUM_THREADS" in os.environ: del os.environ["DOCLING_NUM_THREADS"]

os.environ["HF_TOKEN"] = config.HF_TOKEN


def clean_metadata_for_chroma(documents):
    allowed_types = (str, int, float, bool)
    for doc in documents:
        doc.metadata = {
            k: v for k, v in doc.metadata.items()
            if isinstance(v, allowed_types)
        }
    return documents


def parse_arxiv_id(stem: str) -> dict:
    match = re.match(r"(\d{4}\.\d{4,5})(v\d+)?", stem)
    if match:
        paper_id = match.group(1)
        version = match.group(2) or "v1"
        return {
            "paper_id": paper_id,
            "version": version,
            "arxiv_url": f"https://arxiv.org/abs/{paper_id}",
        }
    return {"paper_id": stem, "version": "unknown", "arxiv_url": ""}


# Pipeline options — OCR off, tables on (FAST mode for speed)
pipeline_options = PdfPipelineOptions()
pipeline_options.allow_external_plugins = True
pipeline_options.do_ocr = False
pipeline_options.do_table_structure = True
pipeline_options.table_structure_options.mode = TableFormerMode.FAST
pipeline_options.table_structure_options.do_cell_matching = True

# Tokenizer for accurate token counting
tokenizer = AutoTokenizer.from_pretrained(
    config.MODEL_NAME, trust_remote_code=True
)


def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


# --- OPTIMIZATION: Device Targeting (GPU Setup) ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device.upper()} for vector embeddings.")

embeddings = HuggingFaceEmbeddings(
    model_name=config.MODEL_NAME,
    model_kwargs={"trust_remote_code": True, "device": device}
)

vector_store = Chroma(
    persist_directory=config.CHROMA_DB_DIR,
    embedding_function=embeddings
)

# Fallback splitter
fallback_splitter = RecursiveCharacterTextSplitter(
    chunk_size=4000, chunk_overlap=400
)

# Deduplication tracker
seen_file = Path(config.CHROMA_DB_DIR) / "ingested.json"
seen_ids = set(json.loads(seen_file.read_text())) if seen_file.exists() else set()

# --- OPTIMIZATION: State Management for BM25 Chunks ---
bm25_store_path = Path(config.CHROMA_DB_DIR) / "bm25_chunks.pkl"
if bm25_store_path.exists():
    with open(bm25_store_path, "rb") as f:
        bmdoc = pickle.load(f)
    print(f"Loaded {len(bmdoc)} existing BM25 chunks from disk.")
else:
    bmdoc = []
    print("No existing BM25 cache found. Initializing empty array.")


def process_single_pdf(pdf_file: Path):
    """
    Worker function to process a single PDF using Docling.
    Runs completely isolated inside a thread pool worker.
    """
    if pdf_file.stem in seen_ids:
        return None, f"Skipping {pdf_file.name} (Already Ingested)"

    try:
        # Separate converter instance per thread to ensure thread safety
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )

        loader = DoclingLoader(
            file_path=str(pdf_file),
            export_type=ExportType.DOC_CHUNKS,
            chunker=HybridChunker(
                tokenizer=config.MODEL_NAME,
                max_tokens=config.MAX_TOKENS,
                merge_peers=config.MERGE_PEERS,
            ),
            converter=converter,
        )

        raw_chunks = list(loader.lazy_load())
        chunks = []

        if raw_chunks:
            for rc in raw_chunks:
                token_count = count_tokens(rc.page_content)
                if token_count > config.MAX_TOKENS:
                    sub_docs = fallback_splitter.split_documents([rc])
                    chunks.extend(sub_docs)
                else:
                    chunks.append(rc)

            meta = parse_arxiv_id(pdf_file.stem)
            for i, chunk in enumerate(chunks):
                chunk.metadata["source"] = str(pdf_file)
                chunk.metadata["paper_id"] = meta["paper_id"]
                chunk.metadata["version"] = meta["version"]
                chunk.metadata["arxiv_url"] = meta["arxiv_url"]
                chunk.metadata["chunk_index"] = i

            filtered_chunks = clean_metadata_for_chroma(chunks)
            return filtered_chunks, f"Successfully parsed {pdf_file.name} ({len(filtered_chunks)} chunks)"

        return [], f"⚠️ Warning: No chunks generated for {pdf_file.name}"

    except Exception as e:
        return None, f"❌ Failed to process {pdf_file.name}: {str(e)}"


# --- MAIN EXECUTION PIPELINE ---
if __name__ == "__main__":
    pdf_files = list(config.PDF_LOCATION.glob("*.pdf"))
    print(f"\nFound {len(pdf_files)} total PDFs | {len(seen_ids)} verified inside cache.\n")

    total_chunks_saved = 0
    failed = []
    new_chunks_added = False

    # Adjust max_workers depending on hardware capabilities.
    # 4 is optimal for balanced RAM usage with Docling layouts.
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_pdf = {executor.submit(process_single_pdf, pdf): pdf for pdf in pdf_files}

        for future in concurrent.futures.as_completed(future_to_pdf):
            pdf_file = future_to_pdf[future]
            try:
                result_chunks, status_msg = future.result()
                print(status_msg)

                if result_chunks is None:
                    # Captured exception or already processed
                    if "Failed" in status_msg:
                        failed.append({"file": pdf_file.name, "error": status_msg})
                    continue

                if result_chunks:
                    new_chunks_added = True

                    # Batch save straight into Chroma vector store
                    batch_size = 50
                    for i in range(0, len(result_chunks), batch_size):
                        batch = result_chunks[i:i + batch_size]
                        vector_store.add_documents(documents=batch)

                    # Append chunks to global list for hybrid BM25 retrieval mapping
                    bmdoc.extend(result_chunks)
                    total_chunks_saved += len(result_chunks)

                    # Update local state tracking
                    seen_ids.add(pdf_file.stem)
                    seen_file.write_text(json.dumps(list(seen_ids)))

            except Exception as exc:
                print(f"Thread execution crashed for {pdf_file.name}: {exc}")
                failed.append({"file": pdf_file.name, "error": str(exc)})

            finally:
                # Active garbage collection collection inside main thread loops
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # --- SAVE UPDATED BM25 CACHE ---
    if new_chunks_added:
        with open(bm25_store_path, "wb") as f:
            pickle.dump(bmdoc, f)
        print(f"\n💾 Saved updated BM25 database state. Complete index count: {len(bmdoc)}")
    else:
        print("\nℹ️ No new files processed. BM25 storage file left intact.")

    print(f"\n🎉 Pipeline Complete! New Chunks Saved: {total_chunks_saved} | Failed Runs: {len(failed)}")
    if failed:
        for f in failed:
            print(f"  - {f['file']}: {f['error']}")