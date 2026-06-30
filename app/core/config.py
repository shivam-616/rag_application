# config.py
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub.constants import HF_TOKEN_PATH

load_dotenv(verbose=True)
# Security Tokens
HF_TOKEN = os.environ.get("HF_TOKEN")

# Model 2
Gemini_MODEL_NAME = "gemini-2.5-flash"
# Model  1
EMBEDDING_MODEL_NAME = "jinaai/jina-embeddings-v3"
MAX_TOKENS = 1000
MERGE_PEERS = True

# Storage & Directory Paths
PDF_LOCATION = Path(r"C:\learning_python\Learning_Rag\arxiv_dataset")
CHROMA_DB_DIR = r"C:\learning_python\Learning_Rag\app\chroma_db"
