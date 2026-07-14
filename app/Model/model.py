from dotenv import load_dotenv
from core import config
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
import torch
from langchain.chat_models import init_chat_model

torch.set_num_threads(1)
load_dotenv(verbose=True)

embeddings = HuggingFaceEmbeddings(
    model_name=config.EMBEDDING_MODEL_NAME,
    model_kwargs={"trust_remote_code": True, "device": "cpu"},
    encode_kwargs={"batch_size": 8},
)
vector_store = Chroma(
    persist_directory=config.CHROMA_DB_DIR, embedding_function=embeddings
)
gemmni_llm = ChatGoogleGenerativeAI(model=config.Gemini_MODEL_NAME, temperature=0.3)

llm = init_chat_model(configurable_fields="any")  # noqa: F821
