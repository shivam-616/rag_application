from dotenv import load_dotenv

from core import config
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv(verbose=True)

embeddings = HuggingFaceEmbeddings(
    model_name=config.EMBEDDING_MODEL_NAME ,
    model_kwargs={"trust_remote_code": True}
)
vector_store = Chroma(
    persist_directory=config.CHROMA_DB_DIR,
    embedding_function=embeddings
)
llm = ChatGoogleGenerativeAI(
    model=config.Gemini_MODEL_NAME,
    temperature=0.3
)
