import os

from dotenv import load_dotenv

load_dotenv(verbose=True)
print(os.getenv("LANGCHAIN_API_KEY"))
print(os.getenv("LANGCHAIN_TRACING_V2"))
