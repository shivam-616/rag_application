import re
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request  # Added Request here
from slowapi.util import get_remote_address
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from pipeline.Generation.chains.chain import multiquery_rag_chain
from pipeline.Retrieval.retriever import load_bm25_retriever, load_vector_retriever
from pydantic import BaseModel, Field

app = FastAPI()
load_dotenv()

# Initialize retrievers and the core RAG execution chain
bm_retriver = load_bm25_retriever(20)
vector_retriver = load_vector_retriever(20)
multi_retriver = [bm_retriver, vector_retriver]
multi_query_chain = multiquery_rag_chain()


# Data Transfer Objects (DTOs) for strict payload validation
class QueryRequest(BaseModel):
    userquery: str = Field(..., min_length=5, max_length=500)


class QueryResponse(BaseModel):
    query: str
    answer: str
    source: str


# Rate limiting setup targeting client IP addresses
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.get("/health")
def health():
    return {"status": "ok"}


def extract_sources(answer: str):
    # Extracts source metadata patterns from the generated LLM text
    sources = re.findall(r"\[Source:(.*?)\]", answer)
    return ", ".join(sources).strip()


@app.post("/rag/")
@limiter.limit("5/minute")
def root(
    request: Request,
    queryRequest: QueryRequest,
    x_api_key: str = Header(None),
    x_model_choice: str = Header("openai:gpt-4o-mini"),
):
    query = queryRequest.userquery

    # 1. Determine if the requested model is a local open-source model
    is_local_model = x_model_choice and x_model_choice.startswith("ollama")
    is_key_blank = x_api_key in (None, "", "None", "undefined")
    # 3. ONLY throw a 401 if it's a cloud model AND the key is blank
    if not is_local_model and is_key_blank:
        raise HTTPException(
            status_code=401,
            detail="API key is missing from headers. Cloud models require authentication.",
        )

    # 4. Dynamically build your configuration dictionary
    run_config = {"model": x_model_choice}
    if not is_local_model:
        run_config["api_key"] = x_api_key

    # 5. Invoke your chain with the runtime configuration
    final_output = multi_query_chain.invoke(
        {"question": query, "context": multi_retriver},
        config={"configurable": run_config},
    )

    source = extract_sources(final_output)
    return QueryResponse(query=query, answer=final_output, source=source)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
