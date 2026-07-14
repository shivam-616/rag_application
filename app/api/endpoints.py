import re

from dotenv import load_dotenv

load_dotenv()
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

from pipeline.Generation.chains.chain import multiquery_rag_chain
from pipeline.Retrieval.retriever import load_vector_retriever, load_bm25_retriever

app = FastAPI()

bm_retriver = load_bm25_retriever(20)
vector_retriver = load_vector_retriever(20)
multi_retriver = [bm_retriver, vector_retriver]
multi_query_chain = multiquery_rag_chain(multi_retriver)


class QueryRequest(BaseModel):
    userquery: str = 'who are you'


class QueryResponse(BaseModel):
    query: str
    answer: str
    source: str


@app.get("/health")
def health():
    return {"status": "ok"}


def extract_sources(answer: str):
    sources = re.findall(r'\[Source:(.*?)\]', answer)
    return ", ".join(sources).strip()


@app.post("/rag/")
def root(queryRequest: QueryRequest):
    query = queryRequest.userquery
    final_output = multi_query_chain.invoke({"question": query,
                                             "context": multi_retriver})

    source = extract_sources(final_output)
    return QueryResponse(query=query, answer=final_output, source=source)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
