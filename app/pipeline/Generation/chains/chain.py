# chains where we have a
#       context =  | question  || promt || llm || output
# #
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from Model.model import llm
from pipeline.Retrieval.retriever import (rrf_retriver)
from promt import rag_promt
from promt.rag_promt import rag_basic_prompt


def format_docs(docs):
    formatted = []
    for doc in docs:
        document = doc[1].get("doc")
        chunk_id = document.metadata["chunk_index"]
        paper_id = document.metadata["paper_id"]
        content = document.page_content
        formatted.append(f"[paper_id: {paper_id} | chunk: {chunk_id}]\n{content}")
    return "\n\n".join(formatted)


def rag_chain(retriver):
    output_parser = StrOutputParser()
    runnable_retriver = RunnableLambda(retriver)
    runnable_format_doc = RunnableLambda(format_docs)
    chain = (
            {"context": runnable_retriver | runnable_format_doc, "question": RunnablePassthrough()}
            | rag_basic_prompt
            | llm
            | output_parser
    )
    return chain


if __name__ == "__main__":
    query = "sucide "

    chain = (rag_chain(rrf_retriver))
    result = chain.invoke(query)
    print(result)
