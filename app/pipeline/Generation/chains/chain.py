# chains where we have a
#       context =  | question  || promt || llm || output
# #
from operator import itemgetter

from dotenv import load_dotenv
from langsmith import traceable

load_dotenv()

from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import psutil

from Model.model import llm
from pipeline.Retrieval.retriever import (
    rrf_multiquery_retriver,
    load_bm25_retriever,
    load_vector_retriever,
)
from promt.multiquery_promt import MULTI_QUERY_PROMPT, LineListOutputParser
from promt.rag_promt import rag_basic_prompt


@traceable(run_type="chain", name="Document Context Formatter")
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

    def full_retrieval(input: dict):
        query = input["question"]

        # step 2: run multiquery retrieval
        docs = rrf_multiquery_retriver(input["context"], query)

        # step 3: format docs
        return format_docs(docs)

    runnable_retriver = RunnableLambda(full_retrieval)

    single_query_chain = (
        {"context": runnable_retriver, "question": RunnablePassthrough()}
        | rag_basic_prompt
        | llm
        | output_parser
    )
    return single_query_chain


def multi_query_generation_chain():
    """return a a list of query"""
    generated_query = MULTI_QUERY_PROMPT | llm | LineListOutputParser()
    return generated_query


def multiquery_rag_chain():
    """return a a list of query"""

    output_parser = StrOutputParser()
    query_gen_chain = multi_query_generation_chain()

    def full_retrieval(input: dict):
        query = input["question"]

        # step 1: generate multiple queries
        generated_queries = query_gen_chain.invoke(query)

        # step 2: run multiquery retrieval
        docs = rrf_multiquery_retriver(input["context"], generated_queries)

        # ADD THIS
        print(f"\n=== Retrieved {len(docs)} docs ===")
        """        for doc in docs:
            print(f"paper: {doc[0]} | score: {doc[1]['score']:.4f}")
            print(f"snippet: {doc[1]['doc'].page_content[:100]}")
            print("---")
        """

        # step 3: format docs
        return format_docs(docs)

    runnable_retrieval = RunnableLambda(full_retrieval)

    Multi_query_chain = (
        {"context": runnable_retrieval, "question": itemgetter("question")}
        | rag_basic_prompt
        | llm
        | output_parser
    )
    return Multi_query_chain


if __name__ == "__main__":
    questions = [
        "rag applicationWhat is retrieval augmented generation?",
        "What is the attention mechanism in transformers?",
        "How does the LoRA (Low-Rank Adaptation) method reduce the number of trainable parameters during fine-tuning?",
        "What are the core differences between BERT and GPT architectures regarding their attention mechanisms?",
        "According to standard RLHF papers, what are the three main stages involved in training a model with human feedback?",
        "What is the primary motivation behind FlashAttention, and how does it achieve better performance?",
        "What is catastrophic forgetting in the context of continual learning in neural networks?",
        "How does the Mixture of Experts (MoE) architecture scale the capacity of a language model without a proportional increase in computational cost?",
        "What is the function of the Temperature parameter during LLM text generation/sampling?",
        "In the context of scaling laws for language models (e.g., Chinchilla paper), what is the optimal relationship between dataset size and model parameters?",
        "What is Direct Preference Optimization (DPO) and how does it differ from traditional RLHF?",
    ]
    print(f"RAM available: {psutil.virtual_memory().available / 1024**3:.1f} GB")
    print(f"RAM total: {psutil.virtual_memory().total / 1024**3:.1f} GB")

    bm_retriver = load_bm25_retriever(20)
    vector_retriver = load_vector_retriever(20)

    multi_retriver = [bm_retriver, vector_retriver]

    """
     single_query_chain = (rag_chain(rrf_retriver))
        result = single_query_chain.invoke({"question": questions[0], "context": multi_retriver})
        print(f"{result} \n")
    """

    multi_query_chain = multi_query_generation_chain()
    multi_query = multi_query_chain.invoke(questions[0])
    print(f"{multi_query} \n")

    multi_query_chain = multiquery_rag_chain()
    final_output = multi_query_chain.invoke(
        {"question": questions[0], "context": multi_retriver}
    )

    print(final_output)
