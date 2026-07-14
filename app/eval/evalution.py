import json
from operator import itemgetter

import pandas as pd
import psutil
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from Model.model import gemmni_llm
from pipeline.Generation.chains.chain import (
    multiquery_rag_chain,
    multi_query_generation_chain,
)
from pipeline.Retrieval.retriever import (
    load_bm25_retriever,
    load_vector_retriever,
    rrf_multiquery_retriver,
)


def build_faith_dataset(questions, ground_truth):
    print(f"RAM available: {psutil.virtual_memory().available / 1024**3:.1f} GB")
    print(f"RAM total: {psutil.virtual_memory().total / 1024**3:.1f} GB")

    bm_retriver = load_bm25_retriever(20)
    vector_retriver = load_vector_retriever(20)

    row = []

    multi_retriver = [bm_retriver, vector_retriver]
    multi_query_chain = multiquery_rag_chain(multi_retriver)
    scorer = faithfull_scorer()
    for test, gtruth in zip(questions, ground_truth):
        question = test
        print(f"Running : {question} \n\n {gtruth}")

        final_output = multi_query_chain.invoke(
            {"question": question, "context": multi_retriver}
        )

        query_chain = multi_query_generation_chain()
        generated_queries = query_chain.invoke(question)
        retrieved_docs = rrf_multiquery_retriver(multi_retriver, generated_queries)
        contexts = [doc[1]["doc"].page_content for doc in retrieved_docs]
        raw = scorer.invoke(
            {"context": "\n\n".join(contexts), "generated_answer": final_output}
        )
        cleaned = (
            raw.strip()
            .removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        try:
            faith_score = json.loads(cleaned)["score"]
        except:
            faith_score = 0.0

        row.append(
            {
                "question": question,
                "answer": final_output,
                "context": "\n\n".join(contexts),
                # "ground_truth": gtruth,
                "faithfulness": faith_score,
            }
        )
        print(f"QUESTION: {question}")
        print(f"ANSWER: {final_output[:100]}")
        print(f"RAW SCORER OUTPUT: {raw}")
        print(f"FAITH SCORE: {faith_score}")
        print("---")
    df = pd.DataFrame(row)
    df.to_csv("faith_results.csv", index=False)
    print(df[["question", "faithfulness"]])


def faithfull_scorer():
    output_parser = StrOutputParser()

    FAITHFULNESS_SYSTEM_PROMPT = """You are an impartial expert evaluator assessing the "faithfulness" of an AI-generated answer. 
    Faithfulness measures whether the generated answer is entirely grounded in and supported by the provided context.

    STRICT RULES:
    1. You must evaluate if the GENERATED ANSWER can be directly inferred from the CONTEXT provided.
    2. Disregard your prior knowledge. An answer is faithful ONLY if the context explicitly supports it.
    3. SCORING CRITERIA:
       - 1.0: The answer is entirely supported by the context. No external information was introduced.
       - 0.5: The answer is partially faithful. It contains some supported facts, but also includes unsupported claims or hallucinated details.
       - 0.0: The answer directly contradicts the context or is entirely made up of information not found in the context.
       - You may use intermediate scores (e.g., 0.8, 0.2 , 0.3 or any number 0.x (where x being 0 to 9)) if appropriate.
    4. Output your evaluation strictly as a JSON object with exactly two keys: "reasoning" and "score". No other keys allowed. No extra text outside the JSON.
        Example output:
        {{"reasoning": "The answer is directly supported by the context.", "score": 1.0}}
        ANY other format will be considered invalid.
    """

    FAITHFULNESS_HUMAN_PROMPT = """CONTEXT:
    {context} 

    GENERATED ANSWER:
    {generated_answer}

    Respond with ONLY a JSON object with exactly two keys "reasoning" and "score". No other text:"""

    rag_faithfulness_prompt = ChatPromptTemplate.from_messages(
        [("system", FAITHFULNESS_SYSTEM_PROMPT), ("human", FAITHFULNESS_HUMAN_PROMPT)]
    )

    faith_chain = (
        {
            "context": itemgetter("context"),
            "generated_answer": itemgetter("generated_answer"),
        }
        | rag_faithfulness_prompt
        | gemmni_llm
        | output_parser
    )

    return faith_chain


def recall_scorer():
    """
        sentences in ground truth that ARE supported by contexts
    ─────────────────────────────────────────────────────────
        total sentences in ground truth
    """
    output_parser = StrOutputParser()

    RECALL_SYSTEM_PROMPT = """
        You are an impartial evaluator assessing context recall.
    Context recall measures whether the retrieved context contains enough information to support the ground truth answer.

    STRICT RULES:
    1. Break the GROUND TRUTH into individual statements/sentences.
    2. For each statement, check if it is supported by the CONTEXT.
    3. SCORING: supported statements / total statements = score
    4. Output ONLY this JSON with no other text:
    {{"reasoning": "brief explanation", "score": 0.0}}

        """
    RECALL_HUMAN_PROMPT = """
        CONTEXT:
        {context} 

        GROUND TRUTH:
        {ground_truth}

        Respond with ONLY a JSON object with exactly two keys "reasoning" and "score". No other text: 
        """

    rag_recall_prompt = ChatPromptTemplate.from_messages(
        [("system", RECALL_SYSTEM_PROMPT), ("human", RECALL_HUMAN_PROMPT)]
    )

    recall_chain = (
        {"context": itemgetter("context"), "ground_truth": itemgetter("ground_truth")}
        | rag_recall_prompt
        | gemmni_llm
        | output_parser
    )

    return recall_chain


def build_recall_dataset(questions, ground_truth):
    print(f"RAM available: {psutil.virtual_memory().available / 1024**3:.1f} GB")
    print(f"RAM total: {psutil.virtual_memory().total / 1024**3:.1f} GB")

    bm_retriver = load_bm25_retriever(20)
    vector_retriver = load_vector_retriever(20)

    row = []

    multi_retriver = [bm_retriver, vector_retriver]

    scorer = recall_scorer()
    for test, gtruth in zip(questions, ground_truth):
        question = test
        print(f"Running : {question} \n\n {gtruth}")

        query_chain = multi_query_generation_chain()
        generated_queries = query_chain.invoke(question)
        retrieved_docs = rrf_multiquery_retriver(multi_retriver, generated_queries)
        contexts = [doc[1]["doc"].page_content for doc in retrieved_docs]

        raw = scorer.invoke({"context": "\n\n".join(contexts), "ground_truth": gtruth})
        cleaned = (
            raw.strip()
            .removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        try:
            recall_score = json.loads(cleaned)["score"]
        except:
            recall_score = 0.0

        row.append(
            {
                "question": question,
                "context": "\n\n".join(contexts),
                "ground_truth": gtruth,
                "recall_score": recall_score,
            }
        )

        print(f"QUESTION: {question}")
        print(f"RAW SCORER OUTPUT: {raw}")
        print(f"RECALL SCORER: {recall_score}")
        print("---")
    df = pd.DataFrame(row)
    df.to_csv("recall_results.csv", index=False)
    print(df[["question", "recall_score"]])


if __name__ == "__main__":
    question = [
        "What is retrieval augmented generation?",
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

    gtruth = [
        "RAG is a method that combines retrieval of relevant documents with LLM generation to produce grounded answers.",
        "The attention mechanism allows a model to focus on different parts of the input sequence when processing a specific token, mapping a query to a set of keys and values.",
        "LoRA freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, drastically reducing parameters.",
        "BERT uses a bidirectional encoder attention mechanism to look at context from both left and right, while GPT uses a causal (or masked) decoder attention mechanism to only look at tokens to the left.",
        "The three stages are pre-training a language model, training a reward model based on human preferences, and fine-tuning the language model using PPO or similar RL algorithms against the reward model.",
        "FlashAttention aims to speed up training and reduce memory footprint by making attention IO-aware, minimizing the number of memory reads/writes between GPU High Bandwidth Memory (HBM) and SRAM.",
        "Catastrophic forgetting occurs when a neural network is trained on a new task and completely loses or overrides the information it previously learned from prior tasks.",
        "MoE replaces dense layers with sparse MoE layers where each token is routed to only a few 'expert' sub-networks, keeping the compute per token constant while expanding total parameters.",
        "Temperature scales the logits before the softmax layer. Lowering temperature makes the distribution sharper and output more deterministic, while raising it increases diversity and randomness.",
        "The Chinchilla scaling laws state that for optimal compute training, model size and the number of training tokens should be scaled in equal proportion.",
        "DPO parameterizes the reward function directly within the language model policy, allowing it to optimize for human preferences using a simple binary cross-entropy loss without needing a separate reward model or RL training phase.",
    ]
    # dataset = build_eval_dataset(question, gtruth)

    # dataset = build_faith_dataset(question[:7], gtruth[:7])

    dataset = build_recall_dataset(question[:7], gtruth[:7])
