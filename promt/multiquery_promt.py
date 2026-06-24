from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import BaseOutputParser

# custom parser to split the LLM's newline-separated output into a list
class LineListOutputParser(BaseOutputParser):
    def parse(self, text: str):
        lines = text.strip().split("\n")
        return [l.strip() for l in lines if l.strip()]

MULTI_QUERY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an AI assistant helping to improve document retrieval.
Your job is to generate alternative versions of the user's question.

RULES:
1. Generate exactly 4 different versions of the question.
2. Each version should approach the same information need from a different angle — different vocabulary, different phrasing, different specificity.
3. Do NOT change the meaning or ask for different information.
4. Output ONLY the 4 questions, one per line, no numbering, no explanation, no preamble.
5. Do not repeat the original question."""),

    ("human", """Original question: {question}

Generate 4 alternative versions:""")
])