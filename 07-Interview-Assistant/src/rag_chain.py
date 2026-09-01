from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

from src.config import LLM_MODEL
from src.prompts import INTERVIEW_PROMPT


def format_docs(retrieved_docs):
    """
    Convert retrieved Document chunks into one context string.

    retrieved_docs come from:
        Chroma -> Retriever -> Top-K Documents

    Each Document contains:
        page_content -> actual chunk text
        metadata     -> source, page, etc.
    """

    return "\n\n".join(
        doc.page_content for doc in retrieved_docs
    )


def create_generation_chain():
    """
    This chain does NOT perform retrieval.

    It receives:
        {
            "context": "...retrieved text...",
            "question": "...user question..."
        }

    Flow:
        Context + Question
                ↓
              Prompt
                ↓
            Llama 3.2
                ↓
         StrOutputParser
                ↓
              Answer
    """

    llm = ChatOllama(
        model=LLM_MODEL,
        temperature=0
    )

    parser = StrOutputParser()

    generation_chain = (
        INTERVIEW_PROMPT
        | llm
        | parser
    )

    return generation_chain