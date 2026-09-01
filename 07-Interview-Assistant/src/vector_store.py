import os

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from src.config import (
    VECTOR_DB_PATH,
    EMBEDDING_MODEL
)


COLLECTION_NAME = "interview_assistant"


def get_embeddings():

    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL
    )

    return embeddings


def create_vector_store(chunks):

    embeddings = get_embeddings()

    print("Creating embeddings and storing chunks in Chroma...")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=VECTOR_DB_PATH
    )

    return vector_store


def load_vector_store():

    embeddings = get_embeddings()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embeddings
    )

    return vector_store

def add_documents_to_vector_store(chunks):
    """
    Add new document chunks to the existing Chroma database.

    Used when a user uploads additional PDFs.

    Flow:
        New PDF
          ↓
        Chunks
          ↓
        Embeddings
          ↓
        Existing Chroma Collection

    Unlike create_vector_store(), this does not rebuild or
    delete the existing vector database.
    """

    vector_store = load_vector_store()

    vector_store.add_documents(chunks)

    return vector_store