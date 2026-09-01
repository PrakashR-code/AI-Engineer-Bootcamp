"""
=========================================================
OFFLINE / INDEXING PHASE
=========================================================

Purpose:
Prepare our interview documents and store them in the
Vector Database.

This is NOT executed for every user question.

Flow:

PDF Files
    ↓
PyPDFLoader
    ↓
LangChain Documents (page level)
    ↓
RecursiveCharacterTextSplitter
    ↓
Chunks
    ↓
OllamaEmbeddings (nomic-embed-text)
    ↓
Embedding Vectors
    ↓
Chroma Vector Database
    ↓
Persisted in vector_db/


Example:

22. JAVA 8 Features.pdf
        ↓
37 Documents / Pages
        ↓
140 Chunks
        ↓
140 Embeddings
        ↓
Chroma DB


IMPORTANT:
After indexing is completed, app.py can directly load
the existing Vector DB.

We DO NOT need to:
- Load PDFs again
- Split documents again
- Embed all document chunks again

Run ingest.py when:
1. Creating the Vector DB for the first time
2. Adding new documents
3. Rebuilding/re-indexing the knowledge base
=========================================================
"""

from src.config import DATA_PATH
from src.document_loader import load_documents
from src.text_splitter import split_documents
from src.vector_store import create_vector_store


def main():

    # STEP 1: Load PDF files
    documents = load_documents(DATA_PATH)

    # STEP 2: Split pages into smaller chunks
    chunks = split_documents(documents)

    # STEP 3:
    # Create embeddings for each chunk
    # and store them in Chroma Vector DB
    vector_store = create_vector_store(chunks)

    print(f"Chunks indexed: {len(chunks)}")


if __name__ == "__main__":
    main()