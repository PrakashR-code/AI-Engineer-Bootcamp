"""
=========================================================
ONLINE / QUERY PHASE
=========================================================

Purpose:
Answer user questions using the ALREADY CREATED
Vector Database.

app.py does NOT load and chunk the original PDFs again.

Flow:

User Question
    ↓
Load existing Chroma Vector DB
    ↓
Retriever
    ↓
Question is converted into an embedding
    ↓
Similarity Search
    ↓
Retrieve Top-K relevant stored chunks
    ↓
Later:
Context + Original Question
    ↓
Prompt
    ↓
LLM
    ↓
Grounded Answer


IMPORTANT:

Document embeddings were already created by ingest.py.

But every NEW USER QUESTION still needs to be embedded
so that its vector can be compared with the stored
document vectors.

Example:

Question:
"What is a functional interface?"

        ↓

Question Embedding

        ↓

Compare against stored vectors in Chroma

        ↓

Top 3 relevant chunks

        ↓

Later these chunks become context for the LLM.
=========================================================
"""
from pathlib import Path

from src.retriever import get_retriever
from src.rag_chain import create_generation_chain, format_docs


def main():

    print("========================================")
    print(" AI INTERVIEW PREPARATION ASSISTANT")
    print("========================================")
    print("Ask any question from your interview documents.")
    print("Type 'exit' to quit.")

    # Create these only once when application starts.
    # We do NOT reload them for every question.
    retriever = get_retriever()
    generation_chain = create_generation_chain()

    # Keep application running until user enters "exit".
    while True:

        print()
        question = input("Question: ").strip()

        # Exit condition
        if question.lower() == "exit":
            print("Goodbye!")
            break

        # Ignore empty input
        if not question:
            print("Please enter a question.")
            continue

        # =================================================
        # STEP 1 - RETRIEVAL
        # =================================================
        #
        # Question
        #    ↓
        # Query Embedding
        #    ↓
        # Chroma Vector DB
        #    ↓
        # Similarity Search
        #    ↓
        # Top-K Document chunks
        #
        retrieved_docs = retriever.invoke(question)

        # =================================================
        # STEP 2 - CREATE CONTEXT
        # =================================================
        #
        # Extract page_content from the retrieved Documents.
        #
        # Top-K Documents
        #       ↓
        # format_docs()
        #       ↓
        # One context string
        #
        context = format_docs(retrieved_docs)

        # =================================================
        # STEP 3 - GENERATION
        # =================================================
        #
        # Context + Original Question
        #          ↓
        #     Grounded Prompt
        #          ↓
        #       Llama 3.2
        #          ↓
        #        Answer
        #
        answer = generation_chain.invoke({
            "context": context,
            "question": question
        })

        print()
        print("========== ANSWER ==========")
        print(answer)

        # =================================================
        # STEP 4 - SOURCE ATTRIBUTION
        # =================================================
        #
        # Use metadata from the SAME retrieved Documents.
        # No second retrieval is required.
        #
        print()
        print("========== SOURCES ==========")

        for doc in retrieved_docs:

            source = doc.metadata.get(
                "source",
                "Unknown"
            )

            page = doc.metadata.get(
                "page_label",
                "Unknown"
            )

            filename = Path(source).name

            print(f"- {filename} - Page {page}")


if __name__ == "__main__":
    main()