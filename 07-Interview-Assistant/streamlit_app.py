from pathlib import Path

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader

from src.config import (
    DATA_PATH,
    VECTOR_DB_PATH,
    TOP_K,
    EMBEDDING_MODEL,
    LLM_MODEL
)

from src.text_splitter import split_documents

from src.vector_store import (
    create_vector_store,
    load_vector_store,
    add_documents_to_vector_store
)

from src.rag_chain import (
    create_generation_chain,
    format_docs
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Interview Preparation Assistant",
    page_icon="🤖",
    layout="wide"
)


# =========================================================
# HELPER FUNCTION - CHECK INDEX
# =========================================================

def is_index_available():
    """
    Check whether the persistent Chroma vector database exists.

    Flow:
        vector_db/
            ↓
        Contains DB files?
            ↓
        Yes -> Index available
        No  -> Index not available
    """

    db_path = Path(VECTOR_DB_PATH)

    return (
        db_path.exists()
        and any(db_path.iterdir())
    )


# =========================================================
# HELPER FUNCTION - GET RETRIEVER
# =========================================================

def get_retriever():
    """
    Load the existing Chroma database and create a retriever.

    Query Flow:

        User Question
              ↓
        Query Embedding
              ↓
          Chroma DB
              ↓
       Similarity Search
              ↓
        Top-K Documents
    """

    vector_store = load_vector_store()

    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": TOP_K
        }
    )

    return retriever


# =========================================================
# HELPER FUNCTION - INDEX UPLOADED FILES
# =========================================================

def index_uploaded_files(uploaded_files):
    """
    Save and index newly uploaded PDF files.

    IMPORTANT:
    We DO NOT delete the existing vector_db directory.

    Why?

    Chroma may already have database files open while
    Streamlit is running. Windows will not allow those
    files to be deleted while they are in use.

    Instead:

        New PDF
           ↓
        Save into data/
           ↓
        PyPDFLoader
           ↓
        Documents
           ↓
        Text Splitter
           ↓
        Chunks
           ↓
        Embeddings
           ↓
        Existing Chroma DB

    If no Chroma DB exists, a new one is created.
    """

    all_documents = []

    # -----------------------------------------------------
    # Create data directory
    # -----------------------------------------------------

    data_directory = Path(DATA_PATH)

    data_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    # -----------------------------------------------------
    # STEP 1 - SAVE AND LOAD UPLOADED PDFs
    # -----------------------------------------------------

    for uploaded_file in uploaded_files:

        destination = (
            data_directory
            / uploaded_file.name
        )

        # Save uploaded PDF into data/
        with open(destination, "wb") as file:

            file.write(
                uploaded_file.getbuffer()
            )

        # Load PDF
        loader = PyPDFLoader(
            str(destination)
        )

        documents = loader.load()

        all_documents.extend(
            documents
        )

    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------

    if not all_documents:

        return 0, 0

    # -----------------------------------------------------
    # STEP 2 - SPLIT DOCUMENTS INTO CHUNKS
    # -----------------------------------------------------

    chunks = split_documents(
        all_documents
    )

    if not chunks:

        return len(all_documents), 0

    # -----------------------------------------------------
    # STEP 3 - STORE CHUNKS IN CHROMA
    # -----------------------------------------------------

    if is_index_available():

        # Existing DB
        #
        # New chunks are embedded and added to
        # the existing Chroma collection.

        add_documents_to_vector_store(
            chunks
        )

    else:

        # First-time indexing
        #
        # Create new Chroma database.

        create_vector_store(
            chunks
        )

    return (
        len(all_documents),
        len(chunks)
    )


# =========================================================
# APPLICATION HEADER
# =========================================================

st.title(
    "🤖 AI Interview Preparation Assistant"
)

st.write(
    """
    Upload your interview preparation PDFs, build the
    vector index, and ask questions using
    Retrieval-Augmented Generation (RAG).
    """
)

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header(
        "System Information"
    )

    st.write(
        f"**LLM:** {LLM_MODEL}"
    )

    st.write(
        f"**Embedding Model:** {EMBEDDING_MODEL}"
    )

    st.write(
        "**Vector DB:** Chroma"
    )

    st.write(
        f"**Top-K:** {TOP_K}"
    )

    st.divider()

    st.subheader(
        "Index Status"
    )

    if is_index_available():

        st.success(
            "Vector index is available"
        )

    else:

        st.warning(
            "Vector index is not available"
        )


# =========================================================
# SECTION 1 - UPLOAD DOCUMENTS
# =========================================================

st.header(
    "1. Upload Interview Documents"
)

uploaded_files = st.file_uploader(
    "Upload one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True
)


if uploaded_files:

    st.write(
        f"{len(uploaded_files)} file(s) selected."
    )

    if st.button(
        "Upload & Index"
    ):

        with st.spinner(
            "Processing documents and updating vector index..."
        ):

            pages, chunks = (
                index_uploaded_files(
                    uploaded_files
                )
            )

        # -------------------------------------------------
        # IMPORTANT
        #
        # pages and chunks exist only after button click.
        # Therefore this code MUST remain inside
        # the button block.
        # -------------------------------------------------

        if chunks > 0:

            st.success(
                "Documents indexed successfully."
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "New Pages Processed",
                    pages
                )

            with col2:

                st.metric(
                    "New Chunks Indexed",
                    chunks
                )

        else:

            st.error(
                "No document chunks were created."
            )


st.divider()


# =========================================================
# SECTION 2 - INDEX STATUS
# =========================================================

st.header(
    "2. Index Status"
)


if st.button(
    "Check Index Status"
):

    if is_index_available():

        st.success(
            "✅ Chroma vector database is ready for questions."
        )

        col1, col2 = st.columns(2)

        with col1:

            st.write(
                "**Vector Store:** Chroma"
            )

            st.write(
                f"**Top-K:** {TOP_K}"
            )

        with col2:

            st.write(
                f"**Embedding Model:** `{EMBEDDING_MODEL}`"
            )

            st.write(
                f"**LLM:** `{LLM_MODEL}`"
            )

    else:

        st.warning(
            "Vector index is not available."
        )

        st.write(
            "Upload and index at least one PDF first."
        )


st.divider()


# =========================================================
# SECTION 3 - ASK INTERVIEW QUESTION
# =========================================================

st.header(
    "3. Ask an Interview Question"
)


question = st.text_input(
    "Question",
    placeholder="Example: What is a functional interface?"
)


ask_button = st.button(
    "Get Answer",
    type="primary"
)


if ask_button:

    # -----------------------------------------------------
    # VALIDATE QUESTION
    # -----------------------------------------------------

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    # -----------------------------------------------------
    # VALIDATE VECTOR DB
    # -----------------------------------------------------

    elif not is_index_available():

        st.error(
            "Vector index is not available. "
            "Please upload and index documents first."
        )

    else:

        with st.spinner(
            "Searching documents and generating answer..."
        ):

            # =================================================
            # STEP 1 - RETRIEVAL
            # =================================================
            #
            # Question
            #    ↓
            # Embedding
            #    ↓
            # Chroma
            #    ↓
            # Similarity Search
            #    ↓
            # Top-K Document chunks
            #

            retriever = get_retriever()

            retrieved_docs = (
                retriever.invoke(
                    question
                )
            )

            # =================================================
            # STEP 2 - FORMAT CONTEXT
            # =================================================
            #
            # Retrieved Document objects
            #        ↓
            # doc.page_content
            #        ↓
            # format_docs()
            #        ↓
            # Context String
            #

            context = format_docs(
                retrieved_docs
            )

            # =================================================
            # STEP 3 - GENERATION
            # =================================================
            #
            # Question + Context
            #        ↓
            # PromptTemplate
            #        ↓
            # Llama 3.2
            #        ↓
            # StrOutputParser
            #        ↓
            # Final Answer
            #

            generation_chain = (
                create_generation_chain()
            )

            answer = (
                generation_chain.invoke(
                    {
                        "context": context,
                        "question": question
                    }
                )
            )

        # =================================================
        # DISPLAY ANSWER
        # =================================================

        st.subheader(
            "Answer"
        )

        st.markdown(
            answer
        )


        # =================================================
        # DISPLAY SOURCES
        # =================================================

        st.subheader(
            "Sources"
        )


        if not retrieved_docs:

            st.info(
                "No documents were retrieved."
            )

        else:

            # Prevent duplicate filename/page
            # combinations from appearing.

            displayed_sources = set()

            for doc in retrieved_docs:

                source = (
                    doc.metadata.get(
                        "source",
                        "Unknown"
                    )
                )

                page = (
                    doc.metadata.get(
                        "page_label",
                        "Unknown"
                    )
                )

                filename = (
                    Path(source).name
                )

                source_key = (
                    filename,
                    page
                )

                if (
                    source_key
                    in displayed_sources
                ):
                    continue

                displayed_sources.add(
                    source_key
                )

                st.write(
                    f"**{filename} — Page {page}**"
                )


        # =================================================
        # DISPLAY RETRIEVED EVIDENCE
        # =================================================

        with st.expander(
            "View Retrieved Evidence"
        ):

            if not retrieved_docs:

                st.write(
                    "No evidence retrieved."
                )

            else:

                for index, doc in enumerate(
                    retrieved_docs,
                    start=1
                ):

                    source = (
                        doc.metadata.get(
                            "source",
                            "Unknown"
                        )
                    )

                    page = (
                        doc.metadata.get(
                            "page_label",
                            "Unknown"
                        )
                    )

                    filename = (
                        Path(source).name
                    )

                    st.markdown(
                        f"### Chunk {index}"
                    )

                    st.write(
                        f"**Source:** {filename}"
                    )

                    st.write(
                        f"**Page:** {page}"
                    )

                    st.write(
                        "**Retrieved Text:**"
                    )

                    st.write(
                        doc.page_content
                    )

                    st.divider()