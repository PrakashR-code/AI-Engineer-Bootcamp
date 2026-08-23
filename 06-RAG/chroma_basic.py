from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS


print("1. Loading PDF...")

loader = PyPDFLoader(
    "../06-RAG/data/Mamatha_Resume.pdf"
)

documents = loader.load()

print("Pages loaded:", len(documents))


# ---------------------------------------------
# CHUNKING
# ---------------------------------------------

print("\n2. Splitting documents...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)

print("Chunks created:", len(chunks))


# ---------------------------------------------
# EMBEDDINGS
# ---------------------------------------------

print("\n3. Creating embedding model...")

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


# ---------------------------------------------
# CHROMA
# ---------------------------------------------

print("\n4. Creating Chroma vector store...")

chroma_vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="resume_collection",
    persist_directory="./chroma_db"
)

print("Chroma vector store created.")

# ---------------------------------------------
# FAISS
# ---------------------------------------------

print("\n4. Creating FAISS vector store...")

faiss_vector_store = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

print("FAISS vector store created.")

# ---------------------------------------------
# CHROMA RETRIEVER
# ---------------------------------------------

chroma_retriever = chroma_vector_store.as_retriever(
    search_kwargs={"k": 3}
)

# ---------------------------------------------
# FAISS RETRIEVER
# ---------------------------------------------

faiss_retriever = faiss_vector_store.as_retriever(
    search_kwargs={"k": 3}
)

# ---------------------------------------------
# QUERY
# ---------------------------------------------

question = "What she/he studied?"

print("\n5. Question:")
print(question)

chroma_results = chroma_retriever.invoke(question)
faiss_results = faiss_retriever.invoke(question)

# ---------------------------------------------
# RESULTS
# ---------------------------------------------

print("\n===== CHROMA RETRIEVED DOCUMENTS =====")

for index, doc in enumerate(chroma_results, start=1):

    print(f"\n----- RESULT {index} -----")

    print(doc.page_content)

    print("\nMetadata:")
    print(doc.metadata)

print("\n===== FAISS RETRIEVED DOCUMENTS =====")

for index, doc in enumerate(faiss_results, start=1):

    print(f"\n----- RESULT {index} -----")

    print(doc.page_content)

    print("\nMetadata:")
    print(doc.metadata)