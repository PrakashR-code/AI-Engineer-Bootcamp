from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

print("1. Loading PDF...")

loader = PyPDFLoader(
    "../06-RAG/data/Mamatha_Resume.pdf"
)

documents = loader.load()

print("Pages loaded:", len(documents))


# --------------------------------------------------
# 1. CHUNKING
# --------------------------------------------------

print("\n2. Splitting documents...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)

print("Chunks created:", len(chunks))


# --------------------------------------------------
# 2. EMBEDDING MODEL
# --------------------------------------------------

print("\n3. Creating embedding model...")

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


# --------------------------------------------------
# 3. CREATE FAISS VECTOR STORE
# --------------------------------------------------

print("\n4. Creating FAISS vector store...")

vector_store = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

print("FAISS vector store created.")


# --------------------------------------------------
# 4. USER QUESTION
# --------------------------------------------------

question = "whos resume is this?"

print("\n5. Question:")
print(question)


# --------------------------------------------------
# 5. SIMILARITY SEARCH
# --------------------------------------------------

print("\n6. Searching FAISS...")

results = vector_store.similarity_search(
    question,
    k=3
)


# --------------------------------------------------
# 6. DISPLAY RETRIEVED CHUNKS
# --------------------------------------------------

print("\n===== RETRIEVED CHUNKS =====")

for index, document in enumerate(results, start=1):

    print(f"\n----- RESULT {index} -----")

    print("\nContent:")
    print(document.page_content)

    print("\nMetadata:")
    print(document.metadata)

"""----------------------------------------------------------"""

prompt = PromptTemplate.from_template("""
You are a helpful assistant.

Answer the question using ONLY the context below.

Context:
{context}

Question:
{question}

If the answer is not available in the context,
say "I could not find the answer in the provided document."
""")

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

parser = StrOutputParser()

rag_chain = prompt | llm | parser

answer = rag_chain.invoke({
    "context": results,
    "question": question
})

print("\n===== FINAL ANSWER =====")
print(answer)

print("\n===== SOURCES =====")

for index, doc in enumerate(results, start=1):
    print(f"\nSource {index}:")
    print("Page:", doc.metadata.get("page"))
    print("Source:", doc.metadata.get("source"))
    print("Content:",doc.page_content)

