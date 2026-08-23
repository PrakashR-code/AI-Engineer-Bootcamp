import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


PDF_PATH =   "../06-RAG/data/Mamatha_Resume.pdf"
FAISS_PATH = "faiss_index"


print("1. Creating embedding model...")

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


# --------------------------------------------------
# CREATE INDEX ONLY IF IT DOES NOT EXIST
# --------------------------------------------------

if not os.path.exists(FAISS_PATH):

    print("\n2. FAISS index not found.")
    print("Creating index from PDF...")

    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    print("Pages loaded:", len(documents))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    print("Chunks created:", len(chunks))

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    vector_store.save_local(FAISS_PATH)

    print("\nFAISS index saved successfully.")

else:

    print("\n2. Existing FAISS index found.")
    print("Loading index from disk...")

    vector_store = FAISS.load_local(
        FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    print("FAISS index loaded successfully.")


# --------------------------------------------------
# CREATE RETRIEVER
# --------------------------------------------------

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)


# --------------------------------------------------
# QUESTION
# --------------------------------------------------

question = "Whose resume is this?"

print("\n3. Question:")
print(question)


# --------------------------------------------------
# RETRIEVE
# --------------------------------------------------

results = retriever.invoke(question)

print("\n===== RETRIEVED CHUNKS =====")

for index, doc in enumerate(results, start=1):
    print(f"\n--- RESULT {index} ---")
    print(doc.page_content)
    print("Metadata:", doc.metadata)


# --------------------------------------------------
# FORMAT CONTEXT
# --------------------------------------------------

context = "\n\n".join(
    doc.page_content
    for doc in results
)


# --------------------------------------------------
# RAG PROMPT
# --------------------------------------------------

prompt = PromptTemplate.from_template("""
You are a helpful assistant.

Answer the question using ONLY the provided context.

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


# --------------------------------------------------
# FINAL ANSWER
# --------------------------------------------------

answer = rag_chain.invoke({
    "context": context,
    "question": question
})

print("\n===== FINAL ANSWER =====")
print(answer)