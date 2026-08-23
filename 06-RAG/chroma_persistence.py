import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


PDF_PATH = "data/Mamatha_Resume.pdf"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "resume_collection"


print("1. Creating embedding model...")

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


# --------------------------------------------------
# CREATE CHROMA INDEX ONLY IF IT DOES NOT EXIST
# --------------------------------------------------

if not os.path.exists(CHROMA_PATH):

    print("\n2. Chroma DB not found.")
    print("Creating persistent Chroma index...")

    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    print("Pages loaded:", len(documents))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    print("Chunks created:", len(chunks))

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PATH
    )

    print("\nChroma DB created and persisted successfully.")

else:

    print("\n2. Existing Chroma DB found.")
    print("Loading persisted Chroma collection...")

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    print("Chroma DB loaded successfully.")


# --------------------------------------------------
# CREATE RETRIEVER
# --------------------------------------------------

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)


# --------------------------------------------------
# QUESTION
# --------------------------------------------------

question = "What did she study?"

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
# CREATE CONTEXT
# --------------------------------------------------

context = "\n\n".join(
    doc.page_content
    for doc in results
)


# --------------------------------------------------
# PROMPT
# --------------------------------------------------

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


# --------------------------------------------------
# LLM
# --------------------------------------------------

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