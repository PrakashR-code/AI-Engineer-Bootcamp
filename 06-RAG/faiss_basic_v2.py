from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


print("1. Loading PDF...")

loader = PyPDFLoader(
    "../06-RAG/data/Mamatha_Resume.pdf"
)

documents = loader.load()

print("Pages loaded:", len(documents))


# --------------------------------------------------
# 2. CHUNKING
# --------------------------------------------------

print("\n2. Splitting documents...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)

print("Chunks created:", len(chunks))


# --------------------------------------------------
# 3. EMBEDDING MODEL
# --------------------------------------------------

print("\n3. Creating embedding model...")

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


# --------------------------------------------------
# 4. CREATE FAISS VECTOR STORE
# --------------------------------------------------

print("\n4. Creating FAISS vector store...")

vector_store = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

print("FAISS vector store created.")


# --------------------------------------------------
# 5. CREATE RETRIEVER  
# Take top 3 most relevant chunks from the vector store
# --------------------------------------------------

print("\n5. Creating Retriever...")

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)


# --------------------------------------------------
# 6. USER QUESTION
# --------------------------------------------------

question = "Whose resume is this?"

print("\n6. Question:")
print(question)


# --------------------------------------------------
# 7. RETRIEVE RELEVANT DOCUMENTS
# --------------------------------------------------

print("\n7. Retrieving documents...")

results = retriever.invoke(question)


# --------------------------------------------------
# 8. DISPLAY RETRIEVED CHUNKS
# --------------------------------------------------

print("\n===== RETRIEVED CHUNKS =====")

for index, doc in enumerate(results, start=1):

    print(f"\n----- RESULT {index} -----")

    print("\nContent:")
    print(doc.page_content)

    print("\nMetadata:")
    print(doc.metadata)


# --------------------------------------------------
# 9. COMBINE RETRIEVED CHUNKS AS CONTEXT
# --------------------------------------------------

context = "\n\n".join(
    doc.page_content
    for doc in results
)


# --------------------------------------------------
# 10. CREATE RAG PROMPT
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
# 11. CREATE LLM
# --------------------------------------------------

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

parser = StrOutputParser()


# --------------------------------------------------
# 12. CREATE RAG CHAIN
# --------------------------------------------------

rag_chain = prompt | llm | parser


# --------------------------------------------------
# 13. GENERATE FINAL ANSWER
# --------------------------------------------------

print("\n8. Generating final answer...")

answer = rag_chain.invoke({
    "context": context,
    "question": question
})


print("\n===== FINAL ANSWER =====")
print(answer)

# --------------------------------------------------
# GETTING SIMILARITY SCORES FOR THE RETRIEVED CHUNKS
# --------------------------------------------------
"""
1.02  ← closest / best of these 3
1.06
1.10  ← farthest / weakest of these 3
But there's an important point: don't interpret the absolute value 1.02 by itself as “good” or “bad.” It isn't a percentage or confidence score.
What we can confidently say from these three values is:
Result 1 is ranked more relevant than Result 2, which is ranked more relevant than Result 3."""

results_with_scores = vector_store.similarity_search_with_score(
    question,
    k=3
)

# --------------------------------------------------
# 14. DISPLAY SOURCES
# --------------------------------------------------

print("\n===== SOURCES =====")

for index, (doc, score) in enumerate(results_with_scores, start=1):
    print(f"\n----- RESULT {index} -----")
    print("Score:", score)
    print("Content:")
    print(doc.page_content)
    print("Metadata:")
    print(doc.metadata)