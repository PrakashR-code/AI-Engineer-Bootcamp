from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS

"""
                    "Whose resume is this?"
                              |
                 +------------+------------+
                 |                         |
                 v                         v
              Branch 1                  Branch 2
              context                   question
                 |                         |
                 v                         v
            Retriever              RunnablePassthrough
                 |                         |
                 v                         |
         Top-k Documents                  |
                 |                         |
                 v                         |
          format_docs()                   |
                 |                         |
                 v                         v
     "chunk1...\nchunk2..."      "Whose resume is this?"
                 |                         |
                 +------------+------------+
                              |
                              v
                    {
                      "context": "...",
                      "question": "Whose resume is this?"
                    }
                       |
                       v
                    Prompt
                       |
                       v
                     LLM
                       |
                       v
                    Parser
                       |
                       v
                  Final Answer
"""
# --------------------------------------------------
# 1. LOAD EMBEDDINGS
# --------------------------------------------------

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


# --------------------------------------------------
# 2. LOAD EXISTING FAISS INDEX
# --------------------------------------------------

vector_store = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)


# --------------------------------------------------
# 3. CREATE RETRIEVER
# --------------------------------------------------

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)


# --------------------------------------------------
# 4. FORMAT RETRIEVED DOCUMENTS
# --------------------------------------------------

def format_docs(docs):
    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


format_docs_runnable = RunnableLambda(format_docs)


# --------------------------------------------------
# 5. CREATE PROMPT
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
# 6. CREATE LLM
# --------------------------------------------------

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

parser = StrOutputParser()


# --------------------------------------------------
# 7. CREATE COMPLETE RAG PIPELINE
# --------------------------------------------------

rag_chain = (
    {
        "context": retriever | format_docs_runnable,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | parser
)


# --------------------------------------------------
# 8. INVOKE
# --------------------------------------------------

question = "Whose resume is this?"

answer = rag_chain.invoke(question)

print("\n===== FINAL ANSWER =====")
print(answer)