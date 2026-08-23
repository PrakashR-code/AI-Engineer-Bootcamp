import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PDF_PATH = "data/Mamatha_Resume.pdf"

FAISS_PATH = "faiss_eval_index"
CHROMA_PATH = "chroma_eval_db"
CHROMA_COLLECTION = "resume_eval_collection"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 3


# --------------------------------------------------
# TEST CASES
# --------------------------------------------------

test_cases = [
    {
        "question": "Whose resume is this?",
        "expected_keyword": "Mamatha",
        "should_abstain": False
    },
    {
        "question": "Where did she study?",
        "expected_keyword": "Oxford",
        "should_abstain": False
    },
    {
        "question": "What did she study?",
        "expected_keyword": None,
        "should_abstain": True
    },
    {
        "question": "Which technologies has she worked with?",
        "expected_keyword": "Java",
        "should_abstain": False
    }
]


# --------------------------------------------------
# EMBEDDING MODEL
# --------------------------------------------------

print("1. Creating embedding model...")

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


# --------------------------------------------------
# LOAD + CHUNK PDF
# --------------------------------------------------

print("\n2. Loading PDF...")

loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

print("Pages loaded:", len(documents))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

chunks = splitter.split_documents(documents)

print("Chunks created:", len(chunks))


# --------------------------------------------------
# CREATE / LOAD FAISS
# --------------------------------------------------

print("\n3. Preparing FAISS...")

if not os.path.exists(FAISS_PATH):

    faiss_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    faiss_store.save_local(FAISS_PATH)

    print("FAISS index created and saved.")

else:

    faiss_store = FAISS.load_local(
        FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    print("Existing FAISS index loaded.")


faiss_retriever = faiss_store.as_retriever(
    search_kwargs={"k": TOP_K}
)


# --------------------------------------------------
# CREATE / LOAD CHROMA
# --------------------------------------------------

print("\n4. Preparing Chroma...")

if not os.path.exists(CHROMA_PATH):

    chroma_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=CHROMA_COLLECTION,
        persist_directory=CHROMA_PATH
    )

    print("Chroma DB created and persisted.")

else:

    chroma_store = Chroma(
        collection_name=CHROMA_COLLECTION,
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    print("Existing Chroma DB loaded.")


chroma_retriever = chroma_store.as_retriever(
    search_kwargs={"k": TOP_K}
)


# --------------------------------------------------
# LLM + PROMPT
# --------------------------------------------------

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

parser = StrOutputParser()

prompt = PromptTemplate.from_template("""
You are a question-answering assistant.

Answer the user's EXACT question using ONLY the provided context.

Rules:
1. Do not answer a different or related question.
2. Do not guess or infer missing information.
3. If the exact answer is not explicitly available in the context,
   say exactly:
   "The provided document does not contain this information."
4. Keep the answer concise.

Context:
{context}

Question:
{question}

Answer:
""")

rag_chain = prompt | llm | parser


# --------------------------------------------------
# HELPER METHODS
# --------------------------------------------------

def format_docs(docs):
    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


def contains_expected_keyword(docs, keyword):

    if keyword is None:
        return None

    combined_text = format_docs(docs).lower()

    return keyword.lower() in combined_text


def evaluate_answer(answer, expected_keyword, should_abstain):

    expected_abstain_message = (
        "the provided document does not contain this information"
    )

    answer_lower = answer.lower()

    if should_abstain:
        return expected_abstain_message in answer_lower

    if expected_keyword is None:
        return True

    return expected_keyword.lower() in answer_lower


def evaluate_retriever(name, retriever, question):

    docs = retriever.invoke(question)

    print(f"\n===== {name} RETRIEVED CHUNKS =====")

    for index, doc in enumerate(docs, start=1):

        print(f"\n--- Chunk {index} ---")
        print(doc.page_content[:300])

        print("\nMetadata:")
        print(doc.metadata)

    return docs


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

results_summary = []

print("\n\n========================================")
print("STARTING RAG EVALUATION")
print("========================================")


for test_number, test in enumerate(test_cases, start=1):

    question = test["question"]
    expected_keyword = test["expected_keyword"]
    should_abstain = test["should_abstain"]

    print("\n\n########################################")
    print(f"TEST CASE {test_number}")
    print("########################################")

    print("\nQuestion:")
    print(question)

    print("\nExpected keyword:")
    print(expected_keyword)

    print("\nShould abstain:")
    print(should_abstain)


    # --------------------------------------------------
    # FAISS
    # --------------------------------------------------

    faiss_docs = evaluate_retriever(
        "FAISS",
        faiss_retriever,
        question
    )

    faiss_context = format_docs(faiss_docs)

    faiss_answer = rag_chain.invoke({
        "context": faiss_context,
        "question": question
    })

    print("\n===== FAISS FINAL ANSWER =====")
    print(faiss_answer)

    faiss_retrieval_pass = contains_expected_keyword(
        faiss_docs,
        expected_keyword
    )

    faiss_answer_pass = evaluate_answer(
        faiss_answer,
        expected_keyword,
        should_abstain
    )


    # --------------------------------------------------
    # CHROMA
    # --------------------------------------------------

    chroma_docs = evaluate_retriever(
        "CHROMA",
        chroma_retriever,
        question
    )

    chroma_context = format_docs(chroma_docs)

    chroma_answer = rag_chain.invoke({
        "context": chroma_context,
        "question": question
    })

    print("\n===== CHROMA FINAL ANSWER =====")
    print(chroma_answer)

    chroma_retrieval_pass = contains_expected_keyword(
        chroma_docs,
        expected_keyword
    )

    chroma_answer_pass = evaluate_answer(
        chroma_answer,
        expected_keyword,
        should_abstain
    )


    # --------------------------------------------------
    # SUMMARY
    # --------------------------------------------------

    results_summary.append({
        "question": question,
        "faiss_retrieval": faiss_retrieval_pass,
        "faiss_answer": faiss_answer_pass,
        "chroma_retrieval": chroma_retrieval_pass,
        "chroma_answer": chroma_answer_pass
    })


# --------------------------------------------------
# FINAL SUMMARY
# --------------------------------------------------

print("\n\n========================================")
print("FINAL EVALUATION SUMMARY")
print("========================================")


for index, result in enumerate(results_summary, start=1):

    print(f"\nTest {index}: {result['question']}")

    print(
        "FAISS Retrieval:",
        result["faiss_retrieval"]
    )

    print(
        "FAISS Answer:",
        result["faiss_answer"]
    )

    print(
        "Chroma Retrieval:",
        result["chroma_retrieval"]
    )

    print(
        "Chroma Answer:",
        result["chroma_answer"]
    )
