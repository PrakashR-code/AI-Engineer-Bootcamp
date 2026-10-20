from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader


def load_documents(data_path):
    documents = []

    pdf_files = list(Path(data_path).rglob("*.pdf"))

    if not pdf_files:
        print("No PDF files found.")
        return documents

    for pdf_file in pdf_files:
        print(f"Loading: {pdf_file.name}")

        loader = PyPDFLoader(str(pdf_file))
        docs = loader.load()

        documents.extend(docs)

    return documents