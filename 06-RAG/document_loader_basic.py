from langchain_community.document_loaders import PyPDFLoader


print("1. Loading PDF...")

loader = PyPDFLoader(
    "../06-RAG/data/Mamatha_Resume.pdf"
)

documents = loader.load()

print("\n2. Number of Documents")
print(len(documents))

print("\n3. First Document Type")
print(type(documents[0]))

print("\n4. First Document Content")
print(documents[0].page_content)

print("\n5. First Document Metadata")
print(documents[0].metadata)