from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

print("1. Loading PDF...")

loader = PyPDFLoader(
    "../06-RAG/data/Mamatha_Resume.pdf"
)

documents = loader.load()

print("\n2. Pages loaded:")
print(len(documents))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)

print("\n3. Number of chunks:")
print(len(chunks))

print("\n4. First chunk type:")
print(type(chunks[0]))

# chars in first chunk
first_len = len(chunks[0].page_content)
print("chars in chunks[0].page_content:", first_len)

# total chars across all chunks
total_len = sum(len(c.page_content or "") for c in chunks)
print("total chars across all chunks:", total_len)

# optional: list of lengths per chunk
lengths = [len(c.page_content or "") for c in chunks]
print("chars per chunk (first 10):", lengths[:10])

print("\n5. First chunk content:")
print(chunks[0].page_content)

print("\n6. First chunk metadata:")
print(chunks[0].metadata)

print("\n7. Second chunk content:")
print(chunks[1].page_content)