from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

print("1. Program Started")

# Step 1
prompt = PromptTemplate.from_template(
    "Explain {topic} in simple language."
)

print("2. Prompt Created")

# Step 2
llm = ChatOllama(model="llama3.2")

print("3. LLM Created")

# Step 3
chain = prompt | llm

print("4. Calling LLM...")

response = chain.invoke(
    {"topic": "Java Streams"}
)

print("\n5. Response Type")
print(type(response))

print("\n6. Complete Response")
print(response)

print("\n7. Only Content")
print(response.content)

print("\n8. Metadata")
print(response.response_metadata)

print("\n9. Usage Metadata")
print(response.usage_metadata)