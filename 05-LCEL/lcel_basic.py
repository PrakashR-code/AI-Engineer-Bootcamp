from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

print("1. Program started")

prompt = PromptTemplate.from_template(
    """
    You are an expert programming trainer.

    Explain {topic} for a {level} developer.

    Give:
    1. Simple definition
    2. Code example
    3. Common interview question
    """
)
print("2. Prompt created")

llm = ChatOllama(model="llama3.2")
print("3. Ollama client created")

parser = StrOutputParser()

chain = prompt | llm | parser
print("4. LCEL chain created")

print("5. Calling Llama 3.2...")

response = chain.invoke(
    {
    "topic": "Kafka",
    "level": "Advanced"
}
)

print("6. Response received")
print(response)