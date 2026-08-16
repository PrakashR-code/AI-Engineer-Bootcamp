from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama


print("1. Program Started")

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

parser = StrOutputParser()


# Normal Python function
def clean_response(text):
    print("\n--- RunnableLambda received LLM output ---")
    print(type(text))

    return text.strip().upper()


# Convert normal Python function into a Runnable
cleaner = RunnableLambda(clean_response)


prompt = PromptTemplate.from_template(
    """
    Explain {topic} in exactly 2 short sentences.
    """
)
"""
|                         → sequence
RunnablePassthrough       → preserve input
.assign()                 → preserve + enrich
RunnableParallel          → independent parallel branches
RunnableLambda            → custom Python logic

RunnableLambda wraps a normal Python function so that custom application logic can participate in an LCEL pipeline.

FLOW:
-----
{"topic": "Java Streams"}
          ↓
    PromptTemplate
          ↓
      ChatOllama
          ↓
      AIMessage
          ↓
   StrOutputParser
          ↓
        str
          ↓
   RunnableLambda
          ↓
 clean_response(text)
          ↓
    cleaned string
"""

chain = (
    prompt
    | llm
    | parser
    | cleaner
)


print("2. Calling Chain...")

response = chain.invoke({
    "topic": "Java Streams"
})


print("\n===== FINAL RESPONSE =====")
print(response)