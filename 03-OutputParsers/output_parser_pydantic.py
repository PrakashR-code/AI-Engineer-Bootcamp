from pydantic import BaseModel

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_ollama import ChatOllama


class InterviewResponse(BaseModel):
    topic: str
    definition: str
    banking_example: str
    interview_question: str
    best_answer: str


print("1. Program Started")

parser = PydanticOutputParser(
    pydantic_object=InterviewResponse
)

prompt = PromptTemplate.from_template(
"""
You are a Senior Java Interviewer.

Generate ACTUAL DATA about this topic:

Topic: {topic}

IMPORTANT:
- Return an INSTANCE matching the schema below.
- Populate every field with actual values.
- DO NOT return the JSON schema itself.
- DO NOT return fields such as "properties", "title", "type", or "description" as schema metadata.
- DO NOT add markdown or explanations outside the JSON.
- Return only the populated object.

{format_instructions}
""",
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

print("2. Prompt Created")

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

print("3. LLM Created")

chain = prompt | llm | parser

print("4. Calling LLM...")

response = chain.invoke(
    {"topic": "Java Streams"}
)

print("\n5. Response Type")
print(type(response))

print("\n6. Complete Object")
print(response)

print("\n7. Topic")
print(response.topic)

print("\n8. Definition")
print(response.definition)

print("\n9. Banking Example")
print(response.banking_example)

print("\n10. Interview Question")
print(response.interview_question)

print("\n11. Best Answer")
print(response.best_answer)