from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

class InterviewResponse(BaseModel):
    topic: str = Field(description="The topic")
    definition: str = Field(description="Short definition")
    advantages: List[str] = Field(description="List of advantages")
    banking_example: str = Field(description="Banking example")
    interview_question: str = Field(description="Interview question")

print("1. Program Started")

parser = PydanticOutputParser(
    pydantic_object=InterviewResponse
)

prompt = PromptTemplate.from_template("""
You are a JSON API.

Return ONLY one valid JSON object.
Do not add explanations.
Do not add markdown.
Do not use code fences.
Do not add any text before or after the JSON.

Topic: {topic}

Return exactly this structure:

{{
  "topic": "{topic}",
  "definition": "short definition",
  "advantages": [
    "advantage 1",
    "advantage 2",
    "advantage 3"
  ],
  "banking_example":"some",
  "interview_question":"somne"
}}
""",

    input_variables=["topic"],

    partial_variables={
        "format_instructions":
        parser.get_format_instructions()
    })

print("2. Prompt Created")

llm = ChatOllama(model="llama3.2",
    temperature=1)

print("3. LLM Created")



chain = prompt | llm | parser

print("4. Calling LLM...")

response = chain.invoke(
    {
        "topic":"Java Streams"
    }
)

print("\n5. Response Type")
print(type(response))

print("\n6. Complete Response")
print(response)

print("\n7. Only Content")
#print(response["topic"])

print("\n8. Definition")
#print(response["definition"])

print("\n9. Advantages")
#print(response["advantages"])




