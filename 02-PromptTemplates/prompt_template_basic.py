from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template(
    "Explain {topic} in simple language."
)

formatted_prompt = prompt.invoke(
    {"topic": "Java Streams"}
)

print(formatted_prompt)
print(formatted_prompt.to_string())