from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

parser = StrOutputParser()

developer_prompt = PromptTemplate.from_template("""
You are a senior Java developer.

Write simple Java code for the following requirement:

{requirement}

Return the Java code with a short explanation.
""")

reviewer_prompt = PromptTemplate.from_template("""
You are a senior Java code reviewer.

Review the following Java code:

{generated_code}

Review it for:
1. Correctness
2. Readability
3. Performance
4. Java best practices
5. Possible improvements

Give a concise code review.
""")

reviewer_chain = reviewer_prompt | llm | parser

dev_llm_chain = developer_prompt | llm
developer_chain = dev_llm_chain | parser

requirement = "Find duplicate numbers in a List using Java Streams"

# Render and print the prompt
prompt_text = developer_prompt.format(requirement=requirement)
print("===== PROMPT SENT TO LLM =====")
print(prompt_text)

# Invoke prompt -> LLM and show raw LLM output
llm_output = dev_llm_chain.invoke({"requirement": requirement})
print("===== RAW LLM OUTPUT =====")
print(llm_output)

# Invoke full chain (LLM + parser) and show final parsed output
code = developer_chain.invoke({"requirement": requirement})
print("===== DEVELOPER OUTPUT =====")
print(code)

print("\n===== CALLING REVIEWER =====")

review = reviewer_chain.invoke({
    "generated_code": code
})

print("\n===== REVIEWER OUTPUT =====")
print(review)