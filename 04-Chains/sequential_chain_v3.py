from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)
"""
              "Find duplicate numbers..."
                         │
               ┌─────────┴──────────┐
               │                    │
               ▼                    ▼
     RunnablePassthrough      developer_chain
               │                    │
               │                 Java code
               │                    │
               └─────────┬──────────┘
                         ▼
                  {
                    requirement: original text,
                    generated_code: generated code
                  }
                         │
                         ▼
                  reviewer_prompt
                         │
                         ▼
                       LLM
                       """
parser = StrOutputParser()

developer_prompt = PromptTemplate.from_template("""
You are a senior Java developer.

Write simple Java code for the following requirement:

{requirement}

Return the Java code with a short explanation.
""")

reviewer_prompt = PromptTemplate.from_template("""
You are a senior Java code reviewer.

Original Requirement:
{requirement}

Generated Java Code:
{generated_code}

Review whether the code satisfies the original requirement.

Check:
1. Correctness
2. Readability
3. Performance
4. Java best practices
5. Possible bugs
6. Whether the requirement is fully satisfied

Give a concise review.
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

final_chain = (
    {
        "requirement": RunnablePassthrough(),
        "generated_code": developer_chain
    }
    | reviewer_prompt
    | llm
    | parser
)

final_review = final_chain.invoke(
    "Find duplicate numbers in a List using Java Streams"
)

print("\n===== FINAL REVIEW =====")
print(final_review)