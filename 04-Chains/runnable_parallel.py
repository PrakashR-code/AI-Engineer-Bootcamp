from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableParallel

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)
"""
                Java Streams
                      │
          ┌───────────┴───────────┐
          ↓                       ↓
   Explanation Chain       Interview Chain
          ↓                       ↓
     Explanation              Questions
          └───────────┬───────────┘
                      ↓
                   Dictionary
                       """
parser = StrOutputParser()

explanation_prompt = PromptTemplate.from_template(
    "Explain {topic} in simple terms."
)

question_prompt = PromptTemplate.from_template(
    "Create 3 interview questions about {topic}."
)

explanation_chain = explanation_prompt | llm | parser
question_chain = question_prompt | llm | parser

"""
      ┌→ explanation_chain
A ────┤
      └→ question_chain

RunnableParallel executes multiple independent Runnables using the same input and combines their outputs into a dictionary. 
I would use it when operations don't depend on each other, whereas I use a sequential chain when one step requires the output 
of a previous step.
"""

parallel_chain = RunnableParallel(
    explanation=explanation_chain,
    questions=question_chain
)



response = parallel_chain.invoke({
    "topic": "Java Streams"
})

print(type(response))

print("\n===== EXPLANATION =====")
print(response["explanation"])

print("\n===== QUESTIONS =====")
print(response["questions"])