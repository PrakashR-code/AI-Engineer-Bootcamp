
from crewai import Agent, LLM
from banking_tools import search_transaction

# Connect CrewAI to your local Ollama model
llm = LLM(
    model="ollama/llama3.2",
    base_url="http://localhost:11434",
    temperature=0
)

# Create an agent with access to the transaction tool
investigator = Agent(
    role="Bank Transaction Investigator",
    goal="Retrieve and explain transaction details accurately",
    backstory=(
        "You are a banking investigator. "
        "Always use the transaction search tool to retrieve "
        "transaction details. Never invent transaction data."
    ),
    tools=[search_transaction],
    llm=llm,
    verbose=True,
    max_iter=3
)

# Ask the agent to investigate a transaction
result = investigator.kickoff(
    "Investigate transaction TXN999. "
    "Retrieve its details and summarize the transaction."
)

print("\n--- INVESTIGATION RESULT ---")
print(result)
