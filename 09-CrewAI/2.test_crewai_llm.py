from crewai import Agent, LLM

llm = LLM(
    model="ollama/llama3.2",
    base_url="http://localhost:11434",
    temperature=0
)

agent = Agent(
    role="Banking Assistant",
    goal="Explain banking concepts clearly",
    backstory="You are an experienced banking assistant.",
    llm=llm,
    verbose=True
)

result = agent.kickoff(
    "Explain a bank transaction in two sentences."
)

print(result)