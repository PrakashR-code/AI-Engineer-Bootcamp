
from crewai import Agent, Task, Crew, Process, LLM

"""
============================================================
CREWAI BANKING INVESTIGATION - SEQUENTIAL TASK CONTEXT
============================================================

PURPOSE:
--------
Understand how multiple CrewAI agents execute tasks in
sequence and how the output of one task becomes the context
for another task.

CURRENT EXAMPLE:
----------------
We are using hardcoded transaction facts to learn task
orchestration and context passing. The transaction search
tool is NOT called in this example.

EXECUTION FLOW:
---------------

1. LLM CONFIGURATION
   - Connect CrewAI to the local Ollama Llama 3.2 model.
   - The same LLM configuration is used by both agents.

2. AGENTS ARE CREATED
   - Transaction Investigator:
       Summarizes the supplied transaction facts.
   - Risk Analyst:
       Reviews the facts and identifies potential risk
       indicators without declaring fraud without evidence.

   NOTE:
   Creating an Agent defines its role and capabilities.
   It does not start execution.

3. INVESTIGATION TASK IS CREATED
   - TXN102 and its sample transaction facts are supplied
     directly inside investigation_task.description.
   - The Investigator receives this task and produces
     a transaction summary.

4. RISK ANALYSIS TASK IS CREATED
   - The Risk Analyst receives its own task instructions.
   - context=[investigation_task] tells CrewAI to make the
     earlier investigation task's output available here.
   - The Risk Analyst uses that output for its assessment.

5. CREW IS CREATED
   - Both agents and both tasks are registered with Crew.
   - Process.sequential means tasks execute in the order
     specified in the Crew's task list.

6. EXECUTION STARTS
   - crew.kickoff() starts the workflow.
   - Task 1 runs first, followed by Task 2.
   - The final Crew result is stored in 'result'.

7. FINAL OUTPUT
   - print(result) displays the final risk assessment.

FLOW SUMMARY:
-------------

TXN102 + HARDCODED FACTS
          |
          v
  crew.kickoff()
          |
          v
  INVESTIGATION TASK
  Agent: Transaction Investigator
          |
          | Produces investigation summary
          v
  RISK ANALYSIS TASK
  Agent: Risk Analyst
  context=[investigation_task]
          |
          v
  FINAL RISK ASSESSMENT
          |
          v
     print(result)

KEY CONCEPTS:
-------------

Agent   = WHO performs the work
Task    = WHAT work must be done
Crew    = Coordinates agents and tasks
kickoff = STARTS the workflow
context = Makes an earlier task's output available
          to a later task

IMPORTANT:
----------
- TXN102 is currently hardcoded in the task description.
- No custom transaction tool is used in this example.
- The Risk Analyst depends on the Investigator's output.
- Sequential execution is controlled by Process.sequential.
- LLM-generated risk assessments must be validated; a
  large international transfer alone does not prove fraud.

NEXT IMPROVEMENT:
-----------------
Pass the transaction ID as a workflow input and let the
Investigator retrieve the actual transaction details through
the custom search_transaction tool.
============================================================
"""


llm = LLM(
    model="ollama/llama3.2",
    base_url="http://localhost:11434",
    temperature=0
)

investigator = Agent(
    role="Transaction Investigator",
    goal="Summarize the supplied transaction facts",
    backstory="You investigate banking transactions using verified data.",
    llm=llm,
    verbose=True
)

risk_analyst = Agent(
    role="Risk Analyst",
    goal="Identify potential risk indicators from transaction facts",
    backstory="You assess banking transaction risk cautiously.",
    llm=llm,
    verbose=True
)

investigation_task = Task(
    description=(
        "Summarize these sample facts without adding information: "
        "TXN102, international transfer, INR 200000, "
        "Singapore, status Completed."
    ),
    expected_output="A concise summary of the supplied transaction facts.",
    agent=investigator
)

risk_task = Task(
    description=(
        "Review the transaction facts provided by the investigator. "
        "Identify potential risk indicators, explain the limitations "
        "of the evidence, and do not declare fraud based only on "
        "the amount or destination."
    ),
    expected_output="A cautious risk assessment with supporting facts.",
    agent=risk_analyst,
    context=[investigation_task]
)

crew = Crew(
    agents=[investigator, risk_analyst],
    tasks=[investigation_task, risk_task],
    process=Process.sequential,
    verbose=True
)

result = crew.kickoff()
print("\n--- FINAL RISK ASSESSMENT ---")
print(result)
