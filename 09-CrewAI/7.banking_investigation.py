"""CrewAI banking investigation using a local Ollama model.

Workflow: investigate TXN102 with the search tool, pass that task output to
risk analysis, then write a concise report. The final report is checked against
source data using transparent deterministic rules (not semantic understanding).
"""

from crewai import Agent, Crew, LLM, Process, Task

from banking_tools import TRANSACTIONS, search_transaction
from banking_validation import validate_agent_report, validate_transaction


# 1. Connect CrewAI to the local Ollama server. Ollama must be running and
#    the selected model must already be available locally.
llm = LLM(
    model="ollama/llama3.2",
    base_url="http://localhost:11434",
    temperature=0,
)

# 2. Define the three agents. The investigator can call the search tool;
#    the analyst and writer receive earlier task outputs as context below.
investigator = Agent(
    role="Transaction Investigator",
    goal="Retrieve and accurately summarize transaction details",
    backstory=(
        "You are a banking investigator. Use the transaction search tool to "
        "retrieve facts. Never invent transaction details."
    ),
    tools=[search_transaction],
    llm=llm,
    verbose=True,
    max_iter=3,
)

risk_analyst = Agent(
    role="Risk Analyst",
    goal="Assess potential risk indicators using verified facts",
    backstory=(
        "You assess banking transaction risk cautiously. Distinguish observed "
        "facts from assumptions. Do not declare fraud without sufficient evidence."
    ),
    llm=llm,
    verbose=True,
    max_iter=3,
)

report_writer = Agent(
    role="Banking Investigation Report Writer",
    goal="Produce a concise, evidence-based investigation report",
    backstory=(
        "You write clear banking investigation reports. Distinguish verified "
        "facts from potential concerns. Never invent details or present speculation "
        "as a confirmed finding."
    ),
    llm=llm,
    verbose=True,
    max_iter=3,
)

# 3. Define the sequential tasks. CrewAI passes each task's result forward;
#    `context` explicitly supplies the prior task output to the next agent.
investigation_task = Task(
    description=(
        "Investigate transaction ID {transaction_id} using the Transaction Search "
        "tool. Report the transaction ID, type, amount, country, and status exactly "
        "as returned. Do not invent missing fields; report clearly if not found."
    ),
    expected_output=(
        "A factual summary of the transaction ID, type, amount, country, and status "
        "returned by the tool, without unsupported details."
    ),
    agent=investigator,
)

# 4. The analyst reviews the investigator's retrieved facts and clearly
#    separates evidence from unknowns.
risk_task = Task(
    description=(
        "Assess only facts supported by the investigation findings. Do not call an "
        "amount high without a defined threshold or customer baseline. Do not infer "
        "money laundering from an international transfer, destination country, or "
        "completed status alone. Separate verified facts, evidence-based concerns, "
        "and missing information. Never declare fraud without sufficient evidence."
    ),
    expected_output=(
        "A cautious assessment separating verified facts, supported concerns, and "
        "additional evidence needed. Do not make unsupported fraud conclusions."
    ),
    agent=risk_analyst,
    context=[investigation_task],
)

# 5. The report writer turns both earlier outputs into the final report.
report_task = Task(
    description=(
        "Write a concise, evidence-based investigation report using the prior task "
        "outputs. Preserve verified transaction facts. Do not describe an amount as "
        "unusually high without a defined threshold or customer baseline. An "
        "international transfer or completed status alone is not evidence of money "
        "laundering. Separate facts, supported concerns, missing evidence, and next "
        "steps. Never invent facts or present speculation as a finding."
    ),
    expected_output=(
        "A structured report with verified facts, evidence-based concerns, missing "
        "evidence, and practical next steps."
    ),
    agent=report_writer,
    context=[risk_task],
)

# 6. Configure one sequential crew: investigation -> risk assessment -> report.
crew = Crew(
    agents=[investigator, risk_analyst, report_writer],
    tasks=[investigation_task, risk_task, report_task],
    process=Process.sequential,
    verbose=True,
)


def main() -> int:
    # 7. Validate the local source record first. Stop if it is missing or
    #    incomplete so the agents do not build a report on invalid input.
    transaction_id = "TXN102"
    source_check = validate_transaction(transaction_id, TRANSACTIONS)
    if not source_check["valid"]:
        print("Source validation failed:")
        for error in source_check["errors"]:
            print(f"- {error}")
        return 1

    # 8. Start the workflow with TXN102 as a template input. The task text
    #    substitutes this value for {transaction_id}.
    verified_transaction = source_check["verified_transaction"]
    result = crew.kickoff(inputs={"transaction_id": transaction_id})
    final_report = str(result)

    # Print the generated report once.
    print("\n--- FINAL INVESTIGATION REPORT ---")
    print(final_report)

    # 9. Check the actual generated report against the same verified source.
    #    This is a deterministic field/phrase check, not semantic review.
    validation = validate_agent_report(final_report, verified_transaction)
    print("\n--- REPORT VALIDATION ---")
    print(f"Valid: {validation['valid']}")
    for error in validation["errors"]:
        print(f"Issue: {error}")
    print(f"Note: {validation['limitations']}")
    return 0 if validation["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
