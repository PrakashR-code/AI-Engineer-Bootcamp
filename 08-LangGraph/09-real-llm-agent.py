from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama

# Use the same LLM initialization you already use in your project.
"""
                 LLM
                  |
          +-------+-------+
          |               |
          ↓               ↓
    get_balance     get_transactions
          |               |
          +-------+-------+
                  |
              ToolNode
                  |
                  ↓
                 LLM
                  |
                  ↓
            Final Response
"""
# --------------------------------------------------
# 1. Tools
# --------------------------------------------------

CUSTOMERS = {
    "John": {
        "account_number": "A123",
        "balance": 8000,
        "transactions": [
            "ATM withdrawal - INR 2,000",
            "UPI payment - INR 1,500",
            "NEFT credit - INR 5,000",
        ],
    },
    "Ravi": {
        "account_number": "A456",
        "balance": 25000,
        "transactions": [
            "UPI payment - INR 500",
            "Salary credit - INR 50,000",
        ],
    },
}


@tool
def get_balance(customer_name: str) -> str:
    """Get the current bank balance for a customer."""

    customer = CUSTOMERS.get(customer_name)

    if not customer:
        return f"Customer {customer_name} was not found."

    return (
        f"Customer: {customer_name}, "
        f"Account: {customer['account_number']}, "
        f"Balance: INR {customer['balance']:,}"
    )


@tool
def get_transactions(customer_name: str) -> str:
    """Get recent transactions for a customer."""

    customer = CUSTOMERS.get(customer_name)

    if not customer:
        return f"Customer {customer_name} was not found."

    transactions = "\n".join(
        f"- {transaction}"
        for transaction in customer["transactions"]
    )

    return (
        f"Recent transactions for {customer_name}:\n"
        f"{transactions}"
    )


tools = [get_balance, get_transactions]


# --------------------------------------------------
# 2. LLM
# --------------------------------------------------

# Replace this with your existing LLM configuration.
#
# Example:

llm = ChatOllama(
        model="llama3.2",
        temperature=0
    )
# IMPORTANT:
# The following assumes that `llm` has been initialized.

llm_with_tools = llm.bind_tools(tools)


# --------------------------------------------------
# 3. State
# --------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# --------------------------------------------------
# 4. LLM Node
# --------------------------------------------------

def llm_node(state: AgentState):

    print("\n--- LLM NODE ---")

    response = llm_with_tools.invoke(state["messages"])

    if response.tool_calls:
        print("LLM selected tool(s):")

        for tool_call in response.tool_calls:
            print(
                f"  Tool: {tool_call['name']}, "
                f"Args: {tool_call['args']}"
            )
    else:
        print("LLM generated final response.")

    return {
        "messages": [response]
    }


# --------------------------------------------------
# 5. Conditional Routing
# --------------------------------------------------

def route_after_llm(state: AgentState):

    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tools"

    return "end"


# --------------------------------------------------
# 6. Build Graph
# --------------------------------------------------

builder = StateGraph(AgentState)

builder.add_node("LLM", llm_node)
builder.add_node("Tools", ToolNode(tools))

builder.add_edge(START, "LLM")

builder.add_conditional_edges(
    "LLM",
    route_after_llm,
    {
        "tools": "Tools",
        "end": END,
    },
)

builder.add_edge("Tools", "LLM")

graph = builder.compile()


# --------------------------------------------------
# 7. Run
# --------------------------------------------------

if __name__ == "__main__":

    print("========== AGENT START ==========")

    #user_request = "What is John's current balance?"
    user_request = "What is John's current balance and recent transactions?"

    result = graph.invoke(
        {
            "messages": [
                HumanMessage(content=user_request)
            ]
        }
    )

    print("\n========== FINAL RESPONSE ==========")

    final_message = result["messages"][-1]

    print(final_message.content)