from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END


# -----------------------------
# 1. STATE
# -----------------------------
class AgentState(TypedDict, total=False):
    customer_name: str
    account_number: str
    balance: int
    transactions: list
    next_action: str
    response: str


# -----------------------------
# 2. SAMPLE CUSTOMER DATA
# -----------------------------
CUSTOMERS = {
    "John": {
        "account_number": "A123",
        "balance": 8000,
        "transactions": [
            "ATM withdrawal - INR 2,000",
            "UPI payment - INR 500",
            "Salary credit - INR 50,000",
        ],
    }
}


# -----------------------------
# 3. LLM NODE
# -----------------------------
def llm_node(state: AgentState):

    print("\n--- LLM Node ---")

    customer_name = state["customer_name"]

    # First decision
    if "balance" not in state:
        print("LLM decision: call get_balance")
        return {
            "next_action": "get_balance"
        }

    # Second decision
    if "transactions" not in state:
        print("LLM decision: call get_transactions")
        return {
            "next_action": "get_transactions"
        }

    # Final decision
    print("LLM decision: task completed")

    return {
        "next_action": "finish",
        "response": (
            f"{customer_name}'s balance is INR "
            f"{state['balance']:,}. "
            f"Recent transactions were retrieved."
        ),
    }


# -----------------------------
# 4. GET BALANCE TOOL
# -----------------------------
def get_balance_tool(state: AgentState):

    print("\n--- Get Balance Tool ---")

    customer = CUSTOMERS[state["customer_name"]]

    print(f"Balance: INR {customer['balance']:,}")

    return {
        "account_number": customer["account_number"],
        "balance": customer["balance"],
    }


# -----------------------------
# 5. GET TRANSACTIONS TOOL
# -----------------------------
def get_transactions_tool(state: AgentState):

    print("\n--- Get Transactions Tool ---")

    customer = CUSTOMERS[state["customer_name"]]

    transactions = customer["transactions"]

    for transaction in transactions:
        print(transaction)

    return {
        "transactions": transactions
    }


# -----------------------------
# 6. ROUTING FUNCTION
# -----------------------------
def choose_next(
    state: AgentState,
) -> Literal["get_balance", "get_transactions", "finish"]:

    return state["next_action"]


# -----------------------------
# 7. BUILD GRAPH
# -----------------------------
graph_builder = StateGraph(AgentState)

graph_builder.add_node("LLM", llm_node)
graph_builder.add_node("GetBalance", get_balance_tool)
graph_builder.add_node("GetTransactions", get_transactions_tool)


# START → LLM
graph_builder.add_edge(START, "LLM")


# LLM → decision
graph_builder.add_conditional_edges(
    "LLM",
    choose_next,
    {
        "get_balance": "GetBalance",
        "get_transactions": "GetTransactions",
        "finish": END,
    },
)


# IMPORTANT:
# Tool → LLM
graph_builder.add_edge("GetBalance", "LLM")
graph_builder.add_edge("GetTransactions", "LLM")


# Compile
graph = graph_builder.compile()


# -----------------------------
# 8. RUN
# -----------------------------
if __name__ == "__main__":

    initial_state = {
        "customer_name": "John"
    }

    print("========== AGENT START ==========")

    final_state = graph.invoke(initial_state)

    print("\n========== FINAL STATE ==========")
    print(final_state)

    print("\n========== FINAL RESPONSE ==========")
    print(final_state["response"])