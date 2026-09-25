from typing import TypedDict, Literal

from langgraph.graph import StateGraph, START, END

"""
                    ┌──────────────┐
                    │    START     │
                    └──────┬───────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │    LLM Node     │
                  │                 │
                  │ Read State      │
                  │ Decide action   │
                  └────────┬────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │  Conditional Edge │
                 │                   │
                 │ choose_next()     │
                 └────────┬──────────┘
                          │
                 ┌────────┴─────────┐
                 │                  │
        get_balance               finish
                 │                  │
                 ▼                  ▼
       ┌─────────────────┐     ┌─────────┐
       │   Tool Node     │     │   END   │
       │                 │     └─────────┘
       │ Get Balance     │
       │                 │
       │ Update State    │
       └────────┬────────┘
                │
                ▼
             ┌───────┐
             │  END  │
             └───────┘"""
# ---------------------------------------------------------
# 1. STATE SCHEMA
# ---------------------------------------------------------

class AgentState(TypedDict, total=False):
    customer_name: str
    account_number: str
    balance: int
    user_request: str
    next_action: str
    response: str


# ---------------------------------------------------------
# 2. SAMPLE CUSTOMER DATA
# ---------------------------------------------------------

CUSTOMERS = {
    "John": {
        "account_number": "A123",
        "balance": 8000,
    },
    "Ravi": {
        "account_number": "A456",
        "balance": 25000,
    },
}


# ---------------------------------------------------------
# 3. LLM NODE
# ---------------------------------------------------------

def llm_node(state: AgentState):
    """
    Simulates an LLM deciding what action to take.
    """

    print("\n--- LLM Node ---")

    customer_name = state["customer_name"]

    if customer_name in CUSTOMERS:
        print("LLM decision: call get_balance tool")
        return {
            "next_action": "get_balance"
        }

    print("LLM decision: customer not found")
    return {
        "next_action": "finish",
        "response": "Customer not found."
    }


# ---------------------------------------------------------
# 4. TOOL NODE
# ---------------------------------------------------------

def get_balance_tool(state: AgentState):
    """
    Simulates an external banking tool.
    """

    print("\n--- Tool Node ---")

    customer_name = state["customer_name"]

    customer = CUSTOMERS[customer_name]

    account_number = customer["account_number"]
    balance = customer["balance"]

    print(f"Tool: Account = {account_number}")
    print(f"Tool: Balance = INR {balance:,}")

    return {
        "account_number": account_number,
        "balance": balance,
        "response": f"The balance of {customer_name} is INR {balance:,}."
    }


# ---------------------------------------------------------
# 5. CONDITIONAL ROUTING
# ---------------------------------------------------------

def choose_next(
    state: AgentState,
) -> Literal["get_balance", "finish"]:

    if state["next_action"] == "get_balance":
        return "get_balance"

    return "finish"


# ---------------------------------------------------------
# 6. BUILD GRAPH
# ---------------------------------------------------------

graph_builder = StateGraph(AgentState)

graph_builder.add_node("LLM", llm_node)
graph_builder.add_node("GetBalanceTool", get_balance_tool)

graph_builder.add_edge(START, "LLM")

graph_builder.add_conditional_edges(
    "LLM",
    choose_next,
    {
        "get_balance": "GetBalanceTool",
        "finish": END,
    },
)

graph_builder.add_edge("GetBalanceTool", END)


# ---------------------------------------------------------
# 7. COMPILE
# ---------------------------------------------------------

graph = graph_builder.compile()


# ---------------------------------------------------------
# 8. RUN EXAMPLE
# ---------------------------------------------------------

if __name__ == "__main__":

    initial_state = {
        "customer_name": "John",
        "user_request": "What is John's balance?",
    }

    print("========== AGENT START ==========")

    final_state = graph.invoke(initial_state)

    print("\n========== FINAL STATE ==========")
    print(final_state)

    print("\n========== FINAL RESPONSE ==========")
    print(final_state["response"])