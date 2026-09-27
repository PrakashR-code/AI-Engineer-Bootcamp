from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


# -----------------------------
# 1. State Schema
# -----------------------------
class State(TypedDict):
    customer_name: str
    account_number: str
    balance: float
    email_sent: bool


# -----------------------------
# 2. Node: Find Customer
# -----------------------------
def find_customer(state: State):
    print(f"FindCustomer: finding {state['customer_name']}")

    return {
        "account_number": "A123"
    }


# -----------------------------
# 3. Node: Get Balance
# -----------------------------
def get_balance(state: State):
    print(f"GetBalance: checking account {state['account_number']}")

    balance = 8000

    print(f"GetBalance: INR {balance:,}")

    return {
        "balance": balance
    }


# -----------------------------
# 4. Build the Graph
# -----------------------------
builder = StateGraph(State)


# Add Nodes
builder.add_node("find_customer", find_customer)
builder.add_node("get_balance", get_balance)


# Add Edges
builder.add_edge(START, "find_customer")
builder.add_edge("find_customer", "get_balance")
builder.add_edge("get_balance", END)


# -----------------------------
# 5. Compile Graph
# -----------------------------
graph = builder.compile()


# -----------------------------
# 6. Execute Graph
# -----------------------------
print("\n=== Running LangGraph ===")

result = graph.invoke({
    "customer_name": "John",
    "account_number": "",
    "balance": 0,
    "email_sent": False
})


# -----------------------------
# 7. Print Final State
# -----------------------------
print("\n=== Final State ===")
print(result)