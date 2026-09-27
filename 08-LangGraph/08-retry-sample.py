from typing import TypedDict, Literal

from langgraph.graph import StateGraph, START, END


class State(TypedDict, total=False):
    customer: str
    balance: int
    retry_count: int
    error: str
    status: str


def call_bank_api(state: State):
    print("\n--- Call Bank API ---")

    retry_count = state.get("retry_count", 0)

    # Simulate a temporary failure for the first 2 attempts
    if retry_count < 2:
        print("Bank API failed: HTTP 503")
        return {
            "retry_count": retry_count + 1,
            "error": "HTTP 503 Service Unavailable",
            "status": "failed",
        }

    print("Bank API succeeded.")
    return {
        "balance": 8000,
        "error": "",
        "status": "success",
    }


def decide_next(state: State) -> Literal["retry", "success", "fallback"]:

    if state["status"] == "success":
        return "success"

    if state["retry_count"] < 3:
        return "retry"

    return "fallback"


def fallback(state: State):
    print("\n--- Fallback ---")
    print("Bank API failed after maximum retries.")

    return {
        "status": "failed_permanently"
    }


builder = StateGraph(State)

builder.add_node("CallBankAPI", call_bank_api)
builder.add_node("Fallback", fallback)

builder.add_edge(START, "CallBankAPI")

builder.add_conditional_edges(
    "CallBankAPI",
    decide_next,
    {
        "retry": "CallBankAPI",
        "success": END,
        "fallback": "Fallback",
    },
)

builder.add_edge("Fallback", END)

graph = builder.compile()


initial_state = {
    "customer": "John",
    "retry_count": 0,
}

print("========== START ==========")

result = graph.invoke(initial_state)

print("\n========== FINAL STATE ==========")
print(result)