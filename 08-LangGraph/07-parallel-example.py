from typing import Annotated, TypedDict
import operator

from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    customer: str
    #operator.add is reducer whis is used to combine the results from multiple nodes into a single list. 
    # This allows us to accumulate results from different branches of the graph.
    results: Annotated[list[str], operator.add]
    # error will occur if we use the following line instead of the above line
    #results: list[str]


def get_balance(state: State):
    print("Get Balance")
    return {
        "results": ["Balance = INR 8,000"]
    }


def get_transactions(state: State):
    print("Get Transactions")
    return {
        "results": ["Transactions = 5"]
    }


builder = StateGraph(State)

builder.add_node("GetBalance", get_balance)
builder.add_node("GetTransactions", get_transactions)

builder.add_edge(START, "GetBalance")
builder.add_edge(START, "GetTransactions")

builder.add_edge("GetBalance", END)
builder.add_edge("GetTransactions", END)

graph = builder.compile()


initial_state = {
    "customer": "John",
    "results": []
}

result = graph.invoke(initial_state)

print("\n========== FINAL STATE ==========")
print(result)