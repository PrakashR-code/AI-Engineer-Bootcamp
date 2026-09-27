from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver


class State(TypedDict, total=False):
    customer_name: str
    balance: int


def get_balance(state: State):
    print("Getting balance...")

    return {
        "balance": 8000
    }


graph_builder = StateGraph(State)

graph_builder.add_node("GetBalance", get_balance)

graph_builder.add_edge(START, "GetBalance")
graph_builder.add_edge("GetBalance", END)


# Create checkpointer
checkpointer = InMemorySaver()


# Compile graph with checkpointer
graph = graph_builder.compile(
    checkpointer=checkpointer
)


initial_state = {
    "customer_name": "John"
}


config = {
    "configurable": {
        "thread_id": "customer-001"
    }
}


result = graph.invoke(
    initial_state,
    config=config
)


print(result)