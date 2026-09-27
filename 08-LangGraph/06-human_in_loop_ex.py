from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver


class TransferState(TypedDict, total=False):
    customer: str
    amount: int
    approval: str
    result: str


def prepare_transfer(state: TransferState):

    print("\n--- Preparing Transfer ---")

    print(
        f"Transfer INR {state['amount']:,} "
        f"for customer {state['customer']}"
    )

    approval = interrupt(
        {
            "message": "Approve this transfer?",
            "customer": state["customer"],
            "amount": state["amount"],
        }
    )

    return {
        "approval": approval
    }


def execute_transfer(state: TransferState):

    print("\n--- Execute Transfer ---")

    if state["approval"] == "yes":
        print("Transfer executed.")
        return {
            "result": "Transfer successful."
        }

    print("Transfer rejected.")

    return {
        "result": "Transfer rejected."
    }


graph_builder = StateGraph(TransferState)

graph_builder.add_node(
    "PrepareTransfer",
    prepare_transfer
)

graph_builder.add_node(
    "ExecuteTransfer",
    execute_transfer
)

graph_builder.add_edge(
    START,
    "PrepareTransfer"
)

graph_builder.add_edge(
    "PrepareTransfer",
    "ExecuteTransfer"
)

graph_builder.add_edge(
    "ExecuteTransfer",
    END
)


checkpointer = InMemorySaver()

graph = graph_builder.compile(
    checkpointer=checkpointer
)


config = {
    "configurable": {
        "thread_id": "transfer-001"
    }
}


initial_state = {
    "customer": "John",
    "amount": 200000
}


print("========== START ==========")

result = graph.invoke(
    initial_state,
    config=config
)

print("\nGraph paused.")
print(result)



# Human has approved
print("\n========== RESUMING ==========")

result = graph.invoke(
    Command(resume="yes"),
    config=config
)

print("\n========== FINAL RESULT ==========")
print(result)