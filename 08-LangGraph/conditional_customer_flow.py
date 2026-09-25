from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph


#State Schema for the customer flow graph. The state is a dictionary with optional keys for customer_id, customer_name, balance, email_message, and path. The path key is a list of strings that tracks the nodes visited in the graph.
class CustomerState(TypedDict, total=False):
    customer_id: str
    customer_name: str
    balance: int
    email_message: str
    path: list[str]


CUSTOMERS = {
    "C001": {"name": "Anika", "balance": 7_500},
    "C002": {"name": "Ravi", "balance": 15_000},
}
ALERT_THRESHOLD = 10_000


def find_customer(state: CustomerState) -> CustomerState:
    """Look up the requested customer in our small in-memory sample."""
    customer_id = state["customer_id"]
    customer = CUSTOMERS[customer_id]
    print(f"FindCustomer: found {customer['name']} ({customer_id})")
    return {
        "customer_name": customer["name"],
        "balance": customer["balance"],
        "path": [*state.get("path", []), "FindCustomer"],
    }


def get_balance(state: CustomerState) -> CustomerState:
    """Display the balance retrieved for the customer."""
    balance = state["balance"]
    print(f"GetBalance: INR {balance:,}")
    return {"path": [*state.get("path", []), "GetBalance"]}


def choose_next(
    state: CustomerState,
) -> Literal["send_email", "finish"]:
    """Route low balances to the email node; other balances finish."""
    if state["balance"] < ALERT_THRESHOLD:
        return "send_email"
    return "finish"


def send_email(state: CustomerState) -> CustomerState:
    """Simulate composing an alert; this example sends no real email."""
    message = (
        f"To: {state['customer_name']}\n"
        f"Subject: Low balance alert\n"
        f"Your balance is INR {state['balance']:,}, below INR {ALERT_THRESHOLD:,}."
    )
    print("SendEmail: simulated message\n" + message)
    return {
        "email_message": message,
        "path": [*state.get("path", []), "SendEmail"],
    }


graph_builder = StateGraph(CustomerState)
graph_builder.add_node("FindCustomer", find_customer)
graph_builder.add_node("GetBalance", get_balance)
graph_builder.add_node("SendEmail", send_email)
graph_builder.add_edge(START, "FindCustomer")
graph_builder.add_edge("FindCustomer", "GetBalance")
graph_builder.add_conditional_edges(
    "GetBalance",
    choose_next,
    {"send_email": "SendEmail", "finish": END},
)
graph_builder.add_edge("SendEmail", END)
customer_graph = graph_builder.compile()


def run_example(customer_id: str, expected_path: list[str]) -> None:
    print(f"\n=== Run for {customer_id} ===")
    result = customer_graph.invoke({"customer_id": customer_id, "path": []})
    print("Visited:", " -> ".join(result["path"]))
    assert result["path"] == expected_path, result["path"]
    print("Path check: passed")


if __name__ == "__main__":
    run_example("C001", ["FindCustomer", "GetBalance", "SendEmail"])
    run_example("C002", ["FindCustomer", "GetBalance"])
    print("\nGraph validation and both routing examples passed.")


