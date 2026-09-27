from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    messages: Annotated[list, add_messages]

    customer_name: str
    to_customer: str
    account_number: str

    balance: int

    transfer_amount: int
    transfer_status: str

    approval: str

    retry_count: int
    iteration_count: int