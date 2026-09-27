from langchain_core.tools import tool


CUSTOMERS = {
    "John": {
        "account_number": "A123",
        "balance": 500000,
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


# ============================================================
# GET BALANCE
# ============================================================

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


# ============================================================
# GET TRANSACTIONS
# ============================================================

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


# ============================================================
# PREPARE TRANSFER
# ============================================================

@tool
def prepare_transfer(
    from_customer: str,
    to_customer: str,
    amount: int
) -> str:
    """Prepare a bank transfer for human approval."""

    sender = CUSTOMERS.get(from_customer)
    receiver = CUSTOMERS.get(to_customer)

    if not sender:
        return f"Sender {from_customer} was not found."

    if not receiver:
        return f"Receiver {to_customer} was not found."

    if amount <= 0:
        return "Transfer amount must be greater than zero."

    if amount > sender["balance"]:
        return (
            f"Insufficient balance. "
            f"Available balance is INR {sender['balance']:,}."
        )

    return (
        f"Transfer prepared: INR {amount:,} "
        f"from {from_customer} "
        f"to {to_customer}."
    )


# ============================================================
# EXECUTE TRANSFER
# ============================================================

@tool
def execute_transfer(
    from_customer: str,
    to_customer: str,
    amount: int
) -> str:
    """Execute an approved transfer between two customers."""

    sender = CUSTOMERS.get(from_customer)
    receiver = CUSTOMERS.get(to_customer)

    if not sender:
        return f"Sender {from_customer} was not found."

    if not receiver:
        return f"Receiver {to_customer} was not found."

    if amount <= 0:
        return "Transfer amount must be greater than zero."

    if amount > sender["balance"]:
        return (
            f"Transfer failed. "
            f"Available balance is INR {sender['balance']:,}."
        )

    # Debit sender
    sender["balance"] -= amount

    # Credit receiver
    receiver["balance"] += amount

    return (
        f"Transfer successful. "
        f"INR {amount:,} transferred from {from_customer} "
        f"to {to_customer}. "
        f"Sender balance: INR {sender['balance']:,}. "
        f"Receiver balance: INR {receiver['balance']:,}."
    )


# ============================================================
# TOOLS EXPOSED TO THE LLM
# ============================================================

TOOLS = [
    get_balance,
    get_transactions,
    prepare_transfer,
]