
from crewai.tools import tool

# Sample data for learning purposes only.
# Dictionary key = transaction ID.
TRANSACTIONS = {
    "TXN102": {
        "transaction_id": "TXN102",
        "type": "International transfer",
        "amount": 200000,
        "country": "Singapore",
        "status": "Completed",
    },
    "TXN101": {
        "transaction_id": "TXN101",
        "type": "UPI payment",
        "amount": 1500,
        "country": "India",
        "status": "Completed",
    },
}


@tool("Transaction Search")
def search_transaction(transaction_id: str) -> str:
    """Retrieve transaction details using a transaction ID."""

    transaction = TRANSACTIONS.get(transaction_id)

    if transaction is None:
        return f"Transaction {transaction_id} not found."

    # Return the actual fields as JSON-formatted text.
    import json
    return json.dumps(transaction, indent=2)
