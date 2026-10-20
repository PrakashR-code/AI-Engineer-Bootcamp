from banking_tools import search_transaction

result = search_transaction.run(
    transaction_id="TXN102"
)

print("\n--- TRANSACTION TOOL RESULT ---")
print(result)
print("\nResult type:", type(result))