from langchain_core.messages import HumanMessage
from langgraph.types import Command

from graph import graph


config = {
    "configurable": {
        "thread_id": "transfer-001"
    }
}


if __name__ == "__main__":
    user_request = (
        "I want to transfer INR 10,000 "
        "from John's account to Ravi's account. "
        "Please proceed."
    )

    print("\n========================================")
    print("           BANKING AGENT")
    print("========================================")

    print(f"\nUser: {user_request}")

    # First invocation pauses at the human approval interrupt.
    result = graph.invoke(
        {
            "messages": [
                HumanMessage(content=user_request)
            ]
        },
        config=config
    )

    print("\n========== GRAPH PAUSED ==========")
    print(result)

    # Resume the graph with the human decision.
    print("\nHuman decision: YES")

    result = graph.invoke(
        Command(resume="yes"),
        config=config
    )

    print("\n========== FINAL RESULT ==========")
    print(result)