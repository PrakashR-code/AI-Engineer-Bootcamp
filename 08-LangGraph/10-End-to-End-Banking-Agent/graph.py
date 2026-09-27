from langchain_ollama import ChatOllama

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt
from langgraph.checkpoint.memory import InMemorySaver

from state import AgentState
from tools import (
    TOOLS,
    prepare_transfer,
    execute_transfer,
)


# ============================================================
# LLM
# ============================================================

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

llm_with_tools = llm.bind_tools(TOOLS)


# ============================================================
# LLM NODE
# ============================================================

def llm_node(state: AgentState):

    print("\n--- LLM NODE ---")

    response = llm_with_tools.invoke(
        state["messages"]
    )

    if response.tool_calls:

        print("LLM selected tool(s):")

        for tool_call in response.tool_calls:
            print(
                f"Tool: {tool_call['name']}, "
                f"Args: {tool_call['args']}"
            )

    else:

        print("LLM generated final response.")

    return {
        "messages": [response]
    }


# ============================================================
# ROUTE AFTER LLM
# ============================================================

def route_after_llm(state: AgentState):

    last_message = state["messages"][-1]

    # No tool call → final response
    if not last_message.tool_calls:
        return "end"

    # Transfer request → controlled transfer workflow
    for tool_call in last_message.tool_calls:

        tool_name = tool_call["name"]

        if tool_name == "prepare_transfer":
            return "prepare_transfer"

    # Normal tools
    return "tools"


# ============================================================
# PREPARE TRANSFER NODE
# ============================================================

def prepare_transfer_node(state: AgentState):

    print("\n--- PREPARE TRANSFER ---")

    last_message = state["messages"][-1]

    # Get the prepare_transfer tool call
    tool_call = next(
        call
        for call in last_message.tool_calls
        if call["name"] == "prepare_transfer"
    )

    from_customer = tool_call["args"]["from_customer"]
    to_customer = tool_call["args"]["to_customer"]
    amount = int(tool_call["args"]["amount"])

    print(f"From: {from_customer}")
    print(f"To: {to_customer}")
    print(f"Amount: INR {amount:,}")

    # Validate / prepare the transfer
    result = prepare_transfer.invoke({
        "from_customer": from_customer,
        "to_customer": to_customer,
        "amount": amount
    })

    print(f"Prepare result: {result}")

    # Preparation failed
    if (
        "Insufficient balance" in result
        or "not found" in result
        or "must be greater" in result
    ):
        return {
            "customer_name": from_customer,
            "to_customer": to_customer,
            "transfer_amount": amount,
            "transfer_status": "failed"
        }

    # Preparation successful
    return {
        "customer_name": from_customer,
        "to_customer": to_customer,
        "transfer_amount": amount,
        "transfer_status": "prepared"
    }


# ============================================================
# HUMAN APPROVAL
# ============================================================

def human_approval(state: AgentState):

    print("\n--- HUMAN APPROVAL REQUIRED ---")

    approval = interrupt({
        "message": "Approve this transfer?",
        "from_customer": state["customer_name"],
        "to_customer": state["to_customer"],
        "amount": state["transfer_amount"]
    })

    return {
        "approval": approval
    }


# ============================================================
# ROUTE AFTER APPROVAL
# ============================================================

def route_after_approval(state: AgentState):

    if state.get("approval") == "yes":
        return "execute"

    return "end"


# ============================================================
# EXECUTE TRANSFER
# ============================================================

def execute_transfer_node(state: AgentState):

    print("\n--- EXECUTE TRANSFER ---")

    # Safety check
    if state.get("approval") != "yes":

        print("Transfer rejected.")

        return {
            "transfer_status": "rejected"
        }

    # Execute only after human approval
    result = execute_transfer.invoke({
        "from_customer": state["customer_name"],
        "to_customer": state["to_customer"],
        "amount": state["transfer_amount"]
    })

    print(result)

    return {
        "transfer_status": "successful"
    }


# ============================================================
# BUILD GRAPH
# ============================================================

builder = StateGraph(AgentState)


# ============================================================
# NODES
# ============================================================

builder.add_node(
    "LLM",
    llm_node
)

builder.add_node(
    "Tools",
    ToolNode(TOOLS)
)

builder.add_node(
    "PrepareTransfer",
    prepare_transfer_node
)

builder.add_node(
    "HumanApproval",
    human_approval
)

builder.add_node(
    "ExecuteTransfer",
    execute_transfer_node
)


# ============================================================
# START
# ============================================================

builder.add_edge(
    START,
    "LLM"
)


# ============================================================
# LLM ROUTING
# ============================================================

builder.add_conditional_edges(
    "LLM",
    route_after_llm,
    {
        "tools": "Tools",
        "prepare_transfer": "PrepareTransfer",
        "end": END
    }
)


# ============================================================
# NORMAL TOOLS → LLM
# ============================================================

builder.add_edge(
    "Tools",
    "LLM"
)


# ============================================================
# TRANSFER FLOW
# ============================================================

builder.add_edge(
    "PrepareTransfer",
    "HumanApproval"
)


# ============================================================
# APPROVAL ROUTING
# ============================================================

builder.add_conditional_edges(
    "HumanApproval",
    route_after_approval,
    {
        "execute": "ExecuteTransfer",
        "end": END
    }
)


# ============================================================
# EXECUTE → END
# ============================================================

builder.add_edge(
    "ExecuteTransfer",
    END
)


# ============================================================
# CHECKPOINTING
# ============================================================

checkpointer = InMemorySaver()

graph = builder.compile(
    checkpointer=checkpointer
)