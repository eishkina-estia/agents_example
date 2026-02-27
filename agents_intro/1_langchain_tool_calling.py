"""
Agents demo 1 - Tool calling (the core pattern)

- A tool is a Python function with a schema (name + args + docstring).
- An LLM can decide to call tools.
- Tool calling is a loop:
    user -> LLM (tool_calls) -> run tool(s) -> LLM (final answer)
"""

import os
import json
from dataclasses import field

import pandas as pd

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

# ---------------------------------------------------------------------
# 1) Dummy data (instead of a real database)
# ---------------------------------------------------------------------

data_df = pd.DataFrame({
    'transaction_id': ['T1001', 'T1002', 'T1003', 'T1004', 'T1005'],
    'customer_id': ['C001', 'C002', 'C003', 'C002', 'C001'],
    'payment_amount': [125.50, 89.99, 120.00, 54.30, 210.20],
    'payment_date': ['2021-10-05', '2021-10-06', '2021-10-07', '2021-10-05', '2021-10-08'],
    'payment_status': ['Paid', 'Unpaid', 'Paid', 'Paid', 'Pending']
})

# ---------------------------------------------------------------------
# 2) Define tools
# ---------------------------------------------------------------------

@tool
def retrieve_payment_status(transaction_id: str) -> str:
    """
    Return the payment status for a given transaction ID.

    Args:
        transaction_id (str): Unique transaction identifier.

    Returns:
        str: JSON-encoded string:
            - {"status": "<payment_status>"} if the transaction exists.
            - {"error": "transaction id not found."} otherwise.
    """
    if transaction_id in data_df.transaction_id.values:
        return json.dumps({'status': data_df[data_df.transaction_id == transaction_id].payment_status.item()})
    return json.dumps({'error': 'transaction id not found.'})

@tool
def retrieve_amount(transaction_id: str) -> str:
    """
    Return the payment amount for a given transaction ID.

    Args:
        transaction_id (str): Unique transaction identifier.

    Returns:
        str: JSON-encoded string:
            - {"amount": "<amount>"} if the transaction exists.
            - {"error": "transaction id not found."} otherwise.
    """
    if transaction_id in data_df.transaction_id.values:
        return json.dumps({'amount': data_df[data_df.transaction_id == transaction_id].payment_amount.item()})
    return json.dumps({'error': 'transaction id not found.'})

tool_names_to_functions = {
    retrieve_payment_status.name: retrieve_payment_status,
    retrieve_amount.name: retrieve_amount,
}

# ---------------------------------------------------------------------
# 3) Select an LLM
# ---------------------------------------------------------------------

llm = ChatOllama(model="llama3.2:3b", temperature=0)

# Bind tools to the model: the model can now return tool_calls.
llm_with_tools = llm.bind_tools([retrieve_payment_status, retrieve_amount])

# ---------------------------------------------------------------------
# 4) The (manual) tool-calling loop
# ---------------------------------------------------------------------
system_message = SystemMessage(content=("""\
You are a customer support assistant.
When a user provides a transaction ID:
- Always call the appropriate tool(s) to retrieve the requested information.
- If a tool returns an error, inform the user clearly and politely about this error.
- When a tool returns valid data, present the result clearly in natural language without exposing raw JSON.
- Do not fabricate data.
"""))

test_user_queries = [
    "What is the payment status and amount for transaction T1001?",
    "Is order T1002 paid, and how much was the payment?",
    "Check transaction T1005: what's its status and total amount?",
    "Can you check transaction T1004 and tell me whether it went through and what I was charged?",
    "Give me the status and payment amount for transaction T9999."
]

for query in test_user_queries:
    print("User:", query)

    messages = [
        system_message,
        HumanMessage(content=query),
    ]

    # LLM decides whether to call a tool
    ai_message = llm_with_tools.invoke(messages)
    messages.append(ai_message)

    # If tool_calls exist, we execute them and append ToolMessages
    if ai_message.tool_calls:
        for tool_call in ai_message.tool_calls:
            print(f"... calling {tool_call['name']}")

            tool_fn = tool_names_to_functions.get(tool_call['name'])

            if tool_fn is None:
                continue  # or handle the error

            tool_message = tool_fn.invoke(tool_call)
            messages.append(tool_message)

        # Call the LLM again with tool results to produce the final answer
        final_message = llm_with_tools.invoke(messages)
    else:
        # Model answered directly (no tools needed)
        final_message = ai_message

    print("Assistant:", final_message.content, "\n")
