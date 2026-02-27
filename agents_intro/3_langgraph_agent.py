"""
Agents demo 3 - LangGraph agents with routing

LangGraph makes the control flow explicit:
START -> LLM -> (if tool_calls) Tools -> LLM -> ... -> END

We explicitly define:
- The agent state
- The LLM node
- The tools node
- The routing logic

Tools in this demo:
- wikipedia (built-in community tool): broad, encyclopedic knowledge
- nlp_concept_kb_lookup (custom knowledge base): structured, deterministic NLP reference

Routing logic:
- Domain router: NLP-related questions -> agent loop | otherwise -> END (short refusal)

Requirements:
pip install langgraph langchain-community wikipedia
"""

import json
from typing import Annotated, Sequence, TypedDict

from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# ---------------------------------------------------------------------
# region 1) LLM configuration
# ---------------------------------------------------------------------

llm = ChatOllama(model="llama3.2:3b", temperature=0)

SYSTEM_MESSAGE = SystemMessage(content=("""\
You are a helpful assistant.
- Use wikipedia_search for broad factual/contextual information.
- Use nlp_concept_kb_lookup for structured NLP concept metadata.
- If a tool returns an error JSON, handle it and ask a short clarification.
- Be concise."""))

# endregion
# ---------------------------------------------------------------------
# region 2) Tools definition
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# 2.1) Tool: wikipedia (built-in)
# ---------------------------------------------------------------------
wikipedia_runner = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(lang="en")
)

@tool
def wikipedia_search(query: str) -> str:
    """
    Search Wikipedia and return a short summary.

    Args:
        query (str): Search query.

    Returns:
        str: Short Wikipedia summary (plain text).
    """
    return wikipedia_runner.run(query)

# ---------------------------------------------------------------------
# 2.2) Tool: NLP concept reference knowledge-based (custom)
# ---------------------------------------------------------------------
# Custom knowledge base
NLP_REFERENCE_KB = {
    "transformer": {
        "definition": "A neural architecture built around self-attention for sequence modeling.",
        "key_paper": "Attention Is All You Need",
        "year": 2017,
        "keywords": ["self-attention", "encoder-decoder", "parallelization"],
        "category": "sequence modeling architecture",
    },
    "self-attention": {
        "definition": "An attention mechanism where each token attends to all tokens in the same sequence.",
        "key_paper": "Attention Is All You Need",
        "year": 2017,
        "keywords": ["query", "key", "value", "attention weights"],
        "category": "attention mechanism",
    },
    "bert": {
        "definition": "A bidirectional Transformer encoder model pretrained with masked language modeling.",
        "key_paper": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "year": 2018,
        "keywords": ["masked language modeling", "fine-tuning", "pretraining"],
        "category": "encoder-only transformer",
    },
    "tokenization": {
        "definition": "The process of splitting text into units (tokens) used by NLP models.",
        "key_paper": "N/A",
        "year": None,
        "keywords": ["subword", "BPE", "WordPiece"],
        "category": "preprocessing",
    },
}

@tool
def nlp_concept_kb_lookup(term: str) -> str:
    """
    Retrieve structured information about an NLP concept from a reference knowledge base.

    Args:
        term (str): NLP concept name (e.g., "transformer", "bert", "self-attention").

    Returns:
        str: JSON-encoded string:
            - {"definition": ..., "key_paper": ..., "year": ..., "keywords": [...], "category": ...}
            - {"error": "concept not found."} if unknown.
    """
    key = term.lower().strip()

    if key in NLP_REFERENCE_KB:
        return json.dumps(NLP_REFERENCE_KB[key])

    return json.dumps({"error": "concept not found."})

# endregion
# ---------------------------------------------------------------------
# region 3) LangGraph state definition
# ---------------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    steps: int

# endregion
# ---------------------------------------------------------------------
# region 4) Node: LLM
# ---------------------------------------------------------------------

TOOLS = [wikipedia_search, nlp_concept_kb_lookup]

def node_llm(state: AgentState, config: RunnableConfig):
    """
    Call the LLM with tools bound.

    Args:
        state (AgentState): Current state.
        config (RunnableConfig): Runtime config.

    Returns:
        dict: Partial state update.
    """
    llm_with_tools = llm.bind_tools(TOOLS)

    messages = [SYSTEM_MESSAGE] + list(state["messages"])
    response = llm_with_tools.invoke(messages, config=config)

    return {
        "messages": [response],
        "steps": state["steps"] + 1,
    }
# endregion
# ---------------------------------------------------------------------
# region 5) Node: Tools
# ---------------------------------------------------------------------

TOOLS_BY_NAME = {t.name: t for t in TOOLS}

def node_tools(state: AgentState):
    """
    Execute tool calls emitted by the last LLM message.

    Args:
        state (AgentState): Current state.

    Returns:
        dict: Partial state update with ToolMessages.
    """
    last_message = state["messages"][-1]
    tool_messages: list[ToolMessage] = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]

        print("----- TOOL CALL -----")
        print(f"{tool_name}({tool_args})")

        tool_fn = TOOLS_BY_NAME.get(tool_name)

        if tool_fn is None:
            tool_messages.append(
                ToolMessage(
                    content=f"Unknown tool: {tool_name}",
                    name=tool_name,
                    tool_call_id=tool_id
                )
            )
            continue

        result = tool_fn.invoke(tool_args)
        print("----- TOOL RESULT BEGIN -----")
        print(str(result))
        print("----- TOOL RESULT END -----")

        tool_messages.append(
            ToolMessage(
                content=str(result),
                name=tool_name,
                tool_call_id=tool_id
            )
        )

    return {
        "messages": tool_messages,
        "steps": state["steps"] + 1,
    }

# endregion
# ---------------------------------------------------------------------
# region 6) LLM/Tools routing logic
# ---------------------------------------------------------------------
MAX_STEPS = 6

def route_after_llm(state: AgentState) -> str:
    """
    Decide whether to continue the loop or end.

    Returns:
        str: "continue" or "end"
    """
    if state["steps"] >= MAX_STEPS:
        return "end"

    last_message = state["messages"][-1]

    if not getattr(last_message, "tool_calls", None):
        return "end"

    return "continue"

# endregion
# ---------------------------------------------------------------------
# region 7) Router (LLM-based domain validation)
# ---------------------------------------------------------------------
ROUTER_SYSTEM_MESSAGE = SystemMessage(content="""\
You are a router for an NLP-only assistant.

Return STRICT JSON only:
- {"route":"in_domain"} if the question is about NLP (tokenization, embeddings, attention, Transformers, LLMs, etc.)
- {"route":"out_of_domain"} otherwise
""")

def route_domain_llm(state: AgentState) -> str:
    """
    Decide if the last user question is in-domain using an LLM call.

    Returns:
        str: "in_domain" or "out_of_domain"
    """
    user_text = ""
    # Look up for the last HumanMessage
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_text = msg.content or ""
            break

    router_messages = [
        ROUTER_SYSTEM_MESSAGE,
        HumanMessage(content=user_text),
    ]

    response = llm.invoke(router_messages)

    try:
        payload = json.loads((response.content or "").strip())
        if payload.get("route") == "out_of_domain":
            return "out_of_domain"
    except Exception:
        # Demo-friendly default: avoid refusing due to router formatting.
        return "in_domain"

    return "in_domain"


def node_out_of_domain(state: AgentState):
    """
    Return a short response for out-of-domain questions.

    Returns:
        dict: Partial state update (final assistant message).
    """
    msg = SystemMessage(content=(
        "The question is out of scope. I can help with NLP concepts (Transformer, attention, tokenization, BERT, etc.). "
        "Please rephrase your question within that scope."
    ))
    return {"messages": [msg], "steps": state["steps"] + 1}

# endregion
# ---------------------------------------------------------------------
# region 8) Build LangGraph agent
# ---------------------------------------------------------------------
def build_langgraph_agent():
    """
    Build and compile the LangGraph agent.

    Returns:
        Any: Compiled LangGraph runnable.
    """
    graph = StateGraph(AgentState)

    graph.add_node("out_of_domain", node_out_of_domain)
    graph.add_node("llm", node_llm)
    graph.add_node("tools", node_tools)

    # Branch first: domain validation.
    graph.add_conditional_edges(
        START,
        route_domain_llm,
        {"in_domain": "llm", "out_of_domain": "out_of_domain"},
    )
    graph.add_edge("out_of_domain", END)

    # Main tool-calling loop.
    graph.add_conditional_edges(
        "llm",
        route_after_llm,
        {"continue": "tools", "end": END},
    )

    graph.add_edge("tools", "llm")

    return graph.compile()

def export_graph_png(agent, output_path: str) -> None:
    """
    Export the compiled LangGraph to a PNG image.

    Args:
        agent (Any): Compiled LangGraph runnable.
        output_path (str): Output file path.

    Returns:
        None
    """
    graph = agent.get_graph()
    png_bytes = graph.draw_mermaid_png()

    with open(output_path, "wb") as f:
        f.write(png_bytes)

# endregion
# ---------------------------------------------------------------------
# region 9) Test queries
# ---------------------------------------------------------------------
test_user_queries = [
    "Define self-attention and give the key paper + year.",
    "What is a Transformer? Also, summarize it in one sentence for Master's students.",
    "Who introduced the Transformer architecture? Use Wikipedia for context, and the NLP knowledge base for metadata.",
    "Explain BPE briefly and give its key paper and year.",
    "What's the best pizza place in Paris?"
]

# endregion
# ---------------------------------------------------------------------
# region 10) Run
# ---------------------------------------------------------------------

import sys
from contextlib import redirect_stdout

LOG_PATH = "./agents_intro/output/demo3_langgraph_full_log.txt"

if __name__ == "__main__":

    agent = build_langgraph_agent()
    export_graph_png(agent, "./agents_intro/output/demo3_langgraph.png")

    print(f"[LOG] Start execution. The results will be saved to: {LOG_PATH}")

    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        with redirect_stdout(log_file):
            for i, user_query in enumerate(test_user_queries, start=1):
                print("---------------------------------------------------------------------")
                print(f"QUERY {i}: {user_query}")

                final_state = agent.invoke(
                    {"messages": [HumanMessage(content=user_query)], "steps": 0}
                )

                print("----- LANGGRAPH AGENT -----")
                print(final_state["messages"][-1].content)

    print(f"[LOG] Full execution log saved to: {LOG_PATH}")

# endregion