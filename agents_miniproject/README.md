# Miniproject: Agents

---

## Overview

In this project, you will build an interactive **NLP Research Assistant**.

The assistant must answer NLP-related questions using tools: wikipedia search, arxiv search and custom knowledge base.

The system must combine:

- Tool calling (LangChain)
- Explicit control flow (LangGraph)
- A user interface (Chainlit)

---

## Requirements

### 1) Add new tool: arXiv lookup

Implement the [arxiv tool](https://docs.langchain.com/oss/python/integrations/retrievers/arxiv) that must:

- Accept a query (title or keywords)
- Query arXiv
- Return structured JSON containing `title`, `authors`, `year`, `arxiv_id`, `url`
- Return an explicit error JSON if no result is found

The tool must follow the same conventions as in previous demos:

- Naming
- Clear docstring with args / returns
- Structured JSON output
- Explicit error handling

---

### 2) LangGraph Agent

You must build an agent using **LangGraph**, following the same architecture as demo 3:
`LLM node ↔ Tools node (loop)`.

Requirements:

- Tools are bound only inside the **LLM node**
- Tool execution is performed explicitly inside the **Tools node**
- Debug prints must show which tool is called and with what arguments
- The graph must be exported as an image
- Logs for test queries must be saved in a text file

The log for test queries must capture:
- Router decisions
- Tool calls
- Tool outputs
- Final answers
- Debug prints

You must include the following test queries (1-2 for each category):
- KB-style questions
- Wikipedia-style questions
- arXiv-style questions
- Out-of-domain questions

The agent must include:

- `wikipedia_search`
- `nlp_concept_kb_lookup`
- `arxiv_lookup`


### 3) Chainlit UI

You must build a Chainlit application that wraps the LangGraph agent, provides a chat interface, and streams the output.

The application must run with:

```bash
chainlit run app.py
```

## Recommended Repository Structure

```
tools.py
    - KB tool
    - Wikipedia tool
    - arXiv tool

agent_graph.py
    - AgentState
    - Nodes
    - Routing logic
    - Graph export
    - Agent tests with logs saving

app.py
    - Chainlit application
    - Streaming

output/
    - graph PNG
    - test logs
```

## (optional) Extension - Explicit Tool Routing

Add a routing mechanism that decides which tool should be preferred before entering the main loop.

Examples:
- If the question contains "paper", "arXiv", or "authors", then prefer arXiv.
- If the question asks for "definition", "keywords", or "year", then prefer knowledge base.
- Otherwise allow normal LLM-based selection.

The routing can be:
- Rule-based (deterministic)
- LLM-based (must return strict JSON)

The routing must be implemented as explicit conditional edges in the LangGraph graph.

## (optional) Extension - Adding Sources to the Output

When Wikipedia or arXiv is used, the final answer must include a short “Sources” section.

Example:
```
<final answer>
Sources:
- arXiv: 1706.03762
- Wikipedia: Transformer (summary)
```

Do not expose raw JSON. Format it cleanly for the user.