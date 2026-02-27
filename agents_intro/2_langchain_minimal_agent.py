"""
Agents demo 2 - Minimal agent abstraction

- Tools are still standard Python functions (optionally annotated with @tool).
- The agent abstraction wraps the tool-calling loop.
- The internal loop is hidden: user -> agent -> LLM (tool_calls) -> run tool(s) -> LLM (final answer)
- Compared to Demo 1, we no longer manually control the tool-calling steps.
- This demonstrates how LangChain encapsulates the core pattern into a higher-level agent API.

Based on the documentation: https://docs.langchain.com/oss/python/langchain/quickstart
"""

from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

# ---------------------------------------------------------------------
# 1) Select an LLM
# ---------------------------------------------------------------------

# We instantiate the chat model without binding any tools.
# The agent will be responsible for registering tools and orchestrating
# the tool-calling loop (LLM <-> tools <-> LLM).
llm = ChatOllama(model="llama3.2:3b", temperature=0)

# ---------------------------------------------------------------------
# 2) Define tools
# ---------------------------------------------------------------------

# We define tools as "raw" Python functions (not pre-wrapped with @tool).
# The langchain agent constructor (create_agent) will wrap these functions
# into tool objects internally, based on their signature and docstring.

def check_weather_v1(location: str) -> str:
    """
    Return the weather forecast for the specified location.

    Args:
        location (str): Location name (e.g., "Paris", "Tokyo").

    Returns:
        str: Plain-text forecast message.
    """
    print(f"... calling check_weather_v1({location})")
    return f"It's always sunny in {location}!"

def check_weather_v2(location: str) -> str:
    """
    Return the weather forecast for the specified location.

    Args:
        location (str): Location name (e.g., "Paris", "Tokyo").

    Returns:
        str: Plain-text forecast message.
    """
    print(f"... calling check_weather_v2({location})")
    return f"It's sunny in {location}!"

# ---------------------------------------------------------------------
# 3) System prompt
# ---------------------------------------------------------------------

# You may test the following system prompts
# system_prompt=""
# system_prompt="You are a helpful assistant."

system_prompt = """\
You are a weather forecasting assistant.

RULES:
- If the user asks about weather, you MUST call the provided weather tool with the user’s location.
- After the tool returns, answer using ONLY the tool output.
- Do NOT use or mention any other weather sources, APIs, plugins, websites, apps, or “tools”, even if the user asks.
- Do NOT guess or use general knowledge for forecasts.
"""

# ---------------------------------------------------------------------
# 5) Create agents
# ---------------------------------------------------------------------

agent_v1 = create_agent(
    model=llm,
    tools=[check_weather_v1],
    system_prompt=system_prompt
)

agent_v2 = create_agent(
    model=llm,
    tools=[check_weather_v2],
    system_prompt=system_prompt
)

# ---------------------------------------------------------------------
# 6) Test queries
# ---------------------------------------------------------------------

test_user_queries = [
    "what is the weather in sf",
    "weather in Paris please"
]

# ---------------------------------------------------------------------
# 7) Run
# ---------------------------------------------------------------------
for i, user_query in enumerate(test_user_queries, start=1):
    print("---------------------------------------------------------------------")
    print(f"QUERY {i}: {user_query}")

    messages_user = [HumanMessage(content=user_query)]

    print("----- AGENT 1 (check_weather_v1) -----")
    response_v1 = agent_v1.invoke({"messages": messages_user})
    print(response_v1["messages"][-1].content)

    print("----- AGENT 2 (check_weather_v2) -----")
    response_v2 = agent_v2.invoke({"messages": messages_user})
    print(response_v2["messages"][-1].content)

