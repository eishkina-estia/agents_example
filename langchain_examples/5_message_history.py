# Show how LangChain "memory" works with RunnableWithMessageHistory
# and how trimming keeps prompts within context limits.
# Pipeline (high level):
# User input -> ChatPromptTemplate (+history) -> trim_messages -> ChatModel -> output
# RunnableWithMessageHistory injects that history into the prompt on each call.

from langchain_ollama import ChatOllama

from langchain_core.messages import SystemMessage, HumanMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

llm = ChatOllama(model="llama3.2:3b", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{query}"),
])

# here: counts “messages”, not real tokens (token_counter=len)
trimmer = trim_messages(
    strategy="last", token_counter=len, max_tokens=6, start_on="human",
    end_on="human", include_system=True, allow_partial=False)

chain = prompt | trimmer | llm
chat_history = InMemoryChatMessageHistory()

chain_with_history = RunnableWithMessageHistory(
    chain, get_session_history = lambda session_id: chat_history,
    input_messages_key = "query", history_messages_key = "history")

DEFAULT_SESSION_ID = "default"
DEFAULT_CONFIG = config={"configurable": {"session_id": DEFAULT_SESSION_ID}}

ai_message = chain_with_history.invoke({"query": "Hi, my name is Bob!"}, config=DEFAULT_CONFIG)
print("Assistant (with memory):", ai_message.content)

ai_message = chain_with_history.invoke({"query": "What is my name?"}, config=DEFAULT_CONFIG)
print("Assistant (with memory):", ai_message.content)

ai_message = llm.invoke("What is my name?")
print("Assistant (without memory):", ai_message.content)
