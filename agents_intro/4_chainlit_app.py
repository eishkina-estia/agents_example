"""
Demo 4 - Chainlit UI for a stateful chatbot

Run:
1) pip install -qU chainlit
2) chainlit run agents_intro/4_chainlit_app.py
"""

import chainlit as cl

from langchain_ollama import ChatOllama
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import trim_messages
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableConfig


DEFAULT_SESSION_ID = "default"
DOMAIN = "NLP and Transformers"

chat_history = InMemoryChatMessageHistory()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful tutor in {domain}. Answer concisely."),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])

trimmer = trim_messages(
    strategy="last",
    token_counter=len,
    max_tokens=12,
    start_on="human",
    include_system=True,
    allow_partial=False,
)

llm = ChatOllama(model="llama3.2:3b", temperature=0)

chain = prompt | trimmer | llm
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history=lambda session_id: chat_history,
    input_messages_key="question",
    history_messages_key="history",
)
final_chain = chain_with_history | StrOutputParser()


@cl.on_message
# triggered each time a user sends a message
async def handle_message(message: cl.Message):
    msg = cl.Message(content="")
    # the application updates the message incrementally
    async for chunk in final_chain.astream(
        {"domain": DOMAIN, "question": message.content},
        config=RunnableConfig(configurable={"session_id": DEFAULT_SESSION_ID}),
    ):
        await msg.stream_token(chunk)
    # ensures the message is properly committed in the conversation history
    await msg.send()
