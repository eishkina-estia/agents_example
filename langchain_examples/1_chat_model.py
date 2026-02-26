from langchain_ollama import ChatOllama
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import SystemMessage, HumanMessage

from config.settings import MISTRAL_API_KEY

# see available models: ollama list (in a terminal)
llm_llama = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)

llm_mistral = ChatMistralAI(
    model="mistral-large-latest",
    api_key=MISTRAL_API_KEY,
    temperature=0
)

message_system = SystemMessage(content="""
You are a translation assistant.
Your task is ONLY to translate text from English to French.
Rules:
Always translate the user's input literally.
If the input is a question, translate the question.
NEVER answer the question.
Output ONLY the French translation.
User's input:
""")

messages_1_raw_v1 = [
    {"role": "system", "content": message_system.content},
    {"role": "user", "content": "I love Deep Learning!"}
]

messages_1_raw_v2 = [
    ("system", message_system.content),
    ("human", "I love Deep Learning!")
]

messages_1 = [
    message_system,
    HumanMessage(content="I love Deep Learning!")
]
messages_2 = [
    message_system,
    HumanMessage(content="Do you love Deep Learning?")
]

print("Model: ", llm_llama.model)

# Standard
print("Messages:", messages_1_raw_v1)
ai_message = llm_llama.invoke(messages_1_raw_v1)
print("Assistant:", ai_message.content)

print("Messages:", messages_1_raw_v2)
ai_message = llm_llama.invoke(messages_1_raw_v2)
print("Assistant:", ai_message.content)

print("Messages:", messages_1)
ai_message = llm_llama.invoke(messages_1)
print("Assistant:", ai_message.content)

input("\nPress Enter to continue to the next step...")

# Stream
print("Messages:", messages_1)
print("Assistant: ", end="")
for message_chunk in llm_llama.stream(messages_1):
    print(message_chunk.content, end="")
input("\nPress Enter to continue to the next step...")

# Standard
print("Messages:", messages_2)
ai_message = llm_llama.invoke(messages_2)
print("Assistant:", ai_message.content)
input("\nPress Enter to continue to the next step...")

# Batch
print("Messages:", "(1)", messages_1, "(2)", messages_2)
print("Assistant:")
ai_messages = llm_llama.batch([messages_1, messages_2])
for i, msg in enumerate(ai_messages, start=1):
    print(f"({i}): {msg.content}")
input("\nPress Enter to continue to the next step...")

# Try another model provider -> MistralAI
print("Model: ", llm_mistral.model)

# Standard
print("Messages:", messages_2)
ai_message = llm_mistral.invoke(messages_2)
print("Assistant:", ai_message.content)
