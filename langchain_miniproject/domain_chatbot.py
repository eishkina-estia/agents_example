# Mini-project: Domain Chatbot (LangChain)

# ---------------------------------------------------------------------
# 1) Configuration
# ---------------------------------------------------------------------

# TODO: Choose a model + provider
# llm = ...

# ---------------------------------------------------------------------
# 2) Memory: session_id -> InMemoryChatMessageHistory
# ---------------------------------------------------------------------

# TODO: define a dictionary session_id (str) -> chat history (InMemoryChatMessageHistory)

def get_history(session_id: str) -> InMemoryChatMessageHistory:
    # TODO: implement the function

# ---------------------------------------------------------------------
# 3) Prompt: system + history + user question
# ---------------------------------------------------------------------

SYSTEM_TEMPLATE = """\
You are a helpful assistant specialized in the domain: {domain}.

Rules:
- Stay within the domain. If the question is out-of-domain, say so briefly and suggest a domain-relevant alternative.
- Ask one clarifying question if the user request is ambiguous.
- Keep answers concise but concrete (bullets or steps when useful).
"""

# TODO: define prompt template
# prompt = ...

# ---------------------------------------------------------------------
# 4) Trimming: keep conversation within context budget
# ---------------------------------------------------------------------

trimmer = trim_messages(
    token_counter=llm,   # uses the model's token counting if supported
    # TODO: add other arguments
)


# ---------------------------------------------------------------------
# 5) Build chain: prompt -> trimmer -> LLM -> parser
# ---------------------------------------------------------------------

# TODO: combine prompt, trimmer, and llm
# TODO: add StrOutputParser at the end
# chain = ...


# ---------------------------------------------------------------------
# 6) Wrap the chain with message history
# ---------------------------------------------------------------------

# TODO:
# chatbot = RunnableWithMessageHistory(...)

# ---------------------------------------------------------------------
# 7) CLI loop
# ---------------------------------------------------------------------

def main():
    print("\n=== Domain Chatbot (LangChain) ===")
    print("Type '/exit' to quit. Type '/reset' to clear memory.\n")

    # TODO: choose a domain interactively or hardcode it
    domain = input("Choose your domain (e.g., 'SQL tutor', 'NLP revision', 'Cloud security'): ").strip()
    if not domain:
        domain = "General"

    # TODO
    # session_id = ...

    while True:
        user_q = input("\nYou: ").strip()
        if not user_q:
            continue
        if user_q.lower() == "/exit":
            print("Bye!")
            break
        if user_q.lower() == "/reset":
            # TODO: clear memory
            print("Memory cleared.")
            continue

        # TODO: Invoke the stateful chain
        # answer = chatbot.invoke(...)
        print(f"\nBot: {answer}")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------
# Additional tasks
# ---------------------------------------------------------------------

# 1. Out-of-domain detection with a second LLM call
# Goal: before answering, run a lightweight “router” step that decides whether the user question is in the chosen domain.
# Build a small classification prompt that returns IN_DOMAIN or OUT_OF_DOMAIN
# Logic:
# - If IN_DOMAIN: run the main chatbot chain
# - If OUT_OF_DOMAIN: refuse briefly + suggest a domain-relevant alternative

# 2. Structured output
# Goal: make the chatbot return a machine-readable JSON object instead of plain text.
# Output format: {"answer": "...", "clarifying_question": null}
# If the request is ambiguous: answer must be null and clarifying_question must contain the question.
# If the request is clear: clarifying_question must be null.
#
# Use a Pydantic model
# Add explicit prompt constraints for the output
