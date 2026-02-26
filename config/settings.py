import os
from dotenv import load_dotenv

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if MISTRAL_API_KEY is None:
    raise RuntimeError(
        "MISTRAL_API_KEY is not set.\n"
        "Create a .env file in the project root containing:\n"
        "MISTRAL_API_KEY=your_api_key"
    )