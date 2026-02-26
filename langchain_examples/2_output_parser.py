from langchain_mistralai import ChatMistralAI
from config.settings import MISTRAL_API_KEY

llm = ChatMistralAI(
    model="mistral-large-latest",
    temperature=0
)

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

class Person(BaseModel):
    firstname: str = Field(description="first name of hero")
    lastname: str = Field(description="last name of hero")
    age: int = Field(description="age of hero in ages")

parser = PydanticOutputParser(pydantic_object=Person)

messages = [
    ("system", "Handle the user query.\n{format_instructions}"),
    ("human", "{user_query}")
]
prompt_template = ChatPromptTemplate(messages)

query = "Dans le premier livre, Harry Potter à l'école des sorciers,\
Harry Potter a 11 ans."

prompt_value = prompt_template.invoke(
    {
        "format_instructions": parser.get_format_instructions(),
        "user_query": query
    }
)

answer = llm.invoke(prompt_value.to_messages())
parsed = parser.invoke(answer)
print(parsed.model_dump_json())
# { "firstname": "Harry", lastname": "Potter", "age": 11 }

# Irrelevant user input
prompt_value = prompt_template.invoke(
    {
        "format_instructions": parser.get_format_instructions(),
        "user_query": "I love Deep Learning!"
    }
)

answer = llm.invoke(prompt_value.to_messages())
try:
    parsed = parser.invoke(answer)
    print(parsed.model_dump_json())
except Exception as e:
    print("PARSING FAILED:", e)
    print("\nRAW MODEL OUTPUT:\n", answer.content)

# Missing user input
prompt_value = prompt_template.invoke(
    {
        "format_instructions": parser.get_format_instructions(),
        "user_query": ""
    }
)

answer = llm.invoke(prompt_value.to_messages())
try:
    parsed = parser.invoke(answer)
    print(parsed.model_dump_json())
except Exception as e:
    print("PARSING FAILED:", e)
    print("\nRAW MODEL OUTPUT:\n", answer.content)
# {"firstname":"","lastname":"","age":0}

