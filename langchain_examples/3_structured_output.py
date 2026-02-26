from langchain_mistralai import ChatMistralAI
from config.settings import MISTRAL_API_KEY

llm = ChatMistralAI(
    model="mistral-large-latest",
    temperature=0
)

from pydantic import BaseModel, Field, field_validator, ValidationError
from langchain_core.prompts import ChatPromptTemplate

class Person(BaseModel):
    firstname: str = Field(description="firstname of hero")
    lastname: str = Field(description="lastname of hero")
    age: int = Field(description="age of hero")

# forces the model output into the schema and returns a Pydantic object.
prepared_llm = llm.with_structured_output(Person)

prompt_template = ChatPromptTemplate.from_messages([
    # You can try a specific prompt, e.g.:
    # Extract the person's information from the text.
    # Return structured output that matches the Person schema.
    ("system", "Handle the user query"),
    ("human", "{user_query}")
])

def print_invoke_result(llm, prompt_template, query):
    prompt_value = prompt_template.invoke({"user_query": query})
    answer = llm.invoke(prompt_value.to_messages())
    print("User input:", query)
    print("Assistant:", answer.model_dump_json())
    print()

print_invoke_result(prepared_llm, prompt_template,
    query="Dans le premier livre, Harry Potter à l'école des sorciers,Harry Potter a 10 ans.")

print_invoke_result(prepared_llm, prompt_template,
    query="Harry Potter was very young.")

print_invoke_result(prepared_llm, prompt_template,
    query="Jean Dupont was very young.")

# Change Person schema
class Person2(BaseModel):
    firstname: str = Field(description="firstname of hero")
    lastname: str = Field(description="lastname of hero")
    age: int | None = Field(description="Age in years if explicitly stated, otherwise null")

    @field_validator("age")
    def validate_age(cls, v):
        if v is not None and (v < 0 or v > 120):
            raise ValueError("Invalid age")
        return v

prepared_llm = llm.with_structured_output(Person2)

print("AFTER SCHEMA REPLACEMENT")

try:
    print_invoke_result(prepared_llm, prompt_template,
        query="Harry Potter was very young.")

    print_invoke_result(prepared_llm, prompt_template,
        query="Jean Dupont was very young.")

    print_invoke_result(prepared_llm, prompt_template,
        query="Jean Dupont was 300 years old.")

except ValidationError as e:
    print("Validation error:", e)
    print("Model returned invalid structured output.")


prompt_template = ChatPromptTemplate.from_messages([
    ("system",
     "Extract the person's information from the text.\n"
     "Rules:\n"
     "- Only extract information explicitly present in the text.\n"
     "- DO NOT infer or guess missing values.\n"
     "- If age is not explicitly mentioned, return null."),
    ("human", "{user_query}"),
])

print("AFTER SCHEMA & SYSTEM PROMPT REPLACEMENT")

print_invoke_result(prepared_llm, prompt_template,
    query="Harry Potter was very young.")