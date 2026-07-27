import os

from dotenv import load_dotenv
from openai import OpenAI


# Load variables from the .env file
load_dotenv()


API_KEY = os.getenv("GENERATIVE_ENGINE_API_KEY")
BASE_URL = os.getenv("GENERATIVE_ENGINE_BASE_URL")

MODEL_NAME = "openai.gpt-5-mini"


if not API_KEY:
    raise ValueError(
        "GENERATIVE_ENGINE_API_KEY is missing from the .env file."
    )

if not BASE_URL:
    raise ValueError(
        "GENERATIVE_ENGINE_BASE_URL is missing from the .env file."
    )


client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    timeout=60.0,
)


def call_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Sends a system prompt and user prompt to the model
    and returns only the generated text.
    """

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )

    generated_text = response.choices[0].message.content

    if generated_text is None:
        raise ValueError("The model returned an empty response.")

    return generated_text