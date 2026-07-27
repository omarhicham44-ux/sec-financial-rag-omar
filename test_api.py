import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

api_key = os.getenv("GENERATIVE_ENGINE_API_KEY")
base_url = os.getenv("GENERATIVE_ENGINE_BASE_URL")

if not api_key:
    raise ValueError("GENERATIVE_ENGINE_API_KEY is missing from .env")

if not base_url:
    raise ValueError("GENERATIVE_ENGINE_BASE_URL is missing from .env")


client = OpenAI(
    api_key=api_key,
    base_url=base_url,
)


try:
    models = client.models.list()

    print("Available models:\n")

    for model in models.data:
        print(model.id)

except Exception as error:
    print("API test failed.")
    print(type(error).__name__)
    print(error)