from llm import call_llm


try:
    answer = call_llm(
        system_prompt="You are a helpful assistant.",
        user_prompt="Reply with exactly: API connection successful",
    )

    print(answer)

except Exception as error:
    print("LLM test failed.")
    print(type(error).__name__)
    print(error)