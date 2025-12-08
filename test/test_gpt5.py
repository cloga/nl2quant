import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

token = os.getenv("LLM_GITHUB_API_KEY")
# Use the standard endpoint for OpenAI SDK with GitHub Models
endpoint = "https://models.inference.ai.azure.com"
model_name = "gpt-5"

print(f"Testing access to: {model_name}")
print(f"Endpoint: {endpoint}")
print(f"Token: {token[:15]}..." if token else "Token not found")

client = OpenAI(
    base_url=endpoint,
    api_key=token,
)

try:
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": "Hello! Which model are you?",
            }
        ],
        model=model_name,
    )
    print("\n--- Success! Response ---")
    print(response.choices[0].message.content)
except Exception as e:
    print("\n--- Error Occurred ---")
    print(e)
