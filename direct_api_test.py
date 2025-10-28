import os
from openai import OpenAI

# Use the provided API key directly
api_key = "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"

client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)

try:
    print("Attempting to make API call...")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say this is a test"}],
        max_tokens=10
    )
    
    print("Response received:")
    print(response.choices[0].message.content)
except Exception as e:
    print(f"Error occurred: {e}")
    print(f"Error type: {type(e)}")