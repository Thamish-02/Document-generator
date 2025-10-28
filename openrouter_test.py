import os
from openai import OpenAI

# Initialize the OpenAI client for OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
)

try:
    completion = client.chat.completions.create(
        extra_headers={
            "HTTP-Referer": "https://github.com/HarveyHunt/docgen",  # Optional, for including your app on openrouter.ai rankings.
            "X-Title": "DocGen",  # Optional. Shows in rankings on openrouter.ai.
        },
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Say this is a test"},
        ],
        temperature=0.7,
        max_tokens=10
    )
    
    print("Success! Response:")
    print(completion.choices[0].message.content)
except Exception as e:
    print(f"Error: {e}")