import os
from openai import OpenAI

# Initialize client using OpenRouter endpoint
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("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")  # Store your key as environment variable
)

def summarize_code(code_snippet):
    """Summarize or explain code using OpenRouter AI model."""
    try:
        prompt = f"Explain this Python code in simple technical terms:\n\n{code_snippet}"
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # you can switch to 'mistralai/mistral-7b' or 'meta-llama/llama-3-8b-instruct'
            messages=[{"role": "user", "content": prompt}],
        )

        content = response.choices[0].message.content
        return content.strip() if content else ""
    except Exception as e:
        print(f"Error generating AI summary: {e}")
        return "AI summary could not be generated due to an error."
