import os
from openai import OpenAI
from dotenv import load_dotenv
import os

# ✅ Load .env file
load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

# Try OpenRouter first
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

if OPENROUTER_KEY:
    client = OpenAI(
        api_key=OPENROUTER_KEY,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://github.com/HarveyHunt/docgen",  # Optional, for including your app on openrouter.ai rankings
            "X-Title": "DocGen"  # Optional, shows in rankings on openrouter.ai
        }
    )
    MODEL_NAME = "gpt-4o-mini"
elif OPENAI_KEY:
    client = OpenAI(api_key=OPENAI_KEY)
    MODEL_NAME = "gpt-4o-mini"
else:
    raise ValueError("❌ No API key found! Set either OPENROUTER_API_KEY or OPENAI_API_KEY.")

def summarize_code(code: str) -> str:
    """Send code to AI model and get a summary."""
    try:
        prompt = f"Summarize the following Python code in simple terms:\n\n{code}"

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250
        )

        content = response.choices[0].message.content
        return content.strip() if content else "No summary generated."

    except Exception as e:
        return f"AI summary could not be generated due to an error: {e}"