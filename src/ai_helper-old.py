import os
from openai import OpenAI
from dotenv import load_dotenv
import os
import json, os, re, ast, time
from typing import Dict, Any, List, Optional

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
# src/ai_helper.py  (append these helpers near your existing LLM call code)

LOG_DIR = os.path.join(os.getcwd(), "logs", "llm_raw")
os.makedirs(LOG_DIR, exist_ok=True)

JSON_CANDIDATE_RE = re.compile(r"(\{(?:[^{}]|(?R))*\})", flags=re.DOTALL)  # recursive pattern if supported

def _extract_json_from_text(text: str) -> Optional[str]:
    """Try common strategies to extract the JSON blob from model text."""
    # First try direct parse
    try:
        json.loads(text)
        return text
    except Exception:
        pass
    # Try to find brace-delimited candidates
    for m in re.finditer(r"\{(?:[^{}]|\{[^{}]*\})*\}", text, flags=re.DOTALL):
        candidate = m.group(0)
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            continue
    # As last resort try to balance braces progressively
    start = text.find('{')
    if start == -1:
        return None
    depth = 0
    for i in range(start, min(len(text), start + 20000)):
        c = text[i]
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                candidate = text[start:i+1]
                try:
                    json.loads(candidate)
                    return candidate
                except Exception:
                    break
    return None

def save_raw_llm_output(id: str, raw_text: str):
    fname = os.path.join(LOG_DIR, f"{time.time():.0f}_{sanitize_filename(id)}.txt")
    with open(fname, "w", encoding="utf-8") as f:
        f.write(raw_text)

def sanitize_filename(s: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in s)[:120]

def validate_llm_response(raw_text: str, id: str, required_keys: List[str]=None) -> Dict[str, Any]:
    """
    Validate and extract JSON from raw LLM output.
    Returns dict: {"ok": bool, "data": parsed_json_or_None, "error": message}
    Also writes raw output to logs.
    """
    required_keys = required_keys or ["title","short_description","long_description","example_usage"]
    save_raw_llm_output(id, raw_text)
    candidate = _extract_json_from_text(raw_text)
    if not candidate:
        return {"ok": False, "data": None, "error": "No JSON found in LLM output"}
    try:
        parsed = json.loads(candidate)
    except Exception as e:
        return {"ok": False, "data": None, "error": f"JSON decode error: {e}"}
    # Quick schema checks
    missing = [k for k in required_keys if k not in parsed or (isinstance(parsed[k], str) and parsed[k].strip() == "")]
    if missing:
        return {"ok": False, "data": parsed, "error": f"Missing required keys or empty: {missing}"}
    # Validate example_usage: minimal check parseable python
    if "example_usage" in parsed and parsed["example_usage"]:
        example = parsed["example_usage"]
        try:
            # allow code blocks with ```python ... ```
            if example.strip().startswith("```"):
                # strip fences
                example = re.sub(r"^```(?:python)?\n", "", example)
                example = re.sub(r"\n```$", "", example)
            ast.parse(example)
        except Exception as e:
            # Not fatal, but warn
            parsed.setdefault("_warnings", []).append(f"example_usage parse error: {e}")
    return {"ok": True, "data": parsed, "error": None}
