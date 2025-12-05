# src/ai_helper.py
"""
AI helper utilities for docgen.

- Supports OpenRouter or OpenAI (prefers OpenRouter if key set).
- Provides summarize_code(chunk) -> validated JSON doc.
- Logs raw model outputs and validates JSON structure.
"""

import os
import json
import re
import ast
import time
from typing import Dict, Any, List, Optional

import requests

# -----------------------------
# Configuration
# -----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "gpt-4o-mini")

LOG_DIR = os.path.join(os.getcwd(), "logs", "llm_raw")
os.makedirs(LOG_DIR, exist_ok=True)

# -----------------------------
# Low-level helpers
# -----------------------------

def _sanitize_filename(s: str) -> str:
    return "".join(c if c.isalnum() or c in "._- " else "_" for c in s)[:120].strip()

def _save_raw_llm_output(chunk_id: str, raw_text: str) -> None:
    ts = int(time.time())
    fname = f"{ts}_{_sanitize_filename(chunk_id)}.txt"
    path = os.path.join(LOG_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        f.write(raw_text)

def _extract_json_from_text(text: str) -> Optional[str]:
    """
    Try to pull a JSON object out of arbitrary LLM text.
    We try: direct parse, then brace matching.
    """
    # direct parse
    try:
        json.loads(text)
        return text
    except Exception:
        pass

    # regex to find {...}
    for m in re.finditer(r"\{(?:[^{}]|\{[^{}]*\})*\}", text, flags=re.DOTALL):
        candidate = m.group(0)
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            continue

    # fallback: balance braces from first '{'
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, c in enumerate(text[start:], start=start):
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:i+1]
                try:
                    json.loads(candidate)
                    return candidate
                except Exception:
                    break
    return None

def validate_llm_response(raw_text: str, chunk_id: str,
                          required_keys: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Validate and extract JSON from raw LLM output.

    Returns:
      {
        "ok": bool,
        "data": dict or None,
        "error": str or None
      }
    """
    required_keys = required_keys or [
        "title",
        "short_description",
        "long_description",
        "example_usage",
    ]

    _save_raw_llm_output(chunk_id, raw_text)
    candidate = _extract_json_from_text(raw_text)
    if not candidate:
        return {"ok": False, "data": None, "error": "No JSON block found in LLM output."}

    try:
        parsed = json.loads(candidate)
    except Exception as e:
        return {"ok": False, "data": None, "error": f"JSON decode error: {e}"}

    missing = [
        k for k in required_keys
        if k not in parsed or (isinstance(parsed[k], str) and not parsed[k].strip())
    ]
    if missing:
        return {
            "ok": False,
            "data": parsed,
            "error": f"Missing or empty required keys: {missing}",
        }

    # Validate example_usage is parseable Python if possible
    ex = parsed.get("example_usage")
    if ex:
        code = ex.strip()
        if code.startswith("```"):
            code = re.sub(r"^```(?:python)?\s*", "", code)
            code = re.sub(r"\s*```$", "", code)
        try:
            ast.parse(code)
        except Exception as e:
            parsed.setdefault("_warnings", []).append(f"example_usage parse error: {e}")

    return {"ok": True, "data": parsed, "error": None}

# -----------------------------
# LLM calling
# -----------------------------

def call_llm(messages: List[Dict[str, str]],
             temperature: float = 0.1,
             max_tokens: int = 1200) -> str:
    """
    Call OpenRouter if available, otherwise OpenAI.
    Returns raw text (assistant message content).
    """
    if OPENROUTER_API_KEY:
        url = "https://api.openrouter.ai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}"}
        payload = {
            "model": OPENROUTER_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    if OPENAI_API_KEY:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": OPENAI_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    raise RuntimeError(
        "No API key found. Set OPENROUTER_API_KEY or OPENAI_API_KEY in your environment."
    )

# -----------------------------
# Prompt & summarization
# -----------------------------

def build_doc_prompt(chunk: Dict[str, Any]) -> str:
    """
    Construct a prompt to ask the LLM for documentation of a single chunk.
    chunk is expected to have: id, type, name, args, docstring, source (optional).
    """
    src = chunk.get("source", "") or ""
    if len(src) > 2000:
        src = src[:2000] + "\n# <truncated>"

    prompt = f"""
You are a senior software engineer and technical writer.
You will receive a code element and must produce a JSON object describing it.

JSON schema:
{{
  "title": string,                // short readable title
  "short_description": string,    // 1-2 line summary
  "long_description": string,     // detailed behavior + steps
  "example_usage": string,        // Python usage example
  "edge_cases": [string],         // list of edge cases or caveats
  "related": [string],            // related functions/modules/classes
  "mermaid": string               // optional mermaid diagram, or "" if not applicable
}}

Rules:
- Respond ONLY with JSON, no extra commentary.
- Use simple, clear language.
- If you are unsure, be honest; do NOT invent behavior not supported by the code.

Code metadata:
- ID: {chunk.get("id")}
- Type: {chunk.get("type")}
- Name: {chunk.get("name")}
- Args: {chunk.get("args")}
- Docstring: {chunk.get("docstring")}

Code:
{src}
"""
    return prompt.strip()

def summarize_code(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate structured documentation JSON for a given chunk.
    Returns parsed dict (doc JSON). If validation fails, returns an error dict.
    """
    chunk_id = chunk.get("id", chunk.get("name", "unknown_chunk"))
    messages = [
        {"role": "system", "content": "You are a precise, honest documentation generator for source code."},
        {"role": "user", "content": build_doc_prompt(chunk)},
    ]

    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            raw = call_llm(messages, temperature=0.1)
            result = validate_llm_response(raw, chunk_id)
            if result["ok"]:
                return result["data"]
            else:
                # Nudge: ask again with explicit JSON reminder
                messages.append({"role": "assistant", "content": raw})
                messages.append({
                    "role": "user",
                    "content": "Your previous response did not match the JSON schema. Respond again with ONLY valid JSON.",
                })
                time.sleep(0.5 * (attempt + 1))
        except Exception as e:
            last_error = str(e)
            if attempt == max_retries:
                return {"error": f"LLM call failed after retries: {e}", "_chunk": chunk_id}
            time.sleep(0.5 * (attempt + 1))

    return {  "title": chunk.get("name") or chunk_id,
        "short_description": "Documentation could not be generated automatically.",
        "long_description": "The AI summarization step failed for this element. Check logs/llm_raw for details or try regenerating when the network and API are available.",
        "example_usage": "",
        "edge_cases": [],
        "related": [],
        "mermaid": "",
        "_error": last_error if 'last_error' in locals() else "unknown_error",
    }





def build_doc_prompt(chunk: Dict[str, Any]) -> str:
    """
    Construct a high-quality prompt for documentation of a single chunk.
    """
    src = chunk.get("source", "") or ""
    if len(src) > 2500:
        src = src[:2500] + "\n# <truncated>"

    # Few-shot style: show the model what kind of JSON we want
    example_json = """
{
  "title": "Load configuration file",
  "short_description": "Reads a JSON configuration file from disk and returns it as a dictionary.",
  "long_description": "This function loads a JSON configuration file from the given path. It validates that the file exists and raises a clear error if not. If the JSON content is invalid, it raises a ValueError with details. Use this as a single source of truth for configuration.",
  "example_usage": "config = load_config('config.json')\\nprint(config['db']['host'])",
  "edge_cases": [
    "File path does not exist.",
    "JSON content is malformed.",
    "File is empty or missing required keys."
  ],
  "related": [
    "save_config",
    "DEFAULT_CONFIG_PATH"
  ],
  "mermaid": ""
}
""".strip()

    prompt = f"""
You are a senior software engineer and technical writer.

You will receive metadata and source code for one code element (function or class).
Your task is to produce a SINGLE JSON object that documents it.

Follow this JSON schema exactly:

- title: short readable title (string)
- short_description: 1–2 line summary (string)
- long_description: detailed explanation of behavior, parameters, return values, and internal logic (string)
- example_usage: a short Python usage example (string, valid Python code)
- edge_cases: list of edge cases, pitfalls, or caveats (list of strings)
- related: list of related functions/classes/modules by name (list of strings)
- mermaid: optional mermaid diagram as a string, or "" if not applicable

Example of a GOOD answer (format and tone):

{example_json}

Rules:
- Respond ONLY with valid JSON. No extra commentary.
- Do NOT include comments inside the JSON.
- Be honest and conservative: if something is not obvious from the code, do not invent behavior.

Code metadata:
- ID: {chunk.get("id")}
- Type: {chunk.get("type")}
- Name: {chunk.get("name")}
- Args: {chunk.get("args")}
- Docstring: {chunk.get("docstring")}

Source code:
{src}
"""
    return prompt.strip()
