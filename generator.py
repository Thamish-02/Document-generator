import json
import time
from pathlib import Path
from src.ai_helper import summarize_code

# Mock function for testing without API access
def summarize_code(chunk):
    """
    Generate realistic mock documentation based on the code chunk.
    This simulates what the real AI would generate.
    """
    chunk_type = chunk.get("type", "element")
    chunk_name = chunk.get("name", "unknown")
    chunk_args = chunk.get("args", [])
    chunk_source = chunk.get("source", "")
    
    # Generate mock documentation based on the chunk type and content
    if chunk_type == "function" and "hello_world" in chunk_name:
        return {
            "title": "Hello World Function",
            "short_description": "Returns a simple greeting message.",
            "long_description": "This function is a basic implementation that returns the classic \"Hello, World!\" greeting. It takes no parameters and always returns the same string. This is commonly used as a simple demonstration of a function's basic structure and return value.",
            "example_usage": "greeting = hello_world()\nprint(greeting)  # Output: Hello, World!",
            "edge_cases": [
                "No edge cases - function always returns the same static string.",
                "Function does not accept any parameters, so no type validation needed."
            ],
            "related": [
                "print",
                "str"
            ],
            "mermaid": ""
        }
    elif chunk_type == "function" and "add_numbers" in chunk_name:
        return {
            "title": "Add Two Numbers",
            "short_description": "Adds two numeric values and returns the result.",
            "long_description": "This function accepts two arguments, `a` and `b`, and returns their sum. It assumes both inputs support the `+` operator. The function does not perform type validation, so non-numeric or incompatible types may raise a TypeError at runtime.",
            "example_usage": "result = add_numbers(2, 3)\nprint(result)  # 5",
            "edge_cases": [
                "Passing non-numeric types like strings or lists will raise a TypeError.",
                "Very large integers may still work, but floating-point addition can introduce rounding errors."
            ],
            "related": [
                "subtract_numbers",
                "Calculator"
            ],
            "mermaid": ""
        }
    elif chunk_type == "class" and "Calculator" in chunk_name:
        return {
            "title": "Calculator Class",
            "short_description": "A simple calculator class for basic arithmetic operations with history tracking.",
            "long_description": "This class provides basic arithmetic operations, specifically multiplication, with built-in history tracking. When initialized, it creates an empty history list. The multiply method performs multiplication of two numbers and records the operation in the history. The get_history method returns all recorded operations.",
            "example_usage": "calc = Calculator()\nresult = calc.multiply(4, 5)\nprint(result)  # 20\nprint(calc.get_history())  # ['4 * 5 = 20']",
            "edge_cases": [
                "Multiplying very large numbers may lead to overflow in some systems.",
                "Non-numeric inputs will raise a TypeError during multiplication.",
                "History grows unbounded with each operation, which could consume significant memory over time."
            ],
            "related": [
                "add_numbers",
                "math"
            ],
            "mermaid": ""
        }
    else:
        # Generic mock for any other chunk
        return {
            "title": f"{chunk_name} {chunk_type.title()}",
            "short_description": f"A {chunk_type} named {chunk_name}.",
            "long_description": f"This {chunk_type} performs operations related to {chunk_name}. The implementation details can be found in the source code.",
            "example_usage": f"# Example usage would depend on the specific {chunk_type}",
            "edge_cases": [
                "General edge cases would depend on the specific implementation."
            ],
            "related": [
                "Related elements would be identified in a real implementation."
            ],
            "mermaid": ""
        }

CHUNKS_FILE = Path("chunks.json")
OUT_FILE = Path("ai_docs.json")

def load_chunks(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Chunks file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def main():
    start_time = time.time()
    print(f"Loading chunks from {CHUNKS_FILE} ...")
    chunks = load_chunks(CHUNKS_FILE)

    results = {}
    if OUT_FILE.exists():
        with OUT_FILE.open("r", encoding="utf-8") as f:
            try:
                results = json.load(f)
            except Exception:
                results = {}

    total = len(chunks)
    done = 0
    skipped = 0
    failed = 0

    for chunk in chunks:
        chunk_id = chunk.get("id") or chunk.get("name")
        if not chunk_id:
            continue

        if chunk_id in results:
            print(f"[SKIP] {chunk_id} already documented.")
            skipped += 1
            continue

        print(f"[GEN] {chunk_id} ...", end="", flush=True)
        t0 = time.time()
        doc = summarize_code(chunk)
        elapsed = time.time() - t0

        if "error" in doc and not doc.get("title"):
            print(f" ❌ (error: {doc['error']}, {elapsed:.2f}s)")
            failed += 1
        else:
            print(f" ✅ ({elapsed:.2f}s)")
        results[chunk_id] = {"chunk": chunk, "doc": doc}

        with OUT_FILE.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        done += 1
        time.sleep(0.2)

    total_time = time.time() - start_time
    print(f"\nDone. Output: {OUT_FILE}")
    print(f"Summary: total={total}, generated={done}, skipped={skipped}, failed={failed}")
    print(f"Total time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()