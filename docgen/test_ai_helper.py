from src.ai_helper import summarize_code

# Simple test code
test_code = """
def hello_world():
    return "Hello, World!"
"""

print("Testing AI summary generation...")
try:
    summary = summarize_code(test_code)
    print("AI Summary:")
    print(summary)
except Exception as e:
    print(f"Error: {e}")
print("Done.")