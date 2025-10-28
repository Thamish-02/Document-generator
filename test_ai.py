import os
import sys

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ai_helper import summarize_code

if __name__ == "__main__":
    code = "def add(a,b): return a+b"
    result = summarize_code(code)
    print(result)