import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ai_helper import summarize_code

# Read the test file
with open("examples/sample_project/test1.py", "r") as f:
    content = f.read()

# Generate AI summary
summary = summarize_code(content)

# Save the summary to a file
with open("ai_summary_output.txt", "w") as f:
    f.write("## AI Summary\n")
    f.write(summary)
    f.write("\n")

print("AI summary saved to ai_summary_output.txt")