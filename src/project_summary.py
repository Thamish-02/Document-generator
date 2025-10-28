# src/project_summary.py
import os
from ai_helper import summarize_code
from parser import parse_python

def collect_code_summaries(project_dir):
    """Collect AI summaries for all Python files in a project."""
    summaries = []

    for root, _, files in os.walk(project_dir):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                try:
                    print(f"📄 Processing {file_path}...")
                    parsed_text = parse_python(file_path)
                    ai_summary = summarize_code(parsed_text)

                    summaries.append({
                        "file": file,
                        "summary": ai_summary
                    })
                except Exception as e:
                    summaries.append({
                        "file": file,
                        "summary": f"⚠️ Error generating summary: {str(e)}"
                    })

    return summaries


def generate_project_summary(project_dir):
    """Generate one combined project-level summary."""
    summaries = collect_code_summaries(project_dir)

    header = "# 🧠 Project-Level Summary\n\n"
    content = header
    content += f"This document provides a high-level overview of the project located in `{project_dir}`.\n\n"

    for s in summaries:
        content += f"## 📄 {s['file']}\n{s['summary']}\n\n"

    # Save the summary
    # ✅ Create folders automatically
    output_dir = os.path.join(os.path.dirname(__file__), "docs", "generated")
    os.makedirs(output_dir, exist_ok=True)  # ✅ Create folders automatically
    
    output_file = os.path.join(output_dir, "project_summary.md")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"\n✅ Project summary generated at: {output_file}")


if __name__ == "__main__":
    # You can change this to your own project folder
    project_dir = "examples/sample_project"
    generate_project_summary(project_dir)
