# src/project_summary.py
"""
Generate a project-level summary using the docs/ folder.

Run:
  python -m src.project_summary

Output:
  docs/project_summary.md
"""

from pathlib import Path
from .high_level_summary import summarize_project_docs

def main():
    root = Path(".").resolve()
    docs_dir = root / "docs"
    out_path = docs_dir / "project_summary.md"

    if not docs_dir.exists():
        print(f"❌ docs/ folder not found at {docs_dir}. Run renderer.py or docgen_cli.py first.")
        return

    print(f"📂 Creating project summary from {docs_dir} ...")
    summarize_project_docs(docs_dir, out_path)
    print(f"✅ Project summary written to {out_path}")

if __name__ == "__main__":
    main()
