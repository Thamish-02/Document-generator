# src/folder_summary.py
"""
Generate a high-level folder summary using the docs/ folder.

Run:
  python -m src.folder_summary

Output:
  docs/folder_summary.md
"""

from pathlib import Path
from .high_level_summary import summarize_folder_docs

def main():
    root = Path(".").resolve()
    docs_dir = root / "docs"  # using the output docs folder
    out_path = docs_dir / "folder_summary.md"

    if not docs_dir.exists():
        print(f"❌ docs/ folder not found at {docs_dir}. Run renderer.py or docgen_cli.py first.")
        return

    print(f"📂 Creating folder summary from {docs_dir} ...")
    summarize_folder_docs(docs_dir, out_path)
    print(f"✅ Folder summary written to {out_path}")

if __name__ == "__main__":
    main()
