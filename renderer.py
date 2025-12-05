# renderer.py
"""
Render AI docs to Markdown files + a single PDF.

Inputs:
  ai_docs.json  (from generator.py)

Outputs:
  docs/index.md
  docs/<chunk>.md
  docs/documentation.pdf
"""

import os
import json
from pathlib import Path
from datetime import datetime

import fitz  # PyMuPDF
import markdown

AI_FILE = Path("ai_docs.json")
OUT_DIR = Path("docs")

def safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "._- " else "_" for c in s)[:120]

def load_ai_docs(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"AI docs file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def make_markdown_page(doc_id: str, chunk: dict, doc: dict) -> str:
    title = doc.get("title") or chunk.get("name") or doc_id
    sig = chunk.get("signature") or f"{chunk.get('name', '')}({', '.join(chunk.get('args', []))})"
    lines = []
    lines.append(f"# {title}\n\n")
    lines.append(f"**ID:** `{doc_id}`  \n")
    lines.append(f"**Signature:** `{sig}`\n\n")
    if doc.get("short_description"):
        lines.append(f"**Summary:** {doc['short_description']}\n\n")
    lines.append("## Description\n\n")
    lines.append((doc.get("long_description") or "") + "\n\n")
    if doc.get("mermaid"):
        lines.append("```mermaid\n")
        lines.append(doc["mermaid"] + "\n")
        lines.append("```\n\n")
    if doc.get("example_usage"):
        lines.append("## Example\n\n```python\n")
        lines.append(doc["example_usage"].strip() + "\n")
        lines.append("```\n\n")
    edges = doc.get("edge_cases") or []
    if edges:
        lines.append("## Edge cases & Notes\n\n")
        for e in edges:
            lines.append(f"- {e}\n")
        lines.append("\n")
    return "".join(lines)

def write_markdown_files(ai_data: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index_lines = [
        "# Project Documentation\n\n",
        f"Generated: {datetime.utcnow().isoformat()} UTC\n\n",
        "## Index\n\n",
    ]
    for doc_id, payload in ai_data.items():
        chunk = payload.get("chunk", {})
        doc = payload.get("doc", {})
        fname = safe_name(doc_id) + ".md"
        path = OUT_DIR / fname
        md_text = make_markdown_page(doc_id, chunk, doc)
        path.write_text(md_text, encoding="utf-8")
        display_name = chunk.get("name") or doc.get("title") or doc_id
        index_lines.append(f"- [{display_name}](./{fname})\n")
    (OUT_DIR / "index.md").write_text("".join(index_lines), encoding="utf-8")

def markdowns_to_pdf(out_pdf: Path) -> None:
    """
    Very simple Markdown -> HTML -> plain text -> PDF using PyMuPDF.
    (Later you can upgrade to prettier HTML->PDF pipeline.)
    """
    files = [OUT_DIR / "index.md"] + sorted(
        p for p in OUT_DIR.iterdir()
        if p.suffix == ".md" and p.name != "index.md"
    )
    doc = fitz.open()
    for md_file in files:
        md_text = md_file.read_text(encoding="utf-8")
        html = markdown.markdown(md_text, extensions=["fenced_code", "tables"])
        # naive HTML -> plain text
        import re
        text = re.sub(r"<[^>]+>", "", html)
        lines = text.splitlines()
        lines_per_page = 40
        for i in range(0, len(lines), lines_per_page):
            page = doc.new_page()
            block = "\n".join(lines[i:i+lines_per_page])
            page.insert_text((50, 50), block, fontsize=11)
    doc.save(str(out_pdf))

def main():
    print(f"Loading AI docs from {AI_FILE} ...")
    ai_data = load_ai_docs(AI_FILE)
    print("Writing Markdown files to docs/ ...")
    write_markdown_files(ai_data)
    out_pdf = OUT_DIR / "documentation.pdf"
    print("Generating PDF:", out_pdf)
    markdowns_to_pdf(out_pdf)
    print("✅ Done. Markdown + PDF generated in docs/")

if __name__ == "__main__":
    main()
