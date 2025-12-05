# tests/test_pipeline_smoke.py
"""
Very basic smoke test for DocGen pipeline.

Run with:
  python -m pytest tests/test_pipeline_smoke.py
"""

import os
from pathlib import Path
import json

def test_ai_docs_and_pdf_exist():
    root = Path(".").resolve()
    ai_file = root / "ai_docs.json"
    docs_dir = root / "docs"
    pdf_file = docs_dir / "documentation.pdf"

    # ai_docs.json must exist and be valid JSON
    assert ai_file.exists(), "ai_docs.json does not exist; run generator.py first"
    with ai_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, dict), "ai_docs.json should be a dict"

    # docs/documentation.pdf must exist
    assert docs_dir.exists(), "docs/ folder does not exist; run renderer.py"
    assert pdf_file.exists(), "documentation.pdf not found in docs/"
