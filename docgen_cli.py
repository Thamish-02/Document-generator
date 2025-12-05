# docgen_cli.py
"""
High-level CLI for DocGen.

Usage:
  python docgen_cli.py --input examples/sample_project --out docs

Steps:
1. (Optional) Run your parsing + chunking pipeline to generate chunks.json.
2. Run generator.py to produce ai_docs.json.
3. Run renderer.py to produce docs/ and documentation.pdf.
"""

import argparse
import sys
from pathlib import Path

def run_parsing_pipeline(input_path: Path, chunks_path: Path):
    """
    Hook into your existing parsing + chunking pipeline.

    Right now this is a placeholder; you should connect it to your real code.
    For now, we just check that chunks.json exists.
    """
    if not chunks_path.exists():
        print(f"[ERROR] chunks.json not found at {chunks_path}")
        print("Please run your parser/chunker pipeline first to generate chunks.json.")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Run full DocGen pipeline: parse -> LLM -> docs -> PDF"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=False,
        default=".",
        help="Path to the project folder to document (currently used by your parser pipeline).",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=False,
        default="docs",
        help="Output folder for final docs (renderer.py controls details).",
    )
    args = parser.parse_args()

    root = Path(".").resolve()
    input_path = (root / args.input).resolve()
    out_dir = (root / args.out).resolve()

    print(f"Input project: {input_path}")
    print(f"Output docs folder: {out_dir}")

    chunks_path = root / "chunks.json"

    # 1) Run parsing + chunking to produce chunks.json (or just check it exists)
    print("Step 1: Checking parsing/chunking output (chunks.json)...")
    run_parsing_pipeline(input_path, chunks_path)

    # 2) Run generator.py (LLM stage)
    print("Step 2: Running generator.py (AI documentation)...")
    from generator import main as gen_main
    gen_main()

    # 3) Run renderer.py (Markdown + PDF)
    print("Step 3: Running renderer.py (Markdown + PDF)...")
    from renderer import main as rend_main
    rend_main()

    print("Done! Check the docs/ folder for Markdown files and documentation.pdf")

if __name__ == "__main__":
    main()