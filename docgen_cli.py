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
    
    This function will parse the input file and generate chunks for documentation.
    """
    import json
    import sys
    import os
    
    # Add src to path for imports
    src_path = os.path.join(os.path.dirname(__file__), 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    try:
        # Try direct import first
        import parser
        import traverse
        from parser import parse_file
        from traverse import traverse_directory
    except ImportError:
        try:
            # Try relative import
            from src import parser, traverse
            from src.parser import parse_file
            from src.traverse import traverse_directory
        except ImportError:
            print("[ERROR] Could not import parser modules")
            sys.exit(1)
    
    # Parse the input file
    if input_path.is_file():
        print(f"Parsing file: {input_path}")
        parse_file(str(input_path))
    elif input_path.is_dir():
        print(f"Parsing directory: {input_path}")
        files_by_type = traverse_directory(str(input_path))
        for file_list in files_by_type.values():
            for file_path in file_list:
                parse_file(file_path)
    else:
        print(f"[ERROR] Input path does not exist: {input_path}")
        sys.exit(1)
    
    # Generate chunks from parsed files
    chunks = []
    docs_generated_dir = Path("docs/generated")
    if docs_generated_dir.exists():
        for md_file in docs_generated_dir.iterdir():
            if md_file.suffix == ".md":
                with open(md_file, "r", encoding="utf-8") as f:
                    content = f.read()
                # Create a simple chunk for the file
                chunk = {
                    "id": md_file.stem,
                    "type": "file",
                    "name": md_file.stem,
                    "args": [],
                    "docstring": "",
                    "source": content
                }
                chunks.append(chunk)
    
    # Save chunks to chunks.json
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    
    print(f"Generated {len(chunks)} chunks")

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