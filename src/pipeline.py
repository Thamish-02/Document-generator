import os
import sys
from traverse import traverse_directory
from parser import parse_file
from chunker import chunk_markdown  # <-- IMPORT CHUNKER
from ai_utils import prepare_ai_input  # <-- add this


def run_pipeline(target_dir):
    print(f"🚀 Running pipeline on: {target_dir}")

    # Step 1: Traverse the directory
    files_by_type = traverse_directory(target_dir)

    # Step 2: Parse each supported file
    for file_list in files_by_type.values():
        for file_path in file_list:
            parse_file(file_path)

    # Step 3: Chunk the generated markdown files
    generated_dir = "docs/generated"
    for md_file in os.listdir(generated_dir):
        md_path = os.path.join(generated_dir, md_file)
        chunks = chunk_markdown(md_path)
        print(f"📑 {md_file} split into {len(chunks)} chunks")

    print("✅ Pipeline completed. Docs are in docs/generated/")

    # Prepare AI-ready input for summarization
    ai_input = prepare_ai_input()
    print(f"📢 Total AI-ready chunks: {len(ai_input)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/pipeline.py <path_to_repo>")
    else:
        run_pipeline(sys.argv[1])
