# src/ai_utils.py
import os
from chunker import chunk_markdown

def prepare_ai_input(generated_dir="docs/generated"):
    ai_input = []
    for md_file in os.listdir(generated_dir):
        md_path = os.path.join(generated_dir, md_file)
        chunks = chunk_markdown(md_path)
        for i, chunk in enumerate(chunks, start=1):
            ai_input.append({
                "file_name": md_file,
                "chunk_number": i,
                "text": chunk
            })
    return ai_input
