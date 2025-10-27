import os

def chunk_markdown(file_path, max_lines=50):
    """
    Splits a markdown file into smaller text chunks.
    Each chunk will have up to `max_lines`.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    chunks, current = [], []
    for i, line in enumerate(lines, 1):
        current.append(line)
        if i % max_lines == 0:
            chunks.append("".join(current))
            current = []
    if current:
        chunks.append("".join(current))

    return chunks
