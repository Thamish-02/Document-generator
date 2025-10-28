import os
from parser import parse_python  # ✅ correct function name
from ai_helper import summarize_code

def generate_folder_summary(folder_path="examples"):
    """Generate documentation + AI summary for all Python files in a folder."""
    combined_summary = ["# 📂 Project Summary\n"]

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                print(f"🔍 Processing {file_path}...")

                # ✅ Get documentation text from parser
                doc_content = parse_python(file_path)

                # ✅ Generate AI summary using the helper
                ai_summary = summarize_code(doc_content)

                combined_summary.append(f"\n## 🧩 {file}\n")
                combined_summary.append(ai_summary)
                combined_summary.append(doc_content)

    # ✅ Ensure output directory exists
    output_dir = os.path.join("docs", "generated")
    os.makedirs(output_dir, exist_ok=True)

    # ✅ Save summary file
    output_file = os.path.join(output_dir, "folder_summary.md")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(combined_summary))

    print(f"\n✅ Folder summary generated: {output_file}")
    return output_file


if __name__ == "__main__":
    generate_folder_summary("examples")
