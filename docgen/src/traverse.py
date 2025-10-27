import os
import sys

def traverse_directory(root_dir):
    files = {"python": [], "notebooks": [], "markdown": [], "other": []}

    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            file_path = os.path.join(dirpath, file)
            if file.endswith(".py"):
                files["python"].append(file_path)
            elif file.endswith(".ipynb"):
                files["notebooks"].append(file_path)
            elif file.endswith(".md"):
                files["markdown"].append(file_path)
            else:
                files["other"].append(file_path)

    # Print summary
    for category, file_list in files.items():
        print(f"{category}: {len(file_list)} files")
    return files

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/traverse.py <path_to_repo>")
    else:
        traverse_directory(sys.argv[1])
