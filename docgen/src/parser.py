# src/parser.py
import os
import ast
import nbformat
from pathlib import Path
from ai_helper import summarize_code

OUTPUT_DIR = "docs/generated"

def parse_python(file_path):
    """Extract functions and classes from .py file as markdown."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Generate AI summary of the code
        ai_summary = summarize_code(content)
        lines = ["## AI Summary\n", ai_summary, "\n"]
        
        # Parse the AST to get proper function and class definitions
        try:
            tree = ast.parse(content)
            # ... existing code ...
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Get function signature
                    args = [arg.arg for arg in node.args.args]
                    sig = f"{node.name}({', '.join(args)})"
                    lines.append(f"### Function: {sig}\n")
                    
                    # Add docstring if available
                    if ast.get_docstring(node):
                        lines.append(f"**Description:** {ast.get_docstring(node)}\n")
                        
                elif isinstance(node, ast.ClassDef):
                    lines.append(f"## Class: {node.name}\n")
                    
                    # Add class docstring if available
                    if ast.get_docstring(node):
                        lines.append(f"**Description:** {ast.get_docstring(node)}\n")
            
            return "\n".join(lines) or "No functions/classes found."
            
        except SyntaxError as e:
            return f"Syntax error in Python file: {str(e)}"
            
    except FileNotFoundError:
        return f"File not found: {file_path}"
    except PermissionError:
        return f"Permission denied: {file_path}"
    except UnicodeDecodeError:
        return f"Encoding error: {file_path}"
    except Exception as e:
        return f"Error parsing Python file: {str(e)}"

def parse_notebook(file_path):
    """Extract code cells from Jupyter notebook."""
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            return f"Notebook file not found: {file_path}"
            
        nb = nbformat.read(file_path, as_version=4)
        code_cells = []
        
        for cell in nb.cells:
            if cell.cell_type == "code":
                # Handle both string and list formats for cell.source
                if isinstance(cell.source, list):
                    source_content = "".join(cell.source)
                else:
                    source_content = cell.source
                
                if source_content.strip():  # Only add non-empty cells
                    code_cells.append("```python\n" + source_content + "\n```")
        
        return "\n\n".join(code_cells) or "No code cells found."
        
    except FileNotFoundError:
        return f"Notebook file not found: {file_path}"
    except nbformat.ValidationError as e:
        return f"Invalid notebook format: {str(e)}"
    except Exception as e:
        return f"Error parsing notebook: {str(e)}"

def parse_markdown(file_path):
    """Return raw markdown text."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"Markdown file not found: {file_path}"
    except PermissionError:
        return f"Permission denied: {file_path}"
    except UnicodeDecodeError:
        return f"Encoding error: {file_path}"
    except Exception as e:
        return f"Error reading markdown file: {str(e)}"

def save_markdown(file_path, content):
    """Save content to markdown file in output directory."""
    try:
        # Create output directory if it doesn't exist
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        
        base_name = os.path.basename(file_path)
        # Remove original extension and add .md
        name_without_ext = os.path.splitext(base_name)[0]
        md_file = os.path.join(OUTPUT_DIR, name_without_ext + ".md")
        
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✅ Parsed {file_path} → {md_file}")
        
    except PermissionError:
        print(f"❌ Permission denied creating output file for {file_path}")
    except Exception as e:
        print(f"❌ Error saving markdown for {file_path}: {str(e)}")

def parse_file(file_path):
    """Parse a single file based on its extension."""
    try:
        # Validate file exists
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return
            
        if file_path.endswith(".py"):
            content = parse_python(file_path)
        elif file_path.endswith(".ipynb"):
            content = parse_notebook(file_path)
        elif file_path.endswith(".md"):
            content = parse_markdown(file_path)
        else:
            print(f"Skipping unsupported file: {file_path}")
            return
            
        save_markdown(file_path, content)
        
    except Exception as e:
        print(f"❌ Error processing {file_path}: {str(e)}")

if __name__ == "__main__":
    # Process files in both sample_project and my_project directories
    directories = ["examples/sample_project", "examples/my_project"]
    
    for sample_dir in directories:
        # Check if directory exists
        if not os.path.exists(sample_dir):
            print(f"⚠️  Directory not found: {sample_dir} (skipping)")
            continue
        
        print(f"Processing files in {sample_dir}...")
        for root, _, files in os.walk(sample_dir):
            for file in files:
                file_path = os.path.join(root, file)
                parse_file(file_path)