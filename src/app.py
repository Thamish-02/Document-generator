from flask import Flask, render_template, request
import os
from parser import parse_file  # Reuse your existing parser
from datetime import datetime

# Get the directory of the current file (src)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the project root directory (parent of src)
project_root = os.path.dirname(current_dir)
# Set the template directory
template_dir = os.path.join(project_root, 'templates')
# Set the static directory
static_dir = os.path.join(project_root, 'static')

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    generated_doc = None

    if request.method == "POST":
        file = request.files["file"]
        if file and file.filename:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

            try:
                # Call your existing parser logic
                parse_file(file_path)

                # Get the generated markdown file
                base_name = os.path.splitext(file.filename)[0] + ".md"
                generated_path = os.path.join("docs", "generated", base_name)

                if os.path.exists(generated_path):
                    with open(generated_path, "r", encoding="utf-8") as f:
                        generated_doc = f.read()
            except Exception as e:
                generated_doc = f"Error processing file: {str(e)}"

    return render_template("index.html", generated_doc=generated_doc)

if __name__ == "__main__":
    app.run(debug=True)