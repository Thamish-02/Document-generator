from flask import Flask, render_template, request, send_file
import os
from parser import parse_file  # Reuse your existing parser
from datetime import datetime
# --- New route for Project Summary ---
from project_summary import generate_project_summary  # Import your function

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


@app.route("/generate_summary", methods=["GET"])
def generate_summary():
    try:
        # Generate the project summary
        project_dir = "examples/sample_project"  # Default project directory
        generate_project_summary(project_dir)
        summary_path = os.path.join("docs", "generated", "project_summary.md")
        return send_file(summary_path, as_attachment=True)
    except Exception as e:
        return f"<h3>Error generating summary: {str(e)}</h3>"


if __name__ == "__main__":
    app.run(debug=True)