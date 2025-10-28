# src/app_dashboard.py
import os
from flask import Flask, render_template, request, redirect, url_for, send_file, flash
from werkzeug.utils import secure_filename

# Import your existing modules (adjust names if your functions differ)
from parser import parse_file          # expects parse_file(file_path) -> writes docs/generated/<name>.md
from folder_summary import generate_folder_summary
from project_summary import generate_project_summary

# Config
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
DOCS_FOLDER = os.path.join(BASE_DIR, "docs", "generated")
ALLOWED_EXTENSIONS = {".py", ".ipynb", ".md"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DOCS_FOLDER, exist_ok=True)

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"), static_folder=os.path.join(BASE_DIR, "static"))
app.secret_key = "change-this-secret"  # change to a secure random key for production

def allowed_file(filename: str) -> bool:
    _, ext = os.path.splitext(filename)
    return ext.lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET"])
def index():
    return render_template("dashboard.html", generated_doc=None, generated_name=None)

# ------------ Single-file upload and generate ------------
@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        flash("No file part")
        return redirect(url_for("index"))

    file = request.files["file"]
    if file.filename == "":
        flash("No selected file")
        return redirect(url_for("index"))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        # Call your existing parser to handle the file and produce markdown
        try:
            parse_file(file_path)  # should write docs/generated/<basename>.md
        except Exception as e:
            flash(f"Error parsing file: {e}")
            return redirect(url_for("index"))

        md_name = os.path.splitext(filename)[0] + ".md"
        md_path = os.path.join(DOCS_FOLDER, md_name)
        if os.path.exists(md_path):
            with open(md_path, "r", encoding="utf-8") as f:
                content = f.read()
            return render_template("dashboard.html", generated_doc=content, generated_name=md_name)
        else:
            flash("Generated file not found.")
            return redirect(url_for("index"))

    else:
        flash("Unsupported file type.")
        return redirect(url_for("index"))

# ------------ Generate folder/project summary ------------
@app.route("/generate_project_summary", methods=["POST"])
def project_summary():
    try:
        # By default run project_summary on repo root or examples folder
        # generate_project_summary should return the output file path OR we adjust below to find file
        output_path = generate_project_summary(".") if generate_project_summary.__code__.co_argcount == 0 else generate_project_summary("examples")
    except TypeError:
        # fallback if signature expects a project_dir param
        output_path = generate_project_summary("examples")

    if not output_path:
        # if your function writes to docs/generated/project_summary.md, build the path
        output_path = os.path.join(DOCS_FOLDER, "project_summary.md")

    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            content = f.read()
        return render_template("dashboard.html", generated_doc=content, generated_name=os.path.basename(output_path))
    else:
        flash("Project summary not found. Check server logs.")
        return redirect(url_for("index"))

# ------------ Generate folder summary for examples folder ------------
@app.route("/generate_folder_summary", methods=["POST"])
def folder_summary():
    try:
        output_path = generate_folder_summary("examples")
    except Exception as e:
        flash(f"Folder summary error: {e}")
        return redirect(url_for("index"))

    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            content = f.read()
        return render_template("dashboard.html", generated_doc=content, generated_name=os.path.basename(output_path))
    else:
        flash("Folder summary file not created.")
        return redirect(url_for("index"))

# ------------ Download a generated doc ------------
@app.route("/download/<path:filename>", methods=["GET"])
def download(filename):
    safe_path = os.path.join(DOCS_FOLDER, secure_filename(filename))
    if os.path.exists(safe_path):
        return send_file(safe_path, as_attachment=True)
    else:
        flash("File not found.")
        return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
