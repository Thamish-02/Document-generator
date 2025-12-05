# src/app_dashboard.py
"""
Flask dashboard for DocGen.

Flow:
1. User opens /  -> upload form + buttons.
2. User uploads a file and clicks "Generate Documentation".
3. Backend saves file to uploads/, runs docgen_cli.py, and checks docs/documentation.pdf.
4. On success, user can click "View PDF" to open the generated documentation.
"""

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    send_from_directory,
    flash,
)
from pathlib import Path
import subprocess
import os
import traceback

app = Flask(__name__, template_folder='../templates')
app.secret_key = "dev-secret-key"  # needed for flash()

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent  # C:\Users\HAI\docgen
UPLOADS_DIR = BASE_DIR / "uploads"
DOCS_DIR = BASE_DIR / "docs"

UPLOADS_DIR.mkdir(exist_ok=True)
DOCS_DIR.mkdir(exist_ok=True)


# -----------------------
# Routes
# -----------------------

@app.route("/", methods=["GET"])
def index():
    """
    Main dashboard page.

    NOTE: If your main template is index.html instead of dashboard.html,
    change 'dashboard.html' to 'index.html' below.
    """
    return render_template("dashboard.html")


@app.route("/generate", methods=["POST"])
def generate():
    """
    Handle file upload + run DocGen pipeline.

    Steps:
    1) Save uploaded file into uploads/
    2) Run `python docgen_cli.py` in BASE_DIR
    3) Check docs/documentation.pdf exists
    4) Redirect to view_pdf() or back to index() with an error message
    """
    try:
        file = request.files.get("file")
        if not file or not file.filename or file.filename.strip() == "":
            flash("No file uploaded.")
            return redirect(url_for("index"))

        # Save uploaded file
        upload_path = UPLOADS_DIR / file.filename
        file.save(upload_path)
        app.logger.info(f"Saved upload to {upload_path}")

        # Run DocGen pipeline (generator + renderer)
        # If you want to use the uploaded file/folder as input later,
        # you can pass it via env or CLI args to docgen_cli.py.
        app.logger.info("Running docgen_cli.py ...")
        result = subprocess.run(
            ["python", "docgen_cli.py"],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore'
        )

        app.logger.info("DocGen stdout:\n%s", result.stdout)
        app.logger.info("DocGen stderr:\n%s", result.stderr)

        if result.returncode != 0:
            app.logger.error("DocGen failed with return code %s", result.returncode)
            flash("Documentation generation failed. Check server logs for details.")
            return redirect(url_for("index"))

        # Check PDF
        pdf_path = DOCS_DIR / "documentation.pdf"
        if not pdf_path.exists():
            flash("Documentation PDF was not created. Check pipeline.")
            return redirect(url_for("index"))

        flash("Documentation generated successfully!")
        return redirect(url_for("view_pdf"))

    except Exception as e:
        # Do NOT let the server crash; log and show a friendly message
        app.logger.error("Error in /generate: %s\n%s", e, traceback.format_exc())
        flash(f"Unexpected error while generating docs: {e}")
        return redirect(url_for("index"))


@app.route("/docs/pdf")
def view_pdf():
    """
    Serve the generated documentation PDF to the browser.
    """
    pdf_path = DOCS_DIR / "documentation.pdf"
    if not pdf_path.exists():
        flash("No documentation.pdf found yet. Generate documentation first.")
        return redirect(url_for("index"))
    return send_from_directory(DOCS_DIR, "documentation.pdf")


@app.route("/docs/<path:filename>")
def serve_docs_file(filename):
    """
    (Optional) Serve any file inside docs/ (Markdown, etc.).
    """
    file_path = DOCS_DIR / filename
    if not file_path.exists():
        flash(f"File not found: {filename}")
        return redirect(url_for("index"))
    return send_from_directory(DOCS_DIR, filename)


if __name__ == "__main__":
    # debug=True will show errors in the browser if something goes wrong
    app.run(host="127.0.0.1", port=5000, debug=True)
