from flask import Flask, render_template_string, request
from folder_summary import generate_folder_summary
import os

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <title>Folder Summary Generator</title>
  <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
</head>
<body>
  <h1>📂 AI Folder Summary Generator</h1>
  <form method="post">
    <button type="submit">Generate Folder Summary</button>
  </form>
  {% if summary %}
  <h2>✅ Generated Summary:</h2>
  <pre>{{ summary }}</pre>
  {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    summary = None
    if request.method == "POST":
        output_path = generate_folder_summary("examples")
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                summary = f.read()
        else:
            summary = "❌ Summary file not found."
    return render_template_string(HTML_TEMPLATE, summary=summary)

if __name__ == "__main__":
    app.run(debug=True, port=8000)
