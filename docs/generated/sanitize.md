## AI Summary

A file named sanitize.py.


## Class: SanitizeHTML

**Description:** A preprocessor to sanitize html.

### Function: _get_default_css_sanitizer()

### Function: preprocess_cell(self, cell, resources, cell_index)

**Description:** Sanitize potentially-dangerous contents of the cell.

Cell Types:
  raw:
    Sanitize literal HTML
  markdown:
    Sanitize literal HTML
  code:
    Sanitize outputs that could result in code execution

### Function: sanitize_code_outputs(self, outputs)

**Description:** Sanitize code cell outputs.

Removes 'text/javascript' fields from display_data outputs, and
runs `sanitize_html_tags` over 'text/html'.

### Function: sanitize_html_tags(self, html_str)

**Description:** Sanitize a string containing raw HTML tags.
