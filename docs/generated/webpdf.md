## AI Summary

A file named webpdf.py.


## Class: WebPDFExporter

**Description:** Writer designed to write to PDF files.

This inherits from :class:`HTMLExporter`. It creates the HTML using the
template machinery, and then run playwright to create a pdf.

### Function: _file_extension_default(self)

### Function: _template_name_default(self)

### Function: run_playwright(self, html)

**Description:** Run playwright.

### Function: from_notebook_node(self, nb, resources)

**Description:** Convert from a notebook node.

### Function: run_coroutine(coro)

**Description:** Run an internal coroutine.
