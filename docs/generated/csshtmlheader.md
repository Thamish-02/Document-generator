## AI Summary

A file named csshtmlheader.py.


## Class: CSSHTMLHeaderPreprocessor

**Description:** Preprocessor used to pre-process notebook for HTML output.  Adds IPython notebook
front-end CSS and Pygments CSS to HTML output.

### Function: __init__(self)

**Description:** Initialize the preprocessor.

### Function: preprocess(self, nb, resources)

**Description:** Fetch and add CSS to the resource dictionary

Fetch CSS from IPython and Pygments to add at the beginning
of the html files.  Add this css in resources in the
"inlining.css" key

Parameters
----------
nb : NotebookNode
    Notebook being converted
resources : dictionary
    Additional resources used in the conversion process.  Allows
    preprocessors to pass variables into the Jinja engine.

### Function: _generate_header(self, resources)

**Description:** Fills self.header with lines of CSS extracted from IPython
and Pygments.

### Function: _hash(self, filename)

**Description:** Compute the hash of a file.
