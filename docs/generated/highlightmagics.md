## AI Summary

A file named highlightmagics.py.


## Class: HighlightMagicsPreprocessor

**Description:** Detects and tags code cells that use a different languages than Python.

### Function: __init__(self, config)

**Description:** Public constructor

### Function: which_magic_language(self, source)

**Description:** When a cell uses another language through a magic extension,
the other language is returned.
If no language magic is detected, this function returns None.

Parameters
----------
source: str
    Source code of the cell to highlight

### Function: preprocess_cell(self, cell, resources, cell_index)

**Description:** Tags cells using a magic extension language

Parameters
----------
cell : NotebookNode cell
    Notebook cell being processed
resources : dictionary
    Additional resources used in the conversion process.  Allows
    preprocessors to pass variables into the Jinja engine.
cell_index : int
    Index of the cell being processed (see base.py)
