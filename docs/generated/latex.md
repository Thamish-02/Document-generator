## AI Summary

A file named latex.py.


## Class: LatexPreprocessor

**Description:** Preprocessor for latex destined documents.

Populates the ``latex`` key in the resources dict,
adding definitions for pygments highlight styles.

Sets the authors, date and title of the latex document,
overriding the values given in the metadata.

### Function: preprocess(self, nb, resources)

**Description:** Preprocessing to apply on each notebook.

Parameters
----------
nb : NotebookNode
    Notebook being converted
resources : dictionary
    Additional resources used in the conversion process.  Allows
    preprocessors to pass variables into the Jinja engine.
