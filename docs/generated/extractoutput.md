## AI Summary

A file named extractoutput.py.


### Function: guess_extension_without_jpe(mimetype)

**Description:** This function fixes a problem with '.jpe' extensions
of jpeg images which are then not recognised by latex.
For any other case, the function works in the same way
as mimetypes.guess_extension

### Function: platform_utf_8_encode(data)

**Description:** Encode data based on platform.

## Class: ExtractOutputPreprocessor

**Description:** Extracts all of the outputs from the notebook file.  The extracted
outputs are returned in the 'resources' dictionary.

### Function: preprocess_cell(self, cell, resources, cell_index)

**Description:** Apply a transformation on each cell,

Parameters
----------
cell : NotebookNode cell
    Notebook cell being processed
resources : dictionary
    Additional resources used in the conversion process.  Allows
    preprocessors to pass variables into the Jinja engine.
cell_index : int
    Index of the cell being processed (see base.py)
