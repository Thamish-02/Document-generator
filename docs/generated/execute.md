## AI Summary

A file named execute.py.


### Function: executenb()

**Description:** DEPRECATED.

## Class: ExecutePreprocessor

**Description:** Executes all the cells in a notebook

### Function: __init__(self)

**Description:** Initialize the preprocessor.

### Function: _check_assign_resources(self, resources)

### Function: preprocess(self, nb, resources, km)

**Description:** Preprocess notebook executing each code cell.

The input argument *nb* is modified in-place.

Note that this function recalls NotebookClient.__init__, which may look wrong.
However since the preprocess call acts line an init on execution state it's expected.
Therefore, we need to capture it here again to properly reset because traitlet
assignments are not passed. There is a risk if traitlets apply any side effects for
dual init.
The risk should be manageable, and this approach minimizes side-effects relative
to other alternatives.

One alternative but rejected implementation would be to copy the client's init internals
which has already gotten out of sync with nbclient 0.5 release before nbconvert 6.0 released.

Parameters
----------
nb : NotebookNode
    Notebook being executed.
resources : dictionary (optional)
    Additional resources used in the conversion process. For example,
    passing ``{'metadata': {'path': run_path}}`` sets the
    execution path to ``run_path``.
km: KernelManager (optional)
    Optional kernel manager. If none is provided, a kernel manager will
    be created.

Returns
-------
nb : NotebookNode
    The executed notebook.
resources : dictionary
    Additional resources used in the conversion process.

### Function: preprocess_cell(self, cell, resources, index)

**Description:** Override if you want to apply some preprocessing to each cell.
Must return modified cell and resource dictionary.

Parameters
----------
cell : NotebookNode cell
    Notebook cell being processed
resources : dictionary
    Additional resources used in the conversion process.  Allows
    preprocessors to pass variables into the Jinja engine.
index : int
    Index of the cell being processed
