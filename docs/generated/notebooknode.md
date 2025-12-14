## AI Summary

A file named notebooknode.py.


## Class: NotebookNode

**Description:** A dict-like node with attribute-access

### Function: from_dict(d)

**Description:** Convert dict to dict-like NotebookNode

Recursively converts any dict in the container to a NotebookNode.
This does not check that the contents of the dictionary make a valid
notebook or part of a notebook.

### Function: __setitem__(self, key, value)

**Description:** Set an item on the notebook.

### Function: update(self)

**Description:** A dict-like update method based on CPython's MutableMapping `update`
method.
