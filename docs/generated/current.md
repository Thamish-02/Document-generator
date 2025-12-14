## AI Summary

A file named current.py.


## Class: NBFormatError

**Description:** An error raised for an nbformat error.

### Function: _warn_format()

### Function: parse_py(s)

**Description:** Parse a string into a (nbformat, string) tuple.

### Function: reads_json(nbjson)

**Description:** DEPRECATED, use reads

### Function: writes_json(nb)

**Description:** DEPRECATED, use writes

### Function: reads_py(s)

**Description:** DEPRECATED: use nbconvert

### Function: writes_py(nb)

**Description:** DEPRECATED: use nbconvert

### Function: reads(s, format, version)

**Description:** Read a notebook from a string and return the NotebookNode object.

This function properly handles notebooks of any version. The notebook
returned will always be in the current version's format.

Parameters
----------
s : unicode
    The raw unicode string to read the notebook from.

Returns
-------
nb : NotebookNode
    The notebook that was read.

### Function: writes(nb, format, version)

**Description:** Write a notebook to a string in a given format in the current nbformat version.

This function always writes the notebook in the current nbformat version.

Parameters
----------
nb : NotebookNode
    The notebook to write.
version : int
    The nbformat version to write.
    Used for downgrading notebooks.

Returns
-------
s : unicode
    The notebook string.

### Function: read(fp, format)

**Description:** Read a notebook from a file and return the NotebookNode object.

This function properly handles notebooks of any version. The notebook
returned will always be in the current version's format.

Parameters
----------
fp : file
    Any file-like object with a read method.

Returns
-------
nb : NotebookNode
    The notebook that was read.

### Function: write(nb, fp, format)

**Description:** Write a notebook to a file in a given format in the current nbformat version.

This function always writes the notebook in the current nbformat version.

Parameters
----------
nb : NotebookNode
    The notebook to write.
fp : file
    Any file-like object with a write method.
