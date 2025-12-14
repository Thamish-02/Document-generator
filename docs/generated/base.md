## AI Summary

A file named base.py.


## Class: WriterBase

**Description:** Consumes output from nbconvert export...() methods and writes to a
useful location.

### Function: __init__(self, config)

**Description:** Constructor

### Function: write(self, output, resources)

**Description:** Consume and write Jinja output.

Parameters
----------
output : string
    Conversion results.  This string contains the file contents of the
    converted file.
resources : dict
    Resources created and filled by the nbconvert conversion process.
    Includes output from preprocessors, such as the extract figure
    preprocessor.
