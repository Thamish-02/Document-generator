## AI Summary

A file named capture.py.


## Class: RichOutput

## Class: CapturedIO

**Description:** Simple object for containing captured stdout/err and rich display StringIO objects

Each instance `c` has three attributes:

- ``c.stdout`` : standard output as a string
- ``c.stderr`` : standard error as a string
- ``c.outputs``: a list of rich display outputs

Additionally, there's a ``c.show()`` method which will print all of the
above in the same order, and can be invoked simply via ``c()``.

## Class: capture_output

**Description:** context manager for capturing stdout/err

### Function: __init__(self, data, metadata, transient, update)

### Function: display(self)

### Function: _repr_mime_(self, mime)

### Function: _repr_mimebundle_(self, include, exclude)

### Function: _repr_html_(self)

### Function: _repr_latex_(self)

### Function: _repr_json_(self)

### Function: _repr_javascript_(self)

### Function: _repr_png_(self)

### Function: _repr_jpeg_(self)

### Function: _repr_svg_(self)

### Function: __init__(self, stdout, stderr, outputs)

### Function: __str__(self)

### Function: stdout(self)

**Description:** Captured standard output

### Function: stderr(self)

**Description:** Captured standard error

### Function: outputs(self)

**Description:** A list of the captured rich display outputs, if any.

If you have a CapturedIO object ``c``, these can be displayed in IPython
using::

    from IPython.display import display
    for o in c.outputs:
        display(o)

### Function: show(self)

**Description:** write my output to sys.stdout/err as appropriate

### Function: __init__(self, stdout, stderr, display)

### Function: __enter__(self)

### Function: __exit__(self, exc_type, exc_value, traceback)
