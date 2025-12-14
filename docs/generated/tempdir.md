## AI Summary

A file named tempdir.py.


## Class: NamedFileInTemporaryDirectory

## Class: TemporaryWorkingDirectory

**Description:** Creates a temporary directory and sets the cwd to that directory.
Automatically reverts to previous cwd upon cleanup.
Usage example:

    with TemporaryWorkingDirectory() as tmpdir:
        ...

### Function: __init__(self, filename, mode, bufsize, add_to_syspath)

**Description:** Open a file named `filename` in a temporary directory.

This context manager is preferred over `NamedTemporaryFile` in
stdlib `tempfile` when one needs to reopen the file.

Arguments `mode` and `bufsize` are passed to `open`.
Rest of the arguments are passed to `TemporaryDirectory`.

### Function: cleanup(self)

### Function: __enter__(self)

### Function: __exit__(self, type, value, traceback)

### Function: __enter__(self)

### Function: __exit__(self, exc, value, tb)
