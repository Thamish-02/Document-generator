## AI Summary

A file named _tempfile.py.


## Class: TemporaryFile

**Description:** An asynchronous temporary file that is automatically created and cleaned up.

This class provides an asynchronous context manager interface to a temporary file.
The file is created using Python's standard `tempfile.TemporaryFile` function in a
background thread, and is wrapped as an asynchronous file using `AsyncFile`.

:param mode: The mode in which the file is opened. Defaults to "w+b".
:param buffering: The buffering policy (-1 means the default buffering).
:param encoding: The encoding used to decode or encode the file. Only applicable in
    text mode.
:param newline: Controls how universal newlines mode works (only applicable in text
    mode).
:param suffix: The suffix for the temporary file name.
:param prefix: The prefix for the temporary file name.
:param dir: The directory in which the temporary file is created.
:param errors: The error handling scheme used for encoding/decoding errors.

## Class: NamedTemporaryFile

**Description:** An asynchronous named temporary file that is automatically created and cleaned up.

This class provides an asynchronous context manager for a temporary file with a
visible name in the file system. It uses Python's standard
:func:`~tempfile.NamedTemporaryFile` function and wraps the file object with
:class:`AsyncFile` for asynchronous operations.

:param mode: The mode in which the file is opened. Defaults to "w+b".
:param buffering: The buffering policy (-1 means the default buffering).
:param encoding: The encoding used to decode or encode the file. Only applicable in
    text mode.
:param newline: Controls how universal newlines mode works (only applicable in text
    mode).
:param suffix: The suffix for the temporary file name.
:param prefix: The prefix for the temporary file name.
:param dir: The directory in which the temporary file is created.
:param delete: Whether to delete the file when it is closed.
:param errors: The error handling scheme used for encoding/decoding errors.
:param delete_on_close: (Python 3.12+) Whether to delete the file on close.

## Class: SpooledTemporaryFile

**Description:** An asynchronous spooled temporary file that starts in memory and is spooled to disk.

This class provides an asynchronous interface to a spooled temporary file, much like
Python's standard :class:`~tempfile.SpooledTemporaryFile`. It supports asynchronous
write operations and provides a method to force a rollover to disk.

:param max_size: Maximum size in bytes before the file is rolled over to disk.
:param mode: The mode in which the file is opened. Defaults to "w+b".
:param buffering: The buffering policy (-1 means the default buffering).
:param encoding: The encoding used to decode or encode the file (text mode only).
:param newline: Controls how universal newlines mode works (text mode only).
:param suffix: The suffix for the temporary file name.
:param prefix: The prefix for the temporary file name.
:param dir: The directory in which the temporary file is created.
:param errors: The error handling scheme used for encoding/decoding errors.

## Class: TemporaryDirectory

**Description:** An asynchronous temporary directory that is created and cleaned up automatically.

This class provides an asynchronous context manager for creating a temporary
directory. It wraps Python's standard :class:`~tempfile.TemporaryDirectory` to
perform directory creation and cleanup operations in a background thread.

:param suffix: Suffix to be added to the temporary directory name.
:param prefix: Prefix to be added to the temporary directory name.
:param dir: The parent directory where the temporary directory is created.
:param ignore_cleanup_errors: Whether to ignore errors during cleanup
    (Python 3.10+).
:param delete: Whether to delete the directory upon closing (Python 3.12+).

### Function: __init__(self, mode, buffering, encoding, newline, suffix, prefix, dir)

### Function: __init__(self, mode, buffering, encoding, newline, suffix, prefix, dir)

### Function: __init__(self, mode, buffering, encoding, newline, suffix, prefix, dir)

### Function: __init__(self, mode, buffering, encoding, newline, suffix, prefix, dir, delete)

### Function: __init__(self, mode, buffering, encoding, newline, suffix, prefix, dir, delete)

### Function: __init__(self, mode, buffering, encoding, newline, suffix, prefix, dir, delete)

### Function: __init__(self, max_size, mode, buffering, encoding, newline, suffix, prefix, dir)

### Function: __init__(self, max_size, mode, buffering, encoding, newline, suffix, prefix, dir)

### Function: __init__(self, max_size, mode, buffering, encoding, newline, suffix, prefix, dir)

### Function: closed(self)

### Function: __init__(self, suffix, prefix, dir)
