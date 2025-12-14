## AI Summary

A file named fileio.py.


### Function: replace_file(src, dst)

**Description:** replace dst with src

### Function: copy2_safe(src, dst, log)

**Description:** copy src to dst

like shutil.copy2, but log errors in copystat instead of raising

### Function: path_to_intermediate(path)

**Description:** Name of the intermediate file used in atomic writes.

The .~ prefix will make Dropbox ignore the temporary file.

### Function: path_to_invalid(path)

**Description:** Name of invalid file after a failed atomic write and subsequent read.

### Function: atomic_writing(path, text, encoding, log)

**Description:** Context manager to write to a file only if the entire write is successful.

This works by copying the previous file contents to a temporary file in the
same directory, and renaming that file back to the target if the context
exits with an error. If the context is successful, the new data is synced to
disk and the temporary file is removed.

Parameters
----------
path : str
    The target file to write to.
text : bool, optional
    Whether to open the file in text mode (i.e. to write unicode). Default is
    True.
encoding : str, optional
    The encoding to use for files opened in text mode. Default is UTF-8.
**kwargs
    Passed to :func:`io.open`.

### Function: _simple_writing(path, text, encoding, log)

**Description:** Context manager to write file without doing atomic writing
(for weird filesystem eg: nfs).

Parameters
----------
path : str
    The target file to write to.
text : bool, optional
    Whether to open the file in text mode (i.e. to write unicode). Default is
    True.
encoding : str, optional
    The encoding to use for files opened in text mode. Default is UTF-8.
**kwargs
    Passed to :func:`io.open`.

## Class: FileManagerMixin

**Description:** Mixin for ContentsAPI classes that interact with the filesystem.

Provides facilities for reading, writing, and copying files.

Shared by FileContentsManager and FileCheckpoints.

Note
----
Classes using this mixin must provide the following attributes:

root_dir : unicode
    A directory against against which API-style paths are to be resolved.

log : logging.Logger

## Class: AsyncFileManagerMixin

**Description:** Mixin for ContentsAPI classes that interact with the filesystem asynchronously.

### Function: open(self, os_path)

**Description:** wrapper around io.open that turns permission errors into 403

### Function: atomic_writing(self, os_path)

**Description:** wrapper around atomic_writing that turns permission errors to 403.
Depending on flag 'use_atomic_writing', the wrapper perform an actual atomic writing or
simply writes the file (whatever an old exists or not)

### Function: perm_to_403(self, os_path)

**Description:** context manager for turning permission errors into 403.

### Function: _copy(self, src, dest)

**Description:** copy src to dest

like shutil.copy2, but log errors in copystat

### Function: _get_os_path(self, path)

**Description:** Given an API path, return its file system path.

Parameters
----------
path : str
    The relative API path to the named file.

Returns
-------
path : str
    Native, absolute OS path to for a file.

Raises
------
404: if path is outside root

### Function: _read_notebook(self, os_path, as_version, capture_validation_error, raw)

**Description:** Read a notebook from an os path.

### Function: _save_notebook(self, os_path, nb, capture_validation_error)

**Description:** Save a notebook to an os_path.

### Function: _get_hash(self, byte_content)

**Description:** Compute the hash hexdigest for the provided bytes.

The hash algorithm is provided by the `hash_algorithm` attribute.

Parameters
----------
byte_content : bytes
    The bytes to hash

Returns
-------
A dictionary to be appended to a model {"hash": str, "hash_algorithm": str}.

### Function: _read_file(self, os_path, format, raw)

**Description:** Read a non-notebook file.

Parameters
----------
os_path: str
    The path to be read.
format: str
    If 'text', the contents will be decoded as UTF-8.
    If 'base64', the raw bytes contents will be encoded as base64.
    If 'byte', the raw bytes contents will be returned.
    If not specified, try to decode as UTF-8, and fall back to base64
raw: bool
    [Optional] If True, will return as third argument the raw bytes content

Returns
-------
(content, format, byte_content) It returns the content in the given format
as well as the raw byte content.

### Function: _save_file(self, os_path, content, format)

**Description:** Save content of a generic file.
