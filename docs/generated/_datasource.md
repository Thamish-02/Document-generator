## AI Summary

A file named _datasource.py.


### Function: _check_mode(mode, encoding, newline)

**Description:** Check mode and that encoding and newline are compatible.

Parameters
----------
mode : str
    File open mode.
encoding : str
    File encoding.
newline : str
    Newline for text files.

## Class: _FileOpeners

**Description:** Container for different methods to open (un-)compressed files.

`_FileOpeners` contains a dictionary that holds one method for each
supported file format. Attribute lookup is implemented in such a way
that an instance of `_FileOpeners` itself can be indexed with the keys
of that dictionary. Currently uncompressed files as well as files
compressed with ``gzip``, ``bz2`` or ``xz`` compression are supported.

Notes
-----
`_file_openers`, an instance of `_FileOpeners`, is made available for
use in the `_datasource` module.

Examples
--------
>>> import gzip
>>> np.lib._datasource._file_openers.keys()
[None, '.bz2', '.gz', '.xz', '.lzma']
>>> np.lib._datasource._file_openers['.gz'] is gzip.open
True

### Function: open(path, mode, destpath, encoding, newline)

**Description:** Open `path` with `mode` and return the file object.

If ``path`` is an URL, it will be downloaded, stored in the
`DataSource` `destpath` directory and opened from there.

Parameters
----------
path : str or pathlib.Path
    Local file path or URL to open.
mode : str, optional
    Mode to open `path`. Mode 'r' for reading, 'w' for writing, 'a' to
    append. Available modes depend on the type of object specified by
    path.  Default is 'r'.
destpath : str, optional
    Path to the directory where the source file gets downloaded to for
    use.  If `destpath` is None, a temporary directory will be created.
    The default path is the current directory.
encoding : {None, str}, optional
    Open text file with given encoding. The default encoding will be
    what `open` uses.
newline : {None, str}, optional
    Newline to use when reading text file.

Returns
-------
out : file object
    The opened file.

Notes
-----
This is a convenience function that instantiates a `DataSource` and
returns the file object from ``DataSource.open(path)``.

## Class: DataSource

**Description:** DataSource(destpath='.')

A generic data source file (file, http, ftp, ...).

DataSources can be local files or remote files/URLs.  The files may
also be compressed or uncompressed. DataSource hides some of the
low-level details of downloading the file, allowing you to simply pass
in a valid file path (or URL) and obtain a file object.

Parameters
----------
destpath : str or None, optional
    Path to the directory where the source file gets downloaded to for
    use.  If `destpath` is None, a temporary directory will be created.
    The default path is the current directory.

Notes
-----
URLs require a scheme string (``http://``) to be used, without it they
will fail::

    >>> repos = np.lib.npyio.DataSource()
    >>> repos.exists('www.google.com/index.html')
    False
    >>> repos.exists('http://www.google.com/index.html')
    True

Temporary directories are deleted when the DataSource is deleted.

Examples
--------
::

    >>> ds = np.lib.npyio.DataSource('/home/guido')
    >>> urlname = 'http://www.google.com/'
    >>> gfile = ds.open('http://www.google.com/')
    >>> ds.abspath(urlname)
    '/home/guido/www.google.com/index.html'

    >>> ds = np.lib.npyio.DataSource(None)  # use with temporary file
    >>> ds.open('/home/guido/foobar.txt')
    <open file '/home/guido.foobar.txt', mode 'r' at 0x91d4430>
    >>> ds.abspath('/home/guido/foobar.txt')
    '/tmp/.../home/guido/foobar.txt'

## Class: Repository

**Description:** Repository(baseurl, destpath='.')

A data repository where multiple DataSource's share a base
URL/directory.

`Repository` extends `DataSource` by prepending a base URL (or
directory) to all the files it handles. Use `Repository` when you will
be working with multiple files from one base URL.  Initialize
`Repository` with the base URL, then refer to each file by its filename
only.

Parameters
----------
baseurl : str
    Path to the local directory or remote location that contains the
    data files.
destpath : str or None, optional
    Path to the directory where the source file gets downloaded to for
    use.  If `destpath` is None, a temporary directory will be created.
    The default path is the current directory.

Examples
--------
To analyze all files in the repository, do something like this
(note: this is not self-contained code)::

    >>> repos = np.lib._datasource.Repository('/home/user/data/dir/')
    >>> for filename in filelist:
    ...     fp = repos.open(filename)
    ...     fp.analyze()
    ...     fp.close()

Similarly you could use a URL for a repository::

    >>> repos = np.lib._datasource.Repository('http://www.xyz.edu/data')

### Function: __init__(self)

### Function: _load(self)

### Function: keys(self)

**Description:** Return the keys of currently supported file openers.

Parameters
----------
None

Returns
-------
keys : list
    The keys are None for uncompressed files and the file extension
    strings (i.e. ``'.gz'``, ``'.xz'``) for supported compression
    methods.

### Function: __getitem__(self, key)

### Function: __init__(self, destpath)

**Description:** Create a DataSource with a local path at destpath.

### Function: __del__(self)

### Function: _iszip(self, filename)

**Description:** Test if the filename is a zip file by looking at the file extension.

        

### Function: _iswritemode(self, mode)

**Description:** Test if the given mode will open a file for writing.

### Function: _splitzipext(self, filename)

**Description:** Split zip extension from filename and return filename.

Returns
-------
base, zip_ext : {tuple}

### Function: _possible_names(self, filename)

**Description:** Return a tuple containing compressed filename variations.

### Function: _isurl(self, path)

**Description:** Test if path is a net location.  Tests the scheme and netloc.

### Function: _cache(self, path)

**Description:** Cache the file specified by path.

Creates a copy of the file in the datasource cache.

### Function: _findfile(self, path)

**Description:** Searches for ``path`` and returns full path if found.

If path is an URL, _findfile will cache a local copy and return the
path to the cached file.  If path is a local file, _findfile will
return a path to that local file.

The search will include possible compressed versions of the file
and return the first occurrence found.

### Function: abspath(self, path)

**Description:** Return absolute path of file in the DataSource directory.

If `path` is an URL, then `abspath` will return either the location
the file exists locally or the location it would exist when opened
using the `open` method.

Parameters
----------
path : str or pathlib.Path
    Can be a local file or a remote URL.

Returns
-------
out : str
    Complete path, including the `DataSource` destination directory.

Notes
-----
The functionality is based on `os.path.abspath`.

### Function: _sanitize_relative_path(self, path)

**Description:** Return a sanitised relative path for which
os.path.abspath(os.path.join(base, path)).startswith(base)

### Function: exists(self, path)

**Description:** Test if path exists.

Test if `path` exists as (and in this order):

- a local file.
- a remote URL that has been downloaded and stored locally in the
  `DataSource` directory.
- a remote URL that has not been downloaded, but is valid and
  accessible.

Parameters
----------
path : str or pathlib.Path
    Can be a local file or a remote URL.

Returns
-------
out : bool
    True if `path` exists.

Notes
-----
When `path` is an URL, `exists` will return True if it's either
stored locally in the `DataSource` directory, or is a valid remote
URL.  `DataSource` does not discriminate between the two, the file
is accessible if it exists in either location.

### Function: open(self, path, mode, encoding, newline)

**Description:** Open and return file-like object.

If `path` is an URL, it will be downloaded, stored in the
`DataSource` directory and opened from there.

Parameters
----------
path : str or pathlib.Path
    Local file path or URL to open.
mode : {'r', 'w', 'a'}, optional
    Mode to open `path`.  Mode 'r' for reading, 'w' for writing,
    'a' to append. Available modes depend on the type of object
    specified by `path`. Default is 'r'.
encoding : {None, str}, optional
    Open text file with given encoding. The default encoding will be
    what `open` uses.
newline : {None, str}, optional
    Newline to use when reading text file.

Returns
-------
out : file object
    File object.

### Function: __init__(self, baseurl, destpath)

**Description:** Create a Repository with a shared url or directory of baseurl.

### Function: __del__(self)

### Function: _fullpath(self, path)

**Description:** Return complete path for path.  Prepends baseurl if necessary.

### Function: _findfile(self, path)

**Description:** Extend DataSource method to prepend baseurl to ``path``.

### Function: abspath(self, path)

**Description:** Return absolute path of file in the Repository directory.

If `path` is an URL, then `abspath` will return either the location
the file exists locally or the location it would exist when opened
using the `open` method.

Parameters
----------
path : str or pathlib.Path
    Can be a local file or a remote URL. This may, but does not
    have to, include the `baseurl` with which the `Repository` was
    initialized.

Returns
-------
out : str
    Complete path, including the `DataSource` destination directory.

### Function: exists(self, path)

**Description:** Test if path exists prepending Repository base URL to path.

Test if `path` exists as (and in this order):

- a local file.
- a remote URL that has been downloaded and stored locally in the
  `DataSource` directory.
- a remote URL that has not been downloaded, but is valid and
  accessible.

Parameters
----------
path : str or pathlib.Path
    Can be a local file or a remote URL. This may, but does not
    have to, include the `baseurl` with which the `Repository` was
    initialized.

Returns
-------
out : bool
    True if `path` exists.

Notes
-----
When `path` is an URL, `exists` will return True if it's either
stored locally in the `DataSource` directory, or is a valid remote
URL.  `DataSource` does not discriminate between the two, the file
is accessible if it exists in either location.

### Function: open(self, path, mode, encoding, newline)

**Description:** Open and return file-like object prepending Repository base URL.

If `path` is an URL, it will be downloaded, stored in the
DataSource directory and opened from there.

Parameters
----------
path : str or pathlib.Path
    Local file path or URL to open. This may, but does not have to,
    include the `baseurl` with which the `Repository` was
    initialized.
mode : {'r', 'w', 'a'}, optional
    Mode to open `path`.  Mode 'r' for reading, 'w' for writing,
    'a' to append. Available modes depend on the type of object
    specified by `path`. Default is 'r'.
encoding : {None, str}, optional
    Open text file with given encoding. The default encoding will be
    what `open` uses.
newline : {None, str}, optional
    Newline to use when reading text file.

Returns
-------
out : file object
    File object.

### Function: listdir(self)

**Description:** List files in the source Repository.

Returns
-------
files : list of str or pathlib.Path
    List of file names (not containing a directory part).

Notes
-----
Does not currently work for remote repositories.
