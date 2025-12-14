## AI Summary

A file named _fileio.py.


## Class: AsyncFile

**Description:** An asynchronous file object.

This class wraps a standard file object and provides async friendly versions of the
following blocking methods (where available on the original file object):

* read
* read1
* readline
* readlines
* readinto
* readinto1
* write
* writelines
* truncate
* seek
* tell
* flush

All other methods are directly passed through.

This class supports the asynchronous context manager protocol which closes the
underlying file at the end of the context block.

This class also supports asynchronous iteration::

    async with await open_file(...) as f:
        async for line in f:
            print(line)

### Function: wrap_file(file)

**Description:** Wrap an existing file as an asynchronous file.

:param file: an existing file-like object
:return: an asynchronous file object

## Class: _PathIterator

## Class: Path

**Description:** An asynchronous version of :class:`pathlib.Path`.

This class cannot be substituted for :class:`pathlib.Path` or
:class:`pathlib.PurePath`, but it is compatible with the :class:`os.PathLike`
interface.

It implements the Python 3.10 version of :class:`pathlib.Path` interface, except for
the deprecated :meth:`~pathlib.Path.link_to` method.

Some methods may be unavailable or have limited functionality, based on the Python
version:

* :meth:`~pathlib.Path.copy` (available on Python 3.14 or later)
* :meth:`~pathlib.Path.copy_into` (available on Python 3.14 or later)
* :meth:`~pathlib.Path.from_uri` (available on Python 3.13 or later)
* :meth:`~pathlib.PurePath.full_match` (available on Python 3.13 or later)
* :attr:`~pathlib.Path.info` (available on Python 3.14 or later)
* :meth:`~pathlib.Path.is_junction` (available on Python 3.12 or later)
* :meth:`~pathlib.PurePath.match` (the ``case_sensitive`` parameter is only
  available on Python 3.13 or later)
* :meth:`~pathlib.Path.move` (available on Python 3.14 or later)
* :meth:`~pathlib.Path.move_into` (available on Python 3.14 or later)
* :meth:`~pathlib.PurePath.relative_to` (the ``walk_up`` parameter is only available
  on Python 3.12 or later)
* :meth:`~pathlib.Path.walk` (available on Python 3.12 or later)

Any methods that do disk I/O need to be awaited on. These methods are:

* :meth:`~pathlib.Path.absolute`
* :meth:`~pathlib.Path.chmod`
* :meth:`~pathlib.Path.cwd`
* :meth:`~pathlib.Path.exists`
* :meth:`~pathlib.Path.expanduser`
* :meth:`~pathlib.Path.group`
* :meth:`~pathlib.Path.hardlink_to`
* :meth:`~pathlib.Path.home`
* :meth:`~pathlib.Path.is_block_device`
* :meth:`~pathlib.Path.is_char_device`
* :meth:`~pathlib.Path.is_dir`
* :meth:`~pathlib.Path.is_fifo`
* :meth:`~pathlib.Path.is_file`
* :meth:`~pathlib.Path.is_junction`
* :meth:`~pathlib.Path.is_mount`
* :meth:`~pathlib.Path.is_socket`
* :meth:`~pathlib.Path.is_symlink`
* :meth:`~pathlib.Path.lchmod`
* :meth:`~pathlib.Path.lstat`
* :meth:`~pathlib.Path.mkdir`
* :meth:`~pathlib.Path.open`
* :meth:`~pathlib.Path.owner`
* :meth:`~pathlib.Path.read_bytes`
* :meth:`~pathlib.Path.read_text`
* :meth:`~pathlib.Path.readlink`
* :meth:`~pathlib.Path.rename`
* :meth:`~pathlib.Path.replace`
* :meth:`~pathlib.Path.resolve`
* :meth:`~pathlib.Path.rmdir`
* :meth:`~pathlib.Path.samefile`
* :meth:`~pathlib.Path.stat`
* :meth:`~pathlib.Path.symlink_to`
* :meth:`~pathlib.Path.touch`
* :meth:`~pathlib.Path.unlink`
* :meth:`~pathlib.Path.walk`
* :meth:`~pathlib.Path.write_bytes`
* :meth:`~pathlib.Path.write_text`

Additionally, the following methods return an async iterator yielding
:class:`~.Path` objects:

* :meth:`~pathlib.Path.glob`
* :meth:`~pathlib.Path.iterdir`
* :meth:`~pathlib.Path.rglob`

### Function: __init__(self, fp)

### Function: __getattr__(self, name)

### Function: wrapped(self)

**Description:** The wrapped file object.

### Function: __init__(self)

### Function: __fspath__(self)

### Function: __str__(self)

### Function: __repr__(self)

### Function: __bytes__(self)

### Function: __hash__(self)

### Function: __eq__(self, other)

### Function: __lt__(self, other)

### Function: __le__(self, other)

### Function: __gt__(self, other)

### Function: __ge__(self, other)

### Function: __truediv__(self, other)

### Function: __rtruediv__(self, other)

### Function: parts(self)

### Function: drive(self)

### Function: root(self)

### Function: anchor(self)

### Function: parents(self)

### Function: parent(self)

### Function: name(self)

### Function: suffix(self)

### Function: suffixes(self)

### Function: stem(self)

### Function: as_posix(self)

### Function: as_uri(self)

### Function: is_relative_to(self, other)

### Function: glob(self, pattern)

### Function: is_absolute(self)

### Function: is_reserved(self)

### Function: joinpath(self)

### Function: rglob(self, pattern)

### Function: with_name(self, name)

### Function: with_stem(self, stem)

### Function: with_suffix(self, suffix)

### Function: with_segments(self)

### Function: from_uri(cls, uri)

### Function: full_match(self, path_pattern)

### Function: match(self, path_pattern)

### Function: match(self, path_pattern)

### Function: info(self)

### Function: relative_to(self)

### Function: relative_to(self)

### Function: sync_write_text()

### Function: get_next_value()
