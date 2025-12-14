## AI Summary

A file named _version_info.py.


## Class: VersionInfo

**Description:** A version object that can be compared to tuple of length 1--4:

>>> attr.VersionInfo(19, 1, 0, "final")  <= (19, 2)
True
>>> attr.VersionInfo(19, 1, 0, "final") < (19, 1, 1)
True
>>> vi = attr.VersionInfo(19, 2, 0, "final")
>>> vi < (19, 1, 1)
False
>>> vi < (19,)
False
>>> vi == (19, 2,)
True
>>> vi == (19, 2, 1)
False

.. versionadded:: 19.2

### Function: _from_version_string(cls, s)

**Description:** Parse *s* and return a _VersionInfo.

### Function: _ensure_tuple(self, other)

**Description:** Ensure *other* is a tuple of a valid length.

Returns a possibly transformed *other* and ourselves as a tuple of
the same length as *other*.

### Function: __eq__(self, other)

### Function: __lt__(self, other)

### Function: __hash__(self)
