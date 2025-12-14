## AI Summary

A file named _utilities.py.


## Class: Symbol

**Description:** A constant symbol, nicer than ``object()``. Repeated calls return the
same instance.

>>> Symbol('foo') is Symbol('foo')
True
>>> Symbol('foo')
foo

### Function: make_id(obj)

**Description:** Get a stable identifier for a receiver or sender, to be used as a dict
key or in a set.

### Function: make_ref(obj, callback)

### Function: __new__(cls, name)

### Function: __init__(self, name)

### Function: __repr__(self)

### Function: __getnewargs__(self)
