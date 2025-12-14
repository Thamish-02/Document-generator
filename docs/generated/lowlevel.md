## AI Summary

A file named lowlevel.py.


## Class: EventLoopToken

**Description:** An opaque object that holds a reference to an event loop.

.. versionadded:: 4.11.0

### Function: current_token()

**Description:** Return a token object that can be used to call code in the current event loop from
another thread.

.. versionadded:: 4.11.0

## Class: _NoValueSet

## Class: RunvarToken

## Class: RunVar

**Description:** Like a :class:`~contextvars.ContextVar`, except scoped to the running event loop.

### Function: __init__(self, var, value)

### Function: __init__(self, name, default)

### Function: _current_vars(self)

### Function: get(self, default)

### Function: get(self)

### Function: get(self, default)

### Function: set(self, value)

### Function: reset(self, token)

### Function: __repr__(self)
