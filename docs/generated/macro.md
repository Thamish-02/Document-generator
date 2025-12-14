## AI Summary

A file named macro.py.


## Class: Macro

**Description:** Simple class to store the value of macros as strings.

Macro is just a callable that executes a string of IPython
input when called.

### Function: __init__(self, code)

**Description:** store the macro value, as a single string which can be executed

### Function: __str__(self)

### Function: __repr__(self)

### Function: __getstate__(self)

**Description:** needed for safe pickling via %store 

### Function: __add__(self, other)
