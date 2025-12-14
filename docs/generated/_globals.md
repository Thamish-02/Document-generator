## AI Summary

A file named _globals.py.


## Class: _NoValueType

**Description:** Special keyword value.

The instance of this class may be used as the default value assigned to a
keyword if no other obvious default (e.g., `None`) is suitable,

Common reasons for using this keyword are:

- A new keyword is added to a function, and that function forwards its
  inputs to another function or method which can be defined outside of
  NumPy. For example, ``np.std(x)`` calls ``x.std``, so when a ``keepdims``
  keyword was added that could only be forwarded if the user explicitly
  specified ``keepdims``; downstream array libraries may not have added
  the same keyword, so adding ``x.std(..., keepdims=keepdims)``
  unconditionally could have broken previously working code.
- A keyword is being deprecated, and a deprecation warning must only be
  emitted when the keyword is used.

## Class: _CopyMode

**Description:** An enumeration for the copy modes supported
by numpy.copy() and numpy.array(). The following three modes are supported,

- ALWAYS: This means that a deep copy of the input
          array will always be taken.
- IF_NEEDED: This means that a deep copy of the input
             array will be taken only if necessary.
- NEVER: This means that the deep copy will never be taken.
         If a copy cannot be avoided then a `ValueError` will be
         raised.

Note that the buffer-protocol could in theory do copies.  NumPy currently
assumes an object exporting the buffer protocol will never do this.

### Function: __new__(cls)

### Function: __repr__(self)

### Function: __bool__(self)
