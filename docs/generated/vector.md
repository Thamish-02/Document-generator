## AI Summary

A file named vector.py.


## Class: Vector

**Description:** A math-like vector.

Represents an n-dimensional numeric vector. ``Vector`` objects support
vector addition and subtraction, scalar multiplication and division,
negation, rounding, and comparison tests.

### Function: _operator_rsub(a, b)

### Function: _operator_rtruediv(a, b)

### Function: __new__(cls, values, keep)

### Function: __repr__(self)

### Function: _vectorOp(self, other, op)

### Function: _scalarOp(self, other, op)

### Function: _unaryOp(self, op)

### Function: __add__(self, other)

### Function: __sub__(self, other)

### Function: __rsub__(self, other)

### Function: __mul__(self, other)

### Function: __truediv__(self, other)

### Function: __rtruediv__(self, other)

### Function: __pos__(self)

### Function: __neg__(self)

### Function: __round__(self)

### Function: __eq__(self, other)

### Function: __ne__(self, other)

### Function: __bool__(self)

### Function: __abs__(self)

### Function: length(self)

**Description:** Return the length of the vector. Equivalent to abs(vector).

### Function: normalized(self)

**Description:** Return the normalized vector of the vector.

### Function: dot(self, other)

**Description:** Performs vector dot product, returning the sum of
``a[0] * b[0], a[1] * b[1], ...``

### Function: toInt(self)

### Function: values(self)

### Function: values(self, values)

### Function: isclose(self, other)

**Description:** Return True if the vector is close to another Vector.
