## AI Summary

A file named numerictypes.py.


### Function: maximum_sctype(t)

**Description:** Return the scalar type of highest precision of the same kind as the input.

.. deprecated:: 2.0
    Use an explicit dtype like int64 or float64 instead.

Parameters
----------
t : dtype or dtype specifier
    The input data type. This can be a `dtype` object or an object that
    is convertible to a `dtype`.

Returns
-------
out : dtype
    The highest precision data type of the same kind (`dtype.kind`) as `t`.

See Also
--------
obj2sctype, mintypecode, sctype2char
dtype

Examples
--------
>>> from numpy._core.numerictypes import maximum_sctype
>>> maximum_sctype(int)
<class 'numpy.int64'>
>>> maximum_sctype(np.uint8)
<class 'numpy.uint64'>
>>> maximum_sctype(complex)
<class 'numpy.complex256'> # may vary

>>> maximum_sctype(str)
<class 'numpy.str_'>

>>> maximum_sctype('i2')
<class 'numpy.int64'>
>>> maximum_sctype('f4')
<class 'numpy.float128'> # may vary

### Function: issctype(rep)

**Description:** Determines whether the given object represents a scalar data-type.

Parameters
----------
rep : any
    If `rep` is an instance of a scalar dtype, True is returned. If not,
    False is returned.

Returns
-------
out : bool
    Boolean result of check whether `rep` is a scalar dtype.

See Also
--------
issubsctype, issubdtype, obj2sctype, sctype2char

Examples
--------
>>> from numpy._core.numerictypes import issctype
>>> issctype(np.int32)
True
>>> issctype(list)
False
>>> issctype(1.1)
False

Strings are also a scalar type:

>>> issctype(np.dtype('str'))
True

### Function: obj2sctype(rep, default)

**Description:** Return the scalar dtype or NumPy equivalent of Python type of an object.

Parameters
----------
rep : any
    The object of which the type is returned.
default : any, optional
    If given, this is returned for objects whose types can not be
    determined. If not given, None is returned for those objects.

Returns
-------
dtype : dtype or Python type
    The data type of `rep`.

See Also
--------
sctype2char, issctype, issubsctype, issubdtype

Examples
--------
>>> from numpy._core.numerictypes import obj2sctype
>>> obj2sctype(np.int32)
<class 'numpy.int32'>
>>> obj2sctype(np.array([1., 2.]))
<class 'numpy.float64'>
>>> obj2sctype(np.array([1.j]))
<class 'numpy.complex128'>

>>> obj2sctype(dict)
<class 'numpy.object_'>
>>> obj2sctype('string')

>>> obj2sctype(1, default=list)
<class 'list'>

### Function: issubclass_(arg1, arg2)

**Description:** Determine if a class is a subclass of a second class.

`issubclass_` is equivalent to the Python built-in ``issubclass``,
except that it returns False instead of raising a TypeError if one
of the arguments is not a class.

Parameters
----------
arg1 : class
    Input class. True is returned if `arg1` is a subclass of `arg2`.
arg2 : class or tuple of classes.
    Input class. If a tuple of classes, True is returned if `arg1` is a
    subclass of any of the tuple elements.

Returns
-------
out : bool
    Whether `arg1` is a subclass of `arg2` or not.

See Also
--------
issubsctype, issubdtype, issctype

Examples
--------
>>> np.issubclass_(np.int32, int)
False
>>> np.issubclass_(np.int32, float)
False
>>> np.issubclass_(np.float64, float)
True

### Function: issubsctype(arg1, arg2)

**Description:** Determine if the first argument is a subclass of the second argument.

Parameters
----------
arg1, arg2 : dtype or dtype specifier
    Data-types.

Returns
-------
out : bool
    The result.

See Also
--------
issctype, issubdtype, obj2sctype

Examples
--------
>>> from numpy._core import issubsctype
>>> issubsctype('S8', str)
False
>>> issubsctype(np.array([1]), int)
True
>>> issubsctype(np.array([1]), float)
False

## Class: _PreprocessDTypeError

### Function: _preprocess_dtype(dtype)

**Description:** Preprocess dtype argument by:
  1. fetching type from a data type
  2. verifying that types are built-in NumPy dtypes

### Function: isdtype(dtype, kind)

**Description:** Determine if a provided dtype is of a specified data type ``kind``.

This function only supports built-in NumPy's data types.
Third-party dtypes are not yet supported.

Parameters
----------
dtype : dtype
    The input dtype.
kind : dtype or str or tuple of dtypes/strs.
    dtype or dtype kind. Allowed dtype kinds are:
    * ``'bool'`` : boolean kind
    * ``'signed integer'`` : signed integer data types
    * ``'unsigned integer'`` : unsigned integer data types
    * ``'integral'`` : integer data types
    * ``'real floating'`` : real-valued floating-point data types
    * ``'complex floating'`` : complex floating-point data types
    * ``'numeric'`` : numeric data types

Returns
-------
out : bool

See Also
--------
issubdtype

Examples
--------
>>> import numpy as np
>>> np.isdtype(np.float32, np.float64)
False
>>> np.isdtype(np.float32, "real floating")
True
>>> np.isdtype(np.complex128, ("real floating", "complex floating"))
True

### Function: issubdtype(arg1, arg2)

**Description:** Returns True if first argument is a typecode lower/equal in type hierarchy.

This is like the builtin :func:`issubclass`, but for `dtype`\ s.

Parameters
----------
arg1, arg2 : dtype_like
    `dtype` or object coercible to one

Returns
-------
out : bool

See Also
--------
:ref:`arrays.scalars` : Overview of the numpy type hierarchy.

Examples
--------
`issubdtype` can be used to check the type of arrays:

>>> ints = np.array([1, 2, 3], dtype=np.int32)
>>> np.issubdtype(ints.dtype, np.integer)
True
>>> np.issubdtype(ints.dtype, np.floating)
False

>>> floats = np.array([1, 2, 3], dtype=np.float32)
>>> np.issubdtype(floats.dtype, np.integer)
False
>>> np.issubdtype(floats.dtype, np.floating)
True

Similar types of different sizes are not subdtypes of each other:

>>> np.issubdtype(np.float64, np.float32)
False
>>> np.issubdtype(np.float32, np.float64)
False

but both are subtypes of `floating`:

>>> np.issubdtype(np.float64, np.floating)
True
>>> np.issubdtype(np.float32, np.floating)
True

For convenience, dtype-like objects are allowed too:

>>> np.issubdtype('S1', np.bytes_)
True
>>> np.issubdtype('i4', np.signedinteger)
True

### Function: sctype2char(sctype)

**Description:** Return the string representation of a scalar dtype.

Parameters
----------
sctype : scalar dtype or object
    If a scalar dtype, the corresponding string character is
    returned. If an object, `sctype2char` tries to infer its scalar type
    and then return the corresponding string character.

Returns
-------
typechar : str
    The string character corresponding to the scalar type.

Raises
------
ValueError
    If `sctype` is an object for which the type can not be inferred.

See Also
--------
obj2sctype, issctype, issubsctype, mintypecode

Examples
--------
>>> from numpy._core.numerictypes import sctype2char
>>> for sctype in [np.int32, np.double, np.cdouble, np.bytes_, np.ndarray]:
...     print(sctype2char(sctype))
l # may vary
d
D
S
O

>>> x = np.array([1., 2-1.j])
>>> sctype2char(x)
'D'
>>> sctype2char(list)
'O'

### Function: _scalar_type_key(typ)

**Description:** A ``key`` function for `sorted`.

### Function: _register_types()
