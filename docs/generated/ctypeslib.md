## AI Summary

A file named ctypeslib.py.


### Function: _num_fromflags(flaglist)

### Function: _flags_fromnum(num)

## Class: _ndptr

## Class: _concrete_ndptr

**Description:** Like _ndptr, but with `_shape_` and `_dtype_` specified.

Notably, this means the pointer has enough information to reconstruct
the array, which is not generally true.

### Function: ndpointer(dtype, ndim, shape, flags)

**Description:** Array-checking restype/argtypes.

An ndpointer instance is used to describe an ndarray in restypes
and argtypes specifications.  This approach is more flexible than
using, for example, ``POINTER(c_double)``, since several restrictions
can be specified, which are verified upon calling the ctypes function.
These include data type, number of dimensions, shape and flags.  If a
given array does not satisfy the specified restrictions,
a ``TypeError`` is raised.

Parameters
----------
dtype : data-type, optional
    Array data-type.
ndim : int, optional
    Number of array dimensions.
shape : tuple of ints, optional
    Array shape.
flags : str or tuple of str
    Array flags; may be one or more of:

    - C_CONTIGUOUS / C / CONTIGUOUS
    - F_CONTIGUOUS / F / FORTRAN
    - OWNDATA / O
    - WRITEABLE / W
    - ALIGNED / A
    - WRITEBACKIFCOPY / X

Returns
-------
klass : ndpointer type object
    A type object, which is an ``_ndtpr`` instance containing
    dtype, ndim, shape and flags information.

Raises
------
TypeError
    If a given array does not satisfy the specified restrictions.

Examples
--------
>>> clib.somefunc.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64,
...                                                  ndim=1,
...                                                  flags='C_CONTIGUOUS')]
... #doctest: +SKIP
>>> clib.somefunc(np.array([1, 2, 3], dtype=np.float64))
... #doctest: +SKIP

### Function: _dummy()

**Description:** Dummy object that raises an ImportError if ctypes is not available.

Raises
------
ImportError
    If ctypes is not available.

### Function: load_library(libname, loader_path)

**Description:** It is possible to load a library using

>>> lib = ctypes.cdll[<full_path_name>] # doctest: +SKIP

But there are cross-platform considerations, such as library file extensions,
plus the fact Windows will just load the first library it finds with that name.
NumPy supplies the load_library function as a convenience.

.. versionchanged:: 1.20.0
    Allow libname and loader_path to take any
    :term:`python:path-like object`.

Parameters
----------
libname : path-like
    Name of the library, which can have 'lib' as a prefix,
    but without an extension.
loader_path : path-like
    Where the library can be found.

Returns
-------
ctypes.cdll[libpath] : library object
   A ctypes library object

Raises
------
OSError
    If there is no library with the expected extension, or the
    library is defective and cannot be loaded.

### Function: from_param(cls, obj)

### Function: _check_retval_(self)

**Description:** This method is called when this class is used as the .restype
attribute for a shared-library function, to automatically wrap the
pointer into an array.

### Function: contents(self)

**Description:** Get an ndarray viewing the data pointed to by this pointer.

This mirrors the `contents` attribute of a normal ctypes pointer

### Function: _ctype_ndarray(element_type, shape)

**Description:** Create an ndarray of the given element type and shape 

### Function: _get_scalar_type_map()

**Description:** Return a dictionary mapping native endian scalar dtype to ctypes types

### Function: _ctype_from_dtype_scalar(dtype)

### Function: _ctype_from_dtype_subarray(dtype)

### Function: _ctype_from_dtype_structured(dtype)

### Function: _ctype_from_dtype(dtype)

### Function: as_ctypes_type(dtype)

**Description:** Convert a dtype into a ctypes type.

Parameters
----------
dtype : dtype
    The dtype to convert

Returns
-------
ctype
    A ctype scalar, union, array, or struct

Raises
------
NotImplementedError
    If the conversion is not possible

Notes
-----
This function does not losslessly round-trip in either direction.

``np.dtype(as_ctypes_type(dt))`` will:

- insert padding fields
- reorder fields to be sorted by offset
- discard field titles

``as_ctypes_type(np.dtype(ctype))`` will:

- discard the class names of `ctypes.Structure`\ s and
  `ctypes.Union`\ s
- convert single-element `ctypes.Union`\ s into single-element
  `ctypes.Structure`\ s
- insert padding fields

Examples
--------
Converting a simple dtype:

>>> dt = np.dtype('int8')
>>> ctype = np.ctypeslib.as_ctypes_type(dt)
>>> ctype
<class 'ctypes.c_byte'>

Converting a structured dtype:

>>> dt = np.dtype([('x', 'i4'), ('y', 'f4')])
>>> ctype = np.ctypeslib.as_ctypes_type(dt)
>>> ctype
<class 'struct'>

### Function: as_array(obj, shape)

**Description:** Create a numpy array from a ctypes array or POINTER.

The numpy array shares the memory with the ctypes object.

The shape parameter must be given if converting from a ctypes POINTER.
The shape parameter is ignored if converting from a ctypes array

Examples
--------
Converting a ctypes integer array:

>>> import ctypes
>>> ctypes_array = (ctypes.c_int * 5)(0, 1, 2, 3, 4)
>>> np_array = np.ctypeslib.as_array(ctypes_array)
>>> np_array
array([0, 1, 2, 3, 4], dtype=int32)

Converting a ctypes POINTER:

>>> import ctypes
>>> buffer = (ctypes.c_int * 5)(0, 1, 2, 3, 4)
>>> pointer = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_int))
>>> np_array = np.ctypeslib.as_array(pointer, (5,))
>>> np_array
array([0, 1, 2, 3, 4], dtype=int32)

### Function: as_ctypes(obj)

**Description:** Create and return a ctypes object from a numpy array.  Actually
anything that exposes the __array_interface__ is accepted.

Examples
--------
Create ctypes object from inferred int ``np.array``:

>>> inferred_int_array = np.array([1, 2, 3])
>>> c_int_array = np.ctypeslib.as_ctypes(inferred_int_array)
>>> type(c_int_array)
<class 'c_long_Array_3'>
>>> c_int_array[:]
[1, 2, 3]

Create ctypes object from explicit 8 bit unsigned int ``np.array`` :

>>> exp_int_array = np.array([1, 2, 3], dtype=np.uint8)
>>> c_int_array = np.ctypeslib.as_ctypes(exp_int_array)
>>> type(c_int_array)
<class 'c_ubyte_Array_3'>
>>> c_int_array[:]
[1, 2, 3]
