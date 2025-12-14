## AI Summary

A file named _internal.py.


### Function: _makenames_list(adict, align)

### Function: _usefields(adict, align)

### Function: _array_descr(descriptor)

### Function: _commastring(astr)

## Class: dummy_ctype

### Function: _getintp_ctype()

## Class: _missing_ctypes

## Class: _ctypes

### Function: _newnames(datatype, order)

**Description:** Given a datatype and an order object, return a new names tuple, with the
order indicated

### Function: _copy_fields(ary)

**Description:** Return copy of structured array with padding between fields removed.

Parameters
----------
ary : ndarray
   Structured array from which to remove padding bytes

Returns
-------
ary_copy : ndarray
   Copy of ary with padding bytes removed

### Function: _promote_fields(dt1, dt2)

**Description:** Perform type promotion for two structured dtypes.

Parameters
----------
dt1 : structured dtype
    First dtype.
dt2 : structured dtype
    Second dtype.

Returns
-------
out : dtype
    The promoted dtype

Notes
-----
If one of the inputs is aligned, the result will be.  The titles of
both descriptors must match (point to the same field).

### Function: _getfield_is_safe(oldtype, newtype, offset)

**Description:** Checks safety of getfield for object arrays.

As in _view_is_safe, we need to check that memory containing objects is not
reinterpreted as a non-object datatype and vice versa.

Parameters
----------
oldtype : data-type
    Data type of the original ndarray.
newtype : data-type
    Data type of the field being accessed by ndarray.getfield
offset : int
    Offset of the field being accessed by ndarray.getfield

Raises
------
TypeError
    If the field access is invalid

### Function: _view_is_safe(oldtype, newtype)

**Description:** Checks safety of a view involving object arrays, for example when
doing::

    np.zeros(10, dtype=oldtype).view(newtype)

Parameters
----------
oldtype : data-type
    Data type of original ndarray
newtype : data-type
    Data type of the view

Raises
------
TypeError
    If the new type is incompatible with the old type.

## Class: _Stream

### Function: _dtype_from_pep3118(spec)

### Function: __dtype_from_pep3118(stream, is_subdtype)

### Function: _fix_names(field_spec)

**Description:** Replace names which are None with the next unused f%d name 

### Function: _add_trailing_padding(value, padding)

**Description:** Inject the specified number of padding bytes at the end of a dtype

### Function: _prod(a)

### Function: _gcd(a, b)

**Description:** Calculate the greatest common divisor of a and b

### Function: _lcm(a, b)

### Function: array_ufunc_errmsg_formatter(dummy, ufunc, method)

**Description:** Format the error message for when __array_ufunc__ gives up. 

### Function: array_function_errmsg_formatter(public_api, types)

**Description:** Format the error message for when __array_ufunc__ gives up. 

### Function: _ufunc_doc_signature_formatter(ufunc)

**Description:** Builds a signature string which resembles PEP 457

This is used to construct the first line of the docstring

### Function: npy_ctypes_check(cls)

### Function: _convert_to_stringdtype_kwargs(coerce, na_object)

### Function: __init__(self, cls)

### Function: __mul__(self, other)

### Function: __call__(self)

### Function: __eq__(self, other)

### Function: __ne__(self, other)

### Function: cast(self, num, obj)

## Class: c_void_p

### Function: __init__(self, array, ptr)

### Function: data_as(self, obj)

**Description:** Return the data pointer cast to a particular c-types object.
For example, calling ``self._as_parameter_`` is equivalent to
``self.data_as(ctypes.c_void_p)``. Perhaps you want to use
the data as a pointer to a ctypes array of floating-point data:
``self.data_as(ctypes.POINTER(ctypes.c_double))``.

The returned pointer will keep a reference to the array.

### Function: shape_as(self, obj)

**Description:** Return the shape tuple as an array of some other c-types
type. For example: ``self.shape_as(ctypes.c_short)``.

### Function: strides_as(self, obj)

**Description:** Return the strides tuple as an array of some other
c-types type. For example: ``self.strides_as(ctypes.c_longlong)``.

### Function: data(self)

**Description:** A pointer to the memory area of the array as a Python integer.
This memory area may contain data that is not aligned, or not in
correct byte-order. The memory area may not even be writeable.
The array flags and data-type of this array should be respected
when passing this attribute to arbitrary C-code to avoid trouble
that can include Python crashing. User Beware! The value of this
attribute is exactly the same as:
``self._array_interface_['data'][0]``.

Note that unlike ``data_as``, a reference won't be kept to the array:
code like ``ctypes.c_void_p((a + b).ctypes.data)`` will result in a
pointer to a deallocated array, and should be spelt
``(a + b).ctypes.data_as(ctypes.c_void_p)``

### Function: shape(self)

**Description:** (c_intp*self.ndim): A ctypes array of length self.ndim where
the basetype is the C-integer corresponding to ``dtype('p')`` on this
platform (see `~numpy.ctypeslib.c_intp`). This base-type could be
`ctypes.c_int`, `ctypes.c_long`, or `ctypes.c_longlong` depending on
the platform. The ctypes array contains the shape of
the underlying array.

### Function: strides(self)

**Description:** (c_intp*self.ndim): A ctypes array of length self.ndim where
the basetype is the same as for the shape attribute. This ctypes
array contains the strides information from the underlying array.
This strides information is important for showing how many bytes
must be jumped to get to the next element in the array.

### Function: _as_parameter_(self)

**Description:** Overrides the ctypes semi-magic method

Enables `c_func(some_array.ctypes)`

### Function: get_data(self)

**Description:** Deprecated getter for the `_ctypes.data` property.

.. deprecated:: 1.21

### Function: get_shape(self)

**Description:** Deprecated getter for the `_ctypes.shape` property.

.. deprecated:: 1.21

### Function: get_strides(self)

**Description:** Deprecated getter for the `_ctypes.strides` property.

.. deprecated:: 1.21

### Function: get_as_parameter(self)

**Description:** Deprecated getter for the `_ctypes._as_parameter_` property.

.. deprecated:: 1.21

### Function: __init__(self, s)

### Function: advance(self, n)

### Function: consume(self, c)

### Function: consume_until(self, c)

### Function: next(self)

### Function: __bool__(self)

### Function: __init__(self, ptr)
