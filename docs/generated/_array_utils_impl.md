## AI Summary

A file named _array_utils_impl.py.


### Function: byte_bounds(a)

**Description:** Returns pointers to the end-points of an array.

Parameters
----------
a : ndarray
    Input array. It must conform to the Python-side of the array
    interface.

Returns
-------
(low, high) : tuple of 2 integers
    The first integer is the first byte of the array, the second
    integer is just past the last byte of the array.  If `a` is not
    contiguous it will not use every byte between the (`low`, `high`)
    values.

Examples
--------
>>> import numpy as np
>>> I = np.eye(2, dtype='f'); I.dtype
dtype('float32')
>>> low, high = np.lib.array_utils.byte_bounds(I)
>>> high - low == I.size*I.itemsize
True
>>> I = np.eye(2); I.dtype
dtype('float64')
>>> low, high = np.lib.array_utils.byte_bounds(I)
>>> high - low == I.size*I.itemsize
True
