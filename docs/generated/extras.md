## AI Summary

A file named extras.py.


### Function: issequence(seq)

**Description:** Is seq a sequence (ndarray, list or tuple)?

### Function: count_masked(arr, axis)

**Description:** Count the number of masked elements along the given axis.

Parameters
----------
arr : array_like
    An array with (possibly) masked elements.
axis : int, optional
    Axis along which to count. If None (default), a flattened
    version of the array is used.

Returns
-------
count : int, ndarray
    The total number of masked elements (axis=None) or the number
    of masked elements along each slice of the given axis.

See Also
--------
MaskedArray.count : Count non-masked elements.

Examples
--------
>>> import numpy as np
>>> a = np.arange(9).reshape((3,3))
>>> a = np.ma.array(a)
>>> a[1, 0] = np.ma.masked
>>> a[1, 2] = np.ma.masked
>>> a[2, 1] = np.ma.masked
>>> a
masked_array(
  data=[[0, 1, 2],
        [--, 4, --],
        [6, --, 8]],
  mask=[[False, False, False],
        [ True, False,  True],
        [False,  True, False]],
  fill_value=999999)
>>> np.ma.count_masked(a)
3

When the `axis` keyword is used an array is returned.

>>> np.ma.count_masked(a, axis=0)
array([1, 1, 1])
>>> np.ma.count_masked(a, axis=1)
array([0, 2, 1])

### Function: masked_all(shape, dtype)

**Description:** Empty masked array with all elements masked.

Return an empty masked array of the given shape and dtype, where all the
data are masked.

Parameters
----------
shape : int or tuple of ints
    Shape of the required MaskedArray, e.g., ``(2, 3)`` or ``2``.
dtype : dtype, optional
    Data type of the output.

Returns
-------
a : MaskedArray
    A masked array with all data masked.

See Also
--------
masked_all_like : Empty masked array modelled on an existing array.

Notes
-----
Unlike other masked array creation functions (e.g. `numpy.ma.zeros`,
`numpy.ma.ones`, `numpy.ma.full`), `masked_all` does not initialize the
values of the array, and may therefore be marginally faster. However,
the values stored in the newly allocated array are arbitrary. For
reproducible behavior, be sure to set each element of the array before
reading.

Examples
--------
>>> import numpy as np
>>> np.ma.masked_all((3, 3))
masked_array(
  data=[[--, --, --],
        [--, --, --],
        [--, --, --]],
  mask=[[ True,  True,  True],
        [ True,  True,  True],
        [ True,  True,  True]],
  fill_value=1e+20,
  dtype=float64)

The `dtype` parameter defines the underlying data type.

>>> a = np.ma.masked_all((3, 3))
>>> a.dtype
dtype('float64')
>>> a = np.ma.masked_all((3, 3), dtype=np.int32)
>>> a.dtype
dtype('int32')

### Function: masked_all_like(arr)

**Description:** Empty masked array with the properties of an existing array.

Return an empty masked array of the same shape and dtype as
the array `arr`, where all the data are masked.

Parameters
----------
arr : ndarray
    An array describing the shape and dtype of the required MaskedArray.

Returns
-------
a : MaskedArray
    A masked array with all data masked.

Raises
------
AttributeError
    If `arr` doesn't have a shape attribute (i.e. not an ndarray)

See Also
--------
masked_all : Empty masked array with all elements masked.

Notes
-----
Unlike other masked array creation functions (e.g. `numpy.ma.zeros_like`,
`numpy.ma.ones_like`, `numpy.ma.full_like`), `masked_all_like` does not
initialize the values of the array, and may therefore be marginally
faster. However, the values stored in the newly allocated array are
arbitrary. For reproducible behavior, be sure to set each element of the
array before reading.

Examples
--------
>>> import numpy as np
>>> arr = np.zeros((2, 3), dtype=np.float32)
>>> arr
array([[0., 0., 0.],
       [0., 0., 0.]], dtype=float32)
>>> np.ma.masked_all_like(arr)
masked_array(
  data=[[--, --, --],
        [--, --, --]],
  mask=[[ True,  True,  True],
        [ True,  True,  True]],
  fill_value=np.float64(1e+20),
  dtype=float32)

The dtype of the masked array matches the dtype of `arr`.

>>> arr.dtype
dtype('float32')
>>> np.ma.masked_all_like(arr).dtype
dtype('float32')

## Class: _fromnxfunction

**Description:** Defines a wrapper to adapt NumPy functions to masked arrays.


An instance of `_fromnxfunction` can be called with the same parameters
as the wrapped NumPy function. The docstring of `newfunc` is adapted from
the wrapped function as well, see `getdoc`.

This class should not be used directly. Instead, one of its extensions that
provides support for a specific type of input should be used.

Parameters
----------
funcname : str
    The name of the function to be adapted. The function should be
    in the NumPy namespace (i.e. ``np.funcname``).

## Class: _fromnxfunction_single

**Description:** A version of `_fromnxfunction` that is called with a single array
argument followed by auxiliary args that are passed verbatim for
both the data and mask calls.

## Class: _fromnxfunction_seq

**Description:** A version of `_fromnxfunction` that is called with a single sequence
of arrays followed by auxiliary args that are passed verbatim for
both the data and mask calls.

## Class: _fromnxfunction_args

**Description:** A version of `_fromnxfunction` that is called with multiple array
arguments. The first non-array-like input marks the beginning of the
arguments that are passed verbatim for both the data and mask calls.
Array arguments are processed independently and the results are
returned in a list. If only one array is found, the return value is
just the processed array instead of a list.

## Class: _fromnxfunction_allargs

**Description:** A version of `_fromnxfunction` that is called with multiple array
arguments. Similar to `_fromnxfunction_args` except that all args
are converted to arrays even if they are not so already. This makes
it possible to process scalars as 1-D arrays. Only keyword arguments
are passed through verbatim for the data and mask calls. Arrays
arguments are processed independently and the results are returned
in a list. If only one arg is present, the return value is just the
processed array instead of a list.

### Function: flatten_inplace(seq)

**Description:** Flatten a sequence in place.

### Function: apply_along_axis(func1d, axis, arr)

**Description:** (This docstring should be overwritten)

### Function: apply_over_axes(func, a, axes)

**Description:** (This docstring will be overwritten)

### Function: average(a, axis, weights, returned)

**Description:** Return the weighted average of array over the given axis.

Parameters
----------
a : array_like
    Data to be averaged.
    Masked entries are not taken into account in the computation.
axis : None or int or tuple of ints, optional
    Axis or axes along which to average `a`.  The default,
    `axis=None`, will average over all of the elements of the input array.
    If axis is a tuple of ints, averaging is performed on all of the axes
    specified in the tuple instead of a single axis or all the axes as
    before.
weights : array_like, optional
    An array of weights associated with the values in `a`. Each value in
    `a` contributes to the average according to its associated weight.
    The array of weights must be the same shape as `a` if no axis is
    specified, otherwise the weights must have dimensions and shape
    consistent with `a` along the specified axis.
    If `weights=None`, then all data in `a` are assumed to have a
    weight equal to one.
    The calculation is::

        avg = sum(a * weights) / sum(weights)

    where the sum is over all included elements.
    The only constraint on the values of `weights` is that `sum(weights)`
    must not be 0.
returned : bool, optional
    Flag indicating whether a tuple ``(result, sum of weights)``
    should be returned as output (True), or just the result (False).
    Default is False.
keepdims : bool, optional
    If this is set to True, the axes which are reduced are left
    in the result as dimensions with size one. With this option,
    the result will broadcast correctly against the original `a`.
    *Note:* `keepdims` will not work with instances of `numpy.matrix`
    or other classes whose methods do not support `keepdims`.

    .. versionadded:: 1.23.0

Returns
-------
average, [sum_of_weights] : (tuple of) scalar or MaskedArray
    The average along the specified axis. When returned is `True`,
    return a tuple with the average as the first element and the sum
    of the weights as the second element. The return type is `np.float64`
    if `a` is of integer type and floats smaller than `float64`, or the
    input data-type, otherwise. If returned, `sum_of_weights` is always
    `float64`.

Raises
------
ZeroDivisionError
    When all weights along axis are zero. See `numpy.ma.average` for a
    version robust to this type of error.
TypeError
    When `weights` does not have the same shape as `a`, and `axis=None`.
ValueError
    When `weights` does not have dimensions and shape consistent with `a`
    along specified `axis`.

Examples
--------
>>> import numpy as np
>>> a = np.ma.array([1., 2., 3., 4.], mask=[False, False, True, True])
>>> np.ma.average(a, weights=[3, 1, 0, 0])
1.25

>>> x = np.ma.arange(6.).reshape(3, 2)
>>> x
masked_array(
  data=[[0., 1.],
        [2., 3.],
        [4., 5.]],
  mask=False,
  fill_value=1e+20)
>>> data = np.arange(8).reshape((2, 2, 2))
>>> data
array([[[0, 1],
        [2, 3]],
       [[4, 5],
        [6, 7]]])
>>> np.ma.average(data, axis=(0, 1), weights=[[1./4, 3./4], [1., 1./2]])
masked_array(data=[3.4, 4.4],
         mask=[False, False],
   fill_value=1e+20)
>>> np.ma.average(data, axis=0, weights=[[1./4, 3./4], [1., 1./2]])
Traceback (most recent call last):
    ...
ValueError: Shape of weights must be consistent
with shape of a along specified axis.

>>> avg, sumweights = np.ma.average(x, axis=0, weights=[1, 2, 3],
...                                 returned=True)
>>> avg
masked_array(data=[2.6666666666666665, 3.6666666666666665],
             mask=[False, False],
       fill_value=1e+20)

With ``keepdims=True``, the following result has shape (3, 1).

>>> np.ma.average(x, axis=1, keepdims=True)
masked_array(
  data=[[0.5],
        [2.5],
        [4.5]],
  mask=False,
  fill_value=1e+20)

### Function: median(a, axis, out, overwrite_input, keepdims)

**Description:** Compute the median along the specified axis.

Returns the median of the array elements.

Parameters
----------
a : array_like
    Input array or object that can be converted to an array.
axis : int, optional
    Axis along which the medians are computed. The default (None) is
    to compute the median along a flattened version of the array.
out : ndarray, optional
    Alternative output array in which to place the result. It must
    have the same shape and buffer length as the expected output
    but the type will be cast if necessary.
overwrite_input : bool, optional
    If True, then allow use of memory of input array (a) for
    calculations. The input array will be modified by the call to
    median. This will save memory when you do not need to preserve
    the contents of the input array. Treat the input as undefined,
    but it will probably be fully or partially sorted. Default is
    False. Note that, if `overwrite_input` is True, and the input
    is not already an `ndarray`, an error will be raised.
keepdims : bool, optional
    If this is set to True, the axes which are reduced are left
    in the result as dimensions with size one. With this option,
    the result will broadcast correctly against the input array.

Returns
-------
median : ndarray
    A new array holding the result is returned unless out is
    specified, in which case a reference to out is returned.
    Return data-type is `float64` for integers and floats smaller than
    `float64`, or the input data-type, otherwise.

See Also
--------
mean

Notes
-----
Given a vector ``V`` with ``N`` non masked values, the median of ``V``
is the middle value of a sorted copy of ``V`` (``Vs``) - i.e.
``Vs[(N-1)/2]``, when ``N`` is odd, or ``{Vs[N/2 - 1] + Vs[N/2]}/2``
when ``N`` is even.

Examples
--------
>>> import numpy as np
>>> x = np.ma.array(np.arange(8), mask=[0]*4 + [1]*4)
>>> np.ma.median(x)
1.5

>>> x = np.ma.array(np.arange(10).reshape(2, 5), mask=[0]*6 + [1]*4)
>>> np.ma.median(x)
2.5
>>> np.ma.median(x, axis=-1, overwrite_input=True)
masked_array(data=[2.0, 5.0],
             mask=[False, False],
       fill_value=1e+20)

### Function: _median(a, axis, out, overwrite_input)

### Function: compress_nd(x, axis)

**Description:** Suppress slices from multiple dimensions which contain masked values.

Parameters
----------
x : array_like, MaskedArray
    The array to operate on. If not a MaskedArray instance (or if no array
    elements are masked), `x` is interpreted as a MaskedArray with `mask`
    set to `nomask`.
axis : tuple of ints or int, optional
    Which dimensions to suppress slices from can be configured with this
    parameter.
    - If axis is a tuple of ints, those are the axes to suppress slices from.
    - If axis is an int, then that is the only axis to suppress slices from.
    - If axis is None, all axis are selected.

Returns
-------
compress_array : ndarray
    The compressed array.

Examples
--------
>>> import numpy as np
>>> arr = [[1, 2], [3, 4]]
>>> mask = [[0, 1], [0, 0]]
>>> x = np.ma.array(arr, mask=mask)
>>> np.ma.compress_nd(x, axis=0)
array([[3, 4]])
>>> np.ma.compress_nd(x, axis=1)
array([[1],
       [3]])
>>> np.ma.compress_nd(x)
array([[3]])

### Function: compress_rowcols(x, axis)

**Description:** Suppress the rows and/or columns of a 2-D array that contain
masked values.

The suppression behavior is selected with the `axis` parameter.

- If axis is None, both rows and columns are suppressed.
- If axis is 0, only rows are suppressed.
- If axis is 1 or -1, only columns are suppressed.

Parameters
----------
x : array_like, MaskedArray
    The array to operate on.  If not a MaskedArray instance (or if no array
    elements are masked), `x` is interpreted as a MaskedArray with
    `mask` set to `nomask`. Must be a 2D array.
axis : int, optional
    Axis along which to perform the operation. Default is None.

Returns
-------
compressed_array : ndarray
    The compressed array.

Examples
--------
>>> import numpy as np
>>> x = np.ma.array(np.arange(9).reshape(3, 3), mask=[[1, 0, 0],
...                                                   [1, 0, 0],
...                                                   [0, 0, 0]])
>>> x
masked_array(
  data=[[--, 1, 2],
        [--, 4, 5],
        [6, 7, 8]],
  mask=[[ True, False, False],
        [ True, False, False],
        [False, False, False]],
  fill_value=999999)

>>> np.ma.compress_rowcols(x)
array([[7, 8]])
>>> np.ma.compress_rowcols(x, 0)
array([[6, 7, 8]])
>>> np.ma.compress_rowcols(x, 1)
array([[1, 2],
       [4, 5],
       [7, 8]])

### Function: compress_rows(a)

**Description:** Suppress whole rows of a 2-D array that contain masked values.

This is equivalent to ``np.ma.compress_rowcols(a, 0)``, see
`compress_rowcols` for details.

Parameters
----------
x : array_like, MaskedArray
    The array to operate on. If not a MaskedArray instance (or if no array
    elements are masked), `x` is interpreted as a MaskedArray with
    `mask` set to `nomask`. Must be a 2D array.

Returns
-------
compressed_array : ndarray
    The compressed array.

See Also
--------
compress_rowcols

Examples
--------
>>> import numpy as np
>>> a = np.ma.array(np.arange(9).reshape(3, 3), mask=[[1, 0, 0],
...                                                   [1, 0, 0],
...                                                   [0, 0, 0]])
>>> np.ma.compress_rows(a)
array([[6, 7, 8]])

### Function: compress_cols(a)

**Description:** Suppress whole columns of a 2-D array that contain masked values.

This is equivalent to ``np.ma.compress_rowcols(a, 1)``, see
`compress_rowcols` for details.

Parameters
----------
x : array_like, MaskedArray
    The array to operate on.  If not a MaskedArray instance (or if no array
    elements are masked), `x` is interpreted as a MaskedArray with
    `mask` set to `nomask`. Must be a 2D array.

Returns
-------
compressed_array : ndarray
    The compressed array.

See Also
--------
compress_rowcols

Examples
--------
>>> import numpy as np
>>> a = np.ma.array(np.arange(9).reshape(3, 3), mask=[[1, 0, 0],
...                                                   [1, 0, 0],
...                                                   [0, 0, 0]])
>>> np.ma.compress_cols(a)
array([[1, 2],
       [4, 5],
       [7, 8]])

### Function: mask_rowcols(a, axis)

**Description:** Mask rows and/or columns of a 2D array that contain masked values.

Mask whole rows and/or columns of a 2D array that contain
masked values.  The masking behavior is selected using the
`axis` parameter.

  - If `axis` is None, rows *and* columns are masked.
  - If `axis` is 0, only rows are masked.
  - If `axis` is 1 or -1, only columns are masked.

Parameters
----------
a : array_like, MaskedArray
    The array to mask.  If not a MaskedArray instance (or if no array
    elements are masked), the result is a MaskedArray with `mask` set
    to `nomask` (False). Must be a 2D array.
axis : int, optional
    Axis along which to perform the operation. If None, applies to a
    flattened version of the array.

Returns
-------
a : MaskedArray
    A modified version of the input array, masked depending on the value
    of the `axis` parameter.

Raises
------
NotImplementedError
    If input array `a` is not 2D.

See Also
--------
mask_rows : Mask rows of a 2D array that contain masked values.
mask_cols : Mask cols of a 2D array that contain masked values.
masked_where : Mask where a condition is met.

Notes
-----
The input array's mask is modified by this function.

Examples
--------
>>> import numpy as np
>>> a = np.zeros((3, 3), dtype=int)
>>> a[1, 1] = 1
>>> a
array([[0, 0, 0],
       [0, 1, 0],
       [0, 0, 0]])
>>> a = np.ma.masked_equal(a, 1)
>>> a
masked_array(
  data=[[0, 0, 0],
        [0, --, 0],
        [0, 0, 0]],
  mask=[[False, False, False],
        [False,  True, False],
        [False, False, False]],
  fill_value=1)
>>> np.ma.mask_rowcols(a)
masked_array(
  data=[[0, --, 0],
        [--, --, --],
        [0, --, 0]],
  mask=[[False,  True, False],
        [ True,  True,  True],
        [False,  True, False]],
  fill_value=1)

### Function: mask_rows(a, axis)

**Description:** Mask rows of a 2D array that contain masked values.

This function is a shortcut to ``mask_rowcols`` with `axis` equal to 0.

See Also
--------
mask_rowcols : Mask rows and/or columns of a 2D array.
masked_where : Mask where a condition is met.

Examples
--------
>>> import numpy as np
>>> a = np.zeros((3, 3), dtype=int)
>>> a[1, 1] = 1
>>> a
array([[0, 0, 0],
       [0, 1, 0],
       [0, 0, 0]])
>>> a = np.ma.masked_equal(a, 1)
>>> a
masked_array(
  data=[[0, 0, 0],
        [0, --, 0],
        [0, 0, 0]],
  mask=[[False, False, False],
        [False,  True, False],
        [False, False, False]],
  fill_value=1)

>>> np.ma.mask_rows(a)
masked_array(
  data=[[0, 0, 0],
        [--, --, --],
        [0, 0, 0]],
  mask=[[False, False, False],
        [ True,  True,  True],
        [False, False, False]],
  fill_value=1)

### Function: mask_cols(a, axis)

**Description:** Mask columns of a 2D array that contain masked values.

This function is a shortcut to ``mask_rowcols`` with `axis` equal to 1.

See Also
--------
mask_rowcols : Mask rows and/or columns of a 2D array.
masked_where : Mask where a condition is met.

Examples
--------
>>> import numpy as np
>>> a = np.zeros((3, 3), dtype=int)
>>> a[1, 1] = 1
>>> a
array([[0, 0, 0],
       [0, 1, 0],
       [0, 0, 0]])
>>> a = np.ma.masked_equal(a, 1)
>>> a
masked_array(
  data=[[0, 0, 0],
        [0, --, 0],
        [0, 0, 0]],
  mask=[[False, False, False],
        [False,  True, False],
        [False, False, False]],
  fill_value=1)
>>> np.ma.mask_cols(a)
masked_array(
  data=[[0, --, 0],
        [0, --, 0],
        [0, --, 0]],
  mask=[[False,  True, False],
        [False,  True, False],
        [False,  True, False]],
  fill_value=1)

### Function: ediff1d(arr, to_end, to_begin)

**Description:** Compute the differences between consecutive elements of an array.

This function is the equivalent of `numpy.ediff1d` that takes masked
values into account, see `numpy.ediff1d` for details.

See Also
--------
numpy.ediff1d : Equivalent function for ndarrays.

Examples
--------
>>> import numpy as np
>>> arr = np.ma.array([1, 2, 4, 7, 0])
>>> np.ma.ediff1d(arr)
masked_array(data=[ 1,  2,  3, -7],
             mask=False,
       fill_value=999999)

### Function: unique(ar1, return_index, return_inverse)

**Description:** Finds the unique elements of an array.

Masked values are considered the same element (masked). The output array
is always a masked array. See `numpy.unique` for more details.

See Also
--------
numpy.unique : Equivalent function for ndarrays.

Examples
--------
>>> import numpy as np
>>> a = [1, 2, 1000, 2, 3]
>>> mask = [0, 0, 1, 0, 0]
>>> masked_a = np.ma.masked_array(a, mask)
>>> masked_a
masked_array(data=[1, 2, --, 2, 3],
            mask=[False, False,  True, False, False],
    fill_value=999999)
>>> np.ma.unique(masked_a)
masked_array(data=[1, 2, 3, --],
            mask=[False, False, False,  True],
    fill_value=999999)
>>> np.ma.unique(masked_a, return_index=True)
(masked_array(data=[1, 2, 3, --],
            mask=[False, False, False,  True],
    fill_value=999999), array([0, 1, 4, 2]))
>>> np.ma.unique(masked_a, return_inverse=True)
(masked_array(data=[1, 2, 3, --],
            mask=[False, False, False,  True],
    fill_value=999999), array([0, 1, 3, 1, 2]))
>>> np.ma.unique(masked_a, return_index=True, return_inverse=True)
(masked_array(data=[1, 2, 3, --],
            mask=[False, False, False,  True],
    fill_value=999999), array([0, 1, 4, 2]), array([0, 1, 3, 1, 2]))

### Function: intersect1d(ar1, ar2, assume_unique)

**Description:** Returns the unique elements common to both arrays.

Masked values are considered equal one to the other.
The output is always a masked array.

See `numpy.intersect1d` for more details.

See Also
--------
numpy.intersect1d : Equivalent function for ndarrays.

Examples
--------
>>> import numpy as np
>>> x = np.ma.array([1, 3, 3, 3], mask=[0, 0, 0, 1])
>>> y = np.ma.array([3, 1, 1, 1], mask=[0, 0, 0, 1])
>>> np.ma.intersect1d(x, y)
masked_array(data=[1, 3, --],
             mask=[False, False,  True],
       fill_value=999999)

### Function: setxor1d(ar1, ar2, assume_unique)

**Description:** Set exclusive-or of 1-D arrays with unique elements.

The output is always a masked array. See `numpy.setxor1d` for more details.

See Also
--------
numpy.setxor1d : Equivalent function for ndarrays.

Examples
--------
>>> import numpy as np
>>> ar1 = np.ma.array([1, 2, 3, 2, 4])
>>> ar2 = np.ma.array([2, 3, 5, 7, 5])
>>> np.ma.setxor1d(ar1, ar2)
masked_array(data=[1, 4, 5, 7],
             mask=False,
       fill_value=999999)

### Function: in1d(ar1, ar2, assume_unique, invert)

**Description:** Test whether each element of an array is also present in a second
array.

The output is always a masked array. See `numpy.in1d` for more details.

We recommend using :func:`isin` instead of `in1d` for new code.

See Also
--------
isin       : Version of this function that preserves the shape of ar1.
numpy.in1d : Equivalent function for ndarrays.

Examples
--------
>>> import numpy as np
>>> ar1 = np.ma.array([0, 1, 2, 5, 0])
>>> ar2 = [0, 2]
>>> np.ma.in1d(ar1, ar2)
masked_array(data=[ True, False,  True, False,  True],
             mask=False,
       fill_value=True)

### Function: isin(element, test_elements, assume_unique, invert)

**Description:** Calculates `element in test_elements`, broadcasting over
`element` only.

The output is always a masked array of the same shape as `element`.
See `numpy.isin` for more details.

See Also
--------
in1d       : Flattened version of this function.
numpy.isin : Equivalent function for ndarrays.

Examples
--------
>>> import numpy as np
>>> element = np.ma.array([1, 2, 3, 4, 5, 6])
>>> test_elements = [0, 2]
>>> np.ma.isin(element, test_elements)
masked_array(data=[False,  True, False, False, False, False],
             mask=False,
       fill_value=True)

### Function: union1d(ar1, ar2)

**Description:** Union of two arrays.

The output is always a masked array. See `numpy.union1d` for more details.

See Also
--------
numpy.union1d : Equivalent function for ndarrays.

Examples
--------
>>> import numpy as np
>>> ar1 = np.ma.array([1, 2, 3, 4])
>>> ar2 = np.ma.array([3, 4, 5, 6])
>>> np.ma.union1d(ar1, ar2)
masked_array(data=[1, 2, 3, 4, 5, 6],
         mask=False,
   fill_value=999999)

### Function: setdiff1d(ar1, ar2, assume_unique)

**Description:** Set difference of 1D arrays with unique elements.

The output is always a masked array. See `numpy.setdiff1d` for more
details.

See Also
--------
numpy.setdiff1d : Equivalent function for ndarrays.

Examples
--------
>>> import numpy as np
>>> x = np.ma.array([1, 2, 3, 4], mask=[0, 1, 0, 1])
>>> np.ma.setdiff1d(x, [1, 2])
masked_array(data=[3, --],
             mask=[False,  True],
       fill_value=999999)

### Function: _covhelper(x, y, rowvar, allow_masked)

**Description:** Private function for the computation of covariance and correlation
coefficients.

### Function: cov(x, y, rowvar, bias, allow_masked, ddof)

**Description:** Estimate the covariance matrix.

Except for the handling of missing data this function does the same as
`numpy.cov`. For more details and examples, see `numpy.cov`.

By default, masked values are recognized as such. If `x` and `y` have the
same shape, a common mask is allocated: if ``x[i,j]`` is masked, then
``y[i,j]`` will also be masked.
Setting `allow_masked` to False will raise an exception if values are
missing in either of the input arrays.

Parameters
----------
x : array_like
    A 1-D or 2-D array containing multiple variables and observations.
    Each row of `x` represents a variable, and each column a single
    observation of all those variables. Also see `rowvar` below.
y : array_like, optional
    An additional set of variables and observations. `y` has the same
    shape as `x`.
rowvar : bool, optional
    If `rowvar` is True (default), then each row represents a
    variable, with observations in the columns. Otherwise, the relationship
    is transposed: each column represents a variable, while the rows
    contain observations.
bias : bool, optional
    Default normalization (False) is by ``(N-1)``, where ``N`` is the
    number of observations given (unbiased estimate). If `bias` is True,
    then normalization is by ``N``. This keyword can be overridden by
    the keyword ``ddof`` in numpy versions >= 1.5.
allow_masked : bool, optional
    If True, masked values are propagated pair-wise: if a value is masked
    in `x`, the corresponding value is masked in `y`.
    If False, raises a `ValueError` exception when some values are missing.
ddof : {None, int}, optional
    If not ``None`` normalization is by ``(N - ddof)``, where ``N`` is
    the number of observations; this overrides the value implied by
    ``bias``. The default value is ``None``.

Raises
------
ValueError
    Raised if some values are missing and `allow_masked` is False.

See Also
--------
numpy.cov

Examples
--------
>>> import numpy as np
>>> x = np.ma.array([[0, 1], [1, 1]], mask=[0, 1, 0, 1])
>>> y = np.ma.array([[1, 0], [0, 1]], mask=[0, 0, 1, 1])
>>> np.ma.cov(x, y)
masked_array(
data=[[--, --, --, --],
      [--, --, --, --],
      [--, --, --, --],
      [--, --, --, --]],
mask=[[ True,  True,  True,  True],
      [ True,  True,  True,  True],
      [ True,  True,  True,  True],
      [ True,  True,  True,  True]],
fill_value=1e+20,
dtype=float64)

### Function: corrcoef(x, y, rowvar, bias, allow_masked, ddof)

**Description:** Return Pearson product-moment correlation coefficients.

Except for the handling of missing data this function does the same as
`numpy.corrcoef`. For more details and examples, see `numpy.corrcoef`.

Parameters
----------
x : array_like
    A 1-D or 2-D array containing multiple variables and observations.
    Each row of `x` represents a variable, and each column a single
    observation of all those variables. Also see `rowvar` below.
y : array_like, optional
    An additional set of variables and observations. `y` has the same
    shape as `x`.
rowvar : bool, optional
    If `rowvar` is True (default), then each row represents a
    variable, with observations in the columns. Otherwise, the relationship
    is transposed: each column represents a variable, while the rows
    contain observations.
bias : _NoValue, optional
    Has no effect, do not use.

    .. deprecated:: 1.10.0
allow_masked : bool, optional
    If True, masked values are propagated pair-wise: if a value is masked
    in `x`, the corresponding value is masked in `y`.
    If False, raises an exception.  Because `bias` is deprecated, this
    argument needs to be treated as keyword only to avoid a warning.
ddof : _NoValue, optional
    Has no effect, do not use.

    .. deprecated:: 1.10.0

See Also
--------
numpy.corrcoef : Equivalent function in top-level NumPy module.
cov : Estimate the covariance matrix.

Notes
-----
This function accepts but discards arguments `bias` and `ddof`.  This is
for backwards compatibility with previous versions of this function.  These
arguments had no effect on the return values of the function and can be
safely ignored in this and previous versions of numpy.

Examples
--------
>>> import numpy as np
>>> x = np.ma.array([[0, 1], [1, 1]], mask=[0, 1, 0, 1])
>>> np.ma.corrcoef(x)
masked_array(
  data=[[--, --],
        [--, --]],
  mask=[[ True,  True],
        [ True,  True]],
  fill_value=1e+20,
  dtype=float64)

## Class: MAxisConcatenator

**Description:** Translate slice objects to concatenation along an axis.

For documentation on usage, see `mr_class`.

See Also
--------
mr_class

## Class: mr_class

**Description:** Translate slice objects to concatenation along the first axis.

This is the masked array version of `r_`.

See Also
--------
r_

Examples
--------
>>> import numpy as np
>>> np.ma.mr_[np.ma.array([1,2,3]), 0, 0, np.ma.array([4,5,6])]
masked_array(data=[1, 2, 3, ..., 4, 5, 6],
             mask=False,
       fill_value=999999)

### Function: ndenumerate(a, compressed)

**Description:** Multidimensional index iterator.

Return an iterator yielding pairs of array coordinates and values,
skipping elements that are masked. With `compressed=False`,
`ma.masked` is yielded as the value of masked elements. This
behavior differs from that of `numpy.ndenumerate`, which yields the
value of the underlying data array.

Notes
-----
.. versionadded:: 1.23.0

Parameters
----------
a : array_like
    An array with (possibly) masked elements.
compressed : bool, optional
    If True (default), masked elements are skipped.

See Also
--------
numpy.ndenumerate : Equivalent function ignoring any mask.

Examples
--------
>>> import numpy as np
>>> a = np.ma.arange(9).reshape((3, 3))
>>> a[1, 0] = np.ma.masked
>>> a[1, 2] = np.ma.masked
>>> a[2, 1] = np.ma.masked
>>> a
masked_array(
  data=[[0, 1, 2],
        [--, 4, --],
        [6, --, 8]],
  mask=[[False, False, False],
        [ True, False,  True],
        [False,  True, False]],
  fill_value=999999)
>>> for index, x in np.ma.ndenumerate(a):
...     print(index, x)
(0, 0) 0
(0, 1) 1
(0, 2) 2
(1, 1) 4
(2, 0) 6
(2, 2) 8

>>> for index, x in np.ma.ndenumerate(a, compressed=False):
...     print(index, x)
(0, 0) 0
(0, 1) 1
(0, 2) 2
(1, 0) --
(1, 1) 4
(1, 2) --
(2, 0) 6
(2, 1) --
(2, 2) 8

### Function: flatnotmasked_edges(a)

**Description:** Find the indices of the first and last unmasked values.

Expects a 1-D `MaskedArray`, returns None if all values are masked.

Parameters
----------
a : array_like
    Input 1-D `MaskedArray`

Returns
-------
edges : ndarray or None
    The indices of first and last non-masked value in the array.
    Returns None if all values are masked.

See Also
--------
flatnotmasked_contiguous, notmasked_contiguous, notmasked_edges
clump_masked, clump_unmasked

Notes
-----
Only accepts 1-D arrays.

Examples
--------
>>> import numpy as np
>>> a = np.ma.arange(10)
>>> np.ma.flatnotmasked_edges(a)
array([0, 9])

>>> mask = (a < 3) | (a > 8) | (a == 5)
>>> a[mask] = np.ma.masked
>>> np.array(a[~a.mask])
array([3, 4, 6, 7, 8])

>>> np.ma.flatnotmasked_edges(a)
array([3, 8])

>>> a[:] = np.ma.masked
>>> print(np.ma.flatnotmasked_edges(a))
None

### Function: notmasked_edges(a, axis)

**Description:** Find the indices of the first and last unmasked values along an axis.

If all values are masked, return None.  Otherwise, return a list
of two tuples, corresponding to the indices of the first and last
unmasked values respectively.

Parameters
----------
a : array_like
    The input array.
axis : int, optional
    Axis along which to perform the operation.
    If None (default), applies to a flattened version of the array.

Returns
-------
edges : ndarray or list
    An array of start and end indexes if there are any masked data in
    the array. If there are no masked data in the array, `edges` is a
    list of the first and last index.

See Also
--------
flatnotmasked_contiguous, flatnotmasked_edges, notmasked_contiguous
clump_masked, clump_unmasked

Examples
--------
>>> import numpy as np
>>> a = np.arange(9).reshape((3, 3))
>>> m = np.zeros_like(a)
>>> m[1:, 1:] = 1

>>> am = np.ma.array(a, mask=m)
>>> np.array(am[~am.mask])
array([0, 1, 2, 3, 6])

>>> np.ma.notmasked_edges(am)
array([0, 6])

### Function: flatnotmasked_contiguous(a)

**Description:** Find contiguous unmasked data in a masked array.

Parameters
----------
a : array_like
    The input array.

Returns
-------
slice_list : list
    A sorted sequence of `slice` objects (start index, end index).

See Also
--------
flatnotmasked_edges, notmasked_contiguous, notmasked_edges
clump_masked, clump_unmasked

Notes
-----
Only accepts 2-D arrays at most.

Examples
--------
>>> import numpy as np
>>> a = np.ma.arange(10)
>>> np.ma.flatnotmasked_contiguous(a)
[slice(0, 10, None)]

>>> mask = (a < 3) | (a > 8) | (a == 5)
>>> a[mask] = np.ma.masked
>>> np.array(a[~a.mask])
array([3, 4, 6, 7, 8])

>>> np.ma.flatnotmasked_contiguous(a)
[slice(3, 5, None), slice(6, 9, None)]
>>> a[:] = np.ma.masked
>>> np.ma.flatnotmasked_contiguous(a)
[]

### Function: notmasked_contiguous(a, axis)

**Description:** Find contiguous unmasked data in a masked array along the given axis.

Parameters
----------
a : array_like
    The input array.
axis : int, optional
    Axis along which to perform the operation.
    If None (default), applies to a flattened version of the array, and this
    is the same as `flatnotmasked_contiguous`.

Returns
-------
endpoints : list
    A list of slices (start and end indexes) of unmasked indexes
    in the array.

    If the input is 2d and axis is specified, the result is a list of lists.

See Also
--------
flatnotmasked_edges, flatnotmasked_contiguous, notmasked_edges
clump_masked, clump_unmasked

Notes
-----
Only accepts 2-D arrays at most.

Examples
--------
>>> import numpy as np
>>> a = np.arange(12).reshape((3, 4))
>>> mask = np.zeros_like(a)
>>> mask[1:, :-1] = 1; mask[0, 1] = 1; mask[-1, 0] = 0
>>> ma = np.ma.array(a, mask=mask)
>>> ma
masked_array(
  data=[[0, --, 2, 3],
        [--, --, --, 7],
        [8, --, --, 11]],
  mask=[[False,  True, False, False],
        [ True,  True,  True, False],
        [False,  True,  True, False]],
  fill_value=999999)
>>> np.array(ma[~ma.mask])
array([ 0,  2,  3,  7, 8, 11])

>>> np.ma.notmasked_contiguous(ma)
[slice(0, 1, None), slice(2, 4, None), slice(7, 9, None), slice(11, 12, None)]

>>> np.ma.notmasked_contiguous(ma, axis=0)
[[slice(0, 1, None), slice(2, 3, None)], [], [slice(0, 1, None)], [slice(0, 3, None)]]

>>> np.ma.notmasked_contiguous(ma, axis=1)
[[slice(0, 1, None), slice(2, 4, None)], [slice(3, 4, None)], [slice(0, 1, None), slice(3, 4, None)]]

### Function: _ezclump(mask)

**Description:** Finds the clumps (groups of data with the same values) for a 1D bool array.

Returns a series of slices.

### Function: clump_unmasked(a)

**Description:** Return list of slices corresponding to the unmasked clumps of a 1-D array.
(A "clump" is defined as a contiguous region of the array).

Parameters
----------
a : ndarray
    A one-dimensional masked array.

Returns
-------
slices : list of slice
    The list of slices, one for each continuous region of unmasked
    elements in `a`.

See Also
--------
flatnotmasked_edges, flatnotmasked_contiguous, notmasked_edges
notmasked_contiguous, clump_masked

Examples
--------
>>> import numpy as np
>>> a = np.ma.masked_array(np.arange(10))
>>> a[[0, 1, 2, 6, 8, 9]] = np.ma.masked
>>> np.ma.clump_unmasked(a)
[slice(3, 6, None), slice(7, 8, None)]

### Function: clump_masked(a)

**Description:** Returns a list of slices corresponding to the masked clumps of a 1-D array.
(A "clump" is defined as a contiguous region of the array).

Parameters
----------
a : ndarray
    A one-dimensional masked array.

Returns
-------
slices : list of slice
    The list of slices, one for each continuous region of masked elements
    in `a`.

See Also
--------
flatnotmasked_edges, flatnotmasked_contiguous, notmasked_edges
notmasked_contiguous, clump_unmasked

Examples
--------
>>> import numpy as np
>>> a = np.ma.masked_array(np.arange(10))
>>> a[[0, 1, 2, 6, 8, 9]] = np.ma.masked
>>> np.ma.clump_masked(a)
[slice(0, 3, None), slice(6, 7, None), slice(8, 10, None)]

### Function: vander(x, n)

**Description:** Masked values in the input array result in rows of zeros.

### Function: polyfit(x, y, deg, rcond, full, w, cov)

**Description:** Any masked values in x is propagated in y, and vice-versa.

### Function: __init__(self, funcname)

### Function: getdoc(self)

**Description:** Retrieve the docstring and signature from the function.

The ``__doc__`` attribute of the function is used as the docstring for
the new masked array version of the function. A note on application
of the function to the mask is appended.

Parameters
----------
None

### Function: __call__(self)

### Function: __call__(self, x)

### Function: __call__(self, x)

### Function: __call__(self)

### Function: __call__(self)

### Function: replace_masked(s)

### Function: makemat(cls, arr)

### Function: __getitem__(self, key)

### Function: __init__(self)
