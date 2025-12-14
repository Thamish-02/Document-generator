## AI Summary

A file named _arraysetops_impl.py.


### Function: _ediff1d_dispatcher(ary, to_end, to_begin)

### Function: ediff1d(ary, to_end, to_begin)

**Description:** The differences between consecutive elements of an array.

Parameters
----------
ary : array_like
    If necessary, will be flattened before the differences are taken.
to_end : array_like, optional
    Number(s) to append at the end of the returned differences.
to_begin : array_like, optional
    Number(s) to prepend at the beginning of the returned differences.

Returns
-------
ediff1d : ndarray
    The differences. Loosely, this is ``ary.flat[1:] - ary.flat[:-1]``.

See Also
--------
diff, gradient

Notes
-----
When applied to masked arrays, this function drops the mask information
if the `to_begin` and/or `to_end` parameters are used.

Examples
--------
>>> import numpy as np
>>> x = np.array([1, 2, 4, 7, 0])
>>> np.ediff1d(x)
array([ 1,  2,  3, -7])

>>> np.ediff1d(x, to_begin=-99, to_end=np.array([88, 99]))
array([-99,   1,   2, ...,  -7,  88,  99])

The returned array is always 1D.

>>> y = [[1, 2, 4], [1, 6, 24]]
>>> np.ediff1d(y)
array([ 1,  2, -3,  5, 18])

### Function: _unpack_tuple(x)

**Description:** Unpacks one-element tuples for use as return values 

### Function: _unique_dispatcher(ar, return_index, return_inverse, return_counts, axis)

### Function: unique(ar, return_index, return_inverse, return_counts, axis)

**Description:** Find the unique elements of an array.

Returns the sorted unique elements of an array. There are three optional
outputs in addition to the unique elements:

* the indices of the input array that give the unique values
* the indices of the unique array that reconstruct the input array
* the number of times each unique value comes up in the input array

Parameters
----------
ar : array_like
    Input array. Unless `axis` is specified, this will be flattened if it
    is not already 1-D.
return_index : bool, optional
    If True, also return the indices of `ar` (along the specified axis,
    if provided, or in the flattened array) that result in the unique array.
return_inverse : bool, optional
    If True, also return the indices of the unique array (for the specified
    axis, if provided) that can be used to reconstruct `ar`.
return_counts : bool, optional
    If True, also return the number of times each unique item appears
    in `ar`.
axis : int or None, optional
    The axis to operate on. If None, `ar` will be flattened. If an integer,
    the subarrays indexed by the given axis will be flattened and treated
    as the elements of a 1-D array with the dimension of the given axis,
    see the notes for more details.  Object arrays or structured arrays
    that contain objects are not supported if the `axis` kwarg is used. The
    default is None.

equal_nan : bool, optional
    If True, collapses multiple NaN values in the return array into one.

    .. versionadded:: 1.24

Returns
-------
unique : ndarray
    The sorted unique values.
unique_indices : ndarray, optional
    The indices of the first occurrences of the unique values in the
    original array. Only provided if `return_index` is True.
unique_inverse : ndarray, optional
    The indices to reconstruct the original array from the
    unique array. Only provided if `return_inverse` is True.
unique_counts : ndarray, optional
    The number of times each of the unique values comes up in the
    original array. Only provided if `return_counts` is True.

See Also
--------
repeat : Repeat elements of an array.
sort : Return a sorted copy of an array.

Notes
-----
When an axis is specified the subarrays indexed by the axis are sorted.
This is done by making the specified axis the first dimension of the array
(move the axis to the first dimension to keep the order of the other axes)
and then flattening the subarrays in C order. The flattened subarrays are
then viewed as a structured type with each element given a label, with the
effect that we end up with a 1-D array of structured types that can be
treated in the same way as any other 1-D array. The result is that the
flattened subarrays are sorted in lexicographic order starting with the
first element.

.. versionchanged:: 1.21
    Like np.sort, NaN will sort to the end of the values.
    For complex arrays all NaN values are considered equivalent
    (no matter whether the NaN is in the real or imaginary part).
    As the representant for the returned array the smallest one in the
    lexicographical order is chosen - see np.sort for how the lexicographical
    order is defined for complex arrays.

.. versionchanged:: 2.0
    For multi-dimensional inputs, ``unique_inverse`` is reshaped
    such that the input can be reconstructed using
    ``np.take(unique, unique_inverse, axis=axis)``. The result is
    now not 1-dimensional when ``axis=None``.

    Note that in NumPy 2.0.0 a higher dimensional array was returned also
    when ``axis`` was not ``None``.  This was reverted, but
    ``inverse.reshape(-1)`` can be used to ensure compatibility with both
    versions.

Examples
--------
>>> import numpy as np
>>> np.unique([1, 1, 2, 2, 3, 3])
array([1, 2, 3])
>>> a = np.array([[1, 1], [2, 3]])
>>> np.unique(a)
array([1, 2, 3])

Return the unique rows of a 2D array

>>> a = np.array([[1, 0, 0], [1, 0, 0], [2, 3, 4]])
>>> np.unique(a, axis=0)
array([[1, 0, 0], [2, 3, 4]])

Return the indices of the original array that give the unique values:

>>> a = np.array(['a', 'b', 'b', 'c', 'a'])
>>> u, indices = np.unique(a, return_index=True)
>>> u
array(['a', 'b', 'c'], dtype='<U1')
>>> indices
array([0, 1, 3])
>>> a[indices]
array(['a', 'b', 'c'], dtype='<U1')

Reconstruct the input array from the unique values and inverse:

>>> a = np.array([1, 2, 6, 4, 2, 3, 2])
>>> u, indices = np.unique(a, return_inverse=True)
>>> u
array([1, 2, 3, 4, 6])
>>> indices
array([0, 1, 4, 3, 1, 2, 1])
>>> u[indices]
array([1, 2, 6, 4, 2, 3, 2])

Reconstruct the input values from the unique values and counts:

>>> a = np.array([1, 2, 6, 4, 2, 3, 2])
>>> values, counts = np.unique(a, return_counts=True)
>>> values
array([1, 2, 3, 4, 6])
>>> counts
array([1, 3, 1, 1, 1])
>>> np.repeat(values, counts)
array([1, 2, 2, 2, 3, 4, 6])    # original order not preserved

### Function: _unique1d(ar, return_index, return_inverse, return_counts)

**Description:** Find the unique elements of an array, ignoring shape.

## Class: UniqueAllResult

## Class: UniqueCountsResult

## Class: UniqueInverseResult

### Function: _unique_all_dispatcher()

### Function: unique_all(x)

**Description:** Find the unique elements of an array, and counts, inverse, and indices.

This function is an Array API compatible alternative to::

    np.unique(x, return_index=True, return_inverse=True,
              return_counts=True, equal_nan=False)

but returns a namedtuple for easier access to each output.

Parameters
----------
x : array_like
    Input array. It will be flattened if it is not already 1-D.

Returns
-------
out : namedtuple
    The result containing:

    * values - The unique elements of an input array.
    * indices - The first occurring indices for each unique element.
    * inverse_indices - The indices from the set of unique elements
      that reconstruct `x`.
    * counts - The corresponding counts for each unique element.

See Also
--------
unique : Find the unique elements of an array.

Examples
--------
>>> import numpy as np
>>> x = [1, 1, 2]
>>> uniq = np.unique_all(x)
>>> uniq.values
array([1, 2])
>>> uniq.indices
array([0, 2])
>>> uniq.inverse_indices
array([0, 0, 1])
>>> uniq.counts
array([2, 1])

### Function: _unique_counts_dispatcher()

### Function: unique_counts(x)

**Description:** Find the unique elements and counts of an input array `x`.

This function is an Array API compatible alternative to::

    np.unique(x, return_counts=True, equal_nan=False)

but returns a namedtuple for easier access to each output.

Parameters
----------
x : array_like
    Input array. It will be flattened if it is not already 1-D.

Returns
-------
out : namedtuple
    The result containing:

    * values - The unique elements of an input array.
    * counts - The corresponding counts for each unique element.

See Also
--------
unique : Find the unique elements of an array.

Examples
--------
>>> import numpy as np
>>> x = [1, 1, 2]
>>> uniq = np.unique_counts(x)
>>> uniq.values
array([1, 2])
>>> uniq.counts
array([2, 1])

### Function: _unique_inverse_dispatcher()

### Function: unique_inverse(x)

**Description:** Find the unique elements of `x` and indices to reconstruct `x`.

This function is an Array API compatible alternative to::

    np.unique(x, return_inverse=True, equal_nan=False)

but returns a namedtuple for easier access to each output.

Parameters
----------
x : array_like
    Input array. It will be flattened if it is not already 1-D.

Returns
-------
out : namedtuple
    The result containing:

    * values - The unique elements of an input array.
    * inverse_indices - The indices from the set of unique elements
      that reconstruct `x`.

See Also
--------
unique : Find the unique elements of an array.

Examples
--------
>>> import numpy as np
>>> x = [1, 1, 2]
>>> uniq = np.unique_inverse(x)
>>> uniq.values
array([1, 2])
>>> uniq.inverse_indices
array([0, 0, 1])

### Function: _unique_values_dispatcher()

### Function: unique_values(x)

**Description:** Returns the unique elements of an input array `x`.

This function is an Array API compatible alternative to::

    np.unique(x, equal_nan=False)

Parameters
----------
x : array_like
    Input array. It will be flattened if it is not already 1-D.

Returns
-------
out : ndarray
    The unique elements of an input array.

See Also
--------
unique : Find the unique elements of an array.

Examples
--------
>>> import numpy as np
>>> np.unique_values([1, 1, 2])
array([1, 2])

### Function: _intersect1d_dispatcher(ar1, ar2, assume_unique, return_indices)

### Function: intersect1d(ar1, ar2, assume_unique, return_indices)

**Description:** Find the intersection of two arrays.

Return the sorted, unique values that are in both of the input arrays.

Parameters
----------
ar1, ar2 : array_like
    Input arrays. Will be flattened if not already 1D.
assume_unique : bool
    If True, the input arrays are both assumed to be unique, which
    can speed up the calculation.  If True but ``ar1`` or ``ar2`` are not
    unique, incorrect results and out-of-bounds indices could result.
    Default is False.
return_indices : bool
    If True, the indices which correspond to the intersection of the two
    arrays are returned. The first instance of a value is used if there are
    multiple. Default is False.

Returns
-------
intersect1d : ndarray
    Sorted 1D array of common and unique elements.
comm1 : ndarray
    The indices of the first occurrences of the common values in `ar1`.
    Only provided if `return_indices` is True.
comm2 : ndarray
    The indices of the first occurrences of the common values in `ar2`.
    Only provided if `return_indices` is True.

Examples
--------
>>> import numpy as np
>>> np.intersect1d([1, 3, 4, 3], [3, 1, 2, 1])
array([1, 3])

To intersect more than two arrays, use functools.reduce:

>>> from functools import reduce
>>> reduce(np.intersect1d, ([1, 3, 4, 3], [3, 1, 2, 1], [6, 3, 4, 2]))
array([3])

To return the indices of the values common to the input arrays
along with the intersected values:

>>> x = np.array([1, 1, 2, 3, 4])
>>> y = np.array([2, 1, 4, 6])
>>> xy, x_ind, y_ind = np.intersect1d(x, y, return_indices=True)
>>> x_ind, y_ind
(array([0, 2, 4]), array([1, 0, 2]))
>>> xy, x[x_ind], y[y_ind]
(array([1, 2, 4]), array([1, 2, 4]), array([1, 2, 4]))

### Function: _setxor1d_dispatcher(ar1, ar2, assume_unique)

### Function: setxor1d(ar1, ar2, assume_unique)

**Description:** Find the set exclusive-or of two arrays.

Return the sorted, unique values that are in only one (not both) of the
input arrays.

Parameters
----------
ar1, ar2 : array_like
    Input arrays.
assume_unique : bool
    If True, the input arrays are both assumed to be unique, which
    can speed up the calculation. Default is False.

Returns
-------
setxor1d : ndarray
    Sorted 1D array of unique values that are in only one of the input
    arrays.

Examples
--------
>>> import numpy as np
>>> a = np.array([1, 2, 3, 2, 4])
>>> b = np.array([2, 3, 5, 7, 5])
>>> np.setxor1d(a,b)
array([1, 4, 5, 7])

### Function: _in1d_dispatcher(ar1, ar2, assume_unique, invert)

### Function: in1d(ar1, ar2, assume_unique, invert)

**Description:** Test whether each element of a 1-D array is also present in a second array.

.. deprecated:: 2.0
    Use :func:`isin` instead of `in1d` for new code.

Returns a boolean array the same length as `ar1` that is True
where an element of `ar1` is in `ar2` and False otherwise.

Parameters
----------
ar1 : (M,) array_like
    Input array.
ar2 : array_like
    The values against which to test each value of `ar1`.
assume_unique : bool, optional
    If True, the input arrays are both assumed to be unique, which
    can speed up the calculation.  Default is False.
invert : bool, optional
    If True, the values in the returned array are inverted (that is,
    False where an element of `ar1` is in `ar2` and True otherwise).
    Default is False. ``np.in1d(a, b, invert=True)`` is equivalent
    to (but is faster than) ``np.invert(in1d(a, b))``.
kind : {None, 'sort', 'table'}, optional
    The algorithm to use. This will not affect the final result,
    but will affect the speed and memory use. The default, None,
    will select automatically based on memory considerations.

    * If 'sort', will use a mergesort-based approach. This will have
      a memory usage of roughly 6 times the sum of the sizes of
      `ar1` and `ar2`, not accounting for size of dtypes.
    * If 'table', will use a lookup table approach similar
      to a counting sort. This is only available for boolean and
      integer arrays. This will have a memory usage of the
      size of `ar1` plus the max-min value of `ar2`. `assume_unique`
      has no effect when the 'table' option is used.
    * If None, will automatically choose 'table' if
      the required memory allocation is less than or equal to
      6 times the sum of the sizes of `ar1` and `ar2`,
      otherwise will use 'sort'. This is done to not use
      a large amount of memory by default, even though
      'table' may be faster in most cases. If 'table' is chosen,
      `assume_unique` will have no effect.

Returns
-------
in1d : (M,) ndarray, bool
    The values `ar1[in1d]` are in `ar2`.

See Also
--------
isin                  : Version of this function that preserves the
                        shape of ar1.

Notes
-----
`in1d` can be considered as an element-wise function version of the
python keyword `in`, for 1-D sequences. ``in1d(a, b)`` is roughly
equivalent to ``np.array([item in b for item in a])``.
However, this idea fails if `ar2` is a set, or similar (non-sequence)
container:  As ``ar2`` is converted to an array, in those cases
``asarray(ar2)`` is an object array rather than the expected array of
contained values.

Using ``kind='table'`` tends to be faster than `kind='sort'` if the
following relationship is true:
``log10(len(ar2)) > (log10(max(ar2)-min(ar2)) - 2.27) / 0.927``,
but may use greater memory. The default value for `kind` will
be automatically selected based only on memory usage, so one may
manually set ``kind='table'`` if memory constraints can be relaxed.

Examples
--------
>>> import numpy as np
>>> test = np.array([0, 1, 2, 5, 0])
>>> states = [0, 2]
>>> mask = np.in1d(test, states)
>>> mask
array([ True, False,  True, False,  True])
>>> test[mask]
array([0, 2, 0])
>>> mask = np.in1d(test, states, invert=True)
>>> mask
array([False,  True, False,  True, False])
>>> test[mask]
array([1, 5])

### Function: _in1d(ar1, ar2, assume_unique, invert)

### Function: _isin_dispatcher(element, test_elements, assume_unique, invert)

### Function: isin(element, test_elements, assume_unique, invert)

**Description:** Calculates ``element in test_elements``, broadcasting over `element` only.
Returns a boolean array of the same shape as `element` that is True
where an element of `element` is in `test_elements` and False otherwise.

Parameters
----------
element : array_like
    Input array.
test_elements : array_like
    The values against which to test each value of `element`.
    This argument is flattened if it is an array or array_like.
    See notes for behavior with non-array-like parameters.
assume_unique : bool, optional
    If True, the input arrays are both assumed to be unique, which
    can speed up the calculation.  Default is False.
invert : bool, optional
    If True, the values in the returned array are inverted, as if
    calculating `element not in test_elements`. Default is False.
    ``np.isin(a, b, invert=True)`` is equivalent to (but faster
    than) ``np.invert(np.isin(a, b))``.
kind : {None, 'sort', 'table'}, optional
    The algorithm to use. This will not affect the final result,
    but will affect the speed and memory use. The default, None,
    will select automatically based on memory considerations.

    * If 'sort', will use a mergesort-based approach. This will have
      a memory usage of roughly 6 times the sum of the sizes of
      `element` and `test_elements`, not accounting for size of dtypes.
    * If 'table', will use a lookup table approach similar
      to a counting sort. This is only available for boolean and
      integer arrays. This will have a memory usage of the
      size of `element` plus the max-min value of `test_elements`.
      `assume_unique` has no effect when the 'table' option is used.
    * If None, will automatically choose 'table' if
      the required memory allocation is less than or equal to
      6 times the sum of the sizes of `element` and `test_elements`,
      otherwise will use 'sort'. This is done to not use
      a large amount of memory by default, even though
      'table' may be faster in most cases. If 'table' is chosen,
      `assume_unique` will have no effect.


Returns
-------
isin : ndarray, bool
    Has the same shape as `element`. The values `element[isin]`
    are in `test_elements`.

Notes
-----
`isin` is an element-wise function version of the python keyword `in`.
``isin(a, b)`` is roughly equivalent to
``np.array([item in b for item in a])`` if `a` and `b` are 1-D sequences.

`element` and `test_elements` are converted to arrays if they are not
already. If `test_elements` is a set (or other non-sequence collection)
it will be converted to an object array with one element, rather than an
array of the values contained in `test_elements`. This is a consequence
of the `array` constructor's way of handling non-sequence collections.
Converting the set to a list usually gives the desired behavior.

Using ``kind='table'`` tends to be faster than `kind='sort'` if the
following relationship is true:
``log10(len(test_elements)) >
(log10(max(test_elements)-min(test_elements)) - 2.27) / 0.927``,
but may use greater memory. The default value for `kind` will
be automatically selected based only on memory usage, so one may
manually set ``kind='table'`` if memory constraints can be relaxed.

Examples
--------
>>> import numpy as np
>>> element = 2*np.arange(4).reshape((2, 2))
>>> element
array([[0, 2],
       [4, 6]])
>>> test_elements = [1, 2, 4, 8]
>>> mask = np.isin(element, test_elements)
>>> mask
array([[False,  True],
       [ True, False]])
>>> element[mask]
array([2, 4])

The indices of the matched values can be obtained with `nonzero`:

>>> np.nonzero(mask)
(array([0, 1]), array([1, 0]))

The test can also be inverted:

>>> mask = np.isin(element, test_elements, invert=True)
>>> mask
array([[ True, False],
       [False,  True]])
>>> element[mask]
array([0, 6])

Because of how `array` handles sets, the following does not
work as expected:

>>> test_set = {1, 2, 4, 8}
>>> np.isin(element, test_set)
array([[False, False],
       [False, False]])

Casting the set to a list gives the expected result:

>>> np.isin(element, list(test_set))
array([[False,  True],
       [ True, False]])

### Function: _union1d_dispatcher(ar1, ar2)

### Function: union1d(ar1, ar2)

**Description:** Find the union of two arrays.

Return the unique, sorted array of values that are in either of the two
input arrays.

Parameters
----------
ar1, ar2 : array_like
    Input arrays. They are flattened if they are not already 1D.

Returns
-------
union1d : ndarray
    Unique, sorted union of the input arrays.

Examples
--------
>>> import numpy as np
>>> np.union1d([-1, 0, 1], [-2, 0, 2])
array([-2, -1,  0,  1,  2])

To find the union of more than two arrays, use functools.reduce:

>>> from functools import reduce
>>> reduce(np.union1d, ([1, 3, 4, 3], [3, 1, 2, 1], [6, 3, 4, 2]))
array([1, 2, 3, 4, 6])

### Function: _setdiff1d_dispatcher(ar1, ar2, assume_unique)

### Function: setdiff1d(ar1, ar2, assume_unique)

**Description:** Find the set difference of two arrays.

Return the unique values in `ar1` that are not in `ar2`.

Parameters
----------
ar1 : array_like
    Input array.
ar2 : array_like
    Input comparison array.
assume_unique : bool
    If True, the input arrays are both assumed to be unique, which
    can speed up the calculation.  Default is False.

Returns
-------
setdiff1d : ndarray
    1D array of values in `ar1` that are not in `ar2`. The result
    is sorted when `assume_unique=False`, but otherwise only sorted
    if the input is sorted.

Examples
--------
>>> import numpy as np
>>> a = np.array([1, 2, 3, 2, 4, 1])
>>> b = np.array([3, 4, 5, 6])
>>> np.setdiff1d(a, b)
array([1, 2])

### Function: reshape_uniq(uniq)
